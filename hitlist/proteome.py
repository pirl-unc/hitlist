# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Peptide-to-source-protein mapping with flanking sequence extraction.

Builds an efficient k-mer index from a protein sequence database (pyensembl
or FASTA) and maps observed peptides to their source protein(s) with
N-terminal and C-terminal flanking context.

The index is built once per proteome and cached. Lookups are O(1) per
peptide via a dict-based inverted index mapping each unique k-mer to
the set of (protein_id, position) occurrences.

``ProteomeIndex.from_fasta`` additionally caches indexes by resolved
``(path, size, mtime, lengths, gene_name, gene_id)`` across calls in the
same process, so repeated flanking passes against FASTAs shared by
multiple canonical species names (common for viral-strain fallbacks)
pay the indexing cost once. Invalidates automatically when the FASTA
on disk changes (size or mtime).

Memory representation (v1.13.4+):
  The k-mer index ``self.index`` stores **packed int64** postings, not
  ``list[tuple[str, int]]``. Each posting is ``(prot_idx << 32) | pos``
  where ``prot_idx`` refers to ``self._protein_ids``. Single-hit k-mers
  (the vast majority) store a scalar ``int`` directly; multi-hit k-mers
  store a ``numpy.ndarray`` of int64. Callers should use ``lookup()`` or
  ``map_peptides()`` rather than accessing ``self.index`` directly — the
  raw values will not be the ``(protein_id, position)`` tuples they used
  to be. This cuts human-proteome (8/9/10/11-mer) peak RSS from ~13 GB
  to ~2-3 GB and makes full-rebuild feasible on 8-16 GB machines.

Typical usage::

    from hitlist.proteome import ProteomeIndex

    idx = ProteomeIndex.from_ensembl(release=112)
    hits = idx.map_peptides(["SLYNTVATL", "GILGFVFTL"], flank=5)
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import tempfile
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Packing for (prot_idx, pos) → int64: 32 bits for each.
# 32 bits prot_idx: up to 4B proteins (far more than any realistic merged index).
# 32 bits pos: up to 4B residues (titin, the longest human protein, is 35K aa).
_PROT_BITS = 32
_POS_MASK = (1 << _PROT_BITS) - 1


def _pack(prot_idx: int, pos: int) -> int:
    return (prot_idx << _PROT_BITS) | pos


def _unpack(packed: int) -> tuple[int, int]:
    return (packed >> _PROT_BITS, packed & _POS_MASK)


# Bounded LRU cache for ``from_fasta``.  Keyed on the resolved absolute
# path plus (size, mtime) of the file, plus the index-configuration inputs
# (``lengths``, ``gene_name``, ``gene_id``).  Size + mtime invalidate the
# entry automatically when the FASTA is replaced (e.g. by
# ``fetch_species_proteome`` downloading a newer build).
#
# Issue #109: prior to v1.19.2 this was an *unbounded* dict, which made
# the full-build flanking pass accumulate every species' ProteomeIndex
# in RAM (the human index alone is ~10 GB; ~525 canonicals iterated by
# ``_add_flanking`` push peak RSS into 16-32 GB territory and the OS
# starts OOM-killing the build).  An ``OrderedDict``-backed LRU caps
# resident size while still capturing the common case — strain-variant
# canonicals appear close together in the build order, so an LRU of 4
# keeps every cache-hit path that motivated #86.
_FASTA_INDEX_CACHE_MAXSIZE: int = 4
_FASTA_INDEX_CACHE: OrderedDict[tuple, ProteomeIndex] = OrderedDict()


def set_fasta_index_cache_maxsize(maxsize: int) -> None:
    """Adjust the bounded cache size used by :meth:`ProteomeIndex.from_fasta`.

    The default ``4`` is enough to capture the strain-variant cache hits
    motivating #86 (e.g. multiple SARS-CoV-2 / EBV canonicals resolve to
    the same FASTA in close succession).  Tests use this to drive the
    LRU eviction path with a small chain of distinct FASTAs.

    Reducing ``maxsize`` evicts entries beyond the new bound immediately.
    Setting it to ``0`` disables caching entirely (every call re-indexes).
    """
    global _FASTA_INDEX_CACHE_MAXSIZE
    if maxsize < 0:
        raise ValueError("maxsize must be non-negative")
    _FASTA_INDEX_CACHE_MAXSIZE = int(maxsize)
    while len(_FASTA_INDEX_CACHE) > _FASTA_INDEX_CACHE_MAXSIZE:
        _FASTA_INDEX_CACHE.popitem(last=False)


def clear_fasta_index_cache() -> None:
    """Drop all cached ``from_fasta`` indexes.

    Useful in tests that write fresh FASTA files with reused filenames
    within the same process. Production code should let the built-in
    size/mtime invalidation handle it.
    """
    _FASTA_INDEX_CACHE.clear()


# ── On-disk persistent cache for built ProteomeIndexes (#246) ─────────────
#
# Built indexes are pickled to a per-host cache directory keyed by the
# same shape used by the in-memory ``_FASTA_INDEX_CACHE`` plus a format
# version.  Subsequent runs that see the same FASTA on disk (size +
# mtime unchanged) skip the ~minute-scale ``_build`` k-mer pass and load
# the pickle in ~1-3s on SSD.
#
# The dominant build-time stage on cold machines is ``peptide_mappings``
# (#176): cProfile shows 159 ``_build`` calls (40 species x 4 lengths)
# adding to ~67% of total wall.  When IEDB CSVs change but FASTAs don't
# (the common deploy pattern), this cache turns subsequent rebuilds
# into near-no-ops for the mappings stage.

# Bump on any breaking change to ``_build``'s output schema (the dict
# layout, the int64 packing, etc.).  Stale-format files are skipped
# (treated as cache misses) and eventually evicted by the cap policy.
_INDEX_FORMAT_VERSION: int = 1

_PROTEOME_INDEX_DISK_CACHE_DIR: Path = Path.home() / ".hitlist" / "proteome_index_cache"


def _resolve_disk_cache_max_gb() -> float:
    """Read the disk-cache cap (GB) from ``HITLIST_PROTEOME_INDEX_CACHE_GB``
    each call so tests can override via :func:`os.environ`.

    Returns ``0.0`` to disable caching entirely.  Default 50 GB caps the
    cache below typical free-space watermarks while accommodating the
    full Homo sapiens index (~12 GB across 4 lengths) plus the largest
    non-human proteomes (mouse, dog, chimp, macaque, pig, cow each
    ~5-10 GB across 4 lengths).
    """
    raw = os.environ.get("HITLIST_PROTEOME_INDEX_CACHE_GB", "50")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 50.0


def set_disk_cache_dir(path: Path | str | None) -> None:
    """Override the on-disk proteome-index cache directory.

    Pass ``None`` to revert to the default (~/.hitlist/proteome_index_cache).
    Tests use this to point at a tmp_path so they don't pollute the
    real cache and aren't affected by it.
    """
    global _PROTEOME_INDEX_DISK_CACHE_DIR
    if path is None:
        _PROTEOME_INDEX_DISK_CACHE_DIR = Path.home() / ".hitlist" / "proteome_index_cache"
    else:
        _PROTEOME_INDEX_DISK_CACHE_DIR = Path(path)


def clear_disk_cache() -> None:
    """Delete every cached index on disk.  Useful for tests + manual reset.

    No-op when the cache dir doesn't exist yet.
    """
    if _PROTEOME_INDEX_DISK_CACHE_DIR.exists():
        for f in _PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"):
            with contextlib.suppress(FileNotFoundError):
                f.unlink()


def _disk_cache_filename(cache_key: tuple) -> str:
    """Deterministic filename for a cache key.

    The first 16 hex chars of a sha256 of ``repr(cache_key)`` give us a
    collision-proof handle; we prefix with the FASTA basename + lengths
    so files are human-debuggable from ``ls``.  The format-version
    prefix lets us evolve ``_build``'s output shape without
    contaminating new builds with stale-format pickles.
    """
    path_str, _size, _mtime, lengths, _gn, _gid = cache_key
    basename = Path(path_str).stem.lower().replace(" ", "_")
    lengths_str = "-".join(str(L) for L in lengths)
    h = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]
    return f"v{_INDEX_FORMAT_VERSION}_{basename}_L{lengths_str}_{h}.pkl"


def _load_index_from_disk(cache_key: tuple) -> ProteomeIndex | None:
    """Return a cached index if a fresh-format pickle exists, else None.

    On hit, touches the file's mtime so the LRU eviction sees recent use.
    On any I/O / unpickling failure (corrupt file, partial write, schema
    drift slipping past the version prefix), returns None and removes
    the bad file so the caller can rebuild + re-cache.
    """
    if _resolve_disk_cache_max_gb() <= 0:
        return None
    cache_path = _PROTEOME_INDEX_DISK_CACHE_DIR / _disk_cache_filename(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, "rb") as f:
            idx = pickle.load(f)
    except Exception:
        # Corrupt / partial / wrong-format cache file — drop it so the
        # rebuild path runs and writes a fresh pickle.
        with contextlib.suppress(FileNotFoundError):
            cache_path.unlink()
        return None
    # Touch the file so this hit promotes it in the LRU.  Skip ENOENT
    # races (file evicted between load and touch).
    with contextlib.suppress(FileNotFoundError):
        os.utime(cache_path, None)
    return idx


def _write_index_to_disk(cache_key: tuple, idx: ProteomeIndex) -> None:
    """Persist an index to disk via atomic ``.tmp + rename``.

    Failures are logged via ``warnings.warn`` and otherwise swallowed —
    the in-memory cache + a future cold rebuild are sufficient
    fallbacks; we'd rather not crash a multi-hour build because the
    cache dir filled up.
    """
    if _resolve_disk_cache_max_gb() <= 0:
        return
    cache_path = _PROTEOME_INDEX_DISK_CACHE_DIR / _disk_cache_filename(cache_key)
    try:
        _PROTEOME_INDEX_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile in same dir → ``os.replace`` is atomic on
        # POSIX (same filesystem guaranteed).  Don't use ``delete=True``
        # because we want the rename, not auto-cleanup.
        with tempfile.NamedTemporaryFile(
            dir=_PROTEOME_INDEX_DISK_CACHE_DIR,
            prefix=cache_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            pickle.dump(idx, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    except Exception as exc:  # pragma: no cover — best-effort persistence
        warnings.warn(
            f"Failed to write proteome index cache {cache_path}: {exc}",
            stacklevel=2,
        )
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        return
    _evict_disk_cache_if_over_cap()


def _evict_disk_cache_if_over_cap() -> None:
    """Evict oldest-mtime cache files until total size ≤ cap.

    Called after every successful write.  Uses file mtime as the LRU
    proxy — ``_load_index_from_disk`` touches mtime on every hit, so
    it's a recency-of-use signal even though it doubles as the
    last-modified timestamp.
    """
    cap_bytes = int(_resolve_disk_cache_max_gb() * 1024**3)
    if cap_bytes <= 0:
        return
    if not _PROTEOME_INDEX_DISK_CACHE_DIR.is_dir():
        return
    files = []
    total = 0
    for f in _PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"):
        try:
            stat = f.stat()
        except FileNotFoundError:
            continue
        files.append((stat.st_mtime, stat.st_size, f))
        total += stat.st_size
    if total <= cap_bytes:
        return
    # Oldest first.
    files.sort(key=lambda t: t[0])
    for _mtime, size, f in files:
        if total <= cap_bytes:
            break
        with contextlib.suppress(FileNotFoundError):
            f.unlink()
            total -= size


try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


@dataclass
class ProteinSource:
    """A source protein occurrence for a peptide."""

    protein_id: str
    gene_name: str
    gene_id: str
    position: int  # 0-based start in protein
    n_flank: str
    c_flank: str


@dataclass
class ProteomeIndex:
    """Inverted index mapping peptide sequences to source proteins.

    Attributes
    ----------
    proteins : dict[str, str]
        Mapping from protein_id to amino acid sequence.
    protein_meta : dict[str, dict]
        Mapping from protein_id to metadata (gene_name, gene_id).
    index : dict[str, int | np.ndarray]
        Inverted index: peptide -> packed postings. For single-hit k-mers
        the value is an ``int`` packing ``(prot_idx << 32) | pos``; for
        multi-hit k-mers the value is a ``numpy.ndarray`` of int64 packed
        values. Use :meth:`lookup` or :meth:`map_peptides` — do not
        decode ``self.index`` directly in callers.
    lengths : tuple[int, ...]
        Peptide lengths that were indexed.
    """

    proteins: dict[str, str] = field(repr=False)
    protein_meta: dict[str, dict] = field(repr=False)
    index: dict[str, int | np.ndarray] = field(repr=False)
    lengths: tuple[int, ...]
    # Internal reverse lookup: int prot_idx → protein_id string.
    # Populated in _build; kept in insertion order matching ``proteins``.
    _protein_ids: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_ensembl(
        cls,
        release: int = 112,
        lengths: tuple[int, ...] = (8, 9, 10, 11),
        biotype: str = "protein_coding",
        verbose: bool = True,
        species: str = "human",
    ) -> ProteomeIndex:
        """Build index from Ensembl protein sequences.

        Parameters
        ----------
        release
            Ensembl release number.
        lengths
            Peptide lengths to index.
        biotype
            Gene biotype filter (default ``"protein_coding"``).
        verbose
            Print progress messages.
        species
            pyensembl species key (``"human"``, ``"mouse"``, ``"rat"``, ...).

        Returns
        -------
        ProteomeIndex
        """
        from pyensembl import EnsemblRelease

        try:
            ensembl = EnsemblRelease(release, species=species)
        except TypeError:
            # Older pyensembl without species kwarg — falls back to human
            if species != "human":
                raise
            ensembl = EnsemblRelease(release)
        proteins: dict[str, str] = {}
        meta: dict[str, dict] = {}

        genes = list(ensembl.genes())
        gene_iter = (
            _tqdm(genes, desc="Loading proteins", leave=False) if (_tqdm and verbose) else genes
        )

        for gene in gene_iter:
            if gene.biotype != biotype:
                continue
            # Issue #141: index every protein-coding transcript per gene
            # rather than collapsing each gene to its longest transcript.
            # The canonical transcript is identified post-hoc as the
            # longest valid translation (a stable, pyensembl-version-
            # independent definition), and every transcript is recorded
            # with its own ``transcript_id`` so downstream consumers can
            # distinguish gene-level ambiguity from transcript-isoform
            # ambiguity in mapping rows.
            transcript_records: list[tuple[str, str, str]] = []
            for t in gene.transcripts:
                if getattr(t, "biotype", "") != "protein_coding":
                    continue
                try:
                    seq = t.protein_sequence
                except Exception:
                    continue
                if not seq:
                    continue
                # Prefer the stable Ensembl protein/translation ID (ENSP)
                # as the index key.  When pyensembl can't surface it,
                # fall back to the transcript ID (ENST) so the proteome
                # still indexes — but the ``transcript_id`` meta column
                # always carries the ENST.
                protein_key = getattr(t, "protein_id", None) or t.id
                transcript_records.append((protein_key, t.id, seq))

            if not transcript_records:
                continue

            canonical_protein_key = max(transcript_records, key=lambda r: len(r[2]))[0]

            for protein_key, transcript_id, seq in transcript_records:
                if protein_key in proteins:
                    # Same ENSP shared by multiple transcripts (rare but
                    # possible when alternative ENSTs translate identically).
                    # Skip duplicates — the meta we already have is correct.
                    continue
                proteins[protein_key] = seq
                meta[protein_key] = {
                    "gene_name": gene.name,
                    "gene_id": gene.id,
                    "transcript_id": transcript_id,
                    "is_canonical_transcript": protein_key == canonical_protein_key,
                }

        return cls._build(proteins, meta, lengths, verbose)

    @classmethod
    def from_fasta(
        cls,
        path: str | Path,
        lengths: tuple[int, ...] = (8, 9, 10, 11),
        gene_name: str = "",
        gene_id: str = "",
        verbose: bool = True,
    ) -> ProteomeIndex:
        """Build index from a FASTA file.

        Memoized by ``(resolved_path, size, mtime, lengths, gene_name,
        gene_id)`` for the lifetime of the process. Repeated calls with
        the same inputs return the same ``ProteomeIndex`` instance
        without re-parsing the FASTA — important during flanking passes
        where multiple canonical species names (e.g. several LCMV or
        SARS-CoV-2 strain variants) fall back to one shared cached
        FASTA. The (size, mtime) part of the key invalidates the entry
        automatically if the on-disk file changes.

        Parameters
        ----------
        path
            Path to FASTA file.
        lengths
            Peptide lengths to index.
        gene_name, gene_id
            Default gene name/ID for all proteins (overridden by FASTA headers).
        verbose
            Print progress.

        Returns
        -------
        ProteomeIndex
        """
        resolved = Path(path).resolve()
        try:
            stat = resolved.stat()
            cache_key: tuple | None = (
                str(resolved),
                stat.st_size,
                stat.st_mtime_ns,
                tuple(lengths),
                gene_name,
                gene_id,
            )
        except OSError:
            # If the FASTA moved/was deleted between resolution and stat,
            # fall through to the uncached path — the open() below will
            # raise with a more actionable FileNotFoundError.
            cache_key = None

        if cache_key is not None:
            cached = _FASTA_INDEX_CACHE.get(cache_key)
            if cached is not None:
                # Touch the entry so LRU order tracks recency-of-use.
                _FASTA_INDEX_CACHE.move_to_end(cache_key)
                if verbose:
                    print(f"  ProteomeIndex cache hit for {resolved.name}")
                return cached
            # In-memory miss → check the on-disk cache (#246).  Skips
            # the ~minute-scale ``_build`` k-mer pass when this same
            # FASTA was indexed in a previous run (size + mtime
            # unchanged).  Promotes the loaded index into the in-memory
            # cache so subsequent calls within this run hit the fast
            # path.
            disk_cached = _load_index_from_disk(cache_key)
            if disk_cached is not None:
                if verbose:
                    print(f"  ProteomeIndex disk-cache hit for {resolved.name}")
                if _FASTA_INDEX_CACHE_MAXSIZE > 0:
                    _FASTA_INDEX_CACHE[cache_key] = disk_cached
                    _FASTA_INDEX_CACHE.move_to_end(cache_key)
                    while len(_FASTA_INDEX_CACHE) > _FASTA_INDEX_CACHE_MAXSIZE:
                        _FASTA_INDEX_CACHE.popitem(last=False)
                return disk_cached

        proteins: dict[str, str] = {}
        meta: dict[str, dict] = {}
        current_id = ""
        current_seq: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        proteins[current_id] = "".join(current_seq)
                    header = line[1:].strip()
                    current_id = header.split()[0] if header else header
                    # Try to parse gene name from header (UniProt format: GN=GENENAME)
                    gn = gene_name
                    if "GN=" in header:
                        gn = header.split("GN=")[1].split()[0]
                    # FASTA-backed entries don't carry transcript identity,
                    # but emitting the keys keeps the meta schema uniform
                    # with the Ensembl path so map_peptides output has
                    # consistent columns regardless of backend (#141).
                    meta[current_id] = {
                        "gene_name": gn,
                        "gene_id": gene_id,
                        "transcript_id": "",
                        "is_canonical_transcript": False,
                    }
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_id:
            proteins[current_id] = "".join(current_seq)
        idx = cls._build(proteins, meta, lengths, verbose)
        if cache_key is not None and _FASTA_INDEX_CACHE_MAXSIZE > 0:
            _FASTA_INDEX_CACHE[cache_key] = idx
            _FASTA_INDEX_CACHE.move_to_end(cache_key)
            # Evict the least-recently-used entry when over capacity.  Each
            # eviction releases the proteome's ~GB-scale arrays so the build
            # loop's peak RSS stays bounded by `maxsize x largest proteome`
            # rather than the sum of all proteomes seen so far (issue #109).
            while len(_FASTA_INDEX_CACHE) > _FASTA_INDEX_CACHE_MAXSIZE:
                _FASTA_INDEX_CACHE.popitem(last=False)
        # Persist to the on-disk cache so subsequent runs (when the
        # FASTA size + mtime are unchanged) skip the expensive _build
        # k-mer pass entirely.  Best-effort: failures here are warned
        # but don't break the build (#246).
        if cache_key is not None:
            _write_index_to_disk(cache_key, idx)
        return idx

    @classmethod
    def _build(
        cls,
        proteins: dict[str, str],
        meta: dict[str, dict],
        lengths: tuple[int, ...],
        verbose: bool,
    ) -> ProteomeIndex:
        """Build the compact packed-int64 k-mer index.

        Two-phase build:
          1. Collect raw postings in ``dict[str, list[int64]]``.
          2. Compact to ``dict[str, int | np.ndarray]`` — scalar for
             singletons (most k-mers), np.ndarray for multis.

        The compact step cuts the index footprint from ~195 bytes/k-mer
        (str key + list-of-tuple value) to ~80 bytes/k-mer for
        singletons (str key + scalar int value). On human (41.7M
        k-mers) this drops the index from ~8.2 GB to ~3 GB.
        """
        protein_ids: list[str] = list(proteins.keys())
        prot_to_idx: dict[str, int] = {pid: i for i, pid in enumerate(protein_ids)}

        # Phase 1: collect raw packed postings per k-mer.
        #
        # Hot loop: cProfile (#176) showed 1.13B ``_pack`` calls dominating
        # runtime — pure Python function-call overhead since ``_pack`` does
        # only a shift+OR. We inline the pack op and hoist the per-protein
        # shift out of the inner loop; the per-kmer cost drops to a string
        # slice + dict lookup + OR + list op. Tried ``setdefault`` here and
        # it benchmarked ~10% slower because the default ``[]`` is built
        # on every call even on the common existing-key path.
        raw: dict[str, list[int]] = {}
        items = list(proteins.items())
        prot_iter = (
            _tqdm(items, desc="Building index", leave=False) if (_tqdm and verbose) else items
        )
        for prot_id, seq in prot_iter:
            pi_shifted = prot_to_idx[prot_id] << _PROT_BITS
            seq_len = len(seq)
            for k in lengths:
                end = seq_len - k + 1
                for i in range(end):
                    kmer = seq[i : i + k]
                    packed = pi_shifted | i
                    posting = raw.get(kmer)
                    if posting is None:
                        raw[kmer] = [packed]
                    else:
                        posting.append(packed)

        # Phase 2: compact — singletons → scalar int, multis → np.ndarray.
        index: dict[str, int | np.ndarray] = {}
        for kmer, postings in raw.items():
            if len(postings) == 1:
                index[kmer] = postings[0]
            else:
                index[kmer] = np.asarray(postings, dtype=np.int64)
        del raw  # release the list-of-list backing store before return

        if verbose:
            n_single = sum(1 for v in index.values() if isinstance(v, int))
            print(
                f"ProteomeIndex: {len(proteins):,} proteins, "
                f"{len(index):,} unique k-mers ({'/'.join(str(k) for k in lengths)}-mers, "
                f"{n_single:,} single-hit / {len(index) - n_single:,} multi-hit)"
            )
        return cls(
            proteins=proteins,
            protein_meta=meta,
            index=index,
            lengths=lengths,
            _protein_ids=protein_ids,
        )

    def merge(self, other: ProteomeIndex) -> ProteomeIndex:
        """Merge another index into this one.

        Combines proteins, metadata, and k-mer index entries from both.
        Useful for combining human + viral proteome indices. Rebuilds the
        compact posting representation — the simplest correct path since
        protein indices in ``self`` and ``other`` refer to disjoint ID
        spaces and have to be renumbered against the merged protein list.

        Parameters
        ----------
        other
            Another ProteomeIndex to merge in.

        Returns
        -------
        ProteomeIndex
            New merged index (does not modify self or other).
        """
        proteins = {**self.proteins, **other.proteins}
        meta = {**self.protein_meta, **other.protein_meta}
        lengths = tuple(sorted(set(self.lengths) | set(other.lengths)))
        # Cheapest correct merge: rebuild from scratch against the unioned
        # protein dict. For the typical use case (human + a handful of
        # viral FASTAs) this is only a few percent of the original build
        # cost since viral proteomes are tiny.
        return self.__class__._build(proteins, meta, lengths, verbose=False)

    @classmethod
    def from_ensembl_plus_fastas(
        cls,
        fasta_paths: list[str | Path] | None = None,
        release: int = 112,
        lengths: tuple[int, ...] = (8, 9, 10, 11),
        verbose: bool = True,
    ) -> ProteomeIndex:
        """Build a combined human + viral/custom proteome index.

        Parameters
        ----------
        fasta_paths
            List of FASTA files to include (e.g. viral proteomes).
        release
            Ensembl release for human proteome.
        lengths
            Peptide lengths to index.
        verbose
            Print progress.

        Returns
        -------
        ProteomeIndex
            Merged index covering human and all provided FASTA proteomes.
        """
        idx = cls.from_ensembl(release=release, lengths=lengths, verbose=verbose)
        for fasta in fasta_paths or []:
            viral_idx = cls.from_fasta(fasta, lengths=lengths, verbose=verbose)
            idx = idx.merge(viral_idx)
        return idx

    @cached_property
    def all_kmers(self) -> frozenset[str]:
        """Every unique k-mer (at ``self.lengths``) across all indexed proteins.

        Cached on first access. Typical sizes:

        - Human at (8, 9, 10, 11): ~41 M k-mers, ~1 GB as a frozenset of strings.
        - Human at (9,): ~10 M k-mers, ~250 MB.

        The primitive that downstream packages (tsarina, perseus, topiary,
        notebook scripts) use for self-peptide subtraction. See also the
        module-level :func:`proteome_kmer_set` if the caller doesn't need
        the full :class:`ProteomeIndex` object.
        """
        return frozenset(self.index.keys())

    def kmers_for_genes(self, gene_ids: frozenset[str]) -> frozenset[str]:
        """K-mers restricted to proteins with these Ensembl gene IDs.

        Walks the proteins table rather than the k-mer index because the
        compact posting representation stores prot_idx ints, not gene_ids
        — filtering the index directly would be slower than re-extracting
        k-mers from the selected protein sequences.

        Parameters
        ----------
        gene_ids
            Set of Ensembl gene IDs to include. Must be a frozenset for
            hashability (the module-level :func:`proteome_kmer_set` wraps
            callers so they can pass a regular set too).

        Returns
        -------
        frozenset[str]
            Unique k-mers at ``self.lengths`` from proteins whose
            ``protein_meta[...].gene_id`` is in ``gene_ids``.
        """
        if not gene_ids:
            return frozenset()
        out: set[str] = set()
        for prot_id, seq in self.proteins.items():
            gid = self.protein_meta.get(prot_id, {}).get("gene_id", "")
            if gid not in gene_ids:
                continue
            for k in self.lengths:
                for i in range(len(seq) - k + 1):
                    out.add(seq[i : i + k])
        return frozenset(out)

    def lookup(self, peptide: str) -> list[tuple[str, int]]:
        """Look up a peptide in the index.

        Returns list of (protein_id, position) tuples.
        """
        v = self.index.get(peptide)
        if v is None:
            return []
        if isinstance(v, (int, np.integer)):
            prot_idx, pos = _unpack(int(v))
            return [(self._protein_ids[prot_idx], pos)]
        # np.ndarray of int64 packed postings.
        out: list[tuple[str, int]] = []
        for packed in v:
            prot_idx, pos = _unpack(int(packed))
            out.append((self._protein_ids[prot_idx], pos))
        return out

    def map_peptides(
        self,
        peptides: list[str] | set[str],
        flank: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Map peptides to source proteins with flanking context.

        Parameters
        ----------
        peptides
            Peptide sequences to map.
        flank
            Number of flanking residues to extract on each side (default 5).
        verbose
            Show progress bar.

        Returns
        -------
        pd.DataFrame
            One row per (peptide, source protein) occurrence. Columns:
            ``peptide``, ``protein_id``, ``gene_name``, ``gene_id``,
            ``position`` (0-based), ``n_flank``, ``c_flank``,
            ``n_sources`` (total source count for this peptide),
            ``unique_nflank`` (set if all sources have same n_flank, else ""),
            ``unique_cflank`` (same for c_flank).
        """
        peptide_list = list(set(peptides))
        pep_iter = (
            _tqdm(peptide_list, desc="Mapping peptides", leave=False)
            if (_tqdm and verbose)
            else peptide_list
        )

        rows: list[dict] = []
        for pep in pep_iter:
            hits = self.lookup(pep)
            for prot_id, pos in hits:
                seq = self.proteins[prot_id]
                m = self.protein_meta.get(prot_id, {})
                n_flank = seq[max(0, pos - flank) : pos]
                c_flank = seq[pos + len(pep) : pos + len(pep) + flank]
                rows.append(
                    {
                        "peptide": pep,
                        "protein_id": prot_id,
                        "gene_name": m.get("gene_name", ""),
                        "gene_id": m.get("gene_id", ""),
                        # Issue #141: transcript_id is now a first-class
                        # column distinct from protein_id.  For Ensembl-
                        # backed indexes it carries the ENST; for FASTA-
                        # backed indexes it's "".  is_canonical_transcript
                        # marks the longest protein-coding transcript per
                        # gene (the Ensembl-canonical proxy).
                        "transcript_id": m.get("transcript_id", ""),
                        "is_canonical_transcript": bool(m.get("is_canonical_transcript", False)),
                        "position": pos,
                        "n_flank": n_flank,
                        "c_flank": c_flank,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "peptide",
                    "protein_id",
                    "gene_name",
                    "gene_id",
                    "transcript_id",
                    "is_canonical_transcript",
                    "position",
                    "n_flank",
                    "c_flank",
                    "n_sources",
                    "unique_nflank",
                    "unique_cflank",
                ]
            )

        df = pd.DataFrame(rows)

        # Compute per-peptide source counts and unique flank indicators
        source_counts = df.groupby("peptide")["protein_id"].nunique().rename("n_sources")
        df = df.merge(source_counts, on="peptide", how="left")

        # Unique flank: if all sources have the same flank, store it; else ""
        for side in ["n_flank", "c_flank"]:
            unique_col = f"unique_{side}"
            unique_map: dict[str, str] = {}
            for pep, group in df.groupby("peptide"):
                flanks = set(group[side])
                unique_map[str(pep)] = next(iter(flanks)) if len(flanks) == 1 else ""
            df[unique_col] = df["peptide"].map(unique_map)

        return df


# ---------------------------------------------------------------------------
# Module-level k-mer primitive for cross-package caching (#99).
# ---------------------------------------------------------------------------
#
# Shared by tsarina (self-peptide subtraction), perseus, topiary, and any
# future consumer that wants "every protein-coding k-mer at lengths X,
# optionally restricted to a gene subset." Delegates to ProteomeIndex
# but caches the frozenset so repeated calls with the same arguments are
# sub-millisecond.
#
# Note that the cache composes with ``ProteomeIndex.from_ensembl``'s own
# cost: the first call to ``proteome_kmer_set(release, lengths)`` pays
# the ProteomeIndex build (~45 s for human at all 4 MHC-I lengths),
# subsequent identical calls return the cached frozenset in <1 ms.


@lru_cache(maxsize=8)
def _proteome_kmer_set_cached(
    release: int,
    lengths: tuple[int, ...],
    species: str,
    gene_ids_frozen: frozenset[str] | None,
) -> frozenset[str]:
    """Cache backend for :func:`proteome_kmer_set`.

    ``gene_ids_frozen`` must be hashable (``frozenset`` or ``None``) so
    the ``@lru_cache`` key is stable across callers that pass equivalent
    sets in different orders.
    """
    idx = ProteomeIndex.from_ensembl(
        release=release,
        lengths=lengths,
        species=species,
        verbose=False,
    )
    if gene_ids_frozen is None:
        return idx.all_kmers
    return idx.kmers_for_genes(gene_ids_frozen)


def proteome_kmer_set(
    release: int = 112,
    lengths: tuple[int, ...] = (8, 9, 10, 11),
    gene_ids: frozenset[str] | set[str] | None = None,
    species: str = "human",
) -> frozenset[str]:
    """Every protein-coding k-mer in the proteome, cached across calls.

    The canonical primitive for self-peptide subtraction, vaccine /
    neoantigen candidate filtering, and any other workflow that needs a
    fast "is this sequence a human k-mer?" membership test. Replaces
    per-package local implementations in tsarina, perseus, etc.

    Parameters
    ----------
    release
        Ensembl release. Default 112.
    lengths
        Peptide lengths to include. Must be a tuple (hashable) so the
        cache key is stable. Default ``(8, 9, 10, 11)`` (MHC-I range).
    gene_ids
        Optional set of Ensembl gene IDs to restrict to. When ``None``
        (default), returns the full proteome's k-mer set. Accepts ``set``
        for convenience — internally frozen for the cache key.
    species
        pyensembl species key. Default ``"human"``.

    Returns
    -------
    frozenset[str]
        Unique peptide sequences at the specified lengths.

    Notes
    -----
    - First call: ~10-60 s (ProteomeIndex build + iteration). Memory
      peak during the call is bounded by the ProteomeIndex
      representation (v1.14.0+: ~3 GB for human at all 4 lengths).
    - Subsequent calls with identical arguments: <1 ms (cached frozenset).
    - Gene subset calls walk protein sequences filtered by gene_id
      rather than the k-mer index, because the compact packed-int64
      posting representation doesn't carry gene IDs.
    - Cache size is 8. If you need more distinct ``(release, lengths,
      gene_ids)`` combinations live at once, call
      :meth:`ProteomeIndex.kmers_for_genes` directly on a longer-lived
      ProteomeIndex instance instead.
    """
    gene_ids_frozen = frozenset(gene_ids) if gene_ids is not None else None
    return _proteome_kmer_set_cached(release, tuple(lengths), species, gene_ids_frozen)


# ---------------------------------------------------------------------------
# In-silico protease digest (#104).
# ---------------------------------------------------------------------------
#
# Every enzyme in hitlist's bulk proteomics index (sources.yaml) has subtly
# different cleavage rules. GluC is buffer-dependent (E-only in phosphate,
# E+D in ammonium bicarbonate). LysC cleaves K-P bonds (unlike trypsin).
# Chymotrypsin-plus includes M as an aliphatic target (unlike strict
# chymotrypsin). This helper centralizes the rules so callers building
# theoretical-negative peptide sets (e.g. MS-detectability training) don't
# re-derive them by hand (and drift into subtle buffer/variant bugs).
#
# Canonical enzyme strings match ``sources.yaml::digestion_enzyme`` and
# ``ancillary_digests[].digestion_enzyme`` so dispatch keys align with the
# row-level values downstream consumers already filter on.


# (canonical name, aliases) → (cleavage residues, forbidden P1' residues,
# optional custom check for edge cases like "P allowed"). "forbidden P1'
# of 'P'" encodes the "not before P" rule shared by Trypsin, Chymotrypsin,
# and GluC. LysC's MaxQuant spec explicitly allows K-P cleavage.
_ENZYME_RULES: dict[str, tuple[str, str]] = {
    # Canonical string (matches sources.yaml). (cleavage_residues, forbidden_p1_prime)
    "Trypsin/P (cleaves K/R except before P)": ("KR", "P"),
    "Chymotrypsin": ("FWYLM", "P"),  # MaxQuant "Chymotrypsin+" — includes M
    "GluC": ("ED", "P"),  # MaxQuant "GluC;D.P" — bicarbonate buffer
    "LysC": ("K", ""),  # MaxQuant "LysC/P" — cleaves K-P too
}

# Short aliases for ergonomics.
_ENZYME_ALIASES: dict[str, str] = {
    "Trypsin/P": "Trypsin/P (cleaves K/R except before P)",
    "Trypsin": "Trypsin/P (cleaves K/R except before P)",
    "trypsin": "Trypsin/P (cleaves K/R except before P)",
    "chymotrypsin": "Chymotrypsin",
    "Chymotrypsin+": "Chymotrypsin",
    "chymo": "Chymotrypsin",
    "gluc": "GluC",
    "GluC;D.P": "GluC",
    "lysc": "LysC",
    "LysC/P": "LysC",
}


def digest(
    seq: str,
    enzyme: str = "Trypsin/P (cleaves K/R except before P)",
    min_len: int = 7,
    max_len: int = 30,
    max_missed: int = 2,
) -> set[str]:
    """In-silico protease digest of a protein sequence.

    Returns the set of peptides the specified enzyme would theoretically
    produce, up to ``max_missed`` missed cleavages. Dispatches on the
    canonical enzyme strings from ``hitlist/data/bulk_proteomics/sources.yaml``
    so the result is directly comparable to observed peptides from
    :func:`hitlist.bulk_proteomics.load_bulk_peptides` filtered on the
    same ``digestion_enzyme`` value.

    Parameters
    ----------
    seq
        Protein sequence (amino acid letters, no non-residue chars).
    enzyme
        Canonical enzyme name. Accepted values (canonical form or alias):

        - ``"Trypsin/P (cleaves K/R except before P)"`` / ``"Trypsin"`` /
          ``"Trypsin/P"`` / ``"trypsin"`` — cleaves C-term K/R, not before P.
        - ``"Chymotrypsin"`` / ``"Chymotrypsin+"`` / ``"chymo"`` —
          MaxQuant's permissive variant; cleaves C-term F/W/Y/L/M, not
          before P. (MaxQuant's strict ``Chymotrypsin`` without the ``+``
          is F/W/Y only; not currently supported.)
        - ``"GluC"`` / ``"GluC;D.P"`` / ``"gluc"`` — MaxQuant's
          bicarbonate-buffer variant: cleaves C-term E or D, not before
          P. This matches the Bekker-Jensen 2017 ingest.
        - ``"LysC"`` / ``"LysC/P"`` / ``"lysc"`` — cleaves C-term K,
          allowed before P (unlike trypsin).
    min_len, max_len
        Inclusive peptide length bounds. Defaults match typical
        detectability-training-set inputs (7-30 aa).
    max_missed
        Maximum missed cleavages. Default 2 matches MaxQuant defaults
        used by Bekker-Jensen + CCLE.

    Returns
    -------
    set[str]
        Unique peptide sequences.

    Examples
    --------
    >>> prame = "MERRRLWGSIQSRYI..."
    >>> tryptic = digest(prame, enzyme="Trypsin/P")
    >>> observed = set(load_bulk_peptides(gene_name="PRAME",
    ...     digestion_enzyme="Trypsin/P (cleaves K/R except before P)",
    ... )["peptide"])
    >>> positives = observed & tryptic
    >>> negatives = tryptic - observed
    """
    canonical = _ENZYME_ALIASES.get(enzyme, enzyme)
    if canonical not in _ENZYME_RULES:
        known = sorted(set(_ENZYME_RULES) | set(_ENZYME_ALIASES))
        raise ValueError(f"Unknown enzyme {enzyme!r}. Accepted: {known}")
    cleavage_residues, forbidden_p1_prime = _ENZYME_RULES[canonical]

    # Find cleavage positions (0-based indices of where to cut AFTER).
    cuts: list[int] = [0]
    for i in range(len(seq) - 1):
        if seq[i] in cleavage_residues:
            if forbidden_p1_prime and seq[i + 1] in forbidden_p1_prime:
                continue
            cuts.append(i + 1)
    cuts.append(len(seq))

    # Emit peptides with up to max_missed missed internal cleavages.
    peps: set[str] = set()
    n_cuts = len(cuts)
    for i in range(n_cuts - 1):
        for j in range(i + 1, min(i + 2 + max_missed, n_cuts)):
            p = seq[cuts[i] : cuts[j]]
            if min_len <= len(p) <= max_len:
                peps.add(p)
    return peps
