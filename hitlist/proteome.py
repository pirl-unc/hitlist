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

from dataclasses import dataclass, field
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


# Module-level cache for ``from_fasta``. Keyed on the resolved absolute path
# plus (size, mtime) of the file, plus the index-configuration inputs
# (``lengths``, ``gene_name``, ``gene_id``). Size + mtime invalidate the
# entry automatically when the FASTA is replaced (e.g. by
# ``fetch_species_proteome`` downloading a newer build).
_FASTA_INDEX_CACHE: dict[tuple, ProteomeIndex] = {}


def clear_fasta_index_cache() -> None:
    """Drop all cached ``from_fasta`` indexes.

    Useful in tests that write fresh FASTA files with reused filenames
    within the same process. Production code should let the built-in
    size/mtime invalidation handle it.
    """
    _FASTA_INDEX_CACHE.clear()


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
            # Pick canonical (longest) transcript
            best_t = None
            best_len = 0
            for t in gene.transcripts:
                if t.biotype != "protein_coding":
                    continue
                try:
                    seq = t.protein_sequence
                except Exception:
                    continue
                if seq and len(seq) > best_len:
                    best_t = t
                    best_len = len(seq)
            if best_t is None:
                continue
            proteins[best_t.id] = best_t.protein_sequence
            meta[best_t.id] = {"gene_name": gene.name, "gene_id": gene.id}

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
                if verbose:
                    print(f"  ProteomeIndex cache hit for {resolved.name}")
                return cached

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
                    meta[current_id] = {"gene_name": gn, "gene_id": gene_id}
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_id:
            proteins[current_id] = "".join(current_seq)
        idx = cls._build(proteins, meta, lengths, verbose)
        if cache_key is not None:
            _FASTA_INDEX_CACHE[cache_key] = idx
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
        raw: dict[str, list[int]] = {}
        items = list(proteins.items())
        prot_iter = (
            _tqdm(items, desc="Building index", leave=False) if (_tqdm and verbose) else items
        )
        for prot_id, seq in prot_iter:
            pi = prot_to_idx[prot_id]
            for k in lengths:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    packed = _pack(pi, i)
                    if kmer in raw:
                        raw[kmer].append(packed)
                    else:
                        raw[kmer] = [packed]

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
