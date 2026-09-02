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

"""Long-form peptide → source-protein mappings with full multi-mapping.

Every observed peptide is mapped against:
1. Its ``source_organism`` reference proteome (or ``mhc_species`` fallback)
2. Any ``reference_proteomes`` overrides on the owning PMID's ``ms_samples``
   (e.g. EBV for B-LCLs, Influenza A for infected lung)

Unlike the previous ``_add_flanking`` pass, this table preserves every
(peptide, protein, position) occurrence — essential for:

- CT-antigen family attribution (MAGEA1/A4/A10/A12 paralogs share peptides)
- Cross-species hits
- Repeat regions and tandem duplications within one protein
- Short 8-mers with high collision rates

The sidecar is stored at ``~/.hitlist/peptide_mappings.parquet`` with
pyarrow push-down filters on ``peptide``, ``gene_name``, ``gene_id``,
``protein_id`` and ``proteome``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .downloads import data_dir
from .proteome import DEFAULT_FLANK, ENSEMBL_CODING_GENE_BIOTYPES, SEED_KMER_LENGTH

_MAPPING_COLUMNS = (
    "peptide",
    "protein_id",
    "gene_name",
    "gene_id",
    # Ensembl gene biotype distinguishes ordinary protein-coding sources from
    # germline IG/TR segments. FASTA-backed mappings leave it empty because
    # UniProt/custom headers do not provide the equivalent ontology (#399).
    "gene_biotype",
    # Issue #141: transcript_id is a first-class column distinct from
    # protein_id (which now carries ENSP for Ensembl-backed indexes
    # rather than ENST).  is_canonical_transcript flags the longest
    # translated selected transcript per gene as the canonical proxy.
    # FASTA-backed indexes leave transcript_id="" and the flag False.
    "transcript_id",
    "is_canonical_transcript",
    "position",
    "n_flank",
    "c_flank",
    "proteome",
    "proteome_source",
)

# Increment when mapping semantics change in a way not already represented by
# `_mapping_artifact_contract` parameters. Metadata without this version is a
# legacy artifact and must rebuild once on upgrade (#404).
_MAPPING_ARTIFACT_VERSION = 2

# Hard wall-clock deadline for the complete parent-side cache warm-up. This is
# an internal safety invariant rather than an environment-variable tuning knob:
# NaN/inf/negative overrides made the purported deadline unsafe (#402).
_PROTEOME_WARMUP_DEADLINE_SECONDS = 900.0


@dataclass(frozen=True)
class MappingTask:
    """Complete, picklable input contract for one canonical mapping pass.

    A task maps every supplied peptide using ``seed_lengths`` against one
    already-resolved proteome registry entry. Carrying the registry ``entry``
    explicitly prevents worker processes from repeating a UniProt lookup. The
    worker is deliberately cache-only: all permitted downloads happen in the
    supervised warm-up phase before mapping begins.

    ``peptides`` is a tuple rather than a length-keyed mapping: one seed index
    serves all peptide lengths, and preserving that fact in the task shape
    prevents the pre-#398 per-length rebuild design from creeping back in.

    Attributes
    ----------
    canonical
        Stable label written to mapping rows and progress output.
    entry
        Already-resolved registry entry (Ensembl species or UniProt UPID).
    peptides
        Unique peptide sequences to map in one pass.
    seed_lengths
        K-mer lengths used to construct the index, normally the single
        :data:`~hitlist.proteome.SEED_KMER_LENGTH` value.
    release
        Ensembl release used for Ensembl-backed entries.
    flank
        Number of source-protein residues retained on each side of a hit.
    ensembl_gene_biotypes
        Complete translated-gene inclusion policy for Ensembl-backed tasks.
        It is explicit in the process boundary so a worker cannot silently
        fall back to the historical protein-coding-only index.
    """

    canonical: str
    entry: dict
    peptides: tuple[str, ...]
    seed_lengths: tuple[int, ...]
    release: int
    flank: int
    ensembl_gene_biotypes: tuple[str, ...]


@dataclass
class MappingResult:
    """Named output contract for one :class:`MappingTask` execution.

    ``mapping_frame`` is ``None`` only when the requested proteome was not
    available. A successfully searched proteome with zero hits returns an empty
    frame, keeping "not searched" distinct from "searched, no matches".
    """

    canonical: str
    mapping_frame: pd.DataFrame | None
    n_matched_peptides: int
    n_input_peptides: int

    @property
    def proteome_available(self) -> bool:
        """Whether the proteome was searched, including a zero-hit search."""
        return self.mapping_frame is not None


def mappings_path() -> Path:
    """Path to the peptide mappings sidecar."""
    return data_dir() / "peptide_mappings.parquet"


def mappings_meta_path() -> Path:
    """Path to the mappings metadata JSON."""
    return data_dir() / "peptide_mappings_meta.json"


def _mapping_artifact_contract(
    *,
    release: int,
    use_uniprot: bool,
    fetch_missing: bool,
    flank: int,
) -> dict:
    """Behavior-defining contract stamped into every mappings sidecar."""
    return {
        "artifact_version": _MAPPING_ARTIFACT_VERSION,
        "release": int(release),
        "use_uniprot": bool(use_uniprot),
        "fetch_missing": bool(fetch_missing),
        "flank": int(flank),
        "seed_kmer_length": SEED_KMER_LENGTH,
        "ensembl_gene_biotypes": list(ENSEMBL_CODING_GENE_BIOTYPES),
        "columns": list(_MAPPING_COLUMNS),
    }


def is_mappings_built() -> bool:
    """Return True if peptide_mappings.parquet exists on disk."""
    return mappings_path().exists()


@lru_cache(maxsize=4)
def _known_gene_identifiers(_key: tuple) -> frozenset[str]:
    import pyarrow.parquet as pq

    t = pq.read_table(mappings_path(), columns=["gene_name", "gene_id"])
    out: set[str] = set()
    for col in ("gene_name", "gene_id"):
        for v in t.column(col).to_pylist():
            if v:
                out.add(str(v).upper())
    return frozenset(out)


def known_gene_identifiers() -> frozenset[str]:
    """Uppercased ``gene_name`` and ``gene_id`` values in peptide_mappings.parquet.

    The universe of genes the corpus has any peptide evidence for — used to tell
    an unrecognized gene symbol (likely a typo / not in our proteome) apart from
    a recognized gene that simply has no matching observations.  Empty frozenset
    if mappings have not been built.  Keyed on the parquet's size+mtime so a
    rebuild invalidates the cache.
    """
    if not is_mappings_built():
        return frozenset()
    st = mappings_path().stat()
    return _known_gene_identifiers((st.st_size, st.st_mtime_ns))


def _obs_fingerprint() -> dict:
    """Fingerprint both indexes the mappings were built from.

    The mappings sidecar covers peptides from observations.parquet AND
    binding.parquet, so both must invalidate the cache when they change.
    """
    from .observations import binding_path, observations_path

    fp: dict = {}
    for label, p in (("observations", observations_path()), ("binding", binding_path())):
        if p.exists():
            stat = p.stat()
            fp[label] = {"path": str(p), "size": stat.st_size, "mtime": stat.st_mtime}
    return fp


def _cache_is_valid(
    *,
    release: int,
    use_uniprot: bool,
    fetch_missing: bool,
    flank: int,
) -> bool:
    meta_path = mappings_meta_path()
    if not meta_path.exists() or not mappings_path().exists():
        return False
    try:
        stored = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    expected_contract = _mapping_artifact_contract(
        release=release,
        use_uniprot=use_uniprot,
        fetch_missing=fetch_missing,
        flank=flank,
    )
    return (
        stored.get("observations") == _obs_fingerprint()
        and stored.get("contract") == expected_contract
        and not stored.get("unavailable_proteomes")
    )


def load_peptide_mappings(
    peptide: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    gene_biotype: str | list[str] | None = None,
    protein_id: str | list[str] | None = None,
    transcript_id: str | list[str] | None = None,
    is_canonical_transcript: bool | None = None,
    proteome: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the long-form peptide → protein mappings with optional filters.

    Filters are pushed down to pyarrow, so a query like ``gene_name="PRAME"``
    reads only the matching row groups.

    ``gene_biotype`` distinguishes conventional ``protein_coding`` mappings
    from germline IG/TR sources; FASTA-backed rows carry ``""``. Issue #141
    added ``transcript_id`` and ``is_canonical_transcript`` so
    callers can ask "give me only the canonical-transcript mapping rows
    for this peptide" or "give me every mapping row that came from
    ENST00000269305" without an in-memory post-filter.
    """
    path = mappings_path()
    if not path.exists():
        raise FileNotFoundError("Peptide mappings not built.  Run: hitlist build observations")

    def _as_list(v) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s for s in v if s]

    filters: list = []
    if peptide is not None:
        filters.append(("peptide", "in", _as_list(peptide)))
    if gene_name is not None:
        filters.append(("gene_name", "in", _as_list(gene_name)))
    if gene_id is not None:
        filters.append(("gene_id", "in", _as_list(gene_id)))
    if gene_biotype is not None:
        filters.append(("gene_biotype", "in", _as_list(gene_biotype)))
    if protein_id is not None:
        filters.append(("protein_id", "in", _as_list(protein_id)))
    if transcript_id is not None:
        filters.append(("transcript_id", "in", _as_list(transcript_id)))
    if is_canonical_transcript is not None:
        filters.append(("is_canonical_transcript", "=", bool(is_canonical_transcript)))
    if proteome is not None:
        filters.append(("proteome", "in", _as_list(proteome)))

    return pd.read_parquet(path, columns=columns, filters=filters if filters else None)


def _flanking_rows_to_mapping_rows(
    flanking: pd.DataFrame, proteome_label: str, proteome_source: str
) -> pd.DataFrame:
    """Convert the output of ProteomeIndex.map_peptides() to mapping rows.

    ``map_peptides`` already returns ONE ROW per (peptide, protein, position) —
    this function just normalizes column names and tags the proteome.
    """
    if flanking.empty:
        return pd.DataFrame(columns=_MAPPING_COLUMNS)
    base_cols = [
        "peptide",
        "protein_id",
        "gene_name",
        "gene_id",
        "position",
        "n_flank",
        "c_flank",
    ]
    df = flanking[base_cols].copy()
    # Issue #141: ProteomeIndex.map_peptides emits transcript_id and
    # is_canonical_transcript on Ensembl-backed indexes; older fixtures
    # / older proteome-index instances without those columns get safe
    # defaults so the parquet schema stays uniform across backends.
    if "transcript_id" in flanking.columns:
        df["transcript_id"] = flanking["transcript_id"].fillna("").astype(str)
    else:
        df["transcript_id"] = ""
    if "gene_biotype" in flanking.columns:
        df["gene_biotype"] = flanking["gene_biotype"].fillna("").astype(str)
    else:
        df["gene_biotype"] = ""
    if "is_canonical_transcript" in flanking.columns:
        df["is_canonical_transcript"] = flanking["is_canonical_transcript"].astype(bool)
    else:
        df["is_canonical_transcript"] = False
    df["proteome"] = proteome_label
    df["proteome_source"] = proteome_source
    return df[list(_MAPPING_COLUMNS)]


def annotate_observations_with_genes(obs: pd.DataFrame, mappings: pd.DataFrame) -> pd.DataFrame:
    """Add central semicolon-joined gene/protein columns to an observations DataFrame.

    - ``gene_names``: unique gene symbols for this peptide, joined by ``;``
    - ``gene_ids``:   unique Ensembl gene IDs, joined by ``;``
    - ``protein_ids``: unique protein IDs, joined by ``;``
    - ``n_source_proteins``: count of distinct protein matches (int)

    Multi-mapping is preserved (MAGEA4;MAGEA10 for shared peptides).
    """
    if mappings.empty:
        for col in ("gene_names", "gene_ids", "protein_ids"):
            obs[col] = ""
        obs["n_source_proteins"] = 0
        return obs

    def _join_unique(series: pd.Series) -> str:
        seen: list[str] = []
        for v in series.dropna():
            s = str(v).strip()
            if s and s not in seen:
                seen.append(s)
        return ";".join(seen)

    agg = mappings.groupby("peptide").agg(
        gene_names=("gene_name", _join_unique),
        gene_ids=("gene_id", _join_unique),
        protein_ids=("protein_id", _join_unique),
        n_source_proteins=("protein_id", "nunique"),
    )
    return obs.merge(agg, left_on="peptide", right_index=True, how="left").fillna(
        {"gene_names": "", "gene_ids": "", "protein_ids": "", "n_source_proteins": 0}
    )


def _proteome_group_key(entry: dict) -> str:
    """Cluster key for ordering canonicals so same-FASTA neighbors land
    adjacently (#107).

    Two canonicals that resolve to the same on-disk FASTA share a key,
    which means the ``from_fasta`` LRU cache hits on the second through
    Nth member of the group. Returned strings are sortable so callers can
    use the key directly in ``sorted(..., key=...)``.

    Bucketing rules:
      * ``kind="ensembl"`` — keyed by species. The ensembl path uses
        ``from_ensembl`` (not ``from_fasta``) so ordering doesn't help its
        cache, but we still cluster by species for build-log readability.
      * ``kind="uniprot"`` — keyed by ``proteome_id``. Strain variants
        sharing one UniProt proteome (multiple LCMV / SARS-CoV-2 lines)
        end up in one bucket, which is the whole point.
      * Other / missing — keyed by canonical name (essentially keeps
        original alphabetical order for the unrecognised tail).
    """
    kind = entry.get("kind", "")
    if kind == "ensembl":
        return f"0:ensembl:{entry.get('species', '')}"
    if kind == "uniprot":
        pid = entry.get("proteome_id", "") or ""
        return f"1:uniprot:{pid}"
    return f"2:other:{entry.get('canonical_species', '')}"


def build_peptide_mappings(
    release: int = 112,
    fetch_missing: bool = True,
    use_uniprot: bool = False,
    force: bool = False,
    flank: int = DEFAULT_FLANK,
    verbose: bool = True,
    obs_override: pd.DataFrame | None = None,
    binding_override: pd.DataFrame | None = None,
) -> Path:
    """Build ``peptide_mappings.parquet`` from the already-built observations table.

    Reads observations.parquet, collects unique peptides per organism (from
    ``source_organism`` / ``mhc_species`` with ``reference_proteomes``
    overrides), maps each against the appropriate reference proteome, and
    writes all (peptide, protein, position) hits to the sidecar.

    Parameters
    ----------
    obs_override, binding_override
        In-memory MS / binding frames supplied by the builder during an
        atomic rebuild (#105), letting mappings be computed before the
        parquets are written so the canonical files aren't briefly
        missing their ``gene_names`` column.  When None, falls back to
        reading from disk (the standalone / cache-only path).
    """
    from .builder import _collect_pmid_extra_proteomes
    from .downloads import fetch_proteome_by_upid, lookup_proteome
    from .observations import is_binding_built, is_built, load_binding, load_observations
    from .proteome import ProteomeIndex

    out = mappings_path()
    if obs_override is not None:
        # Builder path: frames are in-memory, parquets may not exist yet.
        cols = ["peptide", "source_organism", "mhc_species", "pmid"]
        obs = obs_override[cols].copy()
        if binding_override is not None and len(binding_override):
            obs = pd.concat([obs, binding_override[cols]], ignore_index=True)
    else:
        if not is_built():
            raise FileNotFoundError(
                "Observations table not built.  Run: hitlist build observations"
            )
        if not force and _cache_is_valid(
            release=release,
            use_uniprot=use_uniprot,
            fetch_missing=fetch_missing,
            flank=flank,
        ):
            if verbose:
                print(f"Peptide mappings already up to date: {out}")
            return out
        cols = ["peptide", "source_organism", "mhc_species", "pmid"]
        obs = load_observations(columns=cols)
        if is_binding_built():
            binding = load_binding(columns=cols)
            if len(binding):
                obs = pd.concat([obs, binding], ignore_index=True)
    print(
        f"\nBuilding peptide mappings for {len(obs):,} rows (MS + binding, "
        f"{obs['peptide'].nunique():,} unique peptides) ..."
    )

    organism = obs["source_organism"].astype(str).str.strip()
    organism = organism.where(organism != "", obs["mhc_species"].astype(str).str.strip())

    # ── Primary pass: group peptides by canonical source proteome ────────────
    # Registry resolution itself can contact UniProt for rare organisms. Keep
    # that network work inside the same supervised deadline as FASTA/GTF cache
    # warm-up; the parent and mapping workers only perform offline lookups.
    warmup_started = time.monotonic()
    unavailable_proteomes: set[str] = set()
    unique_organisms = sorted({org for org in organism if org})
    lookup_cache: dict[str, dict | None] = {
        org: lookup_proteome(
            org,
            use_uniprot=use_uniprot,
            allow_network=False,
        )
        for org in unique_organisms
    }
    unresolved = [org for org, entry in lookup_cache.items() if entry is None]
    if unresolved and use_uniprot and fetch_missing:
        resolution_tasks = [
            (
                f"Resolve {org}",
                (org,),
                {"kind": "resolve", "organism": org},
            )
            for org in unresolved
        ]
        unavailable_resolutions = _supervise_prefetch_tasks(
            resolution_tasks,
            release=release,
            verbose=verbose,
            deadline_seconds=_PROTEOME_WARMUP_DEADLINE_SECONDS,
        )
        unavailable_proteomes.update(unavailable_resolutions)
        for org in unresolved:
            if org not in unavailable_resolutions:
                lookup_cache[org] = lookup_proteome(
                    org,
                    use_uniprot=True,
                    allow_network=False,
                )

    species_to_peptides: dict[str, set[str]] = {}
    canonical_to_entry: dict[str, dict] = {}
    unmapped_organisms: dict[str, int] = {}
    for org, pep in zip(organism, obs["peptide"]):
        if not org:
            continue
        entry = lookup_cache.get(org)
        if entry is None:
            unmapped_organisms[org] = unmapped_organisms.get(org, 0) + 1
            continue
        canonical = entry.get("canonical_species", org)
        species_to_peptides.setdefault(canonical, set()).add(pep)
        # First lookup wins — canonicals are stable across organism
        # spellings (mhcgnomes-normalized).
        canonical_to_entry.setdefault(canonical, entry)

    all_mapping_dfs: list[pd.DataFrame] = []
    per_proteome_stats: list[tuple[str, int, int]] = []

    # The index is built at ONE seed length; peptide length is no longer the
    # same knob.  Any peptide at or above the seed is located by looking up
    # its first `k` residues and verifying the protein continues with the
    # rest (`ProteomeIndex.lookup`), so class II at 12-25 residues maps
    # through the same index class I does, at no extra memory cost.
    #
    # Indexing 8/9/10/11 separately bought nothing measurable: seed
    # selectivity is flat in k because multiplicity comes from isoforms
    # rather than sequence repetition (k=7 mean 3.74 hits, k=11 mean 3.49,
    # identical p99), while each length costs the same ~492 MB.  See
    # `hitlist.proteome.SEED_KMER_LENGTH` for the measurements.
    seed_lengths = (SEED_KMER_LENGTH,)

    # ── Build order: cluster canonicals by FASTA so adjacent tasks share an index ──
    # Strain-variant canonicals (e.g. multiple SARS-CoV-2 / LCMV) often
    # share one underlying FASTA via their UniProt proteome_id.  Sorting
    # by (group_key, canonical) clusters same-FASTA canonicals adjacently
    # so the from_fasta in-memory cache hits on the 2nd/3rd member of a
    # group.  Combined with chunksize=2 below, this keeps clustered
    # neighbors on the same worker (#107 + #249).
    ordered_canonicals = sorted(
        species_to_peptides,
        key=lambda c: (_proteome_group_key(canonical_to_entry.get(c, {})), c),
    )

    # Build a flat task list for the worker pool (#249).  One task per
    # canonical -- no bucketing by length, because one index serves every
    # length.
    #
    # Before #394 this filtered peptides to `default_lengths` and took a
    # `continue`, so every class II peptide in the corpus (1,395,872 rows,
    # 569,670 unique) was silently unmapped: no flanks, no position, no gene,
    # no protein. The only peptides dropped now are those below the seed --
    # length 2-6, 2,826 rows, which are not plausible MHC ligands.
    mapping_tasks: list[MappingTask] = []
    for canonical in ordered_canonicals:
        peptides = species_to_peptides[canonical]
        mappable = [p for p in peptides if len(p) >= SEED_KMER_LENGTH]
        if not mappable:
            per_proteome_stats.append((canonical, len(peptides), 0))
            continue
        mapping_tasks.append(
            MappingTask(
                canonical=canonical,
                entry=canonical_to_entry[canonical],
                peptides=tuple(mappable),
                seed_lengths=seed_lengths,
                release=release,
                flank=flank,
                ensembl_gene_biotypes=ENSEMBL_CODING_GENE_BIOTYPES,
            )
        )

    # Per-PMID reference-proteome overrides participate in the same supervised
    # cache warm-up as primary species. Their mapping still runs after the
    # primary worker pool, but it is cache-only by then.
    upid_to_peptides: dict[str, tuple[str, set[str]]] = {}
    pmid_extras = _collect_pmid_extra_proteomes()
    if pmid_extras:
        pmid_col = obs["pmid"]
        for pmid_int, upid_entries in pmid_extras.items():
            selected = pmid_col == pmid_int
            if not selected.any():
                continue
            peptides = set(obs.loc[selected, "peptide"].dropna())
            for extra_entry in upid_entries:
                upid = extra_entry["upid"]
                label = extra_entry["label"]
                if upid not in upid_to_peptides:
                    upid_to_peptides[upid] = (label, set())
                upid_to_peptides[upid][1].update(peptides)

    extra_cache_keys = {
        upid: f"{label} [{upid}]" for upid, (label, _peptides) in upid_to_peptides.items()
    }

    n_workers = _build_workers()
    # Cap workers at task count — more processes than work just adds fork overhead.
    effective_workers = min(n_workers, max(1, len(mapping_tasks)))

    # Pre-fetch all missing proteomes in a supervised child so workers don't
    # race on FASTA / GTF downloads and a blocked network/dependency call can
    # be terminated at the hard phase deadline (#402).
    # We pass (canonical_key, entry) pairs because the canonical KEY
    # labels output and results, while the resolved entry lets both the
    # prefetch child and mapping worker avoid repeating registry/network
    # resolution. Entries don't always carry `canonical_species`.
    if (mapping_tasks or upid_to_peptides) and fetch_missing:
        prefetch_entries = [(task.canonical, task.entry) for task in mapping_tasks]
        prefetch_entries.extend(
            (
                extra_cache_keys[upid],
                {"kind": "uniprot", "proteome_id": upid, "label": label},
            )
            for upid, (label, _peptides) in upid_to_peptides.items()
        )
        remaining_warmup_seconds = _PROTEOME_WARMUP_DEADLINE_SECONDS - (
            time.monotonic() - warmup_started
        )
        unavailable = _prefetch_proteomes_for_workers(
            prefetch_entries,
            release=release,
            verbose=verbose,
            deadline_seconds=remaining_warmup_seconds,
        )
        if unavailable:
            unavailable_proteomes.update(unavailable)
            retained_tasks: list[MappingTask] = []
            for task in mapping_tasks:
                if task.canonical in unavailable:
                    per_proteome_stats.append((task.canonical, len(task.peptides), 0))
                else:
                    retained_tasks.append(task)
            mapping_tasks = retained_tasks
            effective_workers = min(n_workers, max(1, len(mapping_tasks)))

    if verbose and mapping_tasks:
        print(
            f"\n  Mapping {len(mapping_tasks)} canonical proteome(s) "
            f"across {effective_workers} worker(s) ..."
        )

    if effective_workers == 1:
        # Sequential fallback — identical to pre-#249 behavior.  Useful for
        # debugging, deterministic profiling, and HITLIST_BUILD_WORKERS=1.
        for result in (_per_canonical_mapping_worker(task) for task in mapping_tasks):
            if not result.proteome_available:
                unavailable_proteomes.add(result.canonical)
            if result.mapping_frame is not None:
                all_mapping_dfs.append(result.mapping_frame)
            per_proteome_stats.append(
                (
                    result.canonical,
                    result.n_input_peptides,
                    result.n_matched_peptides,
                )
            )
            if verbose:
                print(
                    f"    [{result.canonical}] matched "
                    f"{result.n_matched_peptides:,} / "
                    f"{result.n_input_peptides:,} peptides"
                )
    else:
        from concurrent.futures import ProcessPoolExecutor

        # chunksize=2 keeps adjacent FASTA-clustered tasks on the same
        # worker, recovering some of #107's in-memory LRU benefit that a
        # default chunksize=1 round-robin would scatter.  Strain-variant
        # clusters of size ≥ 2 (the common case) get the 2nd member's
        # index from the same-process cache rather than rebuilding.
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            for result in pool.map(_per_canonical_mapping_worker, mapping_tasks, chunksize=2):
                if not result.proteome_available:
                    unavailable_proteomes.add(result.canonical)
                if result.mapping_frame is not None:
                    all_mapping_dfs.append(result.mapping_frame)
                per_proteome_stats.append(
                    (
                        result.canonical,
                        result.n_input_peptides,
                        result.n_matched_peptides,
                    )
                )
                if verbose:
                    print(
                        f"    [{result.canonical}] matched "
                        f"{result.n_matched_peptides:,} / "
                        f"{result.n_input_peptides:,} peptides"
                    )

    # ── Extra proteomes (per-PMID reference_proteomes overrides) ─────────────
    if upid_to_peptides:
        if verbose:
            n_extra_peps = sum(len(p) for _, p in upid_to_peptides.values())
            print(
                f"\n  [extras] mapping {len(upid_to_peptides)} per-PMID override "
                f"proteome(s) against {n_extra_peps:,} peptides (multi-counted across "
                "PMIDs sharing proteomes)"
            )
        for upid, (label, peptides) in upid_to_peptides.items():
            cache_key = extra_cache_keys[upid]
            if cache_key in unavailable_proteomes:
                per_proteome_stats.append((label, len(peptides), 0))
                continue
            path = fetch_proteome_by_upid(
                upid,
                label=label,
                verbose=verbose,
                # All network fetches ran under the supervised phase deadline
                # above. This path is deliberately cache-only so a second
                # unbounded attempt cannot escape into the mapping phase.
                fetch_missing=False,
            )
            if path is None or not path.exists():
                unavailable_proteomes.add(cache_key)
                per_proteome_stats.append((label, len(peptides), 0))
                continue
            idx = ProteomeIndex.from_fasta(path, verbose=False)
            flanking = idx.map_peptides(sorted(peptides), flank=flank, verbose=False)
            df = _flanking_rows_to_mapping_rows(
                flanking, proteome_label=label, proteome_source="reference_proteomes"
            )
            all_mapping_dfs.append(df)
            per_proteome_stats.append((label, len(peptides), int(df["peptide"].nunique())))
            if verbose:
                print(
                    f"    [{label}] matched {df['peptide'].nunique():,} / {len(peptides):,} peptides"
                )

    # ── Consolidate and write ────────────────────────────────────────────────
    if all_mapping_dfs:
        mappings = pd.concat(all_mapping_dfs, ignore_index=True)
    else:
        mappings = pd.DataFrame(columns=list(_MAPPING_COLUMNS))

    # Preserve multi-mapping: dedupe only exact duplicates (same peptide,
    # protein, position, proteome).
    mappings = mappings.drop_duplicates(subset=["peptide", "protein_id", "position", "proteome"])

    mappings.to_parquet(out, index=False)

    meta = {
        "observations": _obs_fingerprint(),
        "contract": _mapping_artifact_contract(
            release=release,
            use_uniprot=use_uniprot,
            fetch_missing=fetch_missing,
            flank=flank,
        ),
        # A transient fetch/prewarm failure must not become a permanently
        # "valid" incomplete artifact. Any unavailable input makes the next
        # invocation retry cache validation/build (#402/#404).
        "unavailable_proteomes": sorted(unavailable_proteomes),
        "n_rows": len(mappings),
        "n_peptides": int(mappings["peptide"].nunique()) if len(mappings) else 0,
        "n_proteomes": int(mappings["proteome"].nunique()) if len(mappings) else 0,
        "per_proteome": {
            label: {"peptides_searched": n_pep, "peptides_matched": n_mapped}
            for label, n_pep, n_mapped in per_proteome_stats
        },
        "unmapped_organisms": dict(sorted(unmapped_organisms.items(), key=lambda x: -x[1])[:20]),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    mappings_meta_path().write_text(json.dumps(meta, indent=2, default=str) + "\n")

    if verbose:
        print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        print(f"  Rows:            {len(mappings):,}")
        print(f"  Unique peptides: {meta['n_peptides']:,}")
        print(f"  Proteomes:       {meta['n_proteomes']}")
    return out


# ── Parallel mapping execution (#249) ─────────────────────────────────────
#
# Per-species index builds are CPU-bound and embarrassingly parallel —
# each canonical's mapping pass is independent of every other.  cProfile
# of a cold full build (#176) showed ~67% of total wall time spent in
# the per-species mapping block; running 4 canonicals concurrently on a
# 10-core machine cuts that block roughly in proportion to the worker
# count.
#
# The on-disk cache shipped in #246 / #251 means warm builds barely
# touch this code path (loads from pickle), so the parallelism win is
# concentrated on the cold-build path.
#
# Memory ceiling: each worker builds exactly one seed ProteomeIndex and maps
# every peptide length through it, preserving #109's one-index-per-worker
# invariant. With the default of 4 workers, peak resident is ~ 4x
# largest-single-length-index ~ 4 x 3 GB = 12 GB — safely under the
# 16 GB / 32 GB host class targets.


def _build_workers() -> int:
    """Worker count for :func:`build_peptide_mappings` parallelism (#249).

    Defaults to ``min(4, cpu_count // 2)`` so peak resident stays
    bounded by ``workers x largest-single-length-index``.  Override via
    ``HITLIST_BUILD_WORKERS=N``.  Set to ``1`` for the sequential
    fallback (identical behavior to pre-#249).

    The override is NOT capped at ``cpu_count``: a value of 16 on an
    8-core box will spawn 16 workers and likely OOM on the human pass.
    Treat this as a power-user knob — the default is the safe choice.
    """
    raw = os.environ.get("HITLIST_BUILD_WORKERS")
    if raw is not None:
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            pass
    return min(4, max(1, (os.cpu_count() or 1) // 2))


def _prefetch_proteomes_for_workers(
    canonicals_and_entries: list[tuple[str, dict]],
    release: int,
    verbose: bool,
    deadline_seconds: float = _PROTEOME_WARMUP_DEADLINE_SECONDS,
) -> set[str]:
    """Eagerly download/index every proteome the workers will need (#249).

    Workers run in fresh processes (``ProcessPoolExecutor`` defaults to spawn
    on macOS, fork on Linux) and don't share download locks. On a first-ever
    cold build, two workers needing the same UniProt FASTA or pyensembl GTF
    could race on the shared download/index paths. We avoid both races by
    warming the on-disk caches sequentially in one supervised child before
    dispatching mapping tasks.

    Takes ``(canonical_key, entry)`` pairs because the key is the stable label
    for logs/results, while the explicit resolved entry prevents child/worker
    processes from repeating UniProt REST resolution. The entry's internal
    ``canonical_species`` field is not always set.

    No-op when caches are warm (the typical case): each call is idempotent and
    exits in milliseconds when the file/db is already present. The warm-up runs
    in a supervised child process with one absolute wall-clock deadline. A
    blocked call can therefore be terminated rather than wedging the parent.

    Returns the canonical keys whose cache warm-up failed, timed out, or was
    left unattempted after the deadline. Callers must skip those tasks rather
    than retrying the same operation silently inside a mapping worker.

    Note: this warms the FASTA / GTF on disk only — it does NOT pre-build
    the on-disk pickle index from #246/#251.  When the pickle cache is
    cold, multiple workers may redundantly rebuild the same index;
    ``_write_index_to_disk`` uses an atomic ``os.replace`` so concurrent
    writes don't corrupt, just waste CPU.  Pre-building serially in the
    parent would defeat the parallelism this PR adds.
    """
    # Deduplicate UniProt inputs by UPID and Ensembl inputs by species. One
    # successful cache write serves every canonical alias in the group.
    unique = dict(canonicals_and_entries)
    tasks: list[tuple[str, tuple[str, ...], dict]] = []
    uniprot_groups: dict[str, list[str]] = {}
    for canonical, entry in unique.items():
        if entry.get("kind") != "ensembl":
            group_key = entry.get("proteome_id") or f"canonical:{canonical}"
            uniprot_groups.setdefault(group_key, []).append(canonical)
    for _group_key, canonicals in sorted(uniprot_groups.items()):
        aliases = tuple(sorted(canonicals))
        entry = unique[aliases[0]]
        label = entry.get("label") or aliases[0]
        tasks.append((label, aliases, entry))

    ensembl_groups: dict[str, list[str]] = {}
    for canonical, entry in unique.items():
        if entry.get("kind") == "ensembl":
            ensembl_groups.setdefault(entry.get("species", "human"), []).append(canonical)
    for species, canonicals in sorted(ensembl_groups.items()):
        tasks.append(
            (
                f"Ensembl {species} r{release}",
                tuple(sorted(canonicals)),
                {"kind": "ensembl", "species": species},
            )
        )

    if verbose and tasks:
        print(
            f"  Pre-warming {len(tasks)} proteome cache entr{'y' if len(tasks) == 1 else 'ies'} "
            "in a supervised process ..."
        )
    return _supervise_prefetch_tasks(
        tasks,
        release=release,
        verbose=verbose,
        deadline_seconds=deadline_seconds,
    )


def _prefetch_worker(
    label: str,
    entry: dict,
    cache_dir: str,
    release: int,
) -> tuple[str, bool, str]:
    """Perform one cache warm-up in a child process and report its outcome."""
    from .downloads import fetch_proteome_by_upid, set_data_dir

    set_data_dir(cache_dir)
    try:
        if entry.get("kind") == "resolve":
            from .downloads import _uniprot_cache, lookup_proteome

            organism = entry["organism"]
            resolved = lookup_proteome(
                organism,
                use_uniprot=True,
                allow_network=True,
            )
            # A genuine empty result is cached negatively and is a successful
            # resolution. A transport failure deliberately leaves no cache
            # entry, so the build must remain retryable.
            if resolved is None and organism not in _uniprot_cache():
                raise RuntimeError("UniProt resolution failed transiently")
        elif entry.get("kind") == "ensembl":
            from pyensembl import EnsemblRelease

            ensembl = EnsemblRelease(release, species=entry.get("species", "human"))
            ensembl.download()
            ensembl.index()
        else:
            path = fetch_proteome_by_upid(
                entry["proteome_id"],
                label=label,
                verbose=False,
                fetch_missing=True,
            )
            if path is None or not path.exists():
                raise RuntimeError("proteome FASTA was not cached")
        return label, True, ""
    except Exception as error:
        return label, False, f"{type(error).__name__}: {error}"


def _supervise_prefetch_tasks(
    tasks: list[tuple[str, tuple[str, ...], dict]],
    *,
    release: int,
    verbose: bool,
    deadline_seconds: float = _PROTEOME_WARMUP_DEADLINE_SECONDS,
    worker_target=None,
) -> set[str]:
    """Run cache warm-ups behind a killable absolute wall-clock deadline."""
    if not tasks:
        return set()
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        unavailable = {canonical for _, canonicals, _ in tasks for canonical in canonicals}
        print(
            "    proteome warm-up deadline is invalid or already exhausted; skipping "
            f"{len(unavailable)} proteome(s).",
            flush=True,
        )
        return unavailable

    import multiprocessing as mp

    from .downloads import data_dir

    target = worker_target or _prefetch_worker
    context = mp.get_context("spawn")
    unavailable: set[str] = set()
    started = time.monotonic()
    pool = None
    aborted = False

    try:
        pool = context.Pool(processes=1)
        for i, (label, canonicals, entry) in enumerate(tasks, 1):
            elapsed = time.monotonic() - started
            remaining_seconds = deadline_seconds - elapsed
            if remaining_seconds <= 0:
                unavailable.update(c for _, keys, _ in tasks[i - 1 :] for c in keys)
                print(
                    f"    prefetch deadline of {deadline_seconds:.0f}s exhausted; "
                    f"skipping {len(unavailable)} proteome(s).",
                    flush=True,
                )
                aborted = True
                break

            if verbose:
                print(f"    [{i}/{len(tasks)}] {label} ...", flush=True)
            try:
                result = pool.apply_async(
                    target,
                    (label, entry, str(data_dir()), release),
                )
                returned_label, succeeded, error_text = result.get(timeout=remaining_seconds)
            except mp.TimeoutError:
                unavailable.update(c for _, keys, _ in tasks[i - 1 :] for c in keys)
                print(
                    f"    [{label}] prefetch timed out at the {deadline_seconds:.0f}s "
                    f"phase deadline; skipping it and {len(tasks) - i} remaining "
                    "cache request(s).",
                    flush=True,
                )
                aborted = True
                break
            except Exception as error:
                unavailable.update(canonicals)
                print(
                    f"    [{label}] prefetch worker failed ({error}); skipped.",
                    flush=True,
                )
                continue
            if returned_label != label:
                unavailable.update(canonicals)
                print(
                    f"    [{label}] prefetch protocol mismatch ({returned_label!r}); skipped.",
                    flush=True,
                )
            elif not succeeded:
                unavailable.update(canonicals)
                print(f"    [{label}] prefetch skipped: {error_text}", flush=True)
    except (OSError, RuntimeError) as error:
        unavailable.update(c for _, canonicals, _ in tasks for c in canonicals)
        print(f"    prefetch worker could not start: {error}; skipping all proteomes.", flush=True)
        aborted = True
    finally:
        if pool is not None:
            if aborted:
                pool.terminate()
            else:
                pool.close()
            pool.join()

    return unavailable


def _per_canonical_mapping_worker(task: MappingTask) -> MappingResult:
    """Execute one canonical proteome mapping task.

    Module-level so :class:`concurrent.futures.ProcessPoolExecutor` can
    pickle and dispatch it across workers. This is the single implementation
    used by both sequential and process-pool execution: build/load one seed
    index, map every peptide length through it, normalize the long-form rows,
    and report coverage against the original input denominator.

    Workers run with ``verbose=False`` to avoid interleaved progress
    spam in the parent terminal — the orchestrator emits one summary
    line per canonical on completion if it wants progress output.

    If the already-resolved proteome cannot be loaded (missing offline FASTA,
    unavailable pyensembl data, corrupt cache), the task returns an empty
    result with the full input count preserved as its coverage denominator.
    """
    n_input_peptides = len(task.peptides)
    idx = _build_species_index(
        canonical=task.canonical,
        release=task.release,
        use_uniprot=False,
        verbose=False,
        lengths=task.seed_lengths,
        entry=task.entry,
        gene_biotypes=task.ensembl_gene_biotypes,
        # Network work belongs exclusively to the supervised warm-up. Keeping
        # mapping workers cache-only prevents a vanished/inconsistent cache
        # from turning into an unbounded retry outside the phase deadline.
        fetch_missing=False,
    )
    if idx is None:
        return MappingResult(
            canonical=task.canonical,
            mapping_frame=None,
            n_matched_peptides=0,
            n_input_peptides=n_input_peptides,
        )

    # One index, one pass, every length.  This used to loop over lengths and
    # rebuild the index each time -- four builds per canonical to answer what
    # one seed index answers in a single pass.  Peak per-worker RSS is still
    # bounded by a single index, so #109's invariant holds by construction
    # rather than by remembering to `del` between iterations.
    flanking = idx.map_peptides(
        sorted(task.peptides),
        flank=task.flank,
        verbose=False,
    )
    df = _flanking_rows_to_mapping_rows(
        flanking,
        proteome_label=task.canonical,
        proteome_source="species",
    )
    matched = set(flanking["peptide"].unique()) if len(flanking) else set()
    del idx, flanking

    return MappingResult(
        canonical=task.canonical,
        mapping_frame=df,
        n_matched_peptides=len(matched),
        n_input_peptides=n_input_peptides,
    )


def _build_species_index(
    canonical: str,
    release: int,
    use_uniprot: bool,
    verbose: bool,
    lengths: tuple[int, ...] = (SEED_KMER_LENGTH,),
    entry: dict | None = None,
    fetch_missing: bool = True,
    gene_biotypes: tuple[str, ...] = ENSEMBL_CODING_GENE_BIOTYPES,
):
    """Build a proteome index from a resolved entry or canonical species.

    ``entry`` lets callers separate registry/network resolution from index
    construction. When it is omitted, ``use_uniprot`` controls dynamic
    resolution and ``fetch_missing`` controls both resolution and FASTA
    downloads. ``gene_biotypes`` is forwarded only to Ensembl indexes and
    makes the translated-gene policy part of the worker's explicit contract.
    Passing a resolved entry with ``fetch_missing=False`` is the mapping
    worker's strictly cache-only path.

    Returns ``None`` if resolution, cache access, or index construction fails.
    """
    from .downloads import fetch_proteome_by_upid, lookup_proteome
    from .proteome import ProteomeIndex

    if entry is None:
        entry = lookup_proteome(
            canonical,
            use_uniprot=use_uniprot,
            allow_network=fetch_missing,
        )
    if entry is None:
        return None

    if entry["kind"] == "ensembl":
        species = entry.get("species", "human")
        try:
            return ProteomeIndex.from_ensembl(
                release=release,
                species=species,
                lengths=lengths,
                verbose=verbose,
                gene_biotypes=gene_biotypes,
            )
        except Exception as e:
            if verbose:
                print(f"    [{canonical}] pyensembl failed: {e}")
            return None

    try:
        path = fetch_proteome_by_upid(
            entry["proteome_id"],
            label=canonical,
            verbose=verbose,
            fetch_missing=fetch_missing,
        )
        if path is None or not path.exists():
            return None
        return ProteomeIndex.from_fasta(path, lengths=lengths, verbose=False)
    except Exception as error:
        if verbose:
            print(f"    [{canonical}] UniProt proteome failed: {error}")
        return None
