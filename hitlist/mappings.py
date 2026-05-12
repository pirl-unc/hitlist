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
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .downloads import data_dir

_MAPPING_COLUMNS = (
    "peptide",
    "protein_id",
    "gene_name",
    "gene_id",
    # Issue #141: transcript_id is a first-class column distinct from
    # protein_id (which now carries ENSP for Ensembl-backed indexes
    # rather than ENST).  is_canonical_transcript flags the longest
    # protein-coding transcript per gene as the canonical proxy.
    # FASTA-backed indexes leave transcript_id="" and the flag False.
    "transcript_id",
    "is_canonical_transcript",
    "position",
    "n_flank",
    "c_flank",
    "proteome",
    "proteome_source",
)


def mappings_path() -> Path:
    """Path to the peptide mappings sidecar."""
    return data_dir() / "peptide_mappings.parquet"


def mappings_meta_path() -> Path:
    """Path to the mappings metadata JSON."""
    return data_dir() / "peptide_mappings_meta.json"


def is_mappings_built() -> bool:
    """Return True if peptide_mappings.parquet exists on disk."""
    return mappings_path().exists()


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


def _cache_is_valid() -> bool:
    meta_path = mappings_meta_path()
    if not meta_path.exists() or not mappings_path().exists():
        return False
    try:
        stored = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return stored.get("observations") == _obs_fingerprint()


def load_peptide_mappings(
    peptide: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    protein_id: str | list[str] | None = None,
    transcript_id: str | list[str] | None = None,
    is_canonical_transcript: bool | None = None,
    proteome: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the long-form peptide → protein mappings with optional filters.

    Filters are pushed down to pyarrow, so a query like ``gene_name="PRAME"``
    reads only the matching row groups.

    Issue #141 added ``transcript_id`` and ``is_canonical_transcript`` so
    callers can ask "give me only the canonical-transcript mapping rows
    for this peptide" or "give me every mapping row that came from
    ENST00000269305" without an in-memory post-filter.
    """
    path = mappings_path()
    if not path.exists():
        raise FileNotFoundError("Peptide mappings not built.  Run: hitlist data build")

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
    flank: int = 10,
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
            raise FileNotFoundError("Observations table not built.  Run: hitlist data build")
        if not force and _cache_is_valid():
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
    lookup_cache: dict[str, dict | None] = {}

    def _lookup(org: str) -> dict | None:
        if org in lookup_cache:
            return lookup_cache[org]
        entry = lookup_proteome(org, use_uniprot=use_uniprot)
        lookup_cache[org] = entry
        return entry

    species_to_peptides: dict[str, set[str]] = {}
    canonical_to_entry: dict[str, dict] = {}
    unmapped_organisms: dict[str, int] = {}
    for org, pep in zip(organism, obs["peptide"]):
        if not org:
            continue
        entry = _lookup(org)
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

    # MHC-I peptide lengths.  Length-on-demand happens per-worker inside
    # _per_canonical_mapping_worker so peak per-worker RSS stays bounded
    # by ONE single-length index (preserves the #109 invariant).
    default_lengths = (8, 9, 10, 11)

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

    # Bucket peptides per canonical, filter out canonicals with no
    # MHC-I-compatible lengths, and build a flat task list for the worker
    # pool (#249).  Empty-length canonicals are recorded synchronously so
    # they don't take a worker slot for a no-op.
    mapping_tasks: list[tuple] = []
    for canonical in ordered_canonicals:
        peptides = species_to_peptides[canonical]
        peptides_by_len: dict[int, list[str]] = {}
        for p in peptides:
            peptides_by_len.setdefault(len(p), []).append(p)
        lengths_in_query = tuple(sorted(L for L in peptides_by_len if L in default_lengths))
        if not lengths_in_query:
            # MHC-II peptides at length 12+ aren't indexed here — pre-#249 behavior.
            per_proteome_stats.append((canonical, len(peptides), 0))
            continue
        mapping_tasks.append(
            (canonical, peptides_by_len, lengths_in_query, release, use_uniprot, flank)
        )

    n_workers = _build_workers()
    # Cap workers at task count — more processes than work just adds fork overhead.
    effective_workers = min(n_workers, max(1, len(mapping_tasks)))

    # Pre-fetch all proteomes in the parent so workers don't race on
    # FASTA / GTF downloads.  No-op when caches are warm (the common case).
    # We pass (canonical_key, entry) pairs because the canonical KEY
    # (which is what the worker hands to fetch_species_proteome via
    # _build_species_index) is what must be deduped — entries don't
    # always carry an explicit `canonical_species` field.
    if mapping_tasks and effective_workers > 1:
        _prefetch_proteomes_for_workers(
            [(t[0], canonical_to_entry[t[0]]) for t in mapping_tasks],
            release=release,
            use_uniprot=use_uniprot,
            verbose=verbose,
        )

    if verbose and mapping_tasks:
        print(
            f"\n  Mapping {len(mapping_tasks)} canonical proteome(s) "
            f"across {effective_workers} worker(s) ..."
        )

    if effective_workers == 1:
        # Sequential fallback — identical to pre-#249 behavior.  Useful for
        # debugging, deterministic profiling, and HITLIST_BUILD_WORKERS=1.
        for canonical, dfs, n_matched, n_total in (
            _per_canonical_mapping_worker(t) for t in mapping_tasks
        ):
            all_mapping_dfs.extend(dfs)
            per_proteome_stats.append((canonical, n_total, n_matched))
            if verbose:
                print(f"    [{canonical}] matched {n_matched:,} / {n_total:,} peptides")
    else:
        from concurrent.futures import ProcessPoolExecutor

        # chunksize=2 keeps adjacent FASTA-clustered tasks on the same
        # worker, recovering some of #107's in-memory LRU benefit that a
        # default chunksize=1 round-robin would scatter.  Strain-variant
        # clusters of size ≥ 2 (the common case) get the 2nd member's
        # index from the same-process cache rather than rebuilding.
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            for canonical, dfs, n_matched, n_total in pool.map(
                _per_canonical_mapping_worker, mapping_tasks, chunksize=2
            ):
                all_mapping_dfs.extend(dfs)
                per_proteome_stats.append((canonical, n_total, n_matched))
                if verbose:
                    print(f"    [{canonical}] matched {n_matched:,} / {n_total:,} peptides")

    # ── Extra proteomes (per-PMID reference_proteomes overrides) ─────────────
    pmid_extras = _collect_pmid_extra_proteomes()
    if pmid_extras:
        pmid_col = obs["pmid"]
        upid_to_peptides: dict[str, tuple[str, set[str]]] = {}
        for pmid_int, upid_entries in pmid_extras.items():
            sel = pmid_col == pmid_int
            if not sel.any():
                continue
            peptides = set(obs.loc[sel, "peptide"].dropna())
            for e in upid_entries:
                upid = e["upid"]
                label = e["label"]
                if upid not in upid_to_peptides:
                    upid_to_peptides[upid] = (label, set())
                upid_to_peptides[upid][1].update(peptides)

        if upid_to_peptides and verbose:
            n_extra_peps = sum(len(p) for _, p in upid_to_peptides.values())
            print(
                f"\n  [extras] mapping {len(upid_to_peptides)} per-PMID override "
                f"proteome(s) against {n_extra_peps:,} peptides (multi-counted across "
                "PMIDs sharing proteomes)"
            )
        for upid, (label, peptides) in upid_to_peptides.items():
            path = fetch_proteome_by_upid(upid, label=label, verbose=verbose)
            if path is None or not path.exists():
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
# Memory ceiling: each worker holds at most one single-length
# ProteomeIndex at a time — the per-length build / drop loop lives
# inside _per_canonical_mapping_worker below, preserving #109's
# invariant.  With the default of 4 workers, peak resident is ~ 4x
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
    use_uniprot: bool,
    verbose: bool,
) -> None:
    """Eagerly download/index every proteome the workers will need (#249).

    Workers run in fresh processes (``ProcessPoolExecutor`` defaults to
    spawn on macOS, fork on Linux) and don't share download locks.  On
    a first-ever cold build, two workers needing the SAME UniProt FASTA
    could race on the ``.tmp`` file (``fetch_species_proteome`` writes
    via ``urllib.request.urlretrieve`` to a non-unique tmp path).  Two
    workers needing the SAME pyensembl GTF could race on its download +
    SQLite index build.  We avoid both races by warming the on-disk
    caches sequentially in the parent before dispatching tasks.

    Takes ``(canonical_key, entry)`` pairs because the canonical KEY
    (the dict key the worker eventually passes to
    :func:`fetch_species_proteome` via :func:`_build_species_index`) is
    what must be deduped — the entry's internal ``canonical_species``
    field isn't always set (the orchestrator falls back to the source
    organism string when missing).

    ``use_uniprot`` mirrors :func:`build_peptide_mappings`'s parameter
    so the pre-fetch resolves canonicals against the same registry path
    the workers will use; otherwise the pre-fetch could silently miss
    species that workers DO need.

    No-op when caches are warm (the typical case): each call is
    idempotent and exits in milliseconds when the file/db is already
    present.  Failures are tolerated — the worker will hit the same
    code path and surface the error there.

    Note: this warms the FASTA / GTF on disk only — it does NOT pre-build
    the on-disk pickle index from #246/#251.  When the pickle cache is
    cold, multiple workers may redundantly rebuild the same index;
    ``_write_index_to_disk`` uses an atomic ``os.replace`` so concurrent
    writes don't corrupt, just waste CPU.  Pre-building serially in the
    parent would defeat the parallelism this PR adds.
    """
    from .downloads import fetch_species_proteome

    # UniProt FASTAs: dedup by canonical KEY (not entry.canonical_species
    # — that field may be absent when the orchestrator's dict key fell
    # back to the source organism).  fetch_species_proteome dedupes
    # further by UPID inside (multiple strain canonicals → one download).
    uniprot_canonicals = sorted(
        {canonical for canonical, entry in canonicals_and_entries if entry.get("kind") != "ensembl"}
    )
    if uniprot_canonicals and verbose:
        print(
            f"  Pre-fetching {len(uniprot_canonicals)} UniProt FASTA(s) "
            f"in parent (avoids worker download races) ..."
        )
    for canonical in uniprot_canonicals:
        try:
            fetch_species_proteome(canonical, verbose=False, use_uniprot=use_uniprot)
        except Exception as e:
            if verbose:
                print(f"    [{canonical}] pre-fetch skipped: {e}")

    # Ensembl species: pre-trigger pyensembl download + SQLite index build
    # once per (release, species) so workers find the local cache populated
    # rather than racing on it.  download() / index() are idempotent.
    ensembl_species = sorted(
        {
            entry.get("species", "human")
            for _canonical, entry in canonicals_and_entries
            if entry.get("kind") == "ensembl"
        }
    )
    if ensembl_species:
        if verbose:
            print(
                f"  Pre-warming {len(ensembl_species)} Ensembl release(s) "
                f"in parent (avoids worker GTF races) ..."
            )
        try:
            from pyensembl import EnsemblRelease
        except ImportError:
            return
        for species in ensembl_species:
            try:
                try:
                    ensembl = EnsemblRelease(release, species=species)
                except TypeError:
                    if species != "human":
                        continue
                    ensembl = EnsemblRelease(release)
                # download() then index() — both idempotent, cheap when warm.
                ensembl.download()
                ensembl.index()
            except Exception as e:
                if verbose:
                    print(f"    [{species}] pre-warm skipped: {e}")


def _per_canonical_mapping_worker(
    args: tuple,
) -> tuple[str, list[pd.DataFrame], int, int]:
    """Build + map for one canonical species across its requested lengths.

    Module-level so :class:`concurrent.futures.ProcessPoolExecutor` can
    pickle and dispatch it across workers.  Logically equivalent to the
    inner loop body of the pre-#249 sequential code:

        for length in lengths_in_query:
            idx = _build_species_index(canonical, ..., lengths=(length,))
            flanking = idx.map_peptides(...)
            df = _flanking_rows_to_mapping_rows(...)
            del idx, flanking

    Returns ``(canonical, dfs, n_matched_peptides, n_input_peptides)``.

    Workers run with ``verbose=False`` to avoid interleaved progress
    spam in the parent terminal — the orchestrator emits one summary
    line per canonical on completion if it wants progress output.

    On a per-length build failure (proteome not registered, FASTA
    download failure, pyensembl missing GTF), the length is silently
    skipped — same behavior as the sequential code.
    """
    canonical, peptides_by_len, lengths_in_query, release, use_uniprot, flank = args

    matched_peps: set[str] = set()
    dfs: list[pd.DataFrame] = []
    n_input = sum(len(v) for v in peptides_by_len.values())

    for length in lengths_in_query:
        idx = _build_species_index(canonical, release, use_uniprot, False, lengths=(length,))
        if idx is None:
            continue
        length_peptides = peptides_by_len[length]
        flanking = idx.map_peptides(sorted(length_peptides), flank=flank, verbose=False)
        df = _flanking_rows_to_mapping_rows(
            flanking, proteome_label=canonical, proteome_source="species"
        )
        dfs.append(df)
        if len(flanking):
            matched_peps.update(flanking["peptide"].unique())
        # Drop the index before the next length builds so per-worker
        # peak RSS stays bounded by ONE single-length index, not
        # all-lengths-at-once (issue #109's invariant).
        del idx, flanking

    return canonical, dfs, len(matched_peps), n_input


def _build_species_index(
    canonical: str,
    release: int,
    use_uniprot: bool,
    verbose: bool,
    lengths: tuple[int, ...] = (8, 9, 10, 11),
):
    """Build a ProteomeIndex for a species, optionally at specific k-mer lengths.

    The ``lengths`` kwarg enables length-on-demand building so callers
    that only need one length at a time (the mapping pass) can keep
    peak memory bounded by a single length's index (~1 GB for human
    9-mers) rather than all four MHC-I lengths combined (~10 GB).

    Returns None on failure.
    """
    from .downloads import fetch_species_proteome, lookup_proteome
    from .proteome import ProteomeIndex

    entry = lookup_proteome(canonical, use_uniprot=use_uniprot)
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
            )
        except Exception as e:
            if verbose:
                print(f"    [{canonical}] pyensembl failed: {e}")
            return None

    path = fetch_species_proteome(canonical, verbose=verbose, use_uniprot=use_uniprot)
    if path is None or not path.exists():
        return None
    return ProteomeIndex.from_fasta(path, lengths=lengths, verbose=False)
