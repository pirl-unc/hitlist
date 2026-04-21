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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .downloads import data_dir

_MAPPING_COLUMNS = (
    "peptide",
    "protein_id",
    "gene_name",
    "gene_id",
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
    proteome: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the long-form peptide → protein mappings with optional filters.

    Filters are pushed down to pyarrow, so a query like ``gene_name="PRAME"``
    reads only the matching row groups.
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
    df = flanking[
        [
            "peptide",
            "protein_id",
            "gene_name",
            "gene_id",
            "position",
            "n_flank",
            "c_flank",
        ]
    ].copy()
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


def build_peptide_mappings(
    release: int = 112,
    fetch_missing: bool = True,
    use_uniprot: bool = False,
    force: bool = False,
    flank: int = 10,
    verbose: bool = True,
) -> Path:
    """Build ``peptide_mappings.parquet`` from the already-built observations table.

    Reads observations.parquet, collects unique peptides per organism (from
    ``source_organism`` / ``mhc_species`` with ``reference_proteomes``
    overrides), maps each against the appropriate reference proteome, and
    writes all (peptide, protein, position) hits to the sidecar.

    This replaces the old single-best ``_add_flanking`` collapse.
    """
    from .builder import _collect_pmid_extra_proteomes
    from .downloads import fetch_proteome_by_upid, lookup_proteome
    from .observations import is_binding_built, is_built, load_binding, load_observations
    from .proteome import ProteomeIndex

    if not is_built():
        raise FileNotFoundError("Observations table not built.  Run: hitlist data build")

    out = mappings_path()
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

    all_mapping_dfs: list[pd.DataFrame] = []
    per_proteome_stats: list[tuple[str, int, int]] = []

    # ── Length-on-demand: build one k-mer length at a time ─────────────────
    # Peak RSS during the human mapping pass drops from ~10 GB (all 4
    # MHC-I lengths held at once) to ~3 GB (one length held at a time).
    # The 4 lengths are built sequentially and dropped between passes;
    # numpy-packed postings keep each single-length index compact.
    default_lengths = (8, 9, 10, 11)

    for canonical in sorted(species_to_peptides):
        peptides = species_to_peptides[canonical]
        # Bucket this canonical's peptides by length so we can run each
        # length's pass against an index built at that single length only.
        peptides_by_len: dict[int, list[str]] = {}
        for p in peptides:
            peptides_by_len.setdefault(len(p), []).append(p)
        lengths_in_query = tuple(sorted(L for L in peptides_by_len if L in default_lengths))
        if not lengths_in_query:
            # No MHC-I-compatible-length peptides for this canonical; nothing
            # to map (MHC-II peptides at length 12+ are not indexed here —
            # that's pre-existing behavior).
            per_proteome_stats.append((canonical, len(peptides), 0))
            continue

        if verbose:
            print(
                f"\n  [{canonical}] mapping {len(peptides):,} peptides "
                f"across lengths {lengths_in_query} ..."
            )

        matched_peps: set[str] = set()
        for length in lengths_in_query:
            length_peptides = peptides_by_len[length]
            idx = _build_species_index(canonical, release, use_uniprot, verbose, lengths=(length,))
            if idx is None:
                continue
            flanking = idx.map_peptides(sorted(length_peptides), flank=flank, verbose=verbose)
            df = _flanking_rows_to_mapping_rows(
                flanking, proteome_label=canonical, proteome_source="species"
            )
            all_mapping_dfs.append(df)
            if len(flanking):
                matched_peps.update(flanking["peptide"].unique())
            # Explicit del so the single-length index is reclaimed before the
            # next length's build allocates its own. Critical for memory.
            del idx, flanking
        per_proteome_stats.append((canonical, len(peptides), len(matched_peps)))

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
