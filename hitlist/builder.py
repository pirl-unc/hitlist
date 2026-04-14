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

"""Build the unified peptide indexes from IEDB + CEDAR.

Runs the full scanner (with source classification from YAML overrides)
on each registered data source, deduplicates by assay IRI, partitions
rows by assay type, maps peptides to source proteins with flanking
context, and writes TWO parquet indexes to ``~/.hitlist/``:

- ``observations.parquet`` — MS-eluted immunopeptidome rows (plus
  manually-curated supplementary data).
- ``binding.parquet`` — binding-assay rows (peptide microarray,
  refolding, MEDi, and quantitative-tier measurements).

The two indexes are never mixed.  Supplementary data is MS-only and
only contributes to observations.parquet.  Both indexes carry the
same gene/protein annotations from the peptide-mappings sidecar.

Usage::

    from hitlist.builder import build_observations

    path = build_observations()                     # IEDB + CEDAR
    path = build_observations(with_flanking=True)   # + proteome mapping

CLI::

    hitlist data build
    hitlist data build --with-flanking
    hitlist data build --force
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pandas as pd

from .downloads import data_dir


def _source_paths() -> dict[str, Path]:
    """Resolve registered IEDB/CEDAR paths."""
    from .downloads import get_path

    sources: dict[str, Path] = {}
    for name in ("iedb", "cedar"):
        with contextlib.suppress(KeyError, FileNotFoundError):
            p = get_path(name)
            if p.exists():
                sources[name] = p
    return sources


def _source_fingerprints(paths: dict[str, Path]) -> dict:
    """File identity for cache invalidation."""
    fp = {
        name: {"path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        for name, p in paths.items()
    }
    # Include supplementary manifest so adding a new supplement invalidates cache
    from .supplement import load_supplementary_manifest, manifest_path

    mp = manifest_path()
    if mp.exists():
        fp["supplementary_manifest"] = {
            "path": str(mp),
            "size": mp.stat().st_size,
            "mtime": mp.stat().st_mtime,
        }
        # Also fingerprint each referenced CSV so edits invalidate cache
        supp_dir = mp.parent / "supplementary"
        for entry in load_supplementary_manifest():
            csv_path = supp_dir / entry["file"]
            if csv_path.exists():
                key = f"supplementary_csv:{entry['file']}"
                fp[key] = {
                    "path": str(csv_path),
                    "size": csv_path.stat().st_size,
                    "mtime": csv_path.stat().st_mtime,
                }
    return fp


def _observations_path() -> Path:
    return data_dir() / "observations.parquet"


def _binding_path() -> Path:
    return data_dir() / "binding.parquet"


def _meta_path() -> Path:
    return data_dir() / "observations_meta.json"


def _cache_is_valid(paths: dict[str, Path], with_flanking: bool = False) -> bool:
    """Check if the cached indexes are still valid for the requested build.

    Both ``observations.parquet`` and ``binding.parquet`` must be present —
    if either is missing the cache is invalid (binding was added in 1.7.0,
    so older installs rebuild once on upgrade).
    """
    meta = _meta_path()
    if not meta.exists():
        return False
    if not _observations_path().exists():
        return False
    if not _binding_path().exists():
        return False
    stored = json.loads(meta.read_text())
    current = _source_fingerprints(paths)
    return stored.get("sources") == current


def _cache_meta() -> dict:
    """Load the observations_meta.json for user-facing info."""
    meta = _meta_path()
    if meta.exists():
        return json.loads(meta.read_text())
    return {}


def build_observations(
    with_flanking: bool = True,
    proteome_release: int = 112,
    force: bool = False,
    fetch_missing_proteomes: bool = True,
    use_uniprot_search: bool = False,
    build_mappings: bool = True,
) -> Path:
    """Build the unified observations table from IEDB + CEDAR.

    Scans each registered source CSV through the full classification
    pipeline (YAML overrides, tissue categories, mono-allelic detection),
    deduplicates by assay IRI, and writes ``observations.parquet``.

    By default also builds the peptide→protein mappings sidecar
    (``peptide_mappings.parquet``) which preserves multi-mapping so that
    paralog attribution (MAGEA1/A4/A10/A12), repeat regions, and
    cross-proteome hits are not collapsed.

    Parameters
    ----------
    with_flanking
        Deprecated.  Retained for backward compat — the mapping sidecar
        now always includes flanking sequences.  Only disables mapping
        entirely when set to False AND ``build_mappings`` is False.
    proteome_release
        Ensembl release for Ensembl-supported species (default 112).
    force
        Rebuild even if cache is valid.
    fetch_missing_proteomes
        Auto-download reference proteomes for any species present in
        the observations.  Default True.
    build_mappings
        Build ``peptide_mappings.parquet`` sidecar (default True).  Adds
        ~5-10 min to the build on the first run (cached after that).

    Returns
    -------
    Path
        Path to ``observations.parquet`` (the MS index).  The binding
        index is written alongside at ``binding.parquet``.
    """
    paths = _source_paths()
    if not paths:
        raise FileNotFoundError(
            "No IEDB/CEDAR data registered. Use: hitlist data register iedb /path/to/file.csv"
        )

    out_path = _observations_path()
    binding_out = _binding_path()
    if not force and _cache_is_valid(paths, with_flanking=with_flanking):
        meta = _cache_meta()
        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
        print(f"Observations already built ({meta.get('n_rows', '?'):,} rows, {size_mb:.1f} MB)")
        print(f"  Path:          {out_path}")
        print(f"  With flanking: {meta.get('with_flanking', False)}")
        print(f"  Peptides:      {meta.get('n_peptides', '?'):,}")
        print(f"  Alleles:       {meta.get('n_alleles', '?'):,}")
        print(f"  Species:       {meta.get('n_species', '?')}")
        if binding_out.exists():
            b_size = binding_out.stat().st_size / 1e6
            print(
                f"  Binding index: {meta.get('n_binding_rows', '?'):,} rows, "
                f"{b_size:.1f} MB → {binding_out}"
            )
        print("\nUse --force to rebuild.")
        return out_path

    from .scanner import scan

    ms_dfs: list[pd.DataFrame] = []
    binding_dfs: list[pd.DataFrame] = []
    ms_seen_iris: set[str] = set()
    binding_seen_iris: set[str] = set()

    for name in ("iedb", "cedar"):
        if name not in paths:
            continue
        print(f"Scanning {name} ({paths[name].name})...")
        df = scan(
            peptides=None,
            iedb_path=paths[name] if name == "iedb" else None,
            cedar_path=paths[name] if name == "cedar" else None,
            human_only=False,
            classify_source=True,
        )
        df["source"] = name

        # Partition into MS vs binding — the two indexes are written
        # separately so downstream consumers cannot accidentally mix
        # immunopeptidome elution with affinity/microarray measurements.
        if "is_binding_assay" in df.columns:
            ms_df = df[~df["is_binding_assay"]].copy()
            bd_df = df[df["is_binding_assay"]].copy()
        else:
            ms_df = df
            bd_df = df.iloc[0:0].copy()

        # Deduplicate across sources by assay IRI (per index).
        if ms_seen_iris:
            before = len(ms_df)
            ms_df = ms_df[~ms_df["reference_iri"].isin(ms_seen_iris)]
            dupes = before - len(ms_df)
            if dupes:
                print(f"  Deduplicated {dupes:,} MS rows (shared IRIs with prior source)")
        if binding_seen_iris:
            before = len(bd_df)
            bd_df = bd_df[~bd_df["reference_iri"].isin(binding_seen_iris)]
            dupes = before - len(bd_df)
            if dupes:
                print(f"  Deduplicated {dupes:,} binding rows (shared IRIs with prior source)")

        ms_seen_iris.update(ms_df["reference_iri"].values)
        binding_seen_iris.update(bd_df["reference_iri"].values)
        ms_dfs.append(ms_df)
        binding_dfs.append(bd_df)
        print(f"  {len(ms_df):,} MS rows + {len(bd_df):,} binding rows from {name}")

    if not ms_dfs and not binding_dfs:
        raise RuntimeError("No data scanned.")

    obs = pd.concat(ms_dfs, ignore_index=True) if ms_dfs else pd.DataFrame()
    binding = pd.concat(binding_dfs, ignore_index=True) if binding_dfs else pd.DataFrame()

    # --- Supplementary data (MS only — manually curated from papers) ---
    from .supplement import scan_supplementary

    supp = scan_supplementary(classify_source=True)
    if not supp.empty:
        supp["source"] = "supplement"
        # Deduplicate: IEDB/CEDAR rows win over supplementary
        existing_keys = set(zip(obs["peptide"], obs["mhc_restriction"], obs["pmid"]))
        before = len(supp)
        supp = supp[
            ~pd.Series(
                [
                    (p, a, m) in existing_keys
                    for p, a, m in zip(supp["peptide"], supp["mhc_restriction"], supp["pmid"])
                ],
                index=supp.index,
            )
        ]
        dupes = before - len(supp)
        if dupes:
            print(f"  Deduplicated {dupes:,} supplementary rows (already in IEDB/CEDAR)")
        obs = pd.concat([obs, supp], ignore_index=True)
        print(f"  {len(supp):,} rows from supplementary data (MS)")

    print(f"\nMS observations: {len(obs):,} rows")
    if len(obs):
        print(f"  Unique peptides: {obs['peptide'].nunique():,}")
        print(f"  Unique alleles:  {obs['mhc_restriction'].nunique():,}")
        print(f"  Species:         {obs['mhc_species'].nunique()}")
    print(f"Binding rows:    {len(binding):,}")
    if len(binding):
        print(f"  Unique peptides: {binding['peptide'].nunique():,}")
        print(f"  Unique alleles:  {binding['mhc_restriction'].nunique():,}")

    # Fix mixed types for parquet compatibility
    for frame in (obs, binding):
        if "pmid" in frame.columns:
            frame["pmid"] = pd.to_numeric(frame["pmid"], errors="coerce").astype("Int64")

    # Write both parquets — the mappings build reads from them to cover
    # the union of peptides.
    obs.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    binding.to_parquet(binding_out, index=False)
    print(f"Wrote {binding_out} ({binding_out.stat().st_size / 1e6:.1f} MB)")

    # Build the long-form peptide → protein mappings sidecar, then
    # annotate BOTH parquets with semicolon-joined gene/protein columns.
    # The sidecar covers the union of peptides from observations + binding.
    if build_mappings:
        from .mappings import (
            annotate_observations_with_genes,
            build_peptide_mappings,
            load_peptide_mappings,
        )

        build_peptide_mappings(
            release=proteome_release,
            fetch_missing=fetch_missing_proteomes,
            use_uniprot=use_uniprot_search,
            force=force,
        )

        print("\nAnnotating indexes with gene/protein columns ...")
        mappings_df = load_peptide_mappings(
            columns=["peptide", "gene_name", "gene_id", "protein_id"]
        )
        obs = annotate_observations_with_genes(obs, mappings_df)
        obs.to_parquet(out_path, index=False)
        if len(obs):
            n_with_gene = (obs["gene_names"] != "").sum() if "gene_names" in obs.columns else 0
            print(
                f"  MS:      {n_with_gene:,} / {len(obs):,} rows annotated "
                f"({100 * n_with_gene / len(obs):.1f}%)"
            )
        binding = annotate_observations_with_genes(binding, mappings_df)
        binding.to_parquet(binding_out, index=False)
        if len(binding):
            n_with_gene_b = (
                (binding["gene_names"] != "").sum() if "gene_names" in binding.columns else 0
            )
            print(
                f"  Binding: {n_with_gene_b:,} / {len(binding):,} rows annotated "
                f"({100 * n_with_gene_b / len(binding):.1f}%)"
            )

    # Save metadata
    meta = {
        "sources": _source_fingerprints(paths),
        "n_rows": len(obs),
        "n_peptides": int(obs["peptide"].nunique()) if len(obs) else 0,
        "n_alleles": int(obs["mhc_restriction"].nunique()) if len(obs) else 0,
        "n_species": int(obs["mhc_species"].nunique()) if len(obs) else 0,
        "n_binding_rows": len(binding),
        "n_binding_peptides": int(binding["peptide"].nunique()) if len(binding) else 0,
        "with_flanking": with_flanking,
        "with_mappings": build_mappings,
    }
    _meta_path().write_text(json.dumps(meta, indent=2, default=str) + "\n")

    return out_path


_FLANKING_COLS = (
    "gene_name",
    "gene_id",
    "protein_id",
    "position",
    "n_flank",
    "c_flank",
    "n_source_proteins",
    "flanking_species",
)


def _load_species_index(
    organism: str,
    release: int,
    verbose: bool,
    use_uniprot: bool = False,
):
    """Build a ProteomeIndex for a given species/organism, or None if unsupported."""
    from .downloads import lookup_proteome
    from .proteome import ProteomeIndex

    entry = lookup_proteome(organism, use_uniprot=use_uniprot)
    if entry is None:
        return None, None

    canonical = entry.get("canonical_species", organism)
    kind = entry["kind"]

    if kind == "ensembl":
        species = entry.get("species", "human")
        try:
            if verbose:
                print(f"  [{canonical}] loading pyensembl (species={species}, release={release})")
            idx = ProteomeIndex.from_ensembl(
                release=release,
                species=species,
                verbose=verbose,
            )
            return idx, canonical
        except TypeError:
            # Older signature without species kwarg — only human works then
            if species == "human":
                idx = ProteomeIndex.from_ensembl(release=release, verbose=verbose)
                return idx, canonical
            if verbose:
                print(
                    f"  [{canonical}] pyensembl species={species} not supported "
                    "by installed ProteomeIndex; skipping"
                )
            return None, None
        except Exception as e:
            if verbose:
                print(f"  [{canonical}] pyensembl failed: {e}")
            return None, None

    # UniProt FASTA
    from .downloads import fetch_species_proteome

    path = fetch_species_proteome(organism, verbose=verbose, use_uniprot=use_uniprot)
    if path is None or not path.exists():
        return None, None
    if verbose:
        print(f"  [{canonical}] indexing FASTA {path.name}")
    idx = ProteomeIndex.from_fasta(path, verbose=verbose)
    return idx, canonical


def _collect_pmid_extra_proteomes() -> dict[int, list[dict]]:
    """Load per-PMID reference_proteomes overrides from pmid_overrides.yaml.

    Returns a map ``{pmid_int: [{"upid": "UP...", "label": "..."}, ...]}``
    preserving the per-sample ordering (host first, viruses after).  Entries
    may be either dicts with a ``uniprot`` key or bare UPID strings.
    """
    from .curation import load_pmid_overrides

    overrides = load_pmid_overrides()
    result: dict[int, list[dict]] = {}
    for pmid, entry in overrides.items():
        seen: set[str] = set()
        upids: list[dict] = []
        for sample in entry.get("ms_samples", []):
            for proteome in sample.get("reference_proteomes", []):
                if isinstance(proteome, dict):
                    upid = proteome.get("uniprot")
                    label = proteome.get("label")
                else:
                    upid = str(proteome).strip()
                    label = None
                if not upid or upid in seen:
                    continue
                seen.add(upid)
                upids.append({"upid": upid, "label": label or upid})
        if upids:
            try:
                result[int(pmid)] = upids
            except (ValueError, TypeError):
                continue
    return result


def _map_extra_proteomes(obs: pd.DataFrame, release: int, use_uniprot: bool) -> pd.DataFrame:
    """Fill in flanking for peptides unmapped by the primary species pass
    using per-PMID ``reference_proteomes`` overrides."""
    pmid_extras = _collect_pmid_extra_proteomes()
    if not pmid_extras:
        return obs

    from .downloads import fetch_proteome_by_upid
    from .proteome import ProteomeIndex

    # Collect per-UPID peptides (for any PMID that references it, take only
    # observations that didn't match in the primary pass)
    pmid_col = obs["pmid"] if "pmid" in obs.columns else None
    if pmid_col is None:
        return obs
    unmatched_mask = obs["gene_name"].isna()

    upid_to_peptides: dict[str, tuple[str, set[str]]] = {}
    pmid_priority: dict[int, list[str]] = {}
    for pmid_int, upid_entries in pmid_extras.items():
        sel = unmatched_mask & (pmid_col == pmid_int)
        if not sel.any():
            continue
        peptides = set(obs.loc[sel, "peptide"].dropna())
        pmid_priority[pmid_int] = [e["upid"] for e in upid_entries]
        for e in upid_entries:
            upid = e["upid"]
            label = e["label"]
            if upid not in upid_to_peptides:
                upid_to_peptides[upid] = (label, set())
            upid_to_peptides[upid][1].update(peptides)

    if not upid_to_peptides:
        return obs

    # Map peptides against each extra proteome; store per-UPID best hits
    upid_to_hits: dict[str, pd.DataFrame] = {}
    print(
        f"\n  [extras] mapping {len(upid_to_peptides)} per-PMID override proteome(s) "
        f"for {sum(len(p) for _, p in upid_to_peptides.values()):,} unmatched peptides"
    )
    for upid, (label, peptides) in upid_to_peptides.items():
        path = fetch_proteome_by_upid(upid, label=label, verbose=True)
        if path is None or not path.exists():
            continue
        idx = ProteomeIndex.from_fasta(path, verbose=False)
        flanking = idx.map_peptides(sorted(peptides), flank=10, verbose=False)
        if flanking.empty:
            continue
        best = flanking.sort_values("n_sources").drop_duplicates("peptide", keep="first")
        best = best[
            [
                "peptide",
                "gene_name",
                "gene_id",
                "protein_id",
                "position",
                "n_flank",
                "c_flank",
                "n_sources",
            ]
        ].copy()
        best = best.rename(columns={"n_sources": "n_source_proteins"})
        best["flanking_species"] = label
        upid_to_hits[upid] = best.set_index("peptide")
        print(f"    [{label}] matched {len(best):,} / {len(peptides):,} peptides")

    if not upid_to_hits:
        return obs

    # For each unmatched observation, try its PMID's priority list and take
    # the first hit.  Update in place.
    flank_cols = (
        "gene_name",
        "gene_id",
        "protein_id",
        "position",
        "n_flank",
        "c_flank",
        "n_source_proteins",
        "flanking_species",
    )
    for pmid_int, priority in pmid_priority.items():
        sel = unmatched_mask & (pmid_col == pmid_int)
        if not sel.any():
            continue
        for upid in priority:
            hits = upid_to_hits.get(upid)
            if hits is None:
                continue
            # Find which unmatched rows of this PMID have a hit in this UPID
            still_unmatched = unmatched_mask & (pmid_col == pmid_int) & obs["gene_name"].isna()
            if not still_unmatched.any():
                break
            peps = obs.loc[still_unmatched, "peptide"]
            matched_peps = peps[peps.isin(hits.index)]
            if matched_peps.empty:
                continue
            for col in flank_cols:
                if col in hits.columns:
                    obs.loc[matched_peps.index, col] = matched_peps.map(hits[col])
            # Recompute unmatched_mask locally for subsequent priority items
            # within this PMID
            unmatched_mask = obs["gene_name"].isna()
    return obs


def _add_flanking(
    obs: pd.DataFrame,
    release: int = 112,
    fetch_missing: bool = True,
    use_uniprot: bool = False,
) -> pd.DataFrame:
    """Map peptides to source proteins with flanking context.

    Routes each observation to its species-specific reference proteome
    based on ``source_organism`` (with ``mhc_species`` as a fallback).
    Reports per-species and overall progress.

    Parameters
    ----------
    obs
        Observations DataFrame with ``peptide`` + ``source_organism`` columns.
    release
        Ensembl release for Ensembl-supported species.
    fetch_missing
        If True, auto-download missing UniProt proteomes.
    use_uniprot
        If True, fall back to UniProt REST search for organisms that
        aren't in the curated registry.  Resolved mappings are cached
        in the manifest.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    from .downloads import fetch_species_proteome, lookup_proteome

    # Decide the organism for each observation: source_organism > mhc_species
    organism = obs["source_organism"].astype(str).str.strip()
    organism = organism.where(organism != "", obs["mhc_species"].astype(str).str.strip())
    obs = obs.assign(_flanking_organism=organism)

    # Group peptides by (registered) canonical proteome.  We cache lookups
    # by raw organism string to avoid redundant UniProt queries.
    lookup_cache: dict[str, dict | None] = {}

    def _lookup(org: str) -> dict | None:
        if org in lookup_cache:
            return lookup_cache[org]
        entry = lookup_proteome(org, use_uniprot=use_uniprot)
        lookup_cache[org] = entry
        return entry

    species_to_peptides: dict[str, set[str]] = {}
    unmapped_organisms: dict[str, int] = {}
    for org, pep in zip(obs["_flanking_organism"], obs["peptide"]):
        if not org:
            continue
        entry = _lookup(org)
        if entry is None:
            unmapped_organisms[org] = unmapped_organisms.get(org, 0) + 1
            continue
        canonical = entry.get("canonical_species", org)
        species_to_peptides.setdefault(canonical, set()).add(pep)

    total_peptides = sum(len(v) for v in species_to_peptides.values())
    n_species = len(species_to_peptides)
    print(
        f"\nMapping peptides to source proteins:"
        f"  {total_peptides:,} unique peptides across {n_species} species"
    )

    if fetch_missing and n_species:
        print("\nEnsuring reference proteomes are available ...")
        for canonical in sorted(species_to_peptides):
            fetch_species_proteome(canonical, verbose=True, use_uniprot=use_uniprot)

    # Map per species; collect representative per peptide (global)
    best_rows: list[dict] = []
    per_species_stats: list[tuple[str, int, int]] = []

    species_iter = sorted(species_to_peptides)
    if tqdm is not None and n_species > 1:
        species_iter = tqdm(species_iter, desc="Flanking annotation", unit="species")

    for canonical in species_iter:
        peptides = species_to_peptides[canonical]
        if tqdm is not None and n_species > 1:
            species_iter.set_postfix_str(f"{canonical} ({len(peptides):,} peps)")
        else:
            print(f"\n  [{canonical}] mapping {len(peptides):,} peptides ...")

        idx, _canonical_out = _load_species_index(
            canonical, release=release, verbose=True, use_uniprot=use_uniprot
        )
        if idx is None:
            per_species_stats.append((canonical, len(peptides), 0))
            continue

        flanking = idx.map_peptides(sorted(peptides), flank=10, verbose=True)
        if flanking.empty:
            per_species_stats.append((canonical, len(peptides), 0))
            continue

        best = flanking.sort_values("n_sources").drop_duplicates("peptide", keep="first")
        best = best[
            [
                "peptide",
                "gene_name",
                "gene_id",
                "protein_id",
                "position",
                "n_flank",
                "c_flank",
                "n_sources",
            ]
        ].copy()
        best = best.rename(columns={"n_sources": "n_source_proteins"})
        best["flanking_species"] = canonical

        best_rows.append(best)
        per_species_stats.append((canonical, len(peptides), len(best)))

    # Merge per-species results into single (peptide, species) → flanking map.
    # We key on (peptide, _flanking_organism_canonical) so a given peptide
    # routed to different species via different observations can get different
    # source proteins.
    if best_rows:
        best_all = pd.concat(best_rows, ignore_index=True)
    else:
        best_all = pd.DataFrame(columns=["peptide", "flanking_species", *_FLANKING_COLS])

    # Attach canonical species to each obs row for the join key
    canonical_lookup: dict[str, str] = {}
    for org in obs["_flanking_organism"].unique():
        if not org:
            continue
        entry = _lookup(org)
        if entry is None:
            continue
        canonical_lookup[org] = entry.get("canonical_species", org)
    obs["_canonical_species"] = obs["_flanking_organism"].map(lambda o: canonical_lookup.get(o, ""))

    obs = obs.merge(
        best_all.rename(columns={"flanking_species": "_canonical_species"}),
        on=["peptide", "_canonical_species"],
        how="left",
    )
    obs["flanking_species"] = obs["_canonical_species"].replace("", pd.NA)
    obs.drop(columns=["_canonical_species", "_flanking_organism"], inplace=True)

    # --- Per-sample reference_proteomes override ---
    # For PMIDs whose ms_samples list additional reference_proteomes (e.g.
    # EBV for B-LCLs, Influenza A for infected lung), try each extra
    # proteome for the peptides of that PMID that the primary mapping
    # didn't match.  First hit wins in user-specified order.
    obs = _map_extra_proteomes(obs, release=release, use_uniprot=use_uniprot)

    # --- Coverage report ---
    print("\n  Flanking coverage by species:")
    mapped_total = 0
    for canonical, n_peps, n_mapped in sorted(per_species_stats):
        n_obs = int((obs["flanking_species"] == canonical).sum())
        mapped_total += n_obs
        print(f"    {canonical:40s}  peptides: {n_mapped:>8,} / {n_peps:<8,}  rows: {n_obs:>10,}")

    if unmapped_organisms:
        print(f"\n  Organisms with no registered proteome ({len(unmapped_organisms)}):")
        for org, n in sorted(unmapped_organisms.items(), key=lambda x: -x[1])[:10]:
            print(f"    {org!r:60s}  {n:>10,} rows")
        if len(unmapped_organisms) > 10:
            print(f"    ... and {len(unmapped_organisms) - 10} more")

    n_mapped = int(obs["gene_name"].notna().sum())
    print(
        f"\n  Mapped: {n_mapped:,} / {len(obs):,} observations ({100 * n_mapped / len(obs):.1f}%)"
    )

    return obs
