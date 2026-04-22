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


def _bulk_proteomics_path() -> Path:
    return data_dir() / "bulk_proteomics.parquet"


def _meta_path() -> Path:
    return data_dir() / "observations_meta.json"


def _parquet_fingerprints() -> dict:
    """Size + mtime of the three index parquets, for cache validation.

    Pairs with ``_source_fingerprints`` to detect either (a) source CSV
    changes or (b) a parquet file being manually edited / replaced
    between builds.
    """
    fp: dict = {}
    for label, p in (
        ("observations", _observations_path()),
        ("binding", _binding_path()),
        ("bulk_proteomics", _bulk_proteomics_path()),
    ):
        if p.exists():
            stat = p.stat()
            fp[label] = {"size": stat.st_size, "mtime": stat.st_mtime}
    return fp


def _cache_is_valid(paths: dict[str, Path], with_flanking: bool = False) -> bool:
    """Check if the cached indexes are still valid for the requested build.

    All three parquets (``observations``, ``binding``, ``bulk_proteomics``)
    must be present AND their fingerprints must match the stored metadata.
    Binding was added in 1.7.0 and bulk_proteomics in 1.11.2, so older
    installs rebuild once on upgrade.
    """
    meta = _meta_path()
    if not meta.exists():
        return False
    if not _observations_path().exists():
        return False
    if not _binding_path().exists():
        return False
    if not _bulk_proteomics_path().exists():
        return False
    stored = json.loads(meta.read_text())
    if stored.get("sources") != _source_fingerprints(paths):
        return False
    stored_parquets = stored.get("parquets")
    return not (stored_parquets is not None and stored_parquets != _parquet_fingerprints())


def _cache_meta() -> dict:
    """Load the observations_meta.json for user-facing info."""
    meta = _meta_path()
    if meta.exists():
        return json.loads(meta.read_text())
    return {}


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to a sibling ``.partial`` file, then rename over ``path``.

    Readers calling :func:`load_observations` / :func:`load_binding`
    during a rebuild keep seeing whatever was on ``path`` before this
    call until the rename atomically swaps in the new file.  This closes
    the mid-rebuild window (#105) where the canonical parquet briefly
    lacked its ``gene_names`` column between the initial write and the
    re-annotate step.
    """
    partial = path.with_suffix(path.suffix + ".partial")
    df.to_parquet(partial, index=False)
    partial.replace(path)


def _drop_short_mhc2_rows(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop ``mhc_class == "II"`` rows with peptides shorter than 12 aa.

    MHC-II binding requires ~12-25 aa peptides; 8-11 aa rows annotated
    as class II are overwhelmingly IEDB import errors (e.g. HLA Ligand
    Atlas PMID 33858848 attributing class-I elutions to the class-II
    bucket).  Prints the drop count and top-5 PMIDs affected so the
    curator can audit.  See pirl-unc/hitlist#122.
    """
    if df.empty or "mhc_class" not in df.columns or "peptide" not in df.columns:
        return df
    mask = (df["mhc_class"] == "II") & (df["peptide"].str.len() < 12)
    n_drop = int(mask.sum())
    if n_drop == 0:
        return df
    print(f"  Dropped {n_drop:,} short MHC-II rows from {label} (peptide < 12 aa, #122)")
    if "pmid" in df.columns:
        by_pmid = df.loc[mask, "pmid"].value_counts().head(5)
        for pmid, n in by_pmid.items():
            print(f"    PMID {pmid}: {n:,} rows")
    return df[~mask].reset_index(drop=True)


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
            mhc_species=None,  # builder indexes all species; downstream filters per-call
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

    # Drop short MHC-II rows (#122) — biologically implausible, and
    # overwhelmingly IEDB import errors (e.g. HLA Ligand Atlas 9-mers).
    obs = _drop_short_mhc2_rows(obs, "MS observations")
    binding = _drop_short_mhc2_rows(binding, "binding")

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

    # Build the long-form peptide → protein mappings sidecar, then
    # annotate BOTH parquets with semicolon-joined gene/protein columns.
    # Hold obs/binding in memory so the canonical parquets aren't briefly
    # missing their ``gene_names`` column mid-rebuild (#105).  The
    # mappings build consumes the frames directly via ``obs_override`` /
    # ``binding_override`` instead of re-reading the canonical files.
    if build_mappings:
        from .mappings import (
            _obs_fingerprint,
            annotate_observations_with_genes,
            build_peptide_mappings,
            load_peptide_mappings,
            mappings_meta_path,
        )

        build_peptide_mappings(
            release=proteome_release,
            fetch_missing=fetch_missing_proteomes,
            use_uniprot=use_uniprot_search,
            force=force,
            obs_override=obs,
            binding_override=binding,
        )

        print("\nAnnotating indexes with gene/protein columns ...")
        mappings_df = load_peptide_mappings(
            columns=["peptide", "gene_name", "gene_id", "protein_id"]
        )
        obs = annotate_observations_with_genes(obs, mappings_df)
        if len(obs):
            n_with_gene = (obs["gene_names"] != "").sum() if "gene_names" in obs.columns else 0
            print(
                f"  MS:      {n_with_gene:,} / {len(obs):,} rows annotated "
                f"({100 * n_with_gene / len(obs):.1f}%)"
            )
        binding = annotate_observations_with_genes(binding, mappings_df)
        if len(binding):
            n_with_gene_b = (
                (binding["gene_names"] != "").sum() if "gene_names" in binding.columns else 0
            )
            print(
                f"  Binding: {n_with_gene_b:,} / {len(binding):,} rows annotated "
                f"({100 * n_with_gene_b / len(binding):.1f}%)"
            )

    # Atomic rename write (#105): write to a sibling .partial, then rename
    # over the canonical path.  This keeps any prior index in place — and
    # queryable — throughout the rebuild; readers never see a half-written
    # or not-yet-annotated parquet.
    _atomic_write_parquet(obs, out_path)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    _atomic_write_parquet(binding, binding_out)
    print(f"Wrote {binding_out} ({binding_out.stat().st_size / 1e6:.1f} MB)")

    if build_mappings:
        # Rewrite mappings cache fingerprint to reflect the just-renamed
        # canonical parquets — it was stamped from whatever was on disk
        # before the atomic rename (empty on a fresh build, stale otherwise).
        meta_p = mappings_meta_path()
        if meta_p.exists():
            stored = json.loads(meta_p.read_text())
            stored["observations"] = _obs_fingerprint()
            meta_p.write_text(json.dumps(stored, indent=2, default=str) + "\n")

    # --- Bulk proteomics index (non-MHC shotgun MS) ---
    print("\nBuilding bulk_proteomics.parquet ...")
    bulk_df = build_bulk_proteomics(verbose=True)

    # Save metadata
    meta = {
        "sources": _source_fingerprints(paths),
        "parquets": _parquet_fingerprints(),
        "n_rows": len(obs),
        "n_peptides": int(obs["peptide"].nunique()) if len(obs) else 0,
        "n_alleles": int(obs["mhc_restriction"].nunique()) if len(obs) else 0,
        "n_species": int(obs["mhc_species"].nunique()) if len(obs) else 0,
        "n_binding_rows": len(binding),
        "n_binding_peptides": int(binding["peptide"].nunique()) if len(binding) else 0,
        "n_bulk_rows": len(bulk_df),
        "n_bulk_protein_rows": int((bulk_df["granularity"] == "protein").sum())
        if len(bulk_df)
        else 0,
        "n_bulk_peptide_rows": int((bulk_df["granularity"] == "peptide").sum())
        if len(bulk_df)
        else 0,
        "with_flanking": with_flanking,
        "with_mappings": build_mappings,
    }
    _meta_path().write_text(json.dumps(meta, indent=2, default=str) + "\n")

    return out_path


def build_bulk_proteomics(verbose: bool = False) -> pd.DataFrame:
    """Build ``bulk_proteomics.parquet`` — long-form bulk MS index.

    Emits a single parquet in ``data_dir()`` with protein- and peptide-level
    rows (distinguished by ``granularity``), denormalizing per-source
    acquisition metadata (instrument, digest, fragmentation, labeling,
    fractionation, search engine, FDR) onto every row so the file is
    self-contained for MS-bias modeling.

    Acquisition column names (``instrument``, ``instrument_type``,
    ``fragmentation``, ``acquisition_mode``, ``labeling``,
    ``search_engine``, ``fdr``) are harmonized with the per-sample schema
    used by :mod:`hitlist.export` so the same columns can be extracted
    from ``observations.parquet`` (via ``generate_observations_table``)
    and ``bulk_proteomics.parquet`` for joint MS-bias analysis.

    Source data ships inside the package under
    ``hitlist/data/bulk_proteomics/`` (CSV.gz + ``sources.yaml``). This
    function just reshapes it into the long parquet written to
    ``~/.hitlist/``.

    Returns
    -------
    pd.DataFrame
        The concatenated long-form table that was written. Empty frame
        if the packaged CSVs are missing.
    """
    from .bulk_proteomics import (
        _load_bj,
        _load_bj_protein,
        _load_ccle,
        _load_sources_yaml,
    )
    from .export import _classify_instrument

    sources_yaml = {s["source_id"]: s for s in _load_sources_yaml()}

    def _study_meta(source_id: str) -> dict:
        """Extract the harmonized acquisition/study metadata for a source."""
        s = sources_yaml.get(source_id, {})
        pmid = s.get("pmid")
        instrument = s.get("instrument", "") or ""
        ph = s.get("fractionation_ph")
        return {
            "pmid": int(pmid) if pmid else pd.NA,
            "reference": s.get("reference", "") or "",
            "study_label": s.get("study_label", "") or "",
            "species": s.get("species", "") or "",
            "instrument": instrument,
            "instrument_type": _classify_instrument(instrument),
            "fragmentation": s.get("fragmentation", "") or "",
            "acquisition_mode": s.get("acquisition_mode", "") or "",
            "labeling": s.get("labeling", "") or "",
            "search_engine": s.get("search_engine", "") or "",
            "fdr": s.get("fdr", "") or "",
            "digestion": s.get("digestion", "") or "",
            "digestion_enzyme": s.get("digestion_enzyme", "") or "",
            "fractionation": s.get("fractionation", "") or "",
            "n_fractions": int(s["n_fractions"]) if s.get("n_fractions") else pd.NA,
            # Default high-pH RPLC buffer pH; source-level default is used
            # for rows that don't carry a per-row value (CCLE; BJ rows at
            # the implicit pH 10). Explicit per-row values (e.g. BJ
            # Tryp-Phos-pH8 = 8.0) override via ``_stamp_bj_per_row``.
            "fractionation_ph": float(ph) if ph is not None else pd.NA,
            "quantification": s.get("quantification", "") or "",
        }

    import numpy as np

    frames: list[pd.DataFrame] = []

    def _stamp_bj_per_row(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
        """Merge source-level meta into a Bekker-Jensen frame.

        Five columns are authoritative at the ROW level (set by the
        ingest script from the MaxQuant experiment name):
        ``digestion_enzyme``, ``n_fractions_in_run``, ``enrichment``,
        ``fractionation_ph``, ``modifications``. When any of them is
        present on the source CSV we:
          - skip overwriting it with the source-level default
          - for ``digestion``: derive a coarse 'tryptic' / 'non-tryptic'
            label from the per-row enzyme so ``df.query("digestion == 'tryptic'")``
            still works after the ancillary arms were added
          - for ``n_fractions``: derive from ``n_fractions_in_run`` so
            the coarse source-level ``n_fractions`` matches the arm
            actually indexed in each row (14/39/46/50/70/12 per the
            Fig 1b design matrix).
        """
        has_per_row_enzyme = "digestion_enzyme" in df.columns
        has_per_row_fracs = "n_fractions_in_run" in df.columns
        has_per_row_ph = "fractionation_ph" in df.columns
        for k, v in meta.items():
            if k == "digestion_enzyme" and has_per_row_enzyme:
                continue
            if k == "fractionation_ph" and has_per_row_ph:
                # Per-row pH is authoritative (Tryp-Phos-pH8 = 8.0,
                # everything else 10.0). Don't overwrite.
                continue
            if k == "digestion" and has_per_row_enzyme:
                df["digestion"] = df["digestion_enzyme"].apply(
                    lambda e: "tryptic" if str(e).startswith("Trypsin") else "non-tryptic"
                )
                continue
            if k == "n_fractions" and has_per_row_fracs:
                # Keep the coarse source-level ``n_fractions`` in sync
                # with the authoritative per-row value. Callers filtering
                # ``df["n_fractions"] == 46`` then get the 46-frac arm
                # rows; callers filtering on ``n_fractions_in_run`` get
                # the same thing explicitly.
                df["n_fractions"] = df["n_fractions_in_run"]
                continue
            df[k] = v
        return df

    # --- CCLE protein-level ---
    ccle = _load_ccle().copy()
    if len(ccle):
        ccle["abundance_percentile"] = ccle.groupby("cell_line")["abundance_log2_normalized"].rank(
            pct=True
        )
        ccle["granularity"] = "protein"
        ccle["peptide"] = ""
        ccle["length"] = np.nan
        ccle["start_position"] = np.nan
        ccle["end_position"] = np.nan
        ccle["log2_intensity"] = np.nan
        ccle["n_peptides"] = np.nan
        meta = _study_meta("CCLE_Nusinow_2020")
        for k, v in meta.items():
            ccle[k] = v
        frames.append(ccle)

    # --- Bekker-Jensen protein-level ---
    bj_protein = _load_bj_protein().copy()
    if len(bj_protein):
        bj_protein["granularity"] = "protein"
        bj_protein["peptide"] = ""
        bj_protein["length"] = np.nan
        bj_protein["start_position"] = np.nan
        bj_protein["end_position"] = np.nan
        bj_protein["abundance_log2_normalized"] = np.nan
        meta = _study_meta("Bekker-Jensen_2017")
        bj_protein = _stamp_bj_per_row(bj_protein, meta)
        frames.append(bj_protein)

    # --- Bekker-Jensen peptide-level ---
    bj_peptide = _load_bj().copy()
    if len(bj_peptide):
        bj_peptide["granularity"] = "peptide"
        bj_peptide["log2_intensity"] = np.nan
        bj_peptide["abundance_log2_normalized"] = np.nan
        bj_peptide["abundance_percentile"] = np.nan
        bj_peptide["n_peptides"] = np.nan
        meta = _study_meta("Bekker-Jensen_2017")
        bj_peptide = _stamp_bj_per_row(bj_peptide, meta)
        frames.append(bj_peptide)

    if not frames:
        out = _bulk_proteomics_path()
        empty = pd.DataFrame()
        empty.to_parquet(out, index=False)
        return empty

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Harmonize column names: cell_line -> cell_line_name to match
    # observations.parquet; keep sample_label alias for ms_samples parity.
    df = df.rename(columns={"cell_line": "cell_line_name"})
    df["sample_label"] = df["cell_line_name"]
    df["evidence_kind"] = "bulk_proteomics"
    # perturbation is empty for all currently-indexed sources (untreated
    # baseline cell-line proteomes); column exists so joins with
    # observations/export stay schema-stable.
    df["perturbation"] = ""

    # Integer columns with nullable dtype so parquet round-trips cleanly.
    int_cols = (
        "length",
        "start_position",
        "end_position",
        "n_peptides",
        "n_fractions",
        "n_fractions_in_run",
        "n_replicates_detected",
        "pmid",
    )
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # fractionation_ph is continuous (float) so it gets its own cast.
    if "fractionation_ph" in df.columns:
        df["fractionation_ph"] = pd.to_numeric(df["fractionation_ph"], errors="coerce")

    # For rows that don't carry per-row Fig 1b axes (CCLE protein rows),
    # fill the columns with sensible defaults so parquet has a stable
    # schema rather than mixed-NA dtypes. CCLE is non-enriched baseline
    # shotgun with no modification enrichment, so ``enrichment="none"``
    # and ``modifications=""`` are correct.
    if "enrichment" in df.columns:
        df["enrichment"] = df["enrichment"].fillna("none").replace("", "none")
    else:
        df["enrichment"] = "none"
    if "modifications" not in df.columns:
        df["modifications"] = ""
    else:
        df["modifications"] = df["modifications"].fillna("")

    ordered_cols = [
        # evidence + granularity
        "evidence_kind",
        "granularity",
        # study identity (harmonized with observations.parquet / ms_samples)
        "source",
        "reference",
        "pmid",
        "study_label",
        "species",
        # sample identity (harmonized with observations.parquet)
        "cell_line_name",
        "sample_label",
        "perturbation",
        # biological target
        "gene_symbol",
        "uniprot_acc",
        "peptide",
        "length",
        "start_position",
        "end_position",
        # quant
        "log2_intensity",
        "abundance_log2_normalized",
        "abundance_percentile",
        "n_peptides",
        "n_replicates_detected",
        # acquisition metadata (harmonized with ms_samples schema)
        "instrument",
        "instrument_type",
        "fragmentation",
        "acquisition_mode",
        "labeling",
        "search_engine",
        "fdr",
        # bulk-specific prep/digest info — per-row Fig 1b axes for
        # Bekker-Jensen (authoritative) and source-level defaults for
        # CCLE (all tryptic, 12-frac, no enrichment).
        "digestion",
        "digestion_enzyme",
        "fractionation",
        "n_fractions",
        "n_fractions_in_run",
        "fractionation_ph",
        "enrichment",
        "modifications",
        "quantification",
    ]
    # Drop any stray input columns (e.g. CCLE's FASTA-header protein_id
    # which duplicates uniprot_acc) so the parquet schema is stable.
    df = df[[c for c in ordered_cols if c in df.columns]]

    out = _bulk_proteomics_path()
    df.to_parquet(out, index=False)
    if verbose:
        n_prot = int((df["granularity"] == "protein").sum())
        n_pep = int((df["granularity"] == "peptide").sum())
        size_mb = out.stat().st_size / 1e6
        print(f"  {len(df):,} rows ({n_prot:,} protein + {n_pep:,} peptide) → {size_mb:.1f} MB")
        print(f"  Wrote {out}")
    return df


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
                    label = proteome.get("proteome_label") or proteome.get("label")
                    if proteome.get("label") and not proteome.get("proteome_label"):
                        import warnings

                        warnings.warn(
                            f"PMID {pmid}: reference_proteomes uses deprecated "
                            f"'label:' key, use 'proteome_label:' (v1.7.5).",
                            DeprecationWarning,
                            stacklevel=2,
                        )
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
