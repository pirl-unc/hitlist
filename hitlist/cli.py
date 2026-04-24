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

"""CLI for hitlist data management.

Usage::

    hitlist data list                         # show registered datasets
    hitlist data available                    # show all known datasets
    hitlist data register iedb /path/to/file  # register a manual download
    hitlist data fetch hpv16                  # auto-download
    hitlist data refresh hpv16                # re-download
    hitlist data info iedb                    # detailed metadata
    hitlist data path iedb                    # print file path
    hitlist data remove iedb                  # unregister
    hitlist data remove iedb --delete         # unregister + delete file
"""

from __future__ import annotations

import argparse
import json
import sys

from .downloads import (
    available_datasets,
    data_dir,
    fetch,
    get_path,
    info,
    list_datasets,
    refresh,
    register,
    remove,
)


def _fmt_size(size: int) -> str:
    if size > 1_000_000_000:
        return f"{size / 1e9:.1f} GB"
    if size > 1_000_000:
        return f"{size / 1e6:.1f} MB"
    if size > 1_000:
        return f"{size / 1e3:.1f} KB"
    return f"{size} B"


def _data_list(args: argparse.Namespace) -> None:
    datasets = list_datasets()
    if not datasets:
        print("No datasets registered.")
        print(f"Data directory: {data_dir()}")
        print("Run 'hitlist data available' to see known datasets.")
        return
    print(f"{'Name':<12} {'Size':>12}  {'Date':<12} {'Index':<8} Description")
    print("-" * 85)

    from .indexer import _cache_is_valid

    for name, ds in sorted(datasets.items()):
        size_str = _fmt_size(ds.get("size_bytes", 0))
        date = ds.get("registered", "")[:10]
        desc = ds.get("description", "")
        idx_status = ""
        if name in ("iedb", "cedar"):
            from pathlib import Path

            p = Path(ds.get("path", ""))
            if p.exists():
                idx_status = "cached" if _cache_is_valid(name, p) else "stale"
            else:
                idx_status = "missing"
        print(f"{name:<12} {size_str:>12}  {date:<12} {idx_status:<8} {desc}")
    print(f"\nData directory: {data_dir()}")
    print("Run 'hitlist data index' to build/rebuild the search index.")


def _data_available(args: argparse.Namespace) -> None:
    datasets = available_datasets()
    registered = set(list_datasets().keys())
    print(f"{'Name':<12} {'Status':<12} Description")
    print("-" * 75)
    for name, desc in sorted(datasets.items()):
        status = "installed" if name in registered else ""
        print(f"{name:<12} {status:<12} {desc}")


def _data_register(args: argparse.Namespace) -> None:
    try:
        p = register(args.name, args.path, description=args.description)
        print(f"Registered '{args.name}' -> {p}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_fetch(args: argparse.Namespace) -> None:
    try:
        p = fetch(args.name, force=args.force)
        print(f"Ready: {p}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_refresh(args: argparse.Namespace) -> None:
    try:
        p = refresh(args.name)
        print(f"Refreshed: {p}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_info(args: argparse.Namespace) -> None:
    try:
        d = info(args.name)
        print(json.dumps(d, indent=2, default=str))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_path(args: argparse.Namespace) -> None:
    try:
        print(str(get_path(args.name)))
    except (KeyError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_remove(args: argparse.Namespace) -> None:
    remove(args.name, delete_file=args.delete)
    if args.delete:
        print(f"Unregistered and deleted '{args.name}'.")
    else:
        print(f"Unregistered '{args.name}' (file kept on disk).")


def _data_build(args: argparse.Namespace) -> None:
    from .builder import build_observations

    build_observations(
        with_flanking=args.with_flanking,
        proteome_release=args.proteome_release,
        force=args.force,
        fetch_missing_proteomes=not args.no_fetch_proteomes,
        use_uniprot_search=args.use_uniprot,
        build_mappings=not args.no_mappings,
    )


def _data_fetch_proteomes(args: argparse.Namespace) -> None:
    """Auto-fetch reference proteomes for species present in observations."""
    import pandas as pd

    from .downloads import fetch_species_proteome, lookup_proteome
    from .observations import is_built, load_observations

    if not is_built():
        print(
            "Observations table not built. Run: hitlist data build",
            file=sys.stderr,
        )
        sys.exit(1)

    obs = load_observations(columns=["source_organism", "mhc_species"])
    counts = pd.Series(dtype=int)
    if "source_organism" in obs.columns:
        counts = obs["source_organism"].value_counts()

    # Also include mhc_species entries that aren't already covered by
    # source_organism — this captures species where source_organism is
    # missing but the MHC allele reveals the host species.
    mhc_counts = obs["mhc_species"].value_counts()
    for sp, n in mhc_counts.items():
        if sp and sp not in counts.index:
            counts[sp] = n

    if counts.empty:
        print("No species information found in observations.")
        sys.exit(0)

    use_uniprot = getattr(args, "use_uniprot", False)

    # Plan the work
    plan: list[tuple[str, int, dict]] = []
    skipped: list[tuple[str, int]] = []
    for organism, n in counts.items():
        if n < args.min_observations:
            continue
        entry = lookup_proteome(str(organism), use_uniprot=use_uniprot)
        if entry is None:
            skipped.append((str(organism), int(n)))
            continue
        plan.append((str(organism), int(n), entry))

    if not plan:
        print(
            f"No registered proteomes for any species with >= {args.min_observations} observations."
        )
        if skipped:
            print("\nUnmatched organisms (no proteome registered):")
            for organism, n in skipped[:10]:
                print(f"  {organism!r}: {n:,} obs")
        sys.exit(0)

    print(f"Fetching reference proteomes for {len(plan)} species/organism(s):\n")
    for organism, n, entry in plan:
        canonical = entry.get("canonical_species", organism)
        kind = entry["kind"]
        proteome_id = entry.get("proteome_id", f"ensembl-{entry.get('release')}")
        print(f"  {organism!r}  ({n:,} obs)  → {canonical}  [{kind}:{proteome_id}]")
    print()

    fetched: list[str] = []
    cached: list[str] = []
    for organism, _n, _entry in plan:
        path = fetch_species_proteome(
            organism, force=args.force, verbose=True, use_uniprot=use_uniprot
        )
        if path is None:
            cached.append(organism)  # ensembl — no local file to track
        else:
            if path.exists():
                fetched.append(organism)

    print(
        f"\nDone. Fetched: {len(fetched)}, Ensembl (no download): {len(cached)}, "
        f"Skipped unknown: {len(skipped)}"
    )
    if skipped:
        print("\nUnmatched organisms (no proteome registered):")
        for organism, n in skipped[:10]:
            print(f"  {organism!r}: {n:,} obs")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


def _data_list_proteomes(args: argparse.Namespace) -> None:
    from .downloads import list_proteomes

    proteomes = list_proteomes()
    if not proteomes:
        print("No proteomes registered.  Run: hitlist data fetch-proteomes")
        return
    print(f"{len(proteomes)} proteomes registered:\n")
    for species, meta in sorted(proteomes.items()):
        kind = meta.get("kind", "?")
        if kind == "ensembl":
            print(
                f"  {species:35s}  ensembl (species={meta.get('species')}, "
                f"release={meta.get('release')})"
            )
        else:
            size = meta.get("size_bytes", 0)
            print(f"  {species:35s}  uniprot:{meta.get('proteome_id')}  ({size:,} bytes)")


def _data_index(args: argparse.Namespace) -> None:
    from .indexer import _cache_dir, _cache_is_valid, _resolve_source_paths, get_index

    paths = _resolve_source_paths()
    if not paths:
        print("No IEDB/CEDAR data registered. Nothing to index.")
        sys.exit(1)

    source = args.source or "merged"
    force = args.force

    # Show cache status before indexing
    if not force:
        for label, path in sorted(paths.items()):
            valid = _cache_is_valid(label, path)
            status = "cached" if valid else "stale/missing"
            print(f"  {label}: {status}")

    study_df, allele_df = get_index(source=source, force=force)
    print(f"\nIndex ({source}):")
    print(f"  Studies:  {len(study_df):,}")
    print(f"  Alleles:  {len(allele_df):,}")
    print(f"  Species:  {study_df['mhc_species'].nunique()}")
    print(f"  Cache:    {_cache_dir()}")


def _build_data_parser(sub: argparse._SubParsersAction) -> None:
    dp = sub.add_parser("data", help="Manage external datasets")
    ds = dp.add_subparsers(dest="data_command")

    ds.add_parser("list", help="Show registered datasets")
    ds.add_parser("available", help="Show all known datasets")

    p = ds.add_parser("register", help="Register a local file")
    p.add_argument("name", help="Dataset name")
    p.add_argument("path", help="Path to the file")
    p.add_argument("--description", "-d", help="Optional description")

    p = ds.add_parser("fetch", help="Download a fetchable dataset")
    p.add_argument("name", help="Dataset name")
    p.add_argument("--force", "-f", action="store_true", help="Re-download")

    p = ds.add_parser("refresh", help="Re-download a fetchable dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("info", help="Detailed metadata for a dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("path", help="Print path to a dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("remove", help="Unregister a dataset")
    p.add_argument("name", help="Dataset name")
    p.add_argument("--delete", action="store_true", help="Also delete the file")

    p = ds.add_parser("build", help="Build unified observations table from IEDB/CEDAR")
    p.add_argument(
        "--no-mappings",
        action="store_true",
        default=False,
        help=(
            "Skip building the peptide_mappings sidecar.  By default, every "
            "build produces multi-mapping peptide→protein attribution and "
            "annotates observations with semicolon-joined gene_names/gene_ids/"
            "protein_ids columns."
        ),
    )
    p.add_argument(
        "--with-flanking",
        action="store_true",
        help="Deprecated — flanks are always stored in peptide_mappings.parquet.",
    )
    p.add_argument(
        "--proteome-release", type=int, default=112, help="Ensembl release (default 112)"
    )
    p.add_argument(
        "--no-fetch-proteomes",
        action="store_true",
        default=False,
        help="Do not auto-fetch missing proteomes when --with-flanking is set",
    )
    p.add_argument(
        "--use-uniprot",
        action="store_true",
        default=False,
        help=(
            "Query UniProt REST for organisms not in the curated registry. "
            "Required to map peptides from rare/pathogen source proteomes."
        ),
    )
    p.add_argument("--force", "-f", action="store_true", help="Rebuild even if cached")

    p = ds.add_parser(
        "fetch-proteomes",
        help="Auto-fetch reference proteomes for species present in observations",
    )
    p.add_argument(
        "--min-observations",
        type=int,
        default=100,
        help="Only fetch proteomes for species with >= N observations (default 100)",
    )
    p.add_argument(
        "--use-uniprot",
        action="store_true",
        default=False,
        help=(
            "Query UniProt REST for organisms not in the curated registry. "
            "Resolved mappings are cached in the manifest."
        ),
    )
    p.add_argument("--force", "-f", action="store_true", help="Re-download cached proteomes")

    ds.add_parser("list-proteomes", help="List downloaded reference proteomes")

    p = ds.add_parser("index", help="Build/rebuild cached index of IEDB/CEDAR data")
    p.add_argument(
        "--source",
        choices=["iedb", "cedar", "merged", "all"],
        help="Source to index (default: all registered sources independently + merged)",
    )
    p.add_argument(
        "--force", "-f", action="store_true", help="Force re-index even if cache is valid"
    )


def _handle_data(args: argparse.Namespace) -> None:
    handlers = {
        "list": _data_list,
        "available": _data_available,
        "register": _data_register,
        "fetch": _data_fetch,
        "refresh": _data_refresh,
        "info": _data_info,
        "path": _data_path,
        "remove": _data_remove,
        "build": _data_build,
        "index": _data_index,
        "fetch-proteomes": _data_fetch_proteomes,
        "list-proteomes": _data_list_proteomes,
    }
    if args.data_command is None:
        print(
            "Usage: hitlist data {list,available,register,fetch,refresh,info,path,"
            "remove,build,index,fetch-proteomes,list-proteomes}"
        )
        sys.exit(1)
    handlers[args.data_command](args)


def _report(args: argparse.Namespace) -> None:
    from .report import run_report

    text = run_report(mhc_class=args.mhc_class, output=args.output)
    if not args.output:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hitlist",
        description="hitlist: curated mass spectrometry evidence for MHC ligand data",
    )
    sub = parser.add_subparsers(dest="command")
    _build_data_parser(sub)

    p_report = sub.add_parser(
        "report", help="Generate data quality report from registered IEDB/CEDAR"
    )
    p_report.add_argument("--class", dest="mhc_class", help="MHC class filter (I or II)")
    p_report.add_argument("--output", "-o", help="Save report to file")

    # ── export subcommand ──────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export curated study metadata as CSV")
    export_sub = p_export.add_subparsers(dest="export_command")

    p_samples = export_sub.add_parser("samples", help="Per-sample MS conditions table")
    p_samples.add_argument("--class", dest="mhc_class", help="Filter to MHC class (I or II)")
    p_samples.add_argument(
        "--with-expression-anchors",
        action="store_true",
        help=(
            "Emit the line-expression anchor resolution for every sample "
            "(expression_backend, expression_key, expression_match_tier, "
            "expression_parent_key; issue #140). Replaces the default "
            "acquisition-metadata sample table."
        ),
    )
    p_samples.add_argument("--output", "-o", help="Write CSV to file")

    p_summary = export_sub.add_parser("summary", help="Species x MHC class summary")
    p_summary.add_argument("--class", dest="mhc_class", help="Filter to MHC class (I or II)")
    p_summary.add_argument("--output", "-o", help="Write CSV to file")

    p_alleles = export_sub.add_parser("alleles", help="Validate MHC alleles with mhcgnomes")
    p_alleles.add_argument("--output", "-o", help="Write CSV to file")

    p_data_alleles = export_sub.add_parser(
        "data-alleles", help="Validate all MHC alleles in local IEDB/CEDAR with mhcgnomes"
    )
    p_data_alleles.add_argument("--output", "-o", help="Write CSV to file")

    p_obs = export_sub.add_parser(
        "observations", help="Unified observations table: peptides + sample metadata"
    )
    p_obs.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_obs.add_argument("--species", help="Filter by MHC species")
    p_obs.add_argument("--instrument-type", help="Instrument type (Orbitrap, timsTOF)")
    p_obs.add_argument("--acquisition-mode", help="Acquisition mode (DDA, DIA, PRM)")
    p_obs.add_argument(
        "--mono-allelic",
        dest="mono_allelic",
        action="store_true",
        default=None,
        help="Only mono-allelic samples",
    )
    p_obs.add_argument(
        "--multi-allelic",
        dest="mono_allelic",
        action="store_false",
        help="Only multi-allelic samples",
    )
    p_obs.add_argument(
        "--min-allele-resolution",
        choices=["four_digit", "two_digit", "serological", "class_only"],
        help="Minimum allele resolution",
    )
    p_obs.add_argument(
        "--mhc-allele",
        action="append",
        help=(
            "Filter to peptides whose mhc_restriction matches (after allele "
            "normalization).  Repeatable or comma-separated, e.g. "
            "'--mhc-allele HLA-A*02:01 --mhc-allele HLA-B*07:02'."
        ),
    )
    p_obs.add_argument(
        "--gene",
        action="append",
        help=(
            "Filter to peptides from these genes.  Accepts current symbols, "
            "Ensembl gene IDs (ENSG...), and old/alias symbols (resolved via "
            "HGNC).  Repeatable or comma-separated.  Requires a flanking-built "
            "observations table."
        ),
    )
    p_obs.add_argument(
        "--gene-name",
        action="append",
        help="Exact match on gene_name column (no HGNC synonym lookup).",
    )
    p_obs.add_argument(
        "--gene-id",
        action="append",
        help="Exact match on gene_id column (Ensembl ENSG ID).",
    )
    p_obs.add_argument(
        "--serotype",
        action="append",
        help=(
            "Filter by HLA serotype.  Accepts locus-specific names (A2, A24, "
            "B57, DR15) or public epitopes (Bw4, Bw6, C1, C2).  Matches any "
            "serotype an allele belongs to, so --serotype Bw4 returns A*24:02, "
            "B*27:05, B*57:01, etc.  Repeatable or comma-separated."
        ),
    )
    p_obs.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_bind = export_sub.add_parser(
        "binding",
        help=(
            "Binding-assay index (peptide microarray, refolding, MEDi, "
            "quantitative tiers).  Separate from MS observations."
        ),
    )
    p_bind.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_bind.add_argument("--species", help="Filter by MHC species")
    p_bind.add_argument(
        "--source",
        choices=["iedb", "cedar"],
        help="Filter by data source (supplementary data is MS-only and never appears here)",
    )
    p_bind.add_argument(
        "--min-allele-resolution",
        choices=["four_digit", "two_digit", "serological", "class_only"],
        help="Minimum allele resolution",
    )
    p_bind.add_argument(
        "--mhc-allele",
        action="append",
        help=(
            "Filter to rows whose mhc_restriction matches (after allele "
            "normalization).  Repeatable or comma-separated."
        ),
    )
    p_bind.add_argument(
        "--gene",
        action="append",
        help=(
            "Filter by gene (HGNC symbol, Ensembl ID, or old alias).  "
            "Repeatable or comma-separated.  Requires a mappings-built index."
        ),
    )
    p_bind.add_argument("--gene-name", action="append", help="Exact match on gene_name column.")
    p_bind.add_argument("--gene-id", action="append", help="Exact match on gene_id column.")
    p_bind.add_argument(
        "--serotype",
        action="append",
        help=(
            "Filter by HLA serotype (locus-specific A24/B57/DR15 or public "
            "epitopes Bw4/Bw6/C1/C2).  Repeatable or comma-separated."
        ),
    )
    p_bind.add_argument(
        "--assay-method",
        action="append",
        help=(
            "Filter to rows whose IEDB/CEDAR assay_method matches (case-"
            "insensitive substring).  Examples: 'purified MHC/direct/"
            "fluorescence', 'cellular MHC/direct'.  Repeatable."
        ),
    )
    p_bind.add_argument(
        "--measurement-units",
        action="append",
        help=(
            "Filter to rows reporting in these units (case-insensitive "
            "exact match).  Pair with --quantitative-value-{min,max} to "
            "avoid mixing unit systems (e.g. nM vs log10(nM))."
        ),
    )
    p_bind.add_argument(
        "--quantitative-value-min",
        type=float,
        help="Inclusive lower bound on quantitative_value.  Excludes NaN.",
    )
    p_bind.add_argument(
        "--quantitative-value-max",
        type=float,
        help="Inclusive upper bound on quantitative_value.  Excludes NaN.",
    )
    p_bind.add_argument(
        "--has-quantitative-value",
        dest="has_quantitative_value",
        action="store_true",
        default=None,
        help="Keep only rows with a non-NaN quantitative_value (IC50/EC50/Kd rows).",
    )
    p_bind.add_argument(
        "--qualitative-only",
        dest="has_quantitative_value",
        action="store_false",
        help="Keep only qualitative-tier rows (no numeric value reported).",
    )
    p_bind.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_training = export_sub.add_parser(
        "training",
        help=(
            "Unified pMHC training export composed from the canonical MS, "
            "binding, and peptide-mappings indexes."
        ),
    )
    p_training.add_argument(
        "--include-evidence",
        choices=["ms", "binding", "both"],
        default="both",
        help="Which evidence families to include (default: both).",
    )
    p_training.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_training.add_argument("--species", help="Filter by MHC species")
    p_training.add_argument(
        "--source",
        choices=["iedb", "cedar", "supplement"],
        help=(
            "Filter by data source. Supplementary data is MS-only, so it "
            "only contributes rows when --include-evidence includes ms."
        ),
    )
    p_training.add_argument(
        "--instrument-type",
        help="MS-only filter on instrument type (Orbitrap, timsTOF).",
    )
    p_training.add_argument(
        "--acquisition-mode",
        help="MS-only filter on acquisition mode (DDA, DIA, PRM).",
    )
    p_training.add_argument(
        "--mono-allelic",
        dest="mono_allelic",
        action="store_true",
        default=None,
        help="MS-only filter: keep mono-allelic rows.",
    )
    p_training.add_argument(
        "--multi-allelic",
        dest="mono_allelic",
        action="store_false",
        help="MS-only filter: keep multi-allelic rows.",
    )
    p_training.add_argument(
        "--min-allele-resolution",
        choices=["four_digit", "two_digit", "serological", "class_only"],
        help="Minimum allele resolution",
    )
    p_training.add_argument(
        "--mhc-allele",
        action="append",
        help="Exact allele filter. Repeatable or comma-separated.",
    )
    p_training.add_argument(
        "--gene",
        action="append",
        help=(
            "Filter by gene (HGNC symbol, Ensembl ID, or old alias). "
            "Repeatable or comma-separated. Requires a mappings-built index."
        ),
    )
    p_training.add_argument("--gene-name", action="append", help="Exact match on gene_name.")
    p_training.add_argument("--gene-id", action="append", help="Exact match on gene_id.")
    p_training.add_argument(
        "--peptide",
        action="append",
        help="Filter to one or more peptide sequences. Repeatable or comma-separated.",
    )
    p_training.add_argument(
        "--serotype",
        action="append",
        help="Filter by HLA serotype. Repeatable or comma-separated.",
    )
    p_training.add_argument("--length-min", type=int, help="Minimum peptide length (inclusive).")
    p_training.add_argument("--length-max", type=int, help="Maximum peptide length (inclusive).")
    p_training.add_argument(
        "--explode-mappings",
        action="store_true",
        help=(
            "Expand to one row per (evidence row, peptide mapping), adding "
            "protein/position/flank columns for training pipelines such as Presto."
        ),
    )
    p_training.add_argument(
        "--with-peptide-origin",
        action="store_true",
        help=(
            "Enrich each row with a per-sample expression anchor and "
            "peptide-origin call (argmax-TPM gene with provenance). "
            "Adds expression_backend / expression_key / "
            "expression_match_tier / expression_parent_key and "
            "peptide_origin_gene / peptide_origin_tpm / "
            "peptide_origin_dominant_transcript columns. Issue #140."
        ),
    )
    p_training.add_argument(
        "--proteome-release",
        type=int,
        default=112,
        help=(
            "Ensembl release used for transcript-isoform lookup when "
            "--with-peptide-origin is set and transcript-level TPM is "
            "available (default: 112)."
        ),
    )
    p_training.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_bulk = export_sub.add_parser(
        "bulk",
        help=(
            "Bulk (non-MHC) proteomics index: shotgun MS peptides + protein "
            "abundances from CCLE and Bekker-Jensen 2017.  NOT MHC-ligand "
            "data — see 'observations' / 'binding' for those."
        ),
    )
    p_bulk.add_argument(
        "--granularity",
        choices=["peptide", "protein", "both"],
        default="both",
        help="Rows to export (default: both).",
    )
    p_bulk.add_argument("--cell-line", action="append", help="Filter to one or more cell lines.")
    p_bulk.add_argument(
        "--gene-name",
        action="append",
        help="Filter to one or more HGNC gene symbols (exact match).",
    )
    p_bulk.add_argument(
        "--uniprot-acc",
        action="append",
        help="Filter peptide rows to one or more UniProt accessions (exact match).",
    )
    p_bulk.add_argument(
        "--source",
        choices=["CCLE_Nusinow_2020", "Bekker-Jensen_2017"],
        help="Restrict to one bulk proteomics source.",
    )
    p_bulk.add_argument(
        "--digestion-enzyme",
        action="append",
        help=(
            "Filter to one or more enzymes: 'Trypsin/P (cleaves K/R except before P)', "
            "'Chymotrypsin', 'GluC', 'LysC'."
        ),
    )
    p_bulk.add_argument(
        "--n-fractions",
        action="append",
        type=int,
        help="Filter to one or more fractionation depths (12/14/39/46/50/70).",
    )
    p_bulk.add_argument(
        "--enrichment",
        choices=["none", "TiO2", "both"],
        default="none",
        help=(
            "Enrichment filter: 'none' (baseline, default), 'TiO2' (phospho "
            "only), or 'both' (include both populations)."
        ),
    )
    p_bulk.add_argument(
        "--fractionation-ph",
        action="append",
        type=float,
        help="Filter by high-pH SPE buffer pH (8.0 or 10.0).",
    )
    p_bulk.add_argument(
        "--length-min", type=int, help="Minimum peptide length (inclusive; peptide rows only)."
    )
    p_bulk.add_argument(
        "--length-max", type=int, help="Maximum peptide length (inclusive; peptide rows only)."
    )
    p_bulk.add_argument(
        "--abundance-percentile-min",
        type=float,
        help="Minimum abundance percentile (0.0-1.0; protein rows only).",
    )
    p_bulk.add_argument(
        "--abundance-percentile-max",
        type=float,
        help="Maximum abundance percentile (0.0-1.0; protein rows only).",
    )
    p_bulk.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_line_expr = export_sub.add_parser(
        "line-expression",
        help=(
            "Per-line RNA / transcript TPM with provenance (source, "
            "normalization, license).  Emits line_expression.parquet "
            "contents or the packaged CSVs when no build has run."
        ),
    )
    p_line_expr.add_argument(
        "--line-key",
        action="append",
        help="Filter to one or more line_key values (e.g. GM12878, HeLa).",
    )
    p_line_expr.add_argument(
        "--gene-name",
        action="append",
        help="Filter to one or more HGNC gene symbols.",
    )
    p_line_expr.add_argument(
        "--gene-id",
        action="append",
        help="Filter to one or more Ensembl gene IDs (unversioned).",
    )
    p_line_expr.add_argument(
        "--granularity",
        choices=["gene", "transcript"],
        help="Restrict to one granularity.",
    )
    p_line_expr.add_argument(
        "--source-id",
        action="append",
        help="Restrict to one or more source_ids from sources.yaml.",
    )
    p_line_expr.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_counts = export_sub.add_parser(
        "counts", help="Count peptides per study from local IEDB/CEDAR"
    )
    p_counts.add_argument(
        "--source",
        choices=["iedb", "cedar", "merged", "all"],
        default="merged",
        help="Data source: iedb, cedar, merged (deduped, default), or all (side-by-side)",
    )
    p_counts.add_argument("--output", "-o", help="Write CSV to file")

    p_reassign = sub.add_parser(
        "reassign-alleles",
        help=(
            "Reassign class-only peptides (mhc_restriction='HLA class I') "
            "to their best-scoring allele via MHCflurry / NetMHCpan"
        ),
    )
    p_reassign.add_argument(
        "--method",
        choices=["mhcflurry", "netmhcpan"],
        default="mhcflurry",
        help="Binding predictor (default: mhcflurry)",
    )
    p_reassign.add_argument(
        "--class",
        dest="mhc_class",
        choices=["I"],
        default="I",
        help="MHC class (only I supported in v1.8.0)",
    )
    p_reassign.add_argument(
        "--max-alleles-per-sample",
        type=int,
        default=30,
        help=(
            "Skip samples with more alleles than this — likely pooled-donor "
            "curation artifacts whose 'best allele' is meaningless (default: 30)"
        ),
    )
    p_reassign.add_argument("--output", "-o", help="Write CSV to file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "data":
        _handle_data(args)
    elif args.command == "report":
        _report(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "reassign-alleles":
        _reassign(args)


def _reassign(args: argparse.Namespace) -> None:
    from .predict import reassign_class_only_alleles

    try:
        df = reassign_class_only_alleles(
            method=args.method,
            mhc_class=args.mhc_class,
            max_alleles_per_sample=args.max_alleles_per_sample,
        )
    except (RuntimeError, NotImplementedError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Scored {len(df):,} class-only peptides.  "
        f"Strong binders: {df['is_strong_binder'].sum():,}  "
        f"Weak binders: {df['is_weak_binder'].sum():,}",
        file=sys.stderr,
    )

    if args.output:
        if args.output.endswith(".parquet"):
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Wrote {len(df):,} rows to {args.output}", file=sys.stderr)
    else:
        print(df.to_csv(index=False), end="")


def _export_bulk(args: argparse.Namespace):
    """Run the ``hitlist export bulk`` subcommand.

    Wires argparse flags to the :mod:`hitlist.bulk_proteomics` loader kwargs.
    Returns the concatenated peptide + protein frame (or just one, per
    ``--granularity``) with all requested filters applied.
    """
    import pandas as pd

    from .bulk_proteomics import load_bulk_peptides, load_bulk_proteomics

    # Resolve --enrichment: CLI accepts 'none' / 'TiO2' / 'both'. 'both' maps
    # to the loader's None sentinel (both populations).
    enrichment = getattr(args, "enrichment", "none")
    enrichment_kwarg = None if enrichment == "both" else enrichment

    common_kwargs: dict = {
        "cell_line": getattr(args, "cell_line", None) or None,
        "gene_name": getattr(args, "gene_name", None) or None,
        "digestion_enzyme": getattr(args, "digestion_enzyme", None) or None,
        "n_fractions_in_run": getattr(args, "n_fractions", None) or None,
        "enrichment": enrichment_kwarg,
        "fractionation_ph": getattr(args, "fractionation_ph", None) or None,
    }

    granularity = getattr(args, "granularity", "both")
    frames: list[pd.DataFrame] = []

    if granularity in ("peptide", "both"):
        pep = load_bulk_peptides(
            uniprot_acc=getattr(args, "uniprot_acc", None) or None,
            length_min=getattr(args, "length_min", None),
            length_max=getattr(args, "length_max", None),
            **common_kwargs,
        )
        # Tag granularity for the combined-output case so callers can
        # tell peptide vs protein rows apart without schema introspection.
        if "granularity" not in pep.columns:
            pep = pep.assign(granularity="peptide")
        frames.append(pep)

    if granularity in ("protein", "both"):
        prot = load_bulk_proteomics(
            source=getattr(args, "source", None),
            abundance_percentile_min=getattr(args, "abundance_percentile_min", None),
            abundance_percentile_max=getattr(args, "abundance_percentile_max", None),
            **common_kwargs,
        )
        if "granularity" not in prot.columns:
            prot = prot.assign(granularity="protein")
        frames.append(prot)

    # Concat preserves both sets of columns (NaN-fill where unique).
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _export_training(args: argparse.Namespace):
    """Run the ``hitlist export training`` subcommand."""
    from .export import generate_training_table

    return generate_training_table(
        include_evidence=getattr(args, "include_evidence", "both"),
        mhc_class=getattr(args, "mhc_class", None),
        species=getattr(args, "species", None),
        source=getattr(args, "source", None),
        instrument_type=getattr(args, "instrument_type", None),
        acquisition_mode=getattr(args, "acquisition_mode", None),
        is_mono_allelic=getattr(args, "mono_allelic", None),
        min_allele_resolution=getattr(args, "min_allele_resolution", None),
        mhc_allele=getattr(args, "mhc_allele", None),
        gene=getattr(args, "gene", None),
        gene_name=getattr(args, "gene_name", None),
        gene_id=getattr(args, "gene_id", None),
        peptide=getattr(args, "peptide", None),
        serotype=getattr(args, "serotype", None),
        length_min=getattr(args, "length_min", None),
        length_max=getattr(args, "length_max", None),
        explode_mappings=getattr(args, "explode_mappings", False),
        with_peptide_origin=getattr(args, "with_peptide_origin", False),
        proteome_release=getattr(args, "proteome_release", 112),
    )


def _export_progress(msg: str) -> None:
    """Print a single-line progress message to stderr (not stdout).

    stderr keeps the CSV output pipe-clean. Only emits when stderr is a
    TTY (interactive terminal) so scripts piping both stdout and stderr
    don't get spammed. See pirl-unc/hitlist#119.
    """
    if sys.stderr.isatty():
        print(f"[hitlist] {msg}", file=sys.stderr, flush=True)


def _export(args: argparse.Namespace) -> None:
    import time as _time

    from .export import (
        collect_alleles_from_data,
        count_peptides_by_study,
        generate_binding_table,
        generate_ms_samples_table,
        generate_observations_table,
        generate_species_summary,
        validate_mhc_alleles,
    )

    cmd = args.export_command
    t0 = _time.perf_counter()

    if cmd == "samples":
        if getattr(args, "with_expression_anchors", False):
            _export_progress("Resolving expression anchors for every ms_sample (issue #140) ...")
            from .export import generate_sample_expression_table

            df = generate_sample_expression_table(mhc_class=args.mhc_class)
        else:
            _export_progress("Building ms_samples table from pmid_overrides.yaml ...")
            df = generate_ms_samples_table(mhc_class=args.mhc_class)
    elif cmd == "summary":
        _export_progress("Loading observations.parquet for species summary ...")
        df = generate_species_summary(mhc_class=args.mhc_class)
    elif cmd == "alleles":
        _export_progress("Validating allele strings via mhcgnomes ...")
        df = validate_mhc_alleles()
    elif cmd == "data-alleles":
        _export_progress("Collecting alleles from local IEDB/CEDAR data ...")
        df = collect_alleles_from_data(source=getattr(args, "source", "merged"))
    elif cmd == "counts":
        _export_progress("Counting peptides per study from local IEDB/CEDAR ...")
        df = count_peptides_by_study(source=args.source)
    elif cmd == "observations":
        _export_progress("Loading observations.parquet + joining sample metadata ...")
        try:
            df = generate_observations_table(
                mhc_class=args.mhc_class,
                species=getattr(args, "species", None),
                source=getattr(args, "source", None),
                instrument_type=getattr(args, "instrument_type", None),
                acquisition_mode=getattr(args, "acquisition_mode", None),
                is_mono_allelic=getattr(args, "mono_allelic", None),
                min_allele_resolution=getattr(args, "min_allele_resolution", None),
                mhc_allele=getattr(args, "mhc_allele", None),
                gene=getattr(args, "gene", None),
                gene_name=getattr(args, "gene_name", None),
                gene_id=getattr(args, "gene_id", None),
                serotype=getattr(args, "serotype", None),
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "binding":
        _export_progress("Loading binding.parquet ...")
        try:
            df = generate_binding_table(
                mhc_class=args.mhc_class,
                species=getattr(args, "species", None),
                source=getattr(args, "source", None),
                min_allele_resolution=getattr(args, "min_allele_resolution", None),
                mhc_allele=getattr(args, "mhc_allele", None),
                gene=getattr(args, "gene", None),
                gene_name=getattr(args, "gene_name", None),
                gene_id=getattr(args, "gene_id", None),
                serotype=getattr(args, "serotype", None),
                assay_method=getattr(args, "assay_method", None),
                measurement_units=getattr(args, "measurement_units", None),
                quantitative_value_min=getattr(args, "quantitative_value_min", None),
                quantitative_value_max=getattr(args, "quantitative_value_max", None),
                has_quantitative_value=getattr(args, "has_quantitative_value", None),
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "training":
        _export_progress(
            "Composing the training export from observations.parquet, binding.parquet, "
            "and peptide_mappings.parquet ..."
        )
        try:
            df = _export_training(args)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "bulk":
        _export_progress("Loading bulk_proteomics.parquet ...")
        df = _export_bulk(args)
    elif cmd == "line-expression":
        _export_progress("Loading line_expression.parquet ...")
        from .line_expression import load_line_expression

        df = load_line_expression(
            line_key=getattr(args, "line_key", None),
            gene_name=getattr(args, "gene_name", None),
            gene_id=getattr(args, "gene_id", None),
            granularity=getattr(args, "granularity", None),
            source_id=getattr(args, "source_id", None),
        )
    else:
        print(
            "Usage: hitlist export "
            "{samples,summary,alleles,data-alleles,counts,observations,"
            "binding,training,bulk,line-expression}"
        )
        sys.exit(1)

    elapsed = _time.perf_counter() - t0
    _export_progress(f"Got {len(df):,} rows in {elapsed:.1f}s")

    if args.output:
        _export_progress(f"Writing to {args.output} ...")
        if args.output.endswith(".parquet"):
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
