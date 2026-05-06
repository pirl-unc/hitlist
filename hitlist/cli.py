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


def _print_banner(stream=sys.stderr) -> None:
    """One-line version + observations-build status banner.

    Shown above subgroup help so users get quick orientation: which
    version they're on, and whether the canonical artifact downstream
    commands depend on has been built yet.
    """
    from .version import __version__

    print(f"hitlist v{__version__}", file=stream)
    try:
        from .observations import is_built

        built = is_built()
    except Exception:
        built = False
    state = "built" if built else "NOT BUILT — run `hitlist build observations`"
    print(f"observations.parquet: {state}", file=stream)
    print(file=stream)


def _print_subgroup_help(parser: argparse.ArgumentParser) -> None:
    """Banner + full help for a subgroup parser.

    Used when a user runs e.g. ``hitlist pmhc`` with no subcommand —
    instead of a one-line "Usage: hitlist pmhc {query}" string, show
    what each subcommand actually does and the args it takes.
    """
    _print_banner()
    parser.print_help(sys.stderr)


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
    dp.set_defaults(_subgroup_parser=dp)
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

    p = ds.add_parser(
        "build",
        help="DEPRECATED — use `hitlist build observations`. Builds observations.parquet.",
    )
    _add_build_observations_args(p)

    p = ds.add_parser(
        "fetch-proteomes",
        help="DEPRECATED — use `hitlist build proteomes`. Auto-fetch reference proteomes.",
    )
    _add_build_proteomes_args(p)

    ds.add_parser("list-proteomes", help="List downloaded reference proteomes")

    p = ds.add_parser(
        "index",
        help="DEPRECATED — use `hitlist build index`. Build cached IEDB/CEDAR scan index.",
    )
    _add_build_index_args(p)


def _add_build_observations_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for the `hitlist build observations` (== legacy `data build`)
    command.  Shared between the canonical and deprecated entry points so the
    flag set never drifts between them."""
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


def _add_build_proteomes_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist build proteomes` (== legacy `data fetch-proteomes`)."""
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


def _add_build_index_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist build index` (== legacy `data index`)."""
    p.add_argument(
        "--source",
        choices=["iedb", "cedar", "merged", "all"],
        help="Source to index (default: all registered sources independently + merged)",
    )
    p.add_argument(
        "--force", "-f", action="store_true", help="Force re-index even if cache is valid"
    )


def _add_export_bulk_proteomics_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist export bulk-proteomics` (== legacy `bulk`).

    Bulk (non-MHC) shotgun proteomics index — CCLE + Bekker-Jensen 2017.
    Renamed in v1.27.0 because the legacy ``bulk`` name read like part of the
    MHC-ligand family it lives next to; ``bulk-proteomics`` makes the scope
    explicit.
    """
    p.add_argument(
        "--granularity",
        choices=["peptide", "protein", "both"],
        default="both",
        help="Rows to export (default: both).",
    )
    p.add_argument(
        "--cell-line", action="extend", nargs="+", help="Filter to one or more cell lines."
    )
    p.add_argument(
        "--gene-name",
        action="extend",
        nargs="+",
        help="Filter to one or more HGNC gene symbols (exact match).",
    )
    p.add_argument(
        "--uniprot-acc",
        action="extend",
        nargs="+",
        help="Filter peptide rows to one or more UniProt accessions (exact match).",
    )
    p.add_argument(
        "--source",
        choices=["CCLE_Nusinow_2020", "Bekker-Jensen_2017"],
        help="Restrict to one bulk proteomics source.",
    )
    p.add_argument(
        "--digestion-enzyme",
        action="extend",
        nargs="+",
        help=(
            "Filter to one or more enzymes: 'Trypsin/P (cleaves K/R except before P)', "
            "'Chymotrypsin', 'GluC', 'LysC'."
        ),
    )
    p.add_argument(
        "--n-fractions",
        action="extend",
        nargs="+",
        type=int,
        help="Filter to one or more fractionation depths (12/14/39/46/50/70).",
    )
    p.add_argument(
        "--enrichment",
        choices=["none", "TiO2", "both"],
        default="none",
        help=(
            "Enrichment filter: 'none' (baseline, default), 'TiO2' (phospho "
            "only), or 'both' (include both populations)."
        ),
    )
    p.add_argument(
        "--fractionation-ph",
        action="extend",
        nargs="+",
        type=float,
        help="Filter by high-pH SPE buffer pH (8.0 or 10.0).",
    )
    p.add_argument(
        "--length-min", type=int, help="Minimum peptide length (inclusive; peptide rows only)."
    )
    p.add_argument(
        "--length-max", type=int, help="Maximum peptide length (inclusive; peptide rows only)."
    )
    p.add_argument(
        "--abundance-percentile-min",
        type=float,
        help="Minimum abundance percentile (0.0-1.0; protein rows only).",
    )
    p.add_argument(
        "--abundance-percentile-max",
        type=float,
        help="Maximum abundance percentile (0.0-1.0; protein rows only).",
    )
    p.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")


def _add_peptide_counts_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist export peptide-counts` (merges legacy
    ``summary`` and ``counts`` into one command differentiated by ``--by``)."""
    p.add_argument(
        "--by",
        choices=["class", "study"],
        default="class",
        help=(
            "Aggregation level: 'class' (one row per species x mhc_class, "
            "default — was `export summary`), 'study' (one row per "
            "PMID x class x species — was `export counts`)."
        ),
    )
    p.add_argument(
        "--class",
        dest="mhc_class",
        help="Filter to MHC class (I or II). Applies to --by class.",
    )
    p.add_argument(
        "--source",
        choices=["iedb", "cedar", "merged", "all"],
        default="merged",
        help="Data source for --by study (default: merged).",
    )
    p.add_argument("--output", "-o", help="Write CSV to file")


def _add_samples_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist samples` (== legacy `export samples`)."""
    p.add_argument("--class", dest="mhc_class", help="Filter to MHC class (I or II)")
    p.add_argument(
        "--with-expression-anchors",
        action="store_true",
        help=(
            "Emit the line-expression anchor resolution for every sample "
            "(expression_backend, expression_key, expression_match_tier, "
            "expression_parent_key; issue #140). Replaces the default "
            "acquisition-metadata sample table."
        ),
    )
    p.add_argument(
        "--apm-only",
        action="store_true",
        help=(
            "Filter to samples where any antigen-processing-machinery "
            "gene was perturbed (B2M, TAP1, TAP2, TAPBP, ERAP1/2, PDIA3, "
            "CALR, CANX, IRF2, GANAB, SPPL3, NLRC5, CIITA, HLA-DM, "
            "HLA-DO, CD74, cathepsin, RFX, bare-lymphocyte-syndrome). "
            "See apm_perturbed and apm_genes_perturbed output columns."
        ),
    )
    p.add_argument("--output", "-o", help="Write CSV to file")


def _add_line_expression_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist expression` (== legacy `export line-expression`)."""
    p.add_argument(
        "--line-key",
        action="extend",
        nargs="+",
        help="Filter to one or more line_key values (e.g. GM12878, HeLa).",
    )
    p.add_argument(
        "--gene-name",
        action="extend",
        nargs="+",
        help="Filter to one or more HGNC gene symbols.",
    )
    p.add_argument(
        "--gene-id",
        action="extend",
        nargs="+",
        help="Filter to one or more Ensembl gene IDs (unversioned).",
    )
    p.add_argument(
        "--granularity",
        choices=["gene", "transcript"],
        help="Restrict to one granularity.",
    )
    p.add_argument(
        "--source-id",
        action="extend",
        nargs="+",
        help="Restrict to one or more source_ids from sources.yaml.",
    )
    p.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")


def _build_top_level_build_parser(sub: argparse._SubParsersAction) -> None:
    """Register the canonical ``hitlist build`` group."""
    p_build = sub.add_parser(
        "build",
        help=(
            "Build canonical artifacts: observations.parquet (from IEDB/CEDAR), "
            "cached scan index, reference proteomes."
        ),
    )
    p_build.set_defaults(_subgroup_parser=p_build)
    bs = p_build.add_subparsers(dest="build_command")

    p = bs.add_parser("observations", help="Build unified observations table from IEDB/CEDAR")
    _add_build_observations_args(p)

    p = bs.add_parser("index", help="Build/rebuild cached scan index of IEDB/CEDAR data")
    _add_build_index_args(p)

    p = bs.add_parser(
        "proteomes",
        help="Auto-fetch reference proteomes for species present in observations",
    )
    _add_build_proteomes_args(p)


def _handle_build(args: argparse.Namespace) -> None:
    """Dispatch ``hitlist build <subcommand>`` to the same handlers as the
    legacy ``hitlist data {build,index,fetch-proteomes}`` entry points."""
    cmd = getattr(args, "build_command", None)
    if cmd is None:
        _print_subgroup_help(args._subgroup_parser)
        sys.exit(1)
    handlers = {
        "observations": _data_build,
        "index": _data_index,
        "proteomes": _data_fetch_proteomes,
    }
    handlers[cmd](args)


_DEPRECATED_DATA_SUBCOMMANDS = {
    "build": "hitlist build observations",
    "index": "hitlist build index",
    "fetch-proteomes": "hitlist build proteomes",
}


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
        _print_subgroup_help(args._subgroup_parser)
        sys.exit(1)
    if args.data_command in _DEPRECATED_DATA_SUBCOMMANDS:
        new_path = _DEPRECATED_DATA_SUBCOMMANDS[args.data_command]
        print(
            f"Note: `hitlist data {args.data_command}` is deprecated; use "
            f"`{new_path}` instead.  Will be removed in v2.0.",
            file=sys.stderr,
        )
    handlers[args.data_command](args)


def _report(args: argparse.Namespace) -> None:
    from .report import run_report

    text = run_report(
        mhc_class=args.mhc_class,
        output=args.output,
        from_csv=getattr(args, "from_csv", False),
    )
    if not args.output:
        print(text)


def main() -> None:
    from .version import __version__

    parser = argparse.ArgumentParser(
        prog="hitlist",
        description="hitlist: curated mass spectrometry evidence for MHC ligand data",
    )
    parser.add_argument("--version", action="version", version=f"hitlist {__version__}")
    sub = parser.add_subparsers(dest="command")
    _build_data_parser(sub)
    _build_top_level_build_parser(sub)

    p_report = sub.add_parser(
        "report",
        help=(
            "Data quality report.  Reads observations.parquet by default "
            "(instant); pass --from-csv to fall back to a live raw-CSV scan."
        ),
    )
    p_report.add_argument("--class", dest="mhc_class", help="MHC class filter (I or II)")
    p_report.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "Scan the raw IEDB/CEDAR CSVs instead of reading observations.parquet. "
            "Slow (minutes); useful before `hitlist build observations` has run."
        ),
    )
    p_report.add_argument("--output", "-o", help="Save report to file")

    # ── export subcommand ──────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export curated study metadata as CSV")
    p_export.set_defaults(_subgroup_parser=p_export)
    export_sub = p_export.add_subparsers(dest="export_command")

    p_samples = export_sub.add_parser(
        "samples",
        help="DEPRECATED — use `hitlist samples`. Per-sample MS conditions table.",
    )
    _add_samples_args(p_samples)

    # ── canonical: peptide-counts (was: summary + counts) ──────────────
    p_pc = export_sub.add_parser(
        "peptide-counts",
        help="Count peptides per (species x class) or per (study x class x species).",
    )
    _add_peptide_counts_args(p_pc)

    # Legacy aliases — kept for one release with a deprecation notice.
    p_summary = export_sub.add_parser(
        "summary",
        help="DEPRECATED — use `hitlist export peptide-counts --by class`.",
    )
    p_summary.add_argument("--class", dest="mhc_class", help="Filter to MHC class (I or II)")
    p_summary.add_argument("--output", "-o", help="Write CSV to file")

    p_alleles = export_sub.add_parser(
        "alleles",
        help="DEPRECATED — use `hitlist qc normalization` for actionable allele drift.",
    )
    p_alleles.add_argument("--output", "-o", help="Write CSV to file")

    p_data_alleles = export_sub.add_parser(
        "data-alleles",
        help="DEPRECATED — use `hitlist qc resolution` for an actionable allele histogram.",
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
        action="extend",
        nargs="+",
        help=(
            "Filter to peptides whose mhc_restriction matches (after allele "
            "normalization).  Space-separated, comma-separated, or repeated, e.g. "
            "'--mhc-allele HLA-A*02:01 --mhc-allele HLA-B*07:02'."
        ),
    )
    p_obs.add_argument(
        "--mhc-allele-in-set",
        action="extend",
        nargs="+",
        help=(
            "Filter to rows whose mhc_allele_set (issue #137 expanded "
            "candidate-allele set) contains any of the listed alleles. "
            "Captures both 4-digit rows AND multi-allele class-only rows whose "
            "donor genotype includes the allele.  Repeatable."
        ),
    )
    p_obs.add_argument(
        "--mhc-allele-provenance",
        action="extend",
        nargs="+",
        choices=["exact", "sample_allele_match", "pmid_class_pool", "unmatched"],
        help=(
            "Filter by how the allele set was obtained.  Use 'exact' for "
            "strict-resolution training; ['exact', 'sample_allele_match'] for "
            "MIL / noisy-OR over small trusted sets."
        ),
    )
    p_obs.add_argument(
        "--gene",
        action="extend",
        nargs="+",
        help=(
            "Filter to peptides from these genes.  Accepts current symbols, "
            "Ensembl gene IDs (ENSG...), and old/alias symbols (resolved via "
            "HGNC).  Space-separated, comma-separated, or repeated.  Requires a flanking-built "
            "observations table."
        ),
    )
    p_obs.add_argument(
        "--gene-name",
        action="extend",
        nargs="+",
        help="Exact match on gene_name column (no HGNC synonym lookup).",
    )
    p_obs.add_argument(
        "--gene-id",
        action="extend",
        nargs="+",
        help="Exact match on gene_id column (Ensembl ENSG ID).",
    )
    p_obs.add_argument(
        "--serotype",
        action="extend",
        nargs="+",
        help=(
            "Filter by HLA serotype.  Accepts locus-specific names (A2, A24, "
            "B57, DR15) or public epitopes (Bw4, Bw6, C1, C2).  Matches any "
            "serotype an allele belongs to, so --serotype Bw4 returns A*24:02, "
            "B*27:05, B*57:01, etc.  Space-separated, comma-separated, or repeated."
        ),
    )
    p_obs.add_argument(
        "--exclude-class-label-suspect",
        action="store_true",
        help=(
            "Drop rows where the curated MHC class disagrees with peptide "
            "length severely enough to be flagged 'suspect' or 'implausible' "
            "by mhc_class_label_severity (#182, #201). The strict variant — "
            "drops bulged class-I peptides 15-17aa as well."
        ),
    )
    p_obs.add_argument(
        "--exclude-class-label-implausible",
        action="store_true",
        help=(
            "Drop only rows flagged 'implausible' by mhc_class_label_severity "
            "(class-I ≥18aa or ≤7aa, class-II ≤4 or ≥31aa). Keeps borderline "
            "(13-14aa class-I, 8-10aa class-II) and suspect (15-17aa class-I, "
            "5-7aa class-II) rows. Useful when bulged class-I peptides should "
            "be retained (#201)."
        ),
    )
    p_obs.add_argument(
        "--apm-only",
        action="store_true",
        help=(
            "Filter to peptide rows from samples where any APM gene was "
            "perturbed (#202). Joins on pmid + sample condition."
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
        action="extend",
        nargs="+",
        help=(
            "Filter to rows whose mhc_restriction matches (after allele "
            "normalization).  Space-separated, comma-separated, or repeated."
        ),
    )
    p_bind.add_argument(
        "--mhc-allele-in-set",
        action="extend",
        nargs="+",
        help=(
            "Filter to rows whose mhc_allele_set (issue #137 candidate-allele "
            "set) contains any of the listed alleles.  Repeatable."
        ),
    )
    p_bind.add_argument(
        "--mhc-allele-provenance",
        action="extend",
        nargs="+",
        choices=["exact", "sample_allele_match", "pmid_class_pool", "unmatched"],
        help="Filter by how the allele set was obtained (issue #137).",
    )
    p_bind.add_argument(
        "--gene",
        action="extend",
        nargs="+",
        help=(
            "Filter by gene (HGNC symbol, Ensembl ID, or old alias).  "
            "Space-separated, comma-separated, or repeated.  Requires a mappings-built index."
        ),
    )
    p_bind.add_argument(
        "--gene-name", action="extend", nargs="+", help="Exact match on gene_name column."
    )
    p_bind.add_argument(
        "--gene-id", action="extend", nargs="+", help="Exact match on gene_id column."
    )
    p_bind.add_argument(
        "--serotype",
        action="extend",
        nargs="+",
        help=(
            "Filter by HLA serotype (locus-specific A24/B57/DR15 or public "
            "epitopes Bw4/Bw6/C1/C2).  Space-separated, comma-separated, or repeated."
        ),
    )
    p_bind.add_argument(
        "--assay-method",
        action="extend",
        nargs="+",
        help=(
            "Filter to rows whose IEDB/CEDAR assay_method matches (case-"
            "insensitive substring).  Examples: 'purified MHC/direct/"
            "fluorescence', 'cellular MHC/direct'.  Repeatable."
        ),
    )
    p_bind.add_argument(
        "--response-measured",
        action="extend",
        nargs="+",
        help=(
            "Filter to rows whose IEDB/CEDAR Response-measured matches "
            "(case-insensitive exact match).  Examples: 'qualitative "
            "binding', 'dissociation constant KD', 'half life', 'ligand "
            "presentation'.  Combine with --assay-method and "
            "--measurement-units to disambiguate IC50 vs Kd vs t_half. "
            "Repeatable."
        ),
    )
    p_bind.add_argument(
        "--measurement-units",
        action="extend",
        nargs="+",
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
        action="extend",
        nargs="+",
        help="Exact allele filter. Space-separated, comma-separated, or repeated.",
    )
    p_training.add_argument(
        "--mhc-allele-in-set",
        action="extend",
        nargs="+",
        help=(
            "Filter to rows whose mhc_allele_set (issue #137 expanded "
            "candidate-allele set) contains any of the listed alleles. "
            "Captures both 4-digit rows AND multi-allele class-only rows whose "
            "donor genotype includes the allele.  Repeatable."
        ),
    )
    p_training.add_argument(
        "--mhc-allele-provenance",
        action="extend",
        nargs="+",
        choices=["exact", "sample_allele_match", "pmid_class_pool", "unmatched"],
        help=(
            "Filter by how the allele set was obtained.  Use 'exact' for "
            "strict-resolution training; ['exact', 'sample_allele_match'] for "
            "MIL / noisy-OR over small trusted sets."
        ),
    )
    p_training.add_argument(
        "--gene",
        action="extend",
        nargs="+",
        help=(
            "Filter by gene (HGNC symbol, Ensembl ID, or old alias). "
            "Space-separated, comma-separated, or repeated. Requires a mappings-built index."
        ),
    )
    p_training.add_argument(
        "--gene-name", action="extend", nargs="+", help="Exact match on gene_name."
    )
    p_training.add_argument("--gene-id", action="extend", nargs="+", help="Exact match on gene_id.")
    p_training.add_argument(
        "--peptide",
        action="extend",
        nargs="+",
        help="Filter to one or more peptide sequences. Space-separated, comma-separated, or repeated.",
    )
    p_training.add_argument(
        "--serotype",
        action="extend",
        nargs="+",
        help="Filter by HLA serotype. Space-separated, comma-separated, or repeated.",
    )
    p_training.add_argument("--length-min", type=int, help="Minimum peptide length (inclusive).")
    p_training.add_argument("--length-max", type=int, help="Maximum peptide length (inclusive).")
    p_training.add_argument(
        "--map-source-proteins",
        "--explode-mappings",
        dest="map_source_proteins",
        action="store_true",
        help=(
            "Expand to one row per (evidence row, source-protein mapping), "
            "adding protein_id / gene_name / gene_id / transcript_id / "
            "position / n_flank / c_flank from peptide_mappings.parquet. "
            "Suitable for flank-aware training pipelines such as Presto. "
            "(--explode-mappings is a deprecated alias.)"
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

    p_bulk_prot = export_sub.add_parser(
        "bulk-proteomics",
        help=(
            "Bulk (non-MHC) proteomics index: shotgun MS peptides + protein "
            "abundances from CCLE and Bekker-Jensen 2017.  NOT MHC-ligand "
            "data — see 'observations' / 'binding' for those."
        ),
    )
    _add_export_bulk_proteomics_args(p_bulk_prot)

    p_bulk = export_sub.add_parser(
        "bulk", help="DEPRECATED — use `hitlist export bulk-proteomics`."
    )
    _add_export_bulk_proteomics_args(p_bulk)

    p_line_expr = export_sub.add_parser(
        "line-expression",
        help="DEPRECATED — use `hitlist expression`. Per-line RNA/transcript TPM.",
    )
    _add_line_expression_args(p_line_expr)

    p_counts = export_sub.add_parser(
        "counts",
        help="DEPRECATED — use `hitlist export peptide-counts --by study`.",
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

    # ── qc subcommand ──────────────────────────────────────────────────
    p_qc = sub.add_parser(
        "qc",
        help=(
            "Audit corpus + curation: allele-resolution histogram, "
            "normalization drift in pmid_overrides, YAML/data allele "
            "cross-reference."
        ),
    )
    p_qc.set_defaults(_subgroup_parser=p_qc)
    qc_sub = p_qc.add_subparsers(dest="qc_command")

    p_qc_res = qc_sub.add_parser(
        "resolution",
        help="Histogram of (mhc_class, source, allele_resolution) buckets",
    )
    p_qc_res.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_qc_res.add_argument("--species", help="Filter by MHC species")
    p_qc_res.add_argument(
        "--source", choices=["iedb", "cedar", "supplement"], help="Filter by data source"
    )
    p_qc_res.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_norm = qc_sub.add_parser(
        "normalization",
        help="Alleles in pmid_overrides whose normalize_allele() output differs from input",
    )
    p_qc_norm.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_xref = qc_sub.add_parser(
        "cross-reference",
        help=(
            "Alleles listed in YAML samples but absent from observation rows "
            "(yaml_only), or vice versa (data_only)"
        ),
    )
    p_qc_xref.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_qc_xref.add_argument(
        "--direction",
        choices=["yaml_only", "data_only", "both"],
        default="both",
        help="Which divergence to report (default: both)",
    )
    p_qc_xref.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_disc = qc_sub.add_parser(
        "discrepancies",
        help=(
            "Per-PMID rate of biologically suspicious patterns "
            "(class-label/length mismatches, mono-allelic class-only "
            "rows, class-pool fallback, non-standard amino acids). "
            "Triage list for curation work — sorted by total suspect "
            "count, descending."
        ),
    )
    p_qc_disc.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_qc_disc.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help=(
            "Drop PMID buckets with fewer than this many rows; small "
            "buckets give noisy length percentiles. Default 50."
        ),
    )
    p_qc_disc.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top N buckets by suspect score (default: all).",
    )
    p_qc_disc.add_argument(
        "--by",
        choices=["pmid", "sample"],
        default="pmid",
        help=(
            "Aggregation level: 'pmid' (default) groups per study/class; "
            "'sample' groups per (pmid, mhc_class, cell_name) so a curator "
            "can spot per-sample issues like 'this transfectant has 30%% "
            "suspect rows but its sibling has 0%%'."
        ),
    )
    p_qc_disc.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_plan = qc_sub.add_parser(
        "plan",
        help=(
            "Curation roadmap: per-PMID priority queue combining "
            "discrepancies, cross_reference, and normalization_drift "
            "into one ranked table — answers 'which study should I "
            "curate next?' without bouncing between three reports."
        ),
    )
    p_qc_plan.add_argument("--class", dest="mhc_class", help="MHC class (I or II)")
    p_qc_plan.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help="Drop PMID buckets with fewer than this many rows (default 50).",
    )
    p_qc_plan.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top N PMIDs by priority score (default: all).",
    )
    p_qc_plan.add_argument(
        "--severity",
        choices=["info", "warn"],
        help="Filter to one severity level (default: all).",
    )
    p_qc_plan.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_pc = qc_sub.add_parser(
        "proteome-coverage",
        help=(
            "Per-source-organism proteome registry coverage. Surfaces "
            "the species gap that blocks downstream features needing a "
            "per-organism proteome (flank extraction, gene-name lookup, "
            "source-protein audits) without fetching anything (#39)."
        ),
    )
    p_qc_pc.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Drop organism buckets with fewer than this many rows (default 1).",
    )
    p_qc_pc.add_argument(
        "--missing-only",
        action="store_true",
        help="Show only organisms WITHOUT a registered proteome.",
    )
    p_qc_pc.add_argument(
        "--severity",
        choices=["info", "warn"],
        help="Filter to one severity level (default: all).",
    )
    p_qc_pc.add_argument(
        "--use-uniprot",
        action="store_true",
        help=(
            "Fall back to the UniProt REST endpoint for organisms not "
            "in the curated registry. Slower (network) and offline runs "
            "should leave this off."
        ),
    )
    p_qc_pc.add_argument("--output", "-o", help="Write CSV to file")

    # ── pmhc subcommand ────────────────────────────────────────────────
    p_pmhc = sub.add_parser(
        "pmhc",
        help=(
            "Per-protein x allele pMHC evidence: MS-attested peptides grouped "
            "by (gene, allele), with optional NetMHCpan/MHCflurry scoring."
        ),
        description=(
            "Per-protein x allele pMHC evidence query.\n\n"
            "Both --protein and --mhc-allele are optional; pass neither to "
            "scan the whole corpus, just one to fix that axis. Pass "
            "--predictor {mhcflurry,netmhcpan} to add binding-affinity scoring."
        ),
    )
    p_pmhc.add_argument(
        "--protein",
        "--gene",
        action="extend",
        nargs="+",
        help=("One or more proteins: HGNC symbol, Ensembl ENSG, or alias. Omit to scan all genes."),
    )
    p_pmhc.add_argument(
        "--mhc-allele",
        action="extend",
        nargs="+",
        help=(
            "One or more 4-digit MHC alleles (HLA-A*02:01, HLA-B*07:02, ...). "
            "Omit to scan all alleles. Mutually exclusive with --sample / --samples — "
            "use those when each sample has its own allele set instead of a "
            "cross-product across all of them."
        ),
    )
    p_pmhc.add_argument(
        "--sample",
        action="append",
        metavar="NAME:ALLELE[,ALLELE,...]",
        help=(
            "Per-sample paired (name, allele set). Repeatable; "
            "alleles are comma-separated after the colon. "
            "Example: --sample patient1:HLA-A*02:01,HLA-A*24:02 "
            "--sample patient2:HLA-A*01:01,HLA-A*24:02. "
            "Each sample is queried independently against the protein "
            "list — results are sectioned by sample in the output."
        ),
    )
    p_pmhc.add_argument(
        "--samples",
        metavar="PATH.tsv",
        help=(
            "Batch form of --sample: path to a TSV with one row per sample, "
            "two columns 'name' and 'alleles' (alleles comma-separated). "
            "Header row optional. Mutually exclusive with --mhc-allele."
        ),
    )
    p_pmhc.add_argument(
        "--predictor",
        choices=["mhcflurry", "netmhcpan"],
        help=(
            "Optional binding-affinity predictor.  If omitted, only MS evidence "
            "is reported (no affinity column)."
        ),
    )
    p_pmhc.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help=(
            "Output shape (default: table — protein > allele section "
            "headers with peptide rows in an aligned table beneath each allele)."
        ),
    )
    p_pmhc.add_argument("--output", "-o", help="Write to file (.csv / .json / .txt)")

    # ── samples subcommand (promoted from `export samples`) ────────────
    p_samples = sub.add_parser(
        "samples",
        help=(
            "Per-sample MS conditions table (cell line, donor, instrument, "
            "perturbation, etc.).  Replaces `export samples`."
        ),
    )
    _add_samples_args(p_samples)

    # ── proteomics (promoted from `export bulk-proteomics`) ────────────
    p_prot = sub.add_parser(
        "proteomics",
        help=(
            "Bulk (non-MHC) proteomics index: shotgun MS peptides + protein "
            "abundances from CCLE and Bekker-Jensen 2017.  Replaces "
            "`export bulk-proteomics`."
        ),
    )
    _add_export_bulk_proteomics_args(p_prot)

    # ── expression (promoted from `export line-expression`) ────────────
    p_expr = sub.add_parser(
        "expression",
        help=("Per-line RNA / transcript TPM with provenance.  Replaces `export line-expression`."),
    )
    _add_line_expression_args(p_expr)

    args = parser.parse_args()
    if args.command is None:
        _print_subgroup_help(parser)
        sys.exit(1)
    if args.command == "data":
        _handle_data(args)
    elif args.command == "build":
        _handle_build(args)
    elif args.command == "report":
        _report(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "reassign-alleles":
        _reassign(args)
    elif args.command == "qc":
        _qc(args)
    elif args.command == "pmhc":
        _pmhc(args)
    elif args.command == "samples":
        _handle_samples(args)
    elif args.command == "proteomics":
        # Reuse the same handler the legacy export bulk-proteomics path uses.
        args.export_command = "bulk-proteomics"
        _export(args)
    elif args.command == "expression":
        args.export_command = "line-expression"
        _export(args)


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


def _qc(args: argparse.Namespace) -> None:
    """Run a qc subcommand and emit CSV (or write to --output)."""
    from . import qc

    cmd = getattr(args, "qc_command", None)
    output = getattr(args, "output", None)

    if cmd is None:
        # Bare ``hitlist qc`` runs all three checks and prints a summary.
        results = qc.run_all(mhc_class=getattr(args, "mhc_class", None))
        for name, df in results.items():
            print(f"# {name}: {len(df):,} rows", file=sys.stderr)
            if not df.empty:
                print(df.to_csv(index=False), end="")
                print()
        return

    if cmd == "resolution":
        df = qc.resolution_histogram(
            mhc_class=getattr(args, "mhc_class", None),
            species=getattr(args, "species", None),
            source=getattr(args, "source", None),
        )
    elif cmd == "normalization":
        df = qc.normalization_drift()
    elif cmd == "cross-reference":
        df = qc.cross_reference(mhc_class=getattr(args, "mhc_class", None))
        direction = getattr(args, "direction", "both")
        if direction != "both" and not df.empty:
            df = df[df["direction"] == direction].reset_index(drop=True)
    elif cmd == "discrepancies":
        df = qc.discrepancies(
            mhc_class=getattr(args, "mhc_class", None),
            min_rows=getattr(args, "min_rows", 50),
            by=getattr(args, "by", "pmid"),
        )
        top = getattr(args, "top", None)
        if top is not None and not df.empty:
            df = df.head(top).reset_index(drop=True)
    elif cmd == "plan":
        df = qc.curation_plan(
            mhc_class=getattr(args, "mhc_class", None),
            min_rows=getattr(args, "min_rows", 50),
        )
        sev = getattr(args, "severity", None)
        if sev is not None and not df.empty:
            df = df[df["severity"] == sev].reset_index(drop=True)
        top = getattr(args, "top", None)
        if top is not None and not df.empty:
            df = df.head(top).reset_index(drop=True)
    elif cmd == "proteome-coverage":
        df = qc.proteome_coverage(
            min_rows=getattr(args, "min_rows", 1),
            use_uniprot=getattr(args, "use_uniprot", False),
        )
        sev = getattr(args, "severity", None)
        if sev is not None and not df.empty:
            df = df[df["severity"] == sev].reset_index(drop=True)
        missing_only = getattr(args, "missing_only", False)
        if missing_only and not df.empty:
            df = df[~df["has_proteome"]].reset_index(drop=True)
    else:
        print(f"Unknown qc subcommand: {cmd}", file=sys.stderr)
        sys.exit(1)

    if output:
        df.to_csv(output, index=False)
        print(f"Wrote {len(df):,} rows to {output}", file=sys.stderr)
    else:
        print(df.to_csv(index=False), end="")


def _parse_pmhc_samples(inline_specs: list[str], tsv_path: str | None) -> dict[str, list[str]]:
    """Parse ``--sample`` / ``--samples`` into ``{name: [allele, ...]}``.

    Inline form: ``"patient1:HLA-A*02:01,HLA-A*24:02"`` — colon splits the
    name from the allele list, commas split alleles. Whitespace around the
    name and alleles is stripped.

    TSV form: two columns, ``name`` and ``alleles`` (alleles comma-separated
    in the second column). Header row optional — if the first row's first
    cell is exactly ``name`` we skip it. Lines starting with ``#`` are
    ignored as comments.

    Raises ``ValueError`` on malformed input — the CLI catches and prints.
    """
    out: dict[str, list[str]] = {}

    for spec in inline_specs:
        if ":" not in spec:
            raise ValueError(
                f"--sample {spec!r} is missing the ':' separator; expected NAME:ALLELE[,ALLELE,...]"
            )
        name, _, allele_csv = spec.partition(":")
        name = name.strip()
        if not name:
            raise ValueError(f"--sample {spec!r} has an empty name")
        alleles = [a.strip() for a in allele_csv.split(",") if a.strip()]
        if not alleles:
            raise ValueError(f"--sample {name!r} has no alleles after the ':'")
        if name in out:
            raise ValueError(f"--sample {name!r} given more than once")
        out[name] = alleles

    if tsv_path:
        from pathlib import Path

        path = Path(tsv_path)
        if not path.exists():
            raise ValueError(f"--samples file not found: {tsv_path}")
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"{tsv_path}:{lineno} expected two tab-separated columns "
                    f"(name, alleles); got {len(parts)} column(s)"
                )
            name = parts[0].strip()
            allele_field = parts[1].strip()
            # Skip the optional header row.
            if lineno == 1 and name.lower() == "name":
                continue
            if not name:
                raise ValueError(f"{tsv_path}:{lineno} has an empty name")
            alleles = [a.strip() for a in allele_field.split(",") if a.strip()]
            if not alleles:
                raise ValueError(f"{tsv_path}:{lineno} sample {name!r} has no alleles")
            if name in out:
                raise ValueError(
                    f"sample {name!r} given more than once (--sample inline + {tsv_path}:{lineno})"
                )
            out[name] = alleles

    if not out:
        raise ValueError("no samples parsed from --sample / --samples input")
    return out


def _pmhc(args: argparse.Namespace) -> None:
    """Run the pmhc evidence query.

    ``--protein`` / ``--mhc-allele`` / ``--sample`` / ``--samples`` are
    all optional. ``--sample``/``--samples`` (paired sample-to-allele
    input) are mutually exclusive with ``--mhc-allele`` (cross-product
    input).
    """
    from . import pmhc_query

    proteins = getattr(args, "protein", []) or []
    alleles = getattr(args, "mhc_allele", []) or []
    inline_samples = getattr(args, "sample", None) or []
    samples_path = getattr(args, "samples", None)
    predictor = getattr(args, "predictor", None)
    fmt = getattr(args, "format", "table")
    output = getattr(args, "output", None)

    has_sample_input = bool(inline_samples) or bool(samples_path)
    if has_sample_input and alleles:
        print(
            "Error: --mhc-allele cannot be combined with --sample / --samples; "
            "drop one. Use --mhc-allele for a cross-product (every protein x "
            "every allele); use --sample/--samples when each sample has its "
            "own allele set.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        if has_sample_input:
            samples_to_alleles = _parse_pmhc_samples(inline_samples, samples_path)
            df = pmhc_query.query_by_samples(
                samples_to_alleles=samples_to_alleles,
                proteins=proteins,
                predictor=predictor,
                verbose=True,
            )
        else:
            df = pmhc_query.query(
                proteins=proteins,
                alleles=alleles,
                predictor=predictor,
                verbose=True,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if fmt == "csv":
        text = df.to_csv(index=False)
    elif fmt == "json":
        text = df.to_json(orient="records", indent=2)
    else:  # table
        text = pmhc_query.format_table(df)

    if output:
        from pathlib import Path

        Path(output).write_text(text)
        print(f"Wrote {output}", file=sys.stderr)
    else:
        print(text)


def _handle_samples(args: argparse.Namespace) -> None:
    """Run ``hitlist samples`` — same backing logic as the legacy
    ``hitlist export samples``."""
    args.export_command = "samples"
    _export(args)


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
        mhc_allele_in_set=getattr(args, "mhc_allele_in_set", None),
        mhc_allele_provenance=getattr(args, "mhc_allele_provenance", None),
        gene=getattr(args, "gene", None),
        gene_name=getattr(args, "gene_name", None),
        gene_id=getattr(args, "gene_id", None),
        peptide=getattr(args, "peptide", None),
        serotype=getattr(args, "serotype", None),
        length_min=getattr(args, "length_min", None),
        length_max=getattr(args, "length_max", None),
        map_source_proteins=getattr(args, "map_source_proteins", False),
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

    # Deprecation notices for legacy export subcommands.  Behavior is
    # preserved for one release; these notices steer users at the
    # canonical replacement before removal in v2.0.
    _deprecated_export = {
        "summary": "hitlist export peptide-counts --by class",
        "counts": "hitlist export peptide-counts --by study",
        "alleles": "hitlist qc normalization",
        "data-alleles": "hitlist qc resolution",
        "bulk": "hitlist export bulk-proteomics",
        "samples": "hitlist samples",
        "bulk-proteomics": "hitlist proteomics",
        "line-expression": "hitlist expression",
    }
    if cmd in _deprecated_export:
        print(
            f"Note: `hitlist export {cmd}` is deprecated; use "
            f"`{_deprecated_export[cmd]}` instead.  Will be removed in v2.0.",
            file=sys.stderr,
        )

    if cmd == "samples":
        if getattr(args, "with_expression_anchors", False):
            _export_progress("Resolving expression anchors for every ms_sample (issue #140) ...")
            from .export import generate_sample_expression_table

            df = generate_sample_expression_table(mhc_class=args.mhc_class)
        else:
            _export_progress("Building ms_samples table from pmid_overrides.yaml ...")
            df = generate_ms_samples_table(
                mhc_class=args.mhc_class,
                apm_only=getattr(args, "apm_only", False),
            )
    elif cmd == "peptide-counts":
        by = getattr(args, "by", "class")
        if by == "class":
            _export_progress(
                "Loading observations.parquet for per-(class, species) peptide counts ..."
            )
            df = generate_species_summary(mhc_class=args.mhc_class)
        else:
            _export_progress("Counting peptides per study from local IEDB/CEDAR ...")
            df = count_peptides_by_study(source=args.source)
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
                mhc_allele_in_set=getattr(args, "mhc_allele_in_set", None),
                mhc_allele_provenance=getattr(args, "mhc_allele_provenance", None),
                gene=getattr(args, "gene", None),
                gene_name=getattr(args, "gene_name", None),
                gene_id=getattr(args, "gene_id", None),
                serotype=getattr(args, "serotype", None),
                exclude_class_label_suspect=getattr(args, "exclude_class_label_suspect", False),
                exclude_class_label_implausible=getattr(
                    args, "exclude_class_label_implausible", False
                ),
                apm_only=getattr(args, "apm_only", False),
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
                mhc_allele_in_set=getattr(args, "mhc_allele_in_set", None),
                mhc_allele_provenance=getattr(args, "mhc_allele_provenance", None),
                gene=getattr(args, "gene", None),
                gene_name=getattr(args, "gene_name", None),
                gene_id=getattr(args, "gene_id", None),
                serotype=getattr(args, "serotype", None),
                assay_method=getattr(args, "assay_method", None),
                response_measured=getattr(args, "response_measured", None),
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
    elif cmd in ("bulk", "bulk-proteomics"):
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
        _print_subgroup_help(args._subgroup_parser)
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
