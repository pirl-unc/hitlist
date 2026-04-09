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
    )


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
        "--with-flanking",
        action="store_true",
        help="Map peptides to source proteins with 10aa flanking",
    )
    p.add_argument(
        "--proteome-release", type=int, default=112, help="Ensembl release (default 112)"
    )
    p.add_argument("--force", "-f", action="store_true", help="Rebuild even if cached")

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
    }
    if args.data_command is None:
        print(
            "Usage: hitlist data {list,available,register,fetch,refresh,info,path,remove,build,index}"
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
    p_obs.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

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


def _export(args: argparse.Namespace) -> None:
    from .export import (
        collect_alleles_from_data,
        count_peptides_by_study,
        generate_ms_samples_table,
        generate_observations_table,
        generate_species_summary,
        validate_mhc_alleles,
    )

    if args.export_command == "samples":
        df = generate_ms_samples_table(mhc_class=args.mhc_class)
    elif args.export_command == "summary":
        df = generate_species_summary(mhc_class=args.mhc_class)
    elif args.export_command == "alleles":
        df = validate_mhc_alleles()
    elif args.export_command == "data-alleles":
        df = collect_alleles_from_data(source=getattr(args, "source", "merged"))
    elif args.export_command == "counts":
        df = count_peptides_by_study(source=args.source)
    elif args.export_command == "observations":
        df = generate_observations_table(
            mhc_class=args.mhc_class,
            species=getattr(args, "species", None),
            instrument_type=getattr(args, "instrument_type", None),
            acquisition_mode=getattr(args, "acquisition_mode", None),
        )
    else:
        print("Usage: hitlist export {samples,summary,alleles,data-alleles,counts,observations}")
        sys.exit(1)

    if args.output:
        if args.output.endswith(".parquet"):
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
