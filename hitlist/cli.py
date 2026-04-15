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
    p_bind.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

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


def _export(args: argparse.Namespace) -> None:
    from .export import (
        collect_alleles_from_data,
        count_peptides_by_study,
        generate_binding_table,
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
        try:
            df = generate_observations_table(
                mhc_class=args.mhc_class,
                species=getattr(args, "species", None),
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
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.export_command == "binding":
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
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Usage: hitlist export "
            "{samples,summary,alleles,data-alleles,counts,observations,binding}"
        )
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
