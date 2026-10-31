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

"""``hitlist export`` and the shared export implementation."""

from __future__ import annotations

import argparse
import sys

from .cli_common import print_subgroup_help
from .curation import (
    ALLELE_RESOLUTION_FLOORS,
    MHC_ALLELE_PROVENANCE_VALUES,
    RESTRICTION_EVIDENCE_VALUES,
    SEROTYPE_SOURCE_VALUES,
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
        help="Filter by reviewed fractionation pH; unknown values are excluded.",
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
        help="Filter to MHC class (I, II, or non-classical). Applies to --by class.",
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
    p.add_argument(
        "--class", dest="mhc_class", help="Filter to MHC class (I, II, or non-classical)"
    )
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


def _add_binding_measurement_args(parser: argparse.ArgumentParser) -> None:
    """Register binding-only filters shared by binding and training exports."""
    parser.add_argument(
        "--assay-method",
        action="extend",
        nargs="+",
        help=(
            "Filter binding rows whose IEDB/CEDAR assay_method matches (case-"
            "insensitive substring).  Examples: 'purified MHC/direct/"
            "fluorescence', 'cellular MHC/direct'.  Repeatable."
        ),
    )
    parser.add_argument(
        "--response-measured",
        action="extend",
        nargs="+",
        help=(
            "Filter binding rows whose IEDB/CEDAR Response-measured matches "
            "(case-insensitive exact match).  Examples: 'qualitative "
            "binding', 'dissociation constant KD', 'half life', 'ligand "
            "presentation'.  Combine with --assay-method and "
            "--measurement-units to disambiguate IC50 vs Kd vs t_half. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--measurement-units",
        action="extend",
        nargs="+",
        help=(
            "Filter binding rows reporting in these units (case-insensitive "
            "exact match).  Pair with --quantitative-value-{min,max} to "
            "avoid mixing unit systems (e.g. nM vs log10(nM))."
        ),
    )
    parser.add_argument(
        "--quantitative-value-min",
        type=float,
        help=(
            "Binding-only inclusive lower bound on the reported value; "
            "excludes NaN, preserves inequalities."
        ),
    )
    parser.add_argument(
        "--quantitative-value-max",
        type=float,
        help=(
            "Binding-only inclusive upper bound on the reported value; "
            "excludes NaN, preserves inequalities."
        ),
    )
    parser.add_argument(
        "--has-quantitative-value",
        dest="has_quantitative_value",
        action="store_true",
        default=None,
        help="Keep binding rows with a numeric value, regardless of endpoint or units.",
    )
    parser.add_argument(
        "--qualitative-only",
        dest="has_quantitative_value",
        action="store_false",
        help="Keep binding rows with no numeric value reported.",
    )


def _add_export_species_axis_args(parser: argparse.ArgumentParser) -> None:
    """Keep independent species-axis filters available on every evidence export."""
    parser.add_argument(
        "--source-species",
        nargs="+",
        help="Filter by peptide-source proteome species (names or aliases)",
    )
    parser.add_argument(
        "--host-species", nargs="+", help="Filter by host-cell species (names or aliases)"
    )
    parser.add_argument(
        "--exclude-chimeric", action="store_true", help="Exclude observations flagged as chimeric"
    )


def _export_training(args: argparse.Namespace):
    """Reproduce the historical MHC training export for source audits."""
    from .export import generate_training_table

    options = {
        "include_evidence": getattr(args, "include_evidence", "both"),
        "mhc_class": getattr(args, "mhc_class", None),
        "species": getattr(args, "species", None),
        "source_species": getattr(args, "source_species", None),
        "host_species": getattr(args, "host_species", None),
        "exclude_chimeric": getattr(args, "exclude_chimeric", False),
        "source": getattr(args, "source", None),
        "instrument_type": getattr(args, "instrument_type", None),
        "acquisition_mode": getattr(args, "acquisition_mode", None),
        "is_mono_allelic": getattr(args, "mono_allelic", None),
        "min_allele_resolution": getattr(args, "min_allele_resolution", None),
        "mhc_allele": getattr(args, "mhc_allele", None),
        "mhc_allele_in_set": getattr(args, "mhc_allele_in_set", None),
        "mhc_allele_provenance": getattr(args, "mhc_allele_provenance", None),
        "restriction_evidence": getattr(args, "restriction_evidence", None),
        "serotype_source": getattr(args, "serotype_source", None),
        "gene": getattr(args, "gene", None),
        "gene_name": getattr(args, "gene_name", None),
        "gene_id": getattr(args, "gene_id", None),
        "peptide": getattr(args, "peptide", None),
        "serotype": getattr(args, "serotype", None),
        "length_min": getattr(args, "length_min", None),
        "length_max": getattr(args, "length_max", None),
        "assay_method": getattr(args, "assay_method", None),
        "response_measured": getattr(args, "response_measured", None),
        "measurement_units": getattr(args, "measurement_units", None),
        "has_quantitative_value": getattr(args, "has_quantitative_value", None),
        "quantitative_value_min": getattr(args, "quantitative_value_min", None),
        "quantitative_value_max": getattr(args, "quantitative_value_max", None),
        "map_source_proteins": getattr(args, "map_source_proteins", False),
        "with_peptide_origin": getattr(args, "with_peptide_origin", False),
        "proteome_release": getattr(args, "proteome_release", 112),
        "columns": getattr(args, "columns", None),
    }
    if getattr(args, "bundle", None):
        from .training_bundle import write_training_bundle

        return write_training_bundle(
            args.bundle,
            split_policy=args.split_policy,
            require_current_indexes=not getattr(args, "allow_stale_indexes", False),
            **options,
        )
    if getattr(args, "allow_stale_indexes", False):
        raise ValueError("--allow-stale-indexes requires --bundle")
    if getattr(args, "split_policy", "report_only") != "report_only":
        raise ValueError("--split-policy requires --bundle; use audit-splits to check partitions")
    return generate_training_table(**options)


def _export_progress(msg: str) -> None:
    """Print a single-line progress message to stderr (not stdout).

    stderr keeps the CSV output pipe-clean. Only emits when stderr is a
    TTY (interactive terminal) so scripts piping both stdout and stderr
    don't get spammed. See pirl-unc/hitlist#119.
    """
    if sys.stderr.isatty():
        print(f"[hitlist] {msg}", file=sys.stderr, flush=True)


def _export_ms(args: argparse.Namespace):
    """Build the MS observations table from parsed ``export ms`` args."""
    from .export import generate_observations_table

    return generate_observations_table(
        mhc_class=args.mhc_class,
        species=getattr(args, "species", None),
        source_species=getattr(args, "source_species", None),
        host_species=getattr(args, "host_species", None),
        exclude_chimeric=getattr(args, "exclude_chimeric", False),
        source=getattr(args, "source", None),
        instrument_type=getattr(args, "instrument_type", None),
        acquisition_mode=getattr(args, "acquisition_mode", None),
        is_mono_allelic=getattr(args, "mono_allelic", None),
        min_allele_resolution=getattr(args, "min_allele_resolution", None),
        mhc_allele=getattr(args, "mhc_allele", None),
        mhc_allele_in_set=getattr(args, "mhc_allele_in_set", None),
        mhc_allele_provenance=getattr(args, "mhc_allele_provenance", None),
        restriction_evidence=getattr(args, "restriction_evidence", None),
        serotype_source=getattr(args, "serotype_source", None),
        gene=getattr(args, "gene", None),
        gene_name=getattr(args, "gene_name", None),
        gene_id=getattr(args, "gene_id", None),
        peptide=getattr(args, "peptide", None),
        serotype=getattr(args, "serotype", None),
        exclude_class_label_suspect=getattr(args, "exclude_class_label_suspect", False),
        exclude_class_label_implausible=getattr(args, "exclude_class_label_implausible", False),
        apm_only=getattr(args, "apm_only", False),
    )


def _export_peptide_summary(args: argparse.Namespace):
    """Build the per-peptide MS support summary from parsed args."""
    from .export import generate_ms_peptide_summary_table

    return generate_ms_peptide_summary_table(
        mhc_class=args.mhc_class,
        species=getattr(args, "species", None),
        source_species=getattr(args, "source_species", None),
        host_species=getattr(args, "host_species", None),
        exclude_chimeric=getattr(args, "exclude_chimeric", False),
        source=getattr(args, "source", None),
        mhc_allele=getattr(args, "mhc_allele", None),
        serotype=getattr(args, "serotype", None),
        gene=getattr(args, "gene", None),
        gene_name=getattr(args, "gene_name", None),
        gene_id=getattr(args, "gene_id", None),
        peptide=getattr(args, "peptide", None),
    )


def _export(args: argparse.Namespace) -> None:
    """Dispatch ``hitlist export <subcommand>``."""
    run_export(args, args.export_command)


def run_export(
    args: argparse.Namespace, cmd: str | None, *, via_legacy_export: bool = True
) -> None:
    """Run export subcommand *cmd* against parsed *args* and emit the table.

    Public so the promoted ``hitlist samples`` / ``proteomics`` / ``expression``
    commands (:mod:`hitlist.cli_samples`) share the implementation of the legacy
    ``hitlist export`` subcommands without rewriting ``args.export_command``.
    ``via_legacy_export`` says the user typed ``hitlist export <cmd>``; only then is the
    deprecation note printed, so the promoted commands do not point users at themselves (#893).
    """
    import time as _time

    from .export import (
        collect_alleles_from_data,
        count_peptides_by_study,
        generate_binding_table,
        generate_ms_samples_table,
        generate_species_summary,
        validate_mhc_alleles,
    )

    if cmd in {"cta-evidence", "tissue-blacklist"}:
        from .cli_evidence import handle_evidence

        handle_evidence(args)
        return
    t0 = _time.perf_counter()

    # Deprecation notices for legacy export subcommands.  Behavior is
    # preserved for one release; these notices steer users at the
    # canonical replacement for historical 1.x export commands.
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
    if via_legacy_export and cmd in _deprecated_export:
        print(
            f"Note: `hitlist export {cmd}` is deprecated; use `{_deprecated_export[cmd]}` instead.",
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
    elif cmd in {"ms", "observations"}:
        _export_progress("Loading observations.parquet + joining sample metadata ...")
        try:
            df = _export_ms(args)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "peptide-summary":
        _export_progress("Loading MS observations and grouping target-support per peptide ...")
        try:
            df = _export_peptide_summary(args)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "binding":
        _export_progress("Loading binding.parquet ...")
        try:
            df = generate_binding_table(
                mhc_class=args.mhc_class,
                species=getattr(args, "species", None),
                source_species=getattr(args, "source_species", None),
                host_species=getattr(args, "host_species", None),
                exclude_chimeric=getattr(args, "exclude_chimeric", False),
                source=getattr(args, "source", None),
                min_allele_resolution=getattr(args, "min_allele_resolution", None),
                mhc_allele=getattr(args, "mhc_allele", None),
                mhc_allele_in_set=getattr(args, "mhc_allele_in_set", None),
                mhc_allele_provenance=getattr(args, "mhc_allele_provenance", None),
                restriction_evidence=getattr(args, "restriction_evidence", None),
                serotype_source=getattr(args, "serotype_source", None),
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
        print_subgroup_help(args._subgroup_parser)
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


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist export`` (plus the sibling commands whose help order depends on it)."""
    p_export = sub.add_parser("export", help="Export curated study metadata as CSV")
    p_export.set_defaults(_subgroup_parser=p_export, func=_export)
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
    p_summary.add_argument(
        "--class", dest="mhc_class", help="Filter to MHC class (I, II, or non-classical)"
    )
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
        "ms",
        aliases=["observations"],
        help="MS observations table: peptides + sample metadata",
    )
    p_obs.add_argument("--class", dest="mhc_class", help="MHC class (I, II, or non-classical)")
    p_obs.add_argument("--species", help="Filter by MHC species")
    _add_export_species_axis_args(p_obs)
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
        choices=ALLELE_RESOLUTION_FLOORS,
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
        choices=MHC_ALLELE_PROVENANCE_VALUES,
        help=(
            "Filter by how the allele set was obtained.  Use 'exact' for "
            "strict-resolution training; ['exact', 'sample_allele_match'] for "
            "MIL / noisy-OR over small trusted sets."
        ),
    )
    p_obs.add_argument(
        "--restriction-evidence",
        action="extend",
        nargs="+",
        choices=RESTRICTION_EVIDENCE_VALUES,
        help="Filter by how the named peptide-to-MHC restriction was established.",
    )
    p_obs.add_argument(
        "--serotype-source",
        action="extend",
        nargs="+",
        choices=SEROTYPE_SOURCE_VALUES,
        help=(
            "Filter by whether the serotype is primary data (reported), a "
            "projection of a named molecule (derived), or a union over a "
            "donor's typed alleles (donor_set)."
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
            "Filter by MHC serotype. Use a species prefix for non-human names (BoLA-A18). "
            "Accepts locus-specific names (A2, A24, "
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
    p_obs.add_argument(
        "--peptide",
        action="extend",
        nargs="+",
        help=(
            "Filter to one or more peptide sequences. Space-separated, "
            "comma-separated, or repeated."
        ),
    )
    p_obs.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_pep_summary = export_sub.add_parser(
        "peptide-summary",
        help="Per-peptide MS support summary for one target allele or serotype",
    )
    p_pep_summary.add_argument(
        "--class", dest="mhc_class", help="MHC class (I, II, or non-classical)"
    )
    p_pep_summary.add_argument("--species", help="Filter by MHC species")
    _add_export_species_axis_args(p_pep_summary)
    p_pep_summary.add_argument(
        "--source",
        choices=["iedb", "cedar", "supplement"],
        help="Filter by data source",
    )
    p_pep_summary.add_argument(
        "--mhc-allele",
        action="extend",
        nargs="+",
        help="Exactly one target allele, e.g. HLA-A*24:02.",
    )
    p_pep_summary.add_argument(
        "--serotype",
        action="extend",
        nargs="+",
        help="Exactly one target serotype, e.g. A24 or Bw4.",
    )
    p_pep_summary.add_argument(
        "--gene",
        action="extend",
        nargs="+",
        help=(
            "Filter by gene (HGNC symbol, Ensembl ID, or old alias). "
            "Space-separated, comma-separated, or repeated."
        ),
    )
    p_pep_summary.add_argument(
        "--gene-name", action="extend", nargs="+", help="Exact match on gene_name."
    )
    p_pep_summary.add_argument(
        "--gene-id", action="extend", nargs="+", help="Exact match on gene_id."
    )
    p_pep_summary.add_argument(
        "--peptide",
        action="extend",
        nargs="+",
        help="Restrict to one or more peptides. Space-separated, comma-separated, or repeated.",
    )
    p_pep_summary.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    p_bind = export_sub.add_parser(
        "binding",
        help=(
            "Binding-assay index (peptide microarray, refolding, MEDi, "
            "quantitative tiers).  Separate from MS observations."
        ),
    )
    p_bind.add_argument("--class", dest="mhc_class", help="MHC class (I, II, or non-classical)")
    p_bind.add_argument("--species", help="Filter by MHC species")
    _add_export_species_axis_args(p_bind)
    p_bind.add_argument(
        "--source",
        choices=["iedb", "cedar"],
        help="Filter by data source (supplementary data is MS-only and never appears here)",
    )
    p_bind.add_argument(
        "--min-allele-resolution",
        choices=ALLELE_RESOLUTION_FLOORS,
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
        choices=MHC_ALLELE_PROVENANCE_VALUES,
        help="Filter by how the allele set was obtained (issue #137).",
    )
    p_bind.add_argument(
        "--restriction-evidence",
        action="extend",
        nargs="+",
        choices=RESTRICTION_EVIDENCE_VALUES,
        help="Filter by how the named peptide-to-MHC restriction was established.",
    )
    p_bind.add_argument(
        "--serotype-source",
        action="extend",
        nargs="+",
        choices=SEROTYPE_SOURCE_VALUES,
        help=(
            "Filter by whether the serotype is primary data (reported), a "
            "projection of a named molecule (derived), or a union over a "
            "donor's typed alleles (donor_set)."
        ),
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
            "Filter by MHC serotype (species-prefixed BoLA-A18, locus-specific A24/B57/DR15 or public "
            "epitopes Bw4/Bw6/C1/C2).  Space-separated, comma-separated, or repeated."
        ),
    )
    _add_binding_measurement_args(p_bind)
    p_bind.add_argument("--output", "-o", help="Write to file (.csv or .parquet)")

    from . import cli_audit
    from .cli_evidence import add_evidence_parsers

    # audit-splits sits between export and verify-evidence-bundle in top-level help.
    cli_audit.register(sub)

    add_evidence_parsers(sub, export_sub)

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
