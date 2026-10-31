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

"""``hitlist qc``: corpus and curation audits."""

from __future__ import annotations

import argparse
import sys


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
    elif cmd == "mhc-tokens":
        df = qc.mhc_token_audit()
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
    elif cmd == "sample-attribution":
        df = qc.sample_attribution_audit()
        bucket = getattr(args, "bucket", None)
        if bucket is not None and not df.empty:
            df = df[df["bucket"] == bucket].reset_index(drop=True)
        sev = getattr(args, "severity", None)
        if sev is not None and not df.empty:
            df = df[df["severity"] == sev].reset_index(drop=True)
        if getattr(args, "actionable_only", False) and not df.empty:
            settled = {"axis_mismatch", "no_row_discriminator", "multi_arm_evidence"}
            df = df[~df["arm_resolution"].isin(settled)].reset_index(drop=True)
    elif cmd == "engineering-drift":
        df = qc.engineering_drift_audit()
        sev = getattr(args, "severity", None)
        if sev is not None and not df.empty:
            df = df[df["severity"] == sev].reset_index(drop=True)
    else:
        print(f"Unknown qc subcommand: {cmd}", file=sys.stderr)
        sys.exit(1)

    if output:
        df.to_csv(output, index=False)
        print(f"Wrote {len(df):,} rows to {output}", file=sys.stderr)
    else:
        print(df.to_csv(index=False), end="")


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist qc``."""
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
    p_qc_res.add_argument("--class", dest="mhc_class", help="MHC class (I, II, or non-classical)")
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

    p_qc_tokens = qc_sub.add_parser(
        "mhc-tokens",
        help=(
            "Audit unparseable MHC restrictions, host typings, serotypes, "
            "and curated sample MHC across MS and binding modalities"
        ),
    )
    p_qc_tokens.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_xref = qc_sub.add_parser(
        "cross-reference",
        help=(
            "Alleles listed in YAML samples but absent from observation rows "
            "(yaml_only), or vice versa (data_only)"
        ),
    )
    p_qc_xref.add_argument("--class", dest="mhc_class", help="MHC class (I, II, or non-classical)")
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
    p_qc_disc.add_argument("--class", dest="mhc_class", help="MHC class (I, II, or non-classical)")
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
    p_qc_plan.add_argument("--class", dest="mhc_class", help="MHC class (I, II, or non-classical)")
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

    p_qc_sa = qc_sub.add_parser(
        "sample-attribution",
        help=(
            "Curated ms_samples arms that were profiled but reach zero "
            "observation rows. A curator adding an arm otherwise gets no "
            "signal about whether it landed (#442). Builds the full "
            "enriched table, so this is slower than the other checks."
        ),
    )
    p_qc_sa.add_argument(
        "--bucket",
        choices=["label_mismatch_candidate", "study_unattributed"],
        help=(
            "Filter to one finding shape. 'label_mismatch_candidate' means "
            "other arms of the same study do attribute, so the label is the "
            "suspect; 'study_unattributed' means no arm of the study "
            "attributes at all (default: both)."
        ),
    )
    p_qc_sa.add_argument(
        "--severity",
        choices=["info", "warn"],
        help="Filter to one severity level (default: all).",
    )
    p_qc_sa.add_argument(
        "--actionable-only",
        action="store_true",
        help=(
            "Hide findings in studies already recorded as unresolvable "
            "(arm_resolution axis_mismatch / no_row_discriminator / "
            "multi_arm_evidence), leaving only what more curation could fix."
        ),
    )
    p_qc_sa.add_argument("--output", "-o", help="Write CSV to file")

    p_qc_ed = qc_sub.add_parser(
        "engineering-drift",
        help=(
            "Curated arms silent on an engineering column that a same-material "
            "sibling asserts (#586). A blank reads as 'not established', so the "
            "arm stops counting as engineered material and takes its reference "
            "line's RNA at tier 1 while the sibling takes it at tier 2."
        ),
    )
    p_qc_ed.add_argument(
        "--severity",
        choices=["warn", "error"],
        help=(
            "Filter to one severity level. 'error' is an arm nothing else "
            "engineers, so an expression tier really differs (default: all)."
        ),
    )
    p_qc_ed.add_argument("--output", "-o", help="Write CSV to file")
    p_qc.set_defaults(func=_qc)
