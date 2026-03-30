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

"""Automated data quality reports for IEDB/CEDAR mass spec evidence.

Generates comprehensive summaries of MHC class distribution, source
classification, tissue coverage, disease breakdown, allele coverage,
peptide counts, and cell line annotations from registered datasets.

CLI::

    hitlist report                     # report on all registered MS data
    hitlist report --class I           # class I only
    hitlist report --output report.txt # save to file
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def generate_report(
    df: pd.DataFrame,
    mhc_class_filter: str | None = None,
    output: str | Path | None = None,
) -> str:
    """Generate a data quality report from a scanned DataFrame.

    Parameters
    ----------
    df
        DataFrame from :func:`hitlist.scanner.scan` with ``classify_source=True``.
    mhc_class_filter
        If set, report only on this MHC class (``"I"`` or ``"II"``).
    output
        If set, write report to this file path.

    Returns
    -------
    str
        The report text.
    """
    lines: list[str] = []

    def p(text: str = "") -> None:
        lines.append(text)

    def table(header: str, rows: list[tuple], col_widths: tuple = (35, 10, 8)) -> None:
        p(header)
        for row in rows:
            parts = []
            for i, val in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else 10
                if isinstance(val, int):
                    parts.append(f"{val:>{w},}")
                elif isinstance(val, float):
                    parts.append(f"{val:>{w}.1f}%")
                elif isinstance(val, str):
                    parts.append(f"{val:<{w}}")
                else:
                    parts.append(f"{val!s:<{w}}")
            p("  " + "  ".join(parts))
        p()

    total = len(df)
    if total == 0:
        p("No data to report.")
        text = "\n".join(lines)
        if output:
            Path(output).write_text(text)
        return text

    p("=" * 72)
    p("HITLIST DATA QUALITY REPORT")
    p("=" * 72)
    p()

    # ── MHC class distribution ──────────────────────────────────────────
    p("── MHC Class Distribution ──")
    p()
    class_counts = df["mhc_class"].value_counts()
    table(
        f"  {'Class':<20} {'Rows':>12} {'Pct':>8}",
        [(cls, count, count / total * 100) for cls, count in class_counts.items()],
        (20, 12, 8),
    )

    # Apply class filter if requested
    if mhc_class_filter:
        df = df[df["mhc_class"] == mhc_class_filter]
        total = len(df)
        p(f"  [Filtered to class {mhc_class_filter}: {total:,} rows]")
        p()

    # ── Source classification ───────────────────────────────────────────
    p("── Source Classification ──")
    p()
    src_flags = [
        ("src_cancer", "Cancer (tumor + non-EBV cell lines)"),
        ("src_adjacent_to_tumor", "Tumor-adjacent normal tissue"),
        ("src_activated_apc", "Activated APCs (DC/macrophage artifact)"),
        ("src_healthy_tissue", "Healthy somatic tissue (SAFETY SIGNAL)"),
        ("src_healthy_thymus", "Healthy thymus (expected for CTAs)"),
        ("src_healthy_reproductive", "Healthy reproductive (expected for CTAs)"),
        ("src_cell_line", "Any cell line (incl. EBV-LCL)"),
        ("src_ebv_lcl", "EBV-transformed B-LCL"),
        ("src_ex_vivo", "Direct ex vivo"),
    ]
    for flag, label in src_flags:
        if flag in df.columns:
            count = int(df[flag].sum())
            p(f"  {label:<50s} {count:>10,} ({count / total * 100:.1f}%)")
    p()

    # ── Unique peptides ─────────────────────────────────────────────────
    p("── Unique Peptides ──")
    p()
    p(f"  Total unique peptides:        {df['peptide'].nunique():>10,}")
    if "src_cancer" in df.columns:
        cancer_peps = set(df[df["src_cancer"]]["peptide"])
        healthy_peps = set(df[df.get("src_healthy_tissue", False) == True]["peptide"])  # noqa: E712
        p(f"  Cancer peptides:              {len(cancer_peps):>10,}")
        p(f"  Healthy somatic peptides:     {len(healthy_peps):>10,}")
        overlap = cancer_peps & healthy_peps
        p(f"  Overlap (cancer AND healthy): {len(overlap):>10,}")
        cancer_only = cancer_peps - healthy_peps
        p(f"  Cancer-specific (no healthy): {len(cancer_only):>10,}")
    p()

    # ── Tissue coverage (healthy) ───────────────────────────────────────
    if "src_healthy_tissue" in df.columns:
        healthy = df[df["src_healthy_tissue"]]
        if len(healthy) > 0:
            p("── Healthy Tissue Coverage ──")
            p()
            tissue_counts = healthy["source_tissue"].value_counts().head(25)
            p(f"  {healthy['source_tissue'].nunique()} tissues represented")
            p()
            for tissue, count in tissue_counts.items():
                peps = healthy[healthy["source_tissue"] == tissue]["peptide"].nunique()
                p(f"  {tissue:<30s} {count:>8,} rows  {peps:>8,} peptides")
            p()

    # ── Cancer disease breakdown ────────────────────────────────────────
    if "src_cancer" in df.columns:
        cancer = df[df["src_cancer"]]
        if len(cancer) > 0:
            p("── Cancer Disease Breakdown ──")
            p()
            nonempty = cancer[cancer["disease"].fillna("").str.strip() != ""]
            p(
                f"  Rows with disease annotation: {len(nonempty):,} / {len(cancer):,} "
                f"({len(nonempty) / len(cancer) * 100:.1f}%)"
            )
            p(f"  Unique diseases: {cancer['disease'].nunique()}")
            p()

            # Filter out "healthy" (mislabeled cell lines) and empty
            real_diseases = cancer[
                ~cancer["disease"].isin(["", "healthy"]) & cancer["disease"].notna()
            ]
            if len(real_diseases) > 0:
                disease_counts = real_diseases["disease"].value_counts().head(15)
                p("  Top cancer types (excluding cell line 'healthy' label):")
                for disease, count in disease_counts.items():
                    p(f"    {disease:<35s} {count:>8,} ({count / len(real_diseases) * 100:.1f}%)")
                p()

    # ── HLA allele coverage ─────────────────────────────────────────────
    p("── HLA Allele Coverage ──")
    p()
    allele_counts = df["mhc_restriction"].value_counts()
    p(f"  Unique alleles: {len(allele_counts)}")
    p()
    for allele, count in allele_counts.head(20).items():
        peps = df[df["mhc_restriction"] == allele]["peptide"].nunique()
        p(f"  {allele:<25s} {count:>8,} rows  {peps:>8,} peptides")
    p()

    # ── Cell line inventory ─────────────────────────────────────────────
    if "cell_line_name" in df.columns:
        cl_data = df[df["cell_line_name"].fillna("").str.strip() != ""]
        if len(cl_data) > 0:
            p("── Cell Line Inventory ──")
            p()
            cl_counts = cl_data["cell_line_name"].value_counts().head(20)
            p(f"  Named cell lines: {cl_data['cell_line_name'].nunique()}")
            p()
            for cl, count in cl_counts.items():
                peps = cl_data[cl_data["cell_line_name"] == cl]["peptide"].nunique()
                p(f"  {cl:<35s} {count:>8,} rows  {peps:>8,} peptides")
            p()

    # ── Per-PMID study summary ──────────────────────────────────────────
    if "pmid" in df.columns:
        p("── Top Studies by Row Count ──")
        p()
        pmid_counts = (
            df.groupby("pmid")
            .agg(
                rows=("peptide", "size"),
                peptides=("peptide", "nunique"),
            )
            .sort_values("rows", ascending=False)
            .head(15)
        )
        for pmid, row in pmid_counts.iterrows():
            p(
                f"  PMID {pmid!s:<12s} {int(row['rows']):>8,} rows  {int(row['peptides']):>8,} peptides"
            )
        p()

    # ── PMID override summary ───────────────────────────────────────────
    from .curation import load_pmid_overrides

    overrides = load_pmid_overrides()
    override_pmids = set(overrides.keys())
    matched = df[df["pmid"].apply(lambda x: _safe_int(x) in override_pmids)]
    if len(matched) > 0:
        p("── Curated Study Overrides Applied ──")
        p()
        for pmid_int, entry in sorted(overrides.items()):
            target_pmid = pmid_int
            pmid_rows = matched[matched["pmid"].apply(lambda x, t=target_pmid: _safe_int(x) == t)]
            if len(pmid_rows) > 0:
                p(f"  PMID {pmid_int}: {entry['label']}")
                p(f"    Override: {entry['override']}  |  Rows: {len(pmid_rows):,}")
                if entry.get("tissue_overrides"):
                    p(f"    Tissue overrides: {len(entry['tissue_overrides'])} tissues")
        p()

    p("=" * 72)
    p("Report complete.")

    text = "\n".join(lines)
    if output:
        Path(output).write_text(text)
        print(f"Report saved to {output}")
    return text


def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return -1


def run_report(
    iedb_path: str | Path | None = None,
    cedar_path: str | Path | None = None,
    mhc_class: str | None = None,
    output: str | Path | None = None,
) -> str:
    """Convenience function: scan + report in one call.

    Parameters
    ----------
    iedb_path, cedar_path
        Paths to IEDB/CEDAR exports. If None, tries to resolve from registry.
    mhc_class
        MHC class filter for the report.
    output
        Optional output file path.

    Returns
    -------
    str
        The report text.
    """
    import contextlib

    from .downloads import get_path
    from .scanner import scan

    if iedb_path is None:
        with contextlib.suppress(KeyError, FileNotFoundError):
            iedb_path = str(get_path("iedb"))
    if cedar_path is None:
        with contextlib.suppress(KeyError, FileNotFoundError):
            cedar_path = str(get_path("cedar"))

    if iedb_path is None and cedar_path is None:
        print(
            "No IEDB or CEDAR data registered. Run 'hitlist data available' for instructions.",
            file=sys.stderr,
        )
        return ""

    df = scan(
        peptides=None,
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        human_only=True,
        hla_only=True,
        classify_source=True,
    )

    return generate_report(df, mhc_class_filter=mhc_class, output=output)
