"""Qualification and synthetic-demo exclusion guard for APC endolysosomal degradation data (#812).

Implements #812 for Öztemiz et al. (2026), PMID 42846314, PMC13643861,
DOI 10.3389/falgy.2026.1936335:
- Inventories the PMC13643861 supplementary PDFs (Datasheet1..5.pdf) and records that no public
  PRIDE/MassIVE/Zenodo tabular peptide time-course dataset is available (qualification status:
  ``qualified_source_gap_no_tabular_peptide_deposit``; zero values fabricated from prose/figures).
- Explicitly separates biological system and assay scope: murine JAWSII DC isolated endolysosomal
  extract (cell-free isolated compartment, NOT intact APC intracellular concentrations), substrates
  (Bet v 1, Aes h 1), preparation chemistry (20 mM iodoacetamide alkylation — never interpreted as
  biological change), and three distinct measurement families:
  1. ``peptide_ms_signal`` (retains short and nonbinding proteolytic fragments)
  2. ``substrate_gel_depletion`` (densitometric gel half-life)
  3. ``cathepsin_zfr_amc_reporter_activity`` (NOT peptide-specific molar calibration)
- Provides an adversarial exclusion guard rejecting ``oztemiz-elif/peptide-studio`` commit
  ``18dd99da4a619abd8a28c7622e0809eb2c03e0ff`` synthetic demo files (``Example_degradome.xlsx``,
  ``example.fasta``, human myoglobin ``P02144`` / ``MB``) from biological training, plus a
  registered recipe enforcing unit qualification and time-point dependence grouping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "oztemiz_2026_endolysosomal.json"
)
_SYNTHETIC_DEMO_FILENAMES = frozenset({"example_degradome.xlsx", "example.fasta"})
_SYNTHETIC_DEMO_PROTEINS = frozenset({"p02144", "myg_human"})
_SYNTHETIC_TIME_COLS = frozenset({"0h", "0.5h", "1h", "2h", "3h", "5h", "8h", "16h", "24h"})


class SyntheticDemoDataError(ValueError):
    """Raised when synthetic software demonstration data (e.g. PeptideStudio) is submitted."""


def oztemiz_2026_source_contract() -> dict[str, Any]:
    """Return the pinned Öztemiz 2026 (PMID 42846314) endolysosomal qualification contract."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def guard_against_synthetic_peptidestudio_data(
    *,
    source_path: str | Path | None = None,
    source_url: str | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Reject synthetic oztemiz-elif/peptide-studio demonstration artifacts before ingestion.

    Inspects filename, repository URL/commit markers, protein accessions/gene symbols (`P02144`,
    `MYG_HUMAN`, myoglobin `MB`), and the synthetic `Example_degradome.xlsx` column signature.
    Raises ``SyntheticDemoDataError`` on any match.
    """
    reasons: list[str] = []
    if source_path is not None:
        name_lower = Path(source_path).name.lower()
        path_lower = str(source_path).lower()
        if name_lower in _SYNTHETIC_DEMO_FILENAMES:
            reasons.append(f"excluded synthetic demo filename: {Path(source_path).name}")
        if "peptide-studio" in path_lower and "sample_data" in path_lower:
            reasons.append("path points to oztemiz-elif/peptide-studio sample_data")

    if source_url is not None:
        url_lower = source_url.lower()
        if "peptide-studio" in url_lower and (
            "sample_data" in url_lower
            or "example_degradome" in url_lower
            or "18dd99da" in url_lower
        ):
            reasons.append(f"excluded synthetic PeptideStudio URL: {source_url}")

    if df is not None and len(df) > 0:
        cols_lower = {str(c).strip().lower() for c in df.columns}
        if _SYNTHETIC_TIME_COLS.issubset(cols_lower) and "protein sequence" in cols_lower:
            reasons.append(
                "table matches PeptideStudio Example_degradome.xlsx synthetic column signature"
            )
        for col in ("protein_accession", "uniprot_accession", "accession", "protein"):
            if col in df.columns:
                vals = {str(v).strip().lower() for v in df[col].dropna().unique()}
                hits = vals & _SYNTHETIC_DEMO_PROTEINS
                if hits:
                    reasons.append(
                        f"table contains excluded PeptideStudio demo protein accession(s): {sorted(hits)}"
                    )
        # Check for human myoglobin 154-aa signature sequence (GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKF...)
        for col in ("protein_sequence", "Protein Sequence"):
            if (
                col in df.columns
                and df[col].astype(str).str.startswith("GLSDGEWQLVLNVWGKVEAD").any()
            ):
                reasons.append(
                    "table contains synthetic PeptideStudio human myoglobin (P02144) sequence"
                )

    if reasons:
        raise SyntheticDemoDataError(
            "Refusing to admit synthetic PeptideStudio demonstration data as biological "
            f"endolysosomal measurements (#812): {'; '.join(reasons)}"
        )

    return {"passed": True, "synthetic_demo_detected": False}


def endolysosomal_degradation_recipe(
    records_df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    source_url: str | None = None,
    target_mode: str = "relative_peptide_signal",
) -> dict[str, Any]:
    """Qualify endolysosomal degradation time-course observations for downstream modeling (#812).

    Enforces:
    - Synthetic PeptideStudio demo rejection (`guard_against_synthetic_peptidestudio_data`).
    - Strict separation of ``peptide_ms_signal`` from ``substrate_gel_depletion`` and
      ``cathepsin_zfr_amc_reporter_activity``.
    - Rejection of unknown/non-molar quantity units when ``target_mode == 'molar_concentration'``.
    - Preservation of dependent time-course trajectory grouping (``trajectory_group_id``) and
      short/nonbinding proteolytic fragments (no HLA-binding length filter applied!).
    - Explicit isolation of cell-free extract scope from intact-cell APC concentrations.
    """
    guard_against_synthetic_peptidestudio_data(
        source_path=source_path, source_url=source_url, df=records_df
    )
    contract = oztemiz_2026_source_contract()
    allowed_molar_units = set(contract["concentration_training_allowed_units"])
    valid_families = set(
        contract["biological_system_and_assay_scope"]["distinct_measurement_families"].keys()
    )

    df = records_df.copy()
    if "measurement_family" not in df.columns:
        raise ValueError("records_df must specify 'measurement_family'")
    unknown_families = set(df["measurement_family"].dropna().astype(str)) - valid_families
    if unknown_families:
        raise ValueError(f"Unsupported measurement_family values: {sorted(unknown_families)}")

    # Ensure trajectory dependence and isolated-extract provenance columns exist
    if "time_hours" not in df.columns:
        raise ValueError("records_df must include 'time_hours' for dependent time-course modeling")
    if "preparation_context" not in df.columns:
        df["preparation_context"] = "cell_free_isolated_extract_not_intact_apc"
    df["is_intact_cell_concentration"] = (
        df["preparation_context"].astype(str) == "intact_cell_native_apc"
    )
    df["trajectory_group_id"] = (
        df.get("substrate_protein", pd.Series(["unknown"] * len(df), index=df.index)).astype(str)
        + "::"
        + df.get("extract_batch_id", pd.Series(["batch_unknown"] * len(df), index=df.index)).astype(
            str
        )
        + "::"
        + df.get("replicate_id", pd.Series(["rep_unknown"] * len(df), index=df.index)).astype(str)
    )

    # Filter admitted rows according to target_mode
    exclusion_reasons: list[str | None] = []
    for _, rec in df.iterrows():
        fam = str(rec["measurement_family"])
        unit = str(rec.get("quantity_unit", "") or "").strip()
        if fam != "peptide_ms_signal":
            exclusion_reasons.append(f"non_peptide_measurement_family:{fam}")
            continue
        if target_mode == "molar_concentration":
            if not unit or unit.lower() in ("unknown", "arbitrary", "au", "intensity", "area"):
                exclusion_reasons.append("unknown_or_non_molar_unit_for_concentration_training")
                continue
            if unit not in allowed_molar_units:
                exclusion_reasons.append(f"disallowed_concentration_unit:{unit}")
                continue
        exclusion_reasons.append(None)

    df["exclusion_reason"] = exclusion_reasons
    admitted = df[df["exclusion_reason"].isna()].reset_index(drop=True)
    excluded = df[df["exclusion_reason"].notna()].reset_index(drop=True)

    return {
        "recipe_id": "hitlist.endolysosomal_degradation_time_course.v1",
        "target_mode": target_mode,
        "time_series_group_column": "trajectory_group_id",
        "preserves_short_and_nonbinding_fragments": True,
        "distinct_from_intact_apc_concentration": True,
        "n_input_rows": len(df),
        "n_admitted_rows": len(admitted),
        "n_excluded_rows": len(excluded),
        "n_admitted_trajectories": (
            int(admitted["trajectory_group_id"].nunique()) if len(admitted) else 0
        ),
        "admitted": admitted,
        "excluded": excluded,
    }
