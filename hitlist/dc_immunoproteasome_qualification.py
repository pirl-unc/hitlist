"""Source-qualified de Verteuil 2010 (PMID 20484733, PMC2938112) DC immunoproteasome contract (#798).

Implements #798:
- Pins verified non-OA access receipts (EuropePMC isOpenAccess=N, hasSuppl=Y, PMC S3 empty,
  PRIDE 0 projects, MCP/Elsevier 403 — no access controls bypassed).
- Encodes the concurrent Psmb8 (LMP7) + Psmb10 (MECL1) double knockout separately from the
  observed secondary protein-level reduction of Psmb9 (LMP2) and the B2m-deficient specificity
  background control (never an ordinary peptide-abundance training negative).
- Enforces the 15,000 detection-limit lower-bound inequality contract: WT/dKO ratios computed
  from the reported 15,000 detection limit are preserved as lower-bound inequalities ('>=')
  and NEVER coerced to exact finite ratios or numeric KO zeros.
- Retains predicted-MHC-motif search filtering as ascertainment metadata (distinct from measured
  single-allele labels and the C57BL/6 H2-Kb/H2-Db repertoire) and separates RNA transcript
  measurements from precursor protein abundance.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "de_verteuil_2010_dc_20484733.json"
)
DETECTION_LIMIT_VALUE = 15000.0
_INEQUALITY_RE = re.compile(r"^\s*(>=|>|<=|<)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")


def de_verteuil_2010_source_contract() -> dict[str, Any]:
    """Return the pinned de Verteuil 2010 (PMID 20484733, PMC2938112) source contract."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _parse_ratio_cell(raw: Any) -> tuple[str, float | None]:
    """Parse a raw WT/dKO ratio cell while preserving inequality qualifiers ('>', '>=')."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "missing", None
    text = str(raw).strip()
    if not text or text.lower() in {"na", "nan", "nd", "n/d", "null", "-"}:
        return "missing", None
    match = _INEQUALITY_RE.match(text)
    if match:
        op, num_str = match.groups()
        val = float(num_str)
        return (">=" if op in {">", ">="} else op), (val if math.isfinite(val) else None)
    try:
        val = float(text)
    except ValueError:
        return "unparsed", None
    return ("=" if math.isfinite(val) else "missing"), (val if math.isfinite(val) else None)


def _parse_optional_float(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"na", "nan", "nd", "n/d", "null", "-"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def qualify_dc_immunoproteasome_records(
    df: pd.DataFrame,
    *,
    source_asset: str,
    source_sha256: str,
) -> pd.DataFrame:
    """Attach de Verteuil 2010 (#798) semantic guards to an authorized local Table S1 frame.

    Required column: ``peptide``.
    Optional columns recognized and preserved without coercion:
    - ``wt_intensity``, ``dko_intensity``, ``wt_to_dko_ratio_raw``
    - ``arm_role`` (defaults to ``"wt_vs_psmb8_psmb10_dko_contrast"``; recognizes
      ``"b2m_ko_specificity_control"``)
    - ``is_detection_limit_ratio`` (bool)
    - ``predicted_mhc_allele`` (e.g. ``H2-Kb`` or ``H2-Db``)
    - ``source_gene``, ``source_protein``, ``rna_log2_fc``, ``protein_log2_fc``
    """
    if "peptide" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'peptide' column")

    rows: list[dict[str, Any]] = []
    for idx, rec in df.iterrows():
        peptide = str(rec["peptide"]).strip()
        arm_role = str(rec.get("arm_role", "wt_vs_psmb8_psmb10_dko_contrast")).strip()
        if not arm_role or arm_role.lower() == "nan":
            arm_role = "wt_vs_psmb8_psmb10_dko_contrast"

        wt_val = _parse_optional_float(rec.get("wt_intensity"))
        dko_val = _parse_optional_float(rec.get("dko_intensity"))
        raw_ratio = rec.get("wt_to_dko_ratio_raw")
        qualifier, parsed_ratio = _parse_ratio_cell(raw_ratio)
        raw_dl = rec.get("is_detection_limit_ratio", False)
        dl_flag = False if pd.isna(raw_dl) else bool(raw_dl)

        # Detect forbidden WT-only -> KO numeric zero coercion or detection-limit inequalities.
        at_detection_limit = (
            dl_flag
            or qualifier in {">", ">="}
            or (
                dko_val is not None
                and math.isclose(dko_val, DETECTION_LIMIT_VALUE, rel_tol=0.0, abs_tol=1e-6)
            )
        )

        if arm_role == "b2m_ko_specificity_control":
            value_state = "b2m_specificity_control"
            ratio_qualifier = qualifier
            ratio_value = parsed_ratio
        elif wt_val is not None and wt_val > 0 and dko_val is not None and dko_val == 0.0:
            # #798 explicit guard: NEVER treat a WT-only detection as a numeric KO zero!
            value_state = "coerced_zero_forbidden"
            ratio_qualifier = "invalid_coerced_zero"
            ratio_value = None
        elif at_detection_limit:
            value_state = "lower_bound_inequality"
            ratio_qualifier = ">="
            if parsed_ratio is not None:
                ratio_value = parsed_ratio
            elif wt_val is not None and wt_val > 0:
                ratio_value = wt_val / DETECTION_LIMIT_VALUE
            else:
                ratio_value = None
        elif parsed_ratio is not None and qualifier == "=":
            value_state = "exact_ratio"
            ratio_qualifier = "="
            ratio_value = parsed_ratio
        elif wt_val is not None and wt_val > 0 and dko_val is not None and dko_val > 0:
            value_state = "exact_ratio"
            ratio_qualifier = "="
            ratio_value = wt_val / dko_val
        elif wt_val is not None and wt_val > 0 and dko_val is None:
            value_state = "wt_only_unquantified"
            ratio_qualifier = "missing"
            ratio_value = None
        elif dko_val is not None and dko_val > 0 and wt_val is None:
            value_state = "dko_only_unquantified"
            ratio_qualifier = "missing"
            ratio_value = None
        else:
            value_state = "missing"
            ratio_qualifier = "missing"
            ratio_value = None

        pred_allele = str(rec.get("predicted_mhc_allele", "")).strip()
        if pred_allele.lower() == "nan":
            pred_allele = ""

        rows.append(
            {
                "study_id": "de_verteuil_2010_dc_20484733",
                "pmid": 20484733,
                "pmc_id": "PMC2938112",
                "source_asset": source_asset,
                "source_sha256": source_sha256,
                "source_row": int(idx) + 2,
                "species": "Mus musculus",
                "strain_background": "C57BL/6",
                "cell_type": "primary_bone_marrow_derived_dendritic_cell",
                "mhc_class": "I",
                "sample_mhc_repertoire": "H2-Kb;H2-Db",
                "predicted_mhc_allele": pred_allele,
                "mhc_assignment_provenance": (
                    "predicted_mhc_motif_filter" if pred_allele else "repertoire_only_unassigned"
                ),
                "predicted_allele_is_measured_single_allele": False,
                "arm_role": arm_role,
                "genotype_knockout_genes": (
                    "B2m" if arm_role == "b2m_ko_specificity_control" else "Psmb8;Psmb10"
                ),
                "genotype_knockout_aliases": (
                    "beta2m" if arm_role == "b2m_ko_specificity_control" else "LMP7;MECL1"
                ),
                "secondary_observed_protein_reduction": (
                    "" if arm_role == "b2m_ko_specificity_control" else "Psmb9 (LMP2)"
                ),
                "b2m_control_eligible_as_abundance_negative": False,
                "peptide": peptide,
                "wt_intensity": wt_val,
                "dko_intensity": dko_val,
                "wt_to_dko_ratio_raw": "" if raw_ratio is None else str(raw_ratio),
                "ratio_qualifier": ratio_qualifier,
                "wt_to_dko_ratio_value": ratio_value,
                "detection_limit_value": DETECTION_LIMIT_VALUE if at_detection_limit else None,
                "value_state": value_state,
                "source_gene": str(rec.get("source_gene", "")).strip(),
                "source_protein": str(rec.get("source_protein", "")).strip(),
                "rna_log2_fc": _parse_optional_float(rec.get("rna_log2_fc")),
                "protein_log2_fc": _parse_optional_float(rec.get("protein_log2_fc")),
                "rna_evidence_is_measured_precursor_protein": False,
                "evidence_scope": "mhc_eluted_response_evidence_not_free_intracellular_concentration",
            }
        )

    return pd.DataFrame(rows)


def dc_immunoproteasome_contrast_recipe(
    qualified_df: pd.DataFrame,
    *,
    include_lower_bound_inequalities: bool = False,
) -> dict[str, Any]:
    """Partition qualified de Verteuil 2010 records into exact, inequality, control, and excluded views.

    Never drops source rows: every input row appears in exactly one output partition and
    contributes to explicit unit-suffixed summary counts.
    """
    required = {"value_state", "arm_role", "ratio_qualifier"}
    missing = sorted(required - set(qualified_df.columns))
    if missing:
        raise ValueError(f"qualified_df is missing required columns: {missing}")

    df = qualified_df.copy()
    dispositions: list[str] = []
    reasons: list[str] = []
    for row in df.itertuples(index=False):
        state = row.value_state
        if state == "b2m_specificity_control" or row.arm_role == "b2m_ko_specificity_control":
            dispositions.append("specificity_control")
            reasons.append("b2m_ko_specificity_background_control_not_abundance_negative")
        elif state == "exact_ratio":
            dispositions.append("admitted_exact_contrast")
            reasons.append("exact_wt_vs_psmb8_psmb10_dko_ratio")
        elif state == "lower_bound_inequality":
            if include_lower_bound_inequalities:
                dispositions.append("admitted_inequality_contrast")
                reasons.append("lower_bound_15000_detection_limit_inequality")
            else:
                dispositions.append("retained_inequality")
                reasons.append("lower_bound_15000_detection_limit_not_exact_ratio")
        else:
            dispositions.append("excluded")
            reasons.append(str(state))

    df["recipe_disposition"] = dispositions
    df["recipe_reason"] = reasons

    exact = df[df["recipe_disposition"] == "admitted_exact_contrast"].reset_index(drop=True)
    inequalities = df[
        df["recipe_disposition"].isin(["retained_inequality", "admitted_inequality_contrast"])
    ].reset_index(drop=True)
    controls = df[df["recipe_disposition"] == "specificity_control"].reset_index(drop=True)
    exclusions = df[df["recipe_disposition"] == "excluded"].reset_index(drop=True)

    summary = {
        "study_id": "de_verteuil_2010_dc_20484733",
        "pmid": 20484733,
        "pmc_id": "PMC2938112",
        "detection_limit_value": DETECTION_LIMIT_VALUE,
        "include_lower_bound_inequalities": bool(include_lower_bound_inequalities),
        "n_source_rows": len(df),
        "n_exact_quantitative": len(exact),
        "n_lower_bound_inequalities": len(inequalities),
        "n_b2m_specificity_controls": len(controls),
        "n_excluded": len(exclusions),
        "n_coerced_zero_rejected": int((df["value_state"] == "coerced_zero_forbidden").sum()),
        "n_admitted_contrasts": int(
            len(exact) + (len(inequalities) if include_lower_bound_inequalities else 0)
        ),
        "predicted_allele_is_measured_single_allele": False,
        "rna_evidence_is_measured_precursor_protein": False,
        "wt_only_conversion_to_ko_zero_allowed": False,
    }
    return {
        "exact_quantitative_contrasts": exact,
        "lower_bound_inequality_records": inequalities,
        "specificity_control_records": controls,
        "exclusions": exclusions,
        "all_records": df,
        "summary": summary,
    }
