"""Source peptidoform and preparation-chemistry provenance for Stražar 2023 & Abelin/MAPTAC (#896).

Implements #896 on top of #854:
- Separates preparation chemistry and timing (e.g., Stražar 2023 PMC10519123 iodoacetamide
  alkylation / fixed cysteine carbamidomethylation + variable modifications) from native
  in-vivo modification status.
- Preserves explicit dispositions: ``unmodified``, ``modified``, ``preparation_derivatized_only``,
  ``multi_form_ambiguous``, and ``unresolved_form`` — never blanket-filling ``has_ptm=False``
  for bare sequences without PSM-level evidence.
- Joins original PSM/identified-form tables (carrying source/run/scan/modified_sequence/ptm_sites)
  to supplement peptide-MHC rows while keeping multiple contributing forms distinct and keeping
  current-study rows (e.g. Stražar ``internal_aff_clean1and2``) separate from reused IEDB/Abelin
  rows, prediction-deconvolved entries, and generated decoys.
- Blocks silent aggregation of modified pMHCs into unmodified affinity measurements.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "peptidoform_preparation_contracts.json"
)
_MOD_TOKEN_RE = re.compile(r"\[[^\]]+\]|\([^)]+\)")
_PREP_DERIV_TOKENS = frozenset(
    {
        "carbamidomethyl",
        "unimod:4",
        "+57.021",
        "+57.0215",
        "+57.02146",
        "cam",
    }
)


def peptidoform_preparation_contracts() -> dict[str, Any]:
    """Return the pinned #896 peptidoform and preparation-chemistry provenance contracts."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def get_peptidoform_preparation_contract(pmid: int | str) -> dict[str, Any]:
    """Return the preparation-chemistry contract for a tracked study (37301199 or 31495665)."""
    key = str(int(pmid))
    studies = peptidoform_preparation_contracts()["studies"]
    if key not in studies:
        raise KeyError(f"PMID {key} has no reviewed #896 peptidoform preparation contract")
    return studies[key]


def _classify_single_psm_form(modified_sequence: str, ptm_annotation: str | None) -> str:
    """Classify one identified PSM form without conflating prep derivatization with native PTMs."""
    mod_seq = str(modified_sequence or "").strip()
    ptm_str = str(ptm_annotation or "").strip()
    tokens = _MOD_TOKEN_RE.findall(mod_seq)
    if not tokens and not ptm_str:
        return "unmodified"

    # Collect normalized modification labels from either inline brackets or ptm_annotation
    labels: list[str] = []
    for tok in tokens:
        labels.append(tok.strip("[]() ").lower())
    if ptm_str and ptm_str.lower() not in ("none", "unmodified", "nan", ""):
        for part in re.split(r"[;,|]+", ptm_str):
            cleaned = part.strip().lower()
            if cleaned:
                labels.append(cleaned)

    if not labels:
        return "unmodified"
    if all(any(prep in lbl for prep in _PREP_DERIV_TOKENS) for lbl in labels):
        return "preparation_derivatized_only"
    return "modified"


def join_psms_to_supplement_rows(
    supplement_df: pd.DataFrame,
    psm_df: pd.DataFrame | None,
    *,
    pmid: int,
    allow_current_study_only: bool = True,
) -> pd.DataFrame:
    """Join original PSM identified forms to supplement peptide-MHC rows with exact provenance.

    Parameters
    ----------
    supplement_df:
        Supplement peptide-MHC table with at least ``peptide`` and ``mhc_restriction`` columns
        (and optionally ``dataset``, ``source_row``, ``source_sheet``, ``source_asset``).
    psm_df:
        Optional original PSM table with ``peptide``, ``modified_sequence``, and optional
        ``mhc_restriction``, ``ptm_annotation``, ``source_file``, ``run_id``, ``scan_number``.
        If ``None`` or empty, every supplement row stays explicitly ``unresolved_form`` (never
        blanket-filled as ``has_ptm=False``).
    pmid:
        Study PMID (e.g. ``37301199`` for Stražar 2023 or ``31495665`` for Abelin/MAPTAC).
    allow_current_study_only:
        When True and a ``dataset`` column is present on Stražar 2023 rows, mark non-current-study
        rows (reused IEDB/Abelin, deconvolved, decoys) as ``excluded_reused_or_decoy_subset`` so
        they are never counted as independent new observations.
    """
    contract = get_peptidoform_preparation_contract(pmid)
    prep = contract["preparation_chemistry"]
    alkylation_reagent = prep["alkylation_reagent"]
    alkylation_timing = prep["alkylation_timing"]
    subset_spec = contract.get("current_study_subset_filter")

    # Index PSMs by (peptide, mhc_restriction) and fallback by (peptide,) when PSM table has no allele
    psm_by_pep_allele: dict[tuple[str, str], list[dict[str, Any]]] = {}
    psm_by_pep: dict[str, list[dict[str, Any]]] = {}
    has_psm_allele = psm_df is not None and len(psm_df) > 0 and "mhc_restriction" in psm_df.columns
    if psm_df is not None and len(psm_df) > 0:
        for idx, rec in psm_df.iterrows():
            pep = str(rec.get("peptide", "")).strip()
            if not pep:
                continue
            mod_seq = str(rec.get("modified_sequence", pep) or pep).strip()
            ptm_ann = rec.get("ptm_annotation")
            ptm_str = "" if (ptm_ann is None or pd.isna(ptm_ann)) else str(ptm_ann).strip()
            form_class = _classify_single_psm_form(mod_seq, ptm_str)
            entry = {
                "modified_sequence": mod_seq,
                "ptm_annotation": ptm_str,
                "form_class": form_class,
                "source_file": str(rec.get("source_file", "") or "").strip(),
                "run_id": str(rec.get("run_id", "") or "").strip(),
                "scan_number": rec.get("scan_number"),
                "psm_row_index": int(idx),
            }
            psm_by_pep.setdefault(pep, []).append(entry)
            if has_psm_allele:
                allele = str(rec.get("mhc_restriction", "") or "").strip()
                psm_by_pep_allele.setdefault((pep, allele), []).append(entry)

    out_rows: list[dict[str, Any]] = []
    for row_idx, rec in supplement_df.iterrows():
        pep = str(rec.get("peptide", "")).strip()
        allele = str(rec.get("mhc_restriction", "") or "").strip()
        dataset_val = rec.get("dataset")
        is_independent_current_study = True
        subset_disposition = "current_study_independent"
        if allow_current_study_only and subset_spec and "dataset" in supplement_df.columns:
            expected_val = subset_spec["value"]
            if str(dataset_val or "").strip() != expected_val:
                is_independent_current_study = False
                subset_disposition = "excluded_reused_or_decoy_subset"

        matches = (
            psm_by_pep_allele.get((pep, allele), []) if has_psm_allele else psm_by_pep.get(pep, [])
        )
        if not matches:
            disposition = "unresolved_form"
            has_ptm: bool | None = None
            has_bio_ptm: bool | None = None
            observed_forms: list[str] = []
            observed_ptms: list[str] = []
            run_ids: list[str] = []
            source_files: list[str] = []
            n_psm_matches = 0
            n_distinct_forms = 0
        else:
            observed_forms = sorted({m["modified_sequence"] for m in matches})
            observed_ptms = sorted({m["ptm_annotation"] for m in matches if m["ptm_annotation"]})
            form_classes = {m["form_class"] for m in matches}
            run_ids = sorted({m["run_id"] for m in matches if m["run_id"]})
            source_files = sorted({m["source_file"] for m in matches if m["source_file"]})
            n_psm_matches = len(matches)
            n_distinct_forms = len(observed_forms)

            if len(form_classes) > 1 or n_distinct_forms > 1:
                disposition = "multi_form_ambiguous"
                has_ptm = None
                has_bio_ptm = None
            else:
                sole_class = next(iter(form_classes))
                disposition = sole_class
                if sole_class == "unmodified":
                    has_ptm = False
                    has_bio_ptm = False
                elif sole_class == "preparation_derivatized_only":
                    has_ptm = True
                    has_bio_ptm = False
                else:
                    has_ptm = True
                    has_bio_ptm = True

        out_rows.append(
            {
                "pmid": int(pmid),
                "peptide": pep,
                "mhc_restriction": allele,
                "source_asset": rec.get("source_asset", contract.get("supplement_file", "")),
                "source_sheet": rec.get("source_sheet", ""),
                "source_row": rec.get("source_row", int(row_idx) + 2),
                "dataset_subset": str(dataset_val) if dataset_val is not None else "",
                "is_independent_current_study": is_independent_current_study,
                "subset_disposition": subset_disposition,
                "peptidoform_disposition": disposition,
                "has_ptm": has_ptm,
                "has_biological_ptm": has_bio_ptm,
                "observed_modified_sequences": ";".join(observed_forms),
                "observed_ptm_annotations": ";".join(observed_ptms),
                "n_psm_matches": n_psm_matches,
                "n_distinct_forms": n_distinct_forms,
                "psm_source_files": ";".join(source_files),
                "psm_run_ids": ";".join(run_ids),
                "preparation_alkylation_reagent": alkylation_reagent,
                "preparation_chemistry_timing": alkylation_timing,
                "eligible_unmodified_ms_observation": (
                    is_independent_current_study and disposition == "unmodified"
                ),
                "eligible_for_unmodified_affinity_aggregation": False,
            }
        )

    return pd.DataFrame(out_rows)


def summarize_peptidoform_eligibility(joined_df: pd.DataFrame) -> dict[str, Any]:
    """Report recovered vs independent counts and explicit peptidoform eligibility breakdowns."""
    if len(joined_df) == 0:
        return {
            "n_total_rows": 0,
            "n_independent_current_study_rows": 0,
            "n_excluded_reused_or_decoy_rows": 0,
            "n_unmodified_rows": 0,
            "n_modified_rows": 0,
            "n_preparation_derivatized_only_rows": 0,
            "n_multi_form_ambiguous_rows": 0,
            "n_unresolved_form_rows": 0,
            "n_eligible_unmodified_independent_rows": 0,
            "allows_silent_modified_into_unmodified_affinity": False,
        }
    disp = joined_df["peptidoform_disposition"]
    indep = joined_df["is_independent_current_study"].astype(bool)
    return {
        "n_total_rows": len(joined_df),
        "n_independent_current_study_rows": int(indep.sum()),
        "n_excluded_reused_or_decoy_rows": int((~indep).sum()),
        "n_unmodified_rows": int((disp == "unmodified").sum()),
        "n_modified_rows": int((disp == "modified").sum()),
        "n_preparation_derivatized_only_rows": int((disp == "preparation_derivatized_only").sum()),
        "n_multi_form_ambiguous_rows": int((disp == "multi_form_ambiguous").sum()),
        "n_unresolved_form_rows": int((disp == "unresolved_form").sum()),
        "n_eligible_unmodified_independent_rows": int(
            joined_df["eligible_unmodified_ms_observation"].astype(bool).sum()
        ),
        "allows_silent_modified_into_unmodified_affinity": False,
    }


def guard_affinity_aggregation(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Reject modified, derivatized, multi-form, or unresolved MS rows from unmodified affinity tables."""
    disallowed = joined_df["peptidoform_disposition"] != "unmodified"
    if disallowed.any():
        bad_counts = joined_df.loc[disallowed, "peptidoform_disposition"].value_counts().to_dict()
        raise ValueError(
            "Cannot silently aggregate non-unmodified or unresolved MS peptidoforms into "
            f"unmodified affinity measurements: {bad_counts}"
        )
    return joined_df.copy()
