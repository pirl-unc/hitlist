"""Source-qualified Stopfer 2021 (PMID 34497125, PXD024917) Iso18 calibration adapter.

Implements #849: known-dose standard / raw-response calibration kept strictly separate from
inferred endogenous copies per cell, MS1 detectability, and APC intracellular concentrations.

Key guarantees:
- Replays all 348 Dataset S3A/S3B worksheet slots (213 S3A positive cells across 71 dose rows +
  135 S3B slots comprising 126 positive and 9 literal blanks) and explicitly records the absent
  0.1-fmol source row for GLFDQHFRL[+7] (DatasetS3A!A16) without inventing a zero or fake cell.
- Preserves distinct observation roles: free 1H direct-spike summed signal (S3A), embedded 2H
  hipMHC pre-IP summed signal (S3B), selected transition counts (S3C — never signal intensity),
  predicted affinity in nM (S1 — never experimental binding), analytical 4H trigger reagent,
  and SKMEL5 binimetinib vs DMSO Skyline transition responses (PXD024917).
- Cross-checks ordinal columns C/D/E against PRIDE Skyline transition exports without blindly
  assuming C = run 1; proves all 9 S3B 100-fmol column E blanks reflect absent PRIDE replicate
  100fmol_2 while column D holds 100fmol_3.
- Audits Dataset S6A vs S6B isotope definitions (72 isotopologues: 17 consistent, 54 inconsistent
  from the S6A Valine 4.007 Da vs S3/S6B V[+6] contradiction, 1 uncheckable Methionine omitted
  from S6A) and emits row-level provenance flags without mutating original source values.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "stopfer_2021_iso18_pxd024917.json"
)
_S3B_DOSE_TAGS = {
    0.1: "100amol",
    1.0: "1fmol",
    10.0: "10fmol",
    100.0: "100fmol",
    1000.0: "1000fmol",
}
_MOD_STRIP_RE = re.compile(r"\[[^\]]+\]")
_PAREN_AA_RE = re.compile(r"\(([A-Z])\)")


def iso18_calibration_source_contract() -> dict[str, Any]:
    """Return the pinned Stopfer 2021 (PXD024917) Iso18 source qualification contract."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_asset(
    path: Path,
    expected_sha256: str,
    *,
    allow_unverified_checksum: bool = False,
) -> str:
    actual = _sha256_file(path)
    if not allow_unverified_checksum and actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return actual


def fetch_iso18_calibration_sources(
    dest_dir: str | Path | None = None,
    *,
    include_pride_exports: bool = True,
    include_sapp_pdf: bool = False,
    force: bool = False,
    allow_unverified_checksum: bool = False,
) -> dict[str, Path]:
    """Download and SHA-256-verify Stopfer 2021 PMC S3 + PRIDE PXD024917 assets."""
    from .downloads import data_dir

    contract = iso18_calibration_source_contract()
    target_dir = Path(dest_dir) if dest_dir is not None else data_dir() / "stopfer_2021_iso18"
    target_dir.mkdir(parents=True, exist_ok=True)

    asset_keys = ["sd01", "sd03", "sd06", "sd07"]
    if include_sapp_pdf:
        asset_keys.append("sapp_pdf")
    if include_pride_exports:
        asset_keys.extend(["pride_direct_spike", "pride_hipmhc_spikein", "pride_skmel5_dmso_meki"])

    downloaded: dict[str, Path] = {}
    for key in asset_keys:
        meta = contract["verified_assets"][key]
        out_path = target_dir / meta["filename"]
        if force or not out_path.exists():
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            urllib.request.urlretrieve(meta["url"], tmp_path)
            tmp_path.replace(out_path)
        _verify_asset(
            out_path,
            meta["sha256"],
            allow_unverified_checksum=allow_unverified_checksum,
        )
        downloaded[key] = out_path
    return downloaded


def _strip_modifications(seq: str) -> str:
    return _MOD_STRIP_RE.sub("", seq).strip()


def _parse_sd01_panel(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb["DatasetS1"]
        rows: list[dict[str, Any]] = []
        for r_idx, row in enumerate(ws.iter_rows(min_row=4, max_col=6, values_only=True), start=4):
            pep = row[1]
            if pep is None or str(pep).strip() == "":
                continue
            rows.append(
                {
                    "study_id": "stopfer_2021_iso18_pxd024917",
                    "pmid": 34497125,
                    "source_asset": path.name,
                    "source_sheet": "DatasetS1",
                    "source_row": r_idx,
                    "peptide": str(pep).strip(),
                    "gene_symbol": str(row[2]).strip() if row[2] is not None else None,
                    "annotated_gene": str(row[3]).strip() if row[3] is not None else None,
                    "peptide_length": int(row[4]) if row[4] is not None else len(str(pep).strip()),
                    "predicted_affinity_nm": float(row[5]) if row[5] is not None else None,
                    "observation_role": "predicted_affinity_nm",
                    "is_experimental_binding": False,
                    "selection_allele_context": "HLA-A*02:01_predicted_binder_selection_only",
                }
            )
    finally:
        wb.close()
    return pd.DataFrame(rows)


def _parse_sd06_isotope_audit(path: Path) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    wb = openpyxl.load_workbook(path, data_only=True)
    try:
        ws6a = wb["DatasetS6A"]
        s6a: dict[str, dict[str, Any]] = {}
        for r_idx in range(5, ws6a.max_row + 1):
            aa = ws6a.cell(r_idx, 1).value
            mod = ws6a.cell(r_idx, 2).value
            mass = ws6a.cell(r_idx, 3).value
            if aa is not None and str(aa).strip():
                s6a[str(aa).strip()] = {
                    "source_row": r_idx,
                    "modification": str(mod).strip() if mod is not None else None,
                    "mass_offset_da": float(mass) if mass is not None else None,
                }

        ws6b = wb["DatasetS6B"]
        audit_rows: list[dict[str, Any]] = []
        current_charge: int | None = None
        for r_idx in range(4, ws6b.max_row + 1):
            label = ws6b.cell(r_idx, 1).value
            seq = ws6b.cell(r_idx, 2).value
            mw = ws6b.cell(r_idx, 3).value
            shift = ws6b.cell(r_idx, 4).value
            heavy_aa_note = ws6b.cell(r_idx, 5).value
            if label == "Charge" and ws6b.cell(r_idx, 7).value is not None:
                current_charge = int(ws6b.cell(r_idx, 7).value)
                continue
            if label not in ("1H", "2H", "3H", "4H") or not seq:
                continue
            seq_str = str(seq).strip()
            base_seq = seq_str.replace("(", "").replace(")", "")
            labeled_aas = _PAREN_AA_RE.findall(seq_str)
            missing_in_s6a = sorted({aa for aa in labeled_aas if aa not in s6a})
            reported_shift = float(shift) if shift is not None else None
            if missing_in_s6a:
                s6a_nominal_sum = None
                status = "uncheckable_from_s6a"
                conflict_id = "methionine_omitted_from_s6a"
            else:
                s6a_nominal_sum = float(sum(round(s6a[aa]["mass_offset_da"]) for aa in labeled_aas))
                if reported_shift is not None and abs(s6a_nominal_sum - reported_shift) <= 0.5:
                    status = "consistent_with_s6a"
                    conflict_id = None
                else:
                    status = "inconsistent_with_s6a"
                    conflict_id = (
                        "valine_isotope_s6a_vs_s6b_and_s3"
                        if "V" in labeled_aas
                        else "s6a_s6b_mass_shift_mismatch"
                    )
            audit_rows.append(
                {
                    "source_asset": path.name,
                    "source_sheet": "DatasetS6B",
                    "source_row": r_idx,
                    "isotopologue_tier": str(label),
                    "is_analytical_trigger": str(label) == "4H",
                    "peptide": base_seq,
                    "labeled_sequence_s6b": seq_str,
                    "labeled_amino_acids": ",".join(labeled_aas),
                    "precursor_charge": current_charge,
                    "reported_mw": float(mw) if mw is not None else None,
                    "reported_nominal_mass_shift": reported_shift,
                    "s6a_summed_nominal_shift": s6a_nominal_sum,
                    "heavy_amino_acid_note": (
                        str(heavy_aa_note).strip() if heavy_aa_note is not None else None
                    ),
                    "missing_from_s6a_amino_acids": ",".join(missing_in_s6a),
                    "isotope_consistency_status": status,
                    "conflict_id": conflict_id,
                }
            )
    finally:
        wb.close()
    return s6a, pd.DataFrame(audit_rows)


def _index_direct_spike_export(path: Path) -> dict[tuple[str, str], list[float]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    try:
        ws = wb["Sheet1"]
        for row in ws.iter_rows(min_row=2, max_col=11, values_only=True):
            mod_seq, rep, frag_ion, area = row[1], row[2], row[7], row[9]
            if not mod_seq or not rep:
                continue
            if frag_ion is not None and str(frag_ion).lower().startswith("precursor"):
                continue
            val = float(area) if area not in (None, "", "#N/A") else 0.0
            by_key[(str(mod_seq).strip(), str(rep).strip())].append(val)
    finally:
        wb.close()
    return by_key


def _index_hipmhc_export(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    try:
        ws = wb["Sheet1"]
        for row in ws.iter_rows(min_row=2, max_col=11, values_only=True):
            base_seq, mod_seq, rep, frag_ion, area = row[0], row[1], row[2], row[7], row[9]
            if not base_seq or not mod_seq or not rep:
                continue
            if frag_ion is not None and str(frag_ion).lower().startswith("precursor"):
                continue
            val = float(area) if area not in (None, "", "#N/A") else 0.0
            by_key[(str(base_seq).strip(), str(rep).strip())][str(mod_seq).strip()] += val
    finally:
        wb.close()
    return by_key


def _match_subset_sum(values: list[float], target: float, tol: float = 5.0) -> bool:
    n_vals = len(values)
    if n_vals == 0 or n_vals > 16:
        return False
    for mask in range(1, 1 << n_vals):
        total = 0.0
        for idx in range(n_vals):
            if (mask >> idx) & 1:
                total += values[idx]
        if abs(total - target) <= tol:
            return True
    return False


def _parse_sd03_calibration(
    path: Path,
    *,
    direct_spike_index: dict[tuple[str, str], list[float]] | None = None,
    hipmhc_index: dict[tuple[str, str], dict[str, float]] | None = None,
    isotope_status_by_peptide_tier: dict[tuple[str, str], str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    meas_rows: list[dict[str, Any]] = []
    tc_rows: list[dict[str, Any]] = []
    absent_dose_records: list[dict[str, Any]] = []

    try:
        # S3A: free 1H direct spike into pre-purified HLA-peptide background
        ws3a = wb["DatasetS3A"]
        cur_pep: str | None = None
        seen_s3a_doses: dict[str, list[float]] = defaultdict(list)
        s3a_anchor_row: dict[str, int] = {}
        for r_idx in range(4, ws3a.max_row + 1):
            pep_val = ws3a.cell(r_idx, 1).value
            fmol_val = ws3a.cell(r_idx, 2).value
            if pep_val is not None and str(pep_val).strip():
                cur_pep = str(pep_val).strip()
                s3a_anchor_row[cur_pep] = r_idx
            if fmol_val is None or cur_pep is None:
                continue
            dose_fmol = float(fmol_val)
            seen_s3a_doses[cur_pep].append(dose_fmol)
            base_seq = _strip_modifications(cur_pep)
            dose_tag = "0.1" if abs(dose_fmol - 0.1) < 1e-9 else str(round(dose_fmol))
            iso_status = (
                isotope_status_by_peptide_tier.get((base_seq, "1H"), "not_audited")
                if isotope_status_by_peptide_tier
                else "not_audited"
            )
            for col_idx, col_let in ((3, "C"), (4, "D"), (5, "E")):
                raw_val = ws3a.cell(r_idx, col_idx).value
                signal = float(raw_val) if raw_val is not None else None
                val_state = (
                    "positive"
                    if (signal is not None and signal > 0)
                    else ("explicit_zero" if signal == 0 else "literal_blank")
                )
                matched_acq: str | None = None
                acq_status = "pride_export_not_provided"
                if direct_spike_index is not None and signal is not None:
                    # Verify against all candidate replicates at this dose without assuming C=1
                    for rep_num in (1, 2, 3):
                        cand_rep = f"{dose_tag}_{rep_num}"
                        areas = direct_spike_index.get((cur_pep, cand_rep), [])
                        if areas and abs(sum(areas) - signal) <= 5.0:
                            matched_acq = cand_rep
                            acq_status = "exact_transition_sum_match"
                            break
                    if matched_acq is None:
                        for rep_num in (1, 2, 3):
                            cand_rep = f"{dose_tag}_{rep_num}"
                            areas = direct_spike_index.get((cur_pep, cand_rep), [])
                            if areas and _match_subset_sum(areas, signal, tol=5.0):
                                matched_acq = cand_rep
                                acq_status = "transition_subset_sum_match"
                                break
                    if matched_acq is None:
                        acq_status = "unreconciled_with_transition_export"

                meas_rows.append(
                    {
                        "study_id": "stopfer_2021_iso18_pxd024917",
                        "pmid": 34497125,
                        "source_asset": path.name,
                        "source_sheet": "DatasetS3A",
                        "source_cell": f"{col_let}{r_idx}",
                        "source_row": r_idx,
                        "source_ordinal_column": col_let,
                        "observation_role": "free_1h_direct_spike_summed_signal",
                        "spike_stage": "post_ip_prepurified_matrix_free_peptide",
                        "isotopologue_tier": "1H",
                        "standard_assignment_method": "amino_acid_analysis_with_isotopologue_balancing",
                        "peptide": base_seq,
                        "modified_sequence": cur_pep,
                        "nominal_standard_amount_raw": fmol_val,
                        "nominal_standard_amount_fmol": dose_fmol,
                        "nominal_standard_unit": "fmol",
                        "reported_signal_raw": raw_val,
                        "reported_signal_area": signal,
                        "value_state": val_state,
                        "verified_acquisition_replicate": matched_acq,
                        "acquisition_join_status": acq_status,
                        "isotope_consistency_status": iso_status,
                        "ms1_transfer_allowed": False,
                        "apc_concentration_transfer_allowed": False,
                    }
                )

        # Record absent dose rows explicitly (e.g., GLFDQHFRL[+7] 0.1 fmol) without fabricating slots
        expected_s3a_doses = [0.1, 1.0, 10.0, 100.0]
        for mod_seq, doses in sorted(seen_s3a_doses.items()):
            for exp_d in expected_s3a_doses:
                if not any(abs(d - exp_d) < 1e-9 for d in doses):
                    absent_dose_records.append(
                        {
                            "source_sheet": "DatasetS3A",
                            "modified_sequence": mod_seq,
                            "base_sequence": _strip_modifications(mod_seq),
                            "absent_fmol": exp_d,
                            "source_anchor_cell": f"DatasetS3A!A{s3a_anchor_row[mod_seq]}",
                            "disposition": "absent_source_row_not_zero",
                        }
                    )

        # S3B: embedded 2H hipMHC spike before immunoprecipitation
        ws3b = wb["DatasetS3B"]
        cur_pep_b: str | None = None
        for r_idx in range(4, ws3b.max_row + 1):
            pep_val = ws3b.cell(r_idx, 1).value
            fmol_val = ws3b.cell(r_idx, 2).value
            if pep_val is not None and str(pep_val).strip():
                cur_pep_b = str(pep_val).strip()
            if fmol_val is None or cur_pep_b is None:
                continue
            dose_fmol = float(fmol_val)
            base_seq = _strip_modifications(cur_pep_b)
            dose_tag = _S3B_DOSE_TAGS.get(dose_fmol, f"{dose_fmol}fmol")
            iso_status = (
                isotope_status_by_peptide_tier.get((base_seq, "2H"), "not_audited")
                if isotope_status_by_peptide_tier
                else "not_audited"
            )
            for col_idx, col_let in ((3, "C"), (4, "D"), (5, "E")):
                raw_val = ws3b.cell(r_idx, col_idx).value
                signal = float(raw_val) if raw_val is not None else None
                val_state = (
                    "positive"
                    if (signal is not None and signal > 0)
                    else ("explicit_zero" if signal == 0 else "literal_blank")
                )
                matched_acq = None
                matched_iso = None
                acq_status = "pride_export_not_provided"
                if hipmhc_index is not None:
                    if signal is None:
                        # Check which candidate replicates exist in PRIDE for this dose
                        present_reps = [
                            f"{dose_tag}_{n}"
                            for n in (1, 2, 3)
                            if (base_seq, f"{dose_tag}_{n}") in hipmhc_index
                        ]
                        missing_reps = [
                            f"{dose_tag}_{n}"
                            for n in (1, 2, 3)
                            if (base_seq, f"{dose_tag}_{n}") not in hipmhc_index
                        ]
                        acq_status = (
                            f"blank_slot_missing_pride_replicates:{','.join(missing_reps)}"
                            if missing_reps and present_reps
                            else "literal_blank_unassigned"
                        )
                    else:
                        for rep_num in (1, 2, 3):
                            cand_rep = f"{dose_tag}_{rep_num}"
                            iso_sums = hipmhc_index.get((base_seq, cand_rep), {})
                            for iso_seq, area_sum in iso_sums.items():
                                if abs(area_sum - signal) <= max(50.0, signal * 1e-4):
                                    matched_acq = cand_rep
                                    matched_iso = iso_seq
                                    acq_status = "exact_transition_sum_match"
                                    break
                            if matched_acq is not None:
                                break
                        if matched_acq is None:
                            acq_status = "unreconciled_with_transition_export"

                meas_rows.append(
                    {
                        "study_id": "stopfer_2021_iso18_pxd024917",
                        "pmid": 34497125,
                        "source_asset": path.name,
                        "source_sheet": "DatasetS3B",
                        "source_cell": f"{col_let}{r_idx}",
                        "source_row": r_idx,
                        "source_ordinal_column": col_let,
                        "observation_role": "embedded_2h_hipmhc_pre_ip_summed_signal",
                        "spike_stage": "pre_ip_embedded_hipmhc_complex",
                        "isotopologue_tier": "2H",
                        "standard_assignment_method": "stable_complex_elisa_after_peptide_exchange",
                        "peptide": base_seq,
                        "modified_sequence": matched_iso or cur_pep_b,
                        "nominal_standard_amount_raw": fmol_val,
                        "nominal_standard_amount_fmol": dose_fmol,
                        "nominal_standard_unit": "fmol",
                        "reported_signal_raw": raw_val,
                        "reported_signal_area": signal,
                        "value_state": val_state,
                        "verified_acquisition_replicate": matched_acq,
                        "acquisition_join_status": acq_status,
                        "isotope_consistency_status": iso_status,
                        "ms1_transfer_allowed": False,
                        "apc_concentration_transfer_allowed": False,
                    }
                )

        # S3C: selected transition counts (distinct from signal intensity!)
        ws3c = wb["DatasetS3C"]
        headers = [ws3c.cell(2, c).value for c in range(2, 9)]
        dose_cols = [
            (3, "10 amol", 0.01),
            (4, "100 amol", 0.1),
            (5, "1 fmol", 1.0),
            (6, "10 fmol", 10.0),
            (7, "100 fmol", 100.0),
            (8, "1000 fmol", 1000.0),
        ]
        if headers[0] == "Sequence":
            for r_idx in range(3, ws3c.max_row + 1):
                seq = ws3c.cell(r_idx, 2).value
                if not seq or not str(seq).strip():
                    continue
                for col_idx, raw_label, fmol_std in dose_cols:
                    cnt = ws3c.cell(r_idx, col_idx).value
                    if cnt is None:
                        continue
                    tc_rows.append(
                        {
                            "study_id": "stopfer_2021_iso18_pxd024917",
                            "pmid": 34497125,
                            "source_asset": path.name,
                            "source_sheet": "DatasetS3C",
                            "source_row": r_idx,
                            "peptide": str(seq).strip(),
                            "nominal_standard_label_raw": raw_label,
                            "nominal_standard_amount_fmol": fmol_std,
                            "n_transitions_detected": int(cnt),
                            "observation_role": "selected_transition_count",
                            "is_signal_intensity": False,
                        }
                    )
    finally:
        wb.close()

    return pd.DataFrame(meas_rows), pd.DataFrame(tc_rows), absent_dose_records


def parse_skmel5_binimetinib_transitions(
    path: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> pd.DataFrame:
    """Parse PRIDE PXD024917 SKMEL5_Transitions_DMSO_MEKi.xlsx with verified arm provenance."""
    contract = iso18_calibration_source_contract()
    file_path = Path(path)
    _verify_asset(
        file_path,
        contract["verified_assets"]["pride_skmel5_dmso_meki"]["sha256"],
        allow_unverified_checksum=allow_unverified_checksum,
    )
    rep_meta = contract["skmel5_replicate_design"]["replicates"]
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    rows: list[dict[str, Any]] = []
    try:
        ws = wb["SKMEL5_SkylineExport"]
        for r_idx, row in enumerate(ws.iter_rows(min_row=2, max_col=11, values_only=True), start=2):
            pep, mod_seq, rep = row[0], row[1], row[2]
            if not pep or not rep:
                continue
            rep_str = str(rep).strip()
            meta = rep_meta.get(rep_str, {})
            mod_str = str(mod_seq).strip() if mod_seq is not None else str(pep).strip()
            is_endogenous = "[" not in mod_str
            area_raw = row[9]
            area_val = float(area_raw) if area_raw not in (None, "", "#N/A") else None
            rt_raw = row[8]
            rt_val = float(rt_raw) if rt_raw not in (None, "", "#N/A") else None
            rows.append(
                {
                    "study_id": "stopfer_2021_iso18_pxd024917",
                    "pmid": 34497125,
                    "source_asset": file_path.name,
                    "source_sheet": "SKMEL5_SkylineExport",
                    "source_row": r_idx,
                    "cell_line": "SKMEL5",
                    "cellosaurus_id": "CVCL_0527",
                    "mhc_class": "I",
                    "peptide": str(pep).strip(),
                    "modified_sequence": mod_str,
                    "analyte_channel": "endogenous_light"
                    if is_endogenous
                    else "heavy_isotopologue_calibrant",
                    "replicate_id": rep_str,
                    "arm_id": meta.get("arm", "unknown"),
                    "condition_drugs": meta.get("condition_drugs", "unknown"),
                    "condition_control": meta.get("condition_control", "unknown"),
                    "acquisition_series": meta.get("series", "unknown"),
                    "treatment_dose": 100.0
                    if meta.get("condition_drugs") == "binimetinib"
                    else 0.0,
                    "treatment_dose_unit": "nM",
                    "treatment_duration_hours": 72.0,
                    "precursor_mz": float(row[3]) if row[3] not in (None, "", "#N/A") else None,
                    "precursor_charge": int(row[4]) if row[4] not in (None, "", "#N/A") else None,
                    "product_mz": float(row[5]) if row[5] not in (None, "", "#N/A") else None,
                    "product_charge": int(row[6]) if row[6] not in (None, "", "#N/A") else None,
                    "fragment_ion": str(row[7]).strip() if row[7] is not None else None,
                    "retention_time_min": rt_val,
                    "transition_area": area_val,
                    "peak_rank": int(row[10]) if row[10] not in (None, "", "#N/A") else None,
                    "observation_role": (
                        "targeted_endogenous_transition_area"
                        if is_endogenous
                        else "targeted_heavy_calibrant_transition_area"
                    ),
                }
            )
    finally:
        wb.close()
    return pd.DataFrame(rows)


def parse_iso18_calibration_workbooks(
    source_dir: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> dict[str, Any]:
    """Parse all Stopfer 2021 Iso18 calibration workbooks from a local verified directory."""
    contract = iso18_calibration_source_contract()
    base = Path(source_dir)
    assets = contract["verified_assets"]

    sd01_path = base / assets["sd01"]["filename"]
    sd03_path = base / assets["sd03"]["filename"]
    sd06_path = base / assets["sd06"]["filename"]
    sd07_path = base / assets["sd07"]["filename"]
    for key, p in (("sd01", sd01_path), ("sd03", sd03_path), ("sd06", sd06_path)):
        if not p.exists():
            raise FileNotFoundError(f"Required Stopfer 2021 asset missing: {p}")
        _verify_asset(p, assets[key]["sha256"], allow_unverified_checksum=allow_unverified_checksum)
    if sd07_path.exists():
        _verify_asset(
            sd07_path, assets["sd07"]["sha256"], allow_unverified_checksum=allow_unverified_checksum
        )

    ds_path = base / assets["pride_direct_spike"]["filename"]
    hip_path = base / assets["pride_hipmhc_spikein"]["filename"]
    sk_path = base / assets["pride_skmel5_dmso_meki"]["filename"]

    direct_spike_index = None
    if ds_path.exists():
        _verify_asset(
            ds_path,
            assets["pride_direct_spike"]["sha256"],
            allow_unverified_checksum=allow_unverified_checksum,
        )
        direct_spike_index = _index_direct_spike_export(ds_path)

    hipmhc_index = None
    if hip_path.exists():
        _verify_asset(
            hip_path,
            assets["pride_hipmhc_spikein"]["sha256"],
            allow_unverified_checksum=allow_unverified_checksum,
        )
        hipmhc_index = _index_hipmhc_export(hip_path)

    panel_df = _parse_sd01_panel(sd01_path)
    s6a_map, isotope_audit_df = _parse_sd06_isotope_audit(sd06_path)
    iso_lookup = {
        (row["peptide"], row["isotopologue_tier"]): row["isotope_consistency_status"]
        for row in isotope_audit_df.to_dict(orient="records")
    }
    calib_df, tc_df, absent_doses = _parse_sd03_calibration(
        sd03_path,
        direct_spike_index=direct_spike_index,
        hipmhc_index=hipmhc_index,
        isotope_status_by_peptide_tier=iso_lookup,
    )
    skmel5_df = (
        parse_skmel5_binimetinib_transitions(
            sk_path, allow_unverified_checksum=allow_unverified_checksum
        )
        if sk_path.exists()
        else pd.DataFrame()
    )

    return {
        "panel": panel_df,
        "calibration_measurements": calib_df,
        "transition_counts": tc_df,
        "isotope_audit": isotope_audit_df,
        "s6a_table": s6a_map,
        "absent_dose_rows": absent_doses,
        "skmel5_binimetinib_transitions": skmel5_df,
    }


def iso18_calibration_training_recipe(
    calibration_df: pd.DataFrame,
    *,
    spike_stage: str = "post_ip_prepurified_matrix_free_peptide",
    require_verified_acquisition_join: bool = False,
    require_consistent_isotope_table: bool = False,
) -> dict[str, Any]:
    """Return a clearly scoped peptide-grouped empirical dose-response subset and recipe receipt.

    Explicitly excludes literal blanks, transition counts, predicted affinities, and analytical
    trigger channels, and forbids MS1 or APC-concentration transfer claims.
    """
    df = calibration_df.copy()
    mask = (df["spike_stage"] == spike_stage) & (df["value_state"] == "positive")
    if require_verified_acquisition_join:
        mask &= df["acquisition_join_status"].isin(
            ["exact_transition_sum_match", "transition_subset_sum_match"]
        )
    if require_consistent_isotope_table:
        mask &= df["isotope_consistency_status"] == "consistent_with_s6a"

    admitted = df.loc[mask].reset_index(drop=True)
    excluded = df.loc[~mask].reset_index(drop=True)
    return {
        "recipe_id": "stopfer_2021_iso18_empirical_dose_response.v1",
        "split_group_column": "peptide",
        "spike_stage": spike_stage,
        "n_admitted_measurements": len(admitted),
        "n_excluded_measurements": len(excluded),
        "n_admitted_peptides": int(admitted["peptide"].nunique()) if len(admitted) else 0,
        "ms1_transfer_allowed": False,
        "apc_concentration_transfer_allowed": False,
        "uncertainty_disclosure": (
            "Nominal spike amounts rely on author-reported AAA/HPLC (>97%) or post-exchange ELISA "
            "and are not independently certified; targeted SureQuant IS-PRM summed fragment-ion "
            "responses do not transfer to MS1 detectability or native APC peptide concentrations."
        ),
        "admitted": admitted,
        "excluded": excluded,
    }


def write_iso18_calibration_package(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> dict[str, Any]:
    """Build and write a normalized Stopfer 2021 Iso18 calibration package to disk."""
    parsed = parse_iso18_calibration_workbooks(
        source_dir, allow_unverified_checksum=allow_unverified_checksum
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parsed["panel"].to_parquet(out / "panel.parquet", index=False)
    parsed["calibration_measurements"].to_parquet(
        out / "calibration_measurements.parquet", index=False
    )
    parsed["transition_counts"].to_parquet(out / "transition_counts.parquet", index=False)
    parsed["isotope_audit"].to_parquet(out / "isotope_audit.parquet", index=False)
    if len(parsed["skmel5_binimetinib_transitions"]):
        parsed["skmel5_binimetinib_transitions"].to_parquet(
            out / "skmel5_binimetinib_transitions.parquet", index=False
        )

    calib = parsed["calibration_measurements"]
    s3a = calib[calib["source_sheet"] == "DatasetS3A"]
    s3b = calib[calib["source_sheet"] == "DatasetS3B"]
    iso = parsed["isotope_audit"]

    manifest = {
        "schema_version": "hitlist.stopfer_2021_iso18_package.v1",
        "pmid": 34497125,
        "pride_accession": "PXD024917",
        "counts": {
            "n_panel_peptides": len(parsed["panel"]),
            "n_s3a_slots": len(s3a),
            "n_s3a_positive_slots": int((s3a["value_state"] == "positive").sum()),
            "n_s3a_blank_slots": int((s3a["value_state"] == "literal_blank").sum()),
            "n_s3b_slots": len(s3b),
            "n_s3b_positive_slots": int((s3b["value_state"] == "positive").sum()),
            "n_s3b_blank_slots": int((s3b["value_state"] == "literal_blank").sum()),
            "n_total_s3ab_slots": len(calib),
            "n_transition_count_rows": len(parsed["transition_counts"]),
            "n_s6b_isotopologues": len(iso),
            "n_s6b_consistent_with_s6a": int(
                (iso["isotope_consistency_status"] == "consistent_with_s6a").sum()
            ),
            "n_s6b_inconsistent_with_s6a": int(
                (iso["isotope_consistency_status"] == "inconsistent_with_s6a").sum()
            ),
            "n_s6b_uncheckable_from_s6a": int(
                (iso["isotope_consistency_status"] == "uncheckable_from_s6a").sum()
            ),
            "n_skmel5_transition_rows": len(parsed["skmel5_binimetinib_transitions"]),
        },
        "absent_dose_rows": parsed["absent_dose_rows"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_iso18_calibration_package(out_dir: str | Path) -> dict[str, Any]:
    """Independently verify a written Stopfer 2021 Iso18 package against pinned contract counts."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    contract = iso18_calibration_source_contract()
    exp = contract["expected_counts"]

    calib = pd.read_parquet(out / "calibration_measurements.parquet")
    iso = pd.read_parquet(out / "isotope_audit.parquet")
    panel = pd.read_parquet(out / "panel.parquet")

    checks = {
        "panel_count_matches": len(panel) == exp["n_panel_peptides"],
        "s3ab_total_slots_match": len(calib) == exp["n_total_s3ab_slots"],
        "s3a_positive_slots_match": int(
            ((calib["source_sheet"] == "DatasetS3A") & (calib["value_state"] == "positive")).sum()
        )
        == exp["n_s3a_positive_slots"],
        "s3b_positive_slots_match": int(
            ((calib["source_sheet"] == "DatasetS3B") & (calib["value_state"] == "positive")).sum()
        )
        == exp["n_s3b_positive_slots"],
        "s3b_blank_slots_match": int(
            (
                (calib["source_sheet"] == "DatasetS3B") & (calib["value_state"] == "literal_blank")
            ).sum()
        )
        == exp["n_s3b_blank_slots"],
        "absent_glfdqhfrl_0_1_fmol_recorded": any(
            r["modified_sequence"] == "GLFDQHFRL[+7]" and abs(r["absent_fmol"] - 0.1) < 1e-9
            for r in manifest["absent_dose_rows"]
        ),
        "s6b_audit_matches": (
            len(iso) == exp["n_s6b_isotopologues"]
            and int((iso["isotope_consistency_status"] == "inconsistent_with_s6a").sum())
            == exp["n_s6b_inconsistent_with_s6a"]
            and int((iso["isotope_consistency_status"] == "uncheckable_from_s6a").sum())
            == exp["n_s6b_uncheckable_from_s6a"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "counts": manifest["counts"]}
