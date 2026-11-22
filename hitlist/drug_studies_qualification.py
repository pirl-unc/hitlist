"""Source qualification and contrast ingestion for the 7 drug studies tracked in #903 / #905.

Qualifies each of the 7 named studies into either:
1. Directly recovered primary-source treatment/control quantitative observations:
   - PMID 34497125 (Stopfer 2021, binimetinib 100 nM vs DMSO in SKMEL5; PXD024917
     SKMEL5_Transitions_DMSO_MEKi.xlsx via hitlist.iso18_calibration).
   - PMID 34129938 (Pollock 2021, dTAG-13 / doxycycline factorial arms on constant 20 ng/mL
     mouse IFN-gamma background in piggyBAC idAdpgkG MC38; pinned sbpollock/NeoToma2021
     Fig4/Fig4C/toma1..3.csv targeted MS3 TMT11-plex & Fig4/Fig4E/untargeted_MS2.csv TMT6-plex).
2. Explicit, machine-readable primary-source gaps with verified PMID/PMCID/DOI/accession,
   exact compound resolution (including Jensen 2018 BET PROTACs ARV-825 / ARV-771 / JQ1-MDM2
   + JQ1 control), searched sources, and precise missing relations — zero fabricated untreated arms:
   - PMID 33981976 (Becker 2021, 5-azacytidine, PXD021755)
   - PMID 33912179 (Mifsud 2021, carbamazepine, PXD023545)
   - PMID 25576301 (Bassani-Sternberg 2015, imatinib SupB15RT chronic selection, PXD000394)
   - PMID 29632711 (Nelde 2018, lenalidomide primary CLL)
   - PMID 30524438 (Jensen 2018, BET PROTACs ARV-825/ARV-771/JQ1-MDM2 in BV173)
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .iso18_calibration import parse_skmel5_binimetinib_transitions

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "drug_studies_qualification_903.json"
)


def drug_studies_qualification_contract() -> dict[str, Any]:
    """Return the pinned #903 qualification contract across all 7 annotated drug studies."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def get_drug_study_qualification(pmid: int | str) -> dict[str, Any]:
    """Return the source qualification record for a single #903 study by PMID."""
    pmid_int = int(pmid)
    for entry in drug_studies_qualification_contract()["studies"]:
        if int(entry["pmid"]) == pmid_int:
            return entry
    raise KeyError(f"PMID {pmid_int} is not in the #903 drug studies qualification contract")


def summarize_drug_studies_qualification() -> pd.DataFrame:
    """Return a normalized summary table of all 7 #903 drug studies and their eligibility status."""
    rows: list[dict[str, Any]] = []
    for entry in drug_studies_qualification_contract()["studies"]:
        rows.append(
            {
                "pmid": int(entry["pmid"]),
                "pmcid": entry["pmcid"],
                "doi": entry["doi"],
                "study_label": entry["study_label"],
                "compounds": ";".join(entry["compounds"]),
                "compound_mechanism_class": entry["compound_mechanism_class"],
                "control_vehicle": entry["control_vehicle"],
                "repository_accessions": ";".join(entry["repository_accessions"]),
                "mhc_classes": ";".join(entry["mhc_classes"]),
                "n_audited_cache_rows": int(entry["audited_cache_rows"]),
                "cache_absence_vs_source_absence": entry["cache_absence_vs_source_absence"],
                "qualification_status": entry["qualification_status"],
                "training_eligibility": entry["training_eligibility"],
                "eligibility_reason": entry["eligibility_reason"],
                "missing_relation": entry["missing_relation"],
                "catalog_dataset": entry["catalog_dataset"],
            }
        )
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_file(path: Path, expected_sha256: str, *, allow_unverified_checksum: bool) -> str:
    actual = _sha256_file(path)
    if not allow_unverified_checksum and actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return actual


def fetch_neotoma_2021_sources(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    allow_unverified_checksum: bool = False,
) -> dict[str, Path]:
    """Fetch and SHA-256-verify pinned sbpollock/NeoToma2021 Fig4 CSVs (PMID 34129938)."""
    from .downloads import data_dir

    pin = drug_studies_qualification_contract()["neotoma_2021_source_pin"]
    target_dir = Path(dest_dir) if dest_dir is not None else data_dir() / "pollock_2021_neotoma"
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for key, meta in pin["verified_assets"].items():
        out_path = target_dir / meta["filename"]
        if force or not out_path.exists():
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            urllib.request.urlretrieve(meta["url"], tmp_path)
            tmp_path.replace(out_path)
        _verify_file(out_path, meta["sha256"], allow_unverified_checksum=allow_unverified_checksum)
        downloaded[key] = out_path
    return downloaded


def _coerce_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in ("", "-", "NA", "NaN", "nan"):
        return None
    return float(text)


def parse_neotoma_2021_tables(
    source_dir: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> dict[str, pd.DataFrame]:
    """Parse Pollock 2021 (PMID 34129938) targeted TOMAHAQ MS3 and untargeted TMT6 MS2 CSVs."""
    pin = drug_studies_qualification_contract()["neotoma_2021_source_pin"]
    base = Path(source_dir)
    tmt11_map = pin["tmt11_targeted_ms3_channels"]
    tmt6_map = pin["tmt6_untargeted_ms2_channels"]

    targeted_rows: list[dict[str, Any]] = []
    for key in ("toma1", "toma2", "toma3"):
        meta = pin["verified_assets"][key]
        path = base / meta["filename"]
        if not path.exists():
            continue
        _verify_file(path, meta["sha256"], allow_unverified_checksum=allow_unverified_checksum)
        df = pd.read_csv(path)
        for idx, rec in df.iterrows():
            peptide = str(rec.get("Peptide", "")).strip()
            if not peptide:
                continue
            protein = str(rec.get("Protein", "")).strip() or None
            sps_ions = _coerce_float(rec.get("MS3SPSIons"))
            ms3_sum_sn = _coerce_float(rec.get("MS3SumSN"))
            for col_name, ch_meta in tmt11_map.items():
                val = _coerce_float(rec.get(col_name))
                val_state = (
                    "positive"
                    if (val is not None and val > 0)
                    else ("explicit_zero" if val == 0 else "missing")
                )
                targeted_rows.append(
                    {
                        "study_id": "pollock_2021_neotoma_34129938",
                        "pmid": 34129938,
                        "source_asset": meta["repo_path"],
                        "source_row": int(idx) + 2,
                        "assay_family": "targeted_tomahaq_ms3_tmt11",
                        "cell_line": "idAdpgkG_MC38",
                        "species": "Mus musculus",
                        "mhc_class": "I",
                        "peptide": peptide,
                        "protein_label": protein,
                        "replicate_id": f"{key}_rep{rec.get('Replicate', '')}",
                        "tmt_channel": col_name,
                        "arm_id": ch_meta["arm_id"],
                        "source_arm_label": ch_meta["source_label"],
                        "condition_drugs": ch_meta["condition_drugs"],
                        "condition_control": ch_meta["condition_control"],
                        "constant_background_factors": ";".join(
                            ch_meta["constant_background_factors"]
                        ),
                        "perturbation_mechanism_class": "engineered_construct_tool",
                        "quantity_value": val,
                        "quantity_unit": "tmt_ms3_reporter_sn",
                        "value_state": val_state,
                        "ms3_sps_ions": int(sps_ions) if sps_ions is not None else None,
                        "ms3_sum_sn": ms3_sum_sn,
                        "ms2_scan_number": _coerce_float(rec.get("MS2ScanNumber")),
                        "ms3_scan_number": _coerce_float(rec.get("MS3ScanNumber")),
                        "retention_time_min": _coerce_float(rec.get("MS2RetentionTime")),
                        "eligibility_status": "eligible_engineered_construct_contrast_only",
                    }
                )

    untargeted_rows: list[dict[str, Any]] = []
    ut_meta = pin["verified_assets"]["untargeted_ms2"]
    ut_path = base / ut_meta["filename"]
    if ut_path.exists():
        _verify_file(
            ut_path, ut_meta["sha256"], allow_unverified_checksum=allow_unverified_checksum
        )
        df_ut = pd.read_csv(ut_path)
        for idx, rec in df_ut.iterrows():
            raw_pep = str(rec.get("Peptide", "")).strip()
            if not raw_pep:
                continue
            for col_name, ch_meta in tmt6_map.items():
                val = _coerce_float(rec.get(col_name))
                val_state = (
                    "positive"
                    if (val is not None and val > 0)
                    else ("explicit_zero" if val == 0 else "missing")
                )
                is_qc = ch_meta["channel_role"] == "blank_qc_channel"
                untargeted_rows.append(
                    {
                        "study_id": "pollock_2021_neotoma_34129938",
                        "pmid": 34129938,
                        "source_asset": ut_meta["repo_path"],
                        "source_row": int(idx) + 2,
                        "assay_family": "untargeted_tmt6_ms2",
                        "cell_line": "idAdpgkG_MC38",
                        "species": "Mus musculus",
                        "mhc_class": "I",
                        "peptide_modified_raw": raw_pep,
                        "protein_accession": str(rec.get("Protein Accession", "")).strip() or None,
                        "tmt_channel": col_name,
                        "channel_role": ch_meta["channel_role"],
                        "arm_id": ch_meta["arm_id"],
                        "source_arm_label": ch_meta["source_label"],
                        "condition_drugs": ch_meta["condition_drugs"],
                        "condition_control": ch_meta["condition_control"],
                        "constant_background_factors": (
                            "IFN-gamma_20ng_per_mL_51h;piggyBAC_idAdpgkG_construct"
                            if not is_qc
                            else "blank_qc_channel"
                        ),
                        "perturbation_mechanism_class": "engineered_construct_tool",
                        "quantity_value": val,
                        "quantity_unit": "tmt6_ms2_reporter_intensity",
                        "value_state": val_state,
                        "ptm_annotation": str(rec.get("PTM", "")).strip() or None,
                        "source_file": str(rec.get("Source File", "")).strip() or None,
                        "scan_number": _coerce_float(rec.get("Scan")),
                        "eligibility_status": (
                            "excluded_blank_qc_channel"
                            if is_qc
                            else "eligible_engineered_construct_contrast_only"
                        ),
                    }
                )

    return {
        "targeted_ms3": pd.DataFrame(targeted_rows),
        "untargeted_ms2": pd.DataFrame(untargeted_rows),
    }


def load_qualified_drug_contrast_records(
    *,
    skmel5_xlsx_path: str | Path | None = None,
    neotoma_source_dir: str | Path | None = None,
    allow_unverified_checksum: bool = False,
) -> pd.DataFrame:
    """Return a uniform EvidenceStore-compatible contrast table across recovered #903 sources."""
    frames: list[pd.DataFrame] = []

    if skmel5_xlsx_path is not None:
        sk = parse_skmel5_binimetinib_transitions(
            skmel5_xlsx_path, allow_unverified_checksum=allow_unverified_checksum
        )
        endo = sk[sk["analyte_channel"] == "endogenous_light"].copy()
        if len(endo):
            frames.append(
                pd.DataFrame(
                    {
                        "study_id": endo["study_id"],
                        "pmid": endo["pmid"],
                        "source_asset": endo["source_asset"],
                        "source_row": endo["source_row"],
                        "assay_family": "targeted_surequant_is_prm_transitions",
                        "cell_line": endo["cell_line"],
                        "mhc_class": endo["mhc_class"],
                        "peptide": endo["peptide"],
                        "arm_id": endo["arm_id"],
                        "condition_drugs": endo["condition_drugs"],
                        "condition_control": endo["condition_control"],
                        "constant_background_factors": "",
                        "perturbation_mechanism_class": "mek_kinase_inhibitor",
                        "replicate_id": endo["replicate_id"],
                        "quantity_value": endo["transition_area"],
                        "quantity_unit": "skyline_transition_area",
                        "eligibility_status": "eligible_targeted_contrast_panel",
                    }
                )
            )

    if neotoma_source_dir is not None:
        nt = parse_neotoma_2021_tables(
            neotoma_source_dir, allow_unverified_checksum=allow_unverified_checksum
        )
        t_df = nt["targeted_ms3"]
        if len(t_df):
            frames.append(
                pd.DataFrame(
                    {
                        "study_id": t_df["study_id"],
                        "pmid": t_df["pmid"],
                        "source_asset": t_df["source_asset"],
                        "source_row": t_df["source_row"],
                        "assay_family": t_df["assay_family"],
                        "cell_line": t_df["cell_line"],
                        "mhc_class": t_df["mhc_class"],
                        "peptide": t_df["peptide"],
                        "arm_id": t_df["arm_id"],
                        "condition_drugs": t_df["condition_drugs"],
                        "condition_control": t_df["condition_control"],
                        "constant_background_factors": t_df["constant_background_factors"],
                        "perturbation_mechanism_class": t_df["perturbation_mechanism_class"],
                        "replicate_id": t_df["replicate_id"],
                        "quantity_value": t_df["quantity_value"],
                        "quantity_unit": t_df["quantity_unit"],
                        "eligibility_status": t_df["eligibility_status"],
                    }
                )
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "study_id",
                "pmid",
                "source_asset",
                "source_row",
                "assay_family",
                "cell_line",
                "mhc_class",
                "peptide",
                "arm_id",
                "condition_drugs",
                "condition_control",
                "constant_background_factors",
                "perturbation_mechanism_class",
                "replicate_id",
                "quantity_value",
                "quantity_unit",
                "eligibility_status",
            ]
        )
    return pd.concat(frames, ignore_index=True)
