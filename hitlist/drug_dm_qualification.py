"""Source-qualified drug perturbation, HLA-DM contrast, and Atlas provenance views.

Unifies primary-source contracts and normalized query/contrast views across:
- #853: Abelin 2019 MAPTAC Class-II HLA-DM arms (PMID 31495665) & Meurer 2021 DP/DM (PMID 33025005)
- #855: HLA Ligand Atlas prediction-derived allele-association provenance (PMID 33858848)
- #901: DLBCL (PMID 35561310 / PMC9327544) Tables 3 & 5 recovery and IEDB reconciliation
- #902: Flucloxacillin (PMID 33633747 / PMC7900192) condition membership and adduct views
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .apc_contrasts import dlbcl_treatment_conflict_contract
from .provenance import file_digest

_DATA_DIR = Path(__file__).parent / "data"
_REVISIONS_DIR = _DATA_DIR / "evidence_revisions"
_SUPP_DIR = _DATA_DIR / "supplementary"

_ATLAS_PROVENANCE_PATH = _REVISIONS_DIR / "hla_ligand_atlas_33858848_prediction_provenance.json"
_FLX_ADDUCT_PATH = _REVISIONS_DIR / "flucloxacillin_33633747_adduct_contract.json"
_DLBCL_SUPPL_PATH = _REVISIONS_DIR / "dlbcl_35561310_supplements.json"
_ABELIN_DM_SOURCE_PATH = _SUPP_DIR / "abelin_2019_maptac_class2_source.yaml"


def atlas_prediction_provenance_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned HLA Ligand Atlas (#855) provenance contract."""
    return json.loads(_ATLAS_PROVENANCE_PATH.read_text(encoding="utf-8"))


def flucloxacillin_adduct_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned flucloxacillin (#902) adduct/condition contract."""
    return json.loads(_FLX_ADDUCT_PATH.read_text(encoding="utf-8"))


def dlbcl_supplement_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned DLBCL (#901) Table 3/5 supplement contract."""
    return json.loads(_DLBCL_SUPPL_PATH.read_text(encoding="utf-8"))


def abelin_dm_source_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned Abelin 2019 MAPTAC Class-II ±DM (#853) source receipt."""
    return yaml.safe_load(_ABELIN_DM_SOURCE_PATH.read_text(encoding="utf-8"))


def _string_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[name].astype("string").fillna("")


def _classify_flx_modification(
    peptide: str,
    mods_raw: str,
    assay_comments: str,
    confirmed_sites: set[tuple[str, str]],
) -> tuple[str, str, str]:
    """Return (adduct_class, adduct_site, localization_status) without guessing sites."""
    mods = mods_raw.strip()
    is_filler = assay_comments.strip() == "Filler Assay - delete."
    if not mods:
        return "unmodified", "", "not_applicable"
    tokens = [tok.strip() for tok in mods.split(";") if tok.strip()]
    mcm_tokens = [tok for tok in tokens if tok.upper().startswith("MCM")]
    if mcm_tokens:
        site_list = [tok.split(":", 1)[1] if ":" in tok else "" for tok in mcm_tokens]
        site_str = ";".join(s for s in site_list if s)
        if is_filler:
            return "flx_filler_unconfirmed_site", site_str, "filler_assay_unconfirmed_site"
        if any((peptide, tok) in confirmed_sites for tok in mcm_tokens):
            return (
                "flx_covalent_hapten_confirmed",
                site_str,
                "primary_table_1_confirmed_msms_and_synthetic_standard",
            )
        if site_str:
            return "flx_covalent_hapten_unverified_site", site_str, "site_reported_not_table1"
        return "flx_covalent_hapten_unknown_site", "", "unknown_localization"
    # Endogenous PTMs (e.g., OX:M3, CITR:R4, CYSTL:C5)
    sites = [tok.split(":", 1)[1] if ":" in tok else "" for tok in tokens]
    site_str = ";".join(s for s in sites if s)
    loc = "reported_site" if site_str else "unknown_localization"
    return "endogenous_ptm", site_str, loc


def prepare_flucloxacillin_evidence_views(observations: pd.DataFrame) -> dict[str, Any]:
    """Partition PMID 33633747 observations into unmodified-only and modification-aware views (#902).

    Preserves unmodified sequences, original modification strings, confirmed FLX
    covalent hapten adduct sites, alternative-site IEDB filler assays, and shared
    treated/untreated memberships without collapsing distinct molecular forms.
    """
    contract = flucloxacillin_adduct_contract()
    confirmed_sites = {
        (item["unmodified_sequence"], item["iedb_modification_code"])
        for item in contract["confirmed_flx_covalent_adducts"]
    }

    if "pmid" in observations.columns:
        pmid_num = pd.to_numeric(observations["pmid"], errors="coerce")
        sub = observations.loc[pmid_num.eq(33633747)].copy()
    else:
        sub = observations.copy()

    peptides = _string_col(sub, "peptide").str.strip()
    mods = _string_col(sub, "peptide_modifications").str.strip()
    assay_comments = _string_col(sub, "assay_comments").str.strip()
    cond_id = _string_col(sub, "condition_id").str.strip()
    supported = _string_col(sub, "supported_condition_ids").str.strip()

    adduct_classes: list[str] = []
    adduct_sites: list[str] = []
    loc_statuses: list[str] = []
    is_filler_flags: list[bool] = []
    membership_roles: list[str] = []

    for pep, mod, comment, cid, sup in zip(
        peptides, mods, assay_comments, cond_id, supported, strict=True
    ):
        aclass, asite, lstat = _classify_flx_modification(
            str(pep), str(mod), str(comment), confirmed_sites
        )
        adduct_classes.append(aclass)
        adduct_sites.append(asite)
        loc_statuses.append(lstat)
        filler = str(comment) == "Filler Assay - delete."
        is_filler_flags.append(filler)

        if filler:
            membership_roles.append("filler_assay_excluded")
        elif str(cid) == "cells_flucloxacillin":
            membership_roles.append("flx_only")
        elif str(cid) == "cells_untreated":
            membership_roles.append("untreated_only")
        elif "cells_flucloxacillin" in str(sup) and "cells_untreated" in str(sup):
            membership_roles.append("shared_both_arms")
        else:
            membership_roles.append("unattributed")

    sub["molecular_form_key"] = peptides + "|" + mods
    sub["adduct_class"] = pd.Series(adduct_classes, index=sub.index, dtype="string")
    sub["adduct_site"] = pd.Series(adduct_sites, index=sub.index, dtype="string")
    sub["localization_status"] = pd.Series(loc_statuses, index=sub.index, dtype="string")
    sub["is_filler_assay"] = pd.Series(is_filler_flags, index=sub.index, dtype="bool")
    sub["condition_membership_role"] = pd.Series(membership_roles, index=sub.index, dtype="string")
    sub["assay_scope"] = "ms_ligand_elution"

    unmodified_mask = sub["adduct_class"].eq("unmodified") & ~sub["is_filler_assay"]
    unmodified_view = sub.loc[unmodified_mask].reset_index(drop=True)
    mod_aware_view = sub.reset_index(drop=True)

    exclusion_reasons = sub.loc[~unmodified_mask, "adduct_class"].value_counts().to_dict()
    audit = {
        "pmid": 33633747,
        "n_input_rows": len(sub),
        "n_unmodified_training_rows": len(unmodified_view),
        "n_excluded_from_unmodified_training": int((~unmodified_mask).sum()),
        "n_modification_aware_rows": len(mod_aware_view),
        "n_distinct_molecular_forms": int(sub["molecular_form_key"].nunique()),
        "n_distinct_unmodified_sequences": int(peptides.nunique()),
        "adduct_class_counts": {
            str(k): int(v) for k, v in sub["adduct_class"].value_counts().items()
        },
        "membership_role_counts": {
            str(k): int(v) for k, v in sub["condition_membership_role"].value_counts().items()
        },
        "unmodified_exclusion_counts": {str(k): int(v) for k, v in exclusion_reasons.items()},
        "row_conservation_ok": len(unmodified_view) + int((~unmodified_mask).sum()) == len(sub),
    }
    return {
        "contract": contract,
        "unmodified_training_view": unmodified_view,
        "modification_aware_view": mod_aware_view,
        "audit": audit,
    }


def load_dlbcl_supplement_rows(
    suppl3_path: str | Path,
    suppl5_path: str | Path,
    *,
    verify_checksums: bool = True,
) -> pd.DataFrame:
    """Parse checksum-pinned PMC9327544 Tables 3 (Class I) and 5 (Class II) (#901).

    Preserves native rows (including repeated forms in SUDHL6_IFN_TAZ), literal
    source header variants (``lengtg``, ``Query #:z``), NetMHCpan 4.0 prediction
    provenance (never mislabeled experimental restriction), explicit missing
    intensity/replicate status, and links to the #930 dose-unit conflict ledger.
    """
    import openpyxl

    contract = dlbcl_supplement_contract()
    conflicts = dlbcl_treatment_conflict_contract()
    sheet_map = contract["sheet_to_condition_map"]

    workbook_specs = [
        ("table_3_class_i", Path(suppl3_path), "I", "predicted"),
        ("table_5_class_ii", Path(suppl5_path), "II", "unknown"),
    ]
    rows: list[dict[str, Any]] = []

    for wb_key, path, mhc_class, rest_ev in workbook_specs:
        meta = contract["workbooks"][wb_key]
        if verify_checksums:
            digest = file_digest(path)
            if digest["sha256"] != meta["sha256"] or digest["size_bytes"] != meta["size_bytes"]:
                raise ValueError(
                    f"DLBCL supplement checksum/size mismatch for {meta['filename']}: "
                    f"expected sha256={meta['sha256']} size={meta['size_bytes']}, got {digest}"
                )

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet_name in wb.sheetnames:
                if sheet_name not in sheet_map:
                    raise ValueError(f"Unexpected DLBCL worksheet {sheet_name!r} in {path.name}")
                arm = sheet_map[sheet_name]
                ws = wb[sheet_name]
                all_rows = list(ws.iter_rows(values_only=True))
                if not all_rows:
                    continue
                raw_headers = [
                    "" if cell is None else str(cell).replace("\n", " ").strip()
                    for cell in all_rows[0]
                ]
                first_col_header = raw_headers[0] if raw_headers else ""
                hla_cols = [
                    (idx, h)
                    for idx, h in enumerate(raw_headers)
                    if h.startswith("HLA-") and mhc_class == "I"
                ]
                for excel_row, values in enumerate(all_rows[1:], start=2):
                    if len(values) < 3 or values[2] is None:
                        continue
                    unmod = str(values[2]).strip()
                    if not unmod:
                        continue
                    mod_raw = "" if values[1] is None else str(values[1]).strip()
                    mod_types = (
                        "" if len(values) < 4 or values[3] is None else str(values[3]).strip()
                    )
                    pred_scores = {}
                    for idx, allele in hla_cols:
                        if idx < len(values) and values[idx] is not None:
                            try:
                                val = float(values[idx])
                                if math.isfinite(val):
                                    pred_scores[allele] = val
                            except (TypeError, ValueError):
                                pass
                    rows.append(
                        {
                            "pmid": 35561310,
                            "source_workbook": meta["filename"],
                            "source_workbook_sha256": meta["sha256"],
                            "source_sheet": sheet_name,
                            "source_row": excel_row,
                            "first_col_header_literal": first_col_header,
                            "peptide": unmod,
                            "modified_sequence_raw": mod_raw,
                            "modification_types_raw": mod_types,
                            "mhc_class": mhc_class,
                            "cell_line_name": arm["cell_line"],
                            "condition_id": arm["condition_id"],
                            "comparator_condition_id": arm["comparator_condition_id"] or "",
                            "contrast_role": arm["contrast_role"],
                            "background": arm["background"],
                            "restriction_evidence": rest_ev,
                            "prediction_method": "NetMHCpan 4.0" if mhc_class == "I" else "",
                            "netmhcpan_4_0_scores_json": json.dumps(pred_scores, sort_keys=True)
                            if pred_scores
                            else "",
                            "quantitative_value": float("nan"),
                            "quantity_availability": contract["quantity_availability"]["status"],
                            "has_source_dose_unit_conflict": True,
                            "dose_conflict_issue": conflicts["issue"],
                        }
                    )
        finally:
            wb.close()

    frame = pd.DataFrame(rows)
    if verify_checksums:
        expected_total = (
            contract["workbooks"]["table_3_class_i"]["n_rows"]
            + contract["workbooks"]["table_5_class_ii"]["n_rows"]
        )
        if len(frame) != expected_total:
            raise ValueError(
                f"DLBCL supplement row count mismatch: expected {expected_total}, got {len(frame)}"
            )
    return frame


def reconcile_dlbcl_supplements_with_iedb(
    supplement_rows: pd.DataFrame,
    iedb_observations: pd.DataFrame,
) -> dict[str, Any]:
    """Reconcile PMC9327544 Table 3/5 rows against IEDB PMID 35561310 rows (#901)."""
    if "pmid" in iedb_observations.columns:
        pmid_num = pd.to_numeric(iedb_observations["pmid"], errors="coerce")
        iedb = iedb_observations.loc[pmid_num.eq(35561310)].copy()
    else:
        iedb = iedb_observations.copy()

    iedb_pep = _string_col(iedb, "peptide").str.strip()
    iedb_cls = _string_col(iedb, "mhc_class").str.strip()
    iedb_cid = _string_col(iedb, "condition_id").str.strip()
    iedb_grp = _string_col(iedb, "sample_group").str.strip()

    supp_pep = _string_col(supplement_rows, "peptide").str.strip()
    supp_cls = _string_col(supplement_rows, "mhc_class").str.strip()

    per_arm: list[dict[str, Any]] = []
    for (mhc_cls, cid), grp in supplement_rows.groupby(["mhc_class", "condition_id"], sort=True):
        s_peps = set(grp["peptide"].astype("string").str.strip())
        i_mask = iedb_cls.eq(mhc_cls) & iedb_cid.eq(cid)
        i_peps = set(iedb_pep.loc[i_mask])
        per_arm.append(
            {
                "mhc_class": str(mhc_cls),
                "condition_id": str(cid),
                "cell_line_name": str(grp["cell_line_name"].iloc[0]),
                "comparator_condition_id": str(grp["comparator_condition_id"].iloc[0]),
                "contrast_role": str(grp["contrast_role"].iloc[0]),
                "n_supplement_rows": len(grp),
                "n_supplement_distinct_peptides": len(s_peps),
                "n_iedb_attributed_rows": int(i_mask.sum()),
                "n_iedb_distinct_peptides": len(i_peps),
                "n_overlap_peptides": len(s_peps & i_peps),
                "n_supplement_only_peptides": len(s_peps - i_peps),
                "n_iedb_only_peptides": len(i_peps - s_peps),
                "recovers_missing_iedb_control": bool(
                    cid.endswith("_dmso") and len(s_peps) > 0 and len(i_peps) == 0
                ),
            }
        )

    n_iedb_ambiguous = int(iedb_cid.eq("").sum())
    return {
        "pmid": 35561310,
        "n_supplement_rows": len(supplement_rows),
        "n_supplement_class_i_rows": int(supp_cls.eq("I").sum()),
        "n_supplement_class_ii_rows": int(supp_cls.eq("II").sum()),
        "n_supplement_distinct_peptides": int(supp_pep.nunique()),
        "n_iedb_rows": len(iedb),
        "n_iedb_attributed_rows": int(iedb_cid.ne("").sum()),
        "n_iedb_ambiguous_rows": n_iedb_ambiguous,
        "n_iedb_groups": int(iedb_grp[iedb_grp.ne("")].nunique()),
        "quantity_availability": dlbcl_supplement_contract()["quantity_availability"]["status"],
        "per_arm": pd.DataFrame(per_arm),
    }


# Paired HLA-DM contrast definitions converging Abelin 2019 (31495665) & Meurer 2021 (33025005)
_HLA_DM_PAIRED_SYSTEMS: dict[tuple[int, str], dict[str, str]] = {
    # Abelin 2019 MAPTAC Class-II (#853): dm- = no added HLA-DM co-transfection
    (31495665, "maptac_hla_drb1_12_01_dm"): {
        "pair_id": "abelin2019:HLA-DRB1*12:01",
        "mhc_pair": "HLA-DRB1*12:01",
        "dm_arm": "dm_minus",
        "partner_condition_id": "maptac_hla_drb1_12_01_dm_2",
        "dm_control_basis": "no_added_hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_drb1_12_01_dm_2"): {
        "pair_id": "abelin2019:HLA-DRB1*12:01",
        "mhc_pair": "HLA-DRB1*12:01",
        "dm_arm": "dm_plus",
        "partner_condition_id": "maptac_hla_drb1_12_01_dm",
        "dm_control_basis": "hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_drb3_01_01_dm"): {
        "pair_id": "abelin2019:HLA-DRB3*01:01",
        "mhc_pair": "HLA-DRB3*01:01",
        "dm_arm": "dm_minus",
        "partner_condition_id": "maptac_hla_drb3_01_01_dm_2",
        "dm_control_basis": "no_added_hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_drb3_01_01_dm_2"): {
        "pair_id": "abelin2019:HLA-DRB3*01:01",
        "mhc_pair": "HLA-DRB3*01:01",
        "dm_arm": "dm_plus",
        "partner_condition_id": "maptac_hla_drb3_01_01_dm",
        "dm_control_basis": "hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_dpb1_06_01_dpa1_01_03_dm"): {
        "pair_id": "abelin2019:HLA-DPA1*01:03/DPB1*06:01",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*06:01",
        "dm_arm": "dm_minus",
        "partner_condition_id": "maptac_hla_dpb1_06_01_dpa1_01_03_dm_2",
        "dm_control_basis": "no_added_hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_dpb1_06_01_dpa1_01_03_dm_2"): {
        "pair_id": "abelin2019:HLA-DPA1*01:03/DPB1*06:01",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*06:01",
        "dm_arm": "dm_plus",
        "partner_condition_id": "maptac_hla_dpb1_06_01_dpa1_01_03_dm",
        "dm_control_basis": "hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_dqb1_06_04_dqa1_01_02_dm"): {
        "pair_id": "abelin2019:HLA-DQA1*01:02/DQB1*06:04",
        "mhc_pair": "HLA-DQA1*01:02/DQB1*06:04",
        "dm_arm": "dm_minus",
        "partner_condition_id": "maptac_hla_dqb1_06_04_dqa1_01_02_dm_2",
        "dm_control_basis": "no_added_hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    (31495665, "maptac_hla_dqb1_06_04_dqa1_01_02_dm_2"): {
        "pair_id": "abelin2019:HLA-DQA1*01:02/DQB1*06:04",
        "mhc_pair": "HLA-DQA1*01:02/DQB1*06:04",
        "dm_arm": "dm_plus",
        "partner_condition_id": "maptac_hla_dqb1_06_04_dqa1_01_02_dm",
        "dm_control_basis": "hla_dm_cotransfection",
        "endogenous_dm_status": "unknown",
        "hla_construct_background": "MAPTAC_tagged_HLA_transfection",
    },
    # Meurer 2021 DP/DM HeLa (#895): paired DP10 and DP402 arms (DP401 is DM-only)
    (33025005, "dp10_a"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*10:01",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*10:01",
        "dm_arm": "dm_minus",
        "partner_condition_id": "dp10_dm_a;dp10_dm_b",
        "dm_control_basis": "no_added_hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp10_b"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*10:01",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*10:01",
        "dm_arm": "dm_minus",
        "partner_condition_id": "dp10_dm_a;dp10_dm_b",
        "dm_control_basis": "no_added_hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp10_dm_a"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*10:01",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*10:01",
        "dm_arm": "dm_plus",
        "partner_condition_id": "dp10_a;dp10_b",
        "dm_control_basis": "hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp10_dm_b"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*10:01",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*10:01",
        "dm_arm": "dm_plus",
        "partner_condition_id": "dp10_a;dp10_b",
        "dm_control_basis": "hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp402_a"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*04:02",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*04:02",
        "dm_arm": "dm_minus",
        "partner_condition_id": "dp402_dm_a;dp402_dm_b",
        "dm_control_basis": "no_added_hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp402_b"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*04:02",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*04:02",
        "dm_arm": "dm_minus",
        "partner_condition_id": "dp402_dm_a;dp402_dm_b",
        "dm_control_basis": "no_added_hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp402_dm_a"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*04:02",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*04:02",
        "dm_arm": "dm_plus",
        "partner_condition_id": "dp402_a;dp402_b",
        "dm_control_basis": "hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
    (33025005, "dp402_dm_b"): {
        "pair_id": "meurer2021:HLA-DPA1*01:03/DPB1*04:02",
        "mhc_pair": "HLA-DPA1*01:03/DPB1*04:02",
        "dm_arm": "dm_plus",
        "partner_condition_id": "dp402_a;dp402_b",
        "dm_control_basis": "hla_dm_transduction",
        "endogenous_dm_status": "hela_baseline_untransduced",
        "hla_construct_background": "CD74_transduced;CD80_transduced;HLA-DP_transduced",
    },
}


def prepare_hla_dm_contrast_data(observations: pd.DataFrame) -> dict[str, Any]:
    """Build a unified paired HLA-DM contrast dataset across Abelin 2019 & Meurer 2021 (#853).

    Converges Abelin 2019 MAPTAC (PMID 31495665) and Meurer 2021 DP/DM (PMID 33025005)
    on the same material/condition contrast records, keeping single-arm DM-only studies
    (such as Stražar 2023 PMID 37301199 or Meurer DP401_DM) and unattributed rows
    explicitly excluded with reasons rather than fabricating cross-study controls.
    """
    pmids = pd.to_numeric(
        observations.get("pmid", pd.Series(pd.NA, index=observations.index)), errors="coerce"
    )
    cids = _string_col(observations, "condition_id").str.strip()
    mhc_cls = _string_col(observations, "mhc_class").str.strip()

    admitted_rows: list[int] = []
    admitted_meta: list[dict[str, str]] = []
    excluded_rows: list[dict[str, Any]] = []

    for pos, (pmid_val, cid_val, cls_val) in enumerate(zip(pmids, cids, mhc_cls, strict=True)):
        pmid_int = int(pmid_val) if pd.notna(pmid_val) else -1
        key = (pmid_int, str(cid_val))
        if cls_val != "II":
            excluded_rows.append({"row_pos": pos, "pmid": pmid_int, "reason": "not_class_ii"})
            continue
        if not str(cid_val):
            excluded_rows.append(
                {"row_pos": pos, "pmid": pmid_int, "reason": "unattributed_condition_id"}
            )
            continue
        if key in _HLA_DM_PAIRED_SYSTEMS:
            admitted_rows.append(pos)
            admitted_meta.append(_HLA_DM_PAIRED_SYSTEMS[key])
        elif pmid_int == 37301199:
            excluded_rows.append(
                {"row_pos": pos, "pmid": pmid_int, "reason": "dm_positive_only_no_paired_control"}
            )
        else:
            excluded_rows.append(
                {"row_pos": pos, "pmid": pmid_int, "reason": "not_paired_hla_dm_arm"}
            )

    admitted = observations.iloc[admitted_rows].copy().reset_index(drop=True)
    if admitted_meta:
        meta_df = pd.DataFrame(admitted_meta)
        for col in meta_df.columns:
            admitted[col] = meta_df[col].astype("string")
    else:
        for col in (
            "pair_id",
            "mhc_pair",
            "dm_arm",
            "partner_condition_id",
            "dm_control_basis",
            "endogenous_dm_status",
            "hla_construct_background",
        ):
            admitted[col] = pd.Series(dtype="string")

    excluded_df = pd.DataFrame(excluded_rows)
    excl_counts = (
        {str(k): int(v) for k, v in excluded_df["reason"].value_counts().items()}
        if not excluded_df.empty
        else {}
    )
    pair_summary = []
    if not admitted.empty:
        for pair_id, grp in admitted.groupby("pair_id", sort=True):
            dm_minus = grp[grp["dm_arm"].eq("dm_minus")]
            dm_plus = grp[grp["dm_arm"].eq("dm_plus")]
            s_m = set(dm_minus["peptide"].astype("string").str.strip())
            s_p = set(dm_plus["peptide"].astype("string").str.strip())
            pair_summary.append(
                {
                    "pair_id": str(pair_id),
                    "mhc_pair": str(grp["mhc_pair"].iloc[0]),
                    "n_dm_minus_rows": len(dm_minus),
                    "n_dm_plus_rows": len(dm_plus),
                    "n_shared_peptides": len(s_m & s_p),
                    "has_both_arms": bool(len(dm_minus) > 0 and len(dm_plus) > 0),
                }
            )

    return {
        "admitted": admitted,
        "excluded_counts": excl_counts,
        "pair_summary": pd.DataFrame(pair_summary),
        "n_input_rows": len(observations),
        "n_admitted_rows": len(admitted),
        "n_excluded_rows": len(excluded_rows),
        "row_conservation_ok": len(admitted) + len(excluded_rows) == len(observations),
    }
