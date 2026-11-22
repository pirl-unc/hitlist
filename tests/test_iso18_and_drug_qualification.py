"""Tests for Stopfer 2021 Iso18 calibration (#849) and 7 drug studies qualification (#903)."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
import pytest

import hitlist
from hitlist.drug_studies_qualification import (
    drug_studies_qualification_contract,
    get_drug_study_qualification,
    load_qualified_drug_contrast_records,
    parse_neotoma_2021_tables,
    summarize_drug_studies_qualification,
)
from hitlist.iso18_calibration import (
    iso18_calibration_source_contract,
    iso18_calibration_training_recipe,
    parse_iso18_calibration_workbooks,
    parse_skmel5_binimetinib_transitions,
    write_iso18_calibration_package,
)


def _build_synthetic_stopfer_dir(root: Path) -> Path:
    """Create minimal synthetic Stopfer 2021 workbooks mirroring exact schema & edge cases."""
    root.mkdir(parents=True, exist_ok=True)

    # sd01: Panel + predicted affinity (NOT experimental binding)
    wb1 = openpyxl.Workbook()
    ws1 = wb1.active
    ws1.title = "DatasetS1"
    ws1.append([None, "Iso18 panel"])
    ws1.append([None])
    ws1.append([None, "Peptide", "Gene", "Annotated Gene", "Length", "Predicted affinity [nM]"])
    ws1.append([None, "GLFDQHFRL", "PMEL", "PMEL_1", 9, 12.4])
    ws1.append([None, "RLLGTEFQV", "PMEL", "PMEL_2", 9, 4.1])
    ws1.append([None, "AMLGTHTMEV", "PMEL", "PMEL_3", 10, 8.7])
    wb1.save(root / "pnas.2111173118.sd01.xlsx")

    # sd03: S3A (absent 0.1 fmol for GLFDQHFRL[+7]), S3B (100 fmol column E blank), S3C (counts)
    wb3 = openpyxl.Workbook()
    ws3a = wb3.active
    ws3a.title = "DatasetS3A"
    ws3a.append(["S3A free 1H direct spike"])
    ws3a.append([None])
    ws3a.append(["Peptide", "fmol", "Summed intensity"])
    # GLFDQHFRL[+7] intentionally lacks 0.1 fmol row!
    ws3a.append(["GLFDQHFRL[+7]", 1, 1000.0, 1100.0, 1200.0])
    ws3a.append([None, 10, 10000.0, 11000.0, 12000.0])
    ws3a.append([None, 100, 100000.0, 110000.0, 120000.0])
    # RLLGTEFQV[+6] has all 4 doses including explicit zero test at 0.1 col E
    ws3a.append(["RLLGTEFQV[+6]", 0.1, 500.0, 600.0, 0.0])
    ws3a.append([None, 1, 5000.0, 6000.0, 7000.0])
    ws3a.append([None, 10, 50000.0, 60000.0, 70000.0])
    ws3a.append([None, 100, 500000.0, 600000.0, 700000.0])

    ws3b = wb3.create_sheet("DatasetS3B")
    ws3b.append(["S3B embedded 2H hipMHC"])
    ws3b.append([None])
    ws3b.append(["Peptide", "Fmol", "Summed Intensity"])
    ws3b.append(["RLLGTEFQV", 100, 171991.0, 78936.0, None])

    ws3c = wb3.create_sheet("DatasetS3C")
    ws3c.append([None, "Selected transition counts"])
    ws3c.append(
        [None, "Sequence", "10 amol", "100 amol", "1 fmol", "10 fmol", "100 fmol", "1000 fmol"]
    )
    ws3c.append([None, "RLLGTEFQV", 5, 6, 6, 6, 6, 6])
    wb3.save(root / "pnas.2111173118.sd03.xlsx")

    # sd06: S6A (Valine 4.007 Da, Methionine absent) + S6B (V[+6] conflict + M omitted)
    wb6 = openpyxl.Workbook()
    ws6a = wb6.active
    ws6a.title = "DatasetS6A"
    ws6a.append([None])
    ws6a.append(["SIL amino acids"])
    ws6a.append([None])
    ws6a.append(["Amino acid (s)", "Modification", "Mass offset (Da)"])
    ws6a.append(["L", "13C6, 15N1", 7.017])
    ws6a.append(["F", "13C9, 15N1", 10.027])
    ws6a.append(["V", "13C3, 15N1", 4.007])
    ws6a.append(["A", "13C3, 15N1", 4.007])

    ws6b = wb6.create_sheet("DatasetS6B")
    ws6b.append(["Isotopologue definitions"])
    ws6b.append([None])
    ws6b.append(
        [
            "Peptide Name",
            "Sequence",
            "MW",
            "Mass shift",
            "Heavy Amino acid",
            "MW",
            "Mass Offset",
            "m/z offset",
        ]
    )
    ws6b.append(["Charge", None, None, None, None, None, 2, None])
    ws6b.append(["1H", "GLFDQHFR(L)", 1140.5, 7, "L: 13C6, 15N1", 570.7, 10.0, 5.0])
    ws6b.append(["1H", "RLLGTEFQ(V)", 1066.5, 6, "V: 13C3, 15N1", 533.7, 10.0, 5.0])
    ws6b.append(["2H", "RLLGTE(F)Q(V)", 1076.5, 16, "F: 13C9, 15N1", 538.7, 6.0, 3.0])
    ws6b.append(["4H", "(A)M(L)GTHT(M)E(V)", 1111.49, 23, "M: 13C5, 15N1", 556.245, None, None])
    wb6.save(root / "pnas.2111173118.sd06.xlsx")

    # PRIDE DirectSpike export (testing subset match and exact match)
    wb_ds = openpyxl.Workbook()
    ws_ds = wb_ds.active
    ws_ds.title = "Sheet1"
    ws_ds.append(
        [
            "Peptide",
            "Modified Sequence",
            "Replicate",
            "Precursor Mz",
            "Precursor Charge",
            "Product Mz",
            "Product Charge",
            "Fragment Ion",
            "Retention Time",
            "Area",
            "Peak Rank",
        ]
    )
    # 1_1 has 2 transitions: 1000 + 250; S3A col C = 1000 -> transition_subset_sum_match!
    ws_ds.append(["GLFDQHFRL", "GLFDQHFRL[+7]", "1_1", 570.0, 2, 700.0, 1, "y6", 35.0, 1000.0, 1])
    ws_ds.append(["GLFDQHFRL", "GLFDQHFRL[+7]", "1_1", 570.0, 2, 500.0, 1, "y4", 35.0, 250.0, 2])
    # 1_2 exact sum = 1100
    ws_ds.append(["GLFDQHFRL", "GLFDQHFRL[+7]", "1_2", 570.0, 2, 700.0, 1, "y6", 35.0, 1100.0, 1])
    wb_ds.save(root / "Iso18_DirectSpike_TransitionResults.xlsx")

    # PRIDE hipMHC export: 100fmol_1 and 100fmol_3 exist (100fmol_2 is ABSENT!)
    wb_hip = openpyxl.Workbook()
    ws_hip = wb_hip.active
    ws_hip.title = "Sheet1"
    ws_hip.append(
        [
            "Peptide",
            "Modified Sequence",
            "Replicate",
            "Precursor Mz",
            "Precursor Charge",
            "Product Mz",
            "Product Charge",
            "Fragment Ion",
            "Retention Time",
            "Area",
            "Peak Rank",
        ]
    )
    ws_hip.append(
        [
            "RLLGTEFQV",
            "RLLGTEF[+10]QV[+6]",
            "100fmol_1",
            538.7,
            2,
            700.0,
            1,
            "y6",
            50.0,
            171991.0,
            1,
        ]
    )
    ws_hip.append(
        ["RLLGTEFQV", "RLLGTEF[+10]QV[+6]", "100fmol_3", 538.7, 2, 700.0, 1, "y6", 50.0, 78936.0, 1]
    )
    wb_hip.save(root / "Iso18_hipMHCspikein_TransitionResults.xlsx")

    # PRIDE SKMEL5 binimetinib vs DMSO export
    wb_sk = openpyxl.Workbook()
    ws_sk = wb_sk.active
    ws_sk.title = "SKMEL5_SkylineExport"
    ws_sk.append(
        [
            "Peptide",
            "Modified Sequence",
            "Replicate",
            "Precursor Mz",
            "Precursor Charge",
            "Product Mz",
            "Product Charge",
            "Fragment Ion",
            "Retention Time",
            "Area",
            "Peak Rank",
        ]
    )
    ws_sk.append(
        ["SLDDYNHLV", "SLDDYNHLV", "SKMEL5_DMSO1", 538.0, 2, 881.4, 1, "y7", 55.8, 12000.0, 1]
    )
    ws_sk.append(
        ["SLDDYNHLV", "SLDDYNHLV", "SKMEL5_100nM1", 538.0, 2, 881.4, 1, "y7", 55.9, 36000.0, 1]
    )
    ws_sk.append(
        ["SLDDYNHLV", "SLDDYNHLV[+6]", "SKMEL5_100nM1", 541.2, 2, 881.4, 1, "y7", 55.9, 276379.0, 2]
    )
    wb_sk.save(root / "SKMEL5_Transitions_DMSO_MEKi.xlsx")

    return root


def test_iso18_contract_and_corrupted_checksum_rejection(tmp_path: Path) -> None:
    contract = iso18_calibration_source_contract()
    assert contract["pmid"] == 34497125
    assert contract["pride_accession"] == "PXD024917"
    assert contract["expected_counts"]["n_total_s3ab_slots"] == 348
    assert (
        contract["verified_assets"]["pride_hipmhc_spikein"]["s7_literal_filename"]
        == "Iso18_hipMHCspiikein_TransitionResults.xlsx"
    )
    assert contract["analytical_boundaries"]["ms1_transfer_allowed"] is False
    assert contract["analytical_boundaries"]["apc_concentration_transfer_allowed"] is False

    src_dir = _build_synthetic_stopfer_dir(tmp_path / "stopfer")
    # Strict checksum verification MUST reject synthetic/corrupted files by default
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_iso18_calibration_workbooks(src_dir, allow_unverified_checksum=False)


def test_iso18_parsing_conflicts_missing_vs_zero_and_column_lineage(tmp_path: Path) -> None:
    src_dir = _build_synthetic_stopfer_dir(tmp_path / "stopfer")
    parsed = parse_iso18_calibration_workbooks(src_dir, allow_unverified_checksum=True)

    # 1. Absent 0.1 fmol row for GLFDQHFRL[+7] is recorded separately, NEVER turned into zero
    absent = parsed["absent_dose_rows"]
    assert len(absent) == 1
    assert absent[0]["modified_sequence"] == "GLFDQHFRL[+7]"
    assert absent[0]["absent_fmol"] == 0.1
    assert absent[0]["disposition"] == "absent_source_row_not_zero"

    # 2. Missing vs explicit zero preserved distinctly
    calib = parsed["calibration_measurements"]
    s3a = calib[calib["source_sheet"] == "DatasetS3A"]
    assert int((s3a["value_state"] == "explicit_zero").sum()) == 1
    assert not (
        (s3a["peptide"] == "GLFDQHFRL") & (s3a["nominal_standard_amount_fmol"] == 0.1)
    ).any()

    # 3. Ordinal column D at S3B 100 fmol resolves to 100fmol_3 (NOT assumed 100fmol_2!),
    # and blank column E explicitly reports missing 100fmol_2!
    s3b = calib[calib["source_sheet"] == "DatasetS3B"]
    col_c = s3b[s3b["source_ordinal_column"] == "C"].iloc[0]
    col_d = s3b[s3b["source_ordinal_column"] == "D"].iloc[0]
    col_e = s3b[s3b["source_ordinal_column"] == "E"].iloc[0]
    assert col_c["verified_acquisition_replicate"] == "100fmol_1"
    assert col_d["verified_acquisition_replicate"] == "100fmol_3"
    assert col_e["value_state"] == "literal_blank"
    assert col_e["acquisition_join_status"] == "blank_slot_missing_pride_replicates:100fmol_2"

    # 4. S3A subset-match and exact-match vs unreconciled
    glf_1 = s3a[(s3a["peptide"] == "GLFDQHFRL") & (s3a["nominal_standard_amount_fmol"] == 1.0)]
    statuses = dict(zip(glf_1["source_ordinal_column"], glf_1["acquisition_join_status"]))
    assert statuses["C"] == "transition_subset_sum_match"
    assert statuses["D"] == "exact_transition_sum_match"
    assert statuses["E"] == "unreconciled_with_transition_export"

    # 5. S6A vs S6B isotope audit flags Valine contradiction and Methionine omission
    iso = parsed["isotope_audit"]
    assert int((iso["isotope_consistency_status"] == "consistent_with_s6a").sum()) == 1
    assert int((iso["isotope_consistency_status"] == "inconsistent_with_s6a").sum()) == 2
    assert int((iso["isotope_consistency_status"] == "uncheckable_from_s6a").sum()) == 1
    m_row = iso[iso["isotope_consistency_status"] == "uncheckable_from_s6a"].iloc[0]
    assert m_row["missing_from_s6a_amino_acids"] == "M"
    assert bool(m_row["is_analytical_trigger"]) is True

    # 6. Separation of roles: S1 predicted affinity vs S3C transition counts vs calibration
    assert (parsed["panel"]["is_experimental_binding"] == False).all()  # noqa: E712
    assert (parsed["transition_counts"]["is_signal_intensity"] == False).all()  # noqa: E712

    # 7. Training recipe enforces peptide grouping & forbids MS1/APC transfer
    recipe = iso18_calibration_training_recipe(
        calib,
        spike_stage="post_ip_prepurified_matrix_free_peptide",
        require_consistent_isotope_table=True,
    )
    assert recipe["split_group_column"] == "peptide"
    assert recipe["ms1_transfer_allowed"] is False
    assert recipe["apc_concentration_transfer_allowed"] is False
    assert set(recipe["admitted"]["peptide"]) == {"GLFDQHFRL"}

    # 8. Package write creates all expected files
    pkg_dir = tmp_path / "pkg"
    manifest = write_iso18_calibration_package(src_dir, pkg_dir, allow_unverified_checksum=True)
    assert manifest["counts"]["n_s3a_slots"] == 21
    assert manifest["counts"]["n_s3b_blank_slots"] == 1
    assert (pkg_dir / "skmel5_binimetinib_transitions.parquet").is_file()


def test_drug_studies_903_contract_and_neotoma_contrast_loader(tmp_path: Path) -> None:
    contract = drug_studies_qualification_contract()
    assert contract["tracking_issue"] == 903
    summary = summarize_drug_studies_qualification()
    assert len(summary) == 7
    expected_pmids = {34497125, 34129938, 33981976, 33912179, 25576301, 29632711, 30524438}
    assert set(summary["pmid"]) == expected_pmids

    recovered = summary[summary["qualification_status"] == "observed_source_recovered"]
    gaps = summary[summary["qualification_status"] == "explicit_source_gap"]
    assert set(recovered["pmid"]) == {34497125, 34129938}
    assert set(gaps["pmid"]) == {33981976, 33912179, 25576301, 29632711, 30524438}
    assert gaps["missing_relation"].notna().all()

    # Jensen 2018 (30524438) resolves exact BET PROTAC compounds
    jensen = get_drug_study_qualification(30524438)
    assert "ARV-825 (JQ1-CRBN)" in jensen["compounds"]
    assert "ARV-771 (JQ1-VHL)" in jensen["compounds"]
    assert "JQ1-MDM2" in jensen["compounds"]
    assert "BV173" in jensen["cell_system"]

    # Pollock 2021 (34129938) keeps dTAG-13/doxycycline distinct from generic processing inhibition
    pollock = get_drug_study_qualification(34129938)
    assert pollock["compound_mechanism_class"] == "engineered_construct_tool"

    # Build synthetic NeoToma2021 CSVs + SKMEL5 workbook and test unified contrast loader
    nt_dir = tmp_path / "neotoma"
    nt_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Replicate": 1,
                "Peptide": "ASMTNMELM",
                "Protein": "Adpgk",
                "MS3SPSIons": 6,
                "MS3SumSN": 400.0,
                "MS2ScanNumber": 100,
                "MS3ScanNumber": 101,
                "MS2RetentionTime": 27.5,
                "MS3Quant2": 10.0,
                "MS3Quant6": 150.0,
                "MS3Quant9": 25.0,
                "MS3Quant10": 5.0,
            }
        ]
    ).to_csv(nt_dir / "toma1.csv", index=False)
    pd.DataFrame(
        [
            {
                "Protein Accession": "CONSTRUCT_FKBP12F36V_AdpgkR304M_tGFP",
                "Peptide": "A.ASMTNMELM.G",
                "Intensity TMT6-126": 100.0,
                "Intensity TMT6-127": 1000.0,
                "Intensity TMT6-128": 50.0,
                "Intensity TMT6-129": 9000.0,
                "Intensity TMT6-130": 1500.0,
                "Intensity TMT6-131": 400.0,
                "PTM": "TMT6plex",
                "Source File": "idAG4plex_untargetedMS2.raw",
                "Scan": 500,
            }
        ]
    ).to_csv(nt_dir / "untargeted_MS2.csv", index=False)

    parsed_nt = parse_neotoma_2021_tables(nt_dir, allow_unverified_checksum=True)
    assert len(parsed_nt["targeted_ms3"]) == 4
    assert set(parsed_nt["targeted_ms3"]["condition_drugs"]) == {
        "none",
        "doxycycline",
        "dTAG-13",
        "dTAG-13;doxycycline",
    }
    assert (
        parsed_nt["targeted_ms3"]["perturbation_mechanism_class"] == "engineered_construct_tool"
    ).all()
    ut = parsed_nt["untargeted_ms2"]
    assert int((ut["eligibility_status"] == "excluded_blank_qc_channel").sum()) == 2
    assert (
        int((ut["eligibility_status"] == "eligible_engineered_construct_contrast_only").sum()) == 4
    )

    # Unified contrast table combines SKMEL5 endogenous transitions + NeoToma targeted MS3
    stopfer_dir = _build_synthetic_stopfer_dir(tmp_path / "stopfer")
    sk_df = parse_skmel5_binimetinib_transitions(
        stopfer_dir / "SKMEL5_Transitions_DMSO_MEKi.xlsx", allow_unverified_checksum=True
    )
    assert set(sk_df["analyte_channel"]) == {"endogenous_light", "heavy_isotopologue_calibrant"}

    combined = load_qualified_drug_contrast_records(
        skmel5_xlsx_path=stopfer_dir / "SKMEL5_Transitions_DMSO_MEKi.xlsx",
        neotoma_source_dir=nt_dir,
        allow_unverified_checksum=True,
    )
    assert set(combined["pmid"]) == {34497125, 34129938}
    assert set(combined["perturbation_mechanism_class"]) == {
        "mek_kinase_inhibitor",
        "engineered_construct_tool",
    }
    # Heavy calibrant row from SKMEL5 must NOT leak into biological contrast table
    assert len(combined[combined["pmid"] == 34497125]) == 2


def test_catalog_registers_stopfer_and_pollock_datasets() -> None:
    datasets = set(hitlist.available_datasets())
    assert "stopfer-2021-iso18" in datasets
    assert "pollock-2021-neotoma" in datasets
    p_iso = hitlist.preflight("stopfer-2021-iso18")
    p_nt = hitlist.preflight("pollock-2021-neotoma")
    assert p_iso["name"] == "stopfer-2021-iso18"
    assert p_nt["name"] == "pollock-2021-neotoma"
