"""Tests for #860 (DO:DM ratio peptidomics PMID 35085787 / PXD024392)
and #816 (Nanaware 2019 HLA-DO source qualification PXD010301).
"""

from __future__ import annotations

import math

import openpyxl
import pyarrow.parquet as pq
import pytest

import hitlist
from hitlist.dohm_ratio_peptidomics import (
    dodm_ratio_source_contract,
    guard_dodm_evidence_namespace,
    parse_dodm_ratio_workbook,
    verify_dodm_ratio_package,
    write_dodm_ratio_package,
)
from hitlist.nanaware_2019_do import (
    extract_title_encoded_intensities,
    nanaware_2019_do_source_contract,
    validate_nanaware_s13_calibration_scope,
)


def _write_synthetic_dodm_workbook(path):
    wb = openpyxl.Workbook()
    ws_id = wb.active
    ws_id.title = "Supp_data_MHC_peptides"
    ws_id.append(
        [
            "Sample ID",
            "Peptide",
            "Proteins",
            "SpecID",
            "Label",
            "Scanf",
            "Xcorr",
            "10lgP",
            "ALC",
            "Peptide",
            "Q_value",
            "PEP_sore",
        ]
    )
    ws_id.append(
        [
            "T2DR4_biorep1",
            "GPQKFC[119.00]IDKVGKETW",
            "sp|O00308|WWP2_HUMAN",
            "target_4_23437_12493",
            0,
            12493,
            2.8254,
            36.6,
            0,
            "N.GPQKFC[119.00]IDKVGKETW.L",
            0.0,
            3.84e-16,
        ]
    )
    ws_id.append(
        [
            "T2DR4DMDOKO_biorep2",
            "EQQRLAVQQVEEAQQLR",
            "sp|O00461|GOLI4_HUMAN",
            "target_3_23435_12648",
            0,
            12648,
            3.7603,
            36.26,
            0,
            "L.EQQRLAVQQVEEAQQLR.E",
            0.0,
            1.2e-12,
        ]
    )

    ws_q = wb.create_sheet("Supp_data_MHC_quant")
    ws_q.append(
        [
            "Accession",
            "Proteins",
            "Peptide",
            "DR4_rep1_1",
            "DR4DM_rep1_1",
            "DR4DMDO+_rep1_1",
            "DR4DMDO++_rep1_1",
            "DR4DMDO+++_rep1_1",
            "DOKO_rep1_1",
        ]
    )
    ws_q.append(
        [
            "O00308",
            "sp|O00308|WWP2_HUMAN",
            "GPQKFC[119.00]IDKVGKETW",
            1.5e7,
            2.0e7,
            None,
            1.2e7,
            9.0e6,
            1.8e7,
        ]
    )
    wb.save(path)


def test_dodm_ratio_contract_clones_and_namespace_guards():
    contract = dodm_ratio_source_contract()
    assert contract["pmid"] == 35085787
    assert contract["pride_accession"] == "PXD024392"
    assert contract["protocol_provenance"]["mhc_allele_set"] == "HLA-DRA*01:01/DRB1*04:01"
    # DOKO is derived from 2D7 ((DO:DM)H) and kept distinct from DR4DM.
    assert contract["conditions"]["DOKO"]["parent_clone"] == "2D7"
    assert contract["conditions"]["DOKO"]["is_doko_comparison"] is True
    assert contract["conditions"]["DR4DM"]["is_doko_comparison"] is False
    assert contract["conditions"]["DR4DMDO+"]["clone_name"] == "G8"
    assert contract["conditions"]["DR4DMDO++"]["clone_name"] == "1C3"
    assert contract["conditions"]["DR4DMDO+++"]["clone_name"] == "2D7"

    # Namespace guards reject predicted Affinity(nM), PLAtEAU imputed cores, and TMT proteome.
    assert guard_dodm_evidence_namespace("peptide_ms1_area") == "peptide_ms1_area"
    with pytest.raises(ValueError, match="NetMHCIIpan prediction"):
        guard_dodm_evidence_namespace("predicted_netmhc_affinity")
    with pytest.raises(ValueError, match="must remain separate"):
        guard_dodm_evidence_namespace("plateau_imputed_cores")
    with pytest.raises(ValueError, match="must remain separate"):
        guard_dodm_evidence_namespace("whole_cell_tmt_proteome")


def test_parse_and_package_dodm_ratio_preserves_ptms_missingness_and_doko(tmp_path):
    wb_path = tmp_path / "dodm_fixture.xlsx"
    _write_synthetic_dodm_workbook(wb_path)

    parsed = parse_dodm_ratio_workbook(wb_path, verify_pin=False)
    ids = parsed["identifications"]
    runs = parsed["runs"].set_index("run_id")
    ms1 = parsed["ms1_quantities"].set_index("measurement_id")

    # Modified peptide string & flanking residues preserved verbatim
    assert ids.loc[0, "peptide_modified"] == "GPQKFC[119.00]IDKVGKETW"
    assert ids.loc[0, "peptide"] == "GPQKFCIDKVGKETW"
    assert ids.loc[0, "peptide_flanked"] == "N.GPQKFC[119.00]IDKVGKETW.L"
    assert bool(ids.loc[0, "has_ptm"]) is True
    assert bool(ids.loc[1, "has_ptm"]) is False

    # Runs preserve clone identities & DOKO separation
    assert runs.loc["DR4DMDO+_rep1_1", "clone_name"] == "G8"
    assert runs.loc["DOKO_rep1_1", "clone_name"] == "2D7_KO"
    assert bool(runs.loc["DOKO_rep1_1", "is_doko_comparison"]) is True
    assert bool(runs.loc["DR4DM_rep1_1", "is_doko_comparison"]) is False

    # Missing MS1 cell remains explicitly missing (never converted to zero/negative!)
    miss_row = ms1.loc["dodm35085787:r2:DR4DMDO+_rep1_1"]
    assert miss_row["source_cell_locator"] == "Supp_data_MHC_quant!F2"
    assert miss_row["value_state"] == "missing"
    assert math.isnan(miss_row["raw_ms1_area"])
    assert bool(miss_row["is_imputed"]) is False

    # Package write + verify + uniform catalog check
    pkg = tmp_path / "dodm_pkg"
    manifest = write_dodm_ratio_package(pkg, workbook_path=wb_path, verify_pin=False)
    assert verify_dodm_ratio_package(pkg) == manifest
    assert pq.read_table(pkg / "ms1_quantities.parquet").num_rows == 6
    assert "dodm-ratio-35085787" in hitlist.available_datasets()
    assert hitlist.preflight("dodm-ratio-35085787")["expected_size_bytes"] == 7838853


def test_nanaware_2019_do_contract_systems_and_title_intensity_parser():
    contract = nanaware_2019_do_source_contract()
    assert hitlist.nanaware_2019_do_source_contract is nanaware_2019_do_source_contract
    assert contract["pmid"] == 30573663
    assert contract["massive_accession"] == "MSV000082570"
    assert set(contract["experimental_systems"]) == {"human_lg2_do_clones", "mouse_splenic_b_cells"}
    assert contract["legacy_iedb_caveat"]["status"] == "iedb_axis_mismatch_unresolved"
    # S10-S14 non-OA retrieval failures preserved explicitly
    for tbl in ("S10", "S11", "S12", "S13", "S14"):
        assert (
            contract["supplement_retrieval_receipts"][tbl]["retrieval_status"]
            == "non_oa_blocked_preserved_failure"
        )
    # Outcome-selected S13 guard blocks unbiased global calibration
    assert validate_nanaware_s13_calibration_scope(global_calibration=False)["n_standards"] == 9
    with pytest.raises(ValueError, match="outcome-selected isotope standards"):
        validate_nanaware_s13_calibration_scope(global_calibration=True)

    # Title-encoded intensity parser preserves duplicates & missingness without summing
    df, summary = extract_title_encoded_intensities(
        [
            {
                "spectrum_result_id": "SR1",
                "identification_item_id": "SII1",
                "peptide": "PKYVKQNTLKLAT",
                "spectrum_title": "scan=101 intensity=4.5e6 charge=2",
            },
            {
                "spectrum_result_id": "SR2",
                "identification_item_id": "SII2",
                "peptide": "PKYVKQNTLKLAT",
                "spectrum_title": "scan=102 intensity=4.5e6 charge=3",
            },
            {
                "spectrum_result_id": "SR3",
                "identification_item_id": "SII3",
                "peptide": "GILGFVFTL",
                "spectrum_title": "scan=103 charge=2",
            },
        ],
        system_id="human_lg2_do_clones",
    )
    assert summary["n_spectrum_results"] == 3
    assert summary["n_positive_title_intensities"] == 2
    assert summary["n_missing_title_intensities"] == 1
    assert summary["n_distinct_positive_values"] == 1
    assert summary["n_shared_value_results"] == 2
    assert df["is_shared_intensity_value"].tolist() == [True, True, False]
    assert df["safe_to_sum_as_precursor_amount"].tolist() == [False, False, False]
    assert set(df["unit_and_aggregation_status"]) == {"unresolved_unit_and_aggregation_semantics"}
