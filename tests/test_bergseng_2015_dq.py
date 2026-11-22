"""Tests for Bergseng 2015 HLA-DQ run-level MS1 ion volumes adapter (#859)."""

from __future__ import annotations

import math

import openpyxl
import pyarrow.parquet as pq
import pytest

import hitlist
from hitlist.bergseng_2015_dq import (
    bergseng_2015_source_contract,
    parse_bergseng_2015_workbook,
    verify_bergseng_2015_dq_package,
    write_bergseng_2015_dq_package,
)


def _write_synthetic_bergseng_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Table S1. Synthetic Bergseng 2015 fixture"])
    ws.append(
        [
            "Peptide number",
            "Sequence",
            "Length",
            "Mass",
            "Proteins",
            "Leading razor protein",
            "Unique (Groups)",
            "Unique (Proteins)",
            "Charges",
            "PEP",
            "Score",
            "Ion peak volume S9037_1a",
            "Ion peak volume A9064_5a",
            "Ion peak volume RN_7a",
            "Ion peak volume P9047_15a",
            "Reverse",
            "Contaminant",
        ]
    )
    # Row 3: normal PEP <= 0.1, mix of positive and explicit zero
    ws.append(
        [
            1,
            "TPLLMQALPMGALPQ",
            15,
            1580.8,
            "P04233",
            "P04233",
            "yes",
            "yes",
            "2;3",
            0.005,
            95.2,
            1000.0,
            250.0,
            0.0,
            500.0,
            "",
            "",
        ]
    )
    # Row 4: PEP > 0.1 retained (never silently filtered!) + explicit zeros + missing cell
    ws.append(
        [
            2,
            "ASFEAQGALANIAVDKA",
            17,
            1688.9,
            "P01903",
            "P01903",
            "yes",
            "yes",
            "2",
            0.145,
            31.4,
            3000.0,
            0.0,
            100.0,
            None,
            "",
            "",
        ]
    )
    wb.save(path)


def test_bergseng_contract_preserves_typing_ambiguity_and_unresolved_rn():
    contract = bergseng_2015_source_contract()
    assert contract["pmid"] == 25502872
    assert contract["license"] == "CC-BY-4.0"
    assert (
        contract["asset"]["sha256"]
        == "30d6206813788ffd05eabded3f276d950605b923e4798aaabba0be6d5a50dd37"
    )
    groups = contract["header_groups"]
    assert len(groups) == 9
    # AMALA/9064 preserves DQA1 05:01/05:03 ambiguity (never collapsed to 05:05!)
    assert groups["A9064"]["table1_cell_line"] == "9064 AMALA"
    assert groups["A9064"]["reported_dqa1"] == "05:01/05:03"
    assert groups["A9064"]["typing_resolution"] == "ambiguous_alpha_pair"
    assert groups["A9064"]["mhc_allele_set"] == (
        "HLA-DQA1*05:01/DQB1*03:01;HLA-DQA1*05:03/DQB1*03:01"
    )
    assert groups["S9037"]["mhc_allele_set"] == "HLA-DQA1*05:05/DQB1*03:01"
    assert groups["B9089"]["mhc_allele_set"] == "HLA-DQA1*05:05/DQB1*03:01"
    # RN alias stays unresolved (never silently mapped to CD114 by elimination!)
    assert groups["RN"]["table1_join_status"] == "unresolved_alias"
    assert groups["RN"]["table1_cell_line"] == ""
    assert groups["RN"]["mhc_allele_set"] == ""


def test_parse_and_package_bergseng_preserves_cells_zeros_normalization_and_flags(tmp_path):
    wb_path = tmp_path / "fixture.xlsx"
    _write_synthetic_bergseng_workbook(wb_path)

    parsed = parse_bergseng_2015_workbook(wb_path, verify_pin=False)
    peptides = parsed["peptides"]
    runs = parsed["runs"].set_index("run_id")
    meas = parsed["measurements"]

    assert len(peptides) == 2
    assert peptides["exceeds_methods_pep_0_1"].tolist() == [False, True]
    assert set(peptides["peptidoform_status"]) == {"sequence_level_only_variable_mods_untracked"}

    # Hierarchy & Table 1 join provenance
    assert runs.loc["S9037_1a", "cell_line_name"] == "9037 SWEIG"
    assert runs.loc["A9064_5a", "typing_resolution"] == "ambiguous_alpha_pair"
    assert runs.loc["RN_7a", "table1_join_status"] == "unresolved_alias"
    assert runs.loc["RN_7a", "cell_line_name"] == ""
    assert runs.loc["P9047_15a", "dq_serotype"] == "DQ2.2"
    assert runs.loc["S9037_1a", "total_ion_peak_volume"] == pytest.approx(4000.0)

    # Exact cell coordinates, raw vs relative normalized values, explicit zero vs missing
    m_by_id = meas.set_index("measurement_id")
    row_pos = m_by_id.loc["bergseng2015:p1:S9037_1a"]
    assert row_pos["source_cell_locator"] == "Sheet1!L3"
    assert row_pos["raw_ion_peak_volume"] == pytest.approx(1000.0)
    assert row_pos["normalized_relative_volume"] == pytest.approx(0.25)
    assert row_pos["value_state"] == "positive"
    assert bool(row_pos["is_absolute_concentration"]) is False
    assert row_pos["evidence_grain"] == "peptide_by_technical_run_separate_from_iedb_aggregate"

    row_zero = m_by_id.loc["bergseng2015:p1:RN_7a"]
    assert row_zero["source_cell_locator"] == "Sheet1!N3"
    assert row_zero["raw_ion_peak_volume"] == 0.0
    assert row_zero["value_state"] == "explicit_zero"

    row_missing = m_by_id.loc["bergseng2015:p2:P9047_15a"]
    assert row_missing["source_cell_locator"] == "Sheet1!O4"
    assert math.isnan(row_missing["raw_ion_peak_volume"])
    assert row_missing["value_state"] == "missing"

    # Package write + verify + tamper detection
    pkg_dir = tmp_path / "pkg"
    manifest = write_bergseng_2015_dq_package(pkg_dir, workbook_path=wb_path, verify_pin=False)
    assert verify_bergseng_2015_dq_package(pkg_dir) == manifest
    assert pq.read_table(pkg_dir / "measurements.parquet").num_rows == 8

    # Pin enforcement rejects wrong bytes when verify_pin=True
    with pytest.raises(ValueError, match="size mismatch"):
        parse_bergseng_2015_workbook(wb_path, verify_pin=True)

    # Tampered artifact is caught by verify
    (pkg_dir / "runs.parquet").write_bytes((pkg_dir / "runs.parquet").read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash differs"):
        verify_bergseng_2015_dq_package(pkg_dir)


def test_bergseng_2015_registered_in_uniform_download_catalog():
    assert hitlist.bergseng_2015_source_contract is bergseng_2015_source_contract
    avail = hitlist.available_datasets()
    assert "bergseng-2015-dq" in avail
    plan = hitlist.preflight("bergseng-2015-dq")
    assert plan["name"] == "bergseng-2015-dq"
    assert plan["expected_size_bytes"] == 4693608
