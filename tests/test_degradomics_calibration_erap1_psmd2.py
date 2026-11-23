"""Tests for Block B: #762 (calibration), #809 (ERAP1), #831 (PXD058121), #837 (PXD018722)."""

from __future__ import annotations

import zipfile

import openpyxl
import pandas as pd
import pytest

import hitlist
from hitlist.degradomics_native_qualification import (
    fetch_pxd018722_sahu_metadata_sources,
    fetch_pxd058121_intact_degradomics_sources,
    guard_intact_precursor_scale,
    guard_sahu_2021_endogenous_vs_reporter,
    intact_degradomics_scale_qualification_contract,
    sahu_2021_psmd2_qualification_contract,
)
from hitlist.downloads import available_dataset_catalog, info, preflight
from hitlist.erap1_biochemical_qualification import (
    erap1_biochemical_inventory_contract,
    fetch_maben_2021_erap1_source,
    parse_maben_2021_archive,
    qualify_erap1_biochemical_records,
)
from hitlist.errors import OfflineError
from hitlist.ms_calibration_qualification import (
    fetch_pxd004886_calibration_sources,
    guard_ms_response_identifiability,
    non_mhc_calibration_contract,
)


class TestErap1BiochemicalInventory809:
    def test_inventory_contract_and_catalog(self, tmp_path):
        c = erap1_biochemical_inventory_contract()
        maben = c["sources"]["maben_2021"]
        assert len(maben["archive_asset"]["sha256"]) == 64
        assert maben["audit_summary"]["passes_sequence_diverse_specificity_training_gate"] is False
        assert maben["audit_summary"]["figs8d_clip_above_100_allowed"] is False
        assert c["sources"]["chang_2005"]["deposited_numeric_tables_available"] is False
        assert c["sources"]["evnouchidou_2008"]["deposited_numeric_tables_available"] is False

        cat = available_dataset_catalog(category="perturbation_and_calibration")
        assert "maben-2021-erap1-source-data" in cat
        plan = preflight("maben-2021-erap1-source-data")
        assert plan["expected_size_bytes"] == 479402
        with pytest.raises(OfflineError):
            fetch_maben_2021_erap1_source(dest_dir=tmp_path / "maben", offline=True)

    def test_qualify_erap1_records_preserves_above_100_missing_and_mixture_guards(self):
        raw = pd.DataFrame(
            [
                {
                    "substrate_sequence": "LSIINFEKL",
                    "product_sequence": "SIINFEKL",
                    "value": 108.5,
                    "unit": "percent",
                },
                {
                    "substrate_sequence": "LSIINFEKL",
                    "product_sequence": "SIINFEKL",
                    "value": 0.0,
                    "unit": "percent",
                },
                {
                    "substrate_sequence": "LSIINFEKL",
                    "product_sequence": "SIINFEKL",
                    "value": float("nan"),
                    "unit": "percent",
                },
                {
                    "substrate_sequence": "X9MIX",
                    "product_sequence": "",
                    "value": 42.0,
                    "is_mixed_library_aggregate": True,
                },
            ]
        )
        out = qualify_erap1_biochemical_records(
            raw, study_id="maben_2021", endpoint_family="endpoint_fraction"
        )
        assert out.loc[0, "value"] == 108.5
        assert out.loc[0, "value_state"] == "positive_above_100_unclipped"
        assert not out.loc[0, "value_clipped_to_probability"]
        assert out.loc[1, "value_state"] == "zero"
        assert out.loc[2, "value_state"] == "missing"
        assert out.loc[2, "qualification_disposition"] == "excluded_missing_value"
        assert (
            out.loc[3, "qualification_disposition"] == "excluded_unresolved_mixed_library_aggregate"
        )
        assert out["isolated_biochemical_not_apc_concentration"].all()

        fig_out = qualify_erap1_biochemical_records(
            raw.iloc[:1],
            study_id="evnouchidou_2008",
            endpoint_family="initial_rate",
            source_kind="digitized_figure",
        )
        assert (
            fig_out.loc[0, "qualification_disposition"]
            == "secondary_digitized_figure_not_independent_replicates"
        )
        assert not fig_out.loc[0, "figure_points_are_independent_replicates"]

    def test_parse_maben_archive_synthetic_zip(self, tmp_path):
        wb_path = tmp_path / "Source Data_Main Figures.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Fig2h"
        ws.append(["Conc", "Resp1", "Resp2"])
        ws.append([10.0, 50.0, None])
        wb.save(wb_path)

        zpath = tmp_path / "maben.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.write(wb_path, arcname="Source Data_Main Figures.xlsx")

        parsed = parse_maben_2021_archive(zpath, allow_unverified_checksum=True)
        assert parsed["n_total_sheets"] == 1
        assert (
            parsed["workbooks"]["Source Data_Main Figures.xlsx"]["sheets"]["Fig2h"][
                "n_numeric_cells"
            ]
            == 2
        )
        assert parsed["passes_sequence_diverse_specificity_training_gate"] is False


class TestNonMhcCalibration762:
    def test_contract_identifiability_gate_and_catalog(self, tmp_path):
        c = non_mhc_calibration_contract()
        assert (
            c["identifiability_rule"]["ratio_only_curves_eligible_for_absolute_response"] is False
        )
        assert (
            c["candidates"]["PASS01163_PASS01200"]["separate_isotope_channel_areas_in_supplements"]
            is False
        )
        assert (
            c["candidates"]["PASS05890"]["access_receipts"]["certificate_validation_disabled"]
            is False
        )
        assert c["candidates"]["PASS05890"]["n_table_s3_peptide_assay_links"] == 48
        assert c["candidates"]["PASS05890"]["n_title_assays"] == 53

        df = pd.DataFrame(
            [
                {
                    "peptide": "P1",
                    "endpoint_kind": "separate_isotope_channel_area",
                    "amount_scope": "on_column_fmol",
                    "signal_value": 12000.0,
                },
                {
                    "peptide": "P2",
                    "endpoint_kind": "isotope_ratio_only",
                    "amount_scope": "on_column_fmol",
                    "signal_value": 1.25,
                },
                {
                    "peptide": "P3",
                    "endpoint_kind": "inferred_concentration",
                    "amount_scope": "stock_solution_concentration_only",
                    "signal_value": 5.0,
                },
                {
                    "peptide": "P4",
                    "endpoint_kind": "raw_peak_area",
                    "amount_scope": "stock_solution_concentration_only",
                    "signal_value": 9000.0,
                },
                {
                    "peptide": "P5",
                    "endpoint_kind": "raw_peak_area",
                    "amount_scope": "on_column_fmol",
                    "signal_value": float("nan"),
                },
            ]
        )
        res = guard_ms_response_identifiability(df)
        s = res["summary"]
        assert s["n_source_rows"] == 5
        assert s["n_admitted_absolute_response"] == 1
        assert s["n_rejected_unidentifiable"] == 2
        assert s["n_deferred_stock_only"] == 1
        assert s["ratio_only_identifies_absolute_response"] is False

        cat = available_dataset_catalog(category="perturbation_and_calibration")
        assert "pxd004886-ms-calibration" in cat
        plan = preflight("pxd004886-ms-calibration")
        assert len(plan["files"]) == 8
        assert plan["expected_size_bytes"] > 400_000
        with pytest.raises(OfflineError):
            fetch_pxd004886_calibration_sources(dest_dir=tmp_path / "cal", offline=True)


class TestDegradomicsAndNative831And837:
    def test_pxd058121_scale_contract_guard_and_catalog(self, tmp_path):
        c = intact_degradomics_scale_qualification_contract()
        assert c["pride_accession"] == "PXD058121"
        assert c["replicates_r1_r3_are_independent_animals"] is False
        assert (
            c["precursor_scale_qualification"]["sum_deconvoluted_input_intensities_is_tic"] is False
        )
        assert len(c["documented_source_exceptions"]) == 7

        df = pd.DataFrame(
            [
                {
                    "intensity_representation": "within_run_product_to_intact_ratio",
                    "normalizer_basis": "within_run_intact",
                },
                {
                    "intensity_representation": "raw_deconvoluted_xtract_intensity",
                    "normalizer_basis": "sum_deconvoluted_as_tic",
                },
                {
                    "intensity_representation": "raw_deconvoluted_xtract_intensity",
                    "normalizer_basis": "protocol_max_accumulation_time",
                    "has_per_scan_accumulation_receipt": False,
                },
                {
                    "intensity_representation": "raw_deconvoluted_xtract_intensity",
                    "normalizer_basis": "none",
                },
                {
                    "intensity_representation": "raw_deconvoluted_xtract_intensity",
                    "normalizer_basis": "independent_stable_standard_calibrated",
                },
            ]
        )
        out = guard_intact_precursor_scale(df)
        assert out["scale_disposition"].tolist() == [
            "admitted_within_run_gain_cancelling_ratio",
            "rejected_invalid_tic_assumption",
            "rejected_unverified_accumulation_normalization",
            "deferred_uncalibrated_cross_acquisition_scale",
            "admitted_calibrated_cross_acquisition_scale",
        ]
        assert (~out["sum_deconvoluted_is_tic"]).all()

        plan = preflight("pxd058121-intact-degradomics")
        assert len(plan["files"]) == 14
        assert plan["expected_size_bytes"] > 4_000_000
        with pytest.raises(OfflineError):
            fetch_pxd058121_intact_degradomics_sources(dest_dir=tmp_path / "intact", offline=True)

    def test_pxd018722_sahu_contract_reporter_guard_and_catalog(self, tmp_path):
        c = sahu_2021_psmd2_qualification_contract()
        assert c["pride_accession"] == "PXD018722"
        assert c["engineered_reporter_accession"] == "P22222"
        assert c["taxonomy_ox_9606_establishes_endogenous_origin"] is False
        assert c["actual_search_length_range"] == [8, 25]
        assert c["hek293t_is_professional_apc"] is False

        df = pd.DataFrame(
            [
                {"peptide": "PEPTIDESEQ", "protein_ids": "P12345", "intensity_scale": "intensity"},
                {"peptide": "REPORTERP", "protein_ids": "P22222", "intensity_scale": "lfq"},
                {
                    "peptide": "SHAREDPEP",
                    "protein_ids": "P12345;NS3CCNB1_HUMAN",
                    "intensity_scale": "intensity",
                },
                {"peptide": "SHORTP", "protein_ids": "P12345", "intensity_scale": "intensity"},
                {
                    "peptide": "DECOYPEPT",
                    "protein_ids": "REV__P99999",
                    "intensity_scale": "intensity",
                },
            ]
        )
        out = guard_sahu_2021_endogenous_vs_reporter(df)
        assert out["origin_class"].tolist() == [
            "endogenous_human_candidate",
            "engineered_reporter_origin",
            "ambiguous_endogenous_and_reporter",
            "endogenous_human_candidate",
            "reverse_or_contaminant",
        ]
        assert out.loc[3, "hi20s_search_scope_status"] == "outside_hi20s_8_25_search_scope"
        assert out.loc[0, "intensity_scale_qualified"] == "raw_intensity_distinct_from_lfq"
        assert out.loc[1, "intensity_scale_qualified"] == "lfq_intensity_distinct_from_raw"
        assert (~out["ox_9606_proves_endogenous"]).all()

        plan = preflight("pxd018722-sahu-2021-psmd2-metadata")
        assert len(plan["files"]) == 5
        assert (
            info("pxd018722-sahu-2021-psmd2-metadata")["name"]
            == "pxd018722-sahu-2021-psmd2-metadata"
        )
        with pytest.raises(OfflineError):
            fetch_pxd018722_sahu_metadata_sources(dest_dir=tmp_path / "sahu", offline=True)


def test_public_exports_2_16_0():
    for name in (
        "erap1_biochemical_inventory_contract",
        "fetch_maben_2021_erap1_source",
        "parse_maben_2021_archive",
        "qualify_erap1_biochemical_records",
        "non_mhc_calibration_contract",
        "fetch_pxd004886_calibration_sources",
        "guard_ms_response_identifiability",
        "intact_degradomics_scale_qualification_contract",
        "sahu_2021_psmd2_qualification_contract",
        "fetch_pxd058121_intact_degradomics_sources",
        "fetch_pxd018722_sahu_metadata_sources",
        "guard_intact_precursor_scale",
        "guard_sahu_2021_endogenous_vs_reporter",
    ):
        assert hasattr(hitlist, name)
