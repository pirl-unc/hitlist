"""Tests for Bekker-Jensen (#654, #96, #95) and MAPP (#752, #757, #759, #820, #825) qualification."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import hitlist
from hitlist.bekker_jensen_qualification import (
    bekker_jensen_source_contract,
    fetch_bekker_jensen_provenance_files,
    parse_bekker_jensen_mod_specific_peptides,
    qualify_bekker_jensen_peptide_table,
)
from hitlist.downloads import available_dataset_catalog, info, preflight
from hitlist.errors import OfflineError
from hitlist.mapp_qualification import (
    fetch_goldberg_mapp_sources,
    guard_goldberg_table3_pooled_slots,
    guard_mapp_material_pairing,
    mapp_detection_mask_summary,
    mapp_precursor_estimation_recipe,
    mapp_unified_qualification_contract,
)


def _write_mock_mod_specific_txt(path: Path) -> Path:
    header = [
        "Sequence",
        "Modifications",
        "Proteins",
        "Gene Names",
        "Reverse",
        "Potential contaminant",
        "Intensity Tryp-46fracs",
        "Intensity HeLa-46fracs-IT-E1",
        "Intensity HeLa-46fracs-IT-E2",
        "Intensity Phospho-50fracs",
        "Identification type Tryp-46fracs",
        "Identification type HeLa-46fracs-IT-E1",
        "Identification type HeLa-46fracs-IT-E2",
        "Identification type Phospho-50fracs",
    ]
    rows = [
        # Unmodified form detected in baseline Tryp-46fracs (By MS/MS) and IT-E1 (By matching)
        [
            "PEPTIDESEQK",
            "Unmodified",
            "P12345",
            "GENE1",
            "",
            "",
            "1024",
            "4096",
            "0",
            "0",
            "By MS/MS",
            "By matching",
            "",
            "",
        ],
        # Phospho form of SAME peptide detected ONLY in Phospho-50fracs TiO2 arm!
        # Must NOT bleed Phospho (STY) onto baseline HeLa 46-frac arm (#96).
        [
            "PEPTIDESEQK",
            "Phospho (STY)",
            "P12345",
            "GENE1",
            "",
            "",
            "0",
            "0",
            "0",
            "8192",
            "",
            "",
            "",
            "By MS/MS",
        ],
        # Reverse decoy — excluded
        [
            "ZZZZZZZZK",
            "Unmodified",
            "REV__P1",
            "",
            "+",
            "",
            "1000",
            "0",
            "0",
            "0",
            "By MS/MS",
            "",
            "",
            "",
        ],
    ]
    path.write_text(
        "\n".join(["\t".join(header), *("\t".join(r) for r in rows)]) + "\n",
        encoding="utf-8",
    )
    return path


class TestBekkerJensenQualification:
    def test_contract_pins_verified_hashes_and_lfq_absence(self):
        c = bekker_jensen_source_contract()
        assert c["pmid"] == 28601559
        assert c["wrong_historical_pmid_corrected"] == 28591648
        assert c["pride_accession"] == "PXD004452"
        members = c["archive"]["verified_members"]
        for key in ("parameters.txt", "summary.txt"):
            assert len(members[key]["sha256"]) == 64
            int(members[key]["sha256"], 16)
        assert members["peptides.txt"]["n_lfq_columns"] == 0
        assert members["modificationSpecificPeptides.txt"]["n_lfq_columns"] == 0
        assert c["measurement_and_modification_contract"]["peptide_level_lfq_available"] is False
        assert (
            c["measurement_and_modification_contract"]["mix_lfq_and_raw_intensity_allowed"] is False
        )
        assert (
            c["search_provenance"]["original_fasta_status"]["original_bytes_available_in_pride"]
            is False
        )
        assert (
            c["search_provenance"]["original_fasta_status"][
                "silent_join_between_pxd004452_and_pxd013455_allowed"
            ]
            is False
        )

    def test_per_arm_modifications_and_separate_raw_vs_lfq_scales(self, tmp_path):
        src = _write_mock_mod_specific_txt(tmp_path / "modificationSpecificPeptides.txt")
        exp_df, arm_df = parse_bekker_jensen_mod_specific_peptides(src)
        assert len(exp_df) == 3
        assert len(arm_df) == 2

        baseline = arm_df[arm_df["enrichment"] == "none"].iloc[0]
        phospho = arm_df[arm_df["enrichment"] == "TiO2"].iloc[0]
        # #96: baseline arm stays Unmodified; TiO2 arm gets Phospho (STY)
        assert baseline["modifications"] == "Unmodified"
        assert phospho["modifications"] == "Phospho (STY)"
        assert baseline["modifications_scope"] == "per_arm_modification_specific_peptides"
        # #95 & #654: separate raw intensity stats, explicit NaN LFQ, exact experiment IDs & MBR
        assert baseline["intensity_scale"] == "raw_ms1_intensity"
        assert math.isclose(baseline["log2_raw_intensity"], 11.0, rel_tol=1e-9)
        assert math.isclose(baseline["log2_raw_intensity_std"], math.sqrt(2.0), rel_tol=1e-9)
        assert pd.isna(baseline["log2_lfq_intensity"])
        assert baseline["experiment_ids_detected"] == "HeLa-46fracs-IT-E1;Tryp-46fracs"
        assert baseline["n_direct_msms_experiments"] == 1
        assert baseline["n_mbr_only_experiments"] == 1

    def test_qualify_legacy_table_and_offline_guard(self, tmp_path):
        legacy = pd.DataFrame([{"peptide": "AAAAR", "modifications": "Phospho (STY)"}])
        qualified = qualify_bekker_jensen_peptide_table(legacy)
        assert qualified.loc[0, "modifications_scope"] == "deposit_aggregate_peptides_txt"
        assert qualified.loc[0, "intensity_scale"] == "unquantified_presence_only"
        assert pd.isna(qualified.loc[0, "log2_lfq_intensity"])

        with pytest.raises(OfflineError):
            fetch_bekker_jensen_provenance_files(dest_dir=tmp_path / "bj", offline=True)

        cat = available_dataset_catalog(category="expression_and_proteomics")
        assert "bekker-jensen-2017-pxd004452-provenance" in cat
        plan = preflight("bekker-jensen-2017-pxd004452-provenance")
        assert plan["expected_size_bytes"] == 1678 + 587345


class TestMAPPUnifiedQualification:
    def test_contract_covers_all_five_mapp_and_three_tryptic_projects(self):
        c = mapp_unified_qualification_contract()
        assert set(c["captured_fragment_projects"]) == {
            "PXD009092",
            "PXD009508",
            "PXD007263",
            "PXD009438",
            "PXD052314",
        }
        assert set(c["companion_tryptic_projects"]) == {"PXD009360", "PXD009181", "PXD009112"}
        assert (
            c["captured_fragment_projects"]["PXD009092"][
                "publisher_workbook_redistribution_rights"
            ]["bundled_in_pypi"]
            is False
        )
        assert (
            c["captured_fragment_projects"]["PXD009438"]["meets_frozen_transfer_support_gate"]
            is False
        )
        assert (
            c["captured_fragment_projects"]["PXD052314"]["table3_pooled_audit"][
                "eligible_for_supervised_cohort"
            ]
            is False
        )

    def test_material_pairing_guard_and_precursor_recipe(self):
        links = pd.DataFrame(
            [
                {"acquisition_id": "run_1", "material_link_precision": "exact_aliquot"},
                {"acquisition_id": "run_2", "material_link_precision": "condition_surrogate"},
                {"acquisition_id": "run_3", "material_link_precision": "order_inferred"},
                {"acquisition_id": "run_4", "material_link_precision": "unresolved"},
            ]
        )
        guarded = guard_mapp_material_pairing(links, allow_condition_surrogate=False).set_index(
            "acquisition_id"
        )
        assert guarded.loc["run_1", "pairing_disposition"] == "admitted_exact_pairing"
        assert guarded.loc["run_1", "is_exact_physical_pairing"]
        assert guarded.loc["run_2", "pairing_disposition"] == "deferred_condition_surrogate"
        assert guarded.loc["run_3", "pairing_disposition"] == "rejected_unsafe_heuristic"
        assert guarded.loc["run_4", "pairing_disposition"] == "deferred_unresolved_material_link"

        pg = pd.DataFrame(
            [
                {
                    "acquisition_id": "run_1",
                    "protein_ids": "P12345",
                    "lfq_intensity": 10000.0,
                    "raw_intensity": 50000.0,
                },
                {
                    "acquisition_id": "run_1",
                    "protein_ids": "P11111;P22222",
                    "lfq_intensity": 20000.0,
                    "raw_intensity": 80000.0,
                },
                {
                    "acquisition_id": "run_2",
                    "protein_ids": "P33333",
                    "lfq_intensity": 15000.0,
                    "raw_intensity": 60000.0,
                },
                {
                    "acquisition_id": "run_3",
                    "protein_ids": "P44444",
                    "lfq_intensity": 12000.0,
                    "raw_intensity": 40000.0,
                },
                {
                    "acquisition_id": "run_1",
                    "protein_ids": "P55555",
                    "lfq_intensity": 0.0,
                    "raw_intensity": 25000.0,
                },
                {
                    "acquisition_id": "run_1",
                    "protein_ids": "REV__P99999",
                    "lfq_intensity": 9000.0,
                    "raw_intensity": 9000.0,
                },
            ]
        )
        recipe = mapp_precursor_estimation_recipe(pg, material_links_df=links)
        s = recipe["summary"]
        assert s["n_source_rows"] == 6
        assert s["n_admitted_precursors"] == 1
        assert s["n_ambiguous_protein_groups"] == 1
        assert s["n_deferred_unresolved_pairing"] == 2
        assert s["n_excluded"] == 2
        assert s["maxquant_lfq_is_copies_per_cell"] is False
        assert s["mix_lfq_and_raw_scales_allowed"] is False

        # Opting into condition_surrogate admits run_2 without admitting unsafe order_inferred run_3
        with_surr = mapp_precursor_estimation_recipe(
            pg, material_links_df=links, allow_condition_surrogate=True
        )
        assert with_surr["summary"]["n_admitted_precursors"] == 2
        assert with_surr["summary"]["n_deferred_unresolved_pairing"] == 1

    def test_detection_mask_summary_and_goldberg_table3_guard(self):
        mapp_runs = pd.DataFrame(
            [
                {"peptide": "P1", "arm_label": "ut", "raw_file": "r1", "intensity": 100.0},
                {"peptide": "P1", "arm_label": "ut", "raw_file": "r2", "intensity": 200.0},
                {"peptide": "P2", "arm_label": "ut", "raw_file": "r1", "intensity": 300.0},
                {"peptide": "P2", "arm_label": "ut", "raw_file": "r2", "intensity": 0.0},
                {"peptide": "P1", "arm_label": "epoxo", "raw_file": "e1", "intensity": 50.0},
                {"peptide": "P1", "arm_label": "epoxo", "raw_file": "e2", "intensity": 60.0},
                {
                    "peptide": "P2",
                    "arm_label": "epoxo",
                    "raw_file": "e1",
                    "intensity": float("nan"),
                },
                {"peptide": "P2", "arm_label": "epoxo", "raw_file": "e2", "intensity": 0.0},
            ]
        )
        mask = mapp_detection_mask_summary(mapp_runs)
        assert mask["n_shared_complete_case_peptides"] == 1
        assert mask["arms"]["ut"]["total_detected_signal"] == 600.0
        assert mask["arms"]["epoxo"]["total_detected_signal"] == 110.0
        assert mask["arms"]["epoxo"]["n_missing_cells"] == 1
        assert mask["unit_sum_composition_captures_global_yield_drop"] is False

        t3 = pd.DataFrame(
            [
                {"Treatment": "CTRL:10;CTRL:20;INF:5", "Gene names": "1-Mar"},
                {"Treatment": "CTRL:12;INF:8", "Gene names": "TP53"},
                {"Treatment": "A:1;A:2", "Gene names": "43891"},
            ]
        )
        audit = guard_goldberg_table3_pooled_slots(t3)
        assert audit["n_rows_with_repeated_slot_labels"] == 2
        assert audit["n_date_styled_gene_cells"] == 2
        assert audit["flagged_gene_row_indices"] == [0, 2]
        assert audit["eligible_for_supervised_cohort"] is False

    def test_goldberg_catalog_and_offline_guard(self, tmp_path):
        cat = available_dataset_catalog(category="perturbation_and_calibration")
        assert "pxd052314-goldberg-mapp-metadata" in cat
        assert "pxd052314-goldberg-mapp" in cat
        meta_plan = preflight("pxd052314-goldberg-mapp-metadata")
        full_plan = preflight("pxd052314-goldberg-mapp")
        assert len(meta_plan["files"]) == 4
        assert len(full_plan["files"]) == 6
        assert meta_plan["expected_size_bytes"] == 6986 + 46362 + 10668 + 19547
        assert full_plan["expected_size_bytes"] > 400_000_000
        assert (
            info("pxd052314-goldberg-mapp-metadata")["name"] == "pxd052314-goldberg-mapp-metadata"
        )
        with pytest.raises(OfflineError):
            fetch_goldberg_mapp_sources(dest_dir=tmp_path / "gb", offline=True)


def test_public_exports_2_15_0():
    for name in (
        "bekker_jensen_source_contract",
        "fetch_bekker_jensen_provenance_files",
        "parse_bekker_jensen_mod_specific_peptides",
        "qualify_bekker_jensen_peptide_table",
        "mapp_unified_qualification_contract",
        "guard_mapp_material_pairing",
        "mapp_precursor_estimation_recipe",
        "mapp_detection_mask_summary",
        "guard_goldberg_table3_pooled_slots",
        "fetch_goldberg_mapp_sources",
    ):
        assert hasattr(hitlist, name)
