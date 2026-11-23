"""Tests for #715 (+ #905), #845, #798, and #778."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import hitlist
from hitlist.curation import load_pmid_overrides
from hitlist.dc_immunoproteasome_qualification import (
    DETECTION_LIMIT_VALUE,
    dc_immunoproteasome_contrast_recipe,
    de_verteuil_2010_source_contract,
    qualify_dc_immunoproteasome_records,
)
from hitlist.dilution_observer import (
    dilution_observer_applicability_contract,
    dilution_observer_nominal_dose_recipe,
)
from hitlist.downloads import available_dataset_catalog, info, preflight
from hitlist.erap2_oe_qualification import (
    _extract_remote_zip_members,
    erap2_oe_contrast_recipe,
    erap2_oe_source_contract,
    fetch_erap2_oe_maxquant_sources,
    parse_erap2_oe_peptides,
    verify_erap2_oe_package,
    write_erap2_oe_package,
)
from hitlist.errors import OfflineError
from hitlist.qc import engineering_drift_audit

_MQ_HEADER = [
    "Sequence",
    "Length",
    "Proteins",
    "Leading razor protein",
    "Gene names",
    "PEP",
    "Score",
    "Identification type A29(-)-1",
    "Identification type A29(-)-2",
    "Identification type A29(-)-3",
    "Identification type A29(+)-1",
    "Identification type A29(+)-2",
    "Identification type A29(+)-3",
    "Intensity A29(-)-1",
    "Intensity A29(-)-2",
    "Intensity A29(-)-3",
    "Intensity A29(+)-1",
    "Intensity A29(+)-2",
    "Intensity A29(+)-3",
    "Reverse",
    "Potential contaminant",
    "id",
]


def _write_mock_peptides_txt(path: Path) -> Path:
    rows = [
        # Shared direct MS/MS in both arms (2 reps GFP, 3 reps ERAP2)
        [
            "YTFGSVEKY",
            "9",
            "P12345",
            "P12345",
            "GENE1",
            "0.001",
            "120.5",
            "By MS/MS",
            "By MS/MS",
            "",
            "By MS/MS",
            "By MS/MS",
            "By MS/MS",
            "1000",
            "2000",
            "0",
            "6000",
            "6000",
            "6000",
            "",
            "",
            "0",
        ],
        # Shared via MBR in GFP arm only (By matching in GFP, By MS/MS in ERAP2)
        [
            "KTFGSVEKY",
            "9",
            "P23456",
            "P23456",
            "GENE2",
            "0.01",
            "90.0",
            "By matching",
            "",
            "",
            "By MS/MS",
            "",
            "",
            "500",
            "0",
            "0",
            "1000",
            "0",
            "0",
            "",
            "",
            "1",
        ],
        # PF-GFP only
        [
            "RTFGSVEKY",
            "9",
            "P34567",
            "P34567",
            "GENE3",
            "0.02",
            "85.0",
            "By MS/MS",
            "",
            "",
            "",
            "",
            "",
            "800",
            "0",
            "0",
            "0",
            "0",
            "0",
            "",
            "",
            "2",
        ],
        # PF-ERAP2 only
        [
            "ATFGSVEKY",
            "9",
            "P45678",
            "P45678",
            "GENE4",
            "0.015",
            "88.0",
            "",
            "",
            "",
            "By MS/MS",
            "By MS/MS",
            "",
            "0",
            "0",
            "0",
            "1200",
            "1400",
            "0",
            "",
            "",
            "3",
        ],
        # Zero intensity in all 6 runs
        [
            "VTFGSVEKY",
            "9",
            "P56789",
            "P56789",
            "GENE5",
            "0.04",
            "55.0",
            "",
            "",
            "",
            "",
            "",
            "",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "",
            "",
            "4",
        ],
        # Reverse decoy
        [
            "ZZZZZZZZY",
            "9",
            "REV__P1",
            "REV__P1",
            "",
            "0.04",
            "50.0",
            "By MS/MS",
            "",
            "",
            "By MS/MS",
            "",
            "",
            "100",
            "0",
            "0",
            "100",
            "0",
            "0",
            "+",
            "",
            "5",
        ],
        # Contaminant
        [
            "KRTCONTAM",
            "9",
            "CON__P1",
            "CON__P1",
            "KRT1",
            "0.01",
            "95.0",
            "By MS/MS",
            "",
            "",
            "",
            "",
            "",
            "900",
            "0",
            "0",
            "0",
            "0",
            "0",
            "",
            "+",
            "6",
        ],
    ]
    lines = ["\t".join(_MQ_HEADER), *("\t".join(r) for r in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestSanzBravo2018bErap2OE:
    def test_pmid_29769354_arms_disambiguated_and_drift_cleared(self):
        entry = load_pmid_overrides()[29769354]
        arms = {s["condition_id"]: s for s in entry["ms_samples"]}
        assert set(arms) == {
            "pf97387_nontransduced",
            "pf97387_gfp_lentiviral_control",
            "pf97387_erap2_lentiviral_oe",
            "gm19397_gm19452_natural_erap2_positive",
        }
        # PF-GFP must keep mock GFP lentiviral engineering, never blank or "none".
        gfp = arms["pf97387_gfp_lentiviral_control"]
        assert gfp["condition_control"] == "mock"
        assert gfp["condition_overexpression_genes"] == "GFP"
        assert gfp["condition_transduction"] == "lentiviral_GFP_control"
        # Nontransduced PF97387 explicitly asserts none.
        assert arms["pf97387_nontransduced"]["condition_overexpression_genes"] == "none"
        assert arms["pf97387_nontransduced"]["condition_transduction"] == "none"
        # Engineering drift audit must emit zero findings for PMID 29769354.
        audit = engineering_drift_audit()
        assert audit[audit["pmid"] == 29769354].empty

    def test_contract_and_catalog_registration(self):
        contract = erap2_oe_source_contract()
        assert contract["pmid"] == 29769354
        assert contract["pride_accession"] == "PXD007679"
        assert contract["iedb_pooled_table_status"]["eligible_for_oe_vs_control_training"] is False
        assert (
            contract["pride_inventory"]["verified_sdrf_asset"]["authoritative_for_enzyme_or_arms"]
            is False
        )
        assert contract["verified_peptides_txt_counts"]["n_total_rows"] == 6308
        assert contract["verified_peptides_txt_counts"]["n_clean_peptides"] == 5502
        assert contract["verified_peptides_txt_counts"]["n_shared_intensity_both_arms"] == 1603

        cat = available_dataset_catalog(category="perturbation_and_calibration")
        assert "sanz-bravo-2018b-erap2-oe" in cat
        plan = preflight("sanz-bravo-2018b-erap2-oe")
        assert plan["expected_size_bytes"] == (2593233 + 3484 + 3347 + 12447 + 6695)
        meta = info("sanz-bravo-2018b-erap2-oe")
        assert meta["name"] == "sanz-bravo-2018b-erap2-oe"

    def test_remote_zip_range_extractor_and_offline_guard(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "A29-19-3-18/combined/txt/summary.txt",
                b"Raw file\tExperiment\nSeq56011_QE2\tA29(-)-1\n",
            )
            zf.writestr(
                "A29-19-3-18/combined/txt/peptides.txt", b"Sequence\tLength\nYTFGSVEKY\t9\n"
            )
            zf.writestr("A29-19-3-18/mqpar.xml", b"<MaxQuantParams/>")
        raw_zip = buf.getvalue()
        requests_made: list[tuple[int, int]] = []

        def fake_range(_url: str, start: int, end_inclusive: int) -> bytes:
            requests_made.append((start, end_inclusive))
            return raw_zip[start : end_inclusive + 1]

        out = _extract_remote_zip_members(
            "https://example.org/A29-19-3-18.zip",
            len(raw_zip),
            {"peptides.txt": "combined/txt/peptides.txt", "mqpar.xml": "mqpar.xml"},
            range_getter=fake_range,
        )
        assert out["peptides.txt"] == b"Sequence\tLength\nYTFGSVEKY\t9\n"
        assert out["mqpar.xml"] == b"<MaxQuantParams/>"
        assert len(requests_made) >= 4

        with pytest.raises(OfflineError):
            fetch_erap2_oe_maxquant_sources(dest_dir=tmp_path / "erap2", offline=True)

    def test_parse_contrast_recipe_and_package_roundtrip(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        pep_path = _write_mock_peptides_txt(src_dir / "peptides.txt")

        tables = parse_erap2_oe_peptides(pep_path, allow_unverified_checksum=True)
        assert len(tables["peptides"]) == 7
        assert len(tables["run_observations"]) == 7 * 6
        # Shared (2 * 2 = 4) + GFP-only (1) + ERAP2-only (1) = 6 membership rows
        assert len(tables["arm_memberships"]) == 6
        assert (~tables["run_observations"]["nondetection_is_measured_negative"]).all()

        # Default contrast recipe admits both shared peptides (direct MS/MS + MBR-assisted)
        recipe = erap2_oe_contrast_recipe(tables, iedb_pooled_peptides={"YTFGSVEKY", "RTFGSVEKY"})
        s = recipe["summary"]
        assert s["n_source_rows"] == 7
        assert s["n_excluded_reverse_or_contaminant"] == 2
        assert s["n_clean_peptides"] == 5
        assert s["n_shared_peptides"] == 2
        assert s["n_shared_direct_msms_both_arms"] == 1
        assert s["n_shared_mbr_assisted"] == 1
        assert s["n_pf_gfp_only"] == 1
        assert s["n_pf_erap2_only"] == 1
        assert s["n_zero_intensity_all_runs"] == 1
        assert s["n_admitted_shared_contrasts"] == 2
        assert s["n_condition_specific_detections"] == 2
        assert s["n_excluded_from_contrast"] == 3
        assert s["n_overlapping_pooled_iedb_peptides"] == 2
        assert s["pooled_iedb_rows_eligible_for_oe_contrast"] is False

        # Strict direct MS/MS in both arms excludes the MBR-only shared peptide explicitly
        strict = erap2_oe_contrast_recipe(tables, require_direct_msms_both_arms=True)
        assert strict["summary"]["n_admitted_shared_contrasts"] == 1
        assert "mbr_only_in_at_least_one_arm" in set(strict["exclusions"]["recipe_reason"])

        # Package round-trip + tamper detection
        pkg_dir = tmp_path / "pkg"
        manifest_path = write_erap2_oe_package(src_dir, pkg_dir, allow_unverified_checksum=True)
        assert manifest_path.is_file()
        verified = verify_erap2_oe_package(pkg_dir)
        assert verified["summary"]["n_clean_peptides"] == 5

        # Reloading contrast recipe directly from package dir works identically
        reloaded = erap2_oe_contrast_recipe(pkg_dir)
        assert reloaded["summary"]["n_admitted_shared_contrasts"] == 2

        # Tampering with a parquet file fails verification
        (pkg_dir / "peptides.parquet").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="integrity mismatch"):
            verify_erap2_oe_package(pkg_dir)


class TestDilutionObserverRecipeAndCatalog:
    def _build_90_slot_tables(self) -> dict[str, pd.DataFrame]:
        standards = pd.DataFrame(
            [
                {
                    "standard_id": f"std_{i}",
                    "peptide_sequence": f"PEPTIDE{i}",
                    "reported_standard_form": f"PEPTIDE{i}_heavy",
                    "expected_modified_sequence": f"PEPTIDE{i}(UniMod:267)",
                    "nominal_status": "intended_heavy_standard",
                }
                for i in range(1, 11)
            ]
        )
        # 9 observed target standards + 1 decoy entry + 1 absent 10th standard
        memberships = pd.DataFrame(
            [
                *[
                    {
                        "standard_id": f"std_{i}",
                        "library_precursor_id": f"lib_{i}",
                        "membership_status": "target",
                        "decoy_raw": "0",
                    }
                    for i in range(1, 10)
                ],
                {
                    "standard_id": "std_1",
                    "library_precursor_id": "lib_decoy_1",
                    "membership_status": "decoy",
                    "decoy_raw": "1",
                },
                {
                    "standard_id": "std_10",
                    "library_precursor_id": None,
                    "membership_status": "absent_from_library",
                    "decoy_raw": None,
                },
            ]
        )
        levels = pd.DataFrame(
            [
                {
                    "nominal_level_id": f"lvl_{idx}",
                    "sample_label_raw": f"No{idx}",
                    "dilution_factor_raw": f"1:{2 ** (idx - 1)}",
                    "nominal_concentration_raw": str(0.2 / (2 ** (idx - 1))),
                    "nominal_concentration": 0.2 / (2 ** (idx - 1)),
                }
                for idx in range(1, 6)
            ]
        )
        acquisitions = pd.DataFrame(
            [
                {
                    "acquisition_id": f"acq_{lvl}_{dia}",
                    "raw_file": f"run_{lvl}_DIA_{dia}.mzML",
                    "source_run_label": f"run_{lvl}_DIA_{dia}",
                    "nominal_level_id": f"lvl_{lvl}",
                    "dia_index_from_name": dia,
                    "method_link_precision": "filename_derived",
                }
                for lvl in range(1, 6)
                for dia in (1, 2)
            ]
        )
        aligned_rows = []
        link_rows = []
        slot_num = 0
        for std_idx in range(1, 10):
            for lvl in range(1, 6):
                for dia in (1, 2):
                    slot_num += 1
                    mid = f"meas_{slot_num}"
                    if slot_num == 1:
                        # 1 positive with missing feature ID
                        vstatus, qval, fid, fstatus = "positive", 123.0, "NA", "missing_feature_id"
                        agrees = False
                    elif slot_num in (2, 3):
                        # 2 literal zeros
                        vstatus, qval, fid, fstatus = "zero", 0.0, "NA", "missing_feature_id"
                        agrees = False
                    else:
                        vstatus, qval, fid, fstatus = (
                            "positive",
                            1000.0 / (2 ** (lvl - 1)),
                            f"feat_{slot_num}",
                            "matched_identity",
                        )
                        agrees = True
                    aligned_rows.append(
                        {
                            "measurement_id": mid,
                            "library_precursor_id": f"lib_{std_idx}",
                            "acquisition_id": f"acq_{lvl}_{dia}",
                            "native_feature_id": fid,
                            "quantitative_raw": str(qval),
                            "quantitative_value": qval,
                            "value_status": vstatus,
                            "measurement_unit": "source_signal_arbitrary_units",
                            "dilution_m_score_raw": "0.001" if agrees else "NA",
                            "dilution_m_score": 0.001 if agrees else None,
                        }
                    )
                    link_rows.append(
                        {
                            "measurement_id": mid,
                            "identification_id": f"id_{slot_num}" if agrees else None,
                            "feature_link_status": fstatus,
                            "form_agrees": agrees,
                            "charge_agrees": agrees,
                            "precursor_agrees": agrees,
                            "target_agrees": agrees,
                        }
                    )
        return {
            "nominal_standards": standards,
            "standard_memberships": memberships,
            "aligned_quantities": pd.DataFrame(aligned_rows),
            "feature_links": pd.DataFrame(link_rows),
            "acquisitions": acquisitions,
            "nominal_levels": levels,
        }

    def test_applicability_contract_and_catalog(self):
        c = dilution_observer_applicability_contract()
        assert c["source_dataset"] == "PXD024809"
        assert c["n_intended_standards"] == 10
        assert c["n_expected_standard_slots"] == 90
        assert c["n_expected_admitted_slots"] == 87
        assert c["absent_standard_is_measured_negative"] is False
        assert c["dia1_and_dia2_exchangeable_replicates"] is False
        assert c["dose_grid_conflict_present"] is True
        assert c["general_sequence_transfer_allowed"] is False
        assert c["apc_concentration_transfer_allowed"] is False

        cat = available_dataset_catalog(category="perturbation_and_calibration")
        assert "pxd024809-dilution-observer" in cat
        plan = preflight("pxd024809-dilution-observer")
        assert plan["expected_size_bytes"] == c["total_pinned_size_bytes"]
        assert len(plan["files"]) == 17

    def test_nominal_dose_recipe_reproduces_87_of_90_and_guards_10th_standard(self):
        recipe = dilution_observer_nominal_dose_recipe(self._build_90_slot_tables())
        s = recipe["summary"]
        assert s["n_intended_standards"] == 10
        assert s["n_observed_standards"] == 9
        assert s["n_absent_standards"] == 1
        assert s["n_total_standard_slots"] == 90
        assert s["n_admitted_slots"] == 87
        assert s["n_excluded_slots"] == 3
        assert s["n_excluded_missing_feature_id_positive"] == 1
        assert s["n_excluded_literal_zero"] == 2
        assert s["n_leave_level_out_folds"] == 5
        assert s["dia_methods_present"] == ["DIA1", "DIA2"]
        assert recipe["absent_standards"]["standard_id"].tolist() == ["std_10"]
        assert "std_10" not in set(recipe["all_slots"]["standard_id"])
        # Both DIA1 and DIA2 share the same leave-level-out fold at each dose level.
        for _, grp in recipe["all_slots"].groupby("nominal_level_id"):
            assert grp["leave_level_out_fold"].nunique() == 1
            assert set(grp["dia_method"]) == {"DIA1", "DIA2"}


class TestDeVerteuil2010DCImmunoproteasome:
    def test_source_contract_pins_non_oa_and_machinery_semantics(self):
        c = de_verteuil_2010_source_contract()
        assert c["pmid"] == 20484733
        assert c["pmc_id"] == "PMC2938112"
        assert c["access_status"]["public_quantitative_supplement_available"] is False
        assert c["access_status"]["receipts"]["access_controls_bypassed"] is False
        ko_genes = {
            g["canonical_mouse_symbol"]
            for g in c["perturbation_contract"]["concurrent_genotype_knockouts"]
        }
        assert ko_genes == {"Psmb8", "Psmb10"}
        sec_genes = {
            g["canonical_mouse_symbol"]
            for g in c["perturbation_contract"]["secondary_observed_protein_changes"]
        }
        assert sec_genes == {"Psmb9"}
        assert (
            c["perturbation_contract"]["b2m_control_contract"][
                "eligible_as_peptide_abundance_training_negative"
            ]
            is False
        )
        assert (
            c["measurement_and_ascertainment_guards"]["detection_limit_value"]
            == DETECTION_LIMIT_VALUE
        )
        assert (
            c["measurement_and_ascertainment_guards"]["wt_only_conversion_to_ko_zero_allowed"]
            is False
        )

    def test_qualification_and_contrast_recipe_guards(self):
        raw = pd.DataFrame(
            [
                {
                    "peptide": "SIINFEKL",
                    "wt_intensity": 60000.0,
                    "dko_intensity": 30000.0,
                    "wt_to_dko_ratio_raw": "2.0",
                    "predicted_mhc_allele": "H2-Kb",
                    "rna_log2_fc": 0.5,
                },
                {
                    "peptide": "RGYVYQGL",
                    "wt_intensity": 90000.0,
                    "dko_intensity": 15000.0,
                    "wt_to_dko_ratio_raw": ">6.0",
                    "is_detection_limit_ratio": True,
                    "predicted_mhc_allele": "H2-Kb",
                },
                {
                    "peptide": "ASNENMETM",
                    "wt_intensity": 45000.0,
                    "dko_intensity": 0.0,
                    "wt_to_dko_ratio_raw": "",
                    "predicted_mhc_allele": "H2-Db",
                },
                {
                    "peptide": "B2MCTRLPE",
                    "arm_role": "b2m_ko_specificity_control",
                    "wt_intensity": 10000.0,
                    "dko_intensity": None,
                    "wt_to_dko_ratio_raw": "",
                    "predicted_mhc_allele": "",
                },
            ]
        )
        qualified = qualify_dc_immunoproteasome_records(
            raw, source_asset="Table_S1_authorized.xlsx", source_sha256="deadbeef"
        )
        assert (~qualified["predicted_allele_is_measured_single_allele"]).all()
        assert (~qualified["rna_evidence_is_measured_precursor_protein"]).all()
        assert (~qualified["b2m_control_eligible_as_abundance_negative"]).all()
        by_pep = qualified.set_index("peptide")
        assert by_pep.loc["SIINFEKL", "value_state"] == "exact_ratio"
        assert by_pep.loc["RGYVYQGL", "value_state"] == "lower_bound_inequality"
        assert by_pep.loc["RGYVYQGL", "ratio_qualifier"] == ">="
        assert by_pep.loc["ASNENMETM", "value_state"] == "coerced_zero_forbidden"
        assert by_pep.loc["B2MCTRLPE", "value_state"] == "b2m_specificity_control"

        recipe = dc_immunoproteasome_contrast_recipe(qualified)
        s = recipe["summary"]
        assert s["n_source_rows"] == 4
        assert s["n_exact_quantitative"] == 1
        assert s["n_lower_bound_inequalities"] == 1
        assert s["n_b2m_specificity_controls"] == 1
        assert s["n_excluded"] == 1
        assert s["n_coerced_zero_rejected"] == 1
        assert s["n_admitted_contrasts"] == 1

        with_ineq = dc_immunoproteasome_contrast_recipe(
            qualified, include_lower_bound_inequalities=True
        )
        assert with_ineq["summary"]["n_admitted_contrasts"] == 2


class TestGomezZepeda2024Closure:
    def test_pmid_38480730_non_transfected_arms_and_drift_cleared(self):
        entry = load_pmid_overrides()[38480730]
        arms = {s["condition_id"]: s for s in entry["ms_samples"]}
        for cid in (
            "jy_ebv_lcl_titration_series",
            "hela_cervical_carcinoma",
            "sk_mel_37_melanoma",
            "raji_burkitt_s_lymphoma_ebv_lcl",
            "human_plasma_commercial_pool",
        ):
            assert arms[cid]["condition_transfection"] == "none"
        assert arms["raji_spike_sars_cov_2_s1_s2_transfectants"]["condition_transfection"] == (
            "SARS-CoV-2 spike S1+S2"
        )
        audit = engineering_drift_audit()
        assert audit[audit["pmid"] == 38480730].empty


def test_public_exports_in_hitlist_init():
    for name in (
        "erap2_oe_source_contract",
        "fetch_erap2_oe_maxquant_sources",
        "parse_erap2_oe_peptides",
        "erap2_oe_contrast_recipe",
        "write_erap2_oe_package",
        "verify_erap2_oe_package",
        "dilution_observer_applicability_contract",
        "dilution_observer_nominal_dose_recipe",
        "fetch_dilution_observer_sources",
        "de_verteuil_2010_source_contract",
        "qualify_dc_immunoproteasome_records",
        "dc_immunoproteasome_contrast_recipe",
    ):
        assert hasattr(hitlist, name)
