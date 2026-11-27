"""Tests for #845 (dilution observer) and #778 (Gomez-Zepeda curation closure)."""

from __future__ import annotations

import pandas as pd

import hitlist
from hitlist.curation import load_pmid_overrides
from hitlist.dilution_observer import (
    dilution_observer_applicability_contract,
    dilution_observer_nominal_dose_recipe,
)
from hitlist.downloads import available_dataset_catalog, preflight
from hitlist.qc import engineering_drift_audit


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

    def test_nominal_dose_recipe_handles_nullable_booleans_and_ambiguous_links(self):
        tables = self._build_90_slot_tables()
        links = tables["feature_links"].copy()
        for col in ("form_agrees", "charge_agrees", "precursor_agrees", "target_agrees"):
            links[col] = links[col].astype("boolean")
            links.loc[links["feature_link_status"] == "missing_feature_id", col] = pd.NA
        dup_row = links[links["measurement_id"] == "meas_4"].copy()
        dup_row["identification_id"] = "id_4_ambig"
        tables["feature_links"] = pd.concat([links, dup_row], ignore_index=True)

        recipe = dilution_observer_nominal_dose_recipe(tables)
        assert recipe["summary"]["n_total_standard_slots"] == 90
        assert recipe["summary"]["n_admitted_slots"] == 86
        assert recipe["summary"]["n_excluded_slots"] == 4
        meas4 = recipe["all_slots"].set_index("measurement_id").loc["meas_4"]
        assert meas4["feature_link_status"] == "ambiguous_feature_id"
        assert meas4["recipe_disposition"] == "excluded"


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
        "dilution_observer_applicability_contract",
        "dilution_observer_nominal_dose_recipe",
        "fetch_dilution_observer_sources",
        "load_study_qualifications",
        "load_study_contract",
        "fetch_study_assets",
    ):
        assert hasattr(hitlist, name)
