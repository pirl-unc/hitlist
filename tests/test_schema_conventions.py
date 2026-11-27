# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for uniform condition placement & canonical assay schema conventions (v2.5.0)."""

from __future__ import annotations

import pandas as pd
import pytest

from hitlist.apc_contrasts import build_stimulus_contrast_catalog
from hitlist.condition_categories import UNPERTURBED, categorize_condition
from hitlist.conditions import (
    CONDITION_COLUMNS,
    INTERVENTION_CONDITION_COLUMNS,
    asserts_condition,
    is_engineered_material,
    split_condition_tokens,
    validate_sample_conditions,
)
from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_ms_samples_table
from hitlist.model_training import (
    prepare_affinity_training_data,
    prepare_ms_training_data,
    prepare_stability_training_data,
)
from hitlist.schema_conventions import (
    ASSAY_SCHEMA_CONVENTIONS,
    COLUMN_SYNONYM_MAP,
    canonical_assay_columns,
    normalize_assay_frame,
)


def _valid_arm(**overrides):
    base = {
        "condition_id": "arm_a",
        "condition_status": "annotated",
        "condition_evidence": "curated_text",
    }
    base.update(overrides)
    return base


def test_condition_validators_enforce_vehicle_dose_and_untreated_rules():
    with pytest.raises(ValueError, match="solvent vehicle 'DMSO'"):
        validate_sample_conditions(_valid_arm(condition_drugs="DMSO"), 1, 0)

    with pytest.raises(ValueError, match="solvent vehicle 'PBS'"):
        validate_sample_conditions(_valid_arm(condition_drugs="PBS"), 1, 0)

    with pytest.raises(ValueError, match="both condition_dose and condition_dose_unit"):
        validate_sample_conditions(_valid_arm(condition_dose="10"), 1, 0)

    with pytest.raises(ValueError, match="both condition_dose and condition_dose_unit"):
        validate_sample_conditions(_valid_arm(condition_dose_unit="uM"), 1, 0)

    with pytest.raises(ValueError, match="positive finite number"):
        validate_sample_conditions(_valid_arm(condition_dose="0", condition_dose_unit="uM"), 1, 0)

    with pytest.raises(ValueError, match="positive numeric string"):
        validate_sample_conditions(_valid_arm(condition_duration_hours="72h"), 1, 0)

    with pytest.raises(ValueError, match="alias for 'ug/mL'"):
        validate_sample_conditions(
            _valid_arm(condition_dose="100", condition_dose_unit="ug/ml"), 1, 0
        )

    with pytest.raises(ValueError, match="condition_control='untreated' alongside active"):
        validate_sample_conditions(
            _valid_arm(condition_control="untreated", condition_transduction="HLA-A"), 1, 0
        )

    # Valid vehicle + dose/duration + baseline engineering pass cleanly.
    validate_sample_conditions(
        _valid_arm(
            condition_control="vehicle",
            condition_drugs="none",
            condition_vehicle="DMSO",
            condition_duration_hours="72",
        ),
        1,
        0,
    )
    validate_sample_conditions(
        _valid_arm(
            condition_control="untreated",
            condition_baseline_engineering="HLA-DPA1_transduced;HLA-DPB1_transduced",
        ),
        1,
        0,
    )


def test_corpus_condition_placement_invariants():
    samples = generate_ms_samples_table()

    # 1. No solvent vehicle in condition_drugs; every vehicle control is unperturbed + control arm.
    for _, row in samples.iterrows():
        drugs = {t.casefold() for t in split_condition_tokens(row["condition_drugs"])}
        assert not (drugs & {"dmso", "pbs"}), (
            f"PMID {row['pmid']} {row['condition_id']} still has vehicle in condition_drugs"
        )
        if row["condition_control"] == "vehicle":
            active = [c for c in INTERVENTION_CONDITION_COLUMNS if asserts_condition(row.get(c))]
            if not active:
                assert row["condition_category"] == UNPERTURBED, (
                    f"PMID {row['pmid']} {row['condition_id']} vehicle classified as "
                    f"{row['condition_category']!r}"
                )
                assert row["is_control_arm"] == "true"

    # 2. Case-insensitive uniqueness on condition_background across the entire corpus.
    bg_by_fold: dict[str, set[str]] = {}
    for cell in samples["condition_background"].dropna().unique():
        for tok in split_condition_tokens(str(cell)):
            bg_by_fold.setdefault(tok.casefold(), set()).add(tok)
            assert not tok.endswith(("_KO", "_transduced", "_transfection")), (
                f"Engineered token {tok!r} belongs in condition_baseline_engineering, not condition_background"
            )
    case_dups = {k: sorted(v) for k, v in bg_by_fold.items() if len(v) > 1}
    assert case_dups == {}, f"Case-inconsistent condition_background tokens: {case_dups}"

    # 3. Baseline engineering marks is_engineered_material() == True.
    ov = load_pmid_overrides()
    for pmid in (30484775, 33025005, 35251023, 34171305):
        for arm in ov[pmid]["ms_samples"]:
            assert arm.get("condition_baseline_engineering")
            assert is_engineered_material(arm)

    # 4. PMID 32488085 structured palbociclib dose/duration distinguishes 1 uM vs 10 uM.
    p324 = {a["condition_id"]: a for a in ov[32488085]["ms_samples"]}
    assert p324["sk_mel_5_palbociclib_1um"]["condition_dose"] == "1"
    assert p324["sk_mel_5_palbociclib_1um"]["condition_dose_unit"] == "uM"
    assert p324["sk_mel_5_palbociclib_1um"]["condition_duration_hours"] == "72"
    assert p324["sk_mel_5_palbociclib_10um"]["condition_dose"] == "10"
    assert p324["sk_mel_5_dmso"]["condition_vehicle"] == "DMSO"
    assert p324["sk_mel_5_dmso"]["condition_drugs"] == "none"

    # 5. PMID 35051231 THP-1 mock carries PMA + infection=none + control_for.
    p350 = {a["condition_id"]: a for a in ov[35051231]["ms_samples"]}
    thp_mock = p350["thp_1_macrophage_uv_mock_uninfected"]
    assert thp_mock["condition_stimulation"] == "PMA"
    assert thp_mock["condition_infection"] == "none"
    assert "thp_1_macrophage_wisconsin_infected" in thp_mock["condition_control_for"]

    # 6. Non-condition prose cleaned on 33123756, 27397612, 42199926 primary tumor arms.
    for pmid, cid in (
        (33123756, "ovarian_carcinoma_patient_tumor"),
        (27397612, "cholangiocarcinoma_patient_tumor"),
        (42199926, "lola_h58a"),
    ):
        row = samples[(samples["pmid"] == pmid) & (samples["condition_id"] == cid)].iloc[0]
        assert row["condition_category"] == UNPERTURBED


def test_categorize_condition_structured_fallback_and_vehicle_consistency():
    assert (
        categorize_condition(
            "DMSO vehicle control; 48 h",
            condition_columns={"condition_control": "vehicle", "condition_drugs": "none"},
        )
        == UNPERTURBED
    )
    assert (
        categorize_condition(
            "novel_small_molecule_xyz 50 nM",
            condition_columns={"condition_drugs": "novel_small_molecule_xyz"},
        )
        == "drug_exposure"
    )
    assert (
        categorize_condition(
            "GM-CSF + IL-4 differentiation",
            condition_columns={"condition_cytokines": "CSF2;IL4"},
        )
        == "cytokine_treatment_generic"
    )


def test_canonical_schema_conventions_and_normalize_assay_frame():
    assert set(ASSAY_SCHEMA_CONVENTIONS) >= {
        "presentation_ms",
        "binding_affinity",
        "stability",
        "tcell",
        "perturbation_contrast",
        "bulk_proteomics",
        "line_expression",
    }
    assert callable(normalize_assay_frame)
    assert callable(canonical_assay_columns)
    assert "peptide_sequence" in COLUMN_SYNONYM_MAP

    raw = pd.DataFrame(
        [
            {
                "peptide_sequence": "SIINFEKL",
                "mhc_allele": "HLA-A*02:01",
                "reported_mhc_class": "I",
                "quantitative_value": 42.0,
                "measurement_units": "nM",
                "normalized_inequality": "=",
                "source_pmid": 12345,
                "treatment_dose": "10",
                "treatment_dose_unit": "uM",
                "duration_hours": "72",
                "control_condition_id": "ctrl_arm",
                "is_comparator_for": "treated_arm",
            }
        ]
    )
    norm = normalize_assay_frame(raw, assay_type="binding_affinity")
    assert list(norm.columns[:6]) == [
        "peptide",
        "mhc_restriction",
        "mhc_class",
        "measurement_value",
        "measurement_unit",
        "measurement_inequality",
    ]
    assert "pmid" in norm.columns
    assert "condition_dose" in norm.columns
    assert "comparator_condition_ids" in norm.columns
    assert "condition_control_for" in norm.columns
    assert "peptide_sequence" not in norm.columns

    # keep_aliases=True preserves originals alongside canonical names.
    kept = normalize_assay_frame(raw, keep_aliases=True)
    assert "peptide" in kept.columns and "peptide_sequence" in kept.columns
    assert kept.loc[0, "peptide"] == "SIINFEKL"

    with pytest.raises(ValueError, match="Unknown assay_type"):
        canonical_assay_columns("not_an_assay")


def test_model_training_and_apc_contrasts_expose_uniform_canonical_columns():
    # Binding affinity & stability: canonical columns on BOTH accepted and excluded.
    binding_df = pd.DataFrame(
        [
            {
                "evidence_row_id": "b1",
                "evidence_source_id": "s1",
                "provenance_id": "p1",
                "evidence_kind": "binding",
                "peptide": "SLYNTVATL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "response_measured": "IC50",
                "quantitative_value": 25.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "assay_method": "purified MHC/competitive/radioactivity",
                "pmid": 11111,
                "has_ptm": False,
            },
            {
                "evidence_row_id": "b2",
                "evidence_source_id": "s1",
                "provenance_id": "p1",
                "evidence_kind": "binding",
                "peptide": "SLYNTVATL",
                "mhc_restriction": "HLA class I",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "response_measured": "IC50",
                "quantitative_value": 50.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "assay_method": "purified MHC/competitive/radioactivity",
                "pmid": 11111,
                "has_ptm": False,
            },
            {
                "evidence_row_id": "st1",
                "evidence_source_id": "s1",
                "provenance_id": "p1",
                "evidence_kind": "binding",
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "response_measured": "half life",
                "quantitative_value": 2.0,
                "measurement_units": "h",
                "measurement_inequality": "=",
                "assay_method": "scintillation proximity assay",
                "pmid": 27402703,
                "has_ptm": False,
            },
        ]
    )
    aff = prepare_affinity_training_data(binding_df)
    for frame in (aff.accepted, aff.excluded):
        assert {
            "peptide",
            "mhc_restriction",
            "mhc_class",
            "measurement_value",
            "measurement_unit",
            "measurement_inequality",
            "pmid",
        } <= set(frame.columns)
    assert "raw_measurement_unit" in aff.accepted.columns

    stab = prepare_stability_training_data(binding_df)
    assert "mhc_restriction" in stab.accepted.columns
    assert stab.accepted.loc[0, "mhc_restriction"] == stab.accepted.loc[0, "allele"]

    # MS presentation training: preserves ALL structured CONDITION_COLUMNS + canonical MHC cols.
    ms_df = pd.DataFrame(
        [
            {
                "evidence_row_id": "m1",
                "evidence_source_id": "s1",
                "provenance_id": "p1",
                "evidence_kind": "ms",
                "peptide": "SLYNTVATL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_allele_set": "HLA-A*02:01",
                "mhc_allele_provenance": "reported_single",
                "restriction_evidence": "monoallelic",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "is_monoallelic": True,
                "pmid": 32488085,
                "condition_id": "sk_mel_5_palbociclib_1um",
                "condition_drugs": "palbociclib",
                "condition_dose": "1",
                "condition_dose_unit": "uM",
                "condition_duration_hours": "72",
                "has_ptm": False,
            }
        ]
    )
    ms_sel = prepare_ms_training_data(ms_df)
    assert set(CONDITION_COLUMNS) <= set(ms_sel.accepted.columns)
    assert {"mhc_restriction", "mhc_allele_set"} <= set(ms_sel.accepted.columns)
    assert ms_sel.accepted.loc[0, "condition_drugs"] == "palbociclib"
    assert ms_sel.accepted.loc[0, "condition_dose"] == "1"

    # APC stimulus contrast catalog exposes condition_control_for & condition_vehicle.
    cat = build_stimulus_contrast_catalog()
    assert {"condition_control_for", "condition_vehicle", "comparator_condition_ids"} <= set(
        cat.columns
    )
    db_dmso = cat[(cat["pmid"] == 35561310) & (cat["condition_id"] == "db_dmso")].iloc[0]
    assert db_dmso["condition_vehicle"] == "DMSO"
    assert db_dmso["condition_drugs"] == ""
    assert "db_ifng" in db_dmso["condition_control_for"]
