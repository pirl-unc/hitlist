"""Fixed native same-parent contrast contract and compact output grains."""

import pyarrow as pa

from .mapp_evidence import _schema

RECIPE_ID = "pxd018722_native_contrast_v1"
TABLES = (
    "peptides",
    "quantities",
    "acquisitions",
    "references",
    "origins",
    "declared_parents",
    "dependency_groups",
    "forms",
    "identifications",
    "assertions",
    "material_contexts",
    "material_links",
)
SAMPLES = ("Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3")
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "endpoint": "same_parent_observed_native_signal_log10_ratio_change",
    "source_revision": "pxd018722_hi20s_native_v1",
    "source_package_schema_version": 27,
    "admission": [
        "standard_amino_acids_length_8_to_25",
        "no_source_reverse_or_contaminant_flag",
        "finite_PEP_in_closed_interval_0_to_0.01",
        "all_origins_literal_human_nonengineered_one_complete_parent_sequence",
        "every_declared_parent_resolves_to_that_complete_sequence",
        "all_six_raw_sequence_acquisition_signals_finite_and_positive",
        "at_least_two_eligible_peptides_per_parent",
    ],
    "reference": "minimum_sha256(pxd018722-native-reference-v1|parent_sha256|peptide_sequence)",
    "target": "mean_KD(log10(I_peptide)-log10(I_reference))-mean_control(log10(I_peptide)-log10(I_reference))",
    "samples": list(SAMPLES),
    "pairing": "unpaired_arm_means_no_Cont1_KD1_specimen_pairing",
    "split": "sha256(pxd018722-native-split-v1|all_source_dependency_group_id)_integer_mod20:0-13_train,14-16_val,17-19_test",
    "weights": "equal_component_then_parent_then_contrast_within_split",
    "support_gate": {
        "train": {"n_components": 40, "n_contrasts": 80},
        "val": {"n_components": 8, "n_contrasts": 16},
        "test": {"n_components": 8, "n_contrasts": 16},
    },
    "limitations": [
        "conditional_on_positive_observations_not_complete_pool_distribution",
        "source_sequence_signal_can_aggregate_modifications_and_matching",
        "run_load_and_precursor_cancellation_require_multiplicative_observation_model",
        "fixed_peptide_response_cancellation_does_not_remove_condition_dependent_MS_or_extraction",
        "numbered_runs_not_established_independent_or_paired_specimens",
        "same_study_acquisitions_across_parent_component_splits",
        "no_molar_calibration_or_elapsed_processing_time",
        "HEK293T_not_professional_APC_transfer",
        "unknown_zeros_and_missing_values_not_negative_or_censored_labels",
        "all_source_parent_dependencies_precede_supervised_eligibility",
    ],
}

DECISIONS = _schema(
    (
        "peptide_id",
        "source_record_id",
        "native_id",
        "peptide_sequence",
        "parent_sequence_sha256",
        "dependency_group_id",
        "split",
        "status",
        "reasons_json",
        "source_quantities_json",
        "candidate_origins_json",
        "declared_parents_json",
        "feature_types_by_acquisition_json",
        "modified_form_support_json",
        "reference_peptide_id",
    ),
    numbers=("posterior_error_probability",),
)
EXAMPLES = _schema(
    (
        "example_id",
        "peptide_id",
        "reference_peptide_id",
        "peptide_sequence",
        "reference_peptide_sequence",
        "parent_sequence_sha256",
        "dependency_group_id",
        "split",
        "measurement_ids_json",
        "acquisition_log_ratios_json",
    ),
    numbers=("target_log10_ratio_change", "weight"),
)
SCHEMAS = {
    "decisions": DECISIONS,
    "parents": _schema(
        (
            "parent_sequence_sha256",
            "parent_sequence",
            "reference_records_json",
            "reference_peptide_id",
            "dependency_group_id",
            "split",
        )
    ),
    "examples": EXAMPLES,
    "covariates": pa.schema(
        [
            f
            for f in EXAMPLES
            if f.name not in {"target_log10_ratio_change", "acquisition_log_ratios_json"}
        ]
    ),
    "contributors": _schema(
        (
            "contributor_id",
            "example_id",
            "measurement_id",
            "peptide_id",
            "source_record_id",
            "acquisition_id",
            "sample_label_raw",
            "quantity_role",
            "quantitative_raw",
        ),
        numbers=("quantitative_value", "target_coefficient"),
    ),
    "components": _schema(("peptide_id", "dependency_group_id", "split", "dependency_policy")),
}
