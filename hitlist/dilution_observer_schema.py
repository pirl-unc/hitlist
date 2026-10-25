"""Original grains for nominal-dose MS evidence; no training admission implied."""

import pyarrow as pa

from .mapp_evidence import _schema
from .material_context import MATERIAL_CONTEXT_SCHEMA, MATERIAL_LINK_SCHEMA

MEASURE_FIELDS = {
    "intensity": ("quantitative_raw", "quantitative_value"),
    "RT": ("dilution_rt_raw", "dilution_rt"),
    "leftWidth": ("dilution_left_width_raw", "dilution_left_width"),
    "rightWidth": ("dilution_right_width_raw", "dilution_right_width"),
    "peak_group_rank": ("dilution_peak_rank_raw", "dilution_peak_rank"),
    "m_score": ("dilution_m_score_raw", "dilution_m_score"),
}
OBSERVATION_TEXT = (
    "source_record_id",
    "acquisition_id",
    "peptide_sequence",
    "modified_sequence_raw",
    "charge_raw",
    "decoy_raw",
    "native_precursor_id",
    "native_feature_id",
    "library_precursor_id",
    "library_link_status",
    "library_label_status",
    "value_status",
    "measurement_unit",
    "observation_kind",
    *(raw for raw, _ in MEASURE_FIELDS.values()),
)
OBSERVATION_NUMBERS = tuple(parsed for _, parsed in MEASURE_FIELDS.values())
SCHEMAS = {
    "assets": _schema(
        (
            "source_asset_id",
            "source_file",
            "source_asset_url",
            "source_asset_role",
            "asset_sha256",
            "license_status",
        ),
        integers=("size_bytes",),
    ),
    "source_records": _schema(
        ("source_record_id", "source_asset_id", "source_file", "source_locator", "source_row_json"),
        integers=("source_row_index",),
    ),
    "assertions": _schema(
        (
            "claim_id",
            "source_record_id",
            "context_id",
            "subject_id",
            "source_scope",
            "assertion_predicate",
            "assertion_value",
            "assertion_status",
            "source_locator",
            "details_json",
        )
    ),
    "nominal_standards": _schema(
        (
            "standard_id",
            "source_record_id",
            "peptide_sequence",
            "reported_standard_form",
            "expected_modified_sequence",
            "standard_group_form",
            "nominal_status",
        ),
        integers=("label_position_1based",),
    ),
    "nominal_levels": _schema(
        (
            "nominal_level_id",
            "source_record_id",
            "sample_label_raw",
            "dilution_factor_raw",
            "nominal_concentration_raw",
            "measurement_unit",
            "nominal_status",
        ),
        numbers=("nominal_concentration",),
    ),
    "acquisitions": _schema(
        (
            "acquisition_id",
            "context_id",
            "source_record_id",
            "raw_file",
            "source_run_id",
            "source_run_label",
            "nominal_level_id",
            "method_link_precision",
            "specimen_id",
            "aliquot_id",
            "material_link_precision",
            "replicate_status",
        ),
        integers=("dia_index_from_name",),
    ),
    "library_precursors": _schema(
        (
            "library_precursor_id",
            "source_record_id",
            "native_library_group_id",
            "modified_sequence_raw",
            "peptide_sequence",
            "charge_raw",
            "decoy_raw",
            "group_label_raw",
        ),
        integers=("n_transitions",),
    ),
    "library_transitions": _schema(
        (
            "transition_id",
            "source_record_id",
            "library_precursor_id",
            "native_transition_id",
            "product_mz_raw",
            "product_charge_raw",
            "library_intensity_raw",
        ),
        numbers=("product_mz", "library_intensity"),
    ),
    "scored_features": _schema(
        ("identification_id", *OBSERVATION_TEXT, "source_run_id", "dilution_d_score_raw"),
        integers=("n_library_candidates",),
        numbers=(*OBSERVATION_NUMBERS, "dilution_d_score"),
    ),
    "aligned_quantities": _schema(
        (
            "measurement_id",
            *OBSERVATION_TEXT,
            "native_peptide_id",
            "group_label_raw",
            "dilution_alignment_rank_raw",
        ),
        integers=("n_library_candidates",),
        numbers=(*OBSERVATION_NUMBERS, "dilution_alignment_rank"),
    ),
    "feature_links": _schema(
        (
            "claim_id",
            "source_record_id",
            "measurement_id",
            "identification_id",
            "feature_link_status",
        ),
        integers=("n_candidates",),
        flags=("form_agrees", "charge_agrees", "precursor_agrees", "target_agrees"),
    ),
    "comparisons": _schema(
        (
            "claim_id",
            "source_record_id",
            "measurement_id",
            "identification_id",
            "compared_source_field",
            "aligned_literal",
            "scored_literal",
            "comparison_status",
            "exact_decimal_delta",
        )
    ),
    "standard_memberships": _schema(
        (
            "claim_id",
            "source_record_id",
            "standard_id",
            "library_precursor_id",
            "membership_status",
            "decoy_raw",
        )
    ),
    "material_contexts": MATERIAL_CONTEXT_SCHEMA,
    "material_links": MATERIAL_LINK_SCHEMA,
}
SCHEMAS = {
    name: pa.schema(
        [
            *schema,
            ("source_dataset", pa.string()),
            ("source_dataset_precision", pa.string()),
            ("source_panel", pa.string()),
        ]
    )
    for name, schema in SCHEMAS.items()
}

RELATIONS = {
    "assets": ("one_pinned_original_file", "relation", "source_asset_id"),
    "source_records": (
        "one_ordered_original_record_or_addressed_metadata_claim",
        "relation",
        "source_record_id",
    ),
    "assertions": (
        "one_source_scoped_claim_or_explicit_unqualified_parameter",
        "relation",
        "claim_id",
    ),
    "nominal_standards": (
        "one_intended_heavy_standard_not_an_observation",
        "context",
        "standard_id",
    ),
    "nominal_levels": (
        "one_reported_nominal_solution_concentration",
        "context",
        "nominal_level_id",
    ),
    "acquisitions": ("one_named_acquisition_not_a_physical_specimen", "context", "acquisition_id"),
    "library_precursors": (
        "one_library_namespace_precursor_form_charge_target_state",
        "relation",
        "library_precursor_id",
    ),
    "library_transitions": ("one_original_library_transition", "relation", "transition_id"),
    "scored_features": (
        "one_exported_scored_feature_candidate",
        "measurement",
        "identification_id",
    ),
    "aligned_quantities": (
        "one_original_aligned_precursor_run_slot",
        "measurement",
        "measurement_id",
    ),
    "feature_links": (
        "one_aligned_disposition_or_ambiguous_candidate_relation",
        "relation",
        "claim_id",
    ),
    "comparisons": ("one_named_aligned_scored_numeric_comparison", "relation", "claim_id"),
    "standard_memberships": (
        "one_intended_standard_library_member_or_absence_claim",
        "relation",
        "claim_id",
    ),
    "material_contexts": (
        "one_source_context_without_inferred_specimen_identity",
        "context",
        "context_id",
    ),
    "material_links": ("one_precision_qualified_material_context_link", "relation", "claim_id"),
}

FIELD_MEANINGS = {
    "expected_modified_sequence": "Intended protocol isotope form expressed in the deposited UniMod notation; not a measurement",
    "standard_id": "Intended source-protocol standard identity, including an unobserved standard",
    "transition_id": "Unique original library transition row identity, separate from its raw source ID",
    "assertion_value": "Original source claim value or explicit unqualified value; not an inferred model label",
    "reported_standard_form": "Literal intended isotope-labeled standard notation in the source protocol",
    "standard_group_form": "Exact protocol form expressed in the deposited library group-label notation",
    "nominal_status": "Applicability of nominal standard or solution concentration; not certified on-column amount",
    "label_position_1based": "One-based leucine isotope-label position explicitly stated in the protocol",
    "nominal_level_id": "Original nominal solution-level claim identity, separate from acquisitions and specimens",
    "dilution_factor_raw": "Original literal dilution-factor claim; conflicts elsewhere remain separate",
    "nominal_concentration_raw": "Original nominal solution concentration literal",
    "nominal_concentration": "Parsed reported nominal solution concentration in the stated unit",
    "source_run_id": "Literal source-local run identifier preserved as text without floating-point conversion",
    "source_run_label": "Original acquisition label, not a physical specimen identity",
    "method_link_precision": "Evidence precision linking a run label to a nominal level or method index",
    "dia_index_from_name": "DIA method index parsed from an acquisition filename; not a replicate identity",
    "library_precursor_id": "Source-qualified library precursor identity, distinct from scored/aligned IDs",
    "native_library_group_id": "Original library TransitionGroupId namespace, never equated to scored precursor numbers",
    "native_transition_id": "Original library transition identifier retained separately from its unique source record",
    "native_precursor_id": "Original scored/aligned precursor identifier, not the library group namespace",
    "native_feature_id": "Original feature identifier as text, including literal missing values",
    "native_peptide_id": "Original aligned peptide identifier as text",
    "product_mz_raw": "Original transition product m/z literal",
    "product_mz": "Parsed transition product mass-to-charge ratio",
    "product_charge_raw": "Original transition product-charge literal",
    "library_intensity_raw": "Original reference-library transition intensity literal",
    "library_intensity": "Parsed reference-library transition intensity, not sample abundance",
    "decoy_raw": "Original target/decoy literal; no biological negative is inferred",
    "group_label_raw": "Original precursor group label, preserving empty versus NA literals",
    "library_link_status": "Exact full-form/charge/target-state membership status, not numeric-ID equivalence",
    "library_label_status": "Additional literal library label comparison, distinct from chemical-form membership",
    "n_library_candidates": "Number of exact library form/charge/target-state candidates",
    "feature_link_status": "Exact run/feature-ID candidate disposition with independent identity checks",
    "n_candidates": "Number of scored feature candidates for the original run/feature-ID key",
    "form_agrees": "Whether the unique or ambiguous source-ID candidate has the exact same full molecular form",
    "charge_agrees": "Whether source charge literals agree on the source-ID candidate",
    "precursor_agrees": "Whether aligned/scored precursor ID literals agree within their shared namespace",
    "target_agrees": "Whether the scored candidate has the source target flag zero",
    "compared_source_field": "Name of the original aligned field in an aligned/scored numeric comparison",
    "aligned_literal": "Original aligned numeric literal, including literal missing tokens",
    "scored_literal": "Original scored-export numeric literal, including literal missing tokens",
    "comparison_status": "Exact, tolerance-close, different or missing numeric comparison; not a confidence rule",
    "exact_decimal_delta": "Exact decimal aligned-minus-scored difference as text, retaining source precision",
    "membership_status": "Intended-standard library membership or explicit absence; not an assay result",
    "n_transitions": "Number of original transition rows in this exact library precursor membership",
}
for _field, (_raw, _parsed) in MEASURE_FIELDS.items():
    if _field == "intensity":
        continue
    FIELD_MEANINGS[_raw] = (
        f"Original DIA {_field} literal, without imputation or confidence filtering"
    )
    FIELD_MEANINGS[_parsed] = (
        f"Parsed source DIA {_field}; original literal and processing grain remain separate"
    )
for _name in ("d_score", "alignment_rank"):
    FIELD_MEANINGS["dilution_" + _name + "_raw"] = f"Original source {_name} literal"
    FIELD_MEANINGS["dilution_" + _name] = f"Parsed source {_name}, not a detection or censor label"
