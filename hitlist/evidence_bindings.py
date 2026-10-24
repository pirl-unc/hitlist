"""Exact physical bindings for the versioned evidence catalog.

Logical aliases are admitted only when a physical column has the same meaning
at the same grain. Every physical column remains queryable as ``native.<name>``.
Selectors distinguish evidence types that intentionally share one Parquet file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .evidence_catalog import CONTROLLED_RELATION_NAMES, DIGESTION_RELATIONS, EVIDENCE


@dataclass(frozen=True)
class Selector:
    column: str
    operator: str
    value: object


@dataclass(frozen=True)
class Link:
    name: str
    source: str
    target: str
    source_key: str
    target_key: str
    cardinality: str
    scope: str


SELECTORS = {
    "mhc_positive_ms": (Selector("evidence_type", "eq", "mhc_positive_ms"),),
    "mhc_binding": (Selector("evidence_type", "eq", "mhc_binding"),),
    "mhc_other": (Selector("evidence_type", "eq", "mhc_other"),),
    "tcell_functional_peptide": (
        Selector("task", "eq", "assay_specific_tcell_qualitative_response"),
    ),
    "tcell_multimer_peptide": (
        Selector("task", "eq", "assay_specific_pmhc_multimer_cell_binding"),
    ),
    "bulk_ms_peptide": (Selector("grain", "eq", "peptide"),),
    "bulk_ms_protein": (Selector("grain", "eq", "protein"),),
    "line_expression_gene": (Selector("grain", "eq", "gene_profile"),),
    "external_mhc_peptide_quantity": (Selector("table_number", "in", (7, 8, 9, 10, 12)),),
    "external_digest_peptide_quantity": (Selector("table_number", "eq", 11),),
}

# These are exact same-grain aliases. Related values are declared as links below;
# a raw field with an uncertain role is intentionally left under native.*.
ALIASES = {
    "proteometools_assets": {
        "source_asset_id": "asset_id",
        "source_asset_url": "source_url",
        "asset_sha256": "sha256",
    },
    "proteometools_planned_products": {
        "source_record_id": "planned_id",
        "sample_label_raw": "pool_name",
    },
    "proteometools_contexts": {
        "source_reference_iri": "source_url",
        "measured_pool": "material_type",
        "digestion_enzyme": "laboratory_digest",
    },
    "proteometools_runs": {"acquisition_id": "run_id", "sample_label_raw": "pool_name"},
    "proteometools_peptides": {"acquisition_id": "run_id"},
    "proteometools_forms": {"acquisition_id": "run_id"},
    "hap1_assets": {"source_asset_role": "source_role"},
    "hap1_runs": {"acquisition_id": "run_id"},
    "hap1_occurrences": {"mapping_position": "start_1based"},
    "tmab_source_asset": {"source_asset_role": "source_role"},
    "calibration_protocol_parameter": {
        "source_record_id": "parameter_id",
        "source_scope": "scope",
        "assertion_predicate": "parameter_name_raw",
        "assertion_value": "value_json",
    },
    "calibration_source_sheet": {"source_record_id": "sheet_id", "source_file": "asset_path"},
    "calibration_source_cell": {
        "source_cell": "cell_id",
        "source_file": "asset_path",
        "source_row_index": "row_1based",
    },
    "calibration_standard": {
        "source_record_id": "standard_id",
        "parent_label_raw": "protein_label_raw",
    },
    "calibration_nominal_load": {
        "measurement_id": "nominal_load_id",
        "sample_label_raw": "sample_label",
        "quantitative_value": "value",
        "measurement_unit": "unit",
    },
    "calibration_reporter_form": {"source_record_id": "reporter_id"},
    "calibration_source_slot": {"source_record_id": "slot_id", "sample_label_raw": "sample_label"},
    "calibration_s4_measurement": {"quantitative_value": "value", "measurement_unit": "unit"},
    "calibration_summary_measurement": {"quantitative_value": "value", "measurement_unit": "unit"},
    "calibration_curve": {"source_record_id": "curve_id"},
    "calibration_curve_measurement": {"quantitative_value": "value", "measurement_unit": "unit"},
    "calibration_source_assertion": {
        "claim_id": "assertion_id",
        "assertion_predicate": "predicate",
        "assertion_value": "value_raw",
        "source_scope": "scope",
    },
    "calibration_normalization_diagnostic": {"source_record_id": "diagnostic_id"},
    "mapp_source_record": {"source_file": "asset_path", "source_row_index": "row_1based"},
    "mapp_peptide": {"peptide_sequence": "sequence_raw", "parent_label_raw": "parent_labels_raw"},
    "mapp_group": {"parent_label_raw": "protein_ids_raw"},
    "mapp_channel": {
        "source_file": "raw_file",
        "sample_label_raw": "experiment_raw",
        "condition_text": "condition",
        "organism_name": "measured_species",
        "cell_line_name": "measured_cell",
        "search_enzyme": "search_enzyme_mode_raw",
    },
    "mapp_identification": {
        "observation_id": "evidence_id",
        "peptide_sequence": "sequence_raw",
        "peptide_modifications_raw": "modifications_raw",
        "quantitative_value": "intensity",
        "value_status": "intensity_status",
    },
    "mapp_quantity_measurement": {"measurement_unit": "unit", "quantitative_value": "value"},
    "mapp_source_assertion": {
        "claim_id": "assertion_id",
        "assertion_predicate": "predicate",
        "assertion_value": "value_raw",
        "source_scope": "scope",
    },
    "mapp_publisher_cell": {
        "source_cell": "cell_id",
        "source_file": "asset_path",
        "quantitative_raw": "value_text",
    },
    "mapp_publisher_quantity": {
        "condition_text": "condition",
        "measurement_unit": "unit",
        "quantitative_value": "value",
    },
    "external_reference_protein": {"reference_sequence": "sequence", "accession_raw": "accession"},
    "external_peptide_parent_occurrence": {"mapping_position": "start_1based"},
    "external_source_parent_claim": {
        "parent_label_raw": "source_accession_raw",
        "mapping_status": "resolution_status",
    },
    "external_source_parent_resolution": {"mapping_status": "occurrence_status"},
    "external_peptide_reference_coverage": {"n_mapping_occurrences": "n_occurrences"},
    "mhc_positive_ms": {"source_reference_id": "source_reference_iri"},
    "mhc_binding": {"source_reference_id": "source_reference_iri"},
    "mhc_other": {"source_reference_id": "source_reference_iri"},
    "gomez_derived_arm_support": {"supported_arm_ids": "supported_arm_ids_json"},
    "gomez_derived_arm_projection": {"supported_arm_ids": "supported_arm_ids_json"},
    "tcell_availability": {
        "assay_readout": "response_measured",
        "qualitative_result": "qualitative_measurement",
        "quantitative_raw": "quantitative_measurement",
        "measurement_unit": "measurement_units",
        "restriction_evidence": "restriction_evidence_code",
    },
    "tcell_functional_peptide": {
        "assay_readout": "response_measured",
        "qualitative_result": "source_qualitative_result",
        "quantitative_raw": "quantitative_measurement",
        "measurement_unit": "measurement_units",
        "measurement_inequality": "measurement_inequality",
        "peptide_sequence": "peptide",
        "restriction_evidence": "restriction_evidence_code",
    },
    "tcell_multimer_peptide": {
        "assay_readout": "response_measured",
        "qualitative_result": "source_qualitative_result",
        "quantitative_raw": "quantitative_measurement",
        "measurement_unit": "measurement_units",
        "peptide_sequence": "peptide",
        "restriction_evidence": "restriction_evidence_code",
    },
    "tcell_reported_outcome": {
        "endpoint_reported": "response_measured_raw",
        "qualitative_result": "qualitative_result_raw",
        "measurement_unit": "measurement_unit_raw",
        "measurement_inequality": "measurement_inequality_raw",
    },
    "tcell_antigen_claim": {
        "peptide_modifications_raw": "epitope_modifications_raw",
        "reported_mhc": "reported_mhc_name_raw",
        "restriction_evidence": "restriction_evidence_code_raw",
    },
    "tcell_source_contributor": {"source_row_index": "csv_data_row_1based"},
    "tcell_cedar_mutation_claim": {"epitope_mutation_raw": "value_raw"},
    "tcell_eligibility": {
        "eligibility_status": "training_status",
        "exclusion_reason": "training_exclusion_reason",
    },
    "bulk_ms_peptide": {
        "cell_line_name": "cell_line_name_raw",
        "digestion_enzyme": "digestion_enzyme_raw",
    },
    "bulk_ms_protein": {
        "gene_name": "gene_symbol",
        "cell_line_name": "cell_line_name_raw",
        "digestion_enzyme": "digestion_enzyme_raw",
    },
    "peptide_protein_occurrence": {
        "peptide_sequence": "peptide",
        "mapping_position": "position",
    },
    "bulk_protocol_scope": {"digestion_enzyme": "digestion_enzyme_raw"},
    "line_expression_source": {"source_dataset": "source_id"},
    "line_anchor_claim": {
        "line_key": "expression_key",
        "material_link_status": "material_link_precision",
    },
    "external_mhc_peptide_quantity": {
        "observation_id": "result_id",
        "source_record_id": "peptide_row_id",
        "source_cell": "source_intensity_cell",
    },
    "external_digest_peptide_quantity": {
        "observation_id": "result_id",
        "source_record_id": "peptide_row_id",
        "source_cell": "source_intensity_cell",
    },
    "external_quantity_measurement": {"measurement_unit": "unit"},
    "external_peptide_source_row": {"peptide_sequence": "sequence_raw"},
    "external_source_panel": {"source_panel_id": "panel_id"},
    "external_source_cell": {
        "source_cell": "cell_ref",
        "quantitative_raw": "value_text",
    },
    "external_panel_sequence": {
        "source_panel_id": "panel_id",
        "source_record_id": "panel_record_id",
        "peptide_sequence": "sequence_raw",
        "source_cell": "sequence_cell",
    },
    "external_contrast": {"channel_id": "treated_channel_id"},
    "external_context_assertion": {
        "claim_id": "assertion_id",
        "assertion_predicate": "predicate",
        "assertion_value": "value_raw",
        "source_scope": "scope",
    },
    "external_sdrf_cell": {
        "source_cell": "sdrf_cell_id",
        "source_file": "data_file",
        "sample_label_raw": "source_name",
    },
    "external_deposited_file": {
        "source_file": "file_name",
    },
    "external_source_asset": {
        "source_asset_id": "asset_id",
        "source_file": "relative_path",
    },
    "source_contributor": {
        "observation_id": "provenance_id",
        "contributor_status": "relationship_status",
    },
    "curated_study": {"source_reference_id": "study_pmid"},
    "curated_material_arm": {"source_reference_id": "study_pmid"},
    "arm_factor": {
        "curation_source_field": "source_field",
        "factor_value": "token",
        "condition_combination": "combination",
    },
    "arm_mhc_claim": {
        "mhc_claim_role": "role",
        "reported_mhc": "raw_value",
        "evidence_basis": "evidence_basis",
        "genotype_cell": "genotype_cell",
        "genotype_source": "genotype_source",
    },
    "protocol_claim": {
        "protocol_field": "field_name",
        "protocol_value": "raw_value",
    },
    "curation_amendment": {
        "amendment_field": "field_name",
        "old_value": "old_value_json",
        "new_value": "new_value_json",
    },
    "reviewed_lineage": {"lineage_axis": "axis", "lineage_status": "status"},
    "external_reference_source_mapping": {
        "source_record_id": "peptide_row_id",
        "peptide_sequence": "sequence_raw",
    },
    "external_reference_sequence": {"peptide_sequence": "sequence"},
    "external_reference_result_context": {"source_record_id": "peptide_row_id"},
    "gomez_prediction_row": {
        "prediction_id": "source_prediction_id",
    },
    "gomez_restriction_projection": {
        "selected_prediction_ids": "selected_prediction_ids_json",
        "alternative_prediction_ids": "alternative_prediction_ids_json",
    },
}

ALIASES.update(
    {
        "pbmc_mapp_source_record": ALIASES["mapp_source_record"],
        "pbmc_mapp_peptide": ALIASES["mapp_peptide"],
        "pbmc_mapp_group": ALIASES["mapp_group"],
        "pbmc_mapp_channel": ALIASES["mapp_channel"],
        "pbmc_mapp_identification": ALIASES["mapp_identification"],
        "pbmc_mapp_quantity_measurement": ALIASES["mapp_quantity_measurement"],
        "pbmc_mapp_source_assertion": ALIASES["mapp_source_assertion"],
        "external_pbmc_proteasome_fragment_quantity": {},
    }
)

# Dataset identity is an immutable property of these source-specific files.
CONSTANTS = {
    name: {"source_dataset": "PXD057588"}
    for name in (
        "external_mhc_peptide_quantity",
        "external_digest_peptide_quantity",
        "external_peptide_source_row",
        "external_source_panel",
        "external_source_cell",
        "external_panel_sequence",
        "external_assay_channel",
        "external_contrast",
        "external_context_assertion",
        "external_sdrf_cell",
        "external_deposited_file",
        "external_source_asset",
        "external_quantity_measurement",
        "external_reference_source_mapping",
        "external_reference_sequence",
        "external_reference_occurrence",
        "external_reference_parent_support",
        "external_reference_result_context",
    )
}
CONSTANTS["external_reference_parent"] = {
    "source_dataset": "PXD057588",
    "organism_taxon_id": "10090",
}
CONSTANTS["external_reference_source_mapping"]["organism_taxon_id"] = "10090"

CONSTANTS.update(
    {
        item.name: {"source_dataset": item.source_population}
        for item in EVIDENCE
        if item.source_population in ("PXD009092", "PXD009438", "PXD004886")
        and "source_dataset" in item.fields
    }
)

LINKS = (
    Link(
        "tmab_measurement_source_row",
        "tmab_measurement",
        "tmab_source_row",
        "source_record_id",
        "source_record_id",
        "one",
        "native_measurement_to_lossless_workbook_row",
    ),
    Link(
        "tmab_measurement_context",
        "tmab_measurement",
        "tmab_protocol_context",
        "context_id",
        "context_id",
        "one",
        "native_measurement_to_source_experiment_context",
    ),
    Link(
        "tmab_lineage_measurement",
        "tmab_measurement_lineage",
        "tmab_measurement",
        "measurement_id",
        "measurement_id",
        "one",
        "lineage_assessment_to_native_measurement",
    ),
    Link(
        "binding_context_availability_observation",
        "binding_context_availability",
        "mhc_binding",
        "observation_id",
        "observation_id",
        "one",
        "one_row_preserving_binding_context_audit",
    ),
    Link(
        "prediction_derived_observation",
        "gomez_prediction_row",
        "mhc_positive_ms",
        "derived_observation_id",
        "observation_id",
        "optional_one",
        "exact_prediction_sequence_and_supplementary_file_link",
    ),
    Link(
        "restriction_projection_observation",
        "gomez_restriction_projection",
        "mhc_positive_ms",
        "derived_observation_id",
        "observation_id",
        "optional_one",
        "explicit_prediction_projection_to_derived_observation",
    ),
    Link(
        "membership",
        "gomez_source_result",
        "gomez_peptide_membership",
        "membership_id",
        "membership_id",
        "one",
        "source_result_to_reported_sample_peptide_membership",
    ),
    Link(
        "derived_peptide",
        "gomez_peptide_membership",
        "mhc_positive_ms",
        "derived_observation_id",
        "observation_id",
        "optional_one",
        "sequence_and_source_file_match_to_merged_derived_peptide",
    ),
    Link(
        "reporter",
        "calibration_s4_measurement",
        "calibration_reporter_form",
        "reporter_id",
        "reporter_id",
        "one",
        "original_reporter_form",
    ),
    Link(
        "source_slot",
        "calibration_s4_measurement",
        "calibration_source_slot",
        "slot_id",
        "slot_id",
        "one",
        "source_column_not_acquisition_identity",
    ),
    Link(
        "nominal_load",
        "calibration_s4_measurement",
        "calibration_nominal_load",
        "nominal_load_id",
        "nominal_load_id",
        "one",
        "nominal_panel_load_not_form_molarity",
    ),
    Link(
        "standard",
        "calibration_reporter_form",
        "calibration_standard",
        "standard_id",
        "standard_id",
        "one",
        "unique_sequence_panel_membership",
    ),
    Link(
        "reporter_forms",
        "calibration_standard",
        "calibration_reporter_form",
        "standard_id",
        "standard_id",
        "many",
        "molecular_form_alternatives",
    ),
    Link(
        "reporter",
        "calibration_curve",
        "calibration_reporter_form",
        "reporter_id",
        "reporter_id",
        "one",
        "original_reporter_form",
    ),
    Link(
        "curve",
        "calibration_curve_measurement",
        "calibration_curve",
        "curve_id",
        "curve_id",
        "one",
        "reported_curve_row",
    ),
    Link(
        "reporter",
        "calibration_summary_measurement",
        "calibration_reporter_form",
        "reporter_id",
        "reporter_id",
        "one",
        "original_reporter_form",
    ),
    Link(
        "source_slot",
        "calibration_normalization_diagnostic",
        "calibration_source_slot",
        "slot_id",
        "slot_id",
        "one",
        "source_column_not_acquisition_identity",
    ),
    Link(
        "source_cell",
        "calibration_s4_measurement",
        "calibration_source_cell",
        "source_cell",
        "cell_id",
        "one",
        "original_source_coordinate",
    ),
    Link(
        "source_cell",
        "calibration_curve_measurement",
        "calibration_source_cell",
        "source_cell",
        "cell_id",
        "one",
        "original_source_coordinate",
    ),
    Link(
        "source_cell",
        "calibration_summary_measurement",
        "calibration_source_cell",
        "source_cell",
        "cell_id",
        "one",
        "original_source_coordinate",
    ),
    Link(
        "peptide",
        "external_proteasome_fragment_quantity",
        "mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "channel",
        "external_proteasome_fragment_quantity",
        "mapp_channel",
        "channel_id",
        "channel_id",
        "one",
        "source_acquisition",
    ),
    Link(
        "material_context",
        "mapp_channel",
        "mapp_material_context",
        "channel_id",
        "context_id",
        "one",
        "source_acquisition_context",
    ),
    Link(
        "source_material_context",
        "mapp_material_link",
        "mapp_material_context",
        "source_context_id",
        "context_id",
        "one",
        "qualified_source_context",
    ),
    Link(
        "target_material_context",
        "mapp_material_link",
        "mapp_material_context",
        "target_context_id",
        "context_id",
        "one",
        "qualified_target_context",
    ),
    Link(
        "measurements",
        "external_proteasome_fragment_quantity",
        "mapp_quantity_measurement",
        "result_id",
        "result_id",
        "many",
        "result_representation",
    ),
    Link(
        "result",
        "mapp_quantity_measurement",
        "external_proteasome_fragment_quantity",
        "result_id",
        "result_id",
        "one",
        "source_result",
    ),
    Link(
        "peptide",
        "mapp_quantity_measurement",
        "mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "peptide",
        "mapp_publisher_quantity",
        "mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "cell",
        "mapp_publisher_quantity",
        "mapp_publisher_cell",
        "source_cell",
        "cell_id",
        "one",
        "source_workbook_cell",
    ),
    Link(
        "channel",
        "mapp_identification",
        "mapp_channel",
        "channel_id",
        "channel_id",
        "one",
        "source_acquisition",
    ),
    Link(
        "peptide",
        "mapp_identification",
        "mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "source",
        "mapp_peptide",
        "mapp_source_record",
        "source_record_id",
        "source_record_id",
        "one",
        "source_table_row",
    ),
    Link(
        "occurrences",
        "mapp_peptide",
        "external_peptide_parent_occurrence",
        "peptide_row_id",
        "peptide_row_id",
        "many",
        "reference_alternatives",
    ),
    Link(
        "material_context",
        "processing_channel",
        "processing_material_context",
        "channel_id",
        "context_id",
        "one",
        "source_acquisition_context",
    ),
    Link(
        "source_material_context",
        "processing_material_link",
        "processing_material_context",
        "source_context_id",
        "context_id",
        "one",
        "qualified_source_context",
    ),
    Link(
        "target_material_context",
        "processing_material_link",
        "processing_material_context",
        "target_context_id",
        "context_id",
        "one",
        "qualified_target_context",
    ),
    Link(
        "protein",
        "external_peptide_parent_occurrence",
        "external_reference_protein",
        "reference_record_id",
        "reference_record_id",
        "one",
        "reference_record",
    ),
    Link(
        "source_contributors",
        "mhc_positive_ms",
        "source_contributor",
        "observation_id",
        "provenance_id",
        "many",
        "source_record",
    ),
    Link(
        "source_contributors",
        "mhc_binding",
        "source_contributor",
        "observation_id",
        "provenance_id",
        "many",
        "source_record",
    ),
    Link(
        "source_contributors",
        "mhc_other",
        "source_contributor",
        "observation_id",
        "provenance_id",
        "many",
        "source_record",
    ),
    Link(
        "source_contributors",
        "mhc_reported_measurement",
        "source_contributor",
        "observation_id",
        "provenance_id",
        "many",
        "source_record",
    ),
    Link(
        "reported_measurement",
        "mhc_positive_ms",
        "mhc_reported_measurement",
        "observation_id",
        "observation_id",
        "one",
        "assay_result",
    ),
    Link(
        "reported_measurement",
        "mhc_binding",
        "mhc_reported_measurement",
        "observation_id",
        "observation_id",
        "one",
        "assay_result",
    ),
    Link(
        "reported_measurement",
        "mhc_other",
        "mhc_reported_measurement",
        "observation_id",
        "observation_id",
        "one",
        "assay_result",
    ),
    Link(
        "assay_context",
        "mhc_positive_ms",
        "assay_context",
        "observation_id",
        "observation_id",
        "optional_one",
        "assay_result",
    ),
    Link(
        "assay_context",
        "mhc_binding",
        "assay_context",
        "observation_id",
        "observation_id",
        "optional_one",
        "assay_result",
    ),
    Link(
        "binding_assay_context",
        "mhc_binding",
        "binding_assay_context",
        "observation_id",
        "observation_id",
        "one",
        "review_state_for_each_binding_result",
    ),
    Link(
        "protocol_claims",
        "binding_assay_context",
        "binding_protocol_claim",
        "protocol_id",
        "protocol_id",
        "many",
        "source_scoped_protocol_assertions",
    ),
    Link(
        "assay_context",
        "mhc_other",
        "assay_context",
        "observation_id",
        "observation_id",
        "optional_one",
        "assay_result",
    ),
    Link(
        "gomez_membership_arm",
        "gomez_source_membership_arm",
        "gomez_source_arm",
        "arm_id",
        "arm_id",
        "one",
        "source_backed_arm_definition",
    ),
    Link(
        "gomez_unique_arm_projection",
        "gomez_derived_arm_projection",
        "gomez_source_arm",
        "unique_arm_id",
        "arm_id",
        "optional_one",
        "unique_source_arm_projection",
    ),
    Link(
        "gomez_projection_context",
        "gomez_assay_context_projection",
        "gomez_derived_arm_projection",
        "observation_id",
        "derived_observation_id",
        "one",
        "source_backed_context_projection",
    ),
    Link(
        "gomez_projection_assay",
        "gomez_assay_context_projection",
        "mhc_positive_ms",
        "observation_id",
        "observation_id",
        "optional_one",
        "assay_result",
    ),
    Link(
        "reported_outcome",
        "tcell_availability",
        "tcell_reported_outcome",
        "source_assay_id",
        "source_assay_id",
        "one",
        "source_assay",
    ),
    Link(
        "antigen_claim",
        "tcell_availability",
        "tcell_antigen_claim",
        "source_assay_id",
        "source_assay_id",
        "one",
        "source_assay",
    ),
    Link(
        "source_contributors",
        "tcell_availability",
        "tcell_source_contributor",
        "source_assay_id",
        "source_assay_id",
        "many",
        "source_assay",
    ),
    Link(
        "source_contributors",
        "tcell_functional_peptide",
        "tcell_source_contributor",
        "source_assay_id",
        "source_assay_id",
        "many",
        "source_assay",
    ),
    Link(
        "source_contributors",
        "tcell_multimer_peptide",
        "tcell_source_contributor",
        "source_assay_id",
        "source_assay_id",
        "many",
        "source_assay",
    ),
    Link(
        "adjudication",
        "tcell_availability",
        "assay_adjudication",
        "source_assay_id",
        "source_assay_id",
        "optional_one",
        "source_assay",
    ),
    Link(
        "source_assay",
        "assay_adjudication",
        "tcell_availability",
        "source_assay_id",
        "source_assay_id",
        "one",
        "source_assay",
    ),
    Link(
        "reported_outcome",
        "assay_adjudication",
        "tcell_reported_outcome",
        "source_assay_id",
        "source_assay_id",
        "one",
        "source_assay",
    ),
    Link(
        "antigen_claim",
        "assay_adjudication",
        "tcell_antigen_claim",
        "source_assay_id",
        "source_assay_id",
        "one",
        "source_assay",
    ),
    Link(
        "eligibility",
        "assay_adjudication",
        "tcell_eligibility",
        "source_assay_id",
        "source_assay_id",
        "one",
        "source_assay",
    ),
    Link(
        "source_contributors",
        "assay_adjudication",
        "tcell_source_contributor",
        "source_assay_id",
        "source_assay_id",
        "many",
        "source_assay",
    ),
    Link(
        "numeric_measurements",
        "bulk_ms_protein",
        "bulk_numeric_measurement",
        "observation_id",
        "observation_id",
        "many",
        "protein_result",
    ),
    Link(
        "protocol_scope",
        "bulk_ms_peptide",
        "bulk_protocol_scope",
        "protocol_scope_id",
        "protocol_scope_id",
        "one",
        "source_protocol",
    ),
    Link(
        "protocol_scope",
        "bulk_ms_protein",
        "bulk_protocol_scope",
        "protocol_scope_id",
        "protocol_scope_id",
        "one",
        "source_protocol",
    ),
    Link(
        "numeric_measurements",
        "line_expression_gene",
        "line_expression_measurement",
        "observation_id",
        "observation_id",
        "many",
        "gene_profile",
    ),
    Link(
        "assay_channel",
        "external_mhc_peptide_quantity",
        "external_assay_channel",
        "channel_id",
        "channel_id",
        "one",
        "assay_channel",
    ),
    Link(
        "assay_channel",
        "external_digest_peptide_quantity",
        "external_assay_channel",
        "channel_id",
        "channel_id",
        "one",
        "assay_channel",
    ),
    Link(
        "peptide_source_row",
        "external_mhc_peptide_quantity",
        "external_peptide_source_row",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "peptide_source_row",
        "external_digest_peptide_quantity",
        "external_peptide_source_row",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "quantity_measurements",
        "external_mhc_peptide_quantity",
        "external_quantity_measurement",
        "result_id",
        "result_id",
        "many",
        "assay_channel_result",
    ),
    Link(
        "quantity_measurements",
        "external_digest_peptide_quantity",
        "external_quantity_measurement",
        "result_id",
        "result_id",
        "many",
        "assay_channel_result",
    ),
    Link(
        "reference_mapping",
        "external_peptide_source_row",
        "external_reference_source_mapping",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "reference_context",
        "external_mhc_peptide_quantity",
        "external_reference_result_context",
        "result_id",
        "result_id",
        "one",
        "assay_channel_result",
    ),
    Link(
        "reference_context",
        "external_digest_peptide_quantity",
        "external_reference_result_context",
        "result_id",
        "result_id",
        "one",
        "assay_channel_result",
    ),
    Link(
        "reference_context",
        "external_quantity_measurement",
        "external_reference_result_context",
        "result_id",
        "result_id",
        "one",
        "assay_channel_result",
    ),
    Link(
        "reference_sequence",
        "external_reference_source_mapping",
        "external_reference_sequence",
        "sequence_id",
        "sequence_id",
        "one",
        "deduplicated_source_sequence",
    ),
    Link(
        "reference_occurrences",
        "external_reference_sequence",
        "external_reference_occurrence",
        "sequence_id",
        "sequence_id",
        "many",
        "reference_position_alternatives",
    ),
    Link(
        "reference_parent",
        "external_reference_occurrence",
        "external_reference_parent",
        "accession",
        "accession",
        "one",
        "reference_identity",
    ),
    Link(
        "source_mapping",
        "external_reference_result_context",
        "external_reference_source_mapping",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
)


ALIASES.update({new: dict(ALIASES.get(old, {})) for old, new in CONTROLLED_RELATION_NAMES.items()})
ALIASES["processing_label_alias"] = {"source_record_id": "alias_id"}
LINKS += tuple(
    replace(
        link,
        source=CONTROLLED_RELATION_NAMES[link.source],
        target=CONTROLLED_RELATION_NAMES[link.target],
    )
    for link in LINKS
    if link.source in CONTROLLED_RELATION_NAMES and link.target in CONTROLLED_RELATION_NAMES
)


ALIASES.update(
    {"digestion_" + name: dict(aliases) for name, (_, aliases) in DIGESTION_RELATIONS.items()}
)
for _table in ("raw_quantities", "ms1", "identifications", "conflicts"):
    LINKS += (
        Link(
            "acquisition",
            "digestion_" + _table,
            "digestion_runs",
            "run_id",
            "run_id",
            "one",
            "exact_original_acquisition",
        ),
    )
for _table in ("raw_quantities", "filtered_vectors", "final_vectors", "final_values", "contexts"):
    LINKS += (
        Link(
            "substrate",
            "digestion_" + _table,
            "digestion_constructs",
            "construct_id",
            "construct_id",
            "one",
            "exact_original_construct",
        ),
    )
for _table in (
    "raw_quantities",
    "filtered_vectors",
    "final_vectors",
    "final_values",
    "ms1",
    "identifications",
):
    LINKS += (
        Link(
            "condition",
            "digestion_" + _table,
            "digestion_contexts",
            "condition_id",
            "condition_id",
            "one",
            "source_condition_not_APC_specimen",
        ),
    )
LINKS += (
    Link(
        "source_vector",
        "digestion_final_values",
        "digestion_final_vectors",
        "vector_id",
        "source_record_id",
        "one",
        "original_vector_slot",
    ),
)


for _table in ("quantities", "protein_quantities", "occurrences"):
    LINKS += (
        Link(
            "source_row",
            "hap1_" + _table,
            "hap1_source_records",
            "source_record_id",
            "source_record_id",
            "one",
            "exact_original_source_row",
        ),
    )
for _table in ("quantities", "protein_quantities", "precursors", "runs"):
    LINKS += (
        Link(
            "sample",
            "hap1_" + _table,
            "hap1_samples",
            "sample_id",
            "sample_id",
            "optional_one" if _table in ("precursors", "runs", "protein_quantities") else "one",
            "declared_modality_sample_not_cross_modality_specimen_pair",
        ),
    )
for _table in ("quantities", "protein_quantities", "precursors", "runs", "samples"):
    LINKS += (
        Link(
            "context",
            "hap1_" + _table,
            "hap1_contexts",
            "context_id",
            "context_id",
            "optional_one" if _table in ("precursors", "runs", "protein_quantities") else "one",
            "source_condition_and_protocol_scope",
        ),
    )
for _table in ("protein_quantities", "precursors"):
    LINKS += (
        Link(
            "run", "hap1_" + _table, "hap1_runs", "run_id", "run_id", "one", "original_source_run"
        ),
    )
LINKS += (
    Link(
        "reference",
        "hap1_occurrences",
        "hap1_references",
        "reference_record_id",
        "reference_record_id",
        "optional_one",
        "exact_sequence_candidate_not_measured_origin",
    ),
)


for _table in ("runs", "peptides", "forms"):
    LINKS += (
        Link(
            "source_record",
            "proteometools_" + _table,
            "proteometools_source_records",
            "source_record_id",
            "source_record_id",
            "one",
            "literal_source_row",
        ),
    )
for _table in ("peptides", "forms"):
    LINKS += (
        Link(
            "run",
            "proteometools_" + _table,
            "proteometools_runs",
            "run_id",
            "run_id",
            "one",
            "deposited_single_run",
        ),
    )
LINKS += (
    Link(
        "peptide",
        "proteometools_forms",
        "proteometools_peptides",
        "peptide_measurement_id",
        "measurement_id",
        "one",
        "original_MaxQuant_Peptide_ID",
    ),
    Link(
        "context",
        "proteometools_runs",
        "proteometools_contexts",
        "context_id",
        "context_id",
        "one",
        "primary_study_protocol",
    ),
)


for _table in ("observations", "quantities"):
    LINKS += (
        Link(
            "source_record",
            "lymph_" + _table,
            "lymph_source_records",
            "source_record_id",
            "source_record_id",
            "one",
            "original_independent_source_block_row",
        ),
    )
for _table in ("samples", "quantities"):
    LINKS += (
        Link(
            "context",
            "lymph_" + _table,
            "lymph_contexts",
            "context_id",
            "context_id",
            "one",
            "source_anatomical_and_modality_scope",
        ),
    )
LINKS += (
    Link(
        "sample",
        "lymph_quantities",
        "lymph_samples",
        "sample_id",
        "sample_id",
        "one",
        "original_channel_label_not_established_independent_specimen",
    ),
    Link(
        "observation",
        "lymph_quantities",
        "lymph_observations",
        "observation_id",
        "observation_id",
        "one",
        "original_peptidoform_feature_row",
    ),
    Link(
        "reference",
        "lymph_identifications",
        "lymph_references",
        "reference_record_id",
        "reference_record_id",
        "one",
        "deposited_search_reference_alternative",
    ),
)

LINKS += (
    Link(
        "peptide",
        "external_pbmc_proteasome_fragment_quantity",
        "pbmc_mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "channel",
        "external_pbmc_proteasome_fragment_quantity",
        "pbmc_mapp_channel",
        "channel_id",
        "channel_id",
        "one",
        "source_acquisition",
    ),
    Link(
        "material_context",
        "pbmc_mapp_channel",
        "pbmc_mapp_material_context",
        "channel_id",
        "context_id",
        "one",
        "source_acquisition_context",
    ),
    Link(
        "source_material_context",
        "pbmc_mapp_material_link",
        "pbmc_mapp_material_context",
        "source_context_id",
        "context_id",
        "one",
        "qualified_source_context",
    ),
    Link(
        "target_material_context",
        "pbmc_mapp_material_link",
        "pbmc_mapp_material_context",
        "target_context_id",
        "context_id",
        "one",
        "qualified_target_context",
    ),
    Link(
        "measurements",
        "external_pbmc_proteasome_fragment_quantity",
        "pbmc_mapp_quantity_measurement",
        "result_id",
        "result_id",
        "many",
        "result_representation",
    ),
    Link(
        "result",
        "pbmc_mapp_quantity_measurement",
        "external_pbmc_proteasome_fragment_quantity",
        "result_id",
        "result_id",
        "one",
        "source_result",
    ),
    Link(
        "peptide",
        "pbmc_mapp_quantity_measurement",
        "pbmc_mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "channel",
        "pbmc_mapp_identification",
        "pbmc_mapp_channel",
        "channel_id",
        "channel_id",
        "one",
        "source_acquisition",
    ),
    Link(
        "peptide",
        "pbmc_mapp_identification",
        "pbmc_mapp_peptide",
        "peptide_row_id",
        "peptide_row_id",
        "one",
        "source_peptide_row",
    ),
    Link(
        "source",
        "pbmc_mapp_peptide",
        "pbmc_mapp_source_record",
        "source_record_id",
        "source_record_id",
        "one",
        "source_table_row",
    ),
)


for _table in (
    "acquisitions",
    "groups",
    "references",
    "peptides",
    "forms",
    "identifications",
    "quantities",
    "assertions",
    "publisher_cells",
    "publisher_quantities",
    "publisher_links",
    "pooled_slots",
):
    LINKS += (
        Link(
            "source_record",
            "goldberg_" + _table,
            "goldberg_source_records",
            "source_record_id",
            "source_record_id",
            "many_to_one",
            "original_source_row_not_physical_sample",
        ),
    )
LINKS += (
    Link(
        "peptide",
        "goldberg_quantities",
        "goldberg_peptides",
        "peptide_id",
        "peptide_id",
        "many_to_one",
        "original_peptide_identity",
    ),
    Link(
        "acquisition",
        "goldberg_quantities",
        "goldberg_acquisitions",
        "acquisition_id",
        "acquisition_id",
        "many_to_one",
        "source_acquisition_not_specimen_or_aliquot",
    ),
)


def materialized_bindings(
    package_schema_version=None, *, appended_source_ids=(), include_adjudications=False
):
    """Return each declared physical relation with an exact row selector."""
    return {
        item.name: (item, SELECTORS.get(item.name, ()), ALIASES.get(item.name, {}))
        for item in EVIDENCE
        if item.status == "v2_materialized"
        and (
            package_schema_version is None
            or package_schema_version >= item.min_package_schema_version
        )
        and (
            package_schema_version is None
            or item.required_appended_source is None
            or item.required_appended_source in appended_source_ids
        )
        and (
            package_schema_version is None
            or item.name != "assay_adjudication"
            or include_adjudications
        )
    }
