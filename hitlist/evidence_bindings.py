"""Exact physical bindings for the versioned evidence catalog.

Logical aliases are admitted only when a physical column has the same meaning
at the same grain. Every physical column remains queryable as ``native.<name>``.
Selectors distinguish evidence types that intentionally share one Parquet file.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence_catalog import EVIDENCE


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
}

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
        item.name: {"source_dataset": "PXD009092"}
        for item in EVIDENCE
        if item.source_population == "PXD009092" and "source_dataset" in item.fields
    }
)

LINKS = (
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
        "assay_context",
        "mhc_other",
        "assay_context",
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


def materialized_bindings(package_schema_version=None, *, appended_source_ids=()):
    """Return each declared physical relation with an exact row selector."""
    return {
        item.name: (item, SELECTORS.get(item.name, ()), ALIASES.get(item.name, {}))
        for item in EVIDENCE
        if item.status == "v2_materialized"
        and (
            package_schema_version is None
            or package_schema_version >= 7
            or item.source_population != "pxd057588_mouse_reference_2026_03"
        )
        and (
            package_schema_version is None
            or item.source_population != "PXD009092"
            or "pxd009092" in appended_source_ids
        )
    }
