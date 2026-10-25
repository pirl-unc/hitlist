"""Source-neutral grains for native intracellular peptide evidence."""

import pyarrow as pa

from .goldberg_mapp import SCHEMAS as MAXQUANT_SCHEMAS
from .intact_degradomics_schema import SCHEMAS as CELL_SCHEMAS
from .mapp_evidence import _schema
from .material_context import MATERIAL_CONTEXT_SCHEMA, MATERIAL_LINK_SCHEMA


def extend(schema, text=(), integers=()):
    return pa.schema(
        [*schema, *[(n, pa.string()) for n in text], *[(n, pa.int64()) for n in integers]]
    )


SCHEMAS = {
    name: MAXQUANT_SCHEMAS[name]
    for name in ("source_records", "peptides", "forms", "identifications", "assertions")
}
SCHEMAS.update(
    {
        "source_records": extend(MAXQUANT_SCHEMAS["source_records"], ("source_asset_id",)),
        "peptides": extend(MAXQUANT_SCHEMAS["peptides"], ("gene_names_raw", "protein_names_raw")),
        "identifications": pa.schema(
            [
                *extend(MAXQUANT_SCHEMAS["identifications"], ("charge_raw",)),
                ("charge_state", pa.float64()),
            ]
        ),
        "assets": extend(
            MAXQUANT_SCHEMAS["assets"], ("license_status", "source_member_provenance_json")
        ),
        "source_cells": CELL_SCHEMAS["source_cells"],
        "acquisitions": extend(
            MAXQUANT_SCHEMAS["acquisitions"], ("condition_id", "replicate_status")
        ),
        "references": extend(
            MAXQUANT_SCHEMAS["references"],
            ("reference_sequence_sha256", "reference_role"),
            ("n_reference_residues",),
        ),
        "quantities": extend(
            MAXQUANT_SCHEMAS["quantities"],
            (
                "observation_kind",
                "form_id",
                "identification_id",
                "normalization_status",
                "quantity_form_scope",
            ),
        ),
        "declared_parents": _schema(
            (
                "parent_claim_id",
                "source_record_id",
                "peptide_id",
                "observation_id",
                "parent_accession",
                "reference_record_id",
                "reference_link_status",
            )
        ),
        "origins": _schema(
            (
                "origin_id",
                "source_record_id",
                "peptide_id",
                "reference_record_id",
                "reference_sequence_sha256",
                "origin_match_type",
                "reference_role",
            ),
            integers=("candidate_start_1based", "candidate_end_1based"),
        ),
        "dependency_groups": _schema(("peptide_id", "dependency_group_id", "dependency_policy")),
        "reconciliations": _schema(
            (
                "claim_id",
                "source_record_id",
                "source_subject_id",
                "target_subject_id",
                "assertion_predicate",
                "assertion_status",
                "source_locator",
                "details_json",
            )
        ),
        "material_contexts": MATERIAL_CONTEXT_SCHEMA,
        "material_links": MATERIAL_LINK_SCHEMA,
    }
)
SCHEMAS = {
    name: extend(schema, ("source_dataset", "source_dataset_precision", "source_panel"))
    for name, schema in SCHEMAS.items()
}

# Grain, evidence kind and exact primary key. No row is automatic training data.
RELATIONS = {
    "assets": (
        "one_pinned_original_asset_or_verified_archive_member",
        "relation",
        "source_asset_id",
    ),
    "source_records": (
        "one_original_text_row_XML_node_or_reference_record",
        "relation",
        "source_record_id",
    ),
    "source_cells": ("one_original_publisher_OOXML_cell", "relation", "cell_id"),
    "acquisitions": (
        "one_reported_acquisition_not_inferred_physical_material",
        "context",
        "acquisition_id",
    ),
    "references": (
        "one_record_in_the_exact_deposited_reference",
        "relation",
        "reference_record_id",
    ),
    "peptides": ("one_original_sequence_level_peptide_identity", "observation", "peptide_id"),
    "forms": ("one_original_modification_specific_peptide_identity", "observation", "form_id"),
    "identifications": (
        "one_original_feature_identification_or_matching_record",
        "observation",
        "identification_id",
    ),
    "quantities": (
        "one_original_quantity_at_its_entity_acquisition_and_processing_grain",
        "measurement",
        "measurement_id",
    ),
    "declared_parents": (
        "one_source_parent_claim_and_its_reference_alternative",
        "relation",
        "parent_claim_id",
    ),
    "origins": ("one_candidate_span_in_one_exact_reference_record", "relation", "origin_id"),
    "dependency_groups": (
        "one_peptide_membership_in_an_all_source_dependency_component",
        "relation",
        "peptide_id",
    ),
    "assertions": ("one_source_scoped_assertion_or_explicit_unknown", "relation", "claim_id"),
    "reconciliations": (
        "one_original_view_identifier_or_quantity_comparison",
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
    "gene_names_raw": "Original gene annotation from the deposited text; alternatives remain uncollapsed",
    "protein_names_raw": "Original source protein-name annotation, separate from resolved parent origins",
    "charge_raw": "Original feature charge-state literal, including missing or invalid representations",
    "source_member_provenance_json": "Original archive member/range provenance; partial retrieval does not verify the full archive digest",
    "source_panel": "Source-supported panel within the deposited project, not an inferred specimen",
    "reference_sequence_sha256": "SHA256 of the complete exact reference sequence, preserving reference alternatives",
    "reference_role": "Qualified reference provenance; engineered constructs remain distinct even with a human taxonomy tag",
    "n_reference_residues": "Number of residues in this exact reference record",
    "observation_kind": "Entity grain of this reported quantity: sequence, modification-specific form or feature",
    "normalization_status": "Source-supported normalization semantics; raw signal and LFQ columns are separate",
    "quantity_form_scope": "Whether the reported quantity aggregates sequence forms or addresses a specified modification/feature",
    "parent_claim_id": "One source record, reported parent token and candidate reference link",
    "reference_link_status": "Whether a reported accession resolves in the original deposited reference; no preferred-parent imputation",
    "origin_id": "One peptide identity, exact reference record and complete candidate span",
    "origin_match_type": "Literal sequence agreement versus a separate isoleucine/leucine mass-equivalent alternative",
    "candidate_start_1based": "Computed inclusive one-based candidate start; not a source-reported unique origin",
    "candidate_end_1based": "Computed inclusive one-based candidate end; not a source-reported unique origin",
    "dependency_group_id": "Connected component of all candidate and declared parent alternatives before supervised selection",
    "dependency_policy": "Explicit all-source grouping contract; membership is not biological eligibility",
}
