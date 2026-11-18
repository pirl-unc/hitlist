"""Source grains for same-plex treatment/control reporter measurements."""

import pyarrow as pa

from .material_context import MATERIAL_CONTEXT_SCHEMA, MATERIAL_LINK_SCHEMA


def _schema(text="", integers="", numbers="", flags=""):
    return pa.schema(
        [(name, pa.string()) for name in text.split()]
        + [(name, pa.int64()) for name in integers.split()]
        + [(name, pa.float64()) for name in numbers.split()]
        + [(name, pa.bool_()) for name in flags.split()]
    )


SCHEMAS = {
    "assets": _schema(
        "source_asset_id source_file source_asset_url asset_sha256 license_status", "size_bytes"
    ),
    "source_sheets": _schema(
        "sheet_id source_asset_id sheet_name sheet_state sheet_dimension merged_ranges_json source_locator"
    ),
    "source_records": _schema(
        "source_record_id sheet_id source_cells_json row_disposition source_locator",
        "source_row_ordinal n_source_cells",
    ),
    "conditions": _schema(
        "condition_id context_id cell_line_name treatment_name treatment_dose_unit condition_text "
        "reported_mhc candidate_mhc_set cellular_mhc_genotype mhc_context_evidence source_locator",
        numbers="treatment_dose duration_hours",
    ),
    "plexes": _schema(
        "plex_id sheet_id cell_line_name control_condition_id treated_condition_id "
        "dose_qualification normalization_scope source_dataset source_reference_id source_locator"
    ),
    "samples": _schema(
        "sample_id plex_id condition_id context_id reporter_channel arm_role "
        "replicate_type specimen_id aliquot_id source_locator",
        "source_slot_ordinal",
    ),
    "file_links": _schema(
        "claim_id plex_id source_file source_file_type source_record_id source_table_label "
        "filemap_qualification source_locator"
    ),
    "forms": _schema(
        "form_id source_record_id plex_id peptide_sequence modified_sequence_raw "
        "modifications_json form_status protein_label_raw gene_symbol_raw reported_length_raw",
        "peptide_length",
    ),
    "measurements": _schema(
        "measurement_id form_id sample_id plex_id source_record_id quantity_kind "
        "quantitative_raw value_status measurement_unit source_column source_header_raw source_locator",
        numbers="quantitative_value",
    ),
    "reported_summaries": _schema(
        "summary_id form_id plex_id source_record_id statistic_name quantity_kind "
        "quantitative_raw value_status measurement_unit source_column source_locator",
        numbers="quantitative_value",
    ),
    "protocol_assertions": _schema(
        "claim_id context_id assertion_predicate assertion_value assertion_status source_scope source_locator"
    ),
    "material_contexts": MATERIAL_CONTEXT_SCHEMA,
    "material_links": MATERIAL_LINK_SCHEMA,
}

RELATIONS = {
    "assets": ("one_pinned_original_workbook", "relation", "source_asset_id"),
    "source_sheets": ("one_original_worksheet", "relation", "sheet_id"),
    "source_records": (
        "one_stored_worksheet_row_including_headers_and_exclusions",
        "relation",
        "source_record_id",
    ),
    "conditions": ("one_reviewed_cell_line_condition_not_specimen", "context", "condition_id"),
    "plexes": ("one_reported_TMT_comparison_not_independent_acquisition", "context", "plex_id"),
    "samples": ("one_plex_scoped_reporter_channel_not_physical_specimen", "context", "sample_id"),
    "file_links": (
        "one_reported_plex_file_membership_not_peptide_detection",
        "relation",
        "claim_id",
    ),
    "forms": ("one_original_peptidoform_row_per_plex", "context", "form_id"),
    "measurements": ("one_reported_form_channel_quantity_kind", "measurement", "measurement_id"),
    "reported_summaries": (
        "one_original_summary_cell_not_recomputed_target",
        "measurement",
        "summary_id",
    ),
    "protocol_assertions": ("one_source_scoped_protocol_claim", "relation", "claim_id"),
    "material_contexts": ("one_precision_qualified_context", "context", "context_id"),
    "material_links": ("one_precision_qualified_context_link", "relation", "claim_id"),
}

# Existing common catalog fields keep their common definitions. These definitions
# qualify the additional fields introduced by this source family.
FIELD_MEANINGS = {
    "summary_id": "Source-scoped identity of an original reported statistic cell",
    "treated_condition_id": "Reviewed treated-arm identity within a source comparison; not a physical specimen",
    "source_cells_json": "All stored source cells in a row, including formulas, literal cached values, attributes and original cell XML; never recalculated",
    "row_disposition": "Peptidoform, excluded aggregate, pooled AUC, source metadata, or explicit blank row; no absence label is inferred",
    "n_source_cells": "Number of explicitly stored source cells in this worksheet row",
    "source_row_ordinal": "One-based original Excel row number, not a replicate identity",
    "plex_id": "Source-scoped TMT comparison identity; its six channels can share multiple technical acquisitions",
    "dose_qualification": "Reviewed agreement or disagreement between channel headers and the original sheet title/file map",
    "normalization_scope": "Source-reported hipMHC correction scope; large shifts can use separate within-condition normalization",
    "reporter_channel": "Original TMT channel label; scoped by plex and not an independent run",
    "arm_role": "Control or treated channel within this one source comparison",
    "source_slot_ordinal": "One-based slot within a reported arm, with no cross-arm pairing implied",
    "replicate_type": "Source reports biological triplicates; physical specimen IDs and cross-plex reuse remain unresolved",
    "mhc_context_evidence": "Curated deposited line-associated MHC labels supply candidates; independent cellular typing and peptide restriction are not established by these workbooks",
    "source_table_label": "Original file-map table number, including obsolete numbering",
    "filemap_qualification": "Evidence for matching a source file-map block to a comparison; source typos remain preserved",
    "source_file_type": "Original deposited-file role, such as Raw, Search or Peak",
    "quantity_kind": "Raw reporter, hipMHC-normalized reporter, or mean-adjusted reporter; these scales are not interchangeable",
    "source_header_raw": "Literal header for this quantity column, including source labeling errors",
    "statistic_name": "Meaning of an original reported summary cell, kept separate from recomputed contrasts",
    "protein_label_raw": "Unresolved original protein annotation; not a uniquely established protein of origin",
    "reported_length_raw": "Original peptide-length cell before consistency checks",
    "form_status": "Explicit unmodified or source-reported phosphorylated form; unsupported syntax is not silently stripped",
    "modifications_json": "Explicit source-reported modified residue positions; chemical identity is qualified by the article search protocol",
}
