"""Source grains for nominal MS2 calibration, with explicit uncertain lineage."""

import pyarrow as pa

from .material_context import MATERIAL_CONTEXT_SCHEMA, MATERIAL_LINK_SCHEMA


def _schema(text="", integers="", numbers="", flags=""):
    return pa.schema(
        [(key, pa.string()) for key in text.split()]
        + [(key, pa.int64()) for key in integers.split()]
        + [(key, pa.float64()) for key in numbers.split()]
        + [(key, pa.bool_()) for key in flags.split()]
    )


SCHEMAS = {
    "assets": _schema(
        "source_asset_id source_file source_asset_url source_asset_role asset_sha256 license_status",
        "size_bytes",
    ),
    "source_sheets": _schema(
        "sheet_id source_asset_id sheet_name sheet_state sheet_dimension merged_ranges_json source_locator"
    ),
    "source_cells": _schema(
        "cell_id sheet_id cell_ref excel_type value_literal raw_cached_value formula "
        "cell_attributes_json formula_attributes_json xml_element source_locator",
        flags="value_present has_formula",
    ),
    "standards": _schema(
        "standard_id peptide_sequence gene_symbol_raw annotated_gene_raw predicted_affinity_raw "
        "prediction_status reported_length_raw reported_length_status length_consistency_status "
        "source_asset_id source_locator",
        "peptide_length",
        "predicted_affinity_nm reported_length_value",
    ),
    "isotope_assertions": _schema(
        "claim_id residue_symbol isotope_atoms_raw mass_offset_raw source_asset_id source_locator",
        numbers="mass_offset_da",
    ),
    "reported_forms": _schema(
        "form_id standard_id peptide_sequence isotopologue_label reported_form "
        "labeled_positions_json nominal_mass_shift_raw heavy_amino_acid_raw "
        "s6a_consistency_status s6a_sum_raw canonical_isotope_status source_asset_id source_locator",
        numbers="nominal_mass_shift_da",
    ),
    "protocol_assertions": _schema(
        "claim_id context_id assertion_predicate assertion_value assertion_status source_scope "
        "measurement_unit source_asset_id source_locator evidence_text"
    ),
    "source_slots": _schema(
        "slot_id context_id sheet_id source_column acquisition_id specimen_id aliquot_id "
        "material_link_precision source_locator",
        "source_slot_ordinal",
    ),
    "nominal_amounts": _schema(
        "nominal_amount_id standard_id context_id reported_form_label isotopologue_label "
        "nominal_amount_raw measurement_unit nominal_status source_asset_id source_locator",
        numbers="nominal_amount_fmol",
    ),
    "measurements": _schema(
        "measurement_id nominal_amount_id slot_id standard_id context_id quantitative_raw "
        "value_status measurement_unit observation_kind aggregation_status cell_id source_locator",
        numbers="quantitative_value",
    ),
    "transition_counts": _schema(
        "measurement_id standard_id nominal_amount_raw amount_unit_raw quantitative_raw "
        "value_status measurement_unit source_asset_id source_locator",
        numbers="nominal_amount_fmol quantitative_value",
    ),
    "export_forms": _schema(
        "export_form_id standard_id peptide_sequence modified_sequence_raw labeled_positions_json "
        "nominal_mass_shift_raw canonical_isotope_status source_asset_id source_locator"
    ),
    "form_correspondences": _schema(
        "claim_id export_form_id form_id correspondence_status correspondence_basis",
        "n_form_candidates",
    ),
    "transition_groups": _schema(
        "transition_group_id export_form_id standard_id context_id source_run_label "
        "nominal_amount_raw amount_unit_raw acquisition_id material_link_precision "
        "source_asset_id source_locator",
        "replicate_suffix_ordinal",
        "nominal_amount_fmol",
    ),
    "transitions": _schema(
        "transition_id transition_group_id precursor_mz_raw precursor_charge_raw product_mz_raw "
        "product_charge_raw fragment_ion_raw retention_time_raw quantitative_raw "
        "value_status peak_rank_raw concat_raw concat_status signal_kind measurement_unit "
        "source_asset_id source_locator",
        "n_duplicate_identity_rows",
        "quantitative_value",
    ),
    "aggregate_signals": _schema(
        "aggregate_id transition_group_id aggregation_mode quantitative_raw value_status "
        "measurement_unit source_transition_ids_json",
        "n_transition_rows n_missing_areas n_invalid_areas n_zero_areas",
        "quantitative_value",
    ),
    "measurement_candidates": _schema(
        "claim_id measurement_id transition_group_id correspondence_status correspondence_basis",
        "n_group_candidates",
        flags="ordinal_suffix_equal acquisition_join_established",
    ),
    "material_contexts": MATERIAL_CONTEXT_SCHEMA,
    "material_links": MATERIAL_LINK_SCHEMA,
}

RELATIONS = {
    "assets": ("one_pinned_source_asset", "relation", "source_asset_id"),
    "source_sheets": ("one_original_worksheet", "relation", "sheet_id"),
    "source_cells": ("one_stored_original_cell", "relation", "cell_id"),
    "standards": ("one_selected_panel_peptide_not_binding_measurement", "context", "standard_id"),
    "isotope_assertions": ("one_reported_residue_isotope_claim", "relation", "claim_id"),
    "reported_forms": ("one_original_S6B_form_claim", "context", "form_id"),
    "protocol_assertions": ("one_source_scoped_protocol_claim", "relation", "claim_id"),
    "source_slots": ("one_worksheet_column_not_acquisition", "context", "slot_id"),
    "nominal_amounts": ("one_reported_peptide_dose_row", "context", "nominal_amount_id"),
    "measurements": (
        "one_original_summed_signal_slot_including_blank",
        "measurement",
        "measurement_id",
    ),
    "transition_counts": (
        "one_reported_selected_transition_count",
        "measurement",
        "measurement_id",
    ),
    "export_forms": ("one_export_reported_form_per_asset", "context", "export_form_id"),
    "form_correspondences": ("one_source_form_candidate_or_missing_claim", "relation", "claim_id"),
    "transition_groups": (
        "one_reported_form_and_replicate_group",
        "context",
        "transition_group_id",
    ),
    "transitions": ("one_original_export_transition_row", "measurement", "transition_id"),
    "aggregate_signals": ("one_fixed_group_sum_not_published_signal", "projection", "aggregate_id"),
    "measurement_candidates": ("one_form_dose_candidate_or_missing_claim", "relation", "claim_id"),
    "material_contexts": ("one_precision_qualified_context", "context", "context_id"),
    "material_links": ("one_precision_qualified_context_link", "relation", "claim_id"),
}

FIELD_MEANINGS = {
    "aggregate_id": "Identity of a declared transition-group aggregation mode",
    "aggregation_mode": "Product-only or precursor-inclusive sum; neither silently substitutes for published signal",
    "amount_unit_raw": "Literal source amount unit before explicit conversion to fmol",
    "annotated_gene_raw": "Original panel annotated gene label, not a resolved gene identity",
    "cell_ref": "Original worksheet cell coordinate",
    "concat_raw": "Original fragment-ion and charge concatenation text, possibly absent",
    "correspondence_basis": "Evidence used to generate candidates, separate from physical identity",
    "correspondence_status": "Unique, ambiguous or missing correspondence at its explicitly declared scope",
    "evidence_text": "Optional source text supporting a protocol assertion; no assertion is inferred from its absence",
    "excel_type": "Original OOXML cell type, not a biological measurement datatype",
    "export_form_id": "Reported modified sequence identity scoped to its deposited export",
    "formula": "Original worksheet formula retained without evaluation",
    "fragment_ion_raw": "Original product-ion label or explicit precursor marker",
    "gene_symbol_raw": "Original panel gene label without external reconciliation",
    "has_formula": "Whether the original source cell has a formula element",
    "heavy_amino_acid_raw": "Original S6B heavy-residue assertion, including claims absent from S6A",
    "isotopologue_label": "Source-reported 1H/2H/3H/4H name, not inferred from label count",
    "mass_offset_da": "Parsed source-reported residue mass offset in daltons",
    "mass_offset_raw": "Original residue mass-offset literal",
    "merged_ranges_json": "Original merged worksheet ranges, not inferred acquisition groups",
    "n_invalid_areas": "Included transition areas unusable because of invalid values, types or formulas",
    "n_missing_areas": "Included transitions without a reported area",
    "n_transition_rows": "Number of original transition rows included by the declared aggregation mode",
    "n_zero_areas": "Included transition areas explicitly reported as zero",
    "nominal_mass_shift_da": "Parsed source-reported form mass shift, not a repaired atom-level calculation",
    "nominal_mass_shift_raw": "Original reported form mass-shift literal",
    "peak_rank_raw": "Original transition peak rank, not a calibrated confidence",
    "peptide_length": "Calculated residue count of the reported panel sequence",
    "reported_length_raw": "Original panel length annotation; may conflict with the sequence",
    "reported_length_value": "Parsed panel length annotation; not the calculated residue count",
    "reported_length_status": "Numeric, missing, formula or invalid status of the length assertion",
    "length_consistency_status": "Agreement with calculated sequence length, conflict or unusable claim",
    "precursor_charge_raw": "Original precursor charge literal",
    "precursor_mz_raw": "Original precursor mass-to-charge literal",
    "prediction_status": "Source annotation status explicitly distinguishing prediction from measured binding",
    "reported_form": "Original source form notation with its declared labeling positions",
    "reported_form_label": "Original S3 block label, which may be bare or modified sequence",
    "residue_symbol": "Original residue name in the source isotope assertion table",
    "retention_time_raw": "Original retention-time literal; units are not inferred",
    "sheet_id": "Stable original asset and worksheet identity",
    "sheet_state": "Original visible, hidden or veryHidden worksheet state",
    "source_column": "Original worksheet column identifier, not a run label",
    "source_slot_ordinal": "Ordinal position of a source response column, not an acquisition identity",
    "value_present": "Whether the cell has a stored value element, including an empty element",
    "xml_element": "Reserialized original cell XML; original asset retains lexical authority",
    "value_literal": "Original decoded cell value as text, without numeric rounding",
    "raw_cached_value": "Original XML cached-value literal, distinct from displayed string",
    "cell_attributes_json": "Original cell attributes including style references",
    "formula_attributes_json": "Original formula attributes, retained without evaluation",
    "sheet_dimension": "Original declared worksheet range; not an observation count",
    "predicted_affinity_raw": "Original predicted affinity annotation, not an experimental label",
    "predicted_affinity_nm": "Parsed predicted affinity in nM, not a measured binding target",
    "isotope_atoms_raw": "Source-reported isotope notation, including conflicting assertions",
    "labeled_positions_json": "Explicitly labeled one-based residue positions in source notation",
    "s6a_consistency_status": "Arithmetic correspondence with S6A, not isotope identity validation",
    "s6a_sum_raw": "Sum of S6A mass assertions at the explicitly labeled positions",
    "canonical_isotope_status": "Whether atom-level isotope identity is established; not repaired here",
    "nominal_amount_raw": "Original nominal standard amount literal, not endogenous concentration",
    "nominal_amount_fmol": "Nominal standard amount converted to fmol using explicit source units",
    "nominal_amount_id": "Source dose-row identity, separate from a specimen or acquisition",
    "aggregation_status": "Qualification of a source-reported aggregate, independent of derived sums",
    "transition_group_id": "Export form and replicate-label group, not a physical sample identity",
    "replicate_suffix_ordinal": "Explicit numerical suffix in a source label, not a worksheet join",
    "concat_status": "Missing, consistent, or conflicting reported fragment-charge concatenation",
    "signal_kind": "Precursor versus product-ion measurement as explicitly reported",
    "n_duplicate_identity_rows": "Rows sharing the same source transition identity; none deduplicated",
    "source_transition_ids_json": "Exact included source transition IDs for a declared sum",
    "n_form_candidates": "Number of source forms matching sequence, label positions and nominal shift",
    "n_group_candidates": "Number of groups matching source form and nominal amount, without run linkage",
    "ordinal_suffix_equal": "Diagnostic equality of source-column and run-label ordinals, not a join",
    "acquisition_join_established": "Whether evidence establishes the physical acquisition correspondence",
}
