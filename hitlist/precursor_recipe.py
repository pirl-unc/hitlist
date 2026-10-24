"""Declared, non-training qualification contract for PXD009360 precursor rows."""

from __future__ import annotations

RECIPE_ID = "pxd009360_precursor_quantity_qualification_v1"
RECIPE_SCHEMA_VERSION = 1

CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": RECIPE_SCHEMA_VERSION,
    "source_dataset": "PXD009360",
    "source_relations": [
        "precursor_evidence_row",
        "precursor_peptide_row",
        "precursor_all_peptide_row",
        "precursor_protein_group_row",
    ],
    "grouping": {
        "parent_scope": ["source_dataset", "source_file", "source_row_index"],
        "parent_split": "source_dataset_and_source_file_only",
        "biological_split": "unresolved_without_source_backed_specimen_crosswalk",
        "mapp_split": "unresolved_without_source_backed_material_link",
    },
    "quantity_policy": {
        "value_fields": ["intensity_raw", "lfq_intensity_raw"],
        "interpretation": "reported_search_output_not_absolute_abundance",
        "unit_policy": "retain_native_representation;no_unit_conversion",
        "imputation": "none",
        "normalization": "none",
    },
    "missingness_policy": {
        "empty_cells": "retain_as_unreported",
        "nan_cells": "retain_as_source_missing_representation",
        "ambiguous_contributors": "retain_and_mark_ambiguous",
        "filtered_rows": "retain_source_row_and_status_flags",
    },
    "pairing_policy": {
        "condition": "unresolved",
        "specimen": "unresolved",
        "aliquot": "unresolved",
        "mapp": "unresolved",
        "evidence_required": "source_backed_crosswalk_or_explicit_assertion",
    },
    "training_status": "unavailable",
}


def describe(store):
    """Describe recipe readiness without implying that it emits training rows."""
    available = set(CONTRACT["source_relations"]) <= store.bindings.keys()
    return {
        "recipe_id": RECIPE_ID,
        "status": "materialized" if available else "not_in_generation",
        "training_status": CONTRACT["training_status"],
        "contract": CONTRACT,
    }
