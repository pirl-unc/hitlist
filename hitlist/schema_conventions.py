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

"""Canonical column conventions and normalization across Hitlist assay families.

Hitlist serves multiple biological assay modalities (MS presentation, binding
affinity, kinetic/thermal stability, T-cell immunogenicity, perturbation
contrasts, bulk proteomics, and line expression) plus raw evidence-store tables.
This module defines the canonical column names per assay family and provides a
non-destructive :func:`normalize_assay_frame` converter so consumers never have
to guess between historical synonyms (e.g. ``peptide`` vs ``peptide_sequence``,
``measurement_unit`` vs ``measurement_units``, ``pmid`` vs ``source_pmid``,
``comparator_condition_ids`` vs ``control_condition_id``).
"""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd

from .conditions import CONDITION_COLUMNS

#: Canonical column contracts by assay family.
ASSAY_SCHEMA_CONVENTIONS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "presentation_ms": (
            "peptide",
            "mhc_restriction",
            "mhc_allele_set",
            "mhc_class",
            "mhc_species",
            "restriction_evidence",
            "sample_label",
            "sample_group",
            "pmid",
            "assay_method",
            "condition",
            "perturbation",
            "condition_category",
            "is_control_arm",
            *CONDITION_COLUMNS,
        ),
        "binding_affinity": (
            "peptide",
            "mhc_restriction",
            "mhc_class",
            "mhc_species",
            "measurement_value",
            "measurement_unit",
            "measurement_inequality",
            "measurement_kind",
            "censoring_status",
            "assay_method",
            "pmid",
        ),
        "stability": (
            "peptide",
            "mhc_restriction",
            "mhc_class",
            "mhc_species",
            "measurement_value",
            "measurement_unit",
            "measurement_inequality",
            "measurement_kind",
            "censoring_status",
            "assay_method",
            "pmid",
        ),
        "tcell": (
            "peptide",
            "mhc_restriction",
            "mhc_class",
            "assay_family",
            "assay_method",
            "response_measured",
            "qualitative_measurement",
            "measurement_value",
            "measurement_unit",
            "measurement_inequality",
            "pmid",
        ),
        "perturbation_contrast": (
            "peptide",
            "mhc_restriction",
            "mhc_class",
            "pmid",
            "condition_id",
            "comparator_condition_ids",
            "condition_control_for",
            "contrast_role",
            "baseline_treatment",
            "condition_vehicle",
            "condition_dose",
            "condition_dose_unit",
            "condition_duration_hours",
            "condition_baseline_engineering",
        ),
        "bulk_proteomics": (
            "peptide",
            "gene_name",
            "sample_label",
            "cell_line_name",
            "measurement_value",
            "measurement_unit",
            "pmid",
        ),
        "line_expression": (
            "gene_id",
            "gene_name",
            "cell_line_name",
            "measurement_value",
            "measurement_unit",
            "pmid",
        ),
        "binding_context_audit": (
            "peptide",
            "mhc_restriction",
            "mhc_class",
            "mhc_species",
            "endpoint_family",
            "measurement_value",
            "measurement_unit",
            "measurement_inequality",
            "censoring_status",
            "assay_method",
            "pmid",
            "source_review_status",
            "protocol_id",
            "context_scope",
            "method_review_status",
            "censor_review_status",
            "limit_applicability_status",
            "replicate_group_id",
            "replicate_group_n_observations",
            "kd_conversion_status",
        ),
        "binding_replicate_group": (
            "replicate_group_id",
            "source_dataset",
            "study_key",
            "pmid",
            "peptide",
            "mhc_restriction",
            "assay_method",
            "endpoint_family",
            "measurement_unit",
            "n_observations",
            "n_exact_or_blank",
            "n_right_censored",
            "n_left_censored",
            "has_conflicting_qualitative",
            "has_conflicting_inequality",
            "aggregation_policy",
        ),
    }
)

#: Explicit synonym -> canonical column name mapping across Hitlist modules.
#: Applied only when the canonical target column is not already present (unless
#: ``keep_aliases=False`` replaces an unambiguous synonym).
COLUMN_SYNONYM_MAP: MappingProxyType[str, str] = MappingProxyType(
    {
        # Peptide sequence
        "peptide_sequence": "peptide",
        # MHC allele / restriction / candidate bag
        "mhc_allele": "mhc_restriction",
        "allele": "mhc_restriction",
        "candidate_alleles": "mhc_allele_set",
        "candidate_mhc_set": "mhc_allele_set",
        "cellular_mhc_genotype": "mhc_genotype",
        "reported_mhc_class": "mhc_class",
        # Quantitative measurements, units, inequalities, qualitative labels
        "quantitative_value": "measurement_value",
        "standardized_value": "measurement_value",
        "measurement_units": "measurement_unit",
        "standardized_unit": "measurement_unit",
        "unit": "measurement_unit",
        "raw_measurement_units": "raw_measurement_unit",
        "normalized_inequality": "measurement_inequality",
        "qualitative_result": "qualitative_measurement",
        "source_qualitative_result": "qualitative_measurement",
        # Condition / contrast / exposure parameters
        "condition_text": "condition",
        "comparator_condition_id": "comparator_condition_ids",
        "control_condition_id": "comparator_condition_ids",
        "partner_condition_id": "comparator_condition_ids",
        "is_comparator_for": "condition_control_for",
        "treatment_dose": "condition_dose",
        "treatment_dose_unit": "condition_dose_unit",
        "duration_hours": "condition_duration_hours",
        # Publication / source identity
        "source_pmid": "pmid",
        "study_pmid": "pmid",
        "reference_pmid": "pmid",
        # Cell / sample identity
        "sample_label_raw": "sample_label",
        "line_key": "cell_line_name",
    }
)


def canonical_assay_columns(assay_type: str) -> tuple[str, ...]:
    """Return the canonical column tuple for *assay_type*."""
    if assay_type not in ASSAY_SCHEMA_CONVENTIONS:
        known = sorted(ASSAY_SCHEMA_CONVENTIONS)
        raise ValueError(f"Unknown assay_type {assay_type!r}; choose from {known}")
    return ASSAY_SCHEMA_CONVENTIONS[assay_type]


def normalize_assay_frame(
    frame: pd.DataFrame,
    assay_type: str | None = None,
    *,
    keep_aliases: bool = False,
) -> pd.DataFrame:
    """Return a copy of *frame* with variant column synonyms mapped to canonical names.

    Parameters
    ----------
    frame
        Any Hitlist export, training-selection, contrast, or evidence-store DataFrame.
    assay_type
        Optional key from :data:`ASSAY_SCHEMA_CONVENTIONS`. When given, validates
        the assay family name and orders matching canonical columns first.
    keep_aliases
        When ``False`` (default), renames synonym columns to their canonical name
        when the canonical column is absent (never overwriting an existing
        canonical column). When ``True``, adds the canonical column alongside the
        original alias column.
    """
    if assay_type is not None:
        canonical_order = canonical_assay_columns(assay_type)
    else:
        canonical_order = ()

    out = frame.copy()
    existing = set(out.columns)
    renames: dict[str, str] = {}
    additions: dict[str, pd.Series] = {}

    for col in out.columns:
        target = COLUMN_SYNONYM_MAP.get(col)
        if target is None or target == col:
            continue
        if target in existing or target in renames.values() or target in additions:
            continue
        if keep_aliases:
            additions[target] = out[col]
        else:
            renames[col] = target

    if renames:
        out = out.rename(columns=renames)
    for target, series in additions.items():
        out[target] = series

    if canonical_order:
        present_canonical = [c for c in canonical_order if c in out.columns]
        rest = [c for c in out.columns if c not in set(present_canonical)]
        out = out[present_canonical + rest]
    return out
