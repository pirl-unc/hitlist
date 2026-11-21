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

"""Task-specific model-training preparation for Presto and MHCflurry.

Bridges Hitlist's source-preserving exports and verified ``training_bundle``
directories into model-ready affinity, stability, and MS presentation tables
without consumer-side endpoint regexes, ad-hoc unit math, silent inequality
drops, arbitrary multi-mapping flank selection, or conflation of study-wide
allele pools with single-sample genotypes.

Every ``prepare_*`` function returns a :class:`TrainingTableSelection` tuple
``(accepted, excluded, summary)`` that conserves 100% of candidate observations
across ``accepted`` and ``excluded`` with machine-readable exclusion reasons
and retained provenance, contributor keys, and lineage-applicability metadata.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from mhcgnomes import Allele, Pair

from .conditions import CONDITION_COLUMNS
from .curation import (
    _cached_parse,
    class_i_prediction_scope,
    classify_allele_resolution,
    mhc_class_of,
    mhc_species_of,
    normalize_mhc_class_token,
    normalize_species,
    resolve_allele_identity,
)
from .lineage import (
    APPLICABILITY_COLUMNS,
    LINEAGE_COLUMNS,
    attach_lineage,
    attach_lineage_applicability,
)
from .mhc_genotypes import infer_host_species, recover_mhc_genotypes_for_frame

CANONICAL_AMINO_ACIDS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

MOLAR_AFFINITY_ENDPOINTS: tuple[str, ...] = (
    "ic50",
    "kd_approx_ic50",
    "kd_approx_ec50",
    "kd",
    "ec50",
)

STABILITY_ENDPOINTS: tuple[str, ...] = (
    "half_life",
    "thermal_stability",
)

BINDING_ENDPOINT_FAMILIES: tuple[str, ...] = (
    *MOLAR_AFFINITY_ENDPOINTS,
    "ka",
    *STABILITY_ENDPOINTS,
    "on_rate",
    "off_rate",
    "qualitative_binding",
    "mhc_binding_unspecified",
    "other_binding",
    "not_binding",
)

_ENDPOINT_MAP: dict[str, str] = {
    "half maximal inhibitory concentration (ic50)": "ic50",
    "dissociation constant kd (~ic50)": "kd_approx_ic50",
    "dissociation constant kd (~ec50)": "kd_approx_ec50",
    "dissociation constant kd": "kd",
    "half maximal effective concentration (ec50)": "ec50",
    "association constant ka": "ka",
    "half life": "half_life",
    "half-life": "half_life",
    "50% dissociation temperature": "thermal_stability",
    "melting temperature": "thermal_stability",
    "on rate": "on_rate",
    "association rate": "on_rate",
    "off rate": "off_rate",
    "dissociation rate": "off_rate",
    "qualitative binding": "qualitative_binding",
    "mhc binding": "mhc_binding_unspecified",
}

# Case-sensitive molar concentration multipliers to nM. Deliberately excludes
# lowercase "m" / "mm" so time/length abbreviations never collide with Molar.
_MOLAR_FACTORS_TO_NM: dict[str, float] = {
    "nM": 1.0,
    "nm": 1.0,
    "pM": 1e-3,
    "pm": 1e-3,
    "fM": 1e-6,
    "fm": 1e-6,
    "uM": 1e3,
    "um": 1e3,
    "µM": 1e3,
    "μM": 1e3,
    "mM": 1e6,
    "M": 1e9,
}

_TIME_FACTORS_TO_MIN: dict[str, float] = {
    "min": 1.0,
    "mins": 1.0,
    "minute": 1.0,
    "minutes": 1.0,
    "h": 60.0,
    "hr": 60.0,
    "hrs": 60.0,
    "hour": 60.0,
    "hours": 60.0,
    "s": 1.0 / 60.0,
    "sec": 1.0 / 60.0,
    "secs": 1.0 / 60.0,
    "second": 1.0 / 60.0,
    "seconds": 1.0 / 60.0,
}

_TEMP_FACTORS_TO_CELSIUS: dict[str, float] = {
    "°C": 1.0,
    "C": 1.0,
    "celsius": 1.0,
    "Celsius": 1.0,
    "deg C": 1.0,
}

_STUDY_POOL_PROVENANCES: frozenset[str] = frozenset(
    {"pmid_class_pool", "pmid_locus_pool", "pmid_serotype_pool"}
)

_SAMPLE_CANDIDATE_PROVENANCES: frozenset[str] = frozenset(
    {
        "peptide_attribution",
        "sample_allele_match",
        "peptide_locus_match",
        "sample_locus_match",
        "peptide_serotype_match",
        "sample_serotype_match",
        "haplotype_roster",
    }
)


class TrainingTableSelection(NamedTuple):
    """Auditable split of candidate observations into accepted and excluded rows."""

    accepted: pd.DataFrame
    excluded: pd.DataFrame
    summary: dict


def _is_canonical_peptide(seq: str, *, min_len: int, max_len: int) -> tuple[bool, str]:
    text = seq.strip().upper()
    if not text:
        return False, "empty_peptide"
    if len(text) < min_len or len(text) > max_len:
        return False, "peptide_length_out_of_bounds"
    if any(ch not in CANONICAL_AMINO_ACIDS for ch in text):
        return False, "non_canonical_amino_acid"
    return True, ""


def _is_mutant_mhc_object(parsed: object) -> bool:
    if parsed is None:
        return False
    if bool(getattr(parsed, "mutations", ())):
        return True
    alpha = getattr(parsed, "alpha", None)
    beta = getattr(parsed, "beta", None)
    return bool(getattr(alpha, "mutations", ())) or bool(getattr(beta, "mutations", ()))


def _allele_completeness_and_identity(
    mhc_restriction: str,
    *,
    scorable_class_i_only: bool = False,
) -> tuple[str, str, str, str]:
    """Return ``(allele, completeness, mhc_class, mhc_species)`` or ``("", reason, "", "")``."""
    raw = str(mhc_restriction or "").strip()
    if not raw or classify_allele_resolution(raw) != "four_digit":
        return "", "unresolved_allele_restriction", "", ""

    if scorable_class_i_only:
        scope = class_i_prediction_scope(raw)
        if not scope.is_eligible or len(scope.scorable) != 1:
            return "", "not_scorable_class_i_allele", "", ""
        allele = scope.scorable[0]
        return allele, "complete_class_i", "I", "Homo sapiens"

    identity = resolve_allele_identity(raw)
    if not identity:
        return "", "unresolved_allele_restriction", "", ""
    parsed = _cached_parse(identity)
    cls = normalize_mhc_class_token(mhc_class_of(identity))
    sp = mhc_species_of(identity)
    is_mutant = _is_mutant_mhc_object(parsed)
    if isinstance(parsed, Pair):
        comp = "mutant_class_ii_pair" if is_mutant else "complete_class_ii_pair"
        return identity, comp, cls or "II", sp
    if isinstance(parsed, Allele):
        if getattr(parsed, "is_class1", False):
            comp = "mutant_class_i" if is_mutant else "complete_class_i"
            return identity, comp, cls or "I", sp
        if getattr(parsed, "is_class2", False):
            gene = str(getattr(parsed, "gene_name", "") or "")
            if parsed.is_human and gene.startswith("DRB"):
                comp = (
                    "mutant_single_chain_class_ii_dr" if is_mutant else "single_chain_class_ii_dr"
                )
                return identity, comp, cls or "II", sp
            return identity, "single_chain_class_ii_unpaired", cls or "II", sp
    return identity, "mutant_other" if is_mutant else "complete_other", cls, sp


def _load_source_frame(
    source: pd.DataFrame | str | Path,
    *,
    default_evidence_kind: str,
) -> tuple[pd.DataFrame, dict | None]:
    """Load and normalize a verified ``training_bundle`` or DataFrame."""
    import pyarrow.parquet as pq

    from .export import _apply_training_defaults
    from .provenance import file_digest
    from .training_bundle import FILES, verify_training_bundle

    bundle_info: dict | None = None
    bundle_registry: dict | None = None
    if isinstance(source, (str, Path)):
        bundle_dir = Path(source)
        manifest = verify_training_bundle(bundle_dir)
        bundle_registry = json.loads((bundle_dir / FILES["lineage"]).read_text())
        training_path = bundle_dir / FILES["training"]
        identities_path = bundle_dir / FILES["identities"]
        kind_filter = [("evidence_kind", "==", default_evidence_kind)]
        training = (
            pd.read_parquet(training_path, filters=kind_filter)
            if pq.ParquetFile(training_path).metadata.num_rows > 0
            else pd.read_parquet(training_path)
        )
        identities = (
            pd.read_parquet(identities_path, filters=kind_filter)
            if pq.ParquetFile(identities_path).metadata.num_rows > 0
            else pd.read_parquet(identities_path)
        )
        missing_id_cols = [
            c for c in identities.columns if c not in training.columns and c != "evidence_row_id"
        ]
        if missing_id_cols:
            training = training.merge(
                identities[["evidence_row_id", *missing_id_cols]],
                on="evidence_row_id",
                how="left",
                validate="many_to_one",
            )
        df = training
        bundle_info = {
            "directory": str(bundle_dir),
            "manifest_sha256": file_digest(bundle_dir / "manifest.json")["sha256"],
            "hitlist_version": manifest.get("hitlist_version"),
            "split_policy": manifest.get("split_policy"),
            "input_validation": manifest.get("input_validation"),
        }
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError("source must be a pandas DataFrame or path to a verified training bundle")

    if "evidence_kind" not in df.columns:
        if "is_binding_assay" in df.columns and "is_ms_observation" in df.columns:
            is_bind = df["is_binding_assay"].astype("boolean").fillna(False)
            is_ms = df["is_ms_observation"].astype("boolean").fillna(False)
            df["evidence_kind"] = np.where(is_ms, "ms", np.where(is_bind, "binding", "other"))
        else:
            df["evidence_kind"] = default_evidence_kind

    if "evidence_row_id" not in df.columns:
        df = _apply_training_defaults(df)
    if not set(LINEAGE_COLUMNS) <= set(df.columns):
        df = attach_lineage(df, registry=bundle_registry, copy=False, include_applicability=True)
    elif not set(APPLICABILITY_COLUMNS) <= set(df.columns):
        df = attach_lineage_applicability(df, copy=False)
    return df, bundle_info


def qualify_binding_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate binding rows with canonical endpoint families, linear unit conversion, and censoring.

    All original columns and rows are preserved untouched. Linear unit conversion
    is restricted to physically commensurate units (molar concentration -> ``nM``,
    half-life -> ``min``, thermal dissociation -> ``celsius``). Nonlinear units
    such as ``log10(IC50)`` are never exponentiated, and ``IC50`` is never
    reinterpreted as ``KD``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    result = df.copy()
    if "assay_preparation" not in result.columns:
        result = attach_lineage_applicability(result, copy=False)

    n_rows = len(result)
    if n_rows == 0:
        result["binding_endpoint_family"] = pd.Series(dtype="string")
        result["standardized_value"] = pd.Series(dtype="float64")
        result["standardized_unit"] = pd.Series(dtype="string")
        result["unit_conversion_status"] = pd.Series(dtype="string")
        result["censoring_status"] = pd.Series(dtype="string")
        result["normalized_inequality"] = pd.Series(dtype="string")
        return result

    empty_str = pd.Series("", index=result.index, dtype="string")
    responses = result.get("response_measured", empty_str).astype("string").fillna("").str.strip()
    units = result.get("measurement_units", empty_str).astype("string").fillna("").str.strip()
    ineqs = result.get("measurement_inequality", empty_str).astype("string").fillna("").str.strip()
    if "evidence_kind" in result.columns:
        is_binding = result["evidence_kind"].astype("string").fillna("").eq("binding")
    elif "is_binding_assay" in result.columns:
        is_binding = result["is_binding_assay"].astype("boolean").fillna(False).astype(bool)
    else:
        is_binding = pd.Series(True, index=result.index, dtype=bool)

    if "quantitative_value" in result.columns:
        raw_vals = pd.to_numeric(result["quantitative_value"], errors="coerce").to_numpy(
            dtype="float64"
        )
    else:
        raw_vals = np.full(n_rows, np.nan, dtype="float64")
    has_finite = np.isfinite(raw_vals)

    unique_meta = pd.DataFrame(
        {
            "is_binding": is_binding,
            "response": responses,
            "unit": units,
            "ineq": ineqs,
            "has_finite": has_finite,
        }
    ).drop_duplicates()

    meta_tuples = list(unique_meta.itertuples(index=False, name=None))
    families: list[str] = []
    std_units: list[str] = []
    unit_statuses: list[str] = []
    factors: list[float] = []
    censor_statuses: list[str] = []
    norm_ineqs: list[str] = []

    for bind_flag, resp, unit, ineq, finite_flag in meta_tuples:
        if not bind_flag:
            families.append("not_binding")
            std_units.append("")
            unit_statuses.append("not_applicable")
            factors.append(np.nan)
            censor_statuses.append("not_applicable")
            norm_ineqs.append("")
            continue

        family = _ENDPOINT_MAP.get(resp.casefold(), "other_binding")
        families.append(family)

        if family in MOLAR_AFFINITY_ENDPOINTS:
            canonical_unit = "nM"
            factor_table = _MOLAR_FACTORS_TO_NM
        elif family == "half_life":
            canonical_unit = "min"
            factor_table = _TIME_FACTORS_TO_MIN
        elif family == "thermal_stability":
            canonical_unit = "celsius"
            factor_table = _TEMP_FACTORS_TO_CELSIUS
        else:
            canonical_unit = ""
            factor_table = None

        if factor_table is None:
            std_units.append("")
            unit_statuses.append("not_applicable")
            factors.append(np.nan)
        elif not finite_flag:
            std_units.append(canonical_unit)
            unit_statuses.append("no_numeric_value")
            factors.append(np.nan)
        elif not unit:
            std_units.append(canonical_unit)
            unit_statuses.append("missing_unit")
            factors.append(np.nan)
        elif unit in factor_table:
            factor = factor_table[unit]
            std_units.append(canonical_unit)
            unit_statuses.append("canonical_unit" if factor == 1.0 else "linear_converted")
            factors.append(factor)
        else:
            std_units.append(canonical_unit)
            unit_statuses.append("nonlinear_or_unsupported_unit")
            factors.append(np.nan)

        if not finite_flag:
            if ineq:
                censor_statuses.append("operator_without_value")
            else:
                censor_statuses.append("no_numeric_value")
            norm_ineqs.append("")
        elif ineq == "":
            censor_statuses.append("unreported_operator_numeric")
            norm_ineqs.append("=")
        elif ineq == "=":
            censor_statuses.append("exact")
            norm_ineqs.append("=")
        elif ineq == "<":
            censor_statuses.append("left_censored_lt")
            norm_ineqs.append("<")
        elif ineq == "<=":
            censor_statuses.append("left_censored_lte")
            norm_ineqs.append("<")
        elif ineq == ">":
            censor_statuses.append("right_censored_gt")
            norm_ineqs.append(">")
        elif ineq == ">=":
            censor_statuses.append("right_censored_gte")
            norm_ineqs.append(">")
        else:
            censor_statuses.append("unrecognized_inequality")
            norm_ineqs.append("")

    lookup = pd.MultiIndex.from_tuples(meta_tuples)
    row_idx = lookup.get_indexer(
        pd.MultiIndex.from_arrays([is_binding, responses, units, ineqs, has_finite])
    )

    factor_arr = np.array(factors, dtype="float64")[row_idx]
    with np.errstate(over="ignore", invalid="ignore"):
        converted = raw_vals * factor_arr
    overflow_mask = np.isfinite(raw_vals) & np.isfinite(factor_arr) & ~np.isfinite(converted)
    unit_status_arr = np.array(unit_statuses, dtype=object)[row_idx]
    if overflow_mask.any():
        converted[overflow_mask] = np.nan
        unit_status_arr[overflow_mask] = "nonlinear_or_unsupported_unit"

    result["binding_endpoint_family"] = pd.Series(
        np.array(families, dtype=object)[row_idx], index=result.index, dtype="string"
    )
    result["standardized_value"] = pd.Series(converted, index=result.index, dtype="float64")
    result["standardized_unit"] = pd.Series(
        np.array(std_units, dtype=object)[row_idx], index=result.index, dtype="string"
    )
    result["unit_conversion_status"] = pd.Series(
        unit_status_arr, index=result.index, dtype="string"
    )
    result["censoring_status"] = pd.Series(
        np.array(censor_statuses, dtype=object)[row_idx], index=result.index, dtype="string"
    )
    result["normalized_inequality"] = pd.Series(
        np.array(norm_ineqs, dtype=object)[row_idx], index=result.index, dtype="string"
    )
    if "method_review_status" not in result.columns and any(
        col in result.columns for col in ("reference_iri", "submission_id", "assay_comments")
    ):
        from .binding_assay_context import review_binding_assay_methods

        result = review_binding_assay_methods(result)
    return result


def _prepare_quantitative_binding_task(
    source: pd.DataFrame | str | Path,
    *,
    task_name: str,
    allowed_endpoints: tuple[str, ...],
    blank_inequality_policy: str,
    allow_censored: bool,
    mhc_class: str | None,
    species: str | None,
    scorable_class_i_only: bool,
    allow_unpaired_class_ii_chains: bool,
    allow_mutant_mhc: bool,
    allow_modified_peptides: bool,
    min_peptide_length: int,
    max_peptide_length: int,
    require_positive_values: bool,
) -> TrainingTableSelection:
    if blank_inequality_policy not in {"treat_as_exact", "exclude"}:
        raise ValueError("blank_inequality_policy must be 'treat_as_exact' or 'exclude'")
    if min_peptide_length < 1 or max_peptide_length < min_peptide_length:
        raise ValueError("Invalid peptide length bounds")
    if not allowed_endpoints:
        raise ValueError("endpoints must be nonempty")

    df, bundle_info = _load_source_frame(source, default_evidence_kind="binding")
    binding_df = df[df["evidence_kind"].astype("string").eq("binding")].reset_index(drop=True)
    qualified = qualify_binding_measurements(binding_df)

    target_class = normalize_mhc_class_token(mhc_class) if mhc_class else ""
    target_species = (
        normalize_species(species) if species and str(species).strip().casefold() != "all" else ""
    )
    allowed_endpoint_set = set(allowed_endpoints)

    from .binding_assay_context import METHOD_REVIEW_COLUMNS
    from .netmhcstabpan import NETMHCSTABPAN_PROVENANCE_COLUMNS

    optional_source_cols = [
        c
        for c in (*METHOD_REVIEW_COLUMNS, *NETMHCSTABPAN_PROVENANCE_COLUMNS)
        if c in qualified.columns
    ]
    accepted_columns = [
        "evidence_row_id",
        "evidence_source_id",
        "provenance_id",
        "provenance_status",
        "allele",
        "mhc_restriction",
        "reported_mhc_restriction",
        "allele_completeness",
        "mhc_class",
        "mhc_species",
        "peptide",
        "measurement_value",
        "measurement_unit",
        "measurement_inequality",
        "measurement_type",
        "measurement_kind",
        "measurement_source",
        "censoring_status",
        "unit_conversion_status",
        "raw_measurement_value",
        "raw_measurement_unit",
        "raw_measurement_units",
        "raw_measurement_inequality",
        "assay_method",
        "pmid",
        "assay_iri",
        "reference_iri",
        *LINEAGE_COLUMNS,
        *APPLICABILITY_COLUMNS,
        *optional_source_cols,
    ]
    excluded_columns = [
        "evidence_row_id",
        "evidence_source_id",
        "provenance_id",
        "peptide",
        "mhc_restriction",
        "mhc_class",
        "response_measured",
        "binding_endpoint_family",
        "measurement_value",
        "measurement_unit",
        "quantitative_value",
        "measurement_units",
        "measurement_inequality",
        "pmid",
        "exclusion_reason",
        *[c for c in optional_source_cols if c not in {"mhc_restriction", "mhc_class"}],
    ]

    if qualified.empty:
        summary = {
            "task": task_name,
            "source_bundle": bundle_info,
            "n_input_candidate_rows": 0,
            "n_accepted_rows": 0,
            "n_excluded_rows": 0,
            "n_unique_peptides": 0,
            "n_unique_alleles": 0,
            "n_unique_pmids": 0,
            "by_endpoint_family": {},
            "by_censoring_status": {},
            "by_assay_preparation": {},
            "by_exclusion_reason": {},
        }
        return TrainingTableSelection(
            accepted=pd.DataFrame(columns=accepted_columns),
            excluded=pd.DataFrame(columns=excluded_columns),
            summary=summary,
        )

    # Pre-resolve unique restriction strings once.
    restrictions = (
        qualified.get("mhc_restriction", pd.Series("", index=qualified.index, dtype="string"))
        .astype("string")
        .fillna("")
    )
    allele_cache = {
        raw: _allele_completeness_and_identity(raw, scorable_class_i_only=scorable_class_i_only)
        for raw in restrictions.unique()
    }

    peptides = (
        qualified.get("peptide", pd.Series("", index=qualified.index, dtype="string"))
        .astype("string")
        .fillna("")
    )
    pep_cache = {
        pep: _is_canonical_peptide(pep, min_len=min_peptide_length, max_len=max_peptide_length)
        for pep in peptides.unique()
    }

    has_ptm = (
        qualified["has_ptm"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "has_ptm" in qualified.columns
        else np.zeros(len(qualified), dtype=bool)
    )
    non_pep = (
        qualified["is_non_peptide_ligand"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "is_non_peptide_ligand" in qualified.columns
        else np.zeros(len(qualified), dtype=bool)
    )
    families = qualified["binding_endpoint_family"].to_numpy(dtype=object)
    unit_statuses = qualified["unit_conversion_status"].to_numpy(dtype=object)
    censor_statuses = qualified["censoring_status"].to_numpy(dtype=object)
    norm_ineqs = qualified["normalized_inequality"].to_numpy(dtype=object)
    std_vals = qualified["standardized_value"].to_numpy(dtype="float64")
    std_units = qualified["standardized_unit"].to_numpy(dtype=object)

    reasons: list[str] = []
    resolved_alleles: list[str] = []
    completeness_vals: list[str] = []
    resolved_classes: list[str] = []
    resolved_species: list[str] = []

    for idx, (raw_rest, pep) in enumerate(zip(restrictions, peptides)):
        fam = families[idx]
        if non_pep[idx]:
            reasons.append("non_peptide_ligand")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if not allow_modified_peptides and has_ptm[idx]:
            reasons.append("modified_peptide_excluded")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        pep_ok, pep_reason = pep_cache[pep]
        if not pep_ok:
            reasons.append(pep_reason)
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if fam not in allowed_endpoint_set:
            reasons.append(f"endpoint_not_selected:{fam}")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue

        ustatus = unit_statuses[idx]
        if ustatus not in {"canonical_unit", "linear_converted"}:
            reasons.append(ustatus)
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if (require_positive_values or fam == "half_life") and not (
            std_vals[idx] > 0.0 and math.isfinite(std_vals[idx])
        ):
            reasons.append("non_positive_measurement_value")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if not math.isfinite(std_vals[idx]):
            reasons.append("no_numeric_value")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue

        cstatus = censor_statuses[idx]
        if cstatus == "unreported_operator_numeric" and blank_inequality_policy == "exclude":
            reasons.append("unreported_inequality_excluded")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if cstatus in {"unrecognized_inequality", "operator_without_value", "no_numeric_value"}:
            reasons.append(cstatus)
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if not allow_censored and norm_ineqs[idx] != "=":
            reasons.append("censored_measurement_excluded")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue

        allele, comp, a_cls, a_sp = allele_cache[raw_rest]
        if not allele:
            reasons.append(comp)
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if comp.startswith("mutant_") and not allow_mutant_mhc:
            reasons.append("mutant_mhc_construct_excluded")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if comp == "single_chain_class_ii_unpaired" and not allow_unpaired_class_ii_chains:
            reasons.append("unpaired_class_ii_chain")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if target_class and a_cls != target_class:
            reasons.append("mhc_class_mismatch")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue
        if target_species and a_sp != target_species:
            reasons.append("mhc_species_mismatch")
            resolved_alleles.append("")
            completeness_vals.append("")
            resolved_classes.append("")
            resolved_species.append("")
            continue

        reasons.append("")
        resolved_alleles.append(allele)
        completeness_vals.append(comp)
        resolved_classes.append(a_cls)
        resolved_species.append(a_sp)

    reason_series = pd.Series(reasons, index=qualified.index, dtype="string")
    ok_mask = reason_series.eq("")

    acc_src = qualified.loc[ok_mask].copy()
    ok_positions = np.flatnonzero(ok_mask.to_numpy(dtype=bool))
    acc_src["allele"] = pd.Series(
        [resolved_alleles[i] for i in ok_positions], index=acc_src.index, dtype="string"
    )
    acc_src["mhc_restriction"] = acc_src["allele"]
    acc_src["reported_mhc_restriction"] = restrictions.loc[ok_mask]
    acc_src["allele_completeness"] = pd.Series(
        [completeness_vals[i] for i in ok_positions], index=acc_src.index, dtype="string"
    )
    acc_src["mhc_class"] = pd.Series(
        [resolved_classes[i] for i in ok_positions], index=acc_src.index, dtype="string"
    )
    acc_src["mhc_species"] = pd.Series(
        [resolved_species[i] for i in ok_positions], index=acc_src.index, dtype="string"
    )
    acc_src["peptide"] = peptides.loc[ok_mask].str.strip().str.upper()
    acc_src["measurement_value"] = pd.Series(
        std_vals[ok_positions], index=acc_src.index, dtype="float64"
    )
    acc_src["measurement_unit"] = pd.Series(
        std_units[ok_positions], index=acc_src.index, dtype="string"
    )
    acc_src["measurement_inequality"] = pd.Series(
        norm_ineqs[ok_positions], index=acc_src.index, dtype="string"
    )
    acc_src["measurement_type"] = pd.Series("quantitative", index=acc_src.index, dtype="string")
    acc_src["measurement_kind"] = acc_src["binding_endpoint_family"]
    pmids_str = (
        pd.to_numeric(acc_src.get("pmid"), errors="coerce")
        .astype("Int64")
        .astype("string")
        .fillna("")
        if "pmid" in acc_src.columns
        else pd.Series("", index=acc_src.index, dtype="string")
    )
    refs_str = (
        acc_src["reference_iri"].astype("string").fillna("")
        if "reference_iri" in acc_src.columns
        else pd.Series("", index=acc_src.index, dtype="string")
    )
    acc_src["measurement_source"] = np.where(pmids_str.ne(""), "pmid:" + pmids_str, refs_str)
    acc_src["raw_measurement_value"] = (
        pd.to_numeric(acc_src["quantitative_value"], errors="coerce")
        if "quantitative_value" in acc_src.columns
        else np.nan
    )
    raw_units_series = (
        acc_src["measurement_units"].astype("string").fillna("")
        if "measurement_units" in acc_src.columns
        else pd.Series("", index=acc_src.index, dtype="string")
    )
    acc_src["raw_measurement_unit"] = raw_units_series
    acc_src["raw_measurement_units"] = raw_units_series
    acc_src["raw_measurement_inequality"] = (
        qualified.loc[ok_mask, "measurement_inequality"].astype("string").fillna("")
        if "measurement_inequality" in qualified.columns
        else ""
    )
    for col in accepted_columns:
        if col not in acc_src.columns:
            acc_src[col] = ""
    accepted = acc_src[accepted_columns].reset_index(drop=True)

    excl_src = qualified.loc[~ok_mask].copy()
    excl_src["exclusion_reason"] = reason_series.loc[~ok_mask]
    excl_src["measurement_value"] = (
        pd.to_numeric(excl_src["quantitative_value"], errors="coerce")
        if "quantitative_value" in excl_src.columns
        else np.nan
    )
    excl_src["measurement_unit"] = (
        excl_src["measurement_units"].astype("string").fillna("")
        if "measurement_units" in excl_src.columns
        else ""
    )
    for col in excluded_columns:
        if col not in excl_src.columns:
            excl_src[col] = np.nan if col in {"quantitative_value", "measurement_value"} else ""
    excluded = excl_src[excluded_columns].reset_index(drop=True)

    pmid_count = (
        int(pd.to_numeric(accepted["pmid"], errors="coerce").dropna().nunique())
        if len(accepted)
        else 0
    )
    summary = {
        "task": task_name,
        "source_bundle": bundle_info,
        "n_input_candidate_rows": len(qualified),
        "n_accepted_rows": len(accepted),
        "n_excluded_rows": len(excluded),
        "n_unique_peptides": int(accepted["peptide"].nunique()),
        "n_unique_alleles": int(accepted["allele"].nunique()),
        "n_unique_pmids": pmid_count,
        "by_endpoint_family": dict(sorted(Counter(accepted["measurement_kind"]).items())),
        "by_censoring_status": dict(sorted(Counter(accepted["censoring_status"]).items())),
        "by_assay_preparation": dict(
            sorted(Counter(accepted["assay_preparation"].astype("string")).items())
        ),
        "by_exclusion_reason": dict(sorted(Counter(excluded["exclusion_reason"]).items())),
    }
    return TrainingTableSelection(accepted=accepted, excluded=excluded, summary=summary)


def prepare_affinity_training_data(
    source: pd.DataFrame | str | Path,
    *,
    endpoints: tuple[str, ...] = MOLAR_AFFINITY_ENDPOINTS,
    blank_inequality_policy: str = "treat_as_exact",
    allow_censored: bool = True,
    mhc_class: str | None = None,
    species: str | None = "Homo sapiens",
    scorable_class_i_only: bool = False,
    allow_unpaired_class_ii_chains: bool = False,
    allow_mutant_mhc: bool = False,
    allow_modified_peptides: bool = False,
    min_peptide_length: int = 5,
    max_peptide_length: int = 45,
) -> TrainingTableSelection:
    """Prepare molar binding-affinity rows (in ``nM``) for MHCflurry or Presto.

    Parameters
    ----------
    source
        Verified ``training_bundle`` directory path or a pandas DataFrame from
        :func:`hitlist.export.generate_training_table` /
        :func:`hitlist.export.generate_binding_table`.
    endpoints
        Allowed molar endpoint families. Pass ``("ic50",)`` for strict literal
        IC50 training (Presto #110/#140) or keep ``MOLAR_AFFINITY_ENDPOINTS``
        when training pan-allele affinity ensembles.
    blank_inequality_policy
        ``"treat_as_exact"`` (default) maps IEDB numeric rows whose inequality
        operator was left blank to ``measurement_inequality == "="`` while
        recording ``censoring_status == "unreported_operator_numeric"``;
        ``"exclude"`` routes those rows to ``excluded``.
    scorable_class_i_only
        When ``True``, restricts to two-field human classical class-Ia proteins
        accepted by MHCflurry ``Class1AffinityPredictor`` (recommend pairing
        with ``min_peptide_length=5, max_peptide_length=15``).
    allow_unpaired_class_ii_chains
        When ``False`` (default), single-chain DP/DQ rows lacking their
        polymorphic partner chain are routed to ``excluded`` rather than
        inventing an alpha chain; human ``HLA-DRB*`` remains accepted as
        ``single_chain_class_ii_dr`` because ``HLA-DRA``'s binding groove is
        monomorphic.
    allow_mutant_mhc
        When ``False`` (default), engineered point-mutant MHC constructs (e.g.
        ``HLA-B*14:01 C67S mutant``) are routed to ``excluded`` with
        ``mutant_mhc_construct_excluded`` so wild-type sequence lookups never
        consume mutant measurements; set ``True`` only when applying explicit
        mutant sequence substitutions downstream.
    """
    invalid = set(endpoints) - set(MOLAR_AFFINITY_ENDPOINTS)
    if invalid:
        raise ValueError(
            f"Unsupported molar affinity endpoints {sorted(invalid)}; "
            f"choose from {MOLAR_AFFINITY_ENDPOINTS}"
        )
    return _prepare_quantitative_binding_task(
        source,
        task_name="mhc_binding_affinity_nm",
        allowed_endpoints=tuple(endpoints),
        blank_inequality_policy=blank_inequality_policy,
        allow_censored=allow_censored,
        mhc_class=mhc_class,
        species=species,
        scorable_class_i_only=scorable_class_i_only,
        allow_unpaired_class_ii_chains=allow_unpaired_class_ii_chains,
        allow_mutant_mhc=allow_mutant_mhc,
        allow_modified_peptides=allow_modified_peptides,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        require_positive_values=True,
    )


def prepare_stability_training_data(
    source: pd.DataFrame | str | Path,
    *,
    endpoints: tuple[str, ...] = STABILITY_ENDPOINTS,
    blank_inequality_policy: str = "treat_as_exact",
    allow_censored: bool = True,
    mhc_class: str | None = None,
    species: str | None = "Homo sapiens",
    scorable_class_i_only: bool = False,
    allow_unpaired_class_ii_chains: bool = False,
    allow_mutant_mhc: bool = False,
    allow_modified_peptides: bool = False,
    min_peptide_length: int = 5,
    max_peptide_length: int = 45,
) -> TrainingTableSelection:
    """Prepare pMHC stability rows (half-life in ``min``, Tm in ``celsius``) for Presto.

    Literal ``0.0`` half-life rows (SPA nondetections / subthreshold dissociation)
    are routed to ``excluded`` with ``non_positive_measurement_value`` rather than
    accepted as exact zero half-lives (#875). Engineered mutant MHC constructs are
    routed to ``excluded`` with ``mutant_mhc_construct_excluded`` unless
    ``allow_mutant_mhc=True``.
    """
    invalid = set(endpoints) - set(STABILITY_ENDPOINTS)
    if invalid:
        raise ValueError(
            f"Unsupported stability endpoints {sorted(invalid)}; choose from {STABILITY_ENDPOINTS}"
        )
    return _prepare_quantitative_binding_task(
        source,
        task_name="mhc_stability",
        allowed_endpoints=tuple(endpoints),
        blank_inequality_policy=blank_inequality_policy,
        allow_censored=allow_censored,
        mhc_class=mhc_class,
        species=species,
        scorable_class_i_only=scorable_class_i_only,
        allow_unpaired_class_ii_chains=allow_unpaired_class_ii_chains,
        allow_mutant_mhc=allow_mutant_mhc,
        allow_modified_peptides=allow_modified_peptides,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        require_positive_values=False,
    )


def _collapse_ms_mappings_to_observations(
    ms_df: pd.DataFrame, *, use_expression_disambiguation: bool
) -> pd.DataFrame:
    """Collapse protein-mapping expansions to 1 row per ``evidence_row_id`` (#34)."""
    mapping_cols_present = {"protein_id", "n_flank", "c_flank"} <= set(ms_df.columns)
    if ms_df.empty:
        out = ms_df.copy()
        for col, dtype in (
            ("n_protein_mappings", "int64"),
            ("n_gene_mappings", "int64"),
            ("n_distinct_flank_pairs", "int64"),
            ("flank_status", "string"),
            ("n_flank", "string"),
            ("c_flank", "string"),
            ("protein_id", "string"),
            ("gene_id", "string"),
            ("gene_name", "string"),
            ("transcript_id", "string"),
            ("candidate_protein_ids", "string"),
            ("candidate_gene_ids", "string"),
            ("candidate_flank_pairs", "string"),
        ):
            out[col] = pd.Series(dtype=dtype)
        return out

    if not mapping_cols_present:
        dedup = ms_df.drop_duplicates("evidence_row_id", keep="first").reset_index(drop=True)
        dedup["n_protein_mappings"] = 0
        dedup["n_gene_mappings"] = 0
        dedup["n_distinct_flank_pairs"] = 0
        dedup["flank_status"] = pd.Series("not_mapped_in_input", index=dedup.index, dtype="string")
        for col in (
            "n_flank",
            "c_flank",
            "protein_id",
            "gene_id",
            "gene_name",
            "transcript_id",
            "candidate_protein_ids",
            "candidate_gene_ids",
        ):
            if col not in dedup.columns:
                dedup[col] = pd.Series("", index=dedup.index, dtype="string")
        dedup["candidate_flank_pairs"] = pd.Series("[]", index=dedup.index, dtype="string")
        return dedup

    for col in ("protein_id", "gene_id", "gene_name", "transcript_id", "n_flank", "c_flank"):
        if col not in ms_df.columns:
            ms_df[col] = ""
        else:
            ms_df[col] = ms_df[col].astype("string").fillna("")

    dup_mask = ms_df["evidence_row_id"].duplicated(keep=False)
    singletons = ms_df.loc[~dup_mask].copy()
    has_hit = singletons["protein_id"].str.strip().ne("")
    gene_token = singletons["gene_id"].where(
        singletons["gene_id"].str.strip().ne(""), singletons["gene_name"]
    )
    singletons["n_protein_mappings"] = has_hit.astype("int64")
    singletons["n_gene_mappings"] = (has_hit & gene_token.str.strip().ne("")).astype("int64")
    singletons["n_distinct_flank_pairs"] = has_hit.astype("int64")
    singletons["flank_status"] = pd.Series(
        np.where(has_hit, "unique_mapping", "unmapped"),
        index=singletons.index,
        dtype="string",
    )
    singletons.loc[
        ~has_hit, ["n_flank", "c_flank", "protein_id", "gene_id", "gene_name", "transcript_id"]
    ] = ""
    singletons["candidate_protein_ids"] = singletons["protein_id"]
    singletons["candidate_gene_ids"] = gene_token.where(has_hit, "")
    singletons["candidate_flank_pairs"] = pd.Series(
        [
            json.dumps([{"n_flank": nf, "c_flank": cf}]) if ok else "[]"
            for nf, cf, ok in zip(singletons["n_flank"], singletons["c_flank"], has_hit)
        ],
        index=singletons.index,
        dtype="string",
    )

    if not dup_mask.any():
        return singletons.reset_index(drop=True)

    multi = ms_df.loc[dup_mask]
    resolved_rows: list[pd.DataFrame] = []
    base_records = multi.drop_duplicates("evidence_row_id", keep="first").set_index(
        "evidence_row_id", drop=False
    )

    n_prot_list: list[int] = []
    n_gene_list: list[int] = []
    n_flank_pairs_list: list[int] = []
    status_list: list[str] = []
    nf_list: list[str] = []
    cf_list: list[str] = []
    prot_list: list[str] = []
    gid_list: list[str] = []
    gname_list: list[str] = []
    tx_list: list[str] = []
    cand_prots_list: list[str] = []
    cand_genes_list: list[str] = []
    cand_flanks_list: list[str] = []

    has_origin_cols = any(
        c in multi.columns for c in ("origin_transcript_id", "origin_gene_id", "origin_gene_name")
    )

    for ev_id, grp in multi.groupby("evidence_row_id", sort=False):
        _ = ev_id
        mapped = grp[grp["protein_id"].str.strip().ne("")]
        if mapped.empty:
            n_prot_list.append(0)
            n_gene_list.append(0)
            n_flank_pairs_list.append(0)
            status_list.append("unmapped")
            nf_list.append("")
            cf_list.append("")
            prot_list.append("")
            gid_list.append("")
            gname_list.append("")
            tx_list.append("")
            cand_prots_list.append("")
            cand_genes_list.append("")
            cand_flanks_list.append("[]")
            continue

        prots = sorted({p.strip() for p in mapped["protein_id"] if p.strip()})
        genes = sorted(
            {
                (gid.strip() or gname.strip())
                for gid, gname in zip(mapped["gene_id"], mapped["gene_name"])
                if (gid.strip() or gname.strip())
            }
        )
        flank_pairs = sorted({(nf, cf) for nf, cf in zip(mapped["n_flank"], mapped["c_flank"])})

        n_prot_list.append(len(prots))
        n_gene_list.append(len(genes))
        n_flank_pairs_list.append(len(flank_pairs))
        cand_prots_list.append(";".join(prots))
        cand_genes_list.append(";".join(genes))
        cand_flanks_list.append(
            json.dumps([{"n_flank": nf, "c_flank": cf} for nf, cf in flank_pairs])
        )

        if len(prots) == 1 and len(flank_pairs) == 1:
            first = mapped.iloc[0]
            status_list.append("unique_mapping")
            nf_list.append(first["n_flank"])
            cf_list.append(first["c_flank"])
            prot_list.append(first["protein_id"])
            gid_list.append(first["gene_id"])
            gname_list.append(first["gene_name"])
            tx_list.append(first["transcript_id"])
            continue

        if len(flank_pairs) == 1:
            nf, cf = flank_pairs[0]
            status_list.append("concordant_flanks")
            nf_list.append(nf)
            cf_list.append(cf)
            prot_list.append(prots[0] if len(prots) == 1 else "")
            gid_list.append(
                genes[0] if len(genes) == 1 and mapped["gene_id"].str.strip().ne("").all() else ""
            )
            uniq_gnames = {g.strip() for g in mapped["gene_name"] if g.strip()}
            gname_list.append(next(iter(uniq_gnames)) if len(uniq_gnames) == 1 else "")
            uniq_tx = {t.strip() for t in mapped["transcript_id"] if t.strip()}
            tx_list.append(next(iter(uniq_tx)) if len(uniq_tx) == 1 else "")
            continue

        chosen_sub = pd.DataFrame()
        if use_expression_disambiguation and has_origin_cols:
            row0 = mapped.iloc[0]
            orig_tx = str(row0.get("origin_transcript_id", "") or "").strip()
            orig_gid = str(row0.get("origin_gene_id", "") or "").strip()
            orig_gname = str(row0.get("origin_gene_name", "") or "").strip()
            if orig_tx and "transcript_id" in mapped.columns:
                sub = mapped[mapped["transcript_id"].str.strip().eq(orig_tx)]
                if not sub.empty:
                    chosen_sub = sub
            if chosen_sub.empty and orig_gid:
                sub = mapped[mapped["gene_id"].str.strip().eq(orig_gid)]
                if not sub.empty:
                    chosen_sub = sub
            if chosen_sub.empty and orig_gname:
                sub = mapped[mapped["gene_name"].str.strip().eq(orig_gname)]
                if not sub.empty:
                    chosen_sub = sub

        sub_flanks = (
            sorted({(nf, cf) for nf, cf in zip(chosen_sub["n_flank"], chosen_sub["c_flank"])})
            if not chosen_sub.empty
            else []
        )
        if len(sub_flanks) == 1:
            nf, cf = sub_flanks[0]
            first = chosen_sub.iloc[0]
            status_list.append("expression_disambiguated")
            nf_list.append(nf)
            cf_list.append(cf)
            sub_prots = {p.strip() for p in chosen_sub["protein_id"] if p.strip()}
            prot_list.append(next(iter(sub_prots)) if len(sub_prots) == 1 else "")
            gid_list.append(first["gene_id"] if chosen_sub["gene_id"].nunique() == 1 else "")
            gname_list.append(first["gene_name"] if chosen_sub["gene_name"].nunique() == 1 else "")
            tx_list.append(
                first["transcript_id"] if chosen_sub["transcript_id"].nunique() == 1 else ""
            )
        else:
            status_list.append("ambiguous_flanks")
            nf_list.append("")
            cf_list.append("")
            prot_list.append("")
            gid_list.append("")
            gname_list.append("")
            tx_list.append("")

    base_records = base_records.copy()
    base_records["n_protein_mappings"] = pd.Series(
        n_prot_list, index=base_records.index, dtype="int64"
    )
    base_records["n_gene_mappings"] = pd.Series(
        n_gene_list, index=base_records.index, dtype="int64"
    )
    base_records["n_distinct_flank_pairs"] = pd.Series(
        n_flank_pairs_list, index=base_records.index, dtype="int64"
    )
    base_records["flank_status"] = pd.Series(status_list, index=base_records.index, dtype="string")
    base_records["n_flank"] = pd.Series(nf_list, index=base_records.index, dtype="string")
    base_records["c_flank"] = pd.Series(cf_list, index=base_records.index, dtype="string")
    base_records["protein_id"] = pd.Series(prot_list, index=base_records.index, dtype="string")
    base_records["gene_id"] = pd.Series(gid_list, index=base_records.index, dtype="string")
    base_records["gene_name"] = pd.Series(gname_list, index=base_records.index, dtype="string")
    base_records["transcript_id"] = pd.Series(tx_list, index=base_records.index, dtype="string")
    base_records["candidate_protein_ids"] = pd.Series(
        cand_prots_list, index=base_records.index, dtype="string"
    )
    base_records["candidate_gene_ids"] = pd.Series(
        cand_genes_list, index=base_records.index, dtype="string"
    )
    base_records["candidate_flank_pairs"] = pd.Series(
        cand_flanks_list, index=base_records.index, dtype="string"
    )
    resolved_rows.append(singletons)
    resolved_rows.append(base_records.reset_index(drop=True))
    combined = pd.concat(resolved_rows, ignore_index=True, sort=False)
    order = ms_df.drop_duplicates("evidence_row_id", keep="first")["evidence_row_id"]
    return combined.set_index("evidence_row_id").loc[order].reset_index()


def prepare_ms_training_data(
    source: pd.DataFrame | str | Path,
    *,
    mhc_class: str | None = None,
    species: str | None = "Homo sapiens",
    scorable_class_i_only: bool = False,
    allow_study_pools: bool = False,
    allow_predicted_restrictions: bool = False,
    allow_unpaired_class_ii_chains: bool = False,
    allow_modified_peptides: bool = False,
    exclude_potential_contaminants: bool = True,
    use_expression_disambiguation: bool = True,
    recover_reference_genotypes: bool = False,
    max_alleles_per_bag: int | None = None,
    min_peptide_length: int = 7,
    max_peptide_length: int = 45,
) -> TrainingTableSelection:
    """Prepare observation-grain MS presentation / MIL training rows for Presto or MHCflurry.

    Collapses protein-mapping expansions back to one row per ``evidence_row_id``
    so multi-mapped peptides never receive inflated training weight and
    conflicting multi-gene flanks are blanked with ``flank_status="ambiguous_flanks"``
    rather than resolved arbitrarily. Distinguishes monoallelic/experimental
    single restrictions, sample-level MIL bags, predicted restrictions, and
    study-wide allele pools. Pass ``species="all"`` or ``species=None`` for
    multi-species training and ``recover_reference_genotypes=True`` to recover
    source-backed condition-level MHC bags from ``host_mhc_types``, curated cell
    lines, and inbred/transgenic strains (#943, #944).
    """
    if min_peptide_length < 1 or max_peptide_length < min_peptide_length:
        raise ValueError("Invalid peptide length bounds")
    if max_alleles_per_bag is not None and max_alleles_per_bag < 1:
        raise ValueError("max_alleles_per_bag must be >= 1 when set")

    df, bundle_info = _load_source_frame(source, default_evidence_kind="ms")
    ms_raw = df[df["evidence_kind"].astype("string").eq("ms")].reset_index(drop=True)
    n_input_rows_before_collapse = len(ms_raw)
    obs = _collapse_ms_mappings_to_observations(
        ms_raw, use_expression_disambiguation=use_expression_disambiguation
    )
    if recover_reference_genotypes and not obs.empty:
        recovered = recover_mhc_genotypes_for_frame(obs)
        existing_set = (
            obs["mhc_allele_set"].astype("string").fillna("")
            if "mhc_allele_set" in obs.columns
            else pd.Series("", index=obs.index, dtype="string")
        )
        existing_prov = (
            obs["mhc_allele_provenance"].astype("string").fillna("")
            if "mhc_allele_provenance" in obs.columns
            else pd.Series("", index=obs.index, dtype="string")
        )
        needs_fill = existing_set.str.strip().eq("") | existing_prov.isin(["", "unmatched"])
        obs["mhc_allele_set"] = existing_set.where(~needs_fill, recovered["mhc_allele_set"])
        obs["mhc_allele_provenance"] = existing_prov.where(
            ~needs_fill, recovered["mhc_allele_set_provenance"]
        )
        obs["host_species"] = recovered["host_species"]
        obs["is_chimeric_mhc_host"] = recovered["is_chimeric_mhc_host"]

    target_class = normalize_mhc_class_token(mhc_class) if mhc_class else ""
    target_species = (
        normalize_species(species) if species and str(species).strip().casefold() != "all" else ""
    )

    extra_condition_columns = [c for c in CONDITION_COLUMNS if c != "condition_id"]
    accepted_columns = [
        "evidence_row_id",
        "evidence_source_id",
        "provenance_id",
        "provenance_status",
        "peptide",
        "mhc_restriction",
        "mhc_allele_set",
        "mhc_allele_cardinality",
        "candidate_alleles",
        "n_candidate_alleles",
        "candidate_bag_status",
        "candidate_bag_provenance",
        "reported_mhc_restriction",
        "restriction_evidence",
        "mhc_class",
        "mhc_species",
        "host_species",
        "is_chimeric_mhc_host",
        "is_monoallelic",
        "sample_group_key",
        "pmid",
        "condition_id",
        "sample_attribution",
        "attributed_sample_label",
        "is_control_arm",
        "apm_perturbed",
        "supported_condition_ids",
        "excluded_condition_ids",
        *extra_condition_columns,
        "flank_status",
        "n_flank",
        "c_flank",
        "protein_id",
        "gene_id",
        "gene_name",
        "transcript_id",
        "n_protein_mappings",
        "n_gene_mappings",
        "n_distinct_flank_pairs",
        "candidate_protein_ids",
        "candidate_gene_ids",
        "candidate_flank_pairs",
        "assay_method",
        "assay_iri",
        "reference_iri",
        *LINEAGE_COLUMNS,
        *APPLICABILITY_COLUMNS,
    ]
    excluded_columns = [
        "evidence_row_id",
        "evidence_source_id",
        "provenance_id",
        "peptide",
        "mhc_restriction",
        "mhc_allele_set",
        "mhc_allele_provenance",
        "restriction_evidence",
        "pmid",
        "condition_id",
        "flank_status",
        "exclusion_reason",
    ]

    if obs.empty:
        summary = {
            "task": "mhc_ms_presentation",
            "source_bundle": bundle_info,
            "n_input_rows": n_input_rows_before_collapse,
            "n_input_candidate_rows": 0,
            "n_mapping_alternative_rows_collapsed": 0,
            "n_accepted_rows": 0,
            "n_excluded_rows": 0,
            "n_unique_peptides": 0,
            "n_unique_candidate_alleles": 0,
            "n_unique_pmids": 0,
            "by_candidate_bag_status": {},
            "by_flank_status": {},
            "by_exclusion_reason": {},
        }
        return TrainingTableSelection(
            accepted=pd.DataFrame(columns=accepted_columns),
            excluded=pd.DataFrame(columns=excluded_columns),
            summary=summary,
        )

    empty_str = pd.Series("", index=obs.index, dtype="string")
    peptides = obs.get("peptide", empty_str).astype("string").fillna("")
    pep_cache = {
        pep: _is_canonical_peptide(pep, min_len=min_peptide_length, max_len=max_peptide_length)
        for pep in peptides.unique()
    }
    has_ptm = (
        obs["has_ptm"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "has_ptm" in obs.columns
        else np.zeros(len(obs), dtype=bool)
    )
    non_pep = (
        obs["is_non_peptide_ligand"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "is_non_peptide_ligand" in obs.columns
        else np.zeros(len(obs), dtype=bool)
    )
    contaminants = (
        obs["is_potential_contaminant"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "is_potential_contaminant" in obs.columns
        else np.zeros(len(obs), dtype=bool)
    )

    restrictions = obs.get("mhc_restriction", empty_str).astype("string").fillna("")
    res_ev = obs.get("restriction_evidence", empty_str).astype("string").fillna("")
    mono = (
        obs["is_monoallelic"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "is_monoallelic" in obs.columns
        else np.zeros(len(obs), dtype=bool)
    )
    allele_sets = obs.get("mhc_allele_set", empty_str).astype("string").fillna("")
    allele_provs = obs.get("mhc_allele_provenance", empty_str).astype("string").fillna("")
    row_classes = obs.get("mhc_class", empty_str).astype("string").fillna("")
    row_species = obs.get("mhc_species", empty_str).astype("string").fillna("")

    unique_mhc = pd.DataFrame(
        {
            "rest": restrictions,
            "ev": res_ev,
            "mono": mono,
            "aset": allele_sets,
            "aprov": allele_provs,
            "cls": row_classes,
            "sp": row_species,
        }
    ).drop_duplicates()

    mhc_cache: dict[tuple, tuple[str, int, str, str, str, str, str]] = {}
    for rest, ev, is_mono, aset, aprov, r_cls, r_sp in unique_mhc.itertuples(
        index=False, name=None
    ):
        key = (rest, ev, is_mono, aset, aprov, r_cls, r_sp)
        if classify_allele_resolution(rest) == "four_digit":
            allele, comp, a_cls, a_sp = _allele_completeness_and_identity(
                rest, scorable_class_i_only=scorable_class_i_only
            )
            if not allele:
                mhc_cache[key] = ("", 0, "", "", "", "", comp)
                continue
            if comp == "single_chain_class_ii_unpaired" and not allow_unpaired_class_ii_chains:
                mhc_cache[key] = ("", 0, "", "", "", "", "unpaired_class_ii_chain")
                continue
            eff_cls = a_cls or normalize_mhc_class_token(r_cls)
            eff_sp = a_sp or (normalize_species(r_sp) if r_sp else "")
            if target_class and eff_cls != target_class:
                mhc_cache[key] = ("", 0, "", "", "", "", "mhc_class_mismatch")
                continue
            if target_species and eff_sp != target_species:
                mhc_cache[key] = ("", 0, "", "", "", "", "mhc_species_mismatch")
                continue
            if ev == "predicted":
                if not allow_predicted_restrictions:
                    mhc_cache[key] = ("", 0, "", "", "", "", "predicted_restriction_excluded")
                    continue
                bag_status = "predicted_single_restriction"
            elif is_mono or ev == "monoallelic":
                bag_status = "monoallelic_single"
            else:
                bag_status = "experimental_single"
            mhc_cache[key] = (
                allele,
                1,
                bag_status,
                "reported_restriction",
                eff_cls,
                eff_sp,
                "",
            )
            continue

        # Coarse or multi-allele restriction: resolve through mhc_allele_set.
        if not aset or aprov in {"", "unmatched"}:
            mhc_cache[key] = ("", 0, "", "", "", "", "unresolved_mhc_candidates")
            continue
        if aprov in _STUDY_POOL_PROVENANCES:
            if not allow_study_pools:
                mhc_cache[key] = ("", 0, "", "", "", "", "study_pool_candidates_excluded")
                continue
            bag_status = "study_pool_candidates"
        elif aprov in _SAMPLE_CANDIDATE_PROVENANCES:
            bag_status = "sample_candidate_bag"
        else:
            bag_status = "sample_candidate_bag"

        raw_tokens = [tok.strip() for tok in aset.split(";") if tok.strip()]
        if not raw_tokens:
            mhc_cache[key] = ("", 0, "", "", "", "", "unresolved_mhc_candidates")
            continue

        if scorable_class_i_only:
            scope = class_i_prediction_scope(" ".join(raw_tokens))
            if not scope.is_eligible or not scope.scorable:
                mhc_cache[key] = ("", 0, "", "", "", "", "not_scorable_class_i_bag")
                continue
            resolved_tokens = list(scope.scorable)
            eff_cls = "I"
            eff_sp = "Homo sapiens"
        else:
            resolved_tokens = []
            classes_seen: set[str] = set()
            species_seen: set[str] = set()
            bag_error = ""
            for tok in raw_tokens:
                allele, comp, a_cls, a_sp = _allele_completeness_and_identity(
                    tok, scorable_class_i_only=False
                )
                if not allele:
                    bag_error = "unresolved_mhc_candidates"
                    break
                if comp == "single_chain_class_ii_unpaired" and not allow_unpaired_class_ii_chains:
                    bag_error = "unpaired_class_ii_chain_in_bag"
                    break
                resolved_tokens.append(allele)
                if a_cls:
                    classes_seen.add(a_cls)
                if a_sp:
                    species_seen.add(a_sp)
            if bag_error:
                mhc_cache[key] = ("", 0, "", "", "", "", bag_error)
                continue
            resolved_tokens = sorted(dict.fromkeys(resolved_tokens))
            eff_cls = (
                next(iter(classes_seen))
                if len(classes_seen) == 1
                else normalize_mhc_class_token(r_cls)
            )
            eff_sp = (
                next(iter(species_seen))
                if len(species_seen) == 1
                else (normalize_species(r_sp) if r_sp else "")
            )

        if target_class and eff_cls != target_class:
            mhc_cache[key] = ("", 0, "", "", "", "", "mhc_class_mismatch")
            continue
        if target_species and eff_sp != target_species:
            mhc_cache[key] = ("", 0, "", "", "", "", "mhc_species_mismatch")
            continue
        if max_alleles_per_bag is not None and len(resolved_tokens) > max_alleles_per_bag:
            mhc_cache[key] = ("", 0, "", "", "", "", "exceeds_max_alleles_per_bag")
            continue

        mhc_cache[key] = (
            ";".join(resolved_tokens),
            len(resolved_tokens),
            bag_status,
            aprov,
            eff_cls,
            eff_sp,
            "",
        )

    reasons: list[str] = []
    bag_alleles: list[str] = []
    bag_sizes: list[int] = []
    bag_statuses: list[str] = []
    bag_provs: list[str] = []
    bag_classes: list[str] = []
    bag_species: list[str] = []

    for idx, (pep, rest, ev, is_mono, aset, aprov, r_cls, r_sp) in enumerate(
        zip(
            peptides,
            restrictions,
            res_ev,
            mono,
            allele_sets,
            allele_provs,
            row_classes,
            row_species,
        )
    ):
        if non_pep[idx]:
            reasons.append("non_peptide_ligand")
            bag_alleles.append("")
            bag_sizes.append(0)
            bag_statuses.append("")
            bag_provs.append("")
            bag_classes.append("")
            bag_species.append("")
            continue
        if exclude_potential_contaminants and contaminants[idx]:
            reasons.append("potential_contaminant_excluded")
            bag_alleles.append("")
            bag_sizes.append(0)
            bag_statuses.append("")
            bag_provs.append("")
            bag_classes.append("")
            bag_species.append("")
            continue
        if not allow_modified_peptides and has_ptm[idx]:
            reasons.append("modified_peptide_excluded")
            bag_alleles.append("")
            bag_sizes.append(0)
            bag_statuses.append("")
            bag_provs.append("")
            bag_classes.append("")
            bag_species.append("")
            continue
        pep_ok, pep_reason = pep_cache[pep]
        if not pep_ok:
            reasons.append(pep_reason)
            bag_alleles.append("")
            bag_sizes.append(0)
            bag_statuses.append("")
            bag_provs.append("")
            bag_classes.append("")
            bag_species.append("")
            continue

        c_alleles, c_size, c_status, c_prov, c_cls, c_sp, mhc_err = mhc_cache[
            (rest, ev, is_mono, aset, aprov, r_cls, r_sp)
        ]
        if mhc_err:
            reasons.append(mhc_err)
            bag_alleles.append("")
            bag_sizes.append(0)
            bag_statuses.append("")
            bag_provs.append("")
            bag_classes.append("")
            bag_species.append("")
            continue

        reasons.append("")
        bag_alleles.append(c_alleles)
        bag_sizes.append(c_size)
        bag_statuses.append(c_status)
        bag_provs.append(c_prov)
        bag_classes.append(c_cls)
        bag_species.append(c_sp)

    reason_series = pd.Series(reasons, index=obs.index, dtype="string")
    ok_mask = reason_series.eq("")
    ok_pos = np.flatnonzero(ok_mask.to_numpy(dtype=bool))

    acc_src = obs.loc[ok_mask].copy()
    acc_src["peptide"] = peptides.loc[ok_mask].str.strip().str.upper()
    acc_src["candidate_alleles"] = pd.Series(
        [bag_alleles[i] for i in ok_pos], index=acc_src.index, dtype="string"
    )
    acc_src["mhc_allele_set"] = acc_src["candidate_alleles"]
    acc_src["n_candidate_alleles"] = pd.Series(
        [bag_sizes[i] for i in ok_pos], index=acc_src.index, dtype="int64"
    )
    acc_src["mhc_allele_cardinality"] = pd.Series(
        np.where(acc_src["n_candidate_alleles"].eq(1), "monoallelic", "multi_allelic_bag"),
        index=acc_src.index,
        dtype="string",
    )
    acc_src["candidate_bag_status"] = pd.Series(
        [bag_statuses[i] for i in ok_pos], index=acc_src.index, dtype="string"
    )
    acc_src["candidate_bag_provenance"] = pd.Series(
        [bag_provs[i] for i in ok_pos], index=acc_src.index, dtype="string"
    )
    acc_src["reported_mhc_restriction"] = restrictions.loc[ok_mask]
    acc_src["mhc_restriction"] = np.where(
        acc_src["n_candidate_alleles"].eq(1),
        acc_src["candidate_alleles"],
        acc_src["reported_mhc_restriction"],
    )
    acc_src["mhc_class"] = pd.Series(
        [bag_classes[i] for i in ok_pos], index=acc_src.index, dtype="string"
    )
    acc_src["mhc_species"] = pd.Series(
        [bag_species[i] for i in ok_pos], index=acc_src.index, dtype="string"
    )
    if "host_species" not in acc_src.columns:
        host_col = (
            acc_src["host"].astype("string").fillna("")
            if "host" in acc_src.columns
            else acc_src["species"].astype("string").fillna("")
            if "species" in acc_src.columns
            else pd.Series("", index=acc_src.index, dtype="string")
        )
        inferred_hosts = [
            infer_host_species(h) or msp
            for h, msp in zip(host_col.tolist(), acc_src["mhc_species"].tolist())
        ]
        acc_src["host_species"] = pd.Series(inferred_hosts, index=acc_src.index, dtype="string")
    acc_src["is_chimeric_mhc_host"] = pd.Series(
        [
            bool(msp and hsp and msp != hsp)
            for msp, hsp in zip(
                acc_src["mhc_species"].astype("string").fillna("").tolist(),
                acc_src["host_species"].astype("string").fillna("").tolist(),
            )
        ],
        index=acc_src.index,
        dtype="bool",
    )

    pmids_str = (
        pd.to_numeric(acc_src.get("pmid"), errors="coerce")
        .astype("Int64")
        .astype("string")
        .fillna("unknown_pmid")
        if "pmid" in acc_src.columns
        else pd.Series("unknown_pmid", index=acc_src.index, dtype="string")
    )
    cond_str = (
        acc_src["condition_id"].astype("string").fillna("")
        if "condition_id" in acc_src.columns
        else pd.Series("", index=acc_src.index, dtype="string")
    )
    attr_lbl = (
        acc_src["attributed_sample_label"].astype("string").fillna("")
        if "attributed_sample_label" in acc_src.columns
        else pd.Series("", index=acc_src.index, dtype="string")
    )
    sample_keys = [
        f"pmid:{p}|arm:{c}|mhc:{m}"
        if c
        else f"pmid:{p}|label:{lbl}|mhc:{m}"
        if lbl
        else f"pmid:{p}|mhc:{m}"
        for p, c, lbl, m in zip(pmids_str, cond_str, attr_lbl, acc_src["candidate_alleles"])
    ]
    acc_src["sample_group_key"] = pd.Series(sample_keys, index=acc_src.index, dtype="string")

    for col in accepted_columns:
        if col not in acc_src.columns:
            acc_src[col] = False if col == "is_monoallelic" else ""
    accepted = acc_src[accepted_columns].reset_index(drop=True)

    excl_src = obs.loc[~ok_mask].copy()
    excl_src["exclusion_reason"] = reason_series.loc[~ok_mask]
    for col in excluded_columns:
        if col not in excl_src.columns:
            excl_src[col] = ""
    excluded = excl_src[excluded_columns].reset_index(drop=True)

    unique_alleles: set[str] = set()
    for bag in accepted["candidate_alleles"].unique():
        if bag:
            unique_alleles.update(bag.split(";"))
    pmid_count = (
        int(pd.to_numeric(accepted["pmid"], errors="coerce").dropna().nunique())
        if len(accepted)
        else 0
    )
    summary = {
        "task": "mhc_ms_presentation",
        "source_bundle": bundle_info,
        "n_input_rows": n_input_rows_before_collapse,
        "n_input_candidate_rows": len(obs),
        "n_mapping_alternative_rows_collapsed": n_input_rows_before_collapse - len(obs),
        "n_accepted_rows": len(accepted),
        "n_excluded_rows": len(excluded),
        "n_unique_peptides": int(accepted["peptide"].nunique()),
        "n_unique_candidate_alleles": len(unique_alleles),
        "n_unique_pmids": pmid_count,
        "by_candidate_bag_status": dict(sorted(Counter(accepted["candidate_bag_status"]).items())),
        "by_flank_status": dict(sorted(Counter(accepted["flank_status"]).items())),
        "by_exclusion_reason": dict(sorted(Counter(excluded["exclusion_reason"]).items())),
    }
    return TrainingTableSelection(accepted=accepted, excluded=excluded, summary=summary)


def to_mhcflurry_presentation_inputs(
    selection_or_df: TrainingTableSelection | pd.DataFrame,
    *,
    max_alleles_per_sample: int = 6,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Convert prepared MS rows into MHCflurry presentation/processing inputs.

    Returns ``(hits_df, sample_alleles)`` where ``sample_alleles`` maps each
    ``sample_id`` (from ``sample_group_key``) to a list of at most
    ``max_alleles_per_sample`` scorable human class-I protein designations and
    ``hits_df`` retains ``peptide``, ``sample_id``, ``n_flank``, ``c_flank``,
    ``flank_status``, and Hitlist observation/lineage identifiers.
    """
    df = (
        selection_or_df.accepted
        if isinstance(selection_or_df, TrainingTableSelection)
        else selection_or_df
    )
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a TrainingTableSelection or pandas DataFrame")
    if max_alleles_per_sample < 1:
        raise ValueError("max_alleles_per_sample must be >= 1")
    required = {"peptide", "candidate_alleles", "sample_group_key"}
    if not required <= set(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(
            f"Input is missing {missing}; pass output of prepare_ms_training_data(..., scorable_class_i_only=True)"
        )

    sample_alleles: dict[str, list[str]] = {}
    for sample_key, bag in (
        df[["sample_group_key", "candidate_alleles"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        tokens = [t for t in str(bag).split(";") if t]
        scope = class_i_prediction_scope(" ".join(tokens))
        if not scope.is_eligible or len(scope.scorable) != len(tokens):
            raise ValueError(
                f"Sample {sample_key!r} has non-scorable class-I alleles {tokens!r}; "
                "call prepare_ms_training_data(..., scorable_class_i_only=True) first"
            )
        if len(scope.scorable) > max_alleles_per_sample:
            raise ValueError(
                f"Sample {sample_key!r} has {len(scope.scorable)} alleles > max_alleles_per_sample={max_alleles_per_sample}; "
                f"call prepare_ms_training_data(..., max_alleles_per_bag={max_alleles_per_sample})"
            )
        if sample_key in sample_alleles and sample_alleles[sample_key] != list(scope.scorable):
            raise ValueError(f"Conflicting allele bags for sample_id {sample_key!r}")
        sample_alleles[str(sample_key)] = list(scope.scorable)

    hits = df.rename(columns={"sample_group_key": "sample_id"}).copy()
    return hits.reset_index(drop=True), sample_alleles
