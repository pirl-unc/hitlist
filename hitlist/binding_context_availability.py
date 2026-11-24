"""Source-availability audit for the retained MHC binding observations.

This successor is intentionally an audit relation, not a protocol imputation
layer. Native assay values remain untouched, while study-scoped review claims
are represented as study-scoped availability rather than copied onto an assay
as if they were measured per row.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .binding_assay_context import (
    CONTEXT_AVAILABILITY_FIELDS,
    _build_study_key_series,
    _pick_numeric_series,
    _pick_series,
    binding_protocol_registry_contract,
    review_binding_assay_methods,
    review_binding_censoring,
    verify_binding_assay_context_package,
)
from .evidence_snapshot import snapshot_file
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 14
CATALOG_SCHEMA_VERSION = 19
REVISION = "binding_assay_context_availability_v1"
PARENT_SCHEMA_VERSION = 10
PARENT_CATALOG_SCHEMA_VERSION = 15
PARENT_MANIFEST = Path("manifests/binding_assay_context.json")
ROOT = Path("binding_context")
AUDIT_TABLE = ROOT / "availability.parquet"
COVERAGE = ROOT / "coverage.json"
EXPECTED_ROWS = 25_896
EXPECTED_IC50_ROWS = 8_084
EXPECTED_KD_IC50_ROWS = 3_232

AUDIT_COLUMNS = (
    "observation_id",
    "source_assay_id",
    "source_pmid",
    "source_dataset",
    "endpoint_reported",
    "endpoint_family",
    "assay_method",
    "quantitative_raw",
    "quantitative_value",
    "measurement_unit",
    "measurement_inequality",
    "assay_comments_raw",
    "row_comment_status",
    "source_review_status",
    "context_scope",
    "assay_format_status",
    "temperature_status",
    "ph_status",
    "tracer_status",
    "competitor_status",
    "incubation_status",
    "range_status",
    "replicate_status",
    "uncertainty_status",
    "kd_conversion_status",
    "audit_reason",
)

FLOAT_COLUMNS = {"quantitative_value"}

CLAIM_FIELD_MAP = {
    "assay_format_status": "assay_format",
    "temperature_status": "incubation_temperature_alternatives",
    "ph_status": "assay_pH",
    "tracer_status": "tracer_identity",
    "competitor_status": "competitor_concentration_span",
    "incubation_status": "incubation_duration",
    "range_status": "tracer_concentration_range",
    "replicate_status": "replicate_identifiers",
    "uncertainty_status": "measurement_uncertainty",
}


def _schema():
    return pa.schema(
        [(name, pa.float64() if name in FLOAT_COLUMNS else pa.string()) for name in AUDIT_COLUMNS]
    )


def _endpoint_family(endpoint):
    if endpoint == "half maximal inhibitory concentration (IC50)":
        return "literal_ic50"
    if endpoint == "dissociation constant KD (~IC50)":
        return "kd_ic50"
    return "other_binding"


def _status_from_claim(claim, *, reviewed):
    if not reviewed or claim is None:
        return "unreviewed"
    status = claim["claim_status"]
    if status == "unreported":
        return "unreported"
    if status == "explicit_absence":
        return "explicit_absence"
    if status in {
        "reported_range_per_assay_unresolved",
        "reported_alternatives_per_assay_unresolved",
    }:
        return "unresolved"
    return "study_scope_only"


def _number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _claims(directory):
    table = pq.read_table(
        directory / "binding_review/binding_protocol_claims.parquet",
        columns=["assertion_predicate", "claim_status", "source_pmid", "claim_applicability"],
    )
    claims = {}
    for row in table.to_pylist():
        if row["claim_applicability"] == "all_study_binding_assays":
            claims[row["assertion_predicate"]] = row
    return claims


def _compile(directory):
    dataset = ds.dataset(directory / "cohort_native/training.parquet", format="parquet")
    available = set(dataset.schema.names)
    required = {
        "provenance_id",
        "assay_iri",
        "pmid",
        "source",
        "response_measured",
        "assay_method",
        "quantitative_measurement",
        "quantitative_value",
        "measurement_units",
        "measurement_inequality",
        "is_binding_assay",
    }
    if not required <= available:
        raise ValueError(f"Binding cohort missing audit fields: {sorted(required - available)}")
    columns = sorted(required | {"assay_comments"} & available)
    table = dataset.to_table(columns=columns, filter=ds.field("is_binding_assay"))
    source_claims = _claims(directory)
    rows = []
    ids = set()
    endpoint_counts = Counter()
    review_counts = Counter()
    status_counts = {field: Counter() for field in CLAIM_FIELD_MAP}
    for source in table.to_pylist():
        observation_id = source["provenance_id"]
        if observation_id in ids:
            raise ValueError(f"Duplicate binding audit observation: {observation_id}")
        ids.add(observation_id)
        endpoint = source["response_measured"] or ""
        endpoint_family = _endpoint_family(endpoint)
        reviewed = source["pmid"] == 28481943
        review_status = "source_reviewed_study_scope" if reviewed else "not_source_reviewed"
        endpoint_counts[endpoint_family] += 1
        review_counts[review_status] += 1
        statuses = {}
        for field, predicate in CLAIM_FIELD_MAP.items():
            statuses[field] = _status_from_claim(source_claims.get(predicate), reviewed=reviewed)
            status_counts[field][statuses[field]] += 1
        comments = source.get("assay_comments") or ""
        rows.append(
            {
                "observation_id": observation_id,
                "source_assay_id": source["assay_iri"],
                "source_pmid": str(source["pmid"]) if source["pmid"] is not None else None,
                "source_dataset": source["source"],
                "endpoint_reported": endpoint or None,
                "endpoint_family": endpoint_family,
                "assay_method": source["assay_method"] or None,
                "quantitative_raw": source["quantitative_measurement"] or None,
                "quantitative_value": _number(source["quantitative_value"]),
                "measurement_unit": source["measurement_units"] or None,
                "measurement_inequality": source["measurement_inequality"] or None,
                "assay_comments_raw": comments or None,
                "row_comment_status": "reported_row_text" if comments.strip() else "unreported",
                "source_review_status": review_status,
                "context_scope": "study_protocol_only" if reviewed else "row_native_only",
                **statuses,
                "kd_conversion_status": "not_performed",
                "audit_reason": (
                    "literal_ic50_endpoint"
                    if endpoint_family == "literal_ic50"
                    else "kd_endpoint_kept_separate"
                    if endpoint_family == "kd_ic50"
                    else "binding_endpoint_not_literal_ic50"
                ),
            }
        )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Binding audit row count changed: {len(rows)}")
    if endpoint_counts["literal_ic50"] != EXPECTED_IC50_ROWS:
        raise ValueError("Binding audit IC50 count changed")
    if endpoint_counts["kd_ic50"] != EXPECTED_KD_IC50_ROWS:
        raise ValueError("Binding audit KD (~IC50) count changed")
    coverage = {
        "n_binding_rows": len(rows),
        "n_unique_observation_ids": len(ids),
        "n_literal_ic50_rows": endpoint_counts["literal_ic50"],
        "n_kd_ic50_rows": endpoint_counts["kd_ic50"],
        "by_endpoint_family": dict(sorted(endpoint_counts.items())),
        "by_source_review_status": dict(sorted(review_counts.items())),
        "field_status_counts": {
            field: dict(sorted(counts.items())) for field, counts in sorted(status_counts.items())
        },
        "native_rows_unchanged": True,
        "kd_conversion_performed": False,
    }
    return pa.Table.from_pylist(rows, schema=_schema()), coverage


def _manifest_files(staging):
    return {
        str(path.relative_to(staging)): file_digest(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path != staging / "manifest.json"
    }


def verify_binding_context_availability_package(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("parent_kind") != "binding_review"
        or manifest.get("parent_manifest_path") != str(PARENT_MANIFEST)
        or manifest.get("revision") != REVISION
    ):
        raise ValueError("Unsupported binding context availability package")
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(manifest.get("files", {})) != actual:
        raise ValueError("Binding context availability inventory mismatch")
    for name, digest in manifest["files"].items():
        if file_digest(directory / name)["sha256"] != digest["sha256"]:
            raise ValueError(f"Changed binding audit artifact: {name}")
    parent = verify_binding_assay_context_package(
        directory, manifest_path=PARENT_MANIFEST, allow_additional_files=True
    )
    if (
        parent.get("schema_version") != PARENT_SCHEMA_VERSION
        or parent.get("catalog_schema_version") != PARENT_CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("Binding audit predecessor is not schema-10 binding review")
    if file_digest(directory / PARENT_MANIFEST)["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("Binding audit predecessor changed")
    expected, coverage = _compile(directory)
    actual_table = pq.read_table(directory / AUDIT_TABLE)
    if not actual_table.equals(expected, check_metadata=False):
        raise ValueError("Binding availability audit rows changed")
    actual_coverage = json.loads((directory / COVERAGE).read_text())
    if actual_coverage != coverage or manifest.get("reconciliation") != coverage:
        raise ValueError("Binding availability coverage changed")
    return manifest


def write_binding_context_availability_package(predecessor_package, destination):
    """Append the one-row-per-binding-observation availability audit."""
    predecessor_package = Path(predecessor_package)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    parent = verify_binding_assay_context_package(predecessor_package)
    if (
        parent.get("schema_version") != PARENT_SCHEMA_VERSION
        or parent.get("catalog_schema_version") != PARENT_CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("Binding availability audit requires schema-10 predecessor")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in parent["files"]:
            snapshot_file(predecessor_package / name, staging / name)
        snapshot_file(predecessor_package / "manifest.json", staging / PARENT_MANIFEST)
        (staging / ROOT).mkdir(parents=True, exist_ok=True)
        audit, coverage = _compile(staging)
        pq.write_table(audit, staging / AUDIT_TABLE, compression="zstd")
        (staging / COVERAGE).write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "parent_kind": "binding_review",
            "parent_manifest_path": str(PARENT_MANIFEST),
            "parent_manifest_sha256": file_digest(staging / PARENT_MANIFEST)["sha256"],
            "adjudication_manifest_path": parent.get("adjudication_manifest_path"),
            "reference": parent.get("reference"),
            "appended_sources": parent.get("appended_sources"),
            "parent_review_revision": parent.get("review_revision"),
            "reconciliation": coverage,
            "files": _manifest_files(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_binding_context_availability_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


BINDING_ENDPOINT_FAMILIES: frozenset[str] = frozenset(
    {
        "literal_ic50",
        "kd_ic50",
        "kd_ec50",
        "literal_kd",
        "ec50",
        "ka",
        "half_life",
        "tm",
        "off_rate",
        "on_rate",
        "qualitative_binding",
        "other_binding",
    }
)

_ENDPOINT_FAMILY_ALIASES: dict[str, str] = {
    "half maximal inhibitory concentration (ic50)": "literal_ic50",
    "ic50": "literal_ic50",
    "literal_ic50": "literal_ic50",
    "dissociation constant kd (~ic50)": "kd_ic50",
    "kd (~ic50)": "kd_ic50",
    "kd_ic50": "kd_ic50",
    "dissociation constant kd (~ec50)": "kd_ec50",
    "kd (~ec50)": "kd_ec50",
    "kd_ec50": "kd_ec50",
    "dissociation constant kd": "literal_kd",
    "kd": "literal_kd",
    "literal_kd": "literal_kd",
    "half maximal effective concentration (ec50)": "ec50",
    "ec50": "ec50",
    "association constant ka": "ka",
    "ka": "ka",
    "half life": "half_life",
    "t1/2": "half_life",
    "half_life": "half_life",
    "stability_t_half": "half_life",
    "50% dissociation temperature": "tm",
    "tm": "tm",
    "off rate": "off_rate",
    "koff": "off_rate",
    "off_rate": "off_rate",
    "on rate": "on_rate",
    "kon": "on_rate",
    "on_rate": "on_rate",
    "qualitative binding": "qualitative_binding",
    "mhc binding": "qualitative_binding",
    "qualitative_binding": "qualitative_binding",
}

_CONCENTRATION_ENDPOINT_FAMILIES: frozenset[str] = frozenset(
    {"literal_ic50", "kd_ic50", "kd_ec50", "literal_kd", "ec50"}
)

REPLICATE_GROUP_COLUMNS: tuple[str, ...] = (
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
    "n_distinct_qualitative",
    "has_conflicting_qualitative",
    "has_conflicting_inequality",
    "value_min",
    "value_median",
    "value_max",
    "aggregation_policy",
)


def classify_binding_endpoint_family(endpoint: str | None) -> str:
    """Classify a reported binding or stability endpoint into a canonical symmetric family (#785).

    Keeps ``literal_ic50`` strictly separate from ``kd_ic50``, ``kd_ec50``,
    ``literal_kd``, and kinetic/thermal stability endpoints (``half_life``,
    ``tm``, ``off_rate``, ``on_rate``).
    """
    norm = (endpoint or "").strip().casefold()
    if not norm:
        return "other_binding"
    return _ENDPOINT_FAMILY_ALIASES.get(norm, "other_binding")


def _replicate_group_digest(parts: tuple[str, ...]) -> str:
    return "repgrp:" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _extract_audit_keys(df: pd.DataFrame) -> pd.DataFrame:
    sources = _pick_series(df, ("source", "source_dataset", "measurement_source"))
    study_keys = _build_study_key_series(df)
    pmids = _pick_series(df, ("pmid", "source_pmid"))
    peptides = _pick_series(df, ("peptide", "peptide_sequence"))
    mhcs = _pick_series(df, ("mhc_restriction", "reported_mhc_restriction", "allele"))
    methods = _pick_series(df, ("assay_method",))
    endpoints_raw = _pick_series(
        df, ("response_measured", "endpoint_reported", "measurement_kind", "measurement_type")
    )
    endpoint_families = endpoints_raw.map(classify_binding_endpoint_family).astype("string")
    units = _pick_series(df, ("measurement_unit", "measurement_units"))
    ineq = _pick_series(df, ("raw_measurement_inequality", "measurement_inequality"))
    vals = _pick_numeric_series(df, ("measurement_value", "quantitative_value"))
    quals = _pick_series(df, ("qualitative_measurement", "qualitative_result")).str.casefold()
    return pd.DataFrame(
        {
            "source_dataset": sources,
            "study_key": study_keys,
            "pmid": pmids,
            "peptide": peptides,
            "mhc_restriction": mhcs,
            "assay_method": methods,
            "endpoint_family": endpoint_families,
            "measurement_unit": units,
            "_ineq": ineq,
            "_val": vals,
            "_qual": quals,
        },
        index=df.index,
    )


def _replicate_group_ids_for_keys(
    keys: pd.DataFrame, group_cols: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Compute per-row replicate_group_id and replicate_group_n_observations vectorially."""
    grp = keys.groupby(group_cols, sort=False, observed=True)
    codes = grp.ngroup()
    sizes = grp["peptide"].transform("size").astype("Int64")
    first_rows = keys.loc[~codes.duplicated(keep="first"), group_cols]
    first_codes = codes.loc[first_rows.index].to_numpy()
    code_to_gid = {
        int(code): _replicate_group_digest(tuple(str(x) for x in tup))
        for code, tup in zip(
            first_codes, first_rows.itertuples(index=False, name=None), strict=True
        )
    }
    row_gids = codes.map(code_to_gid).astype("string")
    return row_gids, sizes


def build_binding_replicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Build a separate-grain replicate/aggregation table for binding and stability assays (#721).

    Groups repeated measurements within ``(source_dataset, study_key, peptide,
    mhc_restriction, assay_method, endpoint_family, measurement_unit)`` at their
    own grain so joining replicate provenance onto observations never multiplies
    observation counts or training weights.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(df) == 0:
        return pd.DataFrame(columns=list(REPLICATE_GROUP_COLUMNS))

    keys = _extract_audit_keys(df)
    group_cols = [
        "source_dataset",
        "study_key",
        "peptide",
        "mhc_restriction",
        "assay_method",
        "endpoint_family",
        "measurement_unit",
    ]
    work = keys[group_cols].copy()
    work["_val"] = keys["_val"]
    work["_is_exact_or_blank"] = keys["_ineq"].isin({"", "="}).astype("int64")
    work["_is_right"] = keys["_ineq"].isin({">", ">="}).astype("int64")
    work["_is_left"] = keys["_ineq"].isin({"<", "<="}).astype("int64")
    work["_nonempty_ineq"] = keys["_ineq"].replace("", pd.NA)
    work["_nonempty_qual"] = keys["_qual"].replace("", pd.NA)
    work["_is_pos"] = keys["_qual"].str.startswith("positive").fillna(False)
    work["_is_neg"] = keys["_qual"].eq("negative").fillna(False)
    work["_pmid_clean"] = keys["pmid"].where(
        keys["pmid"].ne("") & ~keys["pmid"].str.casefold().isin({"nan", "none", "<na>"}),
        pd.NA,
    )

    agg = work.groupby(group_cols, sort=False, observed=True, as_index=False).agg(
        n_observations=("_is_exact_or_blank", "size"),
        n_exact_or_blank=("_is_exact_or_blank", "sum"),
        n_right_censored=("_is_right", "sum"),
        n_left_censored=("_is_left", "sum"),
        n_distinct_qualitative=("_nonempty_qual", "nunique"),
        _any_pos=("_is_pos", "any"),
        _any_neg=("_is_neg", "any"),
        _n_nonempty_ineq=("_nonempty_ineq", "nunique"),
        value_min=("_val", "min"),
        value_median=("_val", "median"),
        value_max=("_val", "max"),
        _pmid_first=("_pmid_clean", "first"),
        _pmid_nunique=("_pmid_clean", "nunique"),
    )

    agg["replicate_group_id"] = pd.Series(
        [
            _replicate_group_digest(tuple(str(x) for x in tup))
            for tup in agg[group_cols].itertuples(index=False, name=None)
        ],
        dtype="string",
    )
    pmid_series = agg["_pmid_first"].astype("string").fillna("")
    multi_pmid_mask = agg["_pmid_nunique"] > 1
    if multi_pmid_mask.any():
        multi_keys = set(agg.loc[multi_pmid_mask, group_cols].itertuples(index=False, name=None))
        pmids_by_key: dict[tuple, set[str]] = {}
        for row_tup, pval in zip(
            work[group_cols].itertuples(index=False, name=None),
            work["_pmid_clean"].tolist(),
            strict=True,
        ):
            if row_tup in multi_keys and pd.notna(pval):
                pmids_by_key.setdefault(row_tup, set()).add(str(pval))
        for idx in agg.index[multi_pmid_mask]:
            row_tup = tuple(agg.loc[idx, c] for c in group_cols)
            pmid_series.loc[idx] = ";".join(sorted(pmids_by_key.get(row_tup, set())))

    agg["pmid"] = pmid_series
    agg["has_conflicting_qualitative"] = (agg["_any_pos"] & agg["_any_neg"]).astype(bool)
    agg["has_conflicting_inequality"] = (
        ((agg["n_right_censored"] > 0) & (agg["n_left_censored"] > 0))
        | (
            ((agg["n_right_censored"] > 0) | (agg["n_left_censored"] > 0))
            & (agg["n_exact_or_blank"] > 0)
        )
        | (agg["_n_nonempty_ineq"] > 1)
    ).astype(bool)
    agg["aggregation_policy"] = pd.Series(
        np.where(
            agg["n_observations"] == 1,
            "singleton_observation",
            "multi_observation_do_not_multiply_weights",
        ),
        dtype="string",
    )
    return agg[list(REPLICATE_GROUP_COLUMNS)]


def audit_binding_context_availability(
    df: pd.DataFrame,
    *,
    protocol_registry: dict | None = None,
    include_method_review: bool = True,
    include_censor_review: bool = True,
    include_replicate_groups: bool = True,
) -> pd.DataFrame:
    """Attach symmetric, non-destructive binding & stability context availability (#721, #785).

    Operates 1:1 on any binding or stability DataFrame (IEDB, CEDAR,
    NetMHCstabpan, ``binding.parquet``, or ``generate_training_table()``) with
    zero row multiplication and zero mutation of native endpoints, values,
    units, or inequalities.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    registry = (
        binding_protocol_registry_contract() if protocol_registry is None else protocol_registry
    )
    result = df.copy()
    if len(result) == 0:
        for col in (
            "endpoint_family",
            "row_comment_status",
            "source_review_status",
            "protocol_id",
            "context_scope",
            *CONTEXT_AVAILABILITY_FIELDS,
            "kd_conversion_status",
        ):
            result[col] = pd.Series(dtype="string")
        if include_method_review and "method_review_status" not in result.columns:
            result = review_binding_assay_methods(result)
        if include_censor_review and "censor_review_status" not in result.columns:
            result = review_binding_censoring(result, protocol_registry=registry)
        if include_replicate_groups:
            result["replicate_group_id"] = pd.Series(dtype="string")
            result["replicate_group_n_observations"] = pd.Series(dtype="Int64")
        return result

    keys = _extract_audit_keys(result)
    comments = _pick_series(result, ("assay_comments", "assay_comments_raw"))
    mhc_classes = _pick_series(result, ("mhc_class",))

    endpoint_family = keys["endpoint_family"]
    row_comment_status = pd.Series("unreported", index=result.index, dtype="string")
    row_comment_status.loc[comments.ne("")] = "reported_row_text"

    review_status = pd.Series("not_source_reviewed", index=result.index, dtype="string")
    protocol_id = pd.Series("", index=result.index, dtype="string")
    context_scope = pd.Series("row_native_only", index=result.index, dtype="string")
    avail_cols = {
        field: pd.Series("unreviewed", index=result.index, dtype="string")
        for field in CONTEXT_AVAILABILITY_FIELDS
    }
    kd_conv = pd.Series("not_applicable", index=result.index, dtype="string")
    kd_conv.loc[endpoint_family.isin(_CONCENTRATION_ENDPOINT_FAMILIES)] = "not_performed"

    for proto in registry.get("protocols", []):
        pmid_str = str(proto["source"]["pmid"])
        scope = proto.get("scope") or {}
        mask = keys["pmid"].eq(pmid_str)
        allowed_methods = scope.get("allowed_methods") or []
        if allowed_methods:
            mask = mask & keys["assay_method"].isin(set(allowed_methods))
        allowed_fams = scope.get("allowed_endpoint_families") or []
        if allowed_fams:
            mask = mask & endpoint_family.isin(set(allowed_fams))
        allowed_classes = scope.get("allowed_mhc_classes") or []
        if allowed_classes and mhc_classes.ne("").any():
            mask = mask & (mhc_classes.eq("") | mhc_classes.isin(set(allowed_classes)))
        allowed_alleles = scope.get("allowed_alleles") or []
        if allowed_alleles:
            mask = mask & keys["mhc_restriction"].isin(set(allowed_alleles))
        # First matching scoped protocol wins; never overwrite an already matched row.
        mask = mask & protocol_id.eq("")
        if not mask.any():
            continue
        review_status.loc[mask] = "source_reviewed_study_scope"
        protocol_id.loc[mask] = str(proto["protocol_id"])
        context_scope.loc[mask] = "study_protocol_only"
        for field_name, field_val in proto["field_availability"].items():
            avail_cols[field_name].loc[mask] = str(field_val)

    result["endpoint_family"] = endpoint_family
    result["row_comment_status"] = row_comment_status
    result["source_review_status"] = review_status
    result["protocol_id"] = protocol_id
    result["context_scope"] = context_scope
    for field_name in CONTEXT_AVAILABILITY_FIELDS:
        result[field_name] = avail_cols[field_name]
    result["kd_conversion_status"] = kd_conv

    if include_method_review and "method_review_status" not in result.columns:
        result = review_binding_assay_methods(result)
    if include_censor_review and "censor_review_status" not in result.columns:
        result = review_binding_censoring(result, protocol_registry=registry)
    if include_replicate_groups:
        group_cols = [
            "source_dataset",
            "study_key",
            "peptide",
            "mhc_restriction",
            "assay_method",
            "endpoint_family",
            "measurement_unit",
        ]
        row_gids, row_sizes = _replicate_group_ids_for_keys(keys, group_cols)
        result["replicate_group_id"] = row_gids
        result["replicate_group_n_observations"] = row_sizes

    assert len(result) == len(df), "audit_binding_context_availability must preserve 1:1 row grain"
    return result


def summarize_binding_context_availability(df: pd.DataFrame) -> dict:
    """Summarize binding/stability context availability, method conflicts, censoring, and replicates (#785, #721)."""
    audited = (
        df
        if "source_review_status" in df.columns and "endpoint_family" in df.columns
        else audit_binding_context_availability(df)
    )
    keys = _extract_audit_keys(audited)
    if (
        "replicate_group_id" in audited.columns
        and "replicate_group_n_observations" in audited.columns
    ):
        gids = audited["replicate_group_id"].astype("string")
        dedup_mask = ~gids.duplicated(keep="first")
        n_rep_groups = int(dedup_mask.sum())
        n_multi_groups = int((audited.loc[dedup_mask, "replicate_group_n_observations"] > 1).sum())
    else:
        rep_groups = build_binding_replicate_groups(audited)
        n_rep_groups = len(rep_groups)
        n_multi_groups = int((rep_groups["n_observations"] > 1).sum()) if len(rep_groups) else 0

    field_status_counts = {
        field: dict(sorted(Counter(audited[field].astype("string")).items()))
        for field in CONTEXT_AVAILABILITY_FIELDS
        if field in audited.columns
    }
    summary = {
        "n_observations": len(audited),
        "n_unique_peptides": int(keys["peptide"].replace("", pd.NA).dropna().nunique()),
        "n_unique_studies": int(keys["study_key"].replace("", pd.NA).dropna().nunique()),
        "n_replicate_groups": n_rep_groups,
        "n_multi_observation_groups": n_multi_groups,
        "by_endpoint_family": dict(
            sorted(Counter(audited["endpoint_family"].astype("string")).items())
        ),
        "by_assay_method": dict(sorted(Counter(keys["assay_method"]).items())),
        "by_source_dataset": dict(sorted(Counter(keys["source_dataset"]).items())),
        "by_source_review_status": dict(
            sorted(Counter(audited["source_review_status"].astype("string")).items())
        ),
        "by_protocol_id": dict(
            sorted(Counter(val for val in audited["protocol_id"].astype("string") if val).items())
        ),
        "field_status_counts": field_status_counts,
        "native_rows_unchanged": True,
        "kd_conversion_performed": False,
    }
    if "method_review_status" in audited.columns:
        summary["by_method_review_status"] = dict(
            sorted(Counter(audited["method_review_status"].astype("string")).items())
        )
    if "censor_review_status" in audited.columns:
        summary["by_censor_review_status"] = dict(
            sorted(Counter(audited["censor_review_status"].astype("string")).items())
        )
    return summary
