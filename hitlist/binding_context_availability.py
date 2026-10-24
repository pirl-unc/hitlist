"""Source-availability audit for the retained MHC binding observations.

This successor is intentionally an audit relation, not a protocol imputation
layer. Native assay values remain untouched, while study-scoped review claims
are represented as study-scoped availability rather than copied onto an assay
as if they were measured per row.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .binding_assay_context import verify_binding_assay_context_package
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
