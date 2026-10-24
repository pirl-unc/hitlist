"""Standalone typed package for the Soh et al. purified-proteasome source."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 19
CATALOG_SCHEMA_VERSION = 24
REVISION = "soh2024_digestion_processing_v1"
PACKAGE_FAMILY = "digestion_processing"
SOURCE_ID = "soh2024_digestion"
SOURCE_PREFIX = f"external/{SOURCE_ID}/"

QUERY_ALIASES = {
    "processing_ms1": "digestion_ms1",
    "processing_raw_quantity": "digestion_raw_quantities",
    "processing_timecourse": "digestion_final_values",
    "processing_run": "digestion_runs",
    "processing_context": "digestion_contexts",
    "processing_conflict": "digestion_conflicts",
}
MATERIALIZED_EVIDENCE_TYPES = frozenset(
    {
        "digestion_members",
        "digestion_metadata",
        "digestion_constructs",
        "digestion_runs",
        "digestion_contexts",
        "digestion_identifications",
        "digestion_ms1",
        "digestion_raw_quantities",
        "digestion_filtered_vectors",
        "digestion_final_vectors",
        "digestion_final_values",
        "digestion_conflicts",
        "digestion_processing_family",
        "digestion_processing_availability",
    }
)

FAMILY_SCHEMA = pa.schema(
    [
        ("family_id", pa.string()),
        ("evidence_family", pa.string()),
        ("availability_state", pa.string()),
        ("source_scope", pa.string()),
        ("n_source_rows", pa.int64()),
        ("n_measurements", pa.int64()),
        ("n_training_examples", pa.int64()),
        ("training_status", pa.string()),
        ("evidence_types_json", pa.string()),
        ("notes", pa.string()),
    ]
)
AVAILABILITY_SCHEMA = pa.schema(
    [
        ("family_id", pa.string()),
        ("source_id", pa.string()),
        ("availability_state", pa.string()),
        ("n_source_rows", pa.int64()),
        ("n_measurements", pa.int64()),
        ("n_training_examples", pa.int64()),
        ("count_unit", pa.string()),
        ("evidence_basis", pa.string()),
        ("source_locator_status", pa.string()),
        ("training_status", pa.string()),
    ]
)


def _manifest(directory):
    path = Path(directory)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads((path / "manifest.json").read_text())


def _source_files(manifest):
    return sorted(name for name in manifest.get("files", {}) if name.startswith(SOURCE_PREFIX))


def _source_counts(manifest):
    record = manifest["appended_sources"][SOURCE_ID]
    return record["reconciliation"]["relation_rows"], record["registry"]


def _rows(manifest):
    relation_rows, registry = _source_counts(manifest)
    families = [
        {
            "family_id": "processing_ms1_identification",
            "evidence_family": "processing_measurement",
            "availability_state": "materialized",
            "source_scope": "purified_standard_proteasome_MS1_search",
            "n_source_rows": relation_rows["ms1"] + relation_rows["identifications"],
            "n_measurements": relation_rows["ms1"],
            "n_training_examples": 0,
            "training_status": "deferred_endpoint_recipe",
            "evidence_types_json": json.dumps(
                ["digestion_ms1", "digestion_identifications"], separators=(",", ":")
            ),
            "notes": "Relative MS response; not molar production or APC concentration.",
        },
        {
            "family_id": "processing_raw_quantities",
            "evidence_family": "processing_measurement",
            "availability_state": "materialized",
            "source_scope": "source_background_and_replicate_layers",
            "n_source_rows": relation_rows["raw_quantities"],
            "n_measurements": relation_rows["raw_quantities"],
            "n_training_examples": 0,
            "training_status": "deferred_endpoint_recipe",
            "evidence_types_json": json.dumps(["digestion_raw_quantities"], separators=(",", ":")),
            "notes": "Original values, controls, time claims and missingness are retained.",
        },
        {
            "family_id": "processing_timecourse_vectors",
            "evidence_family": "processing_measurement",
            "availability_state": "materialized",
            "source_scope": "source_filtered_and_final_vector_layers",
            "n_source_rows": relation_rows["final_vectors"],
            "n_measurements": relation_rows["final_values"],
            "n_training_examples": 0,
            "training_status": "deferred_contiguous_product_recipe",
            "evidence_types_json": json.dumps(
                ["digestion_filtered_vectors", "digestion_final_vectors", "digestion_final_values"],
                separators=(",", ":"),
            ),
            "notes": "Final values are source-selected slots, not a complete negative set.",
        },
        {
            "family_id": "processing_conflict_audit",
            "evidence_family": "source_provenance",
            "availability_state": "materialized",
            "source_scope": "source_claim_conflicts",
            "n_source_rows": relation_rows["conflicts"],
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_training_data",
            "evidence_types_json": json.dumps(["digestion_conflicts"], separators=(",", ":")),
            "notes": "RF1 time and metadata conflicts remain explicit and unresolved.",
        },
        {
            "family_id": "cellular_apc_processing",
            "evidence_family": "processing_measurement",
            "availability_state": "not_materialized_in_pinned_sources",
            "source_scope": "APC_TAP_clearance_MHC",
            "n_source_rows": 0,
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "unavailable",
            "evidence_types_json": "[]",
            "notes": "Purified enzyme digestion does not measure cellular trafficking or MHC selection.",
        },
    ]
    availability = [
        {
            "family_id": row["family_id"],
            "source_id": SOURCE_ID,
            "availability_state": row["availability_state"],
            "n_source_rows": row["n_source_rows"],
            "n_measurements": row["n_measurements"],
            "n_training_examples": row["n_training_examples"],
            "count_unit": "assay_rows_or_typed_relation_rows",
            "evidence_basis": "pinned_schema9_appended_source_manifest",
            "source_locator_status": "hash_pinned",
            "training_status": row["training_status"],
        }
        for row in families
    ]
    return families, availability, registry


def _verify_source_files(directory, predecessor_manifest):
    for name in _source_files(predecessor_manifest):
        path = Path(directory) / name
        if not path.is_file() or file_digest(path) != predecessor_manifest["files"][name]:
            raise ValueError(f"Digestion source artifact mismatch: {name}")


def verify_digestion_processing_package(directory):
    directory = Path(directory)
    manifest = _manifest(directory)
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("package_family") != PACKAGE_FAMILY
        or manifest.get("revision") != REVISION
    ):
        raise ValueError("Unsupported digestion processing package envelope")
    predecessor = _manifest(directory / "manifests/predecessor.json")
    if (
        manifest.get("predecessor_manifest_sha256")
        != file_digest(directory / "manifests/predecessor.json")["sha256"]
    ):
        raise ValueError("Digestion predecessor manifest hash mismatch")
    _verify_source_files(directory, predecessor)
    if manifest.get("query_aliases") != QUERY_ALIASES:
        raise ValueError("Digestion query aliases changed")
    families, availability, _ = _rows(predecessor)
    family_table = pq.read_table(directory / "digestion_processing/families.parquet")
    availability_table = pq.read_table(directory / "digestion_processing/availability.parquet")
    if family_table.schema != FAMILY_SCHEMA or availability_table.schema != AVAILABILITY_SCHEMA:
        raise ValueError("Digestion ledger schema changed")
    if family_table.to_pylist() != families or availability_table.to_pylist() != availability:
        raise ValueError("Digestion ledger changed")
    for name, digest in manifest["files"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Digestion package artifact mismatch: {name}")
    return manifest


def write_digestion_processing_package(predecessor_package, destination):
    """Build a standalone processing package from a verified external generation."""
    predecessor_package, destination = Path(predecessor_package), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    predecessor = _manifest(predecessor_package)
    if predecessor.get("schema_version") not in (9, 10):
        raise ValueError("Expected a schema-9/10 external successor predecessor")
    if SOURCE_ID not in predecessor.get("appended_sources", {}):
        raise ValueError("Predecessor does not contain the Soh digestion source")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (staging / "manifests").mkdir(parents=True)
        snapshot_file(predecessor_package / "manifest.json", staging / "manifests/predecessor.json")
        for name in _source_files(predecessor):
            snapshot_file(predecessor_package / name, staging / name)
        families, availability, registry = _rows(predecessor)
        (staging / "digestion_processing").mkdir()
        pq.write_table(
            pa.Table.from_pylist(families, schema=FAMILY_SCHEMA),
            staging / "digestion_processing/families.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(availability, schema=AVAILABILITY_SCHEMA),
            staging / "digestion_processing/availability.parquet",
        )
        from .version import __version__

        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "package_family": PACKAGE_FAMILY,
            "predecessor_manifest_sha256": files["manifests/predecessor.json"]["sha256"],
            "query_aliases": QUERY_ALIASES,
            "materialized_evidence_types": sorted(MATERIALIZED_EVIDENCE_TYPES),
            "appended_sources": {SOURCE_ID: {}},
            "source_registry": {SOURCE_ID: registry},
            "reconciliation": predecessor["appended_sources"][SOURCE_ID]["reconciliation"],
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_digestion_processing_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
