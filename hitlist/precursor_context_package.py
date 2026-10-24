"""Standalone source-contract package for the PXD009360 precursor context."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 20
CATALOG_SCHEMA_VERSION = 25
REVISION = "pxd009360_precursor_context_v1"
PACKAGE_FAMILY = "precursor_context"
SOURCE_ID = "pxd009360_precursor"
SOURCE_PREFIX = f"external/{SOURCE_ID}/"
ASSET_RECEIPT = Path(__file__).parent / "data/evidence_revisions/precursor_context_assets.json"

QUERY_ALIASES = {
    "precursor_asset": "precursor_source_asset",
    "precursor_run": "precursor_acquisition",
    "precursor_table": "precursor_table_inventory",
    "precursor_pairing": "precursor_pairing_assertion",
}
MATERIALIZED_EVIDENCE_TYPES = frozenset(
    {
        "precursor_context_family",
        "precursor_context_availability",
        "precursor_source_asset",
        "precursor_acquisition",
        "precursor_table_inventory",
        "precursor_pairing_assertion",
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
SOURCE_ASSET_SCHEMA = pa.schema(
    [
        ("source_asset_id", pa.string()),
        ("source_file", pa.string()),
        ("source_asset_role", pa.string()),
        ("asset_sha256", pa.string()),
        ("asset_size_bytes", pa.int64()),
        ("source_locator", pa.string()),
        ("source_locator_status", pa.string()),
        ("availability_state", pa.string()),
        ("notes", pa.string()),
    ]
)
ACQUISITION_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("acquisition_id", pa.string()),
        ("sample_label_raw", pa.string()),
        ("source_file", pa.string()),
        ("source_dataset", pa.string()),
        ("condition_status", pa.string()),
        ("condition_evidence", pa.string()),
        ("specimen_status", pa.string()),
        ("material_link_precision", pa.string()),
        ("source_locator", pa.string()),
        ("source_row_disposition", pa.string()),
        ("notes", pa.string()),
    ]
)
TABLE_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_file", pa.string()),
        ("table_name", pa.string()),
        ("archive_member_size_bytes", pa.int64()),
        ("source_dataset", pa.string()),
        ("availability_state", pa.string()),
        ("table_status", pa.string()),
        ("source_locator", pa.string()),
        ("source_row_disposition", pa.string()),
        ("notes", pa.string()),
    ]
)
PAIRING_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("source_dataset", pa.string()),
        ("assertion_predicate", pa.string()),
        ("assertion_value", pa.string()),
        ("assertion_status", pa.string()),
        ("pairing_scope", pa.string()),
        ("pairing_status", pa.string()),
        ("pairing_evidence", pa.string()),
        ("source_locator", pa.string()),
        ("notes", pa.string()),
    ]
)


def _receipt():
    return json.loads(ASSET_RECEIPT.read_text())


def _manifest(directory):
    path = Path(directory)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads((path / "manifest.json").read_text())


def _source_paths(source_root, receipt):
    source_root = Path(source_root)
    paths = {}
    for relative, expected in receipt["assets"].items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned PXD009360 source asset: {relative}")
        actual = file_digest(path)
        if actual != {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}:
            raise ValueError(f"PXD009360 source asset mismatch: {relative}")
        paths[relative] = path
    return paths


def _rows(receipt):
    assets = [
        {
            "source_asset_id": f"{SOURCE_ID}:asset:{relative}",
            "source_file": relative,
            "source_asset_role": expected["asset_kind"],
            "asset_sha256": expected["sha256"],
            "asset_size_bytes": expected["size_bytes"],
            "source_locator": expected["source_locator"],
            "source_locator_status": "hash_pinned",
            "availability_state": "materialized",
            "notes": "Caller-supplied source bytes copied into this immutable package.",
        }
        for relative, expected in receipt["assets"].items()
    ]
    acquisitions = [
        {
            "source_record_id": row["acquisition_id"],
            "acquisition_id": row["acquisition_id"],
            "sample_label_raw": row["sample_label_raw"],
            "source_file": row["source_file"],
            "source_dataset": receipt["dataset_accession"],
            "condition_status": "unresolved",
            "condition_evidence": "no_source_backed_condition_map",
            "specimen_status": "unresolved",
            "material_link_precision": "assay_label_only",
            "source_locator": f"{receipt['dataset_accession']}/files.json",
            "source_row_disposition": "source_metadata_only",
            "notes": "Run label does not establish condition, specimen or MAPP pairing.",
        }
        for row in receipt["acquisitions"]
    ]
    tables = [
        {
            "source_record_id": f"{SOURCE_ID}:archive_member:{row['path']}",
            "source_file": row["path"],
            "table_name": Path(row["path"]).stem,
            "archive_member_size_bytes": row["size_bytes"],
            "source_dataset": receipt["dataset_accession"],
            "availability_state": "materialized_source_archive",
            "table_status": "deferred_typed_materialization",
            "source_locator": "search/pxd009360-search.tar.gz",
            "source_row_disposition": "source_asset_only",
            "notes": "Archive member is retained; typed quantity rows are child PR 2.",
        }
        for row in receipt["archive_members"]
    ]
    pairings = [
        {
            **row,
            "source_dataset": receipt["dataset_accession"],
            "source_locator": "PXD009360/project.json;PXD009092 source contract",
            "notes": "No run order, abundance correlation or shared label is used as evidence.",
        }
        for row in receipt["pairings"]
    ]
    families = [
        {
            "family_id": "precursor_source_metadata",
            "evidence_family": "source_provenance",
            "availability_state": "materialized",
            "source_scope": "PXD009360_project_and_deposited_search_assets",
            "n_source_rows": len(assets),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_training_data",
            "evidence_types_json": json.dumps(["precursor_source_asset"], separators=(",", ":")),
            "notes": "Hash-pinned project, archive, parameter and summary assets.",
        },
        {
            "family_id": "precursor_acquisition_inventory",
            "evidence_family": "assay_context",
            "availability_state": "materialized",
            "source_scope": "PXD009360_tryptic_acquisition_labels",
            "n_source_rows": len(acquisitions),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_training_data",
            "evidence_types_json": json.dumps(["precursor_acquisition"], separators=(",", ":")),
            "notes": "Nine acquisition labels are retained without inferred biological pairing.",
        },
        {
            "family_id": "precursor_quantity_tables",
            "evidence_family": "precursor_measurement",
            "availability_state": "source_assets_available_rows_deferred",
            "source_scope": "PXD009360_maxquant_search_archive",
            "n_source_rows": 0,
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "deferred_typed_materialization",
            "evidence_types_json": json.dumps(["precursor_table_inventory"], separators=(",", ":")),
            "notes": "Evidence, peptide, all-peptide and protein-group rows await child PR 2.",
        },
        {
            "family_id": "mapp_precursor_pairing",
            "evidence_family": "lineage_context",
            "availability_state": "unresolved_in_pinned_sources",
            "source_scope": "PXD009360_to_PXD009092_cross_project_links",
            "n_source_rows": len(pairings),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "unavailable",
            "evidence_types_json": json.dumps(
                ["precursor_pairing_assertion"], separators=(",", ":")
            ),
            "notes": "Condition, specimen and aliquot links remain unresolved claims.",
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
            "count_unit": "typed_relation_rows_or_training_examples",
            "evidence_basis": "pinned_PXD009360_source_contract",
            "source_locator_status": "hash_pinned",
            "training_status": row["training_status"],
        }
        for row in families
    ]
    return families, availability, assets, acquisitions, tables, pairings


def _write_table(directory, name, rows, schema):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), directory / f"{name}.parquet")


def _verify_assets(directory, receipt):
    for relative, expected in receipt["assets"].items():
        path = directory / SOURCE_PREFIX / "assets" / relative
        actual = file_digest(path)
        if actual != {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}:
            raise ValueError(f"PXD009360 package asset mismatch: {relative}")


def verify_precursor_context_package(directory):
    directory = Path(directory)
    manifest = _manifest(directory)
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("package_family") != PACKAGE_FAMILY
        or manifest.get("revision") != REVISION
    ):
        raise ValueError("Unsupported precursor context package envelope")
    receipt_path = directory / SOURCE_PREFIX / "contracts/source_assets.json"
    if file_digest(receipt_path) != file_digest(ASSET_RECEIPT):
        raise ValueError("PXD009360 source receipt differs from the pinned contract")
    receipt = json.loads(receipt_path.read_text())
    _verify_assets(directory, receipt)
    if manifest.get("query_aliases") != QUERY_ALIASES:
        raise ValueError("Precursor context query aliases changed")
    families, availability, assets, acquisitions, tables, pairings = _rows(receipt)
    expected = {
        "families": (FAMILY_SCHEMA, families),
        "availability": (AVAILABILITY_SCHEMA, availability),
        "source_assets": (SOURCE_ASSET_SCHEMA, assets),
        "acquisitions": (ACQUISITION_SCHEMA, acquisitions),
        "tables": (TABLE_SCHEMA, tables),
        "pairings": (PAIRING_SCHEMA, pairings),
    }
    for name, (schema, rows) in expected.items():
        table = pq.read_table(directory / "precursor_context" / f"{name}.parquet")
        if table.schema != schema or table.to_pylist() != rows:
            raise ValueError(f"Precursor context relation changed: {name}")
    for name, digest in manifest["files"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Precursor context package artifact mismatch: {name}")
    return manifest


def write_precursor_context_package(source_root, destination):
    """Write a standalone source-contract package from verified PXD009360 bytes."""
    source_root, destination = Path(source_root), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    receipt = _receipt()
    paths = _source_paths(source_root, receipt)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        source_dir = staging / SOURCE_PREFIX
        (source_dir / "assets").mkdir(parents=True)
        (source_dir / "contracts").mkdir()
        snapshot_file(ASSET_RECEIPT, source_dir / "contracts/source_assets.json")
        for relative, path in paths.items():
            snapshot_file(path, source_dir / "assets" / relative)
        families, availability, assets, acquisitions, tables, pairings = _rows(receipt)
        typed = staging / "precursor_context"
        typed.mkdir()
        _write_table(typed, "families", families, FAMILY_SCHEMA)
        _write_table(typed, "availability", availability, AVAILABILITY_SCHEMA)
        _write_table(typed, "source_assets", assets, SOURCE_ASSET_SCHEMA)
        _write_table(typed, "acquisitions", acquisitions, ACQUISITION_SCHEMA)
        _write_table(typed, "tables", tables, TABLE_SCHEMA)
        _write_table(typed, "pairings", pairings, PAIRING_SCHEMA)
        from .version import __version__

        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        registry = {
            "source_id": SOURCE_ID,
            "dataset_accession": receipt["dataset_accession"],
            "n_assets": len(assets),
            "n_archive_members": len(receipt["archive_members"]),
            "n_acquisitions": len(acquisitions),
            "condition_map_status": "unresolved",
            "specimen_pairing_status": "unresolved",
            "quantity_materialization_status": "deferred_child_pr_802",
        }
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "package_family": PACKAGE_FAMILY,
            "query_aliases": QUERY_ALIASES,
            "materialized_evidence_types": sorted(MATERIALIZED_EVIDENCE_TYPES),
            "appended_sources": {SOURCE_ID: {}},
            "source_registry": {SOURCE_ID: registry},
            "reconciliation": {
                "n_source_assets": len(assets),
                "n_archive_members": len(receipt["archive_members"]),
                "n_acquisitions": len(acquisitions),
                "n_pairing_assertions": len(pairings),
                "n_precursor_measurements": 0,
                "n_training_examples": 0,
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_precursor_context_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
