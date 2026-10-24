"""Standalone typed T-cell and controlled-processing evidence package.

This package repackages the source-faithful T-cell v2 generation and the
controlled PXD009092 MAPP generation under one query envelope. It copies only
their source-scoped typed relations and source assets, not the unrelated
predecessor native corpus. Scientific families remain separate in the family
registry and availability ledger.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 17
CATALOG_SCHEMA_VERSION = 22
REVISION = "tcell_processing_evidence_v1"
PACKAGE_FAMILY = "tcell_processing"

_TCELL_PREFIX = "tcell/"
_PROCESSING_PREFIX = "external/pxd009092/"
_TCELL_RELATIONS = {
    "tcell_availability",
    "tcell_functional_peptide",
    "tcell_multimer_peptide",
    "tcell_reported_outcome",
    "tcell_antigen_claim",
    "tcell_source_contributor",
    "tcell_cedar_mutation_claim",
    "tcell_eligibility",
}
_PROCESSING_RELATIONS = {
    "mapp_source_record",
    "mapp_peptide",
    "mapp_group",
    "mapp_channel",
    "mapp_identification",
    "mapp_quantity_measurement",
    "mapp_derivation_input",
    "mapp_source_assertion",
    "mapp_publisher_cell",
    "mapp_publisher_quantity",
    "external_proteasome_fragment_quantity",
}
MATERIALIZED_EVIDENCE_TYPES = frozenset(_TCELL_RELATIONS | _PROCESSING_RELATIONS)
QUERY_ALIASES = {
    "tcell_functional_response": "tcell_functional_peptide",
    "pmhc_multimer_binding": "tcell_multimer_peptide",
    "processing_measurement": "external_proteasome_fragment_quantity",
    "mapp_controlled_product": "mapp_quantity_measurement",
}

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


def _manifest(path):
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Predecessor manifest has no file inventory")
    return manifest


def _selected_files(manifest, prefix):
    return sorted(relative for relative in manifest["files"] if relative.startswith(prefix))


def _copy_selected(source, destination, manifest, prefix):
    selected = _selected_files(manifest, prefix)
    if not selected:
        raise ValueError(f"Predecessor manifest has no files under {prefix}")
    for relative in selected:
        source_path = Path(source) / relative
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source_path, target)
    return selected


def _verify_selected(directory, predecessor, prefix):
    selected = _selected_files(predecessor, prefix)
    for relative in selected:
        expected = predecessor["files"][relative]
        actual = file_digest(Path(directory) / relative)
        if actual != expected:
            raise ValueError(f"Predecessor artifact changed: {relative}")
    return selected


def _rows(tcell_manifest, processing_manifest):
    tcell = tcell_manifest["reconciliation"]
    processing = processing_manifest["sources"]["pxd009092"]["reconciliation"]
    families = [
        {
            "family_id": "tcell_functional_response",
            "evidence_family": "tcell_functional_response",
            "availability_state": "materialized",
            "source_scope": "IEDB direct human peptide functional assays; CEDAR contributor evidence",
            "n_source_rows": tcell["n_functional_examples"],
            "n_measurements": tcell["n_functional_examples"],
            "n_training_examples": tcell["n_functional_examples"],
            "training_status": "bounded_source_projection",
            "evidence_types_json": json.dumps(["tcell_functional_peptide"]),
            "notes": "Functional response; multimer binding and other assay families excluded.",
        },
        {
            "family_id": "pmhc_multimer_binding",
            "evidence_family": "pmhc_multimer_binding",
            "availability_state": "materialized",
            "source_scope": "IEDB direct human peptide pMHC multimer assays",
            "n_source_rows": tcell["n_multimer_examples"],
            "n_measurements": tcell["n_multimer_examples"],
            "n_training_examples": tcell["n_multimer_examples"],
            "training_status": "bounded_source_projection",
            "evidence_types_json": json.dumps(["tcell_multimer_peptide"]),
            "notes": "Cell binding is not functional activation or MHC affinity.",
        },
        {
            "family_id": "receptor_evidence",
            "evidence_family": "receptor_evidence",
            "availability_state": "not_materialized_in_pinned_sources",
            "source_scope": "IEDB/CEDAR T-cell export",
            "n_source_rows": 0,
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_available",
            "evidence_types_json": "[]",
            "notes": "A TCR name is not a paired receptor sequence or repertoire measurement.",
        },
        {
            "family_id": "direct_processing_measurement",
            "evidence_family": "direct_processing_measurement",
            "availability_state": "not_observed_in_tcell_export",
            "source_scope": "IEDB/CEDAR T-cell export method/readout fields",
            "n_source_rows": 0,
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_available",
            "evidence_types_json": "[]",
            "notes": "No direct cleavage, transport or proteasome endpoint was routed from this export.",
        },
        {
            "family_id": "mapp_controlled_product",
            "evidence_family": "processing_measurement",
            "availability_state": "materialized_source_evidence",
            "source_scope": "PXD009092 cellular proteasome-associated captured products",
            "n_source_rows": processing["n_quantity_results_rows"],
            "n_measurements": processing["n_quantity_measurements_rows"],
            "n_training_examples": 0,
            "training_status": "deferred_endpoint_recipe",
            "evidence_types_json": json.dumps(
                ["external_proteasome_fragment_quantity", "mapp_quantity_measurement"]
            ),
            "notes": "Captured product is not free intracellular concentration or a direct cleavage rate.",
        },
    ]
    availability = [
        {
            "family_id": row["family_id"],
            "source_id": "tcell_2026_10_20"
            if row["family_id"] != "mapp_controlled_product"
            else "pxd009092",
            "availability_state": row["availability_state"],
            "n_source_rows": row["n_source_rows"],
            "n_measurements": row["n_measurements"],
            "n_training_examples": row["n_training_examples"],
            "count_unit": "source_rows_and_typed_measurements",
            "evidence_basis": row["notes"],
            "source_locator_status": "source_assets_and_typed_relations_pinned",
            "training_status": row["training_status"],
        }
        for row in families
    ]
    return families, availability


def _write_ledgers(staging, tcell_manifest, processing_manifest):
    families, availability = _rows(tcell_manifest, processing_manifest)
    root = Path(staging) / "tcell_processing"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(families, schema=FAMILY_SCHEMA), root / "families.parquet")
    pq.write_table(
        pa.Table.from_pylist(availability, schema=AVAILABILITY_SCHEMA),
        root / "availability.parquet",
    )
    (root / "families.json").write_text(json.dumps(families, indent=2, sort_keys=True) + "\n")
    (root / "availability.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True) + "\n"
    )
    return families, availability


def _manifest_files(directory):
    return {
        str(path.relative_to(directory)): file_digest(path)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def verify_tcell_processing_package(directory):
    directory = Path(directory)
    manifest = _manifest(directory)
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("revision") != REVISION
        or manifest.get("package_family") != PACKAGE_FAMILY
    ):
        raise ValueError("Unsupported T-cell and processing package")
    files = manifest["files"]
    actual = _manifest_files(directory)
    if files != actual:
        raise ValueError("T-cell and processing package inventory mismatch")
    for relative, expected in files.items():
        if file_digest(directory / relative) != expected:
            raise ValueError(f"Changed package artifact: {relative}")
    tcell_manifest = json.loads((directory / "manifests/tcell.json").read_text())
    processing_manifest = json.loads((directory / "manifests/processing.json").read_text())
    if tcell_manifest.get("schema_version") != 5:
        raise ValueError("Unsupported T-cell predecessor")
    if processing_manifest.get("schema_version") != 4:
        raise ValueError("Unsupported processing predecessor")
    if (
        manifest["predecessor_manifest_sha256"]["tcell"]
        != file_digest(directory / "manifests/tcell.json")["sha256"]
    ):
        raise ValueError("T-cell predecessor manifest hash mismatch")
    if (
        manifest["predecessor_manifest_sha256"]["processing"]
        != file_digest(directory / "manifests/processing.json")["sha256"]
    ):
        raise ValueError("Processing predecessor manifest hash mismatch")
    _verify_selected(directory, tcell_manifest, _TCELL_PREFIX)
    _verify_selected(directory, processing_manifest, _PROCESSING_PREFIX)
    if manifest.get("query_aliases") != QUERY_ALIASES:
        raise ValueError("T-cell and processing query aliases changed")
    families, availability = _rows(tcell_manifest, processing_manifest)
    family_table = pq.read_table(directory / "tcell_processing/families.parquet")
    availability_table = pq.read_table(directory / "tcell_processing/availability.parquet")
    if family_table.schema != FAMILY_SCHEMA or availability_table.schema != AVAILABILITY_SCHEMA:
        raise ValueError("T-cell and processing ledger schema changed")
    if family_table.to_pylist() != families or availability_table.to_pylist() != availability:
        raise ValueError("T-cell and processing ledger changed")
    return manifest


def write_tcell_processing_package(tcell_package, processing_package, destination):
    """Build one typed package from verified T-cell and processing generations."""
    tcell_package = Path(tcell_package)
    processing_package = Path(processing_package)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tcell_manifest = _manifest(tcell_package)
    processing_manifest = _manifest(processing_package)
    if tcell_manifest.get("schema_version") != 5:
        raise ValueError("Expected schema-5 T-cell package")
    if processing_manifest.get("schema_version") != 4:
        raise ValueError("Expected schema-4 processing package")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (staging / "manifests").mkdir(parents=True)
        snapshot_file(tcell_package / "manifest.json", staging / "manifests/tcell.json")
        snapshot_file(processing_package / "manifest.json", staging / "manifests/processing.json")
        _copy_selected(tcell_package, staging, tcell_manifest, _TCELL_PREFIX)
        _copy_selected(processing_package, staging, processing_manifest, _PROCESSING_PREFIX)
        _write_ledgers(staging, tcell_manifest, processing_manifest)
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "package_family": PACKAGE_FAMILY,
            "predecessor_manifest_sha256": {
                "tcell": file_digest(staging / "manifests/tcell.json")["sha256"],
                "processing": file_digest(staging / "manifests/processing.json")["sha256"],
            },
            "query_aliases": QUERY_ALIASES,
            "materialized_evidence_types": sorted(MATERIALIZED_EVIDENCE_TYPES),
            "source_registry": {
                "tcell_2026_10_20": tcell_manifest["source_registry"],
                "pxd009092": processing_manifest["sources"]["pxd009092"]["registry"],
            },
            "appended_sources": {
                "pxd009092": processing_manifest["sources"]["pxd009092"]["registry"]
            },
            "reconciliation": {
                "tcell": tcell_manifest["reconciliation"],
                "processing": processing_manifest["sources"]["pxd009092"]["reconciliation"],
            },
            "files": _manifest_files(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_tcell_processing_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
