"""Append reviewed external source generations to one portable evidence package.

Source adapters own source-specific parsing. This module owns immutable assets,
the parent generation, atomic publication, and one common manifest contract.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .mhc_evidence import _link_or_copy
from .provenance import file_digest
from .supporting_evidence import verify_supporting_evidence_package

PACKAGE_SCHEMA_VERSION = 4
CATALOG_SCHEMA_VERSION = 5


def _adapters():
    from .pxd057588_evidence import PXD057588Adapter

    return {PXD057588Adapter.source_id: PXD057588Adapter()}


def _safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid external source path: {name}")
    return path


def _source_paths(adapter, source_root):
    receipt_path = adapter.asset_receipt_path
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt.get("assets"), dict) or not receipt["assets"]:
        raise ValueError(f"Empty external source receipt: {adapter.source_id}")
    source_root = Path(source_root)
    paths = {}
    for relative, expected in receipt["assets"].items():
        path = source_root / _safe_relative(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned external source asset: {relative}")
        if file_digest(path) != {
            "sha256": expected["sha256"],
            "size_bytes": expected["size"],
        }:
            raise ValueError(f"External source asset mismatch: {relative}")
        paths[relative] = path
    return receipt, paths


def _copy_source(adapter, source_root, destination):
    receipt, paths = _source_paths(adapter, source_root)
    asset_dir = destination / "assets"
    contract_dir = destination / "contracts"
    asset_dir.mkdir(parents=True)
    contract_dir.mkdir()
    shutil.copy2(adapter.asset_receipt_path, contract_dir / "source_assets.json")
    for name, path in adapter.support_files.items():
        shutil.copy2(path, contract_dir / _safe_relative(name))
    for relative, path in paths.items():
        target = asset_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return receipt


def _verify_source(adapter, destination):
    receipt_path = destination / "contracts" / "source_assets.json"
    if file_digest(receipt_path) != file_digest(adapter.asset_receipt_path):
        raise ValueError(f"External source receipt differs from adapter: {adapter.source_id}")
    for name, path in adapter.support_files.items():
        if file_digest(destination / "contracts" / _safe_relative(name)) != file_digest(path):
            raise ValueError(f"External source contract mismatch: {name}")
    _source_paths(adapter, destination / "assets")


def verify_external_evidence_package(directory):
    """Verify parent, assets, source relations and exact package inventory."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported external evidence package schema")
    if manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported external evidence catalog schema")
    files = manifest.get("files", {})
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(files) != actual:
        raise ValueError("External package file inventory mismatch")
    for relative, expected in files.items():
        path = _safe_relative(relative)
        if file_digest(directory / path) != expected:
            raise ValueError(f"External package artifact mismatch: {relative}")
    parent_path = "manifests/supporting.json"
    if files[parent_path]["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("External parent manifest mismatch")
    parent = verify_supporting_evidence_package(directory, manifest_path=parent_path)
    if parent["catalog_schema_version"] != 4:
        raise ValueError("Unsupported parent supporting catalog")
    adapters = _adapters()
    source_ids = manifest.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids or source_ids != sorted(set(source_ids)):
        raise ValueError("Invalid external source IDs")
    if any(source_id not in adapters for source_id in source_ids):
        raise ValueError("Unregistered external source adapter")
    if set(manifest.get("sources", {})) != set(source_ids):
        raise ValueError("External source registry mismatch")
    for source_id in source_ids:
        adapter = adapters[source_id]
        source_dir = directory / "external" / source_id
        _verify_source(adapter, source_dir)
        source = manifest["sources"][source_id]
        if source.get("adapter_schema_version") != adapter.schema_version:
            raise ValueError(f"External adapter version mismatch: {source_id}")
        if (
            source.get("asset_receipt_sha256")
            != files[f"external/{source_id}/contracts/source_assets.json"]["sha256"]
        ):
            raise ValueError(f"External source receipt hash mismatch: {source_id}")
        adapter.verify(source_dir, source["reconciliation"])
        if source.get("registry") != adapter.registry(source["reconciliation"]):
            raise ValueError(f"External source metadata mismatch: {source_id}")
    return manifest


def write_external_evidence_package(
    supporting_package,
    destination,
    *,
    source_roots,
    source_ids=("pxd057588",),
):
    """Atomically append reviewed sources to the preceding supporting generation.

    `source_roots` supplies original ingestion bytes by source ID. The written
    generation contains those bytes, so later readers need only `destination`.
    """
    supporting_package = Path(supporting_package)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"External evidence package already exists: {destination}")
    source_ids = tuple(sorted(set(source_ids)))
    adapters = _adapters()
    if not source_ids or any(source_id not in adapters for source_id in source_ids):
        raise ValueError("Unregistered external source adapter")
    if set(source_roots) != set(source_ids):
        raise ValueError("External source roots must match requested source IDs")
    parent = verify_supporting_evidence_package(supporting_package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent["files"]:
            target = staging / _safe_relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(supporting_package / relative, target)
        (staging / "manifests").mkdir(exist_ok=True)
        _link_or_copy(
            supporting_package / "manifest.json", staging / "manifests" / "supporting.json"
        )
        sources = {}
        for source_id in source_ids:
            adapter = adapters[source_id]
            source_dir = staging / "external" / source_id
            _copy_source(adapter, source_roots[source_id], source_dir)
            reconciliation = adapter.write(source_dir)
            _verify_source(adapter, source_dir)
            adapter.verify(source_dir, reconciliation)
            sources[source_id] = {
                "adapter_schema_version": adapter.schema_version,
                "reconciliation": reconciliation,
                "registry": adapter.registry(reconciliation),
            }
        verify_supporting_evidence_package(supporting_package)
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        for source_id in source_ids:
            sources[source_id]["asset_receipt_sha256"] = files[
                f"external/{source_id}/contracts/source_assets.json"
            ]["sha256"]
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_manifest_sha256": files["manifests/supporting.json"]["sha256"],
            "source_ids": list(source_ids),
            "sources": sources,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_external_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
