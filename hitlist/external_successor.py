"""Append registered sources without losing the current evidence generation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .evidence_snapshot import snapshot_file
from .external_evidence import _adapters, _copy_source, _safe_relative, _verify_source
from .provenance import file_digest
from .reference_evidence import verify_reference_evidence_package

PACKAGE_SCHEMA_VERSION = 8
CATALOG_SCHEMA_VERSION = 11
PARENT_MANIFEST = "manifests/reference.json"


def verify_external_successor(directory):
    """Verify the full inventory, embedded predecessor and appended sources."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION or manifest.get(
        "catalog_schema_version"
    ) not in (10, CATALOG_SCHEMA_VERSION):
        raise ValueError("Unsupported external successor schema")
    files = manifest.get("files", {})
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(files) != actual:
        raise ValueError("External successor inventory mismatch")
    for name, digest in files.items():
        if file_digest(directory / _safe_relative(name)) != digest:
            raise ValueError(f"External successor artifact mismatch: {name}")
    if files.get(PARENT_MANIFEST, {}).get("sha256") != manifest.get("parent_manifest_sha256"):
        raise ValueError("External successor predecessor mismatch")
    parent = verify_reference_evidence_package(directory, manifest_path=PARENT_MANIFEST)
    if any(files.get(name) != digest for name, digest in parent["files"].items()):
        raise ValueError("External successor changed predecessor bytes")
    if manifest.get("reference") != parent["reference"]:
        raise ValueError("External successor changed predecessor reference metadata")
    adapters = _adapters()
    sources = manifest.get("appended_sources", {})
    if not sources or any(name not in adapters for name in sources):
        raise ValueError("Unregistered appended source")
    expected = set(parent["files"]) | {PARENT_MANIFEST}
    for name, record in sources.items():
        prefix = f"external/{name}/"
        if any(path.startswith(prefix) for path in parent["files"]):
            raise ValueError(f"Appended source replaces predecessor: {name}")
        adapter = adapters[name]
        source = directory / "external" / name
        _verify_source(adapter, source)
        if record.get("adapter_schema_version") != adapter.schema_version:
            raise ValueError(f"Appended adapter version mismatch: {name}")
        if (
            record.get("asset_receipt_sha256")
            != file_digest(source / "contracts/source_assets.json")["sha256"]
        ):
            raise ValueError(f"Appended source receipt mismatch: {name}")
        adapter.verify(source, record["reconciliation"])
        if record.get("registry") != adapter.registry(record["reconciliation"]):
            raise ValueError(f"Appended source metadata mismatch: {name}")
        expected.update(path for path in files if path.startswith(prefix))
    if set(files) != expected:
        raise ValueError("External successor contains unowned artifacts")
    return manifest


def write_external_successor(reference_package, destination, *, source_roots):
    """Clone the schema-7 package and append only previously absent sources."""
    reference_package, destination = Path(reference_package), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    adapters = _adapters()
    if not source_roots or any(name not in adapters for name in source_roots):
        raise ValueError("Unregistered appended source")
    parent = verify_reference_evidence_package(reference_package)
    for name in source_roots:
        if (reference_package / "external" / name).exists():
            raise ValueError(f"Appended source replaces predecessor: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in parent["files"]:
            snapshot_file(reference_package / _safe_relative(name), staging / name)
        snapshot_file(reference_package / "manifest.json", staging / PARENT_MANIFEST)
        sources = {}
        for name, root in sorted(source_roots.items()):
            adapter = adapters[name]
            source = staging / "external" / name
            _copy_source(adapter, root, source)
            reconciliation = adapter.write(source)
            sources[name] = {
                "adapter_schema_version": adapter.schema_version,
                "asset_receipt_sha256": file_digest(source / "contracts/source_assets.json")[
                    "sha256"
                ],
                "reconciliation": reconciliation,
                "registry": adapter.registry(reconciliation),
            }
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
            "parent_manifest_sha256": files[PARENT_MANIFEST]["sha256"],
            "reference": parent["reference"],
            "appended_sources": sources,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_external_successor(staging)
        # A producer must not silently consume a predecessor changed during its run.
        verify_reference_evidence_package(reference_package)
        if (
            file_digest(reference_package / "manifest.json")["sha256"]
            != manifest["parent_manifest_sha256"]
        ):
            raise ValueError("Predecessor changed during source generation")
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
