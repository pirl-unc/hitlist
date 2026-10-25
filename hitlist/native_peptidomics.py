"""Pinned native intracellular peptide evidence through the shared query store."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .native_peptidomics_compiler import compile_native_source
from .native_peptidomics_schema import SCHEMAS
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 27
CATALOG_SCHEMA_VERSION = 35
PACKAGE_FAMILY = "native_peptidomics"
REVISION = "pxd018722_hi20s_native_v1"
ROOT = Path("native_peptidomics")
CONTRACT = Path(__file__).parent / "data/evidence_revisions/native_peptidomics.json"


def _inventory(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }


def _space(directory):
    if shutil.disk_usage(directory).free < 1024**3:
        raise OSError("Native-peptidomics source work requires 1 GiB free reserve")


def verify_native_peptidomics_package(directory):
    """Verify the source contract and replay every typed row, including exceptions."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = (PACKAGE_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION, PACKAGE_FAMILY, REVISION)
    if (
        tuple(
            manifest.get(k)
            for k in ("schema_version", "catalog_schema_version", "package_family", "revision")
        )
        != expected
    ):
        raise ValueError("Unsupported native-peptidomics package")
    if _inventory(directory) != manifest.get("files"):
        raise ValueError("Native-peptidomics inventory or hashes differ")
    contract = json.loads(CONTRACT.read_text())
    if (
        json.loads((directory / ROOT / "contract.json").read_text()) != contract
        or manifest.get("source_registry") != contract["source_registry"]
    ):
        raise ValueError("Native-peptidomics source contract differs")
    assets = directory / ROOT / "source_assets"
    for asset in contract["assets"]:
        if file_digest(assets / asset["filename"]) != {
            k: asset[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError("Native-peptidomics original source differs")
    _space(directory.parent)
    scratch = Path(tempfile.mkdtemp(prefix="hitlist-native-verify-", dir=directory.parent))
    try:
        counts = compile_native_source(assets, scratch / "typed", contract)
        if counts != manifest.get("reconciliation"):
            raise ValueError("Native-peptidomics row counts differ")
        for name, schema in SCHEMAS.items():
            actual = pq.ParquetFile(directory / ROOT / "typed" / f"{name}.parquet")
            expected_table = pq.ParquetFile(scratch / "typed" / f"{name}.parquet")
            if not actual.schema_arrow.equals(schema):
                raise ValueError(f"Native-peptidomics schema differs: {name}")
            for left, right in zip(
                expected_table.iter_batches(), actual.iter_batches(), strict=True
            ):
                if not left.equals(right):
                    raise ValueError(f"Native-peptidomics original-source replay differs: {name}")
    except Exception:
        (scratch / "failure-context.json").write_text(
            json.dumps({"package": str(directory)}) + "\n"
        )
        raise
    else:
        shutil.rmtree(scratch)
    return manifest


def write_native_peptidomics_package(source_directory, destination):
    """Snapshot qualified inputs once; never imply a supervised concentration cohort."""
    source_directory, destination = Path(source_directory), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    contract = json.loads(CONTRACT.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    _space(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for asset in contract["assets"]:
            source = source_directory / asset["filename"]
            if file_digest(source) != {k: asset[k] for k in ("sha256", "size_bytes")}:
                raise ValueError(f"Native-peptidomics original source differs: {asset['filename']}")
            _space(staging)
            snapshot_file(source, staging / ROOT / "source_assets" / asset["filename"])
        (staging / ROOT / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        counts = compile_native_source(
            staging / ROOT / "source_assets", staging / ROOT / "typed", contract
        )
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "package_family": PACKAGE_FAMILY,
            "revision": REVISION,
            "hitlist_version": __version__,
            "source_registry": contract["source_registry"],
            "reconciliation": counts,
            "files": _inventory(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_native_peptidomics_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            json.dumps({"source": str(source_directory), "destination": str(destination)}) + "\n"
        )
        raise
