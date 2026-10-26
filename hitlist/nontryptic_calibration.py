"""Verified nominal-dose source packages through the shared evidence store."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .nontryptic_calibration_compiler import write_nontryptic_tables
from .nontryptic_calibration_schema import SCHEMAS
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 29
CATALOG_SCHEMA_VERSION = 38
PACKAGE_FAMILY = "nontryptic_calibration"
REVISION = "pxd024917_nominal_ms2_v1"
ROOT = Path("nontryptic_calibration")
CONTRACT = Path(__file__).parent / "data/evidence_revisions/nontryptic_calibration.json"


def _contract():
    contract = json.loads(CONTRACT.read_text())
    names = [a["filename"] for a in contract["assets"]]
    if len(names) != len(set(names)) or any(
        not name or Path(name).name != name or name in {".", ".."} or "\\" in name for name in names
    ):
        raise ValueError("Nontryptic calibration asset names must be unique safe basenames")
    return contract


def _inventory(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }


def _space(directory):
    if shutil.disk_usage(directory).free < 1024**3:
        raise OSError("Nontryptic calibration-source work requires 1 GiB free reserve")


def verify_nontryptic_calibration_package(directory):
    """Replay every typed row from pinned original bytes, including ambiguity."""
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
        raise ValueError("Unsupported nontryptic calibration package")
    if _inventory(directory) != manifest.get("files"):
        raise ValueError("Nontryptic calibration source inventory or hashes differ")
    contract = _contract()
    if (
        json.loads((directory / ROOT / "contract.json").read_text()) != contract
        or manifest.get("source_registry") != contract["source_registry"]
    ):
        raise ValueError("Nontryptic calibration source contract differs")
    assets = directory / ROOT / "source_assets"
    for asset in contract["assets"]:
        if file_digest(assets / asset["filename"]) != {
            k: asset[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError("Pinned nontryptic-calibration source differs")
    _space(directory.parent)
    scratch = Path(
        tempfile.mkdtemp(prefix="hitlist-nontryptic-calibration-verify-", dir=directory.parent)
    )
    try:
        counts = write_nontryptic_tables(assets, scratch / "typed", contract)
        for key, expected_count in contract["expected_reconciliation"].items():
            if counts.get(key) != expected_count:
                raise ValueError(f"Pinned source reconciliation differs: {key}")
        if counts != manifest.get("reconciliation"):
            raise ValueError("Nontryptic calibration source row counts differ")
        for name, schema in SCHEMAS.items():
            actual = pq.ParquetFile(directory / ROOT / "typed" / f"{name}.parquet")
            reference = pq.ParquetFile(scratch / "typed" / f"{name}.parquet")
            if not actual.schema_arrow.equals(schema):
                raise ValueError(f"Nontryptic calibration typed schema differs: {name}")
            for left, right in zip(reference.iter_batches(), actual.iter_batches(), strict=True):
                if not left.equals(right):
                    raise ValueError(
                        f"Nontryptic calibration original-source replay differs: {name}"
                    )
    except Exception:
        (scratch / "failure-context.json").write_text(
            json.dumps({"package": str(directory)}) + "\n"
        )
        raise
    else:
        shutil.rmtree(scratch)
    return manifest


def write_nontryptic_calibration_package(source_directory, destination, *, assets=None):
    """Snapshot admitted originals, optionally located through a basename/path map.

    ``assets`` changes file locations only, never the pinned source contract.
    No destination is overwritten and no source is linked through a shared inode.
    """
    source_directory, destination = Path(source_directory), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    contract = _contract()
    overrides = {} if assets is None else dict(assets)
    if set(overrides) - {a["filename"] for a in contract["assets"]}:
        raise ValueError("Unknown nontryptic-calibration asset location override")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _space(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for asset in contract["assets"]:
            source = Path(overrides.get(asset["filename"], source_directory / asset["filename"]))
            if file_digest(source) != {k: asset[k] for k in ("sha256", "size_bytes")}:
                raise ValueError(
                    f"Pinned nontryptic-calibration source differs: {asset['filename']}"
                )
            _space(staging)
            snapshot_file(source, staging / ROOT / "source_assets" / asset["filename"])
        (staging / ROOT / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        counts = write_nontryptic_tables(
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
        verify_nontryptic_calibration_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            json.dumps({"source": str(source_directory), "destination": str(destination)}) + "\n"
        )
        raise
