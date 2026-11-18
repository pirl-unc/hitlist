"""Acquire and verify original relative perturbation quantities for common queries."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .managed_assets import cache_bytes, check_cache_path
from .perturbation_quantity_compiler import encoded, write_perturbation_tables
from .perturbation_quantity_schema import SCHEMAS
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 30
CATALOG_SCHEMA_VERSION = 40
PACKAGE_FAMILY = "perturbation_quantities"
REVISION = "stopfer2020_reporter_quantities_v1"
ROOT = Path(PACKAGE_FAMILY)
CONTRACT = Path(__file__).parent / "data/evidence_revisions/perturbation_quantities.json"


def source_contract():
    """Return a fresh copy of the admitted source mapping and protocol claims."""
    contract = json.loads(CONTRACT.read_text())
    names = [item["filename"] for item in contract["assets"]]
    if len(names) != len(set(names)) or any(
        not name or Path(name).name != name or name in {".", ".."} or "\\" in name for name in names
    ):
        raise ValueError("Perturbation asset names must be unique safe basenames")
    if contract["revision"] != REVISION:
        raise ValueError("Unsupported perturbation source revision")
    return contract


def _space(directory):
    if shutil.disk_usage(directory).free < 1024**3:
        raise OSError("Perturbation-source work requires 1 GiB free reserve")


def _inventory(directory):
    cache_bytes(directory)  # Reject symlinks and nonregular filesystem objects.
    return {
        str(path.relative_to(directory)): file_digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    }


def fetch_perturbation_sources(cache_directory=None, *, force=False, verbose=True):
    """Fetch the two checksum-pinned originals through Hitlist/datacache.

    The returned basename/path mapping can be passed as ``assets`` when building
    a package. Neither original workbook is edited or recalculated.
    """
    from .downloads import data_dir, download_to_file

    directory = (
        Path(cache_directory)
        if cache_directory is not None
        else data_dir() / "perturbation_sources"
    )
    check_cache_path(directory, directory)
    directory.mkdir(parents=True, exist_ok=True)
    _space(directory)
    result = {}
    for asset in source_contract()["assets"]:
        path = directory / asset["filename"]
        check_cache_path(directory, path)
        result[asset["filename"]] = download_to_file(
            asset["source_url"],
            path,
            force=force,
            verbose=verbose,
            expected_sha256=asset["sha256"],
            expected_size=asset["size_bytes"],
        )
    return result


def _reconcile(counts, contract):
    for name, expected in contract["expected_reconciliation"].items():
        if counts.get(name) != expected:
            raise ValueError(f"Perturbation source reconciliation differs: {name}")


def verify_perturbation_quantity_package(directory):
    """Replay every typed row from the admitted original workbooks."""
    directory = Path(directory)
    check_cache_path(directory, directory / "manifest.json")
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = (PACKAGE_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION, PACKAGE_FAMILY, REVISION)
    if (
        tuple(
            manifest.get(k)
            for k in ("schema_version", "catalog_schema_version", "package_family", "revision")
        )
        != expected
    ):
        raise ValueError("Unsupported perturbation quantity package")
    contract = source_contract()
    expected_paths = {
        str(ROOT / "contract.json"),
        *(str(ROOT / "source_assets" / a["filename"]) for a in contract["assets"]),
        *(str(ROOT / "typed" / f"{name}.parquet") for name in SCHEMAS),
    }
    inventory = _inventory(directory)
    if set(inventory) != expected_paths or inventory != manifest.get("files"):
        raise ValueError("Perturbation quantity source inventory or hashes differ")
    if (
        json.loads((directory / ROOT / "contract.json").read_text()) != contract
        or manifest.get("source_registry") != contract["source_registry"]
    ):
        raise ValueError("Perturbation source contract differs")
    source = directory / ROOT / "source_assets"
    for asset in contract["assets"]:
        if file_digest(source / asset["filename"]) != {
            k: asset[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError("Pinned perturbation original differs")
    _space(directory.parent)
    scratch = Path(tempfile.mkdtemp(prefix="perturbation-source-verify-", dir=directory.parent))
    try:
        counts = write_perturbation_tables(source, scratch / "typed", contract)
        _reconcile(counts, contract)
        if counts != manifest.get("reconciliation"):
            raise ValueError("Perturbation source census differs")
        for name, schema in SCHEMAS.items():
            actual = pq.ParquetFile(directory / ROOT / "typed" / f"{name}.parquet")
            replay = pq.ParquetFile(scratch / "typed" / f"{name}.parquet")
            if (
                not actual.schema_arrow.equals(schema)
                or actual.metadata.num_rows != replay.metadata.num_rows
            ):
                raise ValueError(f"Perturbation typed schema or row count differs: {name}")
            for expected_batch, actual_batch in zip(
                replay.iter_batches(batch_size=1024),
                actual.iter_batches(batch_size=1024),
                strict=True,
            ):
                if not expected_batch.equals(actual_batch):
                    raise ValueError(f"Perturbation original-source replay differs: {name}")
    except Exception:
        (scratch / "failure-context.json").write_text(encoded({"package": str(directory)}) + "\n")
        raise
    else:
        shutil.rmtree(scratch)
    return manifest


def write_perturbation_quantity_package(source_directory, destination, *, assets=None):
    """Create an immutable package; location overrides cannot change source bytes."""
    source_directory, destination = Path(source_directory), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    contract = source_contract()
    overrides = {} if assets is None else dict(assets)
    if set(overrides) - {a["filename"] for a in contract["assets"]}:
        raise ValueError("Unknown perturbation asset location override")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _space(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for asset in contract["assets"]:
            source = Path(overrides.get(asset["filename"], source_directory / asset["filename"]))
            if file_digest(source) != {k: asset[k] for k in ("sha256", "size_bytes")}:
                raise ValueError(f"Pinned perturbation source differs: {asset['filename']}")
            _space(staging)
            snapshot_file(source, staging / ROOT / "source_assets" / asset["filename"])
        (staging / ROOT / "contract.json").write_text(encoded(contract) + "\n")
        counts = write_perturbation_tables(
            staging / ROOT / "source_assets", staging / ROOT / "typed", contract
        )
        _reconcile(counts, contract)
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
        (staging / "manifest.json").write_text(encoded(manifest) + "\n")
        verify_perturbation_quantity_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            encoded({"source": str(source_directory), "destination": str(destination)}) + "\n"
        )
        raise
