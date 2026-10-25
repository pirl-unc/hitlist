"""Registered same-parent contrasts, bound to exact native source evidence."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_catalog import SCHEMA_VERSION
from .evidence_snapshot import snapshot_file
from .native_contrast_compiler import _json, compile_contrasts
from .native_contrast_schema import CONTRACT, RECIPE_ID, SCHEMAS, TABLES
from .native_peptidomics import ROOT, _space
from .provenance import file_digest


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe["schema_version"] != 1
        or recipe["registered_recipe"] != RECIPE_ID
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
    ):
        raise RecipeError("Invalid native same-parent contrast recipe")


def describe(store):
    return {
        "contract": json.loads(_json(CONTRACT)),
        "status": "available"
        if {"native_" + n for n in TABLES} <= store.bindings.keys()
        else "not_in_generation",
    }


def _inventory(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }


def materialize_contrasts(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if (
        store.manifest.get("revision") != CONTRACT["source_revision"]
        or store.manifest.get("schema_version") != CONTRACT["source_package_schema_version"]
    ):
        raise RecipeError("Native contrasts require the qualified original PXD018722 source")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _space(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        for name in TABLES:
            _space(staging)
            snapshot_file(
                store.directory / ROOT / "typed" / (name + ".parquet"),
                staging / "source_tables" / (name + ".parquet"),
            )
        summary = compile_contrasts(staging / "source_tables", staging / "typed")
        (staging / "contract.json").write_text(_json(CONTRACT) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_native_contrast",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(_json(recipe).encode()).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": summary,
            "files": _inventory(staging),
        }
        (staging / "manifest.json").write_text(_json(manifest) + "\n")
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            _json({"source": str(store.directory), "destination": str(destination)}) + "\n"
        )
        raise


def verify_contrast_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    directory = Path(directory)
    _validate(manifest["recipe"])
    inventory = (
        {"source_manifest.json", "contract.json"}
        | {"source_tables/" + n + ".parquet" for n in TABLES}
        | {"typed/" + n + ".parquet" for n in SCHEMAS}
    )
    if (
        set(manifest["files"]) != inventory
        or json.loads((directory / "contract.json").read_text()) != CONTRACT
    ):
        raise RecipeError("Native contrast contract or inventory changed")
    source_manifest = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source_manifest.get("revision") != CONTRACT["source_revision"]
        or source_manifest.get("schema_version") != CONTRACT["source_package_schema_version"]
        or manifest["source_package_schema_version"] != CONTRACT["source_package_schema_version"]
    ):
        raise RecipeError("Native contrast source binding changed")
    for name in TABLES:
        relative = str(ROOT / "typed" / (name + ".parquet"))
        digest = source_manifest["files"][relative]
        if file_digest(directory / "source_tables" / (name + ".parquet")) != digest:
            raise RecipeError("Native contrast source table changed: " + name)
        if source_package is not None and file_digest(Path(source_package) / relative) != digest:
            raise RecipeError("Native contrast original package table changed: " + name)
    _space(directory.parent)
    temporary = Path(tempfile.mkdtemp(prefix="native-contrast-verify-", dir=directory.parent))
    try:
        summary = compile_contrasts(directory / "source_tables", temporary / "typed")
        if summary != manifest["coverage"]:
            raise RecipeError("Native contrast coverage differs from source replay")
        for name, schema in SCHEMAS.items():
            actual = pq.ParquetFile(directory / "typed" / (name + ".parquet"))
            replay = pq.ParquetFile(temporary / "typed" / (name + ".parquet"))
            if not actual.schema_arrow.equals(schema):
                raise RecipeError("Native contrast output schema changed: " + name)
            for expected, observed in zip(
                replay.iter_batches(), actual.iter_batches(), strict=True
            ):
                if not expected.equals(observed):
                    raise RecipeError("Native contrast source replay differs: " + name)
    except Exception:
        (temporary / "failure-context.json").write_text(_json({"package": str(directory)}) + "\n")
        raise
    else:
        shutil.rmtree(temporary)
    return manifest
