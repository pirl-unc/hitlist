"""Real released-format consumer fixtures; no remote acquisition in unit tests."""

import copy
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from orfdata import DatasetManifest, DatasetRegistry
from orfdata.sources.pfizer import import_protein

from hitlist import canine_bulk
from hitlist.provenance import file_digest

FIXTURE = Path(__file__).parent / "data/canine_orfdata"


def fingerprint(manifest, download):
    return hashlib.sha256(
        json.dumps(
            {"manifest": manifest, "download": download}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    manifest = DatasetManifest.from_dict(json.loads((FIXTURE / "manifest.json").read_text()))
    local = {s.asset_id: FIXTURE / s.asset_id for s in manifest.sources}
    for asset in manifest.sources:
        assert file_digest(local[asset.asset_id]) == {
            "sha256": asset.sha256,
            "size_bytes": asset.size,
        }
    dataset = import_protein(local, tmp_path / "dataset.sqlite", manifest)
    compressed = tmp_path / "dataset.sqlite.gz"
    compressed.write_bytes(gzip.compress(dataset.path.read_bytes(), mtime=0))
    digest = file_digest(dataset.path)
    download = {
        "distribution": {
            "access_terms": "Fixture only",
            "redistributable": True,
            "download_size": compressed.stat().st_size,
            "download_sha256": file_digest(compressed)["sha256"],
        },
        "assets": {
            "dataset.sqlite": {
                "url": "https://example.org/dataset.sqlite.gz",
                "sha256": digest["sha256"],
                "size": digest["size_bytes"],
                "decompress": True,
            }
        },
    }
    definitions = {
        manifest.dataset_id: {
            "default_version": manifest.version,
            "versions": {manifest.version: {"manifest": manifest.to_dict(), **download}},
        }
    }
    registry = DatasetRegistry(definitions, cache_root=tmp_path / "cache")
    registry.download(
        manifest.dataset_id, manifest.version, source_paths={"dataset.sqlite": compressed}
    )
    registry.download_reference(
        manifest.dataset_id, manifest.version, source_path=local["search.fasta"]
    )
    monkeypatch.setattr("orfdata.DatasetRegistry", lambda **kwargs: registry)
    catalog = copy.deepcopy(canine_bulk.canine_bulk_catalog())
    source = catalog["sources"]["pfizer_2026_03"]
    source["orfdata"] = {
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "registry_sha256": fingerprint(manifest.to_dict(), download),
    }
    assets = {"table": local["protein.tsv"], "reference": local["search.fasta"]}
    for name, path in assets.items():
        source["assets"][name].update(file_digest(path))
    monkeypatch.setattr(canine_bulk, "canine_bulk_catalog", lambda: catalog)
    return registry, assets, source


def test_orfdata_replays_all_native_cells_and_historical_views(upstream, tmp_path, monkeypatch):
    registry, assets, source = upstream

    def no_download(*args, **kwargs):
        pytest.fail("Offline consumer attempted acquisition")

    monkeypatch.setattr(registry, "download", no_download)
    monkeypatch.setattr(registry, "download_reference", no_download)
    old, new = tmp_path / "original", tmp_path / "shared"
    canine_bulk.curate_canine_bulk("pfizer_2026_03", old, local_assets=assets)
    canine_bulk.curate_canine_bulk("pfizer_2026_03", new, provider="orfdata")
    manifest = canine_bulk.verify_bulk_evidence(new)
    assert manifest["upstream"]["provider"] == "orfdata"
    assert manifest["upstream"]["dataset_manifest"]["dataset_id"] == source["orfdata"]["dataset_id"]
    for table in manifest["n_records_by_table"]:
        pd.testing.assert_frame_equal(
            canine_bulk.load_bulk_evidence(old, table=table),
            canine_bulk.load_bulk_evidence(new, table=table),
        )
    assert (old / "reference.fasta").read_bytes() == (new / "reference.fasta").read_bytes()
    proteins = canine_bulk.load_bulk_evidence(new, table="proteins")
    assert len(proteins) == 5 and proteins.ibaq.isna().sum() == 1
    assert proteins.ibaq.eq(0).sum() == 1
    assert not proteins.is_ms_observation.any()
    assert proteins.donor_id.eq("").all()
    assert canine_bulk.load_bulk_evidence(new).empty
    monkeypatch.setattr(
        canine_bulk, "canine_bulk_catalog", lambda: pytest.fail("Live catalog used")
    )
    canine_bulk.verify_bulk_evidence(old)
    canine_bulk.verify_bulk_evidence(new)


@pytest.mark.parametrize("failure", ["registry", "reference", "missing_unit", "value"])
def test_upstream_mismatch_never_publishes(upstream, tmp_path, monkeypatch, failure):
    registry, _, source = upstream
    if failure == "registry":
        source["orfdata"]["registry_sha256"] = "0" * 64
    elif failure == "reference":
        source["assets"]["reference"]["sha256"] = "0" * 64
    else:
        dataset = registry.open(source["orfdata"]["dataset_id"], source["orfdata"]["version"])
        original = dataset.measurements

        def altered(**kwargs):
            from dataclasses import replace

            for value in original(**kwargs):
                if failure == "missing_unit" and value.unit == "rIBAQ":
                    continue
                if failure == "value":
                    yield replace(value, provenance={**value.provenance, "native_value": "99"})
                else:
                    yield value

        monkeypatch.setattr(dataset, "measurements", altered)
        monkeypatch.setattr(registry, "open", lambda *args, **kwargs: dataset)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        canine_bulk.curate_canine_bulk("pfizer_2026_03", output, provider="orfdata")
    assert not output.exists()


def test_corrupted_cache_rejected_offline(upstream, tmp_path):
    registry, _, source = upstream
    dataset = registry.open(source["orfdata"]["dataset_id"], source["orfdata"]["version"])
    with sqlite3.connect(dataset.path) as db:
        db.execute("UPDATE metadata SET value='{}' WHERE name='counts'")
    with pytest.raises(Exception, match=r"(?i)(hash|sha|invalid|checksum)"):
        canine_bulk.curate_canine_bulk("pfizer_2026_03", tmp_path / "bad", provider="orfdata")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limits": {"max_upstream_bytes": 1}},
        {"limits": {"max_records": 1}},
        {"limits": {"max_output_bytes": 1}},
        {"local_assets": {"table": "unused"}},
        {"provider": "unknown"},
    ],
)
def test_invalid_options_fail_before_publication(upstream, tmp_path, kwargs):
    with pytest.raises(ValueError):
        canine_bulk.curate_canine_bulk(
            "pfizer_2026_03", tmp_path / "bad", **{"provider": "orfdata", **kwargs}
        )
    assert not (tmp_path / "bad").exists()


def test_orfdata_unsupported_source_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="orfdata"):
        canine_bulk.curate_canine_bulk("labrador_2024_01", tmp_path / "bad", provider="orfdata")


def test_explicit_acquisition_uses_pinned_dataset_and_reference(upstream, tmp_path, monkeypatch):
    registry, _, source = upstream
    calls = []
    for method in ("download", "download_reference"):
        original = getattr(registry, method)

        def record(*args, _method=method, _original=original, **kwargs):
            calls.append((_method, args))
            return _original(*args, **kwargs)

        monkeypatch.setattr(registry, method, record)
    canine_bulk.curate_canine_bulk(
        "pfizer_2026_03", tmp_path / "explicit", provider="orfdata", allow_download=True
    )
    identity = (source["orfdata"]["dataset_id"], source["orfdata"]["version"])
    assert calls == [("download", identity), ("download_reference", identity)]


def test_missing_cache_never_triggers_implicit_download(upstream, tmp_path, monkeypatch):
    registry, _, _ = upstream

    def missing(*args, **kwargs):
        raise FileNotFoundError("Explicit download required")

    def no_download(*args, **kwargs):
        pytest.fail("Implicit acquisition")

    monkeypatch.setattr(registry, "open", missing)
    monkeypatch.setattr(registry, "download", no_download)
    monkeypatch.setattr(registry, "download_reference", no_download)
    with pytest.raises(FileNotFoundError, match="Explicit download"):
        canine_bulk.curate_canine_bulk("pfizer_2026_03", tmp_path / "missing", provider="orfdata")
    assert not (tmp_path / "missing").exists()
