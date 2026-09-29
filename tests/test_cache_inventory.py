"""Cross-root inventory is offline, read-only, and explicit about trust."""

import argparse
import hashlib
import json
from pathlib import Path

import datacache
import pytest

from hitlist import cli, downloads, proteome


@pytest.fixture
def locations(tmp_path, monkeypatch):
    built, assets, indexes = (tmp_path / name for name in ("built", "assets", "indexes"))
    monkeypatch.setattr(downloads, "_override_data_dir", built)
    monkeypatch.setattr(downloads, "data_asset_dir", lambda: assets)
    monkeypatch.setattr(proteome, "_PROTEOME_INDEX_DISK_CACHE_DIR", indexes)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(
        downloads,
        "data_assets",
        lambda: {"source.csv": {"size_bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()}},
    )
    return built, assets, indexes


def test_inventory_does_not_create_missing_locations(locations, monkeypatch):
    monkeypatch.setattr(datacache, "fetch_file", lambda *a, **k: pytest.fail("network"))
    rows = downloads.list_cache_files()
    assert len(rows) == 1
    assert rows[0]["status"] == "missing"
    assert not any(path.exists() for path in locations)


def test_inventory_spans_roots_external_files_and_legacy_cache(locations, tmp_path):
    built, assets, indexes = locations
    for path in locations:
        path.mkdir()
    external = tmp_path / "outside.tsv"
    external.write_bytes(b"external")
    downloads.register("external", external)
    downloads.register("duplicate", external)
    (built / "observations.parquet").write_bytes(b"index")
    (assets / "source.csv").write_bytes(b"data")
    (indexes / "proteome.pkl").write_bytes(b"pickle")
    (indexes / "inflight.tmp").write_bytes(b"partial")
    (assets / ".source.csv.datacache.json").write_text("{}")
    partial = assets / ".datacache-partial"
    partial.mkdir()
    (partial / "payload").write_bytes(b"incomplete")
    legacy = tmp_path / "home" / ".hitlist"
    legacy.mkdir(parents=True)
    (legacy / "old.parquet").write_bytes(b"old index")
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    rows = downloads.list_cache_files()
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"available"}
    assert not any(row["verified"] for row in rows)
    assert next(row for row in rows if row["path"] == str(external))["names"] == [
        "duplicate",
        "external",
    ]
    assert next(row for row in rows if row["path"].endswith("proteome.pkl"))["kinds"] == [
        "proteome index"
    ]
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    external.unlink()
    assert (
        next(row for row in downloads.list_cache_files() if row["path"] == str(external))["status"]
        == "missing"
    )


def test_shared_roots_deduplicate_and_verify_trusted_metadata(locations, monkeypatch):
    built, _, _ = locations
    built.mkdir()
    monkeypatch.setattr(downloads, "data_asset_dir", lambda: built)
    path = built / "source.csv"
    path.write_bytes(b"data")
    downloads.register("shared", path)
    rows = downloads.list_cache_files(verify=True)
    assert len(rows) == 1
    assert rows[0]["verified"]
    path.write_bytes(b"fake")  # same size does not prove integrity
    assert downloads.list_cache_files()[0]["status"] == "available"
    assert downloads.list_cache_files(verify=True)[0]["status"] == "corrupt"


def test_inventory_reports_provenance_without_claiming_verification(locations, tmp_path):
    _, assets, _ = locations
    source = tmp_path / "original"
    source.write_bytes(b"data")
    datacache.fetch_file(
        source.as_uri(),
        destination=assets / "source.csv",
        raw=True,
        expected_sha256=hashlib.sha256(b"data").hexdigest(),
        record_provenance=True,
    )
    row = downloads.list_cache_files()[0]
    assert row["source_url"] == source.as_uri()
    assert row["recorded_sha256"] == hashlib.sha256(b"data").hexdigest()
    assert not row["verified"]
    assert downloads.list_cache_files(verify=True)[0]["verified"]


def test_inventory_cli_json_parses_and_dispatches(locations, capsys):
    parser = argparse.ArgumentParser()
    cli._build_data_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["data", "list", "--all", "--json", "--verify"])
    cli._handle_data(args)
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["status"] == "missing"
    assert rows[0]["verified"] is False


def test_inventory_exposes_unreadable_directory(locations, monkeypatch):
    built, _, _ = locations
    built.mkdir()
    import hitlist.cache_inventory as inventory

    def fail_walk(root, *, onerror):
        onerror(PermissionError(13, "Permission denied", str(root)))
        return iter(())

    monkeypatch.setattr(inventory.os, "walk", fail_walk)
    rows = downloads.list_cache_files()
    row = next(row for row in rows if row["path"] == str(built))
    assert row["status"] == "inaccessible"
    assert "Permission denied" in row["error"]


def test_mirrored_assets_share_catalog_commands(locations, monkeypatch):
    _, assets, _ = locations
    metadata = {"size_bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest(), "source": "paper"}
    monkeypatch.setattr(downloads, "data_assets", lambda: {"source.csv": metadata})
    assert "source.csv" in downloads.available_datasets()
    assert downloads.info("source.csv")["status"] == "missing"
    with pytest.raises(FileNotFoundError):
        downloads.get_path("source.csv")
    seen = []

    def fetch_asset(name, *, force=False):
        seen.append((name, force))
        assets.mkdir()
        path = assets / name
        path.write_bytes(b"data")
        return path

    monkeypatch.setattr(downloads, "fetch_data_asset", fetch_asset)
    path = downloads.fetch("source.csv", force=True)
    assert seen == [("source.csv", True)]
    assert downloads.get_path("source.csv") == path
    assert downloads.info("source.csv")["cache_status"] == "available"
    assert downloads.list_datasets() == {}


def test_info_does_not_claim_missing_registered_file_is_installed(locations):
    built, _, _ = locations
    built.mkdir()
    path = built / "registered.tsv"
    path.write_bytes(b"data")
    downloads.register("known", path)
    assert downloads.info("known")["status"] == "installed"
    path.unlink()
    assert downloads.info("known")["status"] == "missing"
