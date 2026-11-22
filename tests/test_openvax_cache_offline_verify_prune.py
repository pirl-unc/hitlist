"""Tests for #960: OPENVAX_DATA_CACHE fallback, OPENVAX_OFFLINE/--offline, and hitlist data verify|prune."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datacache
import pytest

import hitlist
from hitlist import cli, downloads
from hitlist.errors import OfflineError


def test_openvax_data_cache_fallback_and_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_data_dir honors HITLIST_DATA_DIR > OPENVAX_DATA_CACHE > legacy ~/.hitlist > default (#960)."""
    monkeypatch.setattr(downloads, "_override_data_dir", None)
    monkeypatch.delenv("HITLIST_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    openvax_root = tmp_path / "scratch_openvax"
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(openvax_root))
    downloads.reset_data_dir_cache()

    path, origin = downloads.resolve_data_dir()
    assert path == openvax_root / "hitlist"
    assert origin == "openvax_env"
    assert downloads.data_asset_dir() == openvax_root / "hitlist"

    # Explicit HITLIST_DATA_DIR still wins over OPENVAX_DATA_CACHE
    explicit = tmp_path / "explicit_hitlist"
    monkeypatch.setenv("HITLIST_DATA_DIR", str(explicit))
    downloads.reset_data_dir_cache()
    path2, origin2 = downloads.resolve_data_dir()
    assert path2 == explicit
    assert origin2 == "env"

    # Unsetting both falls back cleanly to populated legacy ~/.hitlist if present
    monkeypatch.delenv("HITLIST_DATA_DIR", raising=False)
    monkeypatch.delenv("OPENVAX_DATA_CACHE", raising=False)
    legacy = tmp_path / "home" / ".hitlist"
    legacy.mkdir()
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")
    downloads.reset_data_dir_cache()
    path3, origin3 = downloads.resolve_data_dir()
    assert path3 == legacy
    assert origin3 == "legacy"


def test_offline_kill_switch_blocks_network_but_allows_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OPENVAX_OFFLINE / DATACACHE_OFFLINE / --offline block uncached fetches while permitting cache hits."""
    downloads.set_offline(None)
    monkeypatch.delenv("OPENVAX_OFFLINE", raising=False)
    monkeypatch.delenv("DATACACHE_OFFLINE", raising=False)
    assert hitlist.is_offline() is False

    monkeypatch.setenv("OPENVAX_OFFLINE", "1")
    assert hitlist.is_offline() is True
    with pytest.raises(OfflineError, match="Offline mode is enabled"):
        downloads.fetch("ebv")

    monkeypatch.delenv("OPENVAX_OFFLINE", raising=False)
    monkeypatch.setenv("DATACACHE_OFFLINE", "true")
    assert hitlist.is_offline() is True
    with pytest.raises(OfflineError, match="Offline mode is enabled"):
        downloads.download_to_file("https://example.invalid/file.csv", tmp_path / "missing.csv")

    # Existing cached file with force=False succeeds offline!
    cached = tmp_path / "existing.csv"
    cached.write_text("cached content\n", encoding="utf-8")
    out = downloads.download_to_file("https://example.invalid/file.csv", cached, verbose=False)
    assert out == cached

    # --offline CLI flag fails cleanly with OfflineError -> exit 1
    monkeypatch.delenv("DATACACHE_OFFLINE", raising=False)
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "fetch", "ebv", "--offline"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_data_verify_cli_passes_and_detects_missing_registered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = downloads.data_dir()
    data_root.mkdir(parents=True, exist_ok=True)

    # Valid registered file -> ok=True
    good_csv = tmp_path / "good.csv"
    good_csv.write_text("a,b\n1,2\n", encoding="utf-8")
    downloads.register("cedar", good_csv)

    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "verify", "--json"])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["n_failed"] == 0

    # Delete registered file -> ok=False and exit code 1
    good_csv.unlink()
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "verify", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    bad_report = json.loads(capsys.readouterr().out)
    assert bad_report["ok"] is False
    assert bad_report["n_failed"] >= 1


def test_data_prune_dry_run_default_and_yes_deletes_old_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """hitlist data prune is dry-run by default and prunes old datacache generations only with --yes (#960)."""
    data_root = downloads.data_dir()
    store_dir = data_root / "dataset"
    src_file = tmp_path / "src.txt"
    src_file.write_text("hello\n", encoding="utf-8")

    # Create 3 distinct bundle generations via datacache.materialize
    for gen in (1, 2, 3):
        datacache.materialize(
            store_dir,
            {"src": {"path": str(src_file)}},
            transform={"version": f"gen-{gen}"},
            outputs={"out.txt": {}},
            builder=lambda _in, out, g=gen: Path(out["out.txt"]).write_text(
                f"payload-{g}\n" * 100, encoding="utf-8"
            ),
            force=True,
            download_options={"show_progress": False},
        )
    assert len(datacache.list_bundles(store_dir)) == 3

    # Default dry-run reports 2 prunable bundles but deletes NOTHING
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "prune", "--json"])
    cli.main()
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] is True
    assert dry["n_prunable_bundles"] == 2
    assert dry["n_deleted_bundles"] == 0
    assert len(datacache.list_bundles(store_dir)) == 3

    # With --yes, deletes the 2 oldest generations and keeps the newest 1
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "prune", "--yes", "--json"])
    cli.main()
    applied = json.loads(capsys.readouterr().out)
    assert applied["dry_run"] is False
    assert applied["n_deleted_bundles"] == 2
    assert len(datacache.list_bundles(store_dir)) == 1
