"""Tests for #958: structured dataset catalog, hitlist data preflight, and consistent symlink status."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import hitlist
from hitlist import cli, downloads


def test_available_dataset_catalog_covers_all_datasets_and_filters() -> None:
    flat = hitlist.available_datasets()
    catalog = hitlist.available_dataset_catalog()
    assert set(catalog.keys()) == set(flat.keys())

    valid_categories = set(hitlist.dataset_categories())
    for name, meta in catalog.items():
        assert meta["name"] == name
        assert meta["category"] in valid_categories
        assert meta["group"]
        assert meta["species"]
        assert meta["kind"]
        assert meta["description"]

    core = hitlist.available_dataset_catalog(category="core_immunopeptidomics")
    assert set(core.keys()) == {"iedb", "cedar"}

    viral = hitlist.available_dataset_catalog(category="viral_proteomes")
    assert {"ebv", "hpv16", "hiv1"} <= set(viral.keys())

    with pytest.raises(ValueError, match="Unknown dataset category"):
        hitlist.available_dataset_catalog(category="nonexistent_category")


def test_cli_data_available_and_preflight(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["hitlist", "data", "available", "--category", "core_immunopeptidomics", "--json"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in payload} == {"iedb", "cedar"}
    assert all(row["category"] == "core_immunopeptidomics" for row in payload)
    assert all("species" in row and "kind" in row for row in payload)

    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "available", "--grouped"])
    cli.main()
    grouped_out = capsys.readouterr().out
    assert "[core_immunopeptidomics]" in grouped_out
    assert "[viral_proteomes]" in grouped_out

    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "preflight", "iedb"])
    cli.main()
    pf = json.loads(capsys.readouterr().out)
    assert pf["name"] == "iedb"
    assert "would_download" in pf
    assert "symlink_status" in pf


def test_symlink_status_consistent_across_info_preflight_and_inventory(tmp_path: Path) -> None:
    """info(), preflight(), and list_cache_files() agree on valid, broken, and directory symlinks (#958)."""
    data_root = downloads.data_dir()
    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Valid symlink to a regular CSV file -> installed / available + valid_symlink
    real_csv = tmp_path / "shared_cedar.csv"
    real_csv.write_text("col1,col2\n1,2\n", encoding="utf-8")
    valid_link = tmp_path / "cedar_link.csv"
    valid_link.symlink_to(real_csv)

    # 2. Broken symlink -> missing / stale + broken_symlink
    broken_link = tmp_path / "iedb_broken.csv"
    broken_link.symlink_to(tmp_path / "missing_iedb.csv")

    # 3. Symlink to a directory -> invalid / stale + non_regular_target
    target_dir = tmp_path / "some_dir"
    target_dir.mkdir()
    dir_link = tmp_path / "hpa_dir_link.tsv"
    dir_link.symlink_to(target_dir)

    (data_root / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "cedar": {"path": str(valid_link)},
                    "iedb": {"path": str(broken_link)},
                    "hpa_bulk": {"path": str(dir_link)},
                }
            }
        ),
        encoding="utf-8",
    )

    # info() checks
    cedar_info = downloads.info("cedar")
    assert cedar_info["status"] == "installed"
    assert cedar_info["exists"] is True
    assert cedar_info["is_symlink"] is True
    assert cedar_info["symlink_status"] == "valid_symlink"
    assert Path(cedar_info["resolved_path"]) == real_csv.resolve()

    iedb_info = downloads.info("iedb")
    assert iedb_info["status"] == "missing"
    assert iedb_info["exists"] is False
    assert iedb_info["is_symlink"] is True
    assert iedb_info["symlink_status"] == "broken_symlink"

    hpa_info = downloads.info("hpa_bulk")
    assert hpa_info["status"] == "invalid"
    assert hpa_info["exists"] is False
    assert hpa_info["is_symlink"] is True
    assert hpa_info["symlink_status"] == "non_regular_target"

    # preflight() checks
    cedar_pf = downloads.preflight("cedar")
    assert cedar_pf["stale"] is False
    assert cedar_pf["is_symlink"] is True
    assert cedar_pf["symlink_status"] == "valid_symlink"
    assert Path(cedar_pf["resolved_path"]) == real_csv.resolve()

    iedb_pf = downloads.preflight("iedb")
    assert iedb_pf["stale"] is True
    assert iedb_pf["is_symlink"] is True
    assert iedb_pf["symlink_status"] == "broken_symlink"

    # list_cache_files() checks
    inv = {
        row["names"][0]: row
        for row in downloads.list_cache_files(include_unregistered=False)
        if row["names"]
    }
    assert inv["cedar"]["status"] == "available"
    assert inv["cedar"]["symlink_status"] == "valid_symlink"
    assert inv["iedb"]["status"] == "missing"
    assert inv["iedb"]["symlink_status"] == "broken_symlink"
    assert inv["hpa_bulk"]["status"] == "invalid"
    assert inv["hpa_bulk"]["symlink_status"] == "non_regular_target"
