# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The CLI must report *every* directory hitlist uses (#291).

Before this, ``hitlist data list`` printed ``Data directory: {data_dir()}`` and
nothing else, so the mirrored data assets that ``fetch_data_asset`` writes into
datacache's cache dir — a wholly different location — were undiscoverable: a
user asking "where did my data go?" got half an answer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

from hitlist import cli, downloads


@pytest.fixture
def _split_locations(tmp_path, monkeypatch):
    """Force the two locations apart, as a legacy/configured install has them."""
    built = tmp_path / "built-indexes"
    assets = tmp_path / "datacache-cache"
    built.mkdir()
    assets.mkdir()
    monkeypatch.setattr(downloads, "_override_data_dir", built)
    monkeypatch.setattr(downloads, "_data_dir_cache", {})
    monkeypatch.setattr(cli, "data_asset_dir", lambda: assets)
    monkeypatch.setattr(downloads, "data_asset_dir", lambda: assets)
    monkeypatch.setattr("hitlist.proteome._PROTEOME_INDEX_DISK_CACHE_DIR", tmp_path / "index-cache")
    return built, assets


def test_data_dirs_reports_both_locations(_split_locations, capsys):
    built, assets = _split_locations
    cli._data_dirs(argparse.Namespace())
    out = capsys.readouterr().out
    assert str(built) in out
    assert str(assets) in out
    assert "built indexes" in out
    assert "data assets" in out


def test_data_dirs_names_the_rule_that_chose_the_directory(_split_locations, capsys):
    cli._data_dirs(argparse.Namespace())
    assert "set_data_dir()" in capsys.readouterr().out


def test_data_dirs_reports_the_proteome_index_cache(_split_locations, capsys, tmp_path):
    """A third location, thousands of files, and not moved by any override."""
    cli._data_dirs(argparse.Namespace())
    out = capsys.readouterr().out
    assert str(tmp_path / "index-cache") in out
    assert "proteome index cache" in out


def test_data_list_footer_reports_the_asset_dir(_split_locations, monkeypatch, capsys):
    """The #291 gap itself: ``data list`` used to print only ``data_dir()``."""
    _, assets = _split_locations
    monkeypatch.setattr(downloads, "list_datasets", dict)
    cli._data_list(argparse.Namespace())
    assert str(assets) in capsys.readouterr().out


def test_data_list_footer_reports_the_asset_dir_with_datasets(
    _split_locations, monkeypatch, capsys
):
    built, assets = _split_locations
    monkeypatch.setattr(
        downloads,
        "list_datasets",
        lambda: {
            "iedb": {
                "path": str(built / "iedb.csv"),
                "size_bytes": 10,
                "registered": "2026-01-01",
                "description": "d",
            }
        },
    )
    out_dir_line = cli._data_list(argparse.Namespace())
    assert out_dir_line is None
    out = capsys.readouterr().out
    assert str(built) in out
    assert str(assets) in out


def test_data_available_footer_reports_both(_split_locations, monkeypatch, capsys):
    built, assets = _split_locations
    monkeypatch.setattr(downloads, "list_datasets", dict)
    monkeypatch.setattr(cli, "available_datasets", lambda: {"iedb": "IEDB"})
    cli._data_available(argparse.Namespace())
    out = capsys.readouterr().out
    assert str(built) in out
    assert str(assets) in out


def test_one_line_when_the_two_locations_coincide(tmp_path, monkeypatch, capsys):
    """A fresh install keeps indexes and assets together — say so once."""
    shared = tmp_path / "shared"
    monkeypatch.setattr(downloads, "_override_data_dir", shared)
    monkeypatch.setattr(downloads, "_data_dir_cache", {})
    monkeypatch.setattr(cli, "data_asset_dir", lambda: shared)
    monkeypatch.setattr("hitlist.proteome._PROTEOME_INDEX_DISK_CACHE_DIR", tmp_path / "index-cache")
    lines = cli._data_location_lines()
    assert sum(str(shared) in line for line in lines) == 1
    assert "built indexes + assets" in lines[0]


def test_dirs_subcommand_parses_and_dispatches(monkeypatch):
    parser = argparse.ArgumentParser(prog="hitlist")
    sub = parser.add_subparsers(dest="command")
    cli._build_data_parser(sub)
    args = parser.parse_args(["data", "dirs"])
    assert args.data_command == "dirs"

    called = []
    monkeypatch.setattr(cli, "_data_dirs", lambda a: called.append(a))
    cli._handle_data(args)
    assert len(called) == 1


def test_dir_summary_does_not_create_or_stat_a_missing_dir(tmp_path):
    missing = tmp_path / "nope"
    assert cli._dir_summary(missing) == "not created yet"
    assert not missing.exists()


def test_dir_summary_distinguishes_empty_from_populated(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli._dir_summary(empty) == "empty"
    (empty / "a.parquet").write_bytes(b"x" * 2048)
    (empty / "sub").mkdir()
    summary = cli._dir_summary(empty)
    assert "2.0 KB in 1 files" in summary
    assert "1 subdirectories" in summary


def test_dir_summary_skips_sizing_a_huge_directory(tmp_path, monkeypatch):
    """The proteome index cache holds thousands of files; do not stat them all."""
    monkeypatch.setattr(cli, "_DIR_SUMMARY_STAT_LIMIT", 3)
    big = tmp_path / "big"
    big.mkdir()
    for i in range(4):
        (big / f"{i}.pkl").write_bytes(b"x")
    assert cli._dir_summary(big) == "4 files (size not measured)"


# ── #291 review fixes ────────────────────────────────────────────────────────


def test_every_origin_has_a_label():
    """Drift guard: a fifth resolution rule must not print a bare token, and
    `_DATA_DIR_ORIGIN_LABELS[...]` used to raise KeyError on three commands."""
    assert set(cli._DATA_DIR_ORIGIN_LABELS) == set(downloads.DATA_DIR_ORIGINS)


def test_an_unlabelled_origin_degrades_to_the_token(_split_locations, monkeypatch, capsys):
    monkeypatch.setattr(downloads, "resolve_data_dir", lambda: (Path("/x"), "future-rule"))
    monkeypatch.setattr(cli, "resolve_data_dir", lambda: (Path("/x"), "future-rule"))
    cli._data_dirs(argparse.Namespace())
    assert "future-rule" in capsys.readouterr().out


def test_locations_resolve_the_data_dir_exactly_once(_split_locations, monkeypatch):
    """Two resolutions could print a path that disagrees with its explanation."""
    calls = []
    real = downloads.resolve_data_dir
    monkeypatch.setattr(cli, "resolve_data_dir", lambda: (calls.append(1), real())[1])
    cli._data_location_lines()
    assert len(calls) == 1


def test_index_cache_label_follows_set_disk_cache_dir(tmp_path, monkeypatch, capsys):
    """Report whether the cache follows the data directory or an override."""
    from hitlist import proteome

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "built")
    monkeypatch.setattr(downloads, "_data_dir_cache", {})
    monkeypatch.setattr(cli, "data_asset_dir", lambda: tmp_path / "assets")
    monkeypatch.setattr(proteome, "_PROTEOME_INDEX_DISK_CACHE_DIR", tmp_path / "moved-cache")
    moved = [ln for ln in cli._data_location_lines() if "proteome index cache" in ln]
    assert "set_disk_cache_dir()" in moved[0]
    assert "always ~/.hitlist" not in moved[0]

    proteome.set_disk_cache_dir(None)
    default = [ln for ln in cli._data_location_lines() if "proteome index cache" in ln]
    assert "data directory" in default[0]
    assert str(tmp_path / "built" / "proteome_index_cache") in default[0]


def test_dir_summary_does_not_stat_entries_past_the_limit(tmp_path, monkeypatch):
    """The guard exists to avoid the stats; `Path.iterdir()` + `is_file()` paid
    them before it could apply (~90 ms on a 7.5k-file index cache)."""
    monkeypatch.setattr(cli, "_DIR_SUMMARY_STAT_LIMIT", 3)
    big = tmp_path / "big"
    big.mkdir()
    for i in range(6):
        (big / f"{i}.pkl").write_bytes(b"x")
    stats = []
    original = os.stat

    def counting_stat(path, *a, **k):
        stats.append(path)
        return original(path, *a, **k)

    monkeypatch.setattr(os, "stat", counting_stat)
    assert cli._dir_summary(big) == "6 files (size not measured)"
    assert not [p for p in stats if str(p).endswith(".pkl")]


def test_dir_summary_does_not_call_a_directory_with_a_fifo_empty(tmp_path):
    """`Path.is_file()`/`is_dir()` both swallow the error for a FIFO."""
    d = tmp_path / "odd"
    d.mkdir()
    os.mkfifo(d / "pipe")
    summary = cli._dir_summary(d)
    assert summary != "empty"
    assert "1 other entries" in summary


def test_dir_summary_does_not_call_a_directory_with_a_dangling_symlink_empty(tmp_path):
    d = tmp_path / "odd"
    d.mkdir()
    (d / "link").symlink_to(tmp_path / "gone")
    summary = cli._dir_summary(d)
    assert summary != "empty"
    assert "1 other entries" in summary


def test_main_prints_the_legacy_notice_once(monkeypatch, capsys):
    """The notice moved out of `data_dir()` to the CLI entry point."""
    monkeypatch.setattr(cli, "legacy_data_dir_notice", lambda: "[hitlist] legacy!")
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "dirs"])
    monkeypatch.setattr(cli, "_handle_data", lambda args: None)
    cli.main()
    assert capsys.readouterr().err.count("[hitlist] legacy!") == 1


def test_main_prints_nothing_when_not_on_the_legacy_dir(monkeypatch, capsys):
    monkeypatch.setattr(cli, "legacy_data_dir_notice", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "dirs"])
    monkeypatch.setattr(cli, "_handle_data", lambda args: None)
    cli.main()
    assert capsys.readouterr().err == ""
