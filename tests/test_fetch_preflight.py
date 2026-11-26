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

"""``downloads.preflight``, space refusal, early terms notice, stale-registration reporting,
``exists`` fields (#884) and the matching ``hitlist data fetch --dry-run`` / error handling."""

from __future__ import annotations

import json
import shutil
import sys
from collections import namedtuple
from pathlib import Path

import pytest

import hitlist
from hitlist import cli, downloads
from hitlist.errors import (
    DatasetError,
    DownloadError,
    HitlistError,
    InsufficientSpaceError,
    IntegrityError,
)

Usage = namedtuple("usage", ["total", "used", "free"])


@pytest.fixture
def data_dir(tmp_path):
    downloads.set_data_dir(tmp_path)
    yield tmp_path
    downloads._override_data_dir = None


@pytest.fixture
def fake_download(monkeypatch):
    calls: list[tuple[str, Path]] = []

    def fake(url, dest, **kwargs):
        calls.append((url, Path(dest)))
        label = kwargs.get("label") or ""
        floor = downloads.FETCHABLE_DATASETS.get(label, {}).get("min_size_bytes", 1)
        target = Path(dest)
        if target.suffix == ".fasta":
            payload = b">sp|P1|MOCK\n" + b"M" * max(floor, 1) + b"\n"
        else:
            payload = b"x" * max(floor, 1)
        target.write_bytes(payload)
        return target

    monkeypatch.setattr(downloads, "download_to_file", fake)
    return calls


def _free_space(monkeypatch, free_bytes: int) -> None:
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: Usage(10**13, 10**13 - free_bytes, free_bytes)
    )


def test_preflight_reports_a_fresh_dataset(data_dir, monkeypatch):
    _free_space(monkeypatch, 50 * 10**9)
    plan = downloads.preflight("iedb")
    assert plan["name"] == "iedb"
    assert plan["path"] == str(data_dir / "mhc_ligand_full.csv")
    assert plan["would_download"] is True
    assert plan["registered"] is False and plan["exists"] is False and plan["stale"] is False
    assert plan["expected_size_bytes"] == downloads.FETCHABLE_DATASETS["iedb"]["size_bytes"]
    assert plan["free_bytes"] == 50 * 10**9
    assert plan["enough_space"] is True
    assert plan["terms"] == "https://www.iedb.org/"


def test_preflight_respects_force_for_cached_dataset(data_dir, fake_download, monkeypatch):
    _free_space(monkeypatch, 50 * 10**9)
    downloads.fetch("hpv16")
    assert downloads.preflight("hpv16")["would_download"] is False
    assert downloads.preflight("hpv16", force=True)["would_download"] is True


def test_preflight_uses_expected_size_for_pinned_fetchable_dataset(data_dir, monkeypatch):
    _free_space(monkeypatch, 10_000_000)
    plan = downloads.preflight("netmhcstabpan")
    assert (
        plan["expected_size_bytes"]
        == downloads.FETCHABLE_DATASETS["netmhcstabpan"]["expected_size"]
    )
    assert plan["enough_space"] is True
    assert plan["terms"] == "https://pubmed.ncbi.nlm.nih.gov/27402703/"


def test_preflight_without_a_declared_size_makes_no_space_claim(data_dir, monkeypatch):
    _free_space(monkeypatch, 1)
    plan = downloads.preflight("hpv16")
    assert plan["expected_size_bytes"] is None
    assert plan["enough_space"] is None


def test_preflight_measures_the_nearest_existing_parent(data_dir, monkeypatch):
    seen = []

    def usage(path):
        seen.append(path)
        return Usage(1, 0, 1)

    downloads.set_data_dir(data_dir / "not" / "created" / "yet")
    monkeypatch.setattr(shutil, "disk_usage", usage)
    downloads.preflight("hpv16")
    assert seen and seen[0].exists()
    assert not (data_dir / "not").exists()


@pytest.mark.parametrize("name", ["no-such-dataset", "hpa_bulk"])
def test_preflight_refuses_what_fetch_cannot_do_automatically(data_dir, name):
    with pytest.raises(DatasetError) as raised:
        downloads.preflight(name)
    assert isinstance(raised.value, ValueError) and isinstance(raised.value, HitlistError)


def test_preflight_depmap_returns_composite_bundle_plan(data_dir, monkeypatch):
    _free_space(monkeypatch, 50 * 10**9)
    plan = downloads.preflight("depmap")
    assert plan["name"] == "depmap"
    assert plan["kind"] == "bundle"
    assert plan["would_download"] is True
    assert len(plan["files"]) == len(downloads._DEPMAP_FILES)
    assert plan["terms"] == "https://doi.org/10.25452/figshare.plus.27993248.v1"


def test_fetch_refuses_to_start_when_the_dataset_will_not_fit(data_dir, fake_download, monkeypatch):
    _free_space(monkeypatch, 1_000_000)
    with pytest.raises(InsufficientSpaceError, match="Not enough free space for 'iedb'") as raised:
        downloads.fetch("iedb")
    err = raised.value
    assert isinstance(err, DownloadError)
    assert isinstance(err, HitlistError)
    assert err.path == data_dir
    assert err.required_bytes == downloads.FETCHABLE_DATASETS["iedb"]["size_bytes"]
    assert err.available_bytes == 1_000_000
    assert fake_download == []
    assert not (data_dir / "mhc_ligand_full.csv").exists()


def test_preflight_and_fetch_data_asset_enforce_space(tmp_path, monkeypatch):
    asset_name = sorted(downloads.data_assets())[0]
    asset_size = downloads.data_assets()[asset_name]["size_bytes"]
    monkeypatch.setattr(downloads, "data_asset_dir", lambda: tmp_path)
    _free_space(monkeypatch, max(asset_size - 1, 0))

    plan = downloads.preflight(asset_name)
    assert plan["path"] == str(tmp_path / asset_name)
    assert plan["expected_size_bytes"] == asset_size
    assert plan["enough_space"] is False

    called = []
    import datacache

    monkeypatch.setattr(datacache, "fetch_file", lambda *a, **k: called.append((a, k)))
    with pytest.raises(InsufficientSpaceError) as raised:
        downloads.fetch_data_asset(asset_name)
    assert raised.value.required_bytes == asset_size
    assert raised.value.available_bytes == max(asset_size - 1, 0)
    assert called == []

    # Existing cached file is allowed without forcing even if free space is tight.
    cached = tmp_path / asset_name
    cached.write_bytes(b"cached")
    monkeypatch.setattr(datacache, "fetch_file", lambda *a, **k: str(cached))
    assert downloads.fetch_data_asset(asset_name) == cached


def test_typed_download_errors_translates_upstream_insufficient_space():
    class InsufficientSpace(OSError):
        def __init__(self, msg: str):
            super().__init__(msg)
            self.path = "/var/cache/hitlist"
            self.required_bytes = 5_000_000
            self.available_bytes = 1_000

    with (
        pytest.raises(InsufficientSpaceError) as raised,
        downloads._typed_download_errors("https://example.org/file.bin"),
    ):
        raise InsufficientSpace("disk full")

    err = raised.value
    assert isinstance(err.__cause__, InsufficientSpace)
    assert err.path == Path("/var/cache/hitlist")
    assert err.required_bytes == 5_000_000
    assert err.available_bytes == 1_000


def test_terms_notice_is_printed_before_the_download_starts(data_dir, monkeypatch, capsys):
    _free_space(monkeypatch, 50 * 10**9)
    seen_before_download = []

    def fake(url, dest, **kwargs):
        seen_before_download.append(capsys.readouterr().err)
        Path(dest).write_bytes(b"x" * downloads.FETCHABLE_DATASETS["iedb"]["min_size_bytes"])
        return Path(dest)

    monkeypatch.setattr(downloads, "download_to_file", fake)
    downloads.fetch("iedb")
    assert "terms-of-use-governed source" in seen_before_download[0]
    assert "terms-of-use-governed source" not in capsys.readouterr().err


def test_fetch_says_when_a_registration_is_stale(data_dir, fake_download, capsys):
    gone = data_dir / "elsewhere" / "hpv16.fasta"
    with downloads.manifest_transaction() as manifest:
        manifest["datasets"]["hpv16"] = {"path": str(gone), "description": "x"}
    assert downloads.preflight("hpv16")["stale"] is True
    downloads.fetch("hpv16")
    err = capsys.readouterr().err
    assert f"registered file {gone} is missing" in err
    assert fake_download


def test_fetch_short_body_raises_integrity_error(data_dir, monkeypatch):
    def tiny(url, dest, **kwargs):
        Path(dest).write_bytes(b"oops")
        return Path(dest)

    monkeypatch.setattr(downloads, "download_to_file", tiny)
    with pytest.raises(IntegrityError, match="empty or error body"):
        downloads.fetch("hpv16")
    assert not (data_dir / "hpv16.fasta").exists()


def test_list_datasets_and_info_report_whether_the_file_exists(data_dir, fake_download):
    path = downloads.fetch("hpv16")
    assert downloads.list_datasets()["hpv16"]["exists"] is True
    assert downloads.info("hpv16")["exists"] is True
    path.unlink()
    assert downloads.list_datasets()["hpv16"]["exists"] is False
    assert downloads.info("hpv16")["exists"] is False
    assert downloads.info("ebv")["exists"] is False
    assert downloads.info("hpa_bulk")["exists"] is False
    assert downloads.info("depmap")["exists"] is False


def test_cli_dry_run_prints_the_plan_and_downloads_nothing(
    data_dir, fake_download, monkeypatch, capsys
):
    _free_space(monkeypatch, 50 * 10**9)
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "fetch", "iedb", "--dry-run"])
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan == downloads.preflight("iedb")
    assert fake_download == []


def test_cli_dry_run_honors_force_flag(data_dir, fake_download, monkeypatch, capsys):
    downloads.fetch("hpv16")
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "fetch", "hpv16", "--dry-run", "--force"])
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["exists"] is True
    assert plan["would_download"] is True
    assert len(fake_download) == 1


@pytest.mark.parametrize("verb", ["fetch", "refresh"])
def test_cli_reports_unknown_datasets_as_one_error_line(data_dir, monkeypatch, capsys, verb):
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", verb, "no-such-dataset"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error: Unknown dataset 'no-such-dataset'")
    assert "Traceback" not in captured.err


def test_cli_reports_insufficient_space_as_one_error_line(
    data_dir, fake_download, monkeypatch, capsys
):
    _free_space(monkeypatch, 1_000_000)
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "fetch", "iedb"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error: Not enough free space for 'iedb'")
    assert "Traceback" not in captured.err
    assert fake_download == []


def test_api_callers_can_still_catch_value_error(data_dir):
    with pytest.raises(ValueError, match="Unknown dataset"):
        downloads.fetch("no-such-dataset")


def test_public_exports_include_preflight_and_errors():
    assert hitlist.preflight is downloads.preflight
    assert hitlist.DatasetError is DatasetError
    assert hitlist.InsufficientSpaceError is InsufficientSpaceError
    assert hitlist.list_cache_files is downloads.list_cache_files
    assert hitlist.fetch_all_data_assets is downloads.fetch_all_data_assets
    assert hitlist.list_proteomes is downloads.list_proteomes
