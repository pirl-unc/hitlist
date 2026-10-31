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

"""``downloads.preflight``, stale-registration reporting, ``exists`` fields (#884)
and the matching ``hitlist data fetch --dry-run`` / error handling in the CLI."""

from __future__ import annotations

import json
import shutil
import sys
from collections import namedtuple

import pytest

from hitlist import cli, downloads
from hitlist.errors import DatasetError, DownloadError, HitlistError

GOOD = b">sp|P1|X\n" + b"MKV" * 400 + b"\n"
Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def data_dir(tmp_path):
    downloads.set_data_dir(tmp_path)
    yield tmp_path
    downloads._override_data_dir = None


@pytest.fixture
def fake_download(monkeypatch):
    calls = []

    def fake(url, dest, **kwargs):
        calls.append(url)
        dest.write_bytes(GOOD)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake)
    return calls


def _free_space(monkeypatch, free_bytes):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: Usage(10**13, 10**13 - free_bytes, free_bytes)
    )


def test_preflight_reports_a_fresh_dataset(data_dir, monkeypatch):
    _free_space(monkeypatch, 50 * 10**9)
    plan = downloads.preflight("iedb")
    assert plan["path"] == str(data_dir / "mhc_ligand_full.csv")
    assert plan["would_download"] is True
    assert plan["registered"] is False and plan["exists"] is False and plan["stale"] is False
    assert plan["expected_size_bytes"] == downloads.FETCHABLE_DATASETS["iedb"]["size_bytes"]
    assert plan["free_bytes"] == 50 * 10**9
    assert plan["enough_space"] is True
    assert plan["terms"] == "https://www.iedb.org/"


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


@pytest.mark.parametrize("name", ["no-such-dataset", "hpa_bulk", "depmap"])
def test_preflight_refuses_what_fetch_cannot_do_automatically(data_dir, name):
    with pytest.raises(DatasetError) as raised:
        downloads.preflight(name)
    assert isinstance(raised.value, ValueError) and isinstance(raised.value, HitlistError)


def test_fetch_refuses_to_start_when_the_dataset_will_not_fit(data_dir, fake_download, monkeypatch):
    _free_space(monkeypatch, 1_000_000)
    with pytest.raises(DownloadError, match="Not enough free space for 'iedb'"):
        downloads.fetch("iedb")
    assert fake_download == []
    assert not (data_dir / "mhc_ligand_full.csv").exists()


def test_terms_notice_is_printed_before_the_download_starts(data_dir, monkeypatch, capsys):
    _free_space(monkeypatch, 50 * 10**9)
    seen_before_download = []

    def fake(url, dest, **kwargs):
        seen_before_download.append(capsys.readouterr().err)
        dest.write_bytes(b"x" * downloads.FETCHABLE_DATASETS["iedb"]["min_size_bytes"])
        return dest

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


def test_list_datasets_and_info_report_whether_the_file_exists(data_dir, fake_download):
    path = downloads.fetch("hpv16")
    assert downloads.list_datasets()["hpv16"]["exists"] is True
    assert downloads.info("hpv16")["exists"] is True
    path.unlink()
    assert downloads.list_datasets()["hpv16"]["exists"] is False
    assert downloads.info("hpv16")["exists"] is False
    assert downloads.info("ebv")["exists"] is False


def test_cli_dry_run_prints_the_plan_and_downloads_nothing(
    data_dir, fake_download, monkeypatch, capsys
):
    _free_space(monkeypatch, 50 * 10**9)
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "fetch", "iedb", "--dry-run"])
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan == downloads.preflight("iedb")
    assert fake_download == []


@pytest.mark.parametrize("verb", ["fetch", "refresh"])
def test_cli_reports_unknown_datasets_as_one_error_line(data_dir, monkeypatch, capsys, verb):
    monkeypatch.setattr(sys, "argv", ["hitlist", "data", verb, "no-such-dataset"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error: Unknown dataset 'no-such-dataset'")
    assert "Traceback" not in captured.err


def test_api_callers_can_still_catch_value_error(data_dir):
    with pytest.raises(ValueError, match="Unknown dataset"):
        downloads.fetch("no-such-dataset")
