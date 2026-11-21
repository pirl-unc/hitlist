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

"""Uniform dataset discovery, preflight, fetch, info, path, remove and CLI coverage (v2.4.0)."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from collections import namedtuple
from pathlib import Path

import pytest

import hitlist
from hitlist import cli, downloads, drug_dm_qualification, netmhcstabpan
from hitlist.errors import DownloadError, InsufficientSpaceError, IntegrityError

Usage = namedtuple("usage", ["total", "used", "free"])

EXPECTED_MANAGED_DATASETS = (
    "uniprot/human",
    "uniprot/human/2026_03",
    "uniprot/human/2015_10",
    "hla-ligand-atlas",
    "hla-ligand-atlas-crosswalk",
    "tcell-assays",
    "perturbation-quantities",
    "dlbcl-35561310-supplements",
    "canine-bulk/pfizer_2026_03",
    "canine-bulk/pfizer_2026_03/table",
    "canine-bulk/pfizer_2026_03/reference",
    "canine-bulk/labrador_2024_01",
    "canine-bulk/labrador_2024_01/table",
    "canine-bulk/labrador_2024_01/reference",
    "pxd013455",
    "pxd013455-sources",
)


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    asset_root = tmp_path / "assets"
    monkeypatch.setattr(downloads, "_override_data_dir", data_root)
    monkeypatch.setattr(downloads, "_data_dir_cache", {})
    monkeypatch.setattr(downloads, "data_asset_dir", lambda: asset_root)
    return data_root, asset_root


def _free_space(monkeypatch, free_bytes: int) -> None:
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: Usage(10**13, 10**13 - free_bytes, free_bytes)
    )


def test_every_managed_dataset_is_discoverable_preflightable_and_inspectable(
    isolated_data_dir, monkeypatch
):
    data_root, _ = isolated_data_dir
    _free_space(monkeypatch, 100 * 10**9)

    avail = downloads.available_datasets()
    for name in EXPECTED_MANAGED_DATASETS:
        assert name in avail, f"{name} missing from available_datasets()"
        plan = downloads.preflight(name)
        assert plan["name"] == name
        assert plan["exists"] is False
        assert plan["would_download"] is True
        assert plan["expected_size_bytes"] and plan["expected_size_bytes"] > 0
        assert plan["enough_space"] is True
        meta = downloads.info(name)
        assert meta["exists"] is False
        assert meta["name"] == name
        with pytest.raises(FileNotFoundError, match=f"hitlist data fetch {name}"):
            downloads.get_path(name)

    # Read-only preflight/info must never create the uninitialized data directory.
    assert not data_root.exists()


def test_managed_fetch_enforces_space_preflight_before_network(isolated_data_dir, monkeypatch):
    _free_space(monkeypatch, 1024)
    called = []
    monkeypatch.setattr("hitlist.atlas.atlas_sources", lambda **kwargs: called.append(kwargs))
    with pytest.raises(InsufficientSpaceError) as raised:
        downloads.fetch("hla-ligand-atlas")
    assert raised.value.required_bytes > 1024
    assert raised.value.available_bytes == 1024
    assert called == []


def test_managed_fetch_wraps_raw_errors_as_hitlist_errors(isolated_data_dir, monkeypatch):
    _free_space(monkeypatch, 100 * 10**9)
    import datacache

    monkeypatch.setattr(
        "hitlist.atlas.atlas_sources",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad remote state")),
    )
    with pytest.raises(DownloadError, match="Failed to fetch 'hla-ligand-atlas'"):
        downloads.fetch("hla-ligand-atlas")

    monkeypatch.setattr(
        "hitlist.atlas.atlas_sources",
        lambda **kwargs: (_ for _ in ()).throw(
            datacache.FileValidationError("/tmp/peptides.tsv.gz", "sha mismatch")
        ),
    )
    with pytest.raises(IntegrityError, match="Failed to fetch 'hla-ligand-atlas'"):
        downloads.fetch("hla-ligand-atlas")


def test_netmhcstabpan_default_fetch_uses_registered_downloads_path(isolated_data_dir, monkeypatch):
    data_root, _ = isolated_data_dir
    _free_space(monkeypatch, 100 * 10**9)
    payload = b"HLA\tPep\tThalf\nHLA-A*02:01 SLYNTVATL 2.00\n"
    monkeypatch.setitem(downloads.FETCHABLE_DATASETS["netmhcstabpan"], "min_size_bytes", 1)
    monkeypatch.setitem(
        downloads.FETCHABLE_DATASETS["netmhcstabpan"], "expected_size", len(payload)
    )
    monkeypatch.setitem(
        downloads.FETCHABLE_DATASETS["netmhcstabpan"],
        "expected_sha256",
        hashlib.sha256(payload).hexdigest(),
    )

    def _fake_dl(url, dest, **kwargs):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(payload)
        return Path(dest)

    monkeypatch.setattr(downloads, "download_to_file", _fake_dl)

    fetched = netmhcstabpan.fetch_netmhcstabpan_source()
    assert fetched == data_root / "netmhcstabpan-1.0-Stability.txt"
    assert downloads.get_path("netmhcstabpan") == fetched
    assert downloads.info("netmhcstabpan")["exists"] is True


def test_dlbcl_supplements_uniform_fetch_path_inventory_and_remove(
    isolated_data_dir, monkeypatch, capsys
):
    data_root, _ = isolated_data_dir
    _free_space(monkeypatch, 100 * 10**9)

    b3 = b"fake-xlsx-table-3-payload"
    b5 = b"fake-xlsx-table-5-payload"
    fake_contract = {
        "doi": "10.1182/bloodadvances.2021006069",
        "workbooks": {
            "table_3_class_i": {
                "filename": "advancesADV2021006069-suppl3.xlsx",
                "source_url": "https://example.org/PMC9327544/supplementaryFiles",
                "sha256": hashlib.sha256(b3).hexdigest(),
                "size_bytes": len(b3),
            },
            "table_5_class_ii": {
                "filename": "advancesADV2021006069-suppl5.xlsx",
                "source_url": "https://example.org/PMC9327544/supplementaryFiles",
                "sha256": hashlib.sha256(b5).hexdigest(),
                "size_bytes": len(b5),
            },
        },
    }
    monkeypatch.setattr(drug_dm_qualification, "dlbcl_supplement_contract", lambda: fake_contract)

    def _fake_download(url, dest, **kwargs):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("nested/advancesADV2021006069-suppl3.xlsx", b3)
            zf.writestr("nested/advancesADV2021006069-suppl5.xlsx", b5)
        return Path(dest)

    monkeypatch.setattr(downloads, "download_to_file", _fake_download)

    bundle_dir = downloads.fetch("dlbcl-35561310-supplements")
    assert bundle_dir == data_root / "dlbcl_35561310_supplements"
    assert downloads.get_path("dlbcl-35561310-supplements") == bundle_dir
    assert downloads.info("dlbcl-35561310-supplements")["exists"] is True
    assert "terms-of-use-governed source" in capsys.readouterr().err

    # Installed managed bundle files appear in cache_inventory under their canonical name.
    rows = downloads.list_cache_files(include_unregistered=False)
    dlbcl_rows = [r for r in rows if "dlbcl-35561310-supplements" in r["names"]]
    assert len(dlbcl_rows) == 2
    assert all(r["status"] == "available" for r in dlbcl_rows)

    # remove(..., delete_file=True) cleans up the managed bundle cleanly.
    assert downloads.remove("dlbcl-35561310-supplements", delete_file=True) is True
    assert downloads.info("dlbcl-35561310-supplements")["exists"] is False
    assert downloads.remove("dlbcl-35561310-supplements", delete_file=True) is False


def test_remove_supports_mirrored_data_assets(isolated_data_dir):
    _, asset_root = isolated_data_dir
    asset_name = sorted(downloads.data_assets())[0]
    asset_path = asset_root / asset_name
    assert downloads.remove(asset_name, delete_file=True) is False
    asset_root.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"cached-asset")
    assert downloads.remove(asset_name, delete_file=True) is True
    assert not asset_path.exists()


def test_cli_data_preflight_and_available_json(isolated_data_dir, monkeypatch, capsys):
    _free_space(monkeypatch, 100 * 10**9)

    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "preflight", "hla-ligand-atlas"])
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan == downloads.preflight("hla-ligand-atlas")

    monkeypatch.setattr(sys, "argv", ["hitlist", "data", "available", "--json"])
    cli.main()
    rows = json.loads(capsys.readouterr().out)
    names = {r["name"] for r in rows}
    assert {"iedb", "depmap", "hla-ligand-atlas", "tcell-assays", "uniprot/human"} <= names


def test_top_level_download_api_is_symmetric():
    assert hitlist.data_dir is downloads.data_dir
    assert hitlist.data_asset_dir is downloads.data_asset_dir
    assert hitlist.set_data_dir is downloads.set_data_dir
    assert hitlist.data_assets is downloads.data_assets
    assert hitlist.fetch_data_asset is downloads.fetch_data_asset
    assert hitlist.fetch_dlbcl_supplements is drug_dm_qualification.fetch_dlbcl_supplements
