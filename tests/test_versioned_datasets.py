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

"""Tests for IEDB/CEDAR auto-fetch and the reusable VersionedDatasetRegistry."""

from __future__ import annotations

import io
import zipfile
from contextlib import contextmanager

import pytest

from hitlist import downloads
from hitlist.downloads import (
    FETCHABLE_DATASETS,
    MANUAL_DATASETS,
    VersionedDatasetError,
    VersionedDatasetRegistry,
)


@contextmanager
def _fake_response(payload: bytes):
    yield io.BytesIO(payload)


# ── IEDB / CEDAR are now auto-fetchable ───────────────────────────────────────


def test_iedb_cedar_are_fetchable_not_manual():
    for name in ("iedb", "cedar"):
        assert name in FETCHABLE_DATASETS, f"{name} should be auto-fetchable"
        assert name not in MANUAL_DATASETS, f"{name} should no longer be manual"
        spec = FETCHABLE_DATASETS[name]
        assert spec["url"].endswith(".zip")
        assert spec["filename"].endswith(".csv")
        assert spec["terms"], "fetchable ToU-governed dataset needs a terms URL"


def test_fetch_iedb_streams_unzips_and_notes_terms(tmp_path, monkeypatch, capsys):
    # Serve a zip (as the downloader.php endpoint does) and confirm fetch()
    # unzips it to the CSV and prints the terms notice.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mhc_ligand_full_single_file.csv", b"peptide,allele\nSIINFEKL,H2-Kb\n")
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda url, timeout=None: _fake_response(buf.getvalue()),
    )
    downloads.set_data_dir(tmp_path)
    try:
        path = downloads.fetch("iedb")
    finally:
        downloads._override_data_dir = None

    assert path.name == "mhc_ligand_full.csv"
    assert path.read_bytes() == b"peptide,allele\nSIINFEKL,H2-Kb\n"
    err = capsys.readouterr().err
    assert "terms" in err.lower() and "iedb.org" in err


# ── VersionedDatasetRegistry ──────────────────────────────────────────────────


def _datasets():
    return {
        "thing": {
            "filename": "thing.tsv",
            "urls": {"v1": "https://x/thing.v1.tsv", "v2": "https://x/thing.v2.tsv"},
            "default_version": "v2",
            "description": "A versioned thing",
        },
    }


def _stub_dl(monkeypatch, content=b"DATA", counter=None):
    def _impl(url, dest, *, label="", verbose=True, force=False, decompress=False):
        from pathlib import Path

        dest = Path(dest)
        if dest.exists() and not force:
            return dest
        if counter is not None:
            counter["n"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", _impl)


def test_resolve_version_default_and_errors(tmp_path):
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)
    assert reg.resolve_version("thing") == "v2"  # default
    assert reg.resolve_version("thing", "v1") == "v1"
    with pytest.raises(VersionedDatasetError):
        reg.resolve_version("thing", "v99")
    with pytest.raises(VersionedDatasetError):
        reg.resolve_version("nope")


def test_download_writes_file_and_manifest(tmp_path, monkeypatch):
    _stub_dl(monkeypatch, content=b"col\tval\n")
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)

    path = reg.download("thing", "v1")
    assert path == tmp_path / "thing" / "v1" / "thing.tsv"
    assert path.read_bytes() == b"col\tval\n"

    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    rec = manifest["thing"]
    assert rec["version"] == "v1"
    assert rec["bytes"] == path.stat().st_size
    assert len(rec["sha256"]) == 64
    assert rec["url"].endswith("thing.v1.tsv")


def test_manifest_write_is_atomic(tmp_path, monkeypatch):
    """The registry manifest must be written atomically (temp + os.replace) like
    the module-level _save_manifest — no stray temp file, and a clean round-trip.
    A direct write_text would risk truncating manifest.json and losing all
    provenance on an interrupted/concurrent write."""
    import json

    _stub_dl(monkeypatch)
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)
    reg.download("thing")

    assert json.loads((tmp_path / "manifest.json").read_text())["thing"]["version"] == "v2"
    assert not list(tmp_path.glob(".manifest-*.tmp")), "atomic write left a stray temp file"


def test_cache_hit_skips_redownload(tmp_path, monkeypatch):
    counter = {"n": 0}
    _stub_dl(monkeypatch, counter=counter)
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)

    reg.download("thing")
    reg.ensure("thing")  # already cached -> no new fetch
    assert counter["n"] == 1
    reg.download("thing", force=True)
    assert counter["n"] == 2


def test_status_shape(tmp_path, monkeypatch):
    _stub_dl(monkeypatch)
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)

    before = {r["name"]: r for r in reg.status()}
    assert before["thing"]["cached"] is False
    assert before["thing"]["default_version"] == "v2"
    assert before["thing"]["available_versions"] == ["v1", "v2"]

    reg.download("thing")  # default v2
    after = {r["name"]: r for r in reg.status()}
    assert after["thing"]["cached"] is True
    assert after["thing"]["cached_version"] == "v2"


def test_custom_error_cls(tmp_path):
    class MyError(VersionedDatasetError):
        pass

    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path, error_cls=MyError)
    with pytest.raises(MyError):
        reg.resolve_version("nope")


def test_download_failure_wrapped(tmp_path, monkeypatch):
    def _boom(url, dest, *, label="", verbose=True, force=False, decompress=False):
        raise OSError("network down")

    monkeypatch.setattr(downloads, "download_to_file", _boom)
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)
    with pytest.raises(VersionedDatasetError, match="failed to download"):
        reg.download("thing")
