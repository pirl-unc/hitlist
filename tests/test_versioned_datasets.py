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

import gzip
import io
import json
import zipfile

import pytest
import requests

from hitlist import downloads
from hitlist.downloads import (
    FETCHABLE_DATASETS,
    MANUAL_DATASETS,
    VersionedDatasetError,
    VersionedDatasetRegistry,
)

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

    def respond(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.raw = io.BytesIO(buf.getvalue())
        return response

    monkeypatch.setattr(requests, "get", respond)
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
    def respond(url, **kwargs):
        if counter is not None:
            counter["n"] += 1
        response = requests.Response()
        response.status_code = 200
        response.raw = io.BytesIO(content)
        return response

    monkeypatch.setattr(requests, "get", respond)


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
    def _boom(url, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(requests, "get", _boom)
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)
    with pytest.raises(VersionedDatasetError, match="failed to download"):
        reg.download("thing")


def test_legacy_cache_is_reused_without_mutation(tmp_path, monkeypatch, capsys):
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path)
    path = reg.local_path("thing")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old cache")
    receipt = {"thing": {"version": "v1", "bytes": 5, "downloaded_at": "2020-01-01"}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(receipt))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, manifest)}
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("network on cache hit"))
    assert reg.ensure("thing") == path
    assert capsys.readouterr().out == ""
    assert reg.download("thing") == path
    assert "already cached" in capsys.readouterr().out
    assert reg.status() == [
        {
            "name": "thing",
            "description": "A versioned thing",
            "default_version": "v2",
            "available_versions": ["v1", "v2"],
            "cached": True,
            "cached_version": "v1",
            "bytes": 5,
            "downloaded_at": "2020-01-01",
            "path": str(path),
        }
    ]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, manifest)}
    assert not list(tmp_path.glob(".*")), "read-only reuse created metadata/locks"


@pytest.mark.parametrize("method", ["download", "ensure"])
def test_dynamic_registry_root_is_resolved_once(tmp_path, monkeypatch, method):
    calls = []

    def root():
        calls.append(1)
        return tmp_path / str(len(calls))

    _stub_dl(monkeypatch)
    reg = VersionedDatasetRegistry(_datasets(), cache_dir=root)
    path = getattr(reg, method)("thing")
    assert calls == [1]
    assert path == tmp_path / "1/thing/v2/thing.tsv"
    assert json.loads((tmp_path / "1/manifest.json").read_text())["thing"]["path"] == str(path)


@pytest.mark.parametrize(
    "suffix, filename, expanded",
    [
        (".gz", "file.tsv", True),
        (".gz", "file.tsv.gz", False),
        (".gz?query=1", "file.tsv", False),
    ],
)
def test_registry_preserves_literal_decompression_policy(
    tmp_path, monkeypatch, suffix, filename, expanded
):
    payload = gzip.compress(b"reference")
    _stub_dl(monkeypatch, content=payload)
    reg = VersionedDatasetRegistry(
        {
            "data": {
                "filename": filename,
                "default_version": "v1",
                "urls": {"v1": "http://x/data" + suffix},
            }
        },
        cache_dir=lambda: tmp_path,
    )
    path = reg.download("data", verbose=False)
    assert path.read_bytes() == (b"reference" if expanded else payload)


def test_registry_failed_refresh_retains_legacy_bytes_and_custom_error(tmp_path, monkeypatch):
    class CustomError(VersionedDatasetError):
        pass

    reg = VersionedDatasetRegistry(_datasets(), cache_dir=lambda: tmp_path, error_cls=CustomError)
    path = reg.local_path("thing")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old cache")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"legacy": {}}')
    error = OSError("broken transport")

    def fail(*args, **kwargs):
        assert kwargs["timeout"] == 300.0
        raise error

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(CustomError) as raised:
        reg.download("thing", force=True, verbose=False)
    assert raised.value.__cause__ is error
    assert path.read_bytes() == b"old cache"
    assert manifest.read_text() == '{"legacy": {}}'
