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

"""Tests for the timeout/retry download helper in ``hitlist.downloads`` (#255)."""

from __future__ import annotations

import io
import socket
import urllib.error
from contextlib import contextmanager

import pytest

from hitlist import downloads


@contextmanager
def _fake_response(payload: bytes):
    """Mimic the context-manager object returned by urlopen()."""
    yield io.BytesIO(payload)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Never actually sleep during retry backoff in tests."""
    monkeypatch.setattr(downloads.time, "sleep", lambda _s: None)


def test_download_to_file_success(tmp_path, monkeypatch):
    dest = tmp_path / "out.fasta"
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda url, timeout=None: _fake_response(b">sp|P1\nACDEF\n"),
    )

    downloads._download_to_file("http://example/x", dest, label="x", verbose=False)

    assert dest.read_bytes() == b">sp|P1\nACDEF\n"
    # No leftover temp file.
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()


def test_download_to_file_retries_then_succeeds(tmp_path, monkeypatch):
    dest = tmp_path / "out.fasta"
    monkeypatch.setattr(downloads, "_DOWNLOAD_RETRY_BACKOFF", (0.0, 0.0))
    attempts = {"n": 0}

    def flaky_urlopen(url, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise socket.timeout("stalled")
        return _fake_response(b"DATA")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", flaky_urlopen)

    downloads._download_to_file("http://example/x", dest, verbose=False)

    assert attempts["n"] == 2
    assert dest.read_bytes() == b"DATA"


def test_download_to_file_exhausts_retries(tmp_path, monkeypatch):
    dest = tmp_path / "out.fasta"
    monkeypatch.setattr(downloads, "_DOWNLOAD_RETRY_BACKOFF", (0.0, 0.0))

    def always_timeout(url, timeout=None):
        raise socket.timeout("stalled")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", always_timeout)

    with pytest.raises(RuntimeError, match="Failed to download"):
        downloads._download_to_file("http://example/x", dest, verbose=False)

    # A failed download must not leave a partial file or temp behind.
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()


def test_download_to_file_does_not_retry_on_404(tmp_path, monkeypatch):
    """A permanent 4xx must fail fast — no retries, no backoff sleeps."""
    dest = tmp_path / "out.fasta"
    monkeypatch.setattr(downloads, "_DOWNLOAD_RETRY_BACKOFF", (0.0, 0.0))
    sleeps: list[float] = []
    monkeypatch.setattr(downloads.time, "sleep", lambda s: sleeps.append(s))
    attempts = {"n": 0}

    def not_found(url, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(downloads.urllib.request, "urlopen", not_found)

    with pytest.raises(RuntimeError, match="Failed to download"):
        downloads._download_to_file("http://example/x", dest, verbose=False)

    assert attempts["n"] == 1  # single attempt, no retry
    assert sleeps == []  # never slept


def test_download_to_file_retries_on_500(tmp_path, monkeypatch):
    """A 5xx is transient and should be retried."""
    dest = tmp_path / "out.fasta"
    monkeypatch.setattr(downloads, "_DOWNLOAD_RETRY_BACKOFF", (0.0,))
    attempts = {"n": 0}

    def flaky(url, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
        return _fake_response(b"OK")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", flaky)

    downloads._download_to_file("http://example/x", dest, verbose=False)
    assert attempts["n"] == 2
    assert dest.read_bytes() == b"OK"


def test_download_rejects_empty_body(tmp_path, monkeypatch):
    """A clean 200 with an empty body (e.g. a withdrawn proteome) must NOT be
    cached as a valid 0-byte file — it should fail (after retries) instead."""
    dest = tmp_path / "out.fasta"
    monkeypatch.setattr(downloads, "_DOWNLOAD_RETRY_BACKOFF", (0.0,))
    monkeypatch.setattr(
        downloads.urllib.request, "urlopen", lambda url, timeout=None: _fake_response(b"")
    )
    with pytest.raises(RuntimeError, match="Failed to download"):
        downloads._download_to_file("http://example/x", dest, verbose=False)
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()


def test_remove_reports_whether_dataset_existed(tmp_path):
    """remove() must return False for an unregistered name so the CLI can warn
    on a typo instead of falsely reporting a successful removal."""
    downloads.set_data_dir(tmp_path)
    try:
        assert downloads.remove("never-registered-xyz") is False

        f = tmp_path / "d.csv"
        f.write_text("x")
        downloads.register("d", f, "desc")
        assert downloads.remove("d") is True
        assert downloads.remove("d") is False  # already gone
    finally:
        downloads._override_data_dir = None


def test_uniprot_transient_error_does_not_cache_negative(monkeypatch):
    """A transient UniProt failure must not be cached as a permanent
    ``not_found`` — otherwise the organism is excluded from every later build."""
    saved: list = []
    monkeypatch.setattr(downloads, "_uniprot_cache", lambda: {})
    monkeypatch.setattr(
        downloads, "_save_uniprot_cache_entry", lambda org, entry: saved.append((org, entry))
    )

    def boom(org, timeout=15):
        raise OSError("uniprot down")

    monkeypatch.setattr(downloads, "resolve_proteome_via_uniprot", boom)

    out = downloads.lookup_proteome("Nonexistent rare organism xyz", use_uniprot=True)
    assert out is None
    assert saved == []  # no negative cached -> a later run retries


def test_download_uses_fixed_socket_timeout(tmp_path, monkeypatch):
    dest = tmp_path / "out.fasta"
    seen = {}

    def capture(url, timeout=None):
        seen["timeout"] = timeout
        return _fake_response(b"X")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", capture)

    downloads._download_to_file("http://example/x", dest, verbose=False)
    assert seen["timeout"] == downloads._DOWNLOAD_SOCKET_TIMEOUT


def test_uniprot_lookup_offline_reuses_cached_resolution(monkeypatch):
    monkeypatch.setattr(
        downloads,
        "_uniprot_cache",
        lambda: {
            "Pteropus alecto": {
                "proteome_id": "UP000031014",
                "scientific_name": "Pteropus alecto",
            }
        },
    )
    monkeypatch.setattr(
        downloads,
        "resolve_proteome_via_uniprot",
        lambda *_a, **_kw: pytest.fail("offline lookup must not contact UniProt"),
    )

    assert (
        downloads.lookup_proteome("Uncached organism xyz", use_uniprot=True, allow_network=False)
        is None
    )

    out = downloads.lookup_proteome("Pteropus alecto", use_uniprot=True, allow_network=False)

    assert out is not None
    assert out["proteome_id"] == "UP000031014"


def test_uniprot_lookup_offline_does_not_resolve_uncached_organism(monkeypatch):
    monkeypatch.setattr(downloads, "_uniprot_cache", lambda: {})
    monkeypatch.setattr(
        downloads,
        "resolve_proteome_via_uniprot",
        lambda *_a, **_kw: pytest.fail("offline lookup must not contact UniProt"),
    )

    assert (
        downloads.lookup_proteome(
            "Uncached organism xyz",
            use_uniprot=True,
            allow_network=False,
        )
        is None
    )


def test_fetch_by_upid_offline_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    monkeypatch.setattr(
        downloads,
        "_download_to_file",
        lambda *_a, **_kw: pytest.fail("offline UPID fetch must not download"),
    )

    result = downloads.fetch_proteome_by_upid(
        "UP000000001",
        label="offline",
        verbose=False,
        fetch_missing=False,
    )

    assert result is None


def test_manifest_atomic_write_and_corruption_tolerance(tmp_path, monkeypatch):
    """#331: _save_manifest writes atomically and _load_manifest tolerates a
    corrupt/empty manifest (regenerable cache) instead of crashing the build."""
    monkeypatch.setattr(downloads, "_manifest_path", lambda: tmp_path / "manifest.json")

    # Round-trips.
    downloads._save_manifest({"datasets": {"x": {"file": "x.csv"}}})
    assert downloads._load_manifest()["datasets"]["x"]["file"] == "x.csv"

    # Atomic write leaves no stray temp files behind.
    assert not list(tmp_path.glob(".manifest-*.tmp"))

    # A truncated/empty manifest (the race symptom) reads as empty, not a crash.
    (tmp_path / "manifest.json").write_text("")
    assert downloads._load_manifest() == {"datasets": {}}

    # Garbage is tolerated too.
    (tmp_path / "manifest.json").write_text("{not json")
    assert downloads._load_manifest() == {"datasets": {}}
