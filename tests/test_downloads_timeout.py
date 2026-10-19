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

import pytest

from hitlist import downloads


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
        "download_to_file",
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
