"""The disk cache follows the current data directory unless explicitly set (#591)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hitlist import downloads, proteome


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("HITLIST_DATA_DIR", raising=False)
    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", "1")
    monkeypatch.setattr(downloads, "default_data_dir", lambda: tmp_path / "platform-cache")
    monkeypatch.setattr(downloads, "_override_data_dir", None)
    monkeypatch.setattr(downloads, "_data_dir_cache", {})
    monkeypatch.setattr(proteome, "_PROTEOME_INDEX_DISK_CACHE_DIR", None)
    proteome.set_disk_cache_dir(None)
    proteome.clear_fasta_index_cache()
    yield
    proteome.clear_fasta_index_cache()


def test_default_tracks_environment_and_session_settings(tmp_path, monkeypatch, capsys):
    assert proteome.proteome_index_cache_dir() == (
        downloads.default_data_dir() / "proteome_index_cache"
    )
    for name in ("first", "second"):
        monkeypatch.setenv("HITLIST_DATA_DIR", str(tmp_path / name))
        assert proteome.proteome_index_cache_dir() == tmp_path / name / "proteome_index_cache"
    downloads.set_data_dir(tmp_path / "session")
    assert proteome.proteome_index_cache_dir() == tmp_path / "session" / "proteome_index_cache"
    assert list(tmp_path.iterdir()) == [tmp_path / "home"]
    assert capsys.readouterr().out == ""


def test_explicit_cache_override_and_reset_are_dynamic(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    proteome.set_disk_cache_dir(explicit)
    downloads.set_data_dir(tmp_path / "data")
    assert proteome.proteome_index_cache_dir() == explicit
    proteome.set_disk_cache_dir(None)
    assert proteome.proteome_index_cache_dir() == tmp_path / "data" / "proteome_index_cache"
    downloads.set_data_dir(tmp_path / "later")
    assert proteome.proteome_index_cache_dir() == tmp_path / "later" / "proteome_index_cache"


def test_populated_legacy_cache_is_reused_until_explicit_relocation(tmp_path, monkeypatch):
    legacy = Path.home() / ".hitlist" / "proteome_index_cache"
    legacy.mkdir(parents=True)
    marker = legacy / "existing.pkl"
    marker.write_bytes(b"preserve")
    assert proteome.proteome_index_cache_dir() == legacy
    monkeypatch.setenv("HITLIST_DATA_DIR", str(tmp_path / "relocated"))
    assert proteome.proteome_index_cache_dir() == tmp_path / "relocated" / "proteome_index_cache"
    proteome.clear_disk_cache()
    assert marker.read_bytes() == b"preserve"


def test_disk_operations_use_the_current_data_directory(tmp_path, monkeypatch):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">sp|P|TEST\nACDEFGHIKLMNPQRSTVWY\n")
    roots = [tmp_path / name for name in ("first", "second")]
    for root in roots:
        downloads.set_data_dir(root)
        proteome.clear_fasta_index_cache()
        expected = root / "proteome_index_cache"
        original = proteome.ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
        assert len(list(expected.glob("*.pkl"))) == 1
        proteome.clear_fasta_index_cache()
        with patch.object(proteome.ProteomeIndex, "_build", side_effect=AssertionError("rebuilt")):
            cached = proteome.ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
        assert cached.proteins == original.proteins
        assert cached is not original

    # Eviction and clearing must not touch the previous root.
    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", "0.000000001")
    proteome._evict_disk_cache_if_over_cap()
    assert not list((roots[1] / "proteome_index_cache").glob("*.pkl"))
    assert len(list((roots[0] / "proteome_index_cache").glob("*.pkl"))) == 1
    partial = roots[1] / "proteome_index_cache" / "partial.tmp"
    partial.write_bytes(b"unfinished")
    proteome.clear_disk_cache()
    assert not partial.exists()
    assert len(list((roots[0] / "proteome_index_cache").glob("*.pkl"))) == 1
    assert not (Path.home() / ".hitlist").exists()
