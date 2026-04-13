import json

from hitlist.builder import _cache_is_valid, _meta_path, _source_fingerprints
from hitlist.supplement import load_supplementary_manifest


def test_source_fingerprints_includes_supplementary_csvs(tmp_path):
    """Cache fingerprint should include every supplementary CSV, not just the manifest."""
    # Use a minimal dummy source path dict
    fp = _source_fingerprints({})
    manifest = load_supplementary_manifest()
    for entry in manifest:
        key = f"supplementary_csv:{entry['file']}"
        assert key in fp, f"Missing fingerprint for supplementary CSV: {entry['file']}"
        assert "size" in fp[key]
        assert "mtime" in fp[key]


def test_source_fingerprints_includes_manifest():
    """Cache fingerprint should include the supplementary manifest itself."""
    fp = _source_fingerprints({})
    assert "supplementary_manifest" in fp


def test_cache_invalid_when_flanking_requested_but_absent(tmp_path, monkeypatch):
    """Asking for --with-flanking should invalidate a cache built without it."""
    from hitlist import builder, downloads

    downloads.set_data_dir(tmp_path)

    # Simulate a pre-built observations table without flanking
    fake_parquet = tmp_path / "observations.parquet"
    fake_parquet.write_bytes(b"fake parquet")
    meta = {
        "sources": {},
        "n_rows": 100,
        "n_peptides": 50,
        "n_alleles": 10,
        "n_species": 1,
        "with_flanking": False,
    }
    _meta_path().write_text(json.dumps(meta))

    # Patch _source_fingerprints so the source check passes
    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})

    # Cache is valid when flanking is NOT requested
    assert _cache_is_valid({}, with_flanking=False) is True
    # ...but invalid when flanking IS requested (forces rebuild)
    assert _cache_is_valid({}, with_flanking=True) is False
