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


def test_cache_valid_when_sources_unchanged(tmp_path, monkeypatch):
    """Cache validity requires both observations.parquet and binding.parquet.

    Since 1.7.0 the build writes two sibling parquets; if either is
    missing the cache is invalid so older installs rebuild once on
    upgrade.  ``with_flanking`` is retained on the signature for
    backward compat but no longer changes the result.
    """
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    (tmp_path / "observations.parquet").write_bytes(b"fake parquet")
    (tmp_path / "binding.parquet").write_bytes(b"fake parquet")
    _meta_path().write_text(
        json.dumps(
            {
                "sources": {},
                "n_rows": 100,
                "n_peptides": 50,
                "n_alleles": 10,
                "n_species": 1,
                "n_binding_rows": 20,
                "with_flanking": False,
            }
        )
    )
    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})

    assert _cache_is_valid({}, with_flanking=False) is True
    assert _cache_is_valid({}, with_flanking=True) is True  # no longer invalidates


def test_cache_invalid_when_binding_parquet_missing(tmp_path, monkeypatch):
    """Missing binding.parquet alone should invalidate the cache."""
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    (tmp_path / "observations.parquet").write_bytes(b"fake parquet")
    # Intentionally no binding.parquet
    _meta_path().write_text(json.dumps({"sources": {}, "n_rows": 100}))
    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})

    assert _cache_is_valid({}, with_flanking=False) is False
