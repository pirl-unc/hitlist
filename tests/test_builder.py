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
    """Cache validity now depends on source fingerprints, not the with_flanking flag.

    Under the v1.6.0 architecture, mappings live in a separate sidecar
    (peptide_mappings.parquet), so the observations cache stays valid
    regardless of whether mappings were built.  The with_flanking arg is
    retained on _cache_is_valid for backward compat but no longer changes
    the result.
    """
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    fake_parquet = tmp_path / "observations.parquet"
    fake_parquet.write_bytes(b"fake parquet")
    _meta_path().write_text(
        json.dumps(
            {
                "sources": {},
                "n_rows": 100,
                "n_peptides": 50,
                "n_alleles": 10,
                "n_species": 1,
                "with_flanking": False,
            }
        )
    )
    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})

    assert _cache_is_valid({}, with_flanking=False) is True
    assert _cache_is_valid({}, with_flanking=True) is True  # no longer invalidates
