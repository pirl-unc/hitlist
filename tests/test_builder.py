from hitlist.builder import _source_fingerprints
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
