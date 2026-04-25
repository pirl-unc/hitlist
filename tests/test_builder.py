import json

import pandas as pd

from hitlist.builder import (
    _atomic_write_parquet,
    _cache_is_valid,
    _drop_short_mhc2_rows,
    _meta_path,
    _source_fingerprints,
)
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
    """Cache validity requires all four sibling parquets.

    Since 1.16.0 the build also writes line_expression.parquet; if any
    of the four is missing the cache is invalid so older installs
    rebuild once on upgrade.  ``with_flanking`` is retained on the
    signature for backward compat but no longer changes the result.
    """
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    (tmp_path / "observations.parquet").write_bytes(b"fake parquet")
    (tmp_path / "binding.parquet").write_bytes(b"fake parquet")
    (tmp_path / "bulk_proteomics.parquet").write_bytes(b"fake parquet")
    (tmp_path / "line_expression.parquet").write_bytes(b"fake parquet")
    _meta_path().write_text(
        json.dumps(
            {
                "sources": {},
                "n_rows": 100,
                "n_peptides": 50,
                "n_alleles": 10,
                "n_species": 1,
                "n_binding_rows": 20,
                "n_bulk_rows": 10,
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


def test_cache_invalid_when_observations_parquet_missing(tmp_path, monkeypatch):
    """Missing observations.parquet alone should also invalidate the cache."""
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    (tmp_path / "binding.parquet").write_bytes(b"fake parquet")
    _meta_path().write_text(json.dumps({"sources": {}, "n_rows": 0}))
    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})

    assert _cache_is_valid({}, with_flanking=False) is False


def test_cache_invalid_when_parquet_fingerprint_changes(tmp_path, monkeypatch):
    """If a parquet is replaced (size/mtime changes) after the meta was
    written, the cache must invalidate even if source CSVs look unchanged.
    """
    import os
    import time

    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    obs_p = tmp_path / "observations.parquet"
    bind_p = tmp_path / "binding.parquet"
    bulk_p = tmp_path / "bulk_proteomics.parquet"
    le_p = tmp_path / "line_expression.parquet"
    obs_p.write_bytes(b"original observations")
    bind_p.write_bytes(b"original binding")
    bulk_p.write_bytes(b"original bulk")
    le_p.write_bytes(b"original line expression")

    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})
    _meta_path().write_text(
        json.dumps(
            {
                "sources": {},
                "parquets": builder._parquet_fingerprints(),
            }
        )
    )
    assert _cache_is_valid({}, with_flanking=False) is True

    # Replace observations.parquet — bump mtime past meta's record
    time.sleep(0.01)
    obs_p.write_bytes(b"mutated observations content that is longer")
    os.utime(obs_p, None)
    assert _cache_is_valid({}, with_flanking=False) is False


def test_drop_short_mhc2_rows_removes_subset(capsys):
    """8-11 aa peptides labeled mhc_class="II" are dropped (#122)."""
    df = pd.DataFrame(
        {
            "peptide": ["SLLQHLIGL", "PKYVKQNTLKLAT", "KQNTLKL", "SAMPLEPEPTIDE"],
            "mhc_class": ["II", "II", "II", "II"],
            "pmid": [33858848, 12345678, 33858848, 12345678],
        }
    )
    out = _drop_short_mhc2_rows(df, "MS observations")

    assert list(out["peptide"]) == ["PKYVKQNTLKLAT", "SAMPLEPEPTIDE"]
    captured = capsys.readouterr().out
    assert "Dropped 2" in captured
    assert "#122" in captured
    assert "PMID 33858848" in captured


def test_drop_short_mhc2_rows_preserves_class_i():
    """Class-I rows (any length) and long class-II rows are untouched."""
    df = pd.DataFrame(
        {
            "peptide": ["SLLQHLIGL", "AAAAAA", "LONGCLASSIIPEPTIDE"],
            "mhc_class": ["I", "I", "II"],
            "pmid": [1, 2, 3],
        }
    )
    out = _drop_short_mhc2_rows(df, "MS observations")
    assert len(out) == 3


def test_drop_short_mhc2_rows_empty_frame():
    """An empty input returns an empty output without error."""
    df = pd.DataFrame({"peptide": [], "mhc_class": [], "pmid": []})
    out = _drop_short_mhc2_rows(df, "MS observations")
    assert out.empty


def test_drop_short_mhc2_rows_missing_columns():
    """A frame without mhc_class/peptide columns is returned unchanged."""
    df = pd.DataFrame({"other": [1, 2, 3]})
    out = _drop_short_mhc2_rows(df, "MS observations")
    assert len(out) == 3


def test_atomic_write_parquet_replaces_existing(tmp_path):
    """A subsequent write overwrites the prior file in one rename."""
    import pyarrow.parquet as pq

    path = tmp_path / "observations.parquet"
    first = pd.DataFrame({"peptide": ["AAA"], "gene_names": [""]})
    _atomic_write_parquet(first, path)
    assert path.exists()
    assert "gene_names" in set(pq.read_schema(path).names)

    second = pd.DataFrame(
        {"peptide": ["AAA", "BBB"], "gene_names": ["HER2", "PRAME"], "extra": [1, 2]}
    )
    _atomic_write_parquet(second, path)

    assert not path.with_suffix(".parquet.partial").exists()
    back = pd.read_parquet(path)
    assert set(back.columns) == {"peptide", "gene_names", "extra"}
    assert len(back) == 2


def test_atomic_write_parquet_no_partial_leftover(tmp_path):
    """``.partial`` must not remain after a successful atomic write."""
    path = tmp_path / "binding.parquet"
    _atomic_write_parquet(pd.DataFrame({"peptide": ["X"]}), path)
    assert path.exists()
    assert not path.with_suffix(".parquet.partial").exists()


# ── Line-expression cache fingerprints (issue #150) ────────────────────────


def test_source_fingerprints_includes_line_expression_anchors():
    """Cache fingerprint must cover line_expression_anchors.yaml so curation
    edits invalidate the build cache (issue #150).
    """
    fp = _source_fingerprints({})
    assert "line_expression_anchors" in fp


def test_source_fingerprints_includes_line_expression_sources_yaml():
    fp = _source_fingerprints({})
    assert "line_expression_sources" in fp


def test_source_fingerprints_includes_packaged_line_expression_csvs():
    """Every packaged CSV under hitlist/data/line_expression/ should be
    fingerprinted.  GM12878 ships with the repo, so its key must appear.
    """
    fp = _source_fingerprints({})
    csv_keys = [k for k in fp if k.startswith("line_expression_csv:")]
    assert any("gm12878" in k.lower() for k in csv_keys), (
        "expected packaged GM12878 CSV to be fingerprinted"
    )


def test_source_fingerprints_includes_registered_depmap_inputs(tmp_path, monkeypatch):
    """A registered DepMap input must appear in the cache fingerprint so
    a re-registered file forces the next build to rebuild line_expression.parquet.
    """
    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    fake = tmp_path / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    fake.write_text("ModelID,TP53 (7157)\nACH-001,6.0\n")
    downloads.register("depmap_rna", fake)

    fp = _source_fingerprints({})
    assert "line_expression_input:depmap_rna" in fp
    entry = fp["line_expression_input:depmap_rna"]
    assert entry["path"] == str(fake)
    assert entry["size"] > 0


def test_parquet_fingerprints_includes_line_expression(tmp_path, monkeypatch):
    """The parquet-fingerprint set must include line_expression.parquet so a
    swapped parquet invalidates the cache.
    """
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    le = tmp_path / "line_expression.parquet"
    le.write_bytes(b"fake")
    fp = builder._parquet_fingerprints()
    assert "line_expression" in fp


def test_cache_invalid_when_line_expression_parquet_missing(tmp_path, monkeypatch):
    """Missing line_expression.parquet alone should invalidate the cache."""
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    (tmp_path / "observations.parquet").write_bytes(b"fake parquet")
    (tmp_path / "binding.parquet").write_bytes(b"fake parquet")
    (tmp_path / "bulk_proteomics.parquet").write_bytes(b"fake parquet")
    # Intentionally no line_expression.parquet
    _meta_path().write_text(json.dumps({"sources": {}, "n_rows": 100}))
    monkeypatch.setattr(builder, "_source_fingerprints", lambda paths: {})

    assert _cache_is_valid({}, with_flanking=False) is False
