import json

import pandas as pd
import pytest

from hitlist.builder import (
    _CATEGORICAL_BUILD_COLUMNS,
    _atomic_write_parquet,
    _cache_is_valid,
    _compress_categoricals,
    _drop_duplicate_iris,
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


# ── _compress_categoricals (memory reduction in build_observations) ──────


def _full_obs_fixture(n_rows: int = 50) -> pd.DataFrame:
    """Synthetic obs frame containing every column ``_compress_categoricals``
    targets, plus high-cardinality / numeric columns it must NOT touch."""
    return pd.DataFrame(
        {
            "source": ["iedb"] * n_rows,
            "mhc_class": (["I"] * (n_rows // 2)) + (["II"] * (n_rows - n_rows // 2)),
            "mhc_species": ["Homo sapiens"] * n_rows,
            "mhc_restriction": ["HLA-A*02:01"] * n_rows,
            "mhc_allele_provenance": ["exact"] * n_rows,
            "allele_resolution": ["four_digit"] * n_rows,
            "serotype": ["A2"] * n_rows,
            "host": ["Donor-1"] * n_rows,
            "host_age": [""] * n_rows,
            "process_type": ["natural"] * n_rows,
            "disease": ["healthy"] * n_rows,
            "disease_stage": [""] * n_rows,
            "source_tissue": ["PBMC"] * n_rows,
            "cell_name": ["Line-1"] * n_rows,
            "culture_condition": ["unperturbed"] * n_rows,
            "assay_method": ["mass spectrometry"] * n_rows,
            "response_measured": [""] * n_rows,
            "measurement_units": [""] * n_rows,
            "measurement_inequality": [""] * n_rows,
            "qualitative_measurement": [""] * n_rows,
            "species": ["Homo sapiens"] * n_rows,
            "source_organism": ["Homo sapiens"] * n_rows,
            # NOT in the categorical allowlist — must stay object / numeric:
            "peptide": [f"PEP{i:08d}" for i in range(n_rows)],
            "assay_iri": [f"iedb:{i}" for i in range(n_rows)],
            "pmid": pd.array(list(range(n_rows)), dtype="Int64"),
            "mhc_allele_set_size": list(range(n_rows)),
        }
    )


def test_compress_categoricals_targets_low_cardinality_only():
    """Every column in the allowlist becomes ``category``; high-cardinality
    (peptide, assay_iri) and numeric (pmid, mhc_allele_set_size) preserved."""
    df = _full_obs_fixture()
    _compress_categoricals(df)
    for col in _CATEGORICAL_BUILD_COLUMNS:
        assert df[col].dtype.name == "category", f"{col} not categorical"
    # High-cardinality unique-per-row columns: not categorical.
    assert df["peptide"].dtype == "object"
    assert df["assay_iri"].dtype == "object"
    # Numeric: untouched.
    assert df["pmid"].dtype.name == "Int64"
    assert df["mhc_allele_set_size"].dtype.kind in "iuf"


def test_compress_categoricals_is_idempotent():
    """Re-applying the helper is a no-op (already-categorical columns are
    skipped, not re-converted)."""
    df = _full_obs_fixture()
    _compress_categoricals(df)
    snapshot = {col: df[col].dtype for col in df.columns}
    _compress_categoricals(df)
    for col, dt in snapshot.items():
        assert df[col].dtype == dt


def test_compress_categoricals_empty_frame_is_noop():
    """Empty input must not raise (and must not invent columns)."""
    df = pd.DataFrame()
    _compress_categoricals(df)
    assert df.empty
    assert list(df.columns) == []


def test_compress_categoricals_strict_raises_on_missing_column():
    """``strict=True`` catches typos in the allowlist by failing loudly
    when a listed column is missing from the frame.  Default ``strict=False``
    silently skips so the helper can run on per-source partitions."""
    df = pd.DataFrame({"mhc_class": ["I"] * 5, "peptide": list("abcde")})
    # default mode: silently skip missing columns
    _compress_categoricals(df)
    assert df["mhc_class"].dtype.name == "category"
    # strict mode: raise when columns are missing
    df2 = pd.DataFrame({"mhc_class": ["I"] * 5})
    with pytest.raises(KeyError, match="missing columns"):
        _compress_categoricals(df2, strict=True)


def test_compress_categoricals_partial_frame_default_does_not_raise():
    """Default ``strict=False`` works on a partial frame (per-source partitions
    may not have every column the full obs frame eventually carries)."""
    df = pd.DataFrame({"source": ["iedb"] * 5, "peptide": list("abcde")})
    _compress_categoricals(df)
    assert df["source"].dtype.name == "category"


def test_compress_categoricals_reduces_memory_at_scale():
    """Realistic-cardinality fixture: helper cuts memory ~5x or more.

    Lower bound is conservative (CI variance, pandas version differences).
    The actual production reduction on the 4.4 M-row corpus is ~7-10x.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 100_000
    df = pd.DataFrame(
        {
            "mhc_class": rng.choice(["I", "II"], n),
            "source": rng.choice(["iedb", "cedar"], n),
            "mhc_species": rng.choice(["Homo sapiens", "Mus musculus"], n),
            "mhc_restriction": rng.choice(["HLA-A*02:01", "HLA-B*07:02", "HLA-DRB1*15:01"], n),
            "host": rng.choice([f"Donor-{i}" for i in range(50)], n),
            "disease": rng.choice([f"D-{i}" for i in range(80)], n),
            "cell_name": rng.choice([f"L-{i}" for i in range(200)], n),
            "peptide": [f"P{i:08d}" for i in range(n)],  # unique
        }
    )
    before = df.memory_usage(deep=True).sum()
    _compress_categoricals(df)
    after = df.memory_usage(deep=True).sum()
    assert before / after >= 5.0, f"only {before / after:.2f}x reduction (expected >= 5x)"


# ── _drop_duplicate_iris (replaces ms_seen_iris Python set) ──────────────


def test_drop_duplicate_iris_keeps_first_occurrence():
    """Duplicate ``assay_iri`` values are dropped, first occurrence wins —
    preserves the prior 'IEDB beats CEDAR' tie-break (since IEDB is
    concat'd first)."""
    df = pd.DataFrame(
        {
            "assay_iri": ["a:1", "a:2", "a:1", "a:3"],
            "source": ["iedb", "iedb", "cedar", "cedar"],
            "peptide": ["P1", "P2", "P3", "P4"],
        }
    )
    out = _drop_duplicate_iris(df, label="MS")
    assert len(out) == 3
    assert list(out["assay_iri"]) == ["a:1", "a:2", "a:3"]
    # First-wins semantics: the iedb row for a:1 survives, not the cedar one.
    assert out.loc[out["assay_iri"] == "a:1", "source"].iloc[0] == "iedb"


def test_drop_duplicate_iris_falls_back_to_reference_iri():
    """Rows missing ``assay_iri`` (older intermediates) dedup on
    ``reference_iri`` instead."""
    df = pd.DataFrame(
        {
            "assay_iri": ["", "a:2", "", "a:3"],
            "reference_iri": ["r:1", "r:2", "r:1", "r:3"],
            "peptide": list("abcd"),
        }
    )
    out = _drop_duplicate_iris(df, label="MS")
    # The two empty-iri rows share reference_iri r:1 → second is dropped.
    assert len(out) == 3


def test_drop_duplicate_iris_empty_returns_empty():
    out = _drop_duplicate_iris(pd.DataFrame(), label="MS")
    assert out.empty


def test_drop_duplicate_iris_no_iri_column_returns_unchanged():
    df = pd.DataFrame({"peptide": ["P1", "P2"]})
    out = _drop_duplicate_iris(df, label="MS")
    assert len(out) == 2


# ── pyarrow concat + supplement anti-join (replaces Python set dedup) ────


def test_pyarrow_concat_preserves_categoricals_round_trip():
    """``pa.Table.from_pandas`` → ``concat_tables`` → ``to_pandas`` must
    preserve categorical dtypes (Arrow ``DictionaryArray`` ↔ pandas
    ``category``).  This is the path the builder relies on so the
    post-concat ``obs`` frame keeps the categorical compression applied
    upstream.
    """
    import pyarrow as pa

    a = pd.DataFrame({"mhc_class": pd.Categorical(["I", "II"]), "peptide": ["P1", "P2"]})
    b = pd.DataFrame({"mhc_class": pd.Categorical(["I", "I"]), "peptide": ["P3", "P4"]})
    t = pa.concat_tables(
        [
            pa.Table.from_pandas(a, preserve_index=False),
            pa.Table.from_pandas(b, preserve_index=False),
        ],
        promote_options="default",
    )
    out = t.to_pandas()
    assert out["mhc_class"].dtype.name == "category"
    assert list(out["mhc_class"]) == ["I", "II", "I", "I"]


def test_supplement_dedup_anti_join_drops_existing_rows():
    """Vectorized anti-join replaces the prior ``set(zip(obs[3 cols]))``
    Python set: supp rows with a (peptide, mhc_restriction, pmid) triple
    that already exists in IEDB/CEDAR are dropped; novel triples kept.
    """
    obs = pd.DataFrame(
        {
            "peptide": ["P1", "P2", "P3"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01", "HLA-B*07:02"],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
        }
    )
    supp = pd.DataFrame(
        {
            "peptide": ["P1", "P4", "P3"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*11:01", "HLA-B*07:02"],
            "pmid": pd.array([1, 99, 3], dtype="Int64"),
        }
    )
    join_cols = ["peptide", "mhc_restriction", "pmid"]
    obs_keys = obs[join_cols].drop_duplicates()
    for col in ("peptide", "mhc_restriction"):
        obs_keys[col] = obs_keys[col].astype(str)
        supp[col] = supp[col].astype(str)
    out = (
        supp.merge(obs_keys, on=join_cols, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
        .reset_index(drop=True)
    )
    # Only P4 (novel triple) survives.
    assert list(out["peptide"]) == ["P4"]
    assert list(out["mhc_restriction"]) == ["HLA-A*11:01"]
