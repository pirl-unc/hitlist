"""Experiment inventories, unknown pooled counts and explicit filter semantics."""

import json

import pandas as pd
import pytest

from hitlist.bulk_proteomics import _apply_reproducibility_filter
from hitlist.bulk_replicates import curate_replicate_counts


def rows():
    return pd.DataFrame(
        {
            "source": ["Bekker-Jensen_2017"] * 8 + ["other"],
            "cell_line_name": ["HeLa"] * 9,
            "peptide": list("ABCDEFGHI"),
            "digestion_enzyme": ["Trypsin/P"] * 9,
            "n_fractions_in_run": [14, 14, 46, 70, 70, 70, 70, 99, 14],
            "enrichment": ["none"] * 9,
            "fractionation_ph": [10.0] * 9,
            "n_replicates_detected": [1, 2, 1, 1, 2, 0.5, None, 1, 7],
            "n_replicates_possible": [2] * 9,
        }
    )


def test_curated_inventory_not_observed_max_and_unknowns():
    source = rows()
    result = curate_replicate_counts(source)
    assert result.n_experiments_possible.iloc[:4].tolist() == [3, 3, 3, 1]
    assert result.n_replicates_possible.iloc[:4].tolist() == [3, 3, 3, 1]
    assert result.n_replicates_detected.iloc[[0, 1, 3]].tolist() == [1, 2, 1]
    assert result.n_experiments_detected.iloc[[2, 4, 5, 6, 7, 8]].isna().all()
    assert result.reproducibility_fraction.iloc[[0, 1, 3]].tolist() == [1 / 3, 2 / 3, 1]
    assert result.reproducibility_fraction.iloc[[2, 4, 5, 6, 7, 8]].isna().all()
    assert result.replicate_count_status.tolist() == [
        "available",
        "available",
        "unresolved_aggregate",
        "available",
        "invalid_reported_count",
        "invalid_reported_count",
        "missing_reported_count",
        "unresolved_scope",
        "unresolved_scope",
    ]
    assert result.n_acquisition_runs.iloc[:4].tolist() == [42, 42, 138, 70]
    assert result.n_biological_replicates.isna().all()
    assert json.loads(result.replicate_experiment_ids.iloc[2]) == [
        "HeLa-46fracs-IT-E1",
        "HeLa-46fracs-IT-E2",
        "Tryp-46fracs",
    ]
    assert result.replicate_count_source_sha256.iloc[0] == (
        "5eb5c41af2c7929be8a9354bf7c9fcb021ead7f47c07d0774a146b352e1a8860"
    )
    assert result.replicate_count_reference.iloc[0].endswith("SearchResults.zip#summary.txt")
    assert result.n_replicates_detected.iloc[-1] == 7
    pd.testing.assert_series_equal(
        result.reported_n_replicates_detected, source.n_replicates_detected, check_names=False
    )
    pd.testing.assert_series_equal(
        result.reported_n_replicates_possible, source.n_replicates_possible, check_names=False
    )
    pd.testing.assert_frame_equal(result, curate_replicate_counts(result))
    for threshold, expected in [(0, list("ABD")), (0.5, list("BD")), (1, ["D"])]:
        assert _apply_reproducibility_filter(result, threshold).peptide.tolist() == expected
    assert len(_apply_reproducibility_filter(result, None)) == len(source)


@pytest.mark.parametrize("value", [-1, 1.01, float("nan"), float("inf")])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError, match="finite and between"):
        _apply_reproducibility_filter(pd.DataFrame(), value)


@pytest.mark.parametrize("value", [-1, float("inf"), "not-a-count"])
def test_invalid_reported_numerator_is_preserved_but_unqualified(value):
    source = rows().iloc[[3]].copy()
    source["n_replicates_detected"] = value
    result = curate_replicate_counts(source)
    assert result.reported_n_replicates_detected.iloc[0] == value
    assert result.n_experiments_detected.isna().all()
    assert result.replicate_count_status.iloc[0] == "invalid_reported_count"


def test_missing_scope_and_columns_do_not_pass_a_filter():
    source = rows().iloc[:1].drop(columns=["n_fractions_in_run", "n_replicates_detected"])
    result = curate_replicate_counts(source)
    assert result.n_replicates_possible.isna().all()
    assert result.replicate_count_status.iloc[0] == "unresolved_scope"
    assert _apply_reproducibility_filter(result, 0).empty
    assert _apply_reproducibility_filter(pd.DataFrame({"peptide": ["A"]}), 0).empty


def test_new_and_legacy_parquet_use_the_same_counts(tmp_path):
    from hitlist.bulk_proteomics import _read_parquet_cached

    source = rows().iloc[:8].copy()
    expected = curate_replicate_counts(source)
    for name, content in (("legacy", source), ("current", expected)):
        path = tmp_path / (name + ".parquet")
        content.to_parquet(path)
        stat = path.stat()
        loaded = _read_parquet_cached(str(path), stat.st_mtime_ns, stat.st_size)
        pd.testing.assert_frame_equal(curate_replicate_counts(loaded), expected, check_like=True)
