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

"""Tests for hitlist.qc — corpus + curation diagnostic checks."""

from __future__ import annotations

import pandas as pd


def _write_obs_fixture(tmp_path, rows):
    """Write a minimal observations.parquet fixture with the columns
    the qc functions actually project."""
    df = pd.DataFrame(rows)
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    return path


def test_resolution_histogram_buckets_and_pct(tmp_path, monkeypatch):
    """Histogram returns one row per (class, source, bucket) and pct_within_class
    sums to ~100 within each class."""
    from hitlist import qc

    obs_path = _write_obs_fixture(
        tmp_path,
        [
            # Class I: 3 four_digit, 1 class_only.
            {
                "peptide": "AAAAAAAAA",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-A*02:01",
                "pmid": 1,
            },
            {
                "peptide": "BBBBBBBBB",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-A*02:01",
                "pmid": 1,
            },
            {
                "peptide": "CCCCCCCCC",
                "mhc_class": "I",
                "source": "cedar",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-B*07:02",
                "pmid": 2,
            },
            {
                "peptide": "DDDDDDDDD",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "class_only",
                "mhc_restriction": "",
                "pmid": 3,
            },
            # Class II: 1 two_digit.
            {
                "peptide": "EEEEEEEEEEEEE",
                "mhc_class": "II",
                "source": "iedb",
                "allele_resolution": "two_digit",
                "mhc_restriction": "HLA-DRB1*04",
                "pmid": 4,
            },
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = qc.resolution_histogram()
    # 3 distinct (mhc_class, source, allele_resolution) combinations for class I,
    # 1 for class II = 4 rows total.
    assert len(df) == 4
    # pct_within_class sums to 100 per class.
    for _cls, group in df.groupby("mhc_class"):
        assert abs(group["pct_within_class"].sum() - 100) < 0.5

    # Most-resolved bucket (four_digit) sorts first within each class.
    class_i = df[df["mhc_class"] == "I"]
    assert class_i.iloc[0]["allele_resolution"] == "four_digit"


def test_resolution_histogram_filters(tmp_path, monkeypatch):
    """Class filter narrows the result correctly."""
    from hitlist import qc

    obs_path = _write_obs_fixture(
        tmp_path,
        [
            {
                "peptide": "AAAAAAAAA",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-A*02:01",
                "pmid": 1,
            },
            {
                "peptide": "EEEEEEEEEEEEE",
                "mhc_class": "II",
                "source": "iedb",
                "allele_resolution": "two_digit",
                "mhc_restriction": "HLA-DRB1*04",
                "pmid": 4,
            },
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = qc.resolution_histogram(mhc_class="I")
    assert set(df["mhc_class"]) == {"I"}
    assert len(df) == 1


def test_normalization_drift_flags_changed_alleles(monkeypatch):
    """normalize_allele rewriting an input → drift row.  ``HLA-Cw*04:01``
    is the legacy IMGT C-locus spelling; mhcgnomes rewrites it to the
    modern ``HLA-C*04:01``.  Alleles already in canonical form produce
    no drift row.
    """
    from hitlist import qc

    fake_overrides = {
        12345: {
            "study_label": "Fake Study 2024",
            "hla_alleles": [
                "HLA-Cw*04:01",  # legacy spelling → drifts to HLA-C*04:01
                "HLA-B*07:02",  # already canonical → no drift
            ],
        },
    }
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: fake_overrides)

    df = qc.normalization_drift()
    assert "HLA-Cw*04:01" in df["allele_raw"].values
    assert "HLA-B*07:02" not in df["allele_raw"].values
    drift_row = df[df["allele_raw"] == "HLA-Cw*04:01"].iloc[0]
    assert drift_row["allele_normalized"] == "HLA-C*04:01"
    assert drift_row["pmid"] == 12345


def test_normalization_drift_returns_empty_columns_when_no_drift(monkeypatch):
    """Empty result still has the expected columns (consumers can read
    ``df.columns`` even when nothing's wrong)."""
    from hitlist import qc

    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})
    df = qc.normalization_drift()
    assert list(df.columns) == [
        "pmid",
        "study_label",
        "allele_raw",
        "allele_normalized",
        "severity",
    ]
    assert len(df) == 0


def test_cross_reference_yaml_only_and_data_only(tmp_path, monkeypatch):
    """Cross-reference catches both directions:
    - allele in YAML sample but not in data rows for that PMID → yaml_only
    - 4-digit allele in data but not in any YAML sample for that PMID → data_only
    """
    from hitlist import qc

    obs_path = _write_obs_fixture(
        tmp_path,
        [
            # PMID 100: data has A*02:01 and B*07:02; YAML lists A*02:01 + C*04:01.
            # → C*04:01 yaml_only, B*07:02 data_only.
            {
                "peptide": "AAAAAAAAA",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-A*02:01",
                "pmid": 100,
            },
            {
                "peptide": "BBBBBBBBB",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-B*07:02",
                "pmid": 100,
            },
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    fake_overrides = {
        100: {
            "study_label": "Test Study",
            "ms_samples": [
                {"sample_label": "donor1", "mhc": ["HLA-A*02:01", "HLA-C*04:01"]},
            ],
        },
    }
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: fake_overrides)

    df = qc.cross_reference()
    assert not df.empty
    yaml_only = df[df["direction"] == "yaml_only"]
    data_only = df[df["direction"] == "data_only"]
    assert "HLA-C*04:01" in yaml_only["allele"].values
    assert "HLA-B*07:02" in data_only["allele"].values
    # The agreed allele (A*02:01) should NOT appear in either bucket.
    assert "HLA-A*02:01" not in df["allele"].values


def test_run_all_returns_three_named_dataframes(tmp_path, monkeypatch):
    """The dispatcher returns the expected keys regardless of contents."""
    from hitlist import qc

    obs_path = _write_obs_fixture(
        tmp_path,
        [
            {
                "peptide": "AAAAAAAAA",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-A*02:01",
                "pmid": 1,
            },
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    results = qc.run_all()
    assert set(results.keys()) == {"resolution", "normalization", "cross_reference"}
    for v in results.values():
        assert isinstance(v, pd.DataFrame)
