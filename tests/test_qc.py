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


def test_cross_reference_emits_needs_deconvolution_for_class_only_pmids(tmp_path, monkeypatch):
    """v1.30.15: PMIDs with YAML alleles but ZERO 4-digit data rows
    (data is 100% class-only) used to dump every YAML allele under
    ``yaml_only`` — non-actionable noise that inflated curation_plan
    priorities. Now collapsed into a single ``needs_deconvolution``
    sentinel row per PMID.
    """
    from hitlist import qc

    # PMID 200: 3 rows, all class-only (no 4-digit allele in data).
    obs_path = _write_obs_fixture(
        tmp_path,
        [
            {
                "peptide": "AAAAAAAAA",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "class_only",
                "mhc_restriction": "HLA class I",
                "pmid": 200,
            },
            {
                "peptide": "BBBBBBBBB",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "class_only",
                "mhc_restriction": "HLA class I",
                "pmid": 200,
            },
            {
                "peptide": "CCCCCCCCC",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "class_only",
                "mhc_restriction": "HLA class I",
                "pmid": 200,
            },
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    # YAML lists 5 alleles but data has 0 four-digit rows.
    fake_overrides = {
        200: {
            "study_label": "All Class-Only Study",
            "ms_samples": [
                {
                    "sample_label": "donor1",
                    "mhc": ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*04:01"],
                },
                {
                    "sample_label": "donor2",
                    "mhc": ["HLA-A*03:01", "HLA-B*15:01"],
                },
            ],
        },
    }
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: fake_overrides)

    df = qc.cross_reference()
    # Exactly ONE needs_deconvolution row, no per-allele yaml_only spam.
    needs_decon = df[df["direction"] == "needs_deconvolution"]
    yaml_only = df[df["direction"] == "yaml_only"]
    assert len(needs_decon) == 1
    assert needs_decon.iloc[0]["pmid"] == 200
    assert "5 YAML alleles" in needs_decon.iloc[0]["allele"]
    # No per-allele yaml_only rows for this PMID.
    assert (yaml_only["pmid"] != 200).all() if not yaml_only.empty else True


def test_cross_reference_yaml_only_still_fires_when_some_data_is_4digit(tmp_path, monkeypatch):
    """Don't regress the yaml_only path: as long as ANY 4-digit data
    exists for the PMID, the per-allele yaml_only emission still
    runs (because deconvolution isn't structurally blocked)."""
    from hitlist import qc

    obs_path = _write_obs_fixture(
        tmp_path,
        [
            # Mixed: 1 four-digit row + 5 class-only rows.
            {
                "peptide": "AAAAAAAAA",
                "mhc_class": "I",
                "source": "iedb",
                "allele_resolution": "four_digit",
                "mhc_restriction": "HLA-A*02:01",
                "pmid": 300,
            },
            *[
                {
                    "peptide": f"X{i:08d}",
                    "mhc_class": "I",
                    "source": "iedb",
                    "allele_resolution": "class_only",
                    "mhc_restriction": "HLA class I",
                    "pmid": 300,
                }
                for i in range(5)
            ],
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    fake_overrides = {
        300: {
            "study_label": "Mixed Study",
            "ms_samples": [
                {"sample_label": "donor1", "mhc": ["HLA-A*02:01", "HLA-C*04:01"]},
            ],
        },
    }
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: fake_overrides)

    df = qc.cross_reference()
    # C*04:01 still fires as yaml_only — A*02:01 has data, so this PMID
    # is NOT structurally class-only.
    assert "HLA-C*04:01" in df[df["direction"] == "yaml_only"]["allele"].values
    assert df[df["direction"] == "needs_deconvolution"].empty


def test_run_all_returns_named_dataframes(tmp_path, monkeypatch):
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
                "is_monoallelic": False,
                "mhc_class_label_suspect": False,
                "mhc_allele_provenance": "exact",
            },
        ],
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    results = qc.run_all()
    assert set(results.keys()) == {
        "resolution",
        "normalization",
        "cross_reference",
        "discrepancies",
    }
    for v in results.values():
        assert isinstance(v, pd.DataFrame)


# ── #182 / #45 / #37 / v1.30.9: discrepancies report ──────────────────


def _disc_fixture_row(
    peptide: str,
    mhc_class: str,
    pmid: int,
    *,
    mhc_restriction: str = "",
    is_monoallelic: bool = False,
    mhc_class_label_suspect: bool = False,
    mhc_allele_provenance: str = "exact",
    cell_name: str = "",
):
    """Build a single fixture row with all columns ``discrepancies`` projects."""
    return {
        "peptide": peptide,
        "mhc_class": mhc_class,
        "mhc_restriction": mhc_restriction,
        "is_monoallelic": is_monoallelic,
        "mhc_class_label_suspect": mhc_class_label_suspect,
        "mhc_allele_provenance": mhc_allele_provenance,
        "cell_name": cell_name,
        "pmid": pmid,
        "reference_iri": f"r-{pmid}-{peptide}",
        "source": "iedb",
        "mhc_species": "Homo sapiens",
        "allele_resolution": (
            "four_digit"
            if mhc_restriction.startswith("HLA-") and "*" in mhc_restriction
            else "class_only"
            if mhc_restriction.startswith("HLA class")
            else "unresolved"
        ),
    }


def test_discrepancies_surfaces_class_label_mismatches(tmp_path, monkeypatch):
    """v1.30.9 / #182: per-PMID rate of ``mhc_class_label_suspect``
    rows is reported. A PMID with many class-II 9-mers (the Marcu 2021
    pattern) sorts to the top of the discrepancy report."""
    from hitlist import qc

    rows = []
    # 60 valid class-I rows on PMID 1 (clean).
    for i in range(60):
        rows.append(_disc_fixture_row(f"P{i:09d}", "I", 1, mhc_restriction="HLA-A*02:01"))
    # 60 rows on PMID 2: 50 valid class-II 14-mers + 10 suspect 7-mers
    # labeled class II. v1.30.17: the suspect tier for class-II is now
    # 5-7 aa (not 8-10 aa, which is borderline) so the test peptides
    # must be 7 aa or shorter to fire the suspect flag.
    for i in range(50):
        rows.append(
            _disc_fixture_row(
                f"Q{i:013d}",
                "II",
                2,
                mhc_restriction="HLA-DRB1*15:01",
            )
        )
    for i in range(10):
        rows.append(
            _disc_fixture_row(
                f"R{i:06d}",
                "II",
                2,
                mhc_restriction="HLA-DRB1*15:01",
                mhc_class_label_suspect=True,
            )
        )
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(min_rows=10)
    # PMID 2 (with 10 suspects) sorts above PMID 1 (with 0).
    assert df.iloc[0]["pmid"] == 2
    assert df.iloc[0]["suspect_class_label_n"] == 10
    assert abs(df.iloc[0]["suspect_class_label_rate"] - 10 / 60) < 1e-6
    pmid1_row = df[df["pmid"] == 1].iloc[0]
    assert pmid1_row["suspect_class_label_n"] == 0


def test_discrepancies_breaks_down_borderline_vs_implausible(tmp_path, monkeypatch):
    """v1.30.18 / #201: discrepancies surfaces separate
    ``borderline_class_label_n`` and ``implausible_class_label_n``
    columns so a curator can distinguish bulged class-I peptides
    (uncommon but real, 13-14aa) from clear curation drift (≥18aa
    class I). Both feed the binary suspect flag — but only when
    severity is suspect or implausible, not borderline."""
    from hitlist import qc

    rows = []
    # PMID 99: 50 valid class-I 9-mers + 10 borderline 13-mers
    # (bulged class-I, real biology) + 5 implausible 18-mers
    # (class-I should never be 18aa).
    for i in range(50):
        rows.append(
            _disc_fixture_row(
                f"P{i:09d}",
                "I",
                99,
                mhc_restriction="HLA-A*02:01",
            )
        )
    for i in range(10):
        rows.append(
            _disc_fixture_row(
                f"B{i:013d}",  # 14-char placeholder peptide column; class I
                "I",
                99,
                mhc_restriction="HLA-A*02:01",
            )
        )
    for i in range(5):
        rows.append(
            _disc_fixture_row(
                f"I{i:018d}",  # 19-char placeholder; under v1.30.17 this is implausible
                "I",
                99,
                mhc_restriction="HLA-A*02:01",
            )
        )
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(min_rows=10)
    row = df[df["pmid"] == 99].iloc[0]

    # Borderline (13-14aa class I): uncommon but real, NOT suspect.
    assert row["borderline_class_label_n"] == 10
    # Implausible (≥18aa class I): clearly miscurated.
    assert row["implausible_class_label_n"] == 5
    # Backwards-compat: suspect_class_label_n only counts implausible
    # (no rows in the suspect 15-17 tier in this fixture).
    assert row["suspect_class_label_n"] == 5


def test_discrepancies_flags_monoallelic_class_only(tmp_path, monkeypatch):
    """v1.30.9 / #45: rows that are mono-allelic but carry
    ``HLA class I`` instead of a 4-digit allele are surfaced — IEDB
    lost the per-peptide allele attribution but the paper knows it.
    These appear in the ``monoallelic_class_only_n`` column so a
    curator can prioritize them."""
    from hitlist import qc

    rows = (
        # 5 mono-allelic rows on PMID 99: 3 with proper 4-digit, 2 with class sentinel.
        [
            _disc_fixture_row(
                f"M{i:09d}",
                "I",
                99,
                mhc_restriction="HLA-A*02:01",
                is_monoallelic=True,
            )
            for i in range(3)
        ]
        + [
            _disc_fixture_row(
                f"S{i:09d}",
                "I",
                99,
                mhc_restriction="HLA class I",
                is_monoallelic=True,
            )
            for i in range(2)
        ]
        + [
            # Padding so the bucket clears the min_rows floor.
            _disc_fixture_row(
                f"X{i:09d}",
                "I",
                99,
                mhc_restriction="HLA-A*02:01",
                is_monoallelic=True,
            )
            for i in range(50)
        ]
    )
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(min_rows=10)
    pmid_99 = df[df["pmid"] == 99].iloc[0]
    assert pmid_99["monoallelic_class_only_n"] == 2
    # severity is "warn" because monoallelic_class_only_n > 0.
    assert pmid_99["severity"] == "warn"


def test_discrepancies_flags_class_pool_rate(tmp_path, monkeypatch):
    """v1.30.9 / #37: per-PMID rate of ``pmid_class_pool`` provenance
    is surfaced. A study where IEDB only stored ``HLA class I`` for
    every peptide ends up with class_pool_rate = 1.0; that's the
    deconvolution-target signal."""
    from hitlist import qc

    rows = [
        _disc_fixture_row(
            f"C{i:09d}",
            "I",
            7,
            mhc_restriction="HLA class I",
            mhc_allele_provenance="pmid_class_pool",
        )
        for i in range(60)
    ]
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(min_rows=10)
    pmid_7 = df[df["pmid"] == 7].iloc[0]
    assert pmid_7["class_pool_n"] == 60
    assert pmid_7["class_pool_rate"] == 1.0


def test_discrepancies_flags_nonstandard_amino_acids(tmp_path, monkeypatch):
    """v1.30.9: peptides with X / B / Z / U / lowercase / digits get
    counted in ``nonstandard_aa_n``. Catches ambiguous PSM hits and
    upstream string corruption that would otherwise corrupt model
    training inputs."""
    from hitlist import qc

    # All-AA padding (digits in peptide strings would themselves match the
    # nonstandard regex). Use a fixed valid 9-mer for the 58 padding rows.
    rows = [
        _disc_fixture_row("VALIDPEPT", "I", 5, mhc_restriction="HLA-A*02:01") for _ in range(58)
    ] + [
        # Two non-standard peptides — one with X (ambiguous AA), one
        # lowercase (modification carryover).
        _disc_fixture_row("PXPTIDEX", "I", 5, mhc_restriction="HLA-A*02:01"),
        _disc_fixture_row("ppeptidea", "I", 5, mhc_restriction="HLA-A*02:01"),
    ]
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(min_rows=10)
    pmid_5 = df[df["pmid"] == 5].iloc[0]
    assert pmid_5["nonstandard_aa_n"] == 2
    assert pmid_5["severity"] == "warn"


def test_discrepancies_drops_small_pmid_buckets(tmp_path, monkeypatch):
    """v1.30.9: ``min_rows`` filters out PMID buckets whose row count
    is below the threshold. Length percentiles on tiny buckets (3-5
    rows) are noise; the filter keeps the report focused on real
    cohorts."""
    from hitlist import qc

    rows = (
        # 3 rows on PMID 1 — should be dropped under min_rows=10.
        [_disc_fixture_row(f"A{i:09d}", "I", 1, mhc_restriction="HLA-A*02:01") for i in range(3)]
        # 60 rows on PMID 2 — should survive.
        + [_disc_fixture_row(f"B{i:09d}", "I", 2, mhc_restriction="HLA-A*02:01") for i in range(60)]
    )
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(min_rows=10)
    assert set(df["pmid"]) == {2}


def test_discrepancies_attaches_study_label_from_overrides(tmp_path, monkeypatch):
    """v1.30.9: the report attaches each PMID's ``study_label`` from
    ``pmid_overrides.yaml`` so the triage list is human-readable
    without a second YAML lookup."""
    from hitlist import qc

    rows = [
        _disc_fixture_row(f"P{i:09d}", "I", 42, mhc_restriction="HLA-A*02:01") for i in range(60)
    ]
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    overrides = {42: {"study_label": "Pretty Study Name 2026"}}
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: overrides)

    df = qc.discrepancies(min_rows=10)
    assert df.iloc[0]["study_label"] == "Pretty Study Name 2026"


# ── v1.30.12: by="sample" rollup ──────────────────────────────────────


def test_discrepancies_by_sample_groups_per_cell_name(tmp_path, monkeypatch):
    """v1.30.12: ``by='sample'`` groups by (pmid, mhc_class, cell_name).
    A study with a clean transfectant and a problematic transfectant
    should produce two rows where the dirty one has higher
    suspect_class_label_n than the clean one — exactly the per-sample
    visibility the user asked for ('one transfectant has 30%% suspect
    rows but its sibling has 0%%')."""
    from hitlist import qc

    # Suspect flag is recomputed from peptide length + class at load
    # time, so test fixtures must encode the suspicion in the peptide
    # length itself: class I ≥18aa or class II ≤10aa flags as suspect.
    rows = (
        # K562-A0201: 30 normal 9-aa class-I rows, none suspect.
        [
            _disc_fixture_row(
                "AAAAAAAAA",
                "I",
                7,
                mhc_restriction="HLA-A*02:01",
                cell_name="K562-A0201",
            )
            for _ in range(30)
        ]
        # K562-B0702: 30 clean 9-aa class-I rows + 12 suspect 18-aa rows.
        + [
            _disc_fixture_row(
                "BBBBBBBBB",
                "I",
                7,
                mhc_restriction="HLA-B*07:02",
                cell_name="K562-B0702",
            )
            for _ in range(30)
        ]
        + [
            _disc_fixture_row(
                "CCCCCCCCCCCCCCCCCC",
                "I",
                7,
                mhc_restriction="HLA-B*07:02",
                cell_name="K562-B0702",
            )
            for _ in range(12)
        ]
    )
    obs_path = tmp_path / "observations.parquet"
    pd.DataFrame(rows).to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(by="sample", min_rows=10)
    # Two output rows — one per cell_name.
    assert "cell_name" in df.columns
    assert set(df["cell_name"]) == {"K562-A0201", "K562-B0702"}
    dirty = df[df["cell_name"] == "K562-B0702"].iloc[0]
    clean = df[df["cell_name"] == "K562-A0201"].iloc[0]
    assert dirty["suspect_class_label_n"] == 12
    assert clean["suspect_class_label_n"] == 0
    # Dirty sample should sort first by score.
    assert df.iloc[0]["cell_name"] == "K562-B0702"


def test_discrepancies_by_sample_no_cell_name_falls_back_to_placeholder(tmp_path, monkeypatch):
    """Rows without a cell_name (patient cohorts, raw IEDB without
    sample identifier) should bucket under '(no cell_name)' rather
    than collapsing into NaN groups that pandas may drop."""
    from hitlist import qc

    rows = [
        _disc_fixture_row(f"P{i:09d}", "I", 9, mhc_restriction="HLA-A*02:01") for i in range(60)
    ]
    obs_path = tmp_path / "observations.parquet"
    pd.DataFrame(rows).to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})

    df = qc.discrepancies(by="sample", min_rows=10)
    assert df.iloc[0]["cell_name"] == "(no cell_name)"
    assert df.iloc[0]["n_rows"] == 60


def test_discrepancies_by_invalid_value_raises():
    """Defensive: only 'pmid' or 'sample' are accepted."""
    import pytest

    from hitlist import qc

    with pytest.raises(ValueError, match="must be 'pmid' or 'sample'"):
        qc.discrepancies(by="study")


# ── v1.30.13: curation_plan / qc plan ──────────────────────────────────


def test_curation_plan_combines_all_qc_signals(tmp_path, monkeypatch):
    """v1.30.13: ``curation_plan`` joins discrepancies, cross_reference,
    and normalization_drift into one PMID-priority list. A PMID with
    drift + suspect rows + a yaml-only allele should sort above a PMID
    with only weak signals."""
    from hitlist import qc

    rows = (
        # PMID 100: 60 valid 9-aa class-I rows + 12 long (suspect) rows.
        [
            _disc_fixture_row(
                "AAAAAAAAA",
                "I",
                100,
                mhc_restriction="HLA-A*02:01",
            )
            for _ in range(60)
        ]
        + [
            _disc_fixture_row(
                "BBBBBBBBBBBBBBBBBB",
                "I",
                100,
                mhc_restriction="HLA-A*02:01",
            )
            for _ in range(12)
        ]
        # PMID 200: 60 clean rows, no signals.
        + [
            _disc_fixture_row(
                "CCCCCCCCC",
                "I",
                200,
                mhc_restriction="HLA-B*07:02",
            )
            for _ in range(60)
        ]
    )
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    overrides = {
        100: {
            "study_label": "Study With Issues",
            "hla_alleles": ["HLA-Cw*04:01"],  # drifts to HLA-C*04:01
            "ms_samples": [
                {"sample_label": "donor1", "mhc": ["HLA-A*02:01", "HLA-D*99:99"]},
            ],
        },
        200: {"study_label": "Clean Study", "ms_samples": []},
    }
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)

    plan = qc.curation_plan(min_rows=10)
    assert not plan.empty
    # PMID 100 has signals from all three reports; PMID 200 is clean.
    assert plan.iloc[0]["pmid"] == 100
    pmid_100 = plan[plan["pmid"] == 100].iloc[0]
    assert pmid_100["suspect_class_label_n"] == 12
    # v1.30.19 / #201: severity tiers propagate through the plan.
    # 18-aa class-I peptides fall in the "implausible" tier (which
    # also fires the suspect flag).
    assert "implausible_class_label_n" in plan.columns
    assert "borderline_class_label_n" in plan.columns
    assert pmid_100["implausible_class_label_n"] == 12
    assert pmid_100["borderline_class_label_n"] == 0
    assert pmid_100["normalization_drifts_n"] >= 1  # HLA-Cw*04:01 drift
    assert pmid_100["yaml_only_alleles_n"] >= 1  # HLA-D*99:99 not in data
    assert pmid_100["severity"] == "warn"


def test_curation_plan_ranks_by_priority_score(tmp_path, monkeypatch):
    """A study with normalization drift should beat one with just
    in-data discrepancies — drift means the YAML itself needs editing,
    which is higher-leverage than re-running a length filter."""
    from hitlist import qc

    rows = (
        # PMID 1: 100 suspect-class-label rows. Big number but no drift.
        [
            _disc_fixture_row(
                "BBBBBBBBBBBBBBBBBB",
                "I",
                1,
                mhc_restriction="HLA-A*02:01",
            )
            for _ in range(100)
        ]
        # PMID 2: 60 clean rows. Drift hits.
        + [
            _disc_fixture_row(
                "AAAAAAAAA",
                "I",
                2,
                mhc_restriction="HLA-A*02:01",
            )
            for _ in range(60)
        ]
    )
    obs_path = _write_obs_fixture(tmp_path, rows)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    overrides = {
        1: {"study_label": "PMID 1"},
        2: {
            "study_label": "PMID 2",
            "hla_alleles": ["HLA-Cw*04:01"],  # drifts
        },
    }
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)

    plan = qc.curation_plan(min_rows=10)
    # Drift-weighted score (x10000) should put PMID 2 ahead even with
    # only 1 drift vs PMID 1's 100 suspect rows.
    assert plan.iloc[0]["pmid"] == 2


def test_curation_plan_handles_no_signals(tmp_path, monkeypatch):
    """When every input report is empty (no signals at all), the
    plan returns an empty DataFrame with the expected columns rather
    than crashing on the outer-merge."""
    from hitlist import qc

    # A schema-valid but tiny fixture (1 row, no signals worth
    # surfacing — and below min_rows so discrepancies drops it).
    obs_path = _write_obs_fixture(
        tmp_path, [_disc_fixture_row("AAAAAAAAA", "I", 1, mhc_restriction="HLA-A*02:01")]
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.qc.load_pmid_overrides", lambda: {})
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: {})

    plan = qc.curation_plan(min_rows=1000)  # forces all PMIDs to drop
    assert plan.empty
    # Schema must still include priority + severity so consumers can
    # read .columns even when nothing is wrong.
    assert "priority_score" in plan.columns
    assert "severity" in plan.columns
