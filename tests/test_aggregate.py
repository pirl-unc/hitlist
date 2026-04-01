import pandas as pd

from hitlist.aggregate import aggregate_per_peptide, aggregate_per_pmhc


def test_aggregate_per_peptide_empty():
    df = aggregate_per_peptide(pd.DataFrame())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_aggregate_per_pmhc_empty():
    df = aggregate_per_pmhc(pd.DataFrame())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── Mono-allelic aggregation ───────────────────────────────────────────


def _make_hits_with_mono():
    return pd.DataFrame(
        {
            "peptide": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA-A*02:01",
            ],
            "reference_iri": ["r1", "r2", "r3", "r4", "r5"],
            "pmid": [111, 222, 333, 444, 555],
            "is_monoallelic": [True, False, True, False, True],
            "src_cancer": [True, True, True, True, False],
            "src_healthy_tissue": [False, False, False, False, False],
            "cell_line_name": ["B721.221", "", "K562", "", "B721.221"],
        }
    )


def test_aggregate_per_peptide_mono():
    df = _make_hits_with_mono()
    result = aggregate_per_peptide(df)
    aaa = result[result["peptide"] == "AAA"].iloc[0]
    assert aaa["has_mono_allelic_evidence"] == True  # noqa: E712
    assert aaa["mono_allelic_hit_count"] == 1
    bbb = result[result["peptide"] == "BBB"].iloc[0]
    assert bbb["has_mono_allelic_evidence"] == True  # noqa: E712
    assert bbb["mono_allelic_hit_count"] == 1


def test_aggregate_per_pmhc_mono():
    df = _make_hits_with_mono()
    result = aggregate_per_pmhc(df)
    aaa_a02 = result[
        (result["peptide"] == "AAA") & (result["mhc_restriction"] == "HLA-A*02:01")
    ].iloc[0]
    assert aaa_a02["ms_pmhc_has_mono_evidence"] == True  # noqa: E712
    assert aaa_a02["ms_pmhc_mono_hit_count"] == 1
    bbb_b07 = result[
        (result["peptide"] == "BBB") & (result["mhc_restriction"] == "HLA-B*07:02")
    ].iloc[0]
    assert bbb_b07["ms_pmhc_has_mono_evidence"] == False  # noqa: E712
    assert bbb_b07["ms_pmhc_mono_hit_count"] == 0
