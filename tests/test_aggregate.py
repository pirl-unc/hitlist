import pandas as pd

from hitlist.aggregate import (
    aggregate_per_peptide,
    aggregate_per_pmhc,
    aggregate_per_pmhc_with_refs,
)


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
            "src_apc": [True, False, True, False, True],
            "src_healthy_tissue": [False, False, False, False, False],
            "cell_line_name": ["B721.221", "", "K562", "", "B721.221"],
        }
    )


def test_aggregate_per_peptide_mono():
    df = _make_hits_with_mono()
    result = aggregate_per_peptide(df)
    aaa = result[result["peptide"] == "AAA"].iloc[0]
    assert aaa["found_in_apc"] == True  # noqa: E712
    assert aaa["has_mono_allelic_evidence"] == True  # noqa: E712
    assert aaa["mono_allelic_hit_count"] == 1
    bbb = result[result["peptide"] == "BBB"].iloc[0]
    assert bbb["found_in_apc"] == True  # noqa: E712
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


# ── aggregate_per_pmhc_with_refs ───────────────────────────────────────


def _make_hits_with_provenance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peptide": ["QYIAQFTSQF", "QYIAQFTSQF", "LYVDSLFFL"],
            "mhc_restriction": ["HLA-A*24:02", "HLA-A*24:02", "HLA-A*24:02"],
            "pmid": [27869121, 38000001, 33858848],
            "source_tissue": ["Lymph Node", "Skin", "Thymus"],
            "disease": ["skin melanoma", "skin melanoma", ""],
            "cell_line_name": ["", "", ""],
            "src_cancer": [True, True, False],
            "src_apc": [False, True, True],
            "src_healthy_tissue": [False, False, False],
            "is_monoallelic": [False, True, False],
        }
    )


def test_aggregate_per_pmhc_with_refs_empty():
    result = aggregate_per_pmhc_with_refs(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    # Canonical column list is returned so downstream shape is stable
    assert "ms_pmhc_tissues" in result.columns
    assert "ms_pmhc_in_apc" in result.columns
    assert "ms_pmhc_in_cancer" in result.columns


def test_aggregate_per_pmhc_with_refs_basic():
    result = aggregate_per_pmhc_with_refs(_make_hits_with_provenance())
    assert len(result) == 2

    q = result[result["peptide"] == "QYIAQFTSQF"].iloc[0]
    assert int(q["length"]) == 10
    assert int(q["ms_pmhc_hit_count"]) == 2
    assert int(q["ms_pmhc_ref_count"]) == 2
    assert q["ms_pmhc_pmids"] == "27869121;38000001"
    assert q["ms_pmhc_tissues"] == "Lymph Node;Skin"
    assert q["ms_pmhc_diseases"] == "skin melanoma"
    assert bool(q["ms_pmhc_in_cancer"]) is True
    assert bool(q["ms_pmhc_in_apc"]) is True
    assert bool(q["ms_pmhc_in_healthy_tissue"]) is False
    assert int(q["ms_pmhc_mono_hit_count"]) == 1

    ly_row = result[result["peptide"] == "LYVDSLFFL"].iloc[0]
    assert int(ly_row["length"]) == 9
    assert ly_row["ms_pmhc_pmids"] == "33858848"
    assert ly_row["ms_pmhc_tissues"] == "Thymus"
    assert bool(ly_row["ms_pmhc_in_apc"]) is True
    assert bool(ly_row["ms_pmhc_in_cancer"]) is False


def test_aggregate_per_pmhc_with_refs_numeric_pmid_sort():
    """PMIDs must sort numerically — lexicographic order would place
    ``"1000000000"`` before ``"999999999"``."""
    hits = pd.DataFrame(
        {
            "peptide": ["SAME9PEP"] * 3,
            "mhc_restriction": ["HLA-A*02:01"] * 3,
            "pmid": [999999999, 38000001, 1000000000],
        }
    )
    result = aggregate_per_pmhc_with_refs(hits)
    assert result.loc[0, "ms_pmhc_pmids"] == "38000001;999999999;1000000000"
    assert int(result.loc[0, "ms_pmhc_ref_count"]) == 3


def test_aggregate_per_pmhc_with_refs_in_healthy_tissue_true():
    hits = pd.DataFrame(
        {
            "peptide": ["RISKY9PEP", "RISKY9PEP"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01"],
            "src_cancer": [True, False],
            "src_healthy_tissue": [False, True],
        }
    )
    result = aggregate_per_pmhc_with_refs(hits)
    assert bool(result.loc[0, "ms_pmhc_in_cancer"]) is True
    assert bool(result.loc[0, "ms_pmhc_in_healthy_tissue"]) is True


def test_aggregate_per_pmhc_with_refs_missing_optional_columns():
    """Skinny input (no pmid / tissue / src_*) must produce valid output
    with those columns absent — matches cached-observations fast path."""
    skinny = pd.DataFrame(
        {
            "peptide": ["ABCDEFGHI", "ABCDEFGHI"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01"],
        }
    )
    result = aggregate_per_pmhc_with_refs(skinny)
    assert len(result) == 1
    assert int(result.iloc[0]["ms_pmhc_hit_count"]) == 2
    assert "ms_pmhc_pmids" not in result.columns
    assert "ms_pmhc_tissues" not in result.columns
    assert "ms_pmhc_in_cancer" not in result.columns


def test_aggregate_per_pmhc_with_refs_preserves_non_hla():
    """Non-HLA rows (e.g. murine H2) pass through unchanged.

    Species filtering is the caller's responsibility at the scan /
    load_observations layer via ``mhc_species=``; the aggregator does
    not enforce an HLA-only policy.
    """
    hits = pd.DataFrame(
        {
            "peptide": ["HUMAN9PEP", "MOUSE9PEP"],
            "mhc_restriction": ["HLA-A*02:01", "H2-Db"],
        }
    )
    result = aggregate_per_pmhc_with_refs(hits)
    assert len(result) == 2
    assert set(result["mhc_restriction"]) == {"HLA-A*02:01", "H2-Db"}
