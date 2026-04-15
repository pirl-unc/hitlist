import pandas as pd

from hitlist.predict import _class_i_alleles, _class_ii_alleles


def test_class_i_alleles_parses_space_separated():
    assert _class_i_alleles("HLA-A*02:01 HLA-B*07:02 HLA-C*07:01") == [
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-C*07:01",
    ]


def test_class_i_alleles_skips_class_ii():
    assert _class_i_alleles("HLA-A*02:01 HLA-DRB1*15:01 HLA-DPB1*04:01") == ["HLA-A*02:01"]


def test_class_i_alleles_empty_on_sentinels():
    for s in ("", "unknown", "HLA class I", "HLA class II", None):
        assert _class_i_alleles(s) == []


def test_class_ii_alleles_parses_DRB_DPB_DQB():
    assert _class_ii_alleles("HLA-A*02:01 HLA-DRB1*15:01 HLA-DPB1*04:01 HLA-DQB1*06:02") == [
        "HLA-DRB1*15:01",
        "HLA-DPB1*04:01",
        "HLA-DQB1*06:02",
    ]


def test_reassign_class_ii_not_implemented():
    import pytest

    from hitlist.predict import reassign_class_only_alleles

    with pytest.raises(NotImplementedError):
        reassign_class_only_alleles(mhc_class="II")


def test_reassign_empty_when_no_class_only_rows(monkeypatch):
    """If generate_observations_table returns no class-only rows,
    reassign returns an empty DataFrame with the documented schema.
    """
    from hitlist import predict

    # Stub generate_observations_table to return a dataframe with no
    # class-only rows.
    fake = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "is_monoallelic": [False],
            "sample_mhc": ["HLA-A*02:01 HLA-B*07:02"],
            "sample_label": ["x"],
            "pmid": [1],
        }
    )
    monkeypatch.setattr("hitlist.export.generate_observations_table", lambda *a, **kw: fake)
    result = predict.reassign_class_only_alleles(method="mhcflurry")
    assert result.empty
    assert set(result.columns) >= {
        "peptide",
        "best_allele",
        "best_presentation_percentile",
        "is_strong_binder",
    }
