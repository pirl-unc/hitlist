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
