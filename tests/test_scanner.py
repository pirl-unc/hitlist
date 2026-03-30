import pandas as pd

from hitlist.scanner import scan


def test_scan_no_sources():
    df = scan(peptides={"SLYNTVATL"}, iedb_path=None, cedar_path=None)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_scan_nonexistent():
    df = scan(peptides={"SLYNTVATL"}, iedb_path="/nonexistent.csv")
    assert len(df) == 0


def test_scan_profile_mode_no_sources():
    df = scan(peptides=None, iedb_path=None)
    assert len(df) == 0
