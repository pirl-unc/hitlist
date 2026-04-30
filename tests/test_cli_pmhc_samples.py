"""Tests for the v1.30.5 ``hitlist pmhc --sample`` / ``--samples`` parsers."""

from __future__ import annotations

import pytest


def test_parse_inline_sample_single():
    """One ``--sample NAME:ALLELE,ALLELE`` parses to one mapping entry."""
    from hitlist.cli import _parse_pmhc_samples

    out = _parse_pmhc_samples(["patient1:HLA-A*02:01,HLA-A*24:02"], tsv_path=None)
    assert out == {"patient1": ["HLA-A*02:01", "HLA-A*24:02"]}


def test_parse_inline_sample_multiple():
    """Multiple ``--sample`` flags accumulate into one map; per-sample
    allele sets stay isolated (no merge across samples)."""
    from hitlist.cli import _parse_pmhc_samples

    out = _parse_pmhc_samples(
        ["p1:HLA-A*02:01", "p2:HLA-B*07:02,HLA-A*24:02"],
        tsv_path=None,
    )
    assert out == {
        "p1": ["HLA-A*02:01"],
        "p2": ["HLA-B*07:02", "HLA-A*24:02"],
    }


def test_parse_inline_sample_strips_whitespace():
    """Whitespace around name + alleles is stripped (users hand-type
    these and inevitably leave spaces)."""
    from hitlist.cli import _parse_pmhc_samples

    out = _parse_pmhc_samples(
        ["  p1  : HLA-A*02:01 , HLA-A*24:02 "],
        tsv_path=None,
    )
    assert out == {"p1": ["HLA-A*02:01", "HLA-A*24:02"]}


def test_parse_inline_sample_missing_colon_raises():
    """When the user forgets the ``NAME:`` separator, raise — and use a
    canonical-name string with no internal ``:`` so the missing-separator
    check actually fires (``HLA-A*02:01`` itself contains a ``:``)."""
    from hitlist.cli import _parse_pmhc_samples

    with pytest.raises(ValueError, match="missing the ':'"):
        _parse_pmhc_samples(["p1 HLA-A0201"], tsv_path=None)


def test_parse_inline_sample_empty_name_raises():
    from hitlist.cli import _parse_pmhc_samples

    with pytest.raises(ValueError, match="empty name"):
        _parse_pmhc_samples([":HLA-A*02:01"], tsv_path=None)


def test_parse_inline_sample_no_alleles_raises():
    from hitlist.cli import _parse_pmhc_samples

    with pytest.raises(ValueError, match="no alleles"):
        _parse_pmhc_samples(["p1:"], tsv_path=None)


def test_parse_inline_sample_duplicate_name_raises():
    from hitlist.cli import _parse_pmhc_samples

    with pytest.raises(ValueError, match="more than once"):
        _parse_pmhc_samples(["p1:HLA-A*02:01", "p1:HLA-B*07:02"], tsv_path=None)


def test_parse_samples_tsv_basic(tmp_path):
    """TSV form: name<TAB>allele,allele[,...] — alleles comma-separated
    in the second column."""
    from hitlist.cli import _parse_pmhc_samples

    tsv = tmp_path / "cohort.tsv"
    tsv.write_text("patient1\tHLA-A*02:01,HLA-A*24:02\npatient2\tHLA-A*01:01\n")
    out = _parse_pmhc_samples([], tsv_path=str(tsv))
    assert out == {
        "patient1": ["HLA-A*02:01", "HLA-A*24:02"],
        "patient2": ["HLA-A*01:01"],
    }


def test_parse_samples_tsv_skips_header_row(tmp_path):
    """First-row 'name' header (case-insensitive) is detected and skipped."""
    from hitlist.cli import _parse_pmhc_samples

    tsv = tmp_path / "cohort.tsv"
    tsv.write_text("name\talleles\npatient1\tHLA-A*02:01\n")
    out = _parse_pmhc_samples([], tsv_path=str(tsv))
    assert out == {"patient1": ["HLA-A*02:01"]}


def test_parse_samples_tsv_skips_comments_and_blanks(tmp_path):
    from hitlist.cli import _parse_pmhc_samples

    tsv = tmp_path / "cohort.tsv"
    tsv.write_text("# this is a comment\n\npatient1\tHLA-A*02:01\n\n# trailing comment\n")
    out = _parse_pmhc_samples([], tsv_path=str(tsv))
    assert out == {"patient1": ["HLA-A*02:01"]}


def test_parse_samples_tsv_missing_file_raises():
    from hitlist.cli import _parse_pmhc_samples

    with pytest.raises(ValueError, match="not found"):
        _parse_pmhc_samples([], tsv_path="/nonexistent/cohort.tsv")


def test_parse_samples_tsv_short_row_raises(tmp_path):
    from hitlist.cli import _parse_pmhc_samples

    tsv = tmp_path / "cohort.tsv"
    tsv.write_text("patient1\n")  # missing alleles column
    with pytest.raises(ValueError, match="two tab-separated columns"):
        _parse_pmhc_samples([], tsv_path=str(tsv))


def test_parse_inline_and_tsv_merge(tmp_path):
    """Inline ``--sample`` + ``--samples`` TSV combine into one map.
    Useful for "the cohort + a hand-typed extra patient" workflow."""
    from hitlist.cli import _parse_pmhc_samples

    tsv = tmp_path / "cohort.tsv"
    tsv.write_text("p1\tHLA-A*02:01\n")
    out = _parse_pmhc_samples(
        ["p2:HLA-B*07:02"],
        tsv_path=str(tsv),
    )
    assert out == {
        "p2": ["HLA-B*07:02"],
        "p1": ["HLA-A*02:01"],
    }


def test_parse_inline_and_tsv_duplicate_raises(tmp_path):
    """Same name in both inline + TSV is a user error, not a silent merge."""
    from hitlist.cli import _parse_pmhc_samples

    tsv = tmp_path / "cohort.tsv"
    tsv.write_text("p1\tHLA-A*02:01\n")
    with pytest.raises(ValueError, match="more than once"):
        _parse_pmhc_samples(
            ["p1:HLA-B*07:02"],
            tsv_path=str(tsv),
        )


def test_parse_no_input_raises():
    from hitlist.cli import _parse_pmhc_samples

    with pytest.raises(ValueError, match="no samples parsed"):
        _parse_pmhc_samples([], tsv_path=None)
