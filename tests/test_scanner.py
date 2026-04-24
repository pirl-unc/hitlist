import warnings

import pandas as pd
import pytest

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


# ── Deprecation of human_only (hitlist#72) ─────────────────────────────────


def test_human_only_true_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="human_only is deprecated"):
        scan(peptides={"X"}, iedb_path=None, cedar_path=None, human_only=True)


def test_human_only_false_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="human_only is deprecated"):
        scan(peptides={"X"}, iedb_path=None, cedar_path=None, human_only=False)


def test_no_species_kwargs_does_not_warn():
    """Default call path must stay warning-free — only explicit human_only= warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        scan(peptides={"X"}, iedb_path=None, cedar_path=None)  # must not raise


def test_explicit_mhc_species_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        scan(peptides={"X"}, iedb_path=None, cedar_path=None, mhc_species="Homo sapiens")


def test_mhc_species_wins_when_both_passed():
    """If both legacy and new kwargs are passed, mhc_species takes precedence.
    We can't observe the filter decision without a real CSV, but the call
    must succeed and emit exactly one DeprecationWarning for human_only."""
    with pytest.warns(DeprecationWarning, match="human_only"):
        scan(
            peptides={"X"},
            iedb_path=None,
            cedar_path=None,
            human_only=True,
            mhc_species="Mus musculus",
        )


# ── species_fallback unit tests (via _apply_species_filter behavior) ───────
#
# We can't hit the real filter logic without IEDB CSVs, so these tests
# exercise the shape guarantees: the kwarg is accepted and the call
# succeeds for the four (mhc_species x species_fallback) cross-product
# cases.


@pytest.mark.parametrize(
    "mhc_species,species_fallback",
    [
        ("Homo sapiens", True),
        ("Homo sapiens", False),
        ("Mus musculus", True),
        (None, True),
    ],
)
def test_species_fallback_accepted(mhc_species, species_fallback):
    df = scan(
        peptides={"X"},
        iedb_path=None,
        cedar_path=None,
        mhc_species=mhc_species,
        species_fallback=species_fallback,
    )
    assert isinstance(df, pd.DataFrame)


# ── assay_iri preservation (issue #146) ────────────────────────────────────


def _write_tiny_iedb_csv(path, rows):
    """Write a minimal IEDB-shaped CSV for scanner tests.

    IEDB exports have TWO header rows (category + field).  The scanner
    reads both via ``next(reader)`` twice, then streams data rows.  The
    tests below stream data rows by preceding the headers with a dummy
    category row so the scanner's header handling mirrors real IEDB.
    """
    field_header = [
        "Assay IRI",
        "Reference IRI",
        "PMID",
        "Submission ID",
        "Title",
        "Epitope | Name",
        "Epitope | Source Organism",
        "Epitope | Species",
        "Host",
        "Host Age",
        "Host | Process Type",
        "Host | Disease",
        "Host | Disease Stage",
        "Antigen Processing Comments",
        "Qualitative Measurement",
        "Assay Comments",
        "Effector Cells | Source Tissue",
        "Effector Cells | Cell Name",
        "Assay | Culture Condition",
        "MHC Restriction | Name",
        "MHC Allele Class",
    ]
    category_header = [""] * len(field_header)  # real IEDB uses grouping labels; "" is safe
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(category_header)
        writer.writerow(field_header)
        for row in rows:
            writer.writerow(row)


def test_scan_preserves_assay_iri_per_row(tmp_path):
    """Every scanner-emitted row must carry a populated ``assay_iri`` column
    (issue #146), not just ``reference_iri``.  The three assay rows below
    share one reference IRI but have distinct assay IRIs — all three must
    survive the dedupe (which is already keyed on assay_iri) AND arrive
    in the output with the assay_iri preserved.
    """
    src = tmp_path / "iedb.csv"
    rows = []
    shared_ref = "http://iedb.org/reference/42"
    for i in range(3):
        row = [""] * 21
        row[0] = f"http://iedb.org/assay/{1000001 + i}"  # Assay IRI
        row[1] = shared_ref
        row[2] = "33858848"
        row[5] = f"PEPTIDE{i}AB"  # Epitope | Name
        row[19] = "HLA-A*02:01"  # MHC Restriction
        row[20] = "I"
        rows.append(row)
    _write_tiny_iedb_csv(src, rows)

    df = scan(peptides=None, iedb_path=str(src), cedar_path=None)
    assert len(df) == 3
    assert "assay_iri" in df.columns
    assert df["assay_iri"].nunique() == 3
    assert df["reference_iri"].nunique() == 1  # the whole point: same paper
    # All assay_iri values propagated through unchanged.
    for i in range(3):
        assert f"http://iedb.org/assay/{1000001 + i}" in df["assay_iri"].values


def test_scan_dedupes_by_assay_iri_within_source(tmp_path):
    """Same ``assay_iri`` appearing twice (shouldn't happen in IEDB but the
    dedupe is defensive) must collapse to one row with that IRI preserved.
    """
    src = tmp_path / "iedb.csv"
    rows = []
    for _ in range(2):
        row = [""] * 21
        row[0] = "http://iedb.org/assay/7777777"  # identical
        row[1] = "http://iedb.org/reference/42"
        row[2] = "33858848"
        row[5] = "PEPTIDEAB"
        row[19] = "HLA-A*02:01"
        row[20] = "I"
        rows.append(row)
    _write_tiny_iedb_csv(src, rows)

    df = scan(peptides=None, iedb_path=str(src), cedar_path=None)
    assert len(df) == 1
    assert df["assay_iri"].iloc[0] == "http://iedb.org/assay/7777777"
