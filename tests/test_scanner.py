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


# ── Quantitative binding-assay fields (issue #148) ─────────────────────────


def _write_quant_iedb_csv(path, rows):
    """Write a minimal IEDB CSV with the quantitative-assay columns populated.

    Columns the scanner uses are at fixed indices 0..108; we extend to 97
    so the Quantitative-measurement column (index 96) is writable.
    """
    field_header = [""] * 97
    # Fill the relevant headers at their canonical IEDB positions.
    field_header[0] = "Assay IRI"
    field_header[1] = "Reference IRI"
    field_header[2] = "PMID"
    field_header[5] = "Epitope | Name"
    field_header[90] = "Assay | Method"
    field_header[91] = "Assay | Response measured"
    field_header[92] = "Assay | Units"
    field_header[94] = "Qualitative Measurement"
    field_header[95] = "Assay | Measurement Inequality"
    field_header[96] = "Assay | Quantitative measurement"
    # Note: mhc_restriction / mhc_class are beyond index 96, but the scanner
    # falls back to _FALLBACK_INDICES for columns we don't explicitly name,
    # so indices 107/108 get read from row[107]/row[108].  Pad rows out.
    field_header_full = field_header + [""] * (109 - len(field_header))
    field_header_full[107] = "MHC Restriction | Name"
    field_header_full[108] = "MHC Allele Class"
    category_header = [""] * len(field_header_full)

    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(category_header)
        writer.writerow(field_header_full)
        for row in rows:
            # Pad each data row to match header width.
            padded = list(row) + [""] * (len(field_header_full) - len(row))
            writer.writerow(padded)


def test_scan_preserves_quantitative_binding_fields(tmp_path):
    """Scanner must emit assay_method / measurement_units /
    measurement_inequality / quantitative_measurement / quantitative_value
    from IEDB binding-assay rows (issue #148).  The float cast of
    ``Quantitative measurement`` lives in ``quantitative_value`` for
    downstream filtering.
    """
    src = tmp_path / "iedb.csv"
    rows: list[list] = []
    # IC50 (competitive binding), Kd (direct dissociation), t_half (kinetic),
    # plus a qualitative-only row.  Each carries a different ``Response measured``
    # so consumers can stratify by readout type.  The Response-measured strings
    # used here are the canonical IEDB values (``df['response_measured']
    # .value_counts()`` shows ``"half maximal inhibitory concentration (IC50)"``,
    # ``"dissociation constant KD"``, ``"half life"``, ``"qualitative binding"``,
    # ``"ligand presentation"`` etc.) so consumers can reproduce these tests
    # against a real build.
    for i, (method, response, units, ineq, q, pep) in enumerate(
        [
            (
                "purified MHC/competitive/fluorescence",
                "half maximal inhibitory concentration (IC50)",
                "nM",
                "=",
                "12.5",
                "QUANTROWAB",
            ),
            (
                "purified MHC/direct/fluorescence",
                "dissociation constant KD",
                "nM",
                "<",
                "500",
                "QUANTROWCD",
            ),
            (
                "cellular MHC/direct/fluorescence",
                "half life",
                "min",
                "=",
                "42",
                "KINETICAB",
            ),
            ("purified MHC/direct", "qualitative binding", "", "", "", "QUALONLYAB"),
        ]
    ):
        row = [""] * 112
        row[0] = f"http://iedb.org/assay/{9000001 + i}"
        row[1] = "http://iedb.org/reference/99"
        row[2] = "33858848"
        row[5] = pep
        row[90] = method
        row[91] = response
        row[92] = units
        row[94] = "Positive"
        row[95] = ineq
        row[96] = q
        row[107] = "HLA-A*02:01"
        row[108] = "I"
        rows.append(row)
    _write_quant_iedb_csv(src, rows)

    df = scan(peptides=None, iedb_path=str(src), cedar_path=None)
    assert len(df) == 4
    for col in (
        "assay_method",
        "response_measured",
        "measurement_units",
        "measurement_inequality",
        "quantitative_measurement",
        "quantitative_value",
    ):
        assert col in df.columns, f"missing column: {col}"

    quant = df[df["peptide"] == "QUANTROWAB"].iloc[0]
    assert quant["assay_method"] == "purified MHC/competitive/fluorescence"
    assert quant["response_measured"] == "half maximal inhibitory concentration (IC50)"
    assert quant["measurement_units"] == "nM"
    assert quant["measurement_inequality"] == "="
    assert quant["quantitative_measurement"] == "12.5"
    assert quant["quantitative_value"] == pytest.approx(12.5)

    kd = df[df["peptide"] == "QUANTROWCD"].iloc[0]
    assert kd["measurement_inequality"] == "<"
    assert kd["response_measured"] == "dissociation constant KD"
    assert kd["quantitative_value"] == pytest.approx(500.0)

    kinetic = df[df["peptide"] == "KINETICAB"].iloc[0]
    assert kinetic["response_measured"] == "half life"
    assert kinetic["measurement_units"] == "min"
    assert kinetic["quantitative_value"] == pytest.approx(42.0)

    qual_only = df[df["peptide"] == "QUALONLYAB"].iloc[0]
    assert qual_only["assay_method"] == "purified MHC/direct"
    assert qual_only["response_measured"] == "qualitative binding"
    assert qual_only["measurement_units"] == ""
    # Empty Quantitative measurement → NaN float in quantitative_value.
    import math

    assert math.isnan(qual_only["quantitative_value"])


def test_scan_quantitative_value_handles_garbage(tmp_path):
    """A non-numeric quantitative_measurement value must become NaN, not
    crash the scanner.
    """
    src = tmp_path / "iedb.csv"
    rows = [[""] * 109]
    rows[0][0] = "http://iedb.org/assay/8888888"
    rows[0][1] = "http://iedb.org/reference/99"
    rows[0][2] = "33858848"
    rows[0][5] = "TRASHROWAB"
    rows[0][90] = "something"
    rows[0][94] = "Positive"
    rows[0][96] = "not a number"
    rows[0][107] = "HLA-A*02:01"
    rows[0][108] = "I"
    _write_quant_iedb_csv(src, rows)

    df = scan(peptides=None, iedb_path=str(src), cedar_path=None)
    assert len(df) == 1
    import math

    assert df["quantitative_measurement"].iloc[0] == "not a number"
    assert math.isnan(df["quantitative_value"].iloc[0])
