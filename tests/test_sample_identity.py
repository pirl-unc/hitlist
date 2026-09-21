"""Tests for hitlist.sample_identity — the canonical "what is one sample"
definition (#260, #502).

These pin a load-bearing contract: the counts this module produces are how
a reader judges whether a candidate target is well-attested, so two
callers disagreeing about it is a correctness bug.
"""

from __future__ import annotations

import pathlib

import pandas as pd


def _sample_frame(rows: list[dict]) -> pd.DataFrame:
    """Observation-shaped frame with the sample-identity columns present."""
    base = {
        "pmid": 1,
        "src_cell_line": False,
        "cell_line_name": "",
        "cell_name": "",
        "monoallelic_host": "",
        "attributed_sample_label": "",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_sample_identity_unlabelled_donors_separate_by_pmid():
    """#502: attributed_sample_label is blank on 96.7% of the corpus. Two
    unlabelled rows from DIFFERENT studies are two samples, not one --
    counting the raw label collapsed them into a single bucket."""
    from hitlist.sample_identity import count_distinct_ids, sample_identity_ids

    df = _sample_frame([{"pmid": 111}, {"pmid": 222}])
    _, _, donor_type_id = sample_identity_ids(df)
    assert count_distinct_ids(donor_type_id) == 2


def test_sample_identity_cell_line_row_without_identifiers_still_counts():
    """The #502 sibling: a src_cell_line row carrying neither a line name
    nor a mono-allelic host used to get an empty ID and vanish from the
    count. It now falls back to its PMID, like the donor tier already did."""
    from hitlist.sample_identity import count_distinct_ids, sample_identity_ids

    df = _sample_frame([{"pmid": 333, "src_cell_line": True}])
    line_id, _, _ = sample_identity_ids(df)
    assert count_distinct_ids(line_id) == 1


def test_sample_identity_named_lines_still_key_on_name_and_host():
    """The fallback must not swallow the normal case: two distinct named
    lines from ONE pmid stay two lines."""
    from hitlist.sample_identity import count_distinct_ids, sample_identity_ids

    df = _sample_frame(
        [
            {"pmid": 1, "src_cell_line": True, "cell_line_name": "THP-1"},
            {"pmid": 1, "src_cell_line": True, "cell_line_name": "HeLa"},
            {"pmid": 1, "src_cell_line": True, "monoallelic_host": "C1R"},
        ]
    )
    line_id, _, _ = sample_identity_ids(df)
    assert count_distinct_ids(line_id) == 3


def test_sample_identity_categories_do_not_leak_into_each_other():
    """A cell-line row has no donor ID and vice versa -- blank means "not
    in this category", which is why callers count non-blank rather than
    calling nunique()."""
    from hitlist.sample_identity import count_distinct_ids, sample_identity_ids

    df = _sample_frame(
        [
            {"pmid": 1, "src_cell_line": True, "cell_line_name": "THP-1"},
            {"pmid": 2, "attributed_sample_label": "donor_a"},
        ]
    )
    line_id, donor_id, donor_type_id = sample_identity_ids(df)
    assert count_distinct_ids(line_id) == 1
    assert count_distinct_ids(donor_id) == 1
    assert count_distinct_ids(donor_type_id) == 1
    assert line_id.tolist()[1] == ""  # donor row contributes no line
    assert donor_id.tolist()[0] == ""  # line row contributes no donor


def test_count_distinct_ids_ignores_the_not_in_category_blank():
    from hitlist.sample_identity import count_distinct_ids

    assert count_distinct_ids(pd.Series(["a", "", "b", ""])) == 2
    assert count_distinct_ids(pd.Series(["", ""])) == 0


# ── Public contract + drift guard ─────────────────────────────────────


def test_public_api_surface_is_stable():
    """These names are imported across modules; renaming one is a breaking
    change, not a refactor (the repo's no-private-interfaces rule)."""
    from hitlist import sample_identity as si

    assert si.SAMPLE_IDENTITY_COLUMNS == (
        "pmid",
        "src_cell_line",
        "cell_line_name",
        "cell_name",
        "monoallelic_host",
        "attributed_sample_label",
    )
    assert si.SAMPLE_IDENTITY_ID_COLUMNS == ("line_id", "donor_id", "donor_type_id")
    for name in (
        "sample_identity_ids",
        "add_sample_identity_columns",
        "count_distinct_ids",
        "count_samples",
    ):
        assert callable(getattr(si, name)), name


def test_identity_columns_are_public_not_underscore_prefixed():
    """The attached columns are part of the contract, so they carry public
    names -- an underscore prefix would signal callers may not rely on
    them, which is the opposite of the intent."""
    from hitlist.sample_identity import SAMPLE_IDENTITY_ID_COLUMNS, add_sample_identity_columns

    out = add_sample_identity_columns(_sample_frame([{"pmid": 1}]))
    for col in SAMPLE_IDENTITY_ID_COLUMNS:
        assert not col.startswith("_")
        assert col in out.columns


def test_count_samples_is_lines_plus_donor_cell_types():
    from hitlist.sample_identity import count_samples

    df = _sample_frame(
        [
            {"pmid": 1, "src_cell_line": True, "cell_line_name": "THP-1"},
            {"pmid": 2, "attributed_sample_label": "donor_a", "cell_name": "blood"},
            {"pmid": 2, "attributed_sample_label": "donor_a", "cell_name": "tumour"},
        ]
    )
    # 1 line + 2 (donor, cell-type) profiles from the same donor.
    assert count_samples(df) == 3


def test_no_module_counts_samples_off_the_raw_label():
    """Drift guard.  #502 happened because a second module grew its own
    sample count from ``attributed_sample_label``, which is blank on 96.7%
    of rows. Anything reporting sample counts goes through this module."""
    src = pathlib.Path(__file__).resolve().parents[1] / "hitlist"
    offenders = []
    for path in src.glob("*.py"):
        if path.name in {"sample_identity.py", "scanner.py", "cell_name_parser.py"}:
            continue  # define/populate the field rather than counting it
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if "attributed_sample_label" in line and "nunique" in line:
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "count samples via hitlist.sample_identity, not nunique() on the raw "
        "label (blank on 96.7% of rows):\n" + "\n".join(offenders)
    )
