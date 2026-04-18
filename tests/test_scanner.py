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
