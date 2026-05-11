"""Tests for ``hitlist.export.apply_winners_vectorized``.

Exercise the strict-equivalence semantics promised by the helper's
docstring — in particular, the distinction between "winner dict has
col=NaN" (write NaN through) and "winner dict missing col" (preserve
obs value).  See PR #245 review for context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hitlist.export import apply_winners_vectorized


def test_empty_winners_no_mutation():
    obs = pd.DataFrame(
        {"pmid": [1, 2], "allele": ["A", "B"], "tissue": ["skin", "lung"], "label": ["x", "y"]}
    )
    before = obs.copy()
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True, True]),
        tiebreak_cols=["pmid", "allele"],
        winners={},
        meta_cols=["tissue", "label"],
    )
    pd.testing.assert_frame_equal(obs, before)


def test_winner_overwrites_matched_row_only():
    obs = pd.DataFrame(
        {
            "pmid": [1, 2],
            "allele": ["A", "B"],
            "tissue": ["OLD", "OLD"],
            "label": ["old", "old"],
        }
    )
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True, True]),
        tiebreak_cols=["pmid", "allele"],
        winners={(1, "A"): {"tissue": "NEW", "label": "new"}},
        meta_cols=["tissue", "label"],
    )
    assert obs.loc[0, "tissue"] == "NEW"
    assert obs.loc[0, "label"] == "new"
    # Row 1's key wasn't in winners → preserved.
    assert obs.loc[1, "tissue"] == "OLD"
    assert obs.loc[1, "label"] == "old"


def test_winner_with_explicit_nan_writes_nan():
    """Original semantics: an explicit NaN in the winner dict writes
    NaN to obs (it doesn't fall back to the existing obs value).

    This is the case where naive ``~pd.isna(matched[col])`` masking
    silently diverges — the presence-tracking design exists to make
    this case airtight.
    """
    obs = pd.DataFrame(
        {"pmid": [1], "allele": ["A"], "tissue": ["OLD"]},
    ).astype({"tissue": object})
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True]),
        tiebreak_cols=["pmid", "allele"],
        winners={(1, "A"): {"tissue": np.nan}},
        meta_cols=["tissue"],
    )
    assert pd.isna(obs.loc[0, "tissue"])  # NaN written through, not OLD


def test_winner_missing_meta_col_preserves_obs_value():
    """Original semantics: when a winner dict omits a meta_col entirely,
    that col's existing obs value is preserved (not overwritten with NaN).
    """
    obs = pd.DataFrame(
        {"pmid": [1], "allele": ["A"], "tissue": ["OLD"], "label": ["OLD"]},
    )
    # Winner has 'tissue' but not 'label'.
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True]),
        tiebreak_cols=["pmid", "allele"],
        winners={(1, "A"): {"tissue": "NEW"}},
        meta_cols=["tissue", "label"],
    )
    assert obs.loc[0, "tissue"] == "NEW"
    assert obs.loc[0, "label"] == "OLD"  # preserved — winner didn't carry it


def test_mask_excludes_rows_from_application():
    """Rows where mask=False are not touched, even if their key matches
    a winner."""
    obs = pd.DataFrame(
        {"pmid": [1, 1], "allele": ["A", "A"], "tissue": ["OLD0", "OLD1"]},
    )
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True, False]),
        tiebreak_cols=["pmid", "allele"],
        winners={(1, "A"): {"tissue": "NEW"}},
        meta_cols=["tissue"],
    )
    assert obs.loc[0, "tissue"] == "NEW"
    assert obs.loc[1, "tissue"] == "OLD1"  # mask=False, untouched


def test_no_winners_match_obs_keys():
    """Winners exist but none of their keys appear in ``obs[mask]`` —
    obs is unchanged."""
    obs = pd.DataFrame(
        {"pmid": [1], "allele": ["A"], "tissue": ["OLD"]},
    )
    before = obs.copy()
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True]),
        tiebreak_cols=["pmid", "allele"],
        winners={(99, "Z"): {"tissue": "NEW"}},
        meta_cols=["tissue"],
    )
    pd.testing.assert_frame_equal(obs, before)


def test_multiple_meta_cols_partial_overlap():
    """Two winners, two meta_cols, each winner sets a different col."""
    obs = pd.DataFrame(
        {
            "pmid": [1, 2],
            "allele": ["A", "B"],
            "tissue": ["OLD_T0", "OLD_T1"],
            "label": ["OLD_L0", "OLD_L1"],
        }
    )
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True, True]),
        tiebreak_cols=["pmid", "allele"],
        winners={
            (1, "A"): {"tissue": "NEW_T0"},  # only tissue
            (2, "B"): {"label": "NEW_L1"},  # only label
        },
        meta_cols=["tissue", "label"],
    )
    assert obs.loc[0, "tissue"] == "NEW_T0"
    assert obs.loc[0, "label"] == "OLD_L0"  # winner didn't set label
    assert obs.loc[1, "tissue"] == "OLD_T1"  # winner didn't set tissue
    assert obs.loc[1, "label"] == "NEW_L1"


def test_duplicate_keys_in_obs_all_get_winner():
    """Multiple obs rows with the same tiebreak tuple all receive the
    winner's value for that tuple."""
    obs = pd.DataFrame(
        {
            "pmid": [1, 1, 2],
            "allele": ["A", "A", "B"],
            "tissue": ["OLD", "OLD", "OLD"],
        }
    )
    apply_winners_vectorized(
        obs,
        mask=pd.Series([True, True, True]),
        tiebreak_cols=["pmid", "allele"],
        winners={(1, "A"): {"tissue": "NEW"}},
        meta_cols=["tissue"],
    )
    assert obs.loc[0, "tissue"] == "NEW"
    assert obs.loc[1, "tissue"] == "NEW"
    assert obs.loc[2, "tissue"] == "OLD"
