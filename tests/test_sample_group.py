"""Resolving the sample *system* before the experimental arm (#359).

Several studies curate a treated and an untreated arm of one system. IEDB's
narrative fields name the system on every row and never mention treatment,
so the scorer — which reads `sample_label` and `perturbation` as one bag of
tokens — cannot tell a system word from a condition word. Two consequences,
both observed:

* Extra identifying words on one arm of a pair decide the pair. PMID
  29242379's untreated UWB arm alone carried "(ovarian carcinoma)", and
  `ovarian` matching `source_tissue = "Ovary"` handed it all 3,919 UWB
  rows with nothing about treatment in evidence. PMID 30833945 lost 4,676
  rows the same way, via `lung`.
* Because the narrative fields are blocked wholesale whenever arms
  disagree, a system whose arm is unambiguous is not attributable either —
  PMID 29242379's TIL and meningioma systems have one arm each.

`sample_group` splits the two questions. The system stage admits the
narrative fields, because identifying the system is exactly what they do
well; the arm stage keeps blocking them.
"""

from __future__ import annotations

import pytest
import yaml

from hitlist import curation
from hitlist.curation import MS_SAMPLE_FIELDS, load_pmid_overrides
from hitlist.export import (
    _identifier_tokens,
    _select_group,
    generate_ms_samples_table,
)


def _cands(*specs):
    """Candidate tuples of the shape the join builds: (label, perturbation, meta)."""
    return [
        (label, perturbation, {"sample_group": group, "condition_category": perturbation})
        for label, perturbation, group in specs
    ]


# ── the group selector ──────────────────────────────────────────────────────


def test_narrative_field_identifies_the_system():
    """The field the arm stage refuses is the right evidence for this stage."""
    chosen = _select_group(
        _cands(
            ("UWB.1 289 untreated", "unperturbed", "UWB.1 289 (ovarian carcinoma)"),
            ("UWB.1 289 + IFN-gamma", "IFN_gamma_treatment", "UWB.1 289 (ovarian carcinoma)"),
            ("melanoma TILs", "other_perturbation", "melanoma tumor infiltrating lymphocytes"),
        ),
        cell_name="Lymphocyte",
        source_tissue="Blood",
        antigen_processing_comments="",
        assay_comments=(
            "The epitope was eluted from human tumor infiltrating lymphocytes "
            "of melanoma tumors (TILs)."
        ),
    )
    assert chosen == "melanoma tumor infiltrating lymphocytes"


def test_a_tie_between_systems_declines_rather_than_guessing():
    """Two systems scoring equally is not evidence for either."""
    assert (
        _select_group(
            _cands(
                ("arm a", "unperturbed", "system one"),
                ("arm b", "unperturbed", "system two"),
            ),
            cell_name="",
            source_tissue="",
            antigen_processing_comments="",
            assay_comments="nothing identifying here",
        )
        is None
    )


def test_a_single_system_is_not_a_discrimination():
    """Naming the only system says no more than naming the PMID.

    Returning it would relabel genuinely arm-ambiguous rows from
    `pmid_ambiguous` to `group_ambiguous` on no evidence, and break any
    consumer partitioning on the former to find undetermined arms.
    """
    assert (
        _select_group(
            _cands(
                ("A549 untreated", "unperturbed", "A549 lung cancer"),
                ("A549 + TNFa + IFNg", "other_perturbation", "A549 lung cancer"),
            ),
            cell_name="A549-Epithelial cell",
            source_tissue="Lung",
            antigen_processing_comments="",
            assay_comments="",
        )
        is None
    )


def test_an_ungrouped_study_opts_out():
    """Absent `sample_group`, the study takes the pre-existing path."""
    assert (
        _select_group(
            [("arm a", "unperturbed", {}), ("arm b", "unperturbed", {})],
            cell_name="whatever",
            source_tissue="",
            antigen_processing_comments="",
            assay_comments="",
        )
        is None
    )


# ── curation contract ───────────────────────────────────────────────────────


def test_sample_group_is_declared():
    assert "sample_group" in MS_SAMPLE_FIELDS
    assert "sample_group" in generate_ms_samples_table().columns


def test_grouping_is_all_or_nothing_within_a_study(tmp_path, monkeypatch):
    """A half-grouped study silently falls back — so refuse to load one."""
    bad_yaml = tmp_path / "pmid_overrides.yaml"
    bad_yaml.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 12345678,
                    "ms_samples": [
                        {"sample_label": "grouped", "sample_group": "sys"},
                        {"sample_label": "ungrouped"},
                    ],
                }
            ]
        )
    )
    real_data_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(bad_yaml) if fn == "pmid_overrides.yaml" else real_data_path(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="all or none"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_paired_arms_of_one_system_are_labelled_symmetrically():
    """The asymmetry is the bug; a group with arms must not reintroduce it.

    Every arm of a multi-arm group must contain its group name, so no arm
    carries identifying words its partner lacks.
    """
    samples = generate_ms_samples_table()
    grouped = samples[samples["sample_group"] != ""]
    assert len(grouped) > 0
    sizes = grouped.groupby(["pmid", "sample_group"]).size()
    multi = sizes[sizes > 1].index
    assert len(multi) > 0
    for pmid, group in multi:
        arms = grouped[(grouped["pmid"] == pmid) & (grouped["sample_group"] == group)]
        for _, arm in arms.iterrows():
            label = str(arm["sample_label"])
            assert group in label, (
                f"PMID {pmid}: arm {label!r} does not carry its group name {group!r}; "
                f"an arm with extra identifying words wins on them alone"
            )
            # Carrying the group name is necessary but not sufficient: the
            # regression this guards against is one arm holding identifying
            # words its partner lacks. Whatever remains after removing the
            # group name must be arm text — the perturbation or an untreated
            # marker — and never a further identifier.
            remainder = _identifier_tokens(label.replace(group, " "))
            partners = [
                str(other["sample_label"])
                for _, other in arms.iterrows()
                if str(other["sample_label"]) != label
            ]
            for partner in partners:
                unique = remainder - _identifier_tokens(partner.replace(group, " "))
                arm_words = _identifier_tokens(
                    f"{arm['condition']} {arm['perturbation']} untreated control"
                )
                stray = {t for t in unique - arm_words if not t.isdigit()}
                assert not stray, (
                    f"PMID {pmid}: arm {label!r} carries {sorted(stray)}, which its "
                    f"partner {partner!r} lacks and which is not condition text. That "
                    f"asymmetry is what decided the arm before #359."
                )


# ── end-to-end on the corpus ────────────────────────────────────────────────


@pytest.mark.integration
def test_single_arm_systems_attribute_and_paired_ones_report_the_system(full_observations_df):
    """PMID 29242379 is the whole shape of #359 in one study.

    Its TIL and meningioma systems have one curated arm each and are now
    attributable; its B-LCL and UWB systems have two, so the system is
    reported and the arm is not. Before this, the only attributed rows in
    the study were 3,919 assigned to the wrong arm.
    """
    df = full_observations_df
    study = df[df["pmid"] == 29242379]
    if study.empty:
        pytest.skip("Chong 2018 not present in this build")

    labels = study["sample_label"].astype(str)
    groups = study["sample_group"].astype(str)
    attribution = study["sample_attribution"].astype(str)

    # Single-arm systems reach a specific arm.
    assert (labels == "melanoma TILs (expanded)").sum() > 0
    assert (labels == "primary meningioma tissue").sum() > 0

    # Paired systems report the system and withhold the arm.
    for group in ("B-LCLs (JY, CD165, PD42, CM467, RA957)", "UWB.1 289 (ovarian carcinoma)"):
        rows = study[groups == group]
        assert len(rows) > 0
        assert (rows["sample_attribution"].astype(str) == "group_ambiguous").all()
        assert (rows["sample_label"].astype(str) == "").all()

    # No row is attributed to a specific arm of a paired system.
    assert not labels.str.contains("untreated").any()
    assert not labels.str.contains("IFN-gamma").any()
    assert "group_ambiguous" in set(attribution)


@pytest.mark.integration
def test_group_ambiguous_rows_never_name_an_arm(full_observations_df):
    """The whole point: system known, arm withheld — not arm guessed."""
    df = full_observations_df
    rows = df[df["sample_attribution"].astype(str) == "group_ambiguous"]
    if rows.empty:
        pytest.skip("no grouped studies in this build")
    assert (rows["sample_label"].astype(str) == "").all()
    assert (rows["sample_group"].astype(str) != "").all()
    # Every reported group must be one a curator actually wrote.
    curated = {
        str(s.get("sample_group", "") or "")
        for entry in load_pmid_overrides().values()
        for s in (entry.get("ms_samples") or [])
    }
    assert set(rows["sample_group"].astype(str)) <= curated


def test_attribution_vocabulary_covers_what_the_join_emits():
    """The documented value set drifted once; pin it to one constant.

    `group_ambiguous` shipped in 1.59.0 without being added to the
    docstring's exhaustive list, so a consumer validating against the
    documented set would have dropped every grouped row.
    """
    from hitlist.export import SAMPLE_ATTRIBUTION_VALUES

    assert "group_ambiguous" in SAMPLE_ATTRIBUTION_VALUES
    assert "" in SAMPLE_ATTRIBUTION_VALUES
    assert len(set(SAMPLE_ATTRIBUTION_VALUES)) == len(SAMPLE_ATTRIBUTION_VALUES)


def test_one_arm_per_group_is_rejected(tmp_path, monkeypatch):
    """A 1:1 group/arm mapping turns the group stage into an arm selector.

    That would let narrative fields pick an arm — exactly what #354
    forbids — with no error and nothing in the output to show for it.
    """
    bad = tmp_path / "pmid_overrides.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 12345678,
                    "ms_samples": [
                        {"sample_label": "A549 untreated", "sample_group": "A549 untreated"},
                        {"sample_label": "A549 + TNFa", "sample_group": "A549 + TNFa"},
                    ],
                }
            ]
        )
    )
    real = curation._data_path
    monkeypatch.setattr(
        curation, "_data_path", lambda fn: str(bad) if fn == "pmid_overrides.yaml" else real(fn)
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="one arm"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()
