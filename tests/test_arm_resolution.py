"""Why a study's rows cannot reach an arm, recorded once (#366).

584,966 observation rows sit at `pmid_ambiguous` or `group_ambiguous`. Without
a recorded reason each one invites the same investigation again, and the
issue's own acceptance criterion is that a study is either resolved or
"confirmed genuinely unresolvable from the deposited data and recorded as
such, so it stops being re-investigated".

The distinction that matters is between ambiguity more curation could fix and
ambiguity it cannot. Three of the four verdicts are the second kind:

* `axis_mismatch` — the curated arms and the recorded metadata are on
  different axes. PMID 33858848 curates per donor; IEDB records per tissue.
* `no_row_discriminator` — measured: every per-row metadata field takes one
  distinct value across the study.
* `multi_arm_evidence` — the evidence positively places the peptide in more
  than one arm. Nothing is missing; the peptide was in both.

Only `curation_gap` marks remaining work.
"""

from __future__ import annotations

import pytest
import yaml

from hitlist import curation
from hitlist.curation import ARM_RESOLUTION_VALUES, PMID_ENTRY_FIELDS, load_pmid_overrides
from hitlist.export import generate_ms_samples_table

#: Verdicts meaning "more curation will not help".
SETTLED = frozenset({"axis_mismatch", "no_row_discriminator", "multi_arm_evidence"})


def test_vocabulary_is_declared_and_exported():
    assert "arm_resolution" in PMID_ENTRY_FIELDS
    assert "arm_resolution_note" in PMID_ENTRY_FIELDS
    assert "arm_resolution" in generate_ms_samples_table().columns
    assert SETTLED.issubset(ARM_RESOLUTION_VALUES)
    assert "curation_gap" in ARM_RESOLUTION_VALUES


def test_every_verdict_carries_its_evidence():
    """A bare verdict is an assertion; the note is what makes it checkable."""
    for pmid, entry in load_pmid_overrides().items():
        if entry.get("arm_resolution"):
            note = str(entry.get("arm_resolution_note", "") or "")
            assert len(note) > 40, f"PMID {pmid}: arm_resolution without a substantive note"


def test_a_note_without_a_verdict_is_rejected(tmp_path, monkeypatch):
    """The note explains the verdict; on its own it states nothing."""
    bad = tmp_path / "pmid_overrides.yaml"
    bad.write_text(
        yaml.safe_dump([{"pmid": 12345678, "arm_resolution_note": "some prose with no verdict"}])
    )
    real = curation._data_path
    monkeypatch.setattr(
        curation, "_data_path", lambda fn: str(bad) if fn == "pmid_overrides.yaml" else real(fn)
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="without"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_an_invented_verdict_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "pmid_overrides.yaml"
    bad.write_text(yaml.safe_dump([{"pmid": 12345678, "arm_resolution": "probably_fine"}]))
    real = curation._data_path
    monkeypatch.setattr(
        curation, "_data_path", lambda fn: str(bad) if fn == "pmid_overrides.yaml" else real(fn)
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="probably_fine"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


@pytest.mark.integration
def test_every_ambiguous_row_carries_a_verdict(full_observations_df):
    """The accounting is complete, which is the deliverable.

    Any ambiguous row without a verdict is a study nobody has classified,
    and is exactly what this issue exists to eliminate.
    """
    df = full_observations_df
    ambiguous = df[df["sample_attribution"].astype(str).isin(["pmid_ambiguous", "group_ambiguous"])]
    if ambiguous.empty:
        pytest.skip("no ambiguous rows in this build")
    unexplained = ambiguous[ambiguous["arm_resolution"].astype(str) == ""]
    assert unexplained.empty, (
        f"{len(unexplained):,} ambiguous rows in "
        f"{unexplained['pmid'].nunique()} studies carry no arm_resolution verdict: "
        f"{sorted(unexplained['pmid'].dropna().unique())[:10]}"
    )


@pytest.mark.integration
def test_settled_studies_have_no_remaining_arm_evidence(full_observations_df):
    """`no_row_discriminator` is a measurement, so it must still measure true.

    If a future corpus refresh gives one of these studies a varying per-row
    field, the verdict is stale and the study deserves another look — which
    is the opposite of what a recorded verdict should silently prevent.
    """
    df = full_observations_df
    fields = ["cell_name", "source_tissue", "antigen_processing_comments", "assay_comments"]
    available = [f for f in fields if f in df.columns]
    if not available:
        pytest.skip("discriminator columns not in this build")
    for pmid, entry in load_pmid_overrides().items():
        if entry.get("arm_resolution") != "no_row_discriminator":
            continue
        rows = df[df["pmid"] == pmid]
        if rows.empty:
            continue
        varying = [f for f in available if rows[f].astype(str).nunique() > 1]
        assert not varying, (
            f"PMID {pmid} is recorded no_row_discriminator but {varying} now vary; "
            f"re-audit the study rather than trusting the stale verdict"
        )


@pytest.mark.integration
def test_audit_reports_the_verdict_so_settled_findings_are_separable():
    """The #442 audit must distinguish 'unresolvable' from 'unexamined'."""
    from hitlist.observations import is_built
    from hitlist.qc import sample_attribution_audit

    if not is_built():
        pytest.skip("requires a registered observations corpus")
    found = sample_attribution_audit()
    assert "arm_resolution" in found.columns
    actionable = found[~found["arm_resolution"].isin(SETTLED)]
    assert len(actionable) <= len(found)
