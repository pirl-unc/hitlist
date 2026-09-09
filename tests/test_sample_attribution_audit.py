"""Curated arms that reach no evidence, and the study-level schema guard.

A curated ``ms_samples`` entry is a claim that an experimental arm exists.
232 of 684 profiled arms are never attributed to a single observation row
(#442), and until now nothing told a curator whether an arm they added
landed. These tests pin the audit that surfaces them.

The same module carries the study-level schema guard. #373 closed the
sample level; the study level stayed open, and four keys were curated with
no reader — most seriously ``exclude_from_ms``, documented as excluding a
study from the MS index, set on 11 non-MS studies, honored by nothing
(#444).
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from hitlist import curation
from hitlist.curation import PMID_ENTRY_FIELDS
from hitlist.qc import _SAMPLE_ATTRIBUTION_AUDIT_COLUMNS, sample_attribution_audit


def _samples(*rows) -> pd.DataFrame:
    """Minimal curated-sample frame with the columns the audit reads."""
    return pd.DataFrame(
        [
            {
                "pmid": pmid,
                "study_label": "Study",
                "sample_label": label,
                "mhc_class": "I",
                "condition": "unperturbed",
                "profiled": profiled,
            }
            for pmid, label, profiled in rows
        ]
    )


def _observations(*rows) -> pd.DataFrame:
    return pd.DataFrame([{"pmid": pmid, "sample_label": label} for pmid, label in rows])


# ── the audit ───────────────────────────────────────────────────────────────


def test_orphan_in_a_working_study_is_a_label_mismatch_candidate():
    """A sibling arm attributing is evidence the join works here.

    PMID 31844290 attributes 86 of its 107 arms, so the 21 that fail are
    not a broken study — they are 21 suspect labels.
    """
    found = sample_attribution_audit(
        samples=_samples((1, "works", "true"), (1, "orphan", "true")),
        observations=_observations((1, "works"), (1, "works")),
    )
    assert list(found["sample_label"]) == ["orphan"]
    row = found.iloc[0]
    assert row["bucket"] == "label_mismatch_candidate"
    assert row["severity"] == "warn"
    assert row["n_study_arms"] == 2
    assert row["n_attributed_arms"] == 1


def test_study_with_nothing_attributed_is_its_own_bucket():
    """A different failure, needing per-study triage rather than per-label."""
    found = sample_attribution_audit(
        samples=_samples((2, "arm a", "true"), (2, "arm b", "true")),
        observations=_observations((2, ""), (2, "")),
    )
    assert set(found["bucket"]) == {"study_unattributed"}
    assert set(found["severity"]) == {"info"}
    assert (found["n_attributed_arms"] == 0).all()


def test_unprofiled_arms_are_not_reported():
    """``profiled: false`` arms are supposed to reach no row (#437).

    Counting them as attribution failures would report the four records
    #437 deliberately preserved as defects every time the audit runs.
    """
    found = sample_attribution_audit(
        samples=_samples((3, "never run", "false"), (3, "works", "true")),
        observations=_observations((3, "works")),
    )
    assert found.empty


def test_arms_in_studies_absent_from_the_corpus_are_not_reported():
    """A study missing from this build is not an attribution failure."""
    found = sample_attribution_audit(
        samples=_samples((4, "arm", "true")),
        observations=_observations((5, "other")),
    )
    assert found.empty


def test_empty_inputs_return_a_well_formed_frame():
    """A caller doing ``df.groupby('bucket')`` must not hit a KeyError."""
    found = sample_attribution_audit(
        samples=pd.DataFrame(columns=["pmid", "sample_label", "profiled"]),
        observations=pd.DataFrame(columns=["pmid", "sample_label"]),
    )
    assert list(found.columns) == _SAMPLE_ATTRIBUTION_AUDIT_COLUMNS
    assert found.empty


def test_cli_routes_sample_attribution(monkeypatch, capsys):
    from hitlist.cli import main

    canned = pd.DataFrame(
        [
            {"pmid": 1, "bucket": "label_mismatch_candidate", "severity": "warn"},
            {"pmid": 2, "bucket": "study_unattributed", "severity": "info"},
        ]
    )
    monkeypatch.setattr("hitlist.qc.sample_attribution_audit", lambda: canned)
    monkeypatch.setattr(
        "sys.argv", ["hitlist", "qc", "sample-attribution", "--bucket", "study_unattributed"]
    )
    main()
    out = capsys.readouterr().out
    assert "study_unattributed" in out
    assert "label_mismatch_candidate" not in out


@pytest.mark.integration
def test_real_corpus_orphan_count_is_bucketed(full_observations_df):
    """The audit runs on the real corpus and both shapes are present.

    Feeds the shared fixture in rather than letting the audit build its own
    table: an unparameterized call materializes a second 4.4M-row frame in
    the worker on top of the mmapped one, which OOM-killed CI's coverage job
    (exit 143).  The injectable parameter exists for this.
    """
    found = sample_attribution_audit(observations=full_observations_df)
    assert len(found) > 0
    assert set(found["bucket"]) == {"label_mismatch_candidate", "study_unattributed"}
    # Every finding must name a study that has rows and an arm that has none.
    assert (found["n_study_arms"] > 0).all()
    assert (found["n_attributed_arms"] >= 0).all()


# ── study-level schema guard (#444) ─────────────────────────────────────────


def test_every_study_level_key_is_declared():
    """The guard #373 gave `ms_samples`, one level up.

    Its absence is why ``exclude_from_ms`` could be documented, curated on
    11 studies, and read by nothing.
    """
    with open(curation._data_path("pmid_overrides.yaml")) as f:
        entries = yaml.safe_load(f)
    used = {key for entry in entries for key in entry}
    assert used - set(PMID_ENTRY_FIELDS) == set()


def test_load_pmid_overrides_rejects_an_unknown_study_key(tmp_path, monkeypatch):
    bad_yaml = tmp_path / "pmid_overrides.yaml"
    bad_yaml.write_text(yaml.safe_dump([{"pmid": 12345678, "studyy_label": "typo"}]))
    real_data_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(bad_yaml) if fn == "pmid_overrides.yaml" else real_data_path(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="studyy_label"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_count_fields_carry_a_unit_suffix():
    """House rule: never a bare `samples:` / `tissues:` count.

    Both were renamed when the guard went in; this keeps them renamed.
    """
    assert "n_samples" in PMID_ENTRY_FIELDS
    assert "n_tissues" in PMID_ENTRY_FIELDS
    assert "samples" not in PMID_ENTRY_FIELDS
    assert "tissues" not in PMID_ENTRY_FIELDS


def test_unread_fields_are_declared_as_unread():
    """A field with no reader must not be described as if it worked.

    The guard has to accept these keys or the packaged YAML stops
    loading, so the honest thing is to accept them and say so.
    """
    for field in ("donors", "exclude_from_ms"):
        assert "UNREAD" in PMID_ENTRY_FIELDS[field]
        assert "#444" in PMID_ENTRY_FIELDS[field]


# ── #362: the two APM levels stay separate through the join ─────────────────


@pytest.mark.integration
def test_per_sample_and_study_apm_levels_stay_separate_on_observations(full_observations_df):
    """A WT control inside a KO study must not inherit the panel's genes.

    #353 fixed this in `apm_columns_for_sample`; #362 asked whether it
    survives to the per-row export now that both levels are exposed there.
    It does, and this pins it: the Shapiro HAP1 panel's `HAP1 wildtype`
    rows carry no per-sample gene while still reporting the study panel.
    """
    df = full_observations_df
    hap1 = df[df["pmid"] == 40113210]
    if hap1.empty:
        pytest.skip("Shapiro HAP1 panel not present in this build")
    wildtype = hap1[hap1["sample_label"].astype(str) == "HAP1 wildtype"]
    assert len(wildtype) > 0
    assert (wildtype["apm_genes_perturbed"].astype(str) == "").all()
    assert (wildtype["apm_perturbed"].astype(str) == "false").all()
    # The panel context is still queryable, just not as a sample-level fact.
    assert wildtype["study_apm_perturbed"].all()
    ko = hap1[hap1["sample_label"].astype(str) == "HAP1 ERAP1 KO"]
    assert (ko["apm_genes_perturbed"].astype(str) == "erap1").all()
