"""Sample-level ``override`` and ``note`` reach a consumer (#373).

``generate_ms_samples_table()`` read most ``ms_samples`` keys and silently
dropped three. ``override`` appeared on 13 samples and ``note`` on 3, and
nothing anywhere read either at sample level — the curator's intent existed
only in a field no code consulted.

Two properties matter beyond "the value is exported":

* **Explicit null is not absence.** PMID 34497125 marks two cell-line arms
  ``cell_line`` and its patient-biopsy arm ``override: null``. That null says
  "deliberately no override here", which is a different claim from "this
  sample says nothing, inherit the study's". Collapsing them loses the only
  record that a curator considered the question.
* **Adding metadata may not add evidence.** Carrying these fields through the
  observation join must not change which rows exist or what they say.
"""

from __future__ import annotations

import pytest
import yaml

from hitlist import curation
from hitlist.curation import MS_SAMPLE_FIELDS
from hitlist.export import generate_ms_samples_table


def _row(samples, pmid: int, label_fragment: str):
    matched = samples[
        (samples["pmid"] == pmid) & samples["sample_label"].str.contains(label_fragment)
    ]
    assert len(matched) == 1, f"{label_fragment!r} matched {len(matched)} rows"
    return matched.iloc[0]


# ── override resolution ─────────────────────────────────────────────────────


def test_sample_override_is_exported():
    """A sample-level override no longer evaporates."""
    samples = generate_ms_samples_table()
    arm = _row(samples, 34497125, "SKMEL5 melanoma . binimetinib")
    assert arm["sample_override"] == "cell_line"
    assert arm["effective_override"] == "cell_line"
    assert arm["effective_override_origin"] == "sample"


def test_explicit_null_override_is_distinguishable_from_absence():
    """``override: null`` means "no override *here*", not "inherit".

    PMID 34497125's patient-biopsy arm carries an explicit null alongside two
    ``cell_line`` cell-line arms. Exporting it identically to a sample that
    simply omits the key would erase the distinction the curator drew.
    """
    samples = generate_ms_samples_table()
    biopsies = _row(samples, 34497125, "patient melanoma biopsies")
    assert biopsies["sample_override"] == ""
    assert biopsies["effective_override"] == ""
    assert biopsies["effective_override_origin"] == "sample_null"


def test_absent_sample_override_inherits_the_study_value():
    """PMID 27846572 is ``cell_line`` study-wide; its samples say nothing."""
    samples = generate_ms_samples_table()
    c1r = _row(samples, 27846572, "C1R parental")
    assert c1r["sample_override"] == ""
    assert c1r["effective_override"] == "cell_line"
    assert c1r["effective_override_origin"] == "study"


def test_no_override_anywhere_reports_no_origin():
    """Neither level curated one — say so rather than implying a default."""
    samples = generate_ms_samples_table()
    granta = _row(samples, 34129938, "GRANTA-519")
    # This study has no PMID-level override; the sample carries its own.
    assert granta["effective_override_origin"] == "sample"
    uncurated = samples[samples["effective_override_origin"] == "none"]
    assert len(uncurated) > 0
    assert (uncurated["effective_override"] == "").all()
    assert (uncurated["sample_override"] == "").all()


def test_every_exported_override_is_a_known_value():
    """A typo in the YAML must not travel as a classification claim."""
    samples = generate_ms_samples_table()
    values = set(samples["effective_override"]) | set(samples["sample_override"])
    assert values - {""} <= set(curation.OVERRIDE_VALUES)


# ── note / classification / reason ──────────────────────────────────────────


def test_sample_note_is_exported_separately_from_the_legacy_column():
    """The ccRCC Pat9 caveat describes a sample that *is* in the corpus.

    "No HLA typing — used for proteasomal analysis (Fig. 4) but excluded from
    allele-level validation (Fig. 6)" is an analytic caveat, and it reached no
    consumer at all.
    """
    samples = generate_ms_samples_table()
    pat9 = _row(samples, 31844290, "ccRCC Pat9")
    assert "excluded from allele-level validation" in pat9["note"]


def test_classification_and_reason_are_their_own_columns():
    """``notes`` collapsed two distinct fields into one; keep it, add both."""
    samples = generate_ms_samples_table()
    for column in ("note", "classification", "reason"):
        assert column in samples.columns
    # The legacy column keeps its exact previous meaning: classification if
    # present, else reason.
    populated = samples[(samples["classification"] != "") | (samples["reason"] != "")]
    assert len(populated) > 0
    for _, row in populated.iterrows():
        assert row["notes"] == (row["classification"] or row["reason"])


# ── schema guard ────────────────────────────────────────────────────────────


def test_every_curated_sample_key_is_declared():
    """No ``ms_samples`` key may be silently ignored.

    This is the guard the issue asks for: the audit that found ``override``,
    ``note``, and ``species`` unread should not have needed a human.
    """
    with open(curation._data_path("pmid_overrides.yaml")) as f:
        entries = yaml.safe_load(f)
    used = {key for e in entries for s in (e.get("ms_samples") or []) for key in s}
    assert used - set(MS_SAMPLE_FIELDS) == set()


def test_load_pmid_overrides_rejects_an_unknown_sample_key(tmp_path, monkeypatch):
    """A typo fails at load, not in an audit a year later."""
    bad_yaml = tmp_path / "pmid_overrides.yaml"
    bad_yaml.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 12345678,
                    "ms_samples": [{"sample_label": "a", "mhc_clss": "I"}],
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
        with pytest.raises(ValueError, match="mhc_clss"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_declared_fields_describe_what_reads_them():
    """The mapping is documentation, not just a spelling list."""
    assert all(isinstance(v, str) and v for v in MS_SAMPLE_FIELDS.values())


# ── conservation ────────────────────────────────────────────────────────────


def test_no_unattributed_row_carries_a_sample_level_override(full_observations_df):
    """A sample-origin value on a row with no sample would be a fabrication.

    Study-origin values on unattributed rows are fine and expected: the
    ``pmid_ambiguous`` tier matches a row to its PMID without resolving which
    arm produced it, so ``effective_override_origin == "study"`` there is a
    truthful statement about where the value came from.  What must never
    appear is ``"sample"`` or ``"sample_null"`` — those claim a specific arm.
    """
    df = full_observations_df
    for column in ("sample_override", "effective_override", "effective_override_origin"):
        assert column in df.columns
    unattributed = df[df["sample_label"].astype(str) == ""]
    origins = set(unattributed["effective_override_origin"].astype(str))
    assert origins & {"sample", "sample_null"} == set()
    assert (unattributed["sample_override"].astype(str) == "").all()


def test_study_origin_values_agree_with_the_study_entry(full_observations_df):
    """``origin == "study"`` must actually match that PMID's curated value."""
    from hitlist.curation import load_pmid_overrides

    overrides = load_pmid_overrides()
    df = full_observations_df
    study_rows = df[df["effective_override_origin"].astype(str) == "study"]
    assert len(study_rows) > 0
    sampled = study_rows.drop_duplicates(subset=["pmid"])
    for _, row in sampled.iterrows():
        expected = overrides[int(row["pmid"])].get("override") or ""
        assert str(row["effective_override"]) == expected
