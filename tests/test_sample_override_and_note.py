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
from hitlist.curation import MS_SAMPLE_FIELDS, load_pmid_overrides
from hitlist.export import generate_ms_samples_table


def _row(samples, pmid: int, label_fragment: str):
    """One sample row, matched on a *literal* label substring.

    ``regex=False`` on purpose: the fragments are real label text and
    contain ``+`` and ``(``.  With the default regex mode a fragment like
    ``"SKMEL5 melanoma + binimetinib"`` is not even a valid pattern, and
    papering over that with ``.`` would also match labels that do not
    exist, surfacing as a confusing count failure rather than a missing
    label.
    """
    matched = samples[
        (samples["pmid"] == pmid)
        & samples["sample_label"].str.contains(label_fragment, regex=False)
    ]
    assert len(matched) == 1, f"{label_fragment!r} matched {len(matched)} rows"
    return matched.iloc[0]


# ── override resolution ─────────────────────────────────────────────────────


def test_sample_override_is_exported():
    """A sample-level override no longer evaporates."""
    samples = generate_ms_samples_table()
    arm = _row(samples, 34497125, "SKMEL5 melanoma + binimetinib (MEKi)")
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
    """A study with no ``rules`` inherits cleanly.

    Asserted over the whole ``study`` cohort rather than one named arm:
    which studies carry a sample-level override is curation that moves,
    and this contract does not depend on any particular one.
    """
    samples = generate_ms_samples_table()
    overrides = load_pmid_overrides()
    study_only = samples[samples["effective_override_origin"] == "study"]
    assert len(study_only) > 0
    # Origin "study" means the arm declared nothing and inherited a value.
    assert (study_only["sample_override"] == "").all()
    assert (study_only["effective_override"] != "").all()
    for _, row in study_only.iterrows():
        assert row["effective_override"] == overrides[int(row["pmid"])]["override"]


def test_row_conditional_rules_downgrade_the_inherited_origin():
    """``rules`` match per row, so a sample cannot claim the study value.

    PMID 27846572 inherits ``cell_line`` while its rule routes every
    Direct Ex Vivo fibroblast row to ``healthy``.  Reporting plain
    ``study`` there asserts a value the build contradicts on 3,614 rows,
    so the origin says the default is conditional instead.
    """
    samples = generate_ms_samples_table()
    fibroblasts = _row(samples, 27846572, "primary fibroblasts")
    assert fibroblasts["effective_override"] == "cell_line"
    assert fibroblasts["effective_override_origin"] == "study_conditional"

    overrides = load_pmid_overrides()
    conditional = samples[samples["effective_override_origin"] == "study_conditional"]
    assert len(conditional) > 0
    for pmid in conditional["pmid"].unique():
        assert overrides[int(pmid)].get("rules"), f"PMID {pmid} has no rules"
    plain = samples[samples["effective_override_origin"] == "study"]
    for pmid in plain["pmid"].unique():
        assert not overrides[int(pmid)].get("rules"), f"PMID {pmid} has rules"


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


#: Declared keys whose exported column carries a different name.
_SAMPLE_FIELD_COLUMN_ALIASES = {
    # Exported as the resolved pair, because the sample's own claim and the
    # value that survives inheritance are different facts (#373).
    "override": "sample_override",
}

#: Sample keys that legitimately reach no exported column at all.
_UNEXPORTED_SAMPLE_FIELDS = frozenset(
    {
        "type",  # deprecated spelling; warned about, never read
    }
)


def test_declared_fields_describe_what_reads_them():
    """The mapping is documentation, not just a spelling list."""
    assert all(isinstance(v, str) and v for v in MS_SAMPLE_FIELDS.values())


def test_every_declared_field_actually_reaches_a_consumer():
    """ "Declared" must not become a synonym for "accepted and ignored".

    The guard rejects an *undeclared* key, which stops a typo — but on its
    own it would happily accept a new field added to the mapping with no
    reader, which is the exact failure #373 is about.  Tie the declaration
    to the exported column set so adding a field without a consumer fails.
    """
    exported = set(generate_ms_samples_table().columns)
    undelivered = {
        field
        for field in MS_SAMPLE_FIELDS
        if _SAMPLE_FIELD_COLUMN_ALIASES.get(field, field) not in exported
        and field not in _UNEXPORTED_SAMPLE_FIELDS
    }
    assert undelivered == set(), (
        f"declared but reaching no exported column: {sorted(undelivered)}. "
        f"Add the reader, or list it in _UNEXPORTED_SAMPLE_FIELDS with why."
    )


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
    for column in ("effective_override", "effective_override_origin", "sample_note"):
        assert column in df.columns
    # sample_override is deliberately not carried here — derivable, and a
    # redundant object column is expensive across 4.4M rows.
    assert "sample_override" not in df.columns
    unattributed = df[df["sample_label"].astype(str) == ""]
    origins = set(unattributed["effective_override_origin"].astype(str))
    assert origins & {"sample", "sample_null"} == set()
    # An arm's free text is arm-specific, so it cannot survive either.
    assert (unattributed["sample_note"].astype(str) == "").all()


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


def test_consensus_meta_drops_arm_specific_claims():
    """Agreement between arms is not licence to name one.

    Several arms of a study routinely share a sample-level ``override`` —
    PMID 34129938 marks all six ``cell_line`` — so a plain consensus rule
    keeps ``effective_override_origin == "sample"`` on a row whose
    ``sample_label`` it just blanked. That is a statement about a specific
    arm attached to evidence with no arm. A *study*-origin value is a
    property of the deposit and legitimately survives.
    """
    from hitlist.export import _consensus_meta

    meta_cols = [
        "sample_label",
        "effective_override",
        "effective_override_origin",
        "note",
        "instrument",
    ]

    def _cand(label, origin, override):
        return (
            "",
            "",
            {
                "sample_label": label,
                "effective_override": override,
                "effective_override_origin": origin,
                "note": "an arm-specific caveat",
                "instrument": "Exploris 480",
            },
        )

    agreed_sample = _consensus_meta(
        [_cand("arm a", "sample", "cell_line"), _cand("arm b", "sample", "cell_line")],
        meta_cols,
    )
    assert agreed_sample["effective_override"] == ""
    assert agreed_sample["effective_override_origin"] == ""
    assert agreed_sample["note"] == ""
    # What the arms genuinely share still survives.
    assert agreed_sample["instrument"] == "Exploris 480"
    assert agreed_sample["sample_attribution"] == "pmid_ambiguous"

    agreed_study = _consensus_meta(
        [_cand("arm a", "study", "cell_line"), _cand("arm b", "study", "cell_line")],
        meta_cols,
    )
    assert agreed_study["effective_override"] == "cell_line"
    assert agreed_study["effective_override_origin"] == "study"
