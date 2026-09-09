"""The flat experimental-condition columns (#450).

Three layers, and the corpus ones are the point.  A validator that rejects a
bad cell proves the shape is enforceable; it does not prove the 761 curated
records are annotated, that the columns survive the observation join, or that
an unresolved row refuses to name an arm.  The audits below assert those
directly against the packaged curation.
"""

from __future__ import annotations

import pytest
import yaml

from hitlist import curation
from hitlist.conditions import (
    CLOSED_CONDITION_VOCABULARIES,
    CONDITION_COLUMNS,
    CONDITION_STATUS_VALUES,
    MULTI_VALUE_CONDITION_COLUMNS,
    NONE_PERMITTED_CONDITION_COLUMNS,
    canonical_condition_token,
    load_condition_vocabulary,
    split_condition_tokens,
)
from hitlist.curation import MS_SAMPLE_FIELDS, load_pmid_overrides
from hitlist.export import (
    _empty_ms_samples_columns,
    generate_ms_samples_table,
    generate_sample_expression_table,
)


def _load_with(tmp_path, monkeypatch, entries):
    """Load a synthetic overrides file through the real loader."""
    path = tmp_path / "pmid_overrides.yaml"
    path.write_text(yaml.safe_dump(entries))
    real = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(path) if fn == "pmid_overrides.yaml" else real(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    return curation.load_pmid_overrides


def _arm(**kwargs):
    arm = {
        "sample_label": "arm",
        "condition": "unperturbed",
        "condition_id": "arm",
        "condition_status": "annotated",
        "condition_evidence": "curated_text",
    }
    arm.update(kwargs)
    return arm


# ── contract ────────────────────────────────────────────────────────────────


def test_registry_is_the_only_declaration():
    """One ordered registry, spliced everywhere — not four hand-kept copies."""
    exported = list(generate_ms_samples_table().columns)
    for column in CONDITION_COLUMNS:
        assert column in MS_SAMPLE_FIELDS, f"{column} is not accepted by the loader"
        assert column in exported, f"{column} reaches no exported column"
    assert _empty_ms_samples_columns() == exported, (
        "the empty-frame schema is hand-mirrored from the row dict; a zero-match "
        "filter must return the same columns as a matching one"
    )


def test_declared_fields_describe_what_reads_them():
    for column in CONDITION_COLUMNS:
        assert MS_SAMPLE_FIELDS[column].strip(), f"{column} has no description"


def test_expression_anchor_export_carries_the_block():
    """The anchors variant is one CLI flag from the plain samples export.

    That tuple has already drifted twice, and each time a flag changed which
    curation a user got back.
    """
    anchors = set(generate_sample_expression_table().columns)
    assert set(CONDITION_COLUMNS) <= anchors


def test_vocabulary_is_yaml_not_python():
    """Gene, drug and organism names are curation data (AGENTS.md)."""
    vocab = load_condition_vocabulary()
    assert vocab, "condition_vocabulary.yaml resolved to nothing"
    assert canonical_condition_token("condition_cytokines", "IFN-gamma") == "IFNG"
    assert canonical_condition_token("condition_knockout_genes", "ERAAP") == "ERAP1"
    # An unknown reported entity survives rather than being dropped for
    # missing a list — that is the whole reason these columns are open.
    assert canonical_condition_token("condition_drugs", "novelcompound") == "novelcompound"


def test_ifn_stays_coarse_when_the_source_was_coarse():
    """`IFN treatment` names a class, not a member.

    Aliasing it to IFNG would manufacture a fact the source did not state,
    which is the failure mode the whole evidence/status split exists for.
    """
    assert canonical_condition_token("condition_cytokines", "IFN") == "IFN"


# ── validation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "arm,expected",
    [
        (_arm(condition_status="fully_annotated"), "condition_status"),
        (_arm(condition_id="Arm One"), "malformed"),
        (_arm(condition_cytokines="TNF;IFNG"), "sorted and unique"),
        (_arm(condition_cytokines="IFNG;IFNG"), "sorted and unique"),
        (_arm(condition_cytokines="IFNG;"), "empty token"),
        (_arm(condition_cytokines="IFN-gamma"), "alias"),
        (_arm(condition_knockout_genes="ERAP1 KO"), "not a gene designation"),
        (_arm(condition_material="cryopreserved"), "condition_material"),
        (_arm(condition_culture="none"), "does not support"),
        (_arm(condition_drugs="none;DMSO"), "mixes 'none'"),
        (_arm(condition_evidence="primary_source"), "no condition_reference"),
        (_arm(condition_reference="doi:10/x Methods"), "condition_reference but"),
        (_arm(condition_status="unreported", condition_evidence="curated_text"), "unreported"),
        (_arm(condition_control_for="ghost"), "not a condition_id in this study"),
        (_arm(condition_control_for="arm"), "its own condition_id"),
    ],
)
def test_malformed_condition_curation_is_rejected(tmp_path, monkeypatch, arm, expected):
    load = _load_with(tmp_path, monkeypatch, [{"pmid": 42, "ms_samples": [arm]}])
    try:
        with pytest.raises(ValueError, match=expected):
            load()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_status_without_evidence_is_rejected(tmp_path, monkeypatch):
    """Normalizing curated wording and reading the paper are different claims."""
    arm = _arm()
    del arm["condition_evidence"]
    load = _load_with(tmp_path, monkeypatch, [{"pmid": 42, "ms_samples": [arm]}])
    try:
        with pytest.raises(ValueError, match="without condition_evidence"):
            load()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_duplicate_condition_id_within_a_study_is_rejected(tmp_path, monkeypatch):
    """`(pmid, condition_id)` is the identity; two arms sharing one are one arm."""
    load = _load_with(
        tmp_path,
        monkeypatch,
        [
            {
                "pmid": 42,
                "ms_samples": [
                    _arm(sample_label="a", condition_id="dup"),
                    _arm(sample_label="b", condition_id="dup"),
                ],
            }
        ],
    )
    try:
        with pytest.raises(ValueError, match="is used by ms_samples"):
            load()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_condition_curation_is_all_or_none_within_a_study(tmp_path, monkeypatch):
    """Half-curated exports blanks indistinguishable from "nobody could tell"."""
    load = _load_with(
        tmp_path,
        monkeypatch,
        [
            {
                "pmid": 42,
                "ms_samples": [
                    _arm(sample_label="a", condition_id="a"),
                    {"sample_label": "b", "condition": "unperturbed"},
                ],
            }
        ],
    )
    try:
        with pytest.raises(ValueError, match="all or none"):
            load()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_a_study_may_opt_out_entirely(tmp_path, monkeypatch):
    load = _load_with(
        tmp_path,
        monkeypatch,
        [{"pmid": 42, "ms_samples": [{"sample_label": "a", "condition": "unperturbed"}]}],
    )
    try:
        assert 42 in load()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_duplicate_yaml_keys_are_rejected(tmp_path, monkeypatch):
    """PyYAML keeps the last and discards the first, silently."""
    path = tmp_path / "pmid_overrides.yaml"
    path.write_text(
        "- pmid: 42\n"
        "  ms_samples:\n"
        '    - sample_label: "a"\n'
        '      mhc: "HLA-A*02:01"\n'
        '      mhc: "HLA-B*07:02"\n'
    )
    real = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(path) if fn == "pmid_overrides.yaml" else real(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(Exception, match="duplicate key"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


# ── the corpus ──────────────────────────────────────────────────────────────


def test_every_curated_arm_carries_an_annotated_context():
    """All 761 records, each with a persistent id and an explicit status."""
    records = [
        (pmid, sample)
        for pmid, entry in load_pmid_overrides().items()
        for sample in (entry.get("ms_samples") or [])
    ]
    assert len(records) == 761, f"expected 761 curated arms, found {len(records)}"
    missing = [
        (pmid, s.get("sample_label"))
        for pmid, s in records
        if not s.get("condition_id") or not s.get("condition_status")
    ]
    assert missing == [], f"arms with no condition context: {missing[:5]}"


def test_condition_ids_are_unique_within_each_study():
    for pmid, entry in load_pmid_overrides().items():
        ids = [s.get("condition_id") for s in (entry.get("ms_samples") or [])]
        assert len(ids) == len(set(ids)), f"PMID {pmid} reuses a condition_id"


def test_every_multi_value_cell_is_canonical():
    """Sorted, unique and alias-free — otherwise equality is not comparable."""
    for pmid, entry in load_pmid_overrides().items():
        for sample in entry.get("ms_samples") or []:
            for column in MULTI_VALUE_CONDITION_COLUMNS:
                tokens = split_condition_tokens(str(sample.get(column) or ""))
                if not tokens or tokens == ["none"]:
                    continue
                assert tokens == sorted(set(tokens)), f"PMID {pmid} {column}={tokens}"
                for token in tokens:
                    assert canonical_condition_token(column, token) == token, (
                        f"PMID {pmid} {column} token {token!r} is an alias"
                    )


def test_every_closed_vocabulary_value_is_declared():
    for pmid, entry in load_pmid_overrides().items():
        for sample in entry.get("ms_samples") or []:
            for column, allowed in CLOSED_CONDITION_VOCABULARIES.items():
                for token in split_condition_tokens(str(sample.get(column) or "")):
                    assert token in allowed, f"PMID {pmid} {column}={token!r}"


def test_none_appears_only_where_absence_is_a_claim():
    for pmid, entry in load_pmid_overrides().items():
        for sample in entry.get("ms_samples") or []:
            for column in CONDITION_COLUMNS:
                if str(sample.get(column) or "") == "none":
                    assert column in NONE_PERMITTED_CONDITION_COLUMNS, (
                        f"PMID {pmid} {column} claims 'none' for a context field"
                    )


def test_unreported_arms_never_claim_a_condition():
    """The four arms that record why they were never run state no experiment.

    An unreported condition must not become untreated: that is the one
    direction that turns a silence into a control arm.
    """
    df = generate_ms_samples_table()
    unreported = df[df["condition_status"] == "unreported"]
    assert len(unreported) == 4, f"expected 4 unreported arms, found {len(unreported)}"
    for column in CONDITION_COLUMNS:
        if column in ("condition_id", "condition_status"):
            continue
        assert (unreported[column].astype(str) == "").all(), f"an unreported arm asserts {column}"
    assert (unreported["profiled"].astype(str) == "false").all()


def test_mixed_arms_assert_no_agent():
    """ "various combinations" establishes no agent for every contributing arm."""
    df = generate_ms_samples_table()
    mixed = df[df["condition_status"] == "mixed"]
    assert not mixed.empty
    agent_cols = [
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_cytokines",
        "condition_drugs",
        "condition_infection",
        "condition_stimulation",
    ]
    for column in agent_cols:
        assert (mixed[column].astype(str) == "").all(), (
            f"a mixed arm claims {column} for alternatives the source does not separate"
        )
    # ...and it is not silently a control either.
    assert (mixed["condition_control"].astype(str) == "").all()


def test_status_vocabulary_is_pinned():
    df = generate_ms_samples_table()
    assert set(df["condition_status"].astype(str)) <= set(CONDITION_STATUS_VALUES)


# ── what the columns are for ────────────────────────────────────────────────


def _arm_row(df, pmid, condition_id):
    rows = df[(df["pmid"] == pmid) & (df["condition_id"] == condition_id)]
    assert len(rows) == 1, f"PMID {pmid} {condition_id!r} matched {len(rows)} rows"
    return rows.iloc[0]


def test_two_cytokines_survive_in_one_row():
    """Both factors, in one cell, without inferring either one's effect."""
    row = _arm_row(generate_ms_samples_table(), 30833945, "a549_lung_cancer_tnfa_ifng")
    assert split_condition_tokens(row["condition_cytokines"]) == ["IFNG", "TNF"]
    assert row["condition_combination"] == "simultaneous"


def test_knockout_and_infection_are_separate_axes():
    """`condition_category` collapses this pair to TAP_perturbation.

    The infection reaches no legacy column at all, which is the loss the
    block exists to stop.
    """
    df = generate_ms_samples_table()
    row = df[
        (df["pmid"] == 39438697)
        & (df["condition_knockout_genes"] == "TAP1")
        & (df["condition_infection"] != "")
    ]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["condition_infection"] == "Mycobacterium tuberculosis"
    assert row["condition_combination"] == "simultaneous"
    assert row["condition_category"] == "TAP_perturbation"  # legacy value unchanged


def test_a_knockdown_does_not_become_a_knockout():
    df = generate_ms_samples_table()
    knockdowns = df[df["condition_knockdown_genes"] != ""]
    assert not knockdowns.empty
    for _, row in knockdowns.iterrows():
        assert "ERAP1" not in split_condition_tokens(row["condition_knockout_genes"]) or (
            "ERAP1" not in split_condition_tokens(row["condition_knockdown_genes"])
        ), "the same gene is both knocked down and knocked out on one arm"
    shrna = df[df["condition"] == "ERAP1 shRNA knockdown"]
    assert (shrna["condition_knockdown_genes"] == "ERAP1").all()
    assert (shrna["condition_knockout_genes"] == "").all()


def test_intrinsic_background_stays_distinct_from_an_intervention():
    """A Hap10 ERAP1 background is not an ERAP1 knockout."""
    df = generate_ms_samples_table()
    hap10 = df[df["condition_background"] == "ERAP1_hap10"]
    assert not hap10.empty
    assert (hap10["condition_knockout_genes"] == "").all()
    assert (hap10["condition_control"] == "untreated").all()


def test_a_vehicle_keeps_both_the_vehicle_and_the_control_role():
    row = _arm_row(generate_ms_samples_table(), 34497125, "skmel5_melanoma_dmso_control")
    assert row["condition_control"] == "vehicle"
    assert row["condition_drugs"] == "DMSO"


def test_explicit_absence_is_not_silence():
    """The arms curated as "no HLA-DM co-transfection" against the 42 with it.

    All 46 are one `condition_category`, because `simplify_condition` blanks
    everything after `unperturbed — `.  The block separates them.
    """
    df = generate_ms_samples_table()
    with_dm = df[df["condition"] == "unperturbed — HLA-DM co-transfected"]
    without_dm = df[df["condition"] == "unperturbed — mono-allelic; no HLA-DM co-transfection"]
    assert len(with_dm) == 42 and len(without_dm) == 4
    assert (with_dm["condition_transfection"] == "HLA-DM").all()
    assert (without_dm["condition_transfection"] == "none").all()
    # The legacy column cannot tell the two apart, which is the point: both
    # sides of a real experimental contrast read as one unperturbed bucket.
    assert set(with_dm["condition_category"]) == {"unperturbed"}
    assert set(without_dm["condition_category"]) == {"unperturbed"}


def test_condition_id_survives_a_display_label_change(monkeypatch):
    """Ids are curated and frozen, never re-derived from a mutable label.

    A label is free to change — several already have.  If the id moved with
    it, an observation attributed to `(pmid, condition_id)` would silently
    point at a different arm after a rename.
    """
    from hitlist import export

    overrides = {
        7: {
            "study_label": "renamed",
            "ms_samples": [
                {
                    "sample_label": "a completely different display name",
                    "condition": "ERAP2 CRISPR KO",
                    "condition_id": "erap2_ko",
                    "condition_status": "annotated",
                    "condition_evidence": "curated_text",
                    "condition_knockout_genes": "ERAP2",
                    "mhc_class": "I",
                }
            ],
        }
    }
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)
    row = export.generate_ms_samples_table().iloc[0]
    assert row["condition_id"] == "erap2_ko"
    assert row["condition_knockout_genes"] == "ERAP2"


# ── the join ────────────────────────────────────────────────────────────────


def test_only_a_resolved_row_names_an_arm(full_observations_df):
    """`condition_id` names one arm, so an ambiguous row must not carry one.

    This is the invariant that keeps the block honest across 4.4M rows: a
    consumer filtering `condition_id == "hap1_erap1_ko"` gets peptides the
    join actually placed in that arm, not peptides from a study that has one.
    """
    df = full_observations_df
    has_id = df["condition_id"].astype(str) != ""
    ambiguous = df["sample_attribution"].astype(str).isin(["pmid_ambiguous", "group_ambiguous"])
    assert not (has_id & ambiguous).any(), "an ambiguous row asserts a condition_id"
    resolved = (
        df["sample_attribution"]
        .astype(str)
        .isin(
            [
                "discriminated",
                "single_sample_pmid",
                "curated_sample_label",
                "elution_conditions",
                "serotype_expansion",
            ]
        )
    )
    assert not (resolved & ~has_id).any(), "a row attributed to one arm carries no condition_id"


def test_condition_columns_are_categorical_on_observations(full_observations_df):
    """23 object columns on 4.4M rows is hundreds of MB of str overhead (#263).

    Every value comes from one of 761 arms, so the category sets are tiny.
    """
    import pandas as pd

    for column in CONDITION_COLUMNS:
        assert isinstance(df_dtype := full_observations_df[column].dtype, pd.CategoricalDtype), (
            f"{column} arrived as {df_dtype}, not a Categorical"
        )


def test_a_study_panel_does_not_reach_its_own_control_arm(full_observations_df):
    """The #353 failure, asked of the new columns.

    The Shapiro panel knocks out 11 genes. Its wild-type arm must carry none
    of them — a model that featurizes the study union as if it were per-sample
    inverts its own control.
    """
    df = full_observations_df
    hap1 = df[df["pmid"] == 40113210]
    if hap1.empty:
        pytest.skip("Shapiro HAP1 panel not present in this build")
    wildtype = hap1[hap1["condition_id"].astype(str) == "hap1_wildtype"]
    if wildtype.empty:
        pytest.skip("no rows reached the wildtype arm in this build")
    # `none` (the primary-source claim that this arm has no knockout) and ""
    # both satisfy the invariant; a panel gene does not.
    assert set(wildtype["condition_knockout_genes"].astype(str)) <= {"", "none"}
    erap1 = hap1[hap1["condition_id"].astype(str) == "hap1_erap1_ko"]
    if not erap1.empty:
        assert (erap1["condition_knockout_genes"].astype(str) == "ERAP1").all()


def test_binding_rows_get_blanks_not_an_untreated_arm():
    """Predicted binders have no MS sample and so no experimental condition.

    Any non-blank default would assert an untreated arm for every binding row
    in the training table — the direction that cancels the perturbed-vs-control
    contrast.
    """
    from hitlist.export import _TRAINING_DEFAULTS

    for column in CONDITION_COLUMNS:
        assert column in _TRAINING_DEFAULTS, f"{column} has no training default"
        assert _TRAINING_DEFAULTS[column] == "", (
            f"{column} defaults to {_TRAINING_DEFAULTS[column]!r} on binding rows"
        )
