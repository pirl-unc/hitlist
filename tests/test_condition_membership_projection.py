"""Condition edges must preserve source identity and join unambiguous arms."""

import pandas as pd
import pytest

from hitlist.export import expand_condition_memberships


@pytest.fixture
def membership_frames():
    observations = pd.DataFrame(
        {
            "pmid": [123, 123, 123],
            "peptide": ["ACDEFGHIK", "KLMNPQRST", "ACDEFGHIK"],
            "provenance_id": ["p0", "p1", "p2"],
            "supported_condition_ids": ["treated;control", "treated", "treated"],
            "excluded_condition_ids": ["", "control", ""],
        },
        index=["repeated", 5, "repeated"],
    )
    samples = pd.DataFrame(
        {
            "pmid": [123, 123],
            "condition_id": ["control", "treated"],
            "condition_drugs": ["none", "drug"],
            "mhc": ["HLA-A*02:01", "HLA-A*02:01"],
        }
    )
    return observations, samples


def test_membership_positions_preserve_duplicate_and_mixed_index_labels(membership_frames):
    observations, samples = membership_frames
    edges = expand_condition_memberships(observations, samples_df=samples)
    assert edges.source_observation_position.tolist() == [0, 0, 1, 1, 2]
    assert edges.source_observation_index.tolist() == ["repeated", "repeated", 5, 5, "repeated"]
    assert edges.provenance_id.tolist() == ["p0", "p0", "p1", "p1", "p2"]
    assert edges.condition_drugs.tolist() == ["none", "drug", "drug", "none", "drug"]
    assert edges.membership_status.tolist() == ["detected"] * 3 + [
        "explicit_not_detected",
        "detected",
    ]
    assert len(observations) == 3
    # Repeating identical metadata is harmless; conflicting claims are not.
    pd.testing.assert_frame_equal(
        edges, expand_condition_memberships(observations, samples_df=pd.concat([samples, samples]))
    )


@pytest.mark.parametrize(
    "defect",
    [
        "conflict",
        "unknown",
        "missing_pmid",
        "invalid_pmid",
        "fractional_pmid",
        "no_metadata",
        "contradiction",
        "duplicate",
    ],
)
def test_membership_projection_rejects_unqualified_edges(membership_frames, defect):
    observations, samples = membership_frames
    if defect == "conflict":
        samples = pd.concat([samples, samples.iloc[[0]].assign(condition_drugs="different")])
    elif defect == "unknown":
        observations.iloc[0, observations.columns.get_loc("supported_condition_ids")] = "unknown"
    elif defect == "missing_pmid":
        observations = observations.drop(columns="pmid")
    elif defect == "invalid_pmid":
        observations["pmid"] = None
    elif defect == "fractional_pmid":
        observations["pmid"] = 123.5
    elif defect == "no_metadata":
        samples = samples.iloc[:0]
    elif defect == "contradiction":
        observations["excluded_condition_ids"] = "treated"
    else:
        observations["supported_condition_ids"] = "treated;treated"
    with pytest.raises(ValueError, match=r"[Cc]ondition|PMID"):
        expand_condition_memberships(observations, samples_df=samples)


def test_empty_membership_projection_keeps_types_without_loading_curations(
    membership_frames, monkeypatch
):
    observations, _ = membership_frames
    observations["peptide"] = observations.peptide.astype("category")
    observations[["supported_condition_ids", "excluded_condition_ids"]] = ""

    def unexpected_lookup():
        pytest.fail("Empty memberships must not load curation metadata")

    monkeypatch.setattr("hitlist.export.generate_ms_samples_table", unexpected_lookup)
    for frame in (observations, observations.iloc[:0]):
        edges = expand_condition_memberships(frame)
        assert edges.empty
        assert edges.source_observation_position.dtype == "int64"
        assert edges.peptide.dtype == observations.peptide.dtype
