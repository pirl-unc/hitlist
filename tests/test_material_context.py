"""Material-graph qualification does not promote labels into physical identities."""

import pytest

from hitlist.material_context import links_with_minimum_precision, validate_material_graph


def _context(**overrides):
    return {
        "context_id": "source:run:1",
        "context_kind": "source_acquisition",
        "sample_label_raw": "run_1",
        "condition_id": "source:arm:control",
        "condition_text": "control",
        "condition_status": "source_run_label",
        "specimen_id": None,
        "aliquot_id": None,
        "material_link_precision": "source_acquisition",
        "context_status": "source_acquisition_unresolved_specimen_aliquot",
        "source_locator": "summary.tsv:2",
        "notes": "fixture",
        **overrides,
    }


def test_study_arm_link_cannot_satisfy_physical_pair_request():
    links = [
        {
            "claim_id": "run:in_arm",
            "source_context_id": "source:run:1",
            "target_context_id": "source:arm:control",
            "assertion_predicate": "source_acquisition_in_study_arm",
            "assertion_status": "source_reported",
            "material_link_precision": "study_arm",
            "source_locator": "summary.tsv:2",
            "notes": "fixture",
        }
    ]
    assert links_with_minimum_precision(links, "study_arm") == tuple(links)
    assert not links_with_minimum_precision(links, "exact_specimen")
    assert not links_with_minimum_precision(links, "exact_aliquot")


def test_physical_identity_requires_exact_precision():
    with pytest.raises(ValueError, match="specimen ID requires exact"):
        validate_material_graph([_context(specimen_id="specimen-1")], [])
