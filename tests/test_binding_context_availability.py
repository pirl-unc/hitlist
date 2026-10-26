import pytest

from hitlist.binding_context_availability import _endpoint_family, _status_from_claim
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family


@pytest.mark.parametrize(
    "endpoint,family",
    [
        ("half maximal inhibitory concentration (IC50)", "literal_ic50"),
        ("dissociation constant KD (~IC50)", "kd_ic50"),
        ("equilibrium dissociation constant", "other_binding"),
    ],
)
def test_binding_endpoint_families_stay_separate(endpoint, family):
    assert _endpoint_family(endpoint) == family


def test_study_claim_status_never_becomes_row_value():
    claim = {"claim_status": "reported_study_scope"}
    assert _status_from_claim(claim, reviewed=True) == "study_scope_only"
    assert (
        _status_from_claim(
            {"claim_status": "reported_alternatives_per_assay_unresolved"}, reviewed=True
        )
        == "unresolved"
    )
    assert _status_from_claim(claim, reviewed=False) == "unreviewed"


def test_binding_context_availability_is_a_registered_successor_relation():
    envelope = {
        "schema_version": 14,
        "parent_kind": "binding_review",
        "revision": "binding_assay_context_availability_v1",
    }
    assert package_family(envelope) == "binding_context_availability"
    assert SCHEMA_VERSION == 39
    fields = {
        field["name"] for field in describe_evidence("binding_context_availability")["fields"]
    }
    assert {"observation_id", "temperature_status", "kd_conversion_status"} <= fields
    assert "binding_context_availability" in materialized_bindings(14)
