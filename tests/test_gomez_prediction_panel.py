import pytest

from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import adjudication_manifest_path, package_family
from hitlist.gomez_prediction_panel import normalize_allele


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("A*68:02", "HLA-A*68:02"),
        ("HLA-C0702", "HLA-C*07:02"),
        ("HLA-B*07:02", "HLA-B*07:02"),
        ("", ""),
    ],
)
def test_prediction_allele_normalization(raw, normalized):
    assert normalize_allele(raw) == normalized


def test_prediction_panel_family_and_catalog_contract():
    envelope = {
        "schema_version": 13,
        "parent_kind": "gomez_arm_projection",
        "revision": "gomez_zepeda_2024_prediction_panels_v1",
        "adjudication_manifest_path": "manifests/adjudication.json",
    }
    assert package_family(envelope) == "gomez_prediction_panel"
    assert adjudication_manifest_path(envelope) == "manifests/adjudication.json"
    assert SCHEMA_VERSION == 23
    assert "prediction_id" in {
        field["name"] for field in describe_evidence("gomez_prediction_row")["fields"]
    }
    bindings = materialized_bindings(13)
    assert "gomez_prediction_row" in bindings
    assert "gomez_restriction_projection" in bindings
