import json

import pytest

from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family
from hitlist.tmab_peptidomics import _parse_ratio, _parse_sequence, _treatment


@pytest.mark.parametrize(
    "raw,value,inequality,censoring,status",
    [
        (0.5, 0.5, None, "exact", "reported_exact"),
        ("<0.20", 0.2, "lt", "source_declared_bound", "reported_bound"),
        (">5", 5.0, "gt", "source_declared_bound", "reported_bound"),
        ("po", None, None, "overlapping_peak", "reported_overlap"),
        (None, None, None, "missing_signal", "missing_source_value"),
        ("nd", None, None, "missing_signal", "reported_not_detected"),
    ],
)
def test_native_ratio_states_preserve_censoring_and_missingness(
    raw, value, inequality, censoring, status
):
    parsed = _parse_ratio(raw)
    assert parsed[:4] == (value, inequality, censoring, status)


def test_sequence_and_modification_are_separate():
    assert _parse_sequence("Ac-ASKRALVILA") == ("ASKRALVILA", "N_terminal_acetylation")
    assert _parse_sequence("(ASKRALVILA)") == ("ASKRALVILA", None)
    assert _parse_sequence(None) == (None, None)


def test_treatment_parser_keeps_dose_and_duration_scoped_to_raw_label():
    assert _treatment("bortezomib 30min, 500nM") == (
        "bortezomib 30min, 500nM",
        "bortezomib",
        500.0,
        "nM",
        0.5,
    )
    assert _treatment("control") == ("control", "control", None, None, None)


def test_tmab_package_is_standalone_and_queryable():
    envelope = {
        "schema_version": 17,
        "package_family": "tmab_peptidomics",
        "revision": "tmab_native_peptidomics_v3",
    }
    assert package_family(envelope) == "tmab_peptidomics"
    assert SCHEMA_VERSION == 26
    fields = {field["name"] for field in describe_evidence("tmab_measurement")["fields"]}
    assert {
        "peptide_sequence_raw",
        "quantitative_raw",
        "measurement_inequality",
        "censoring_status",
    } <= fields
    context_fields = {
        field["name"] for field in describe_evidence("tmab_protocol_context")["fields"]
    }
    assert {"instrument", "cell_line_name", "treatment_name"} <= context_fields
    bindings = materialized_bindings(17)
    assert {
        "tmab_measurement",
        "tmab_source_row",
        "tmab_protocol_context",
        "tmab_measurement_lineage",
    } <= set(bindings)


def test_asset_receipt_is_hash_pinned():
    from hitlist.tmab_peptidomics import ASSETS

    payload = json.loads(ASSETS.read_text())
    assert len(payload["assets"]) == 4
    assert all(len(asset["sha256"]) == 64 for asset in payload["assets"])
