from hitlist.digestion_processing_package import QUERY_ALIASES, _rows
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family


def _manifest():
    return {
        "appended_sources": {
            "soh2024_digestion": {
                "registry": {"source_id": "soh2024_digestion"},
                "reconciliation": {
                    "relation_rows": {
                        "ms1": 5_792_969,
                        "identifications": 649_789,
                        "raw_quantities": 545_664,
                        "final_vectors": 53_413,
                        "final_values": 280_664,
                        "conflicts": 2_284,
                    }
                },
            }
        }
    }


def test_processing_family_ledger_preserves_unavailable_apc_scope():
    families, availability, _ = _rows(_manifest())
    by_family = {row["family_id"]: row for row in families}
    assert by_family["processing_ms1_identification"]["n_measurements"] == 5_792_969
    assert by_family["processing_timecourse_vectors"]["n_measurements"] == 280_664
    assert by_family["cellular_apc_processing"]["availability_state"] == (
        "not_materialized_in_pinned_sources"
    )
    assert {row["family_id"] for row in availability} == set(by_family)


def test_processing_package_family_and_aliases_are_versioned():
    assert SCHEMA_VERSION == 29
    assert (
        package_family(
            {
                "schema_version": 19,
                "package_family": "digestion_processing",
                "revision": "soh2024_digestion_processing_v1",
            }
        )
        == "digestion_processing"
    )
    assert QUERY_ALIASES["processing_timecourse"] == "digestion_final_values"
    fields = {field["name"] for field in describe_evidence("digestion_processing_family")["fields"]}
    assert {"family_id", "availability_state", "n_measurements", "notes"} <= fields
