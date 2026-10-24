from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family
from hitlist.tcell_processing_package import QUERY_ALIASES, _rows


def _predecessors():
    return (
        {
            "reconciliation": {
                "n_functional_examples": 47_421,
                "n_multimer_examples": 8_423,
            },
            "source_registry": {"IEDB": {"n_assays": 578_529}},
        },
        {
            "sources": {
                "pxd009092": {
                    "reconciliation": {
                        "n_quantity_results_rows": 192_114,
                        "n_quantity_measurements_rows": 362_882,
                    },
                    "registry": {"source_id": "pxd009092"},
                }
            }
        },
    )


def test_family_ledger_keeps_unavailable_families_explicit():
    families, availability = _rows(*_predecessors())
    by_family = {row["family_id"]: row for row in families}
    assert by_family["tcell_functional_response"]["n_training_examples"] == 47_421
    assert by_family["pmhc_multimer_binding"]["n_training_examples"] == 8_423
    assert by_family["receptor_evidence"]["availability_state"] == (
        "not_materialized_in_pinned_sources"
    )
    assert by_family["direct_processing_measurement"]["n_measurements"] == 0
    assert by_family["mapp_controlled_product"]["n_measurements"] == 362_882
    assert {row["family_id"] for row in availability} == set(by_family)


def test_package_family_and_query_alias_contract_are_versioned():
    assert SCHEMA_VERSION == 26
    assert (
        package_family(
            {
                "schema_version": 18,
                "package_family": "tcell_processing",
                "revision": "tcell_processing_evidence_v1",
            }
        )
        == "tcell_processing"
    )
    assert QUERY_ALIASES == {
        "tcell_functional_response": "tcell_functional_peptide",
        "pmhc_multimer_binding": "tcell_multimer_peptide",
        "processing_measurement": "external_proteasome_fragment_quantity",
        "mapp_controlled_product": "mapp_quantity_measurement",
    }
    family_fields = {
        field["name"] for field in describe_evidence("tcell_processing_family")["fields"]
    }
    assert {"family_id", "availability_state", "n_training_examples"} <= family_fields
