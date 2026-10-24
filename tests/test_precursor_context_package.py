import json

from hitlist import EvidenceStore
from hitlist import precursor_context_package as package
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family
from hitlist.provenance import file_digest


def _receipt(source_root, receipt_path):
    source = source_root / "asset.txt"
    source.write_text("pinned source\n")
    digest = file_digest(source)
    value = {
        "source_id": "test_precursor",
        "dataset_accession": "TEST-PXD",
        "source_reference_iri": "https://example.invalid/test",
        "assets": {
            "asset.txt": {
                "sha256": digest["sha256"],
                "size_bytes": digest["size_bytes"],
                "asset_kind": "test_asset",
                "source_locator": "asset.txt",
            }
        },
        "archive_members": [{"path": "txt/peptides.txt", "size_bytes": 12}],
        "acquisitions": [
            {
                "acquisition_id": "test:run:1",
                "sample_label_raw": "1",
                "source_file": "run.raw",
            }
        ],
        "pairings": [
            {
                "claim_id": "test:pairing",
                "assertion_predicate": "run_to_condition",
                "assertion_value": "unresolved",
                "assertion_status": "unresolved",
                "pairing_scope": "test",
                "pairing_status": "unresolved",
                "pairing_evidence": "test_only",
            }
        ],
    }
    receipt_path.write_text(json.dumps(value))
    return value


def test_precursor_family_ledger_keeps_unresolved_pairing_and_deferred_quantities():
    receipt_path = package.ASSET_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    families, availability, assets, acquisitions, tables, pairings = package._rows(receipt)
    by_family = {row["family_id"]: row for row in families}
    assert by_family["precursor_quantity_tables"]["n_training_examples"] == 0
    assert by_family["mapp_precursor_pairing"]["availability_state"] == (
        "unresolved_in_pinned_sources"
    )
    assert len(assets) == 5
    assert len(acquisitions) == 9
    assert len(tables) == 17
    assert len(pairings) == 3
    assert {row["family_id"] for row in availability} == set(by_family)


def test_precursor_package_family_and_catalog_are_versioned():
    assert SCHEMA_VERSION == 26
    assert (
        package_family(
            {
                "schema_version": 20,
                "package_family": "precursor_context",
                "revision": "pxd009360_precursor_context_v1",
            }
        )
        == "precursor_context"
    )
    fields = {field["name"] for field in describe_evidence("precursor_pairing_assertion")["fields"]}
    assert {"claim_id", "pairing_status", "pairing_evidence"} <= fields


def test_precursor_writer_verifier_and_evidence_store(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    receipt_path = tmp_path / "receipt.json"
    _receipt(source_root, receipt_path)
    monkeypatch.setattr(package, "ASSET_RECEIPT", receipt_path)
    destination = tmp_path / "package"
    package.write_precursor_context_package(source_root, destination)
    manifest = package.verify_precursor_context_package(destination)
    assert manifest["schema_version"] == 20
    store = EvidenceStore.open(destination)
    assert store.count("precursor_run") == 1
    assert store.count("precursor_pairing") == 1
    assert store._source_registry()[package.SOURCE_ID]["n_acquisitions"] == 1
