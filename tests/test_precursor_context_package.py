import io
import json
import tarfile

from hitlist import EvidenceStore
from hitlist import precursor_context_package as package
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family
from hitlist.precursor_recipe import CONTRACT, RECIPE_ID
from hitlist.provenance import file_digest


def _receipt(source_root, receipt_path):
    source = source_root / "asset.txt"
    source.write_text("pinned source\n")
    digest = file_digest(source)
    archive = source_root / "search.tar.gz"
    member = b"Sequence\tModifications\tCharge\tIntensity\nPEPTIDE\tOxidation (M)\t2\tNaN\n"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("txt/peptides.txt")
        info.size = len(member)
        handle.addfile(info, io.BytesIO(member))
    archive_digest = file_digest(archive)
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
            },
            "search.tar.gz": {
                "sha256": archive_digest["sha256"],
                "size_bytes": archive_digest["size_bytes"],
                "asset_kind": "search_archive",
                "source_locator": "search.tar.gz",
            },
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
    assert SCHEMA_VERSION == 33
    assert (
        package_family(
            {
                "schema_version": 22,
                "package_family": "precursor_context",
                "revision": "pxd009360_precursor_context_v3",
            }
        )
        == "precursor_context"
    )
    fields = {field["name"] for field in describe_evidence("precursor_pairing_assertion")["fields"]}
    assert {"claim_id", "pairing_status", "pairing_evidence"} <= fields
    assert CONTRACT["recipe_id"] == RECIPE_ID
    assert CONTRACT["quantity_policy"]["imputation"] == "none"
    assert CONTRACT["quantity_policy"]["normalization"] == "none"
    assert CONTRACT["pairing_policy"]["evidence_required"] == (
        "source_backed_crosswalk_or_explicit_assertion"
    )


def test_precursor_writer_verifier_and_evidence_store(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    receipt_path = tmp_path / "receipt.json"
    _receipt(source_root, receipt_path)
    monkeypatch.setattr(package, "ASSET_RECEIPT", receipt_path)
    destination = tmp_path / "package"
    package.write_precursor_context_package(source_root, destination)
    manifest = package.verify_precursor_context_package(destination)
    assert manifest["schema_version"] == 22
    store = EvidenceStore.open(destination)
    assert store.count("precursor_run") == 1
    assert store.count("precursor_pairing") == 1
    assert store.count("precursor_peptide") == 1
    assert store.count("precursor_recipe") == 1
    assert store.count("precursor_qualification") == 1
    assert store.count("precursor_pairing_qualification") == 1
    family = store.query(
        "precursor_context_family",
        select=("availability_state", "n_source_rows", "n_measurements"),
        where={"family_id": "precursor_quantity_tables"},
        limit=None,
    ).table.to_pylist()[0]
    assert family == {
        "availability_state": "typed_rows_materialized",
        "n_source_rows": 1,
        "n_measurements": 0,
    }
    peptide = store.query(
        "precursor_peptide",
        select=("raw_row", "values_json", "charge_raw", "intensity_raw"),
        limit=None,
    ).table.to_pylist()[0]
    assert peptide["raw_row"].endswith("PEPTIDE\tOxidation (M)\t2\tNaN")
    assert json.loads(peptide["values_json"]) == ["PEPTIDE", "Oxidation (M)", "2", "NaN"]
    assert peptide["charge_raw"] == "2"
    assert peptide["intensity_raw"] == "NaN"
    recipe = store.query(
        "precursor_recipe", select=("recipe_id", "training_status"), limit=None
    ).table.to_pylist()[0]
    assert recipe == {
        "recipe_id": "pxd009360_precursor_quantity_qualification_v1",
        "training_status": "unavailable",
    }
    qualification = store.query("precursor_qualification", limit=None).table.to_pylist()[0]
    assert qualification["n_source_rows"] == 1
    assert qualification["n_rows_with_sequence"] == 1
    assert qualification["n_rows_with_contributors"] == 0
    assert qualification["n_rows_with_intensity"] == 0
    assert qualification["n_rows_missing_intensity"] == 1
    assert qualification["qualification_status"] == "qualified_for_nontraining_only"
    pairing = store.query("precursor_pairing_qualification", limit=None).table.to_pylist()[0]
    assert pairing["source_backed"] is False
    assert pairing["pairing_status"] == "unresolved"
    assert pairing["training_status"] == "unavailable"
    assert pairing["split_policy"] == "no_biological_split_until_source_backed_crosswalk"
    assert store._source_registry()[package.SOURCE_ID]["n_acquisitions"] == 1
    assert store._source_registry()[package.SOURCE_ID]["n_materialized_quantity_rows"] == 1
