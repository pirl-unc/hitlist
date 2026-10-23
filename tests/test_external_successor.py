"""Successors preserve newer modalities and isolate their predecessor bytes."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import external_successor as successor
from hitlist.evidence_bindings import materialized_bindings
from hitlist.provenance import file_digest


def _fixture(tmp_path, monkeypatch, *, adjudication=False):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "newer-modality.txt").write_text("a later T-cell or numeric measurement")
    manifest = {
        "schema_version": 7,
        "reference": {"identity": "pinned"},
        "files": {"newer-modality.txt": file_digest(parent / "newer-modality.txt")},
    }
    (parent / "manifest.json").write_text(json.dumps(manifest))
    if adjudication:
        (parent / "manifests").mkdir()
        (parent / "manifests/reference.json").write_text(json.dumps(manifest))
        (parent / "reviews.txt").write_text("nine reviewed assertions, unchanged")
        manifest = {
            "schema_version": 8,
            "catalog_schema_version": 10,
            "review_revision": "fixture",
            "files": {
                **manifest["files"],
                "manifests/reference.json": file_digest(parent / "manifests/reference.json"),
                "reviews.txt": file_digest(parent / "reviews.txt"),
            },
        }
        (parent / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    def verify_parent(directory, *, manifest_path="manifest.json"):
        # This fixture isolates successor orchestration; the production recursive
        # verifier and full source/package invariants are exercised by real replay.
        calls.append((Path(directory), manifest_path))
        value = json.loads((Path(directory) / manifest_path).read_text())
        for relative, digest in value["files"].items():
            if file_digest(Path(directory) / relative) != digest:
                raise ValueError("fixture predecessor mismatch")
        return value

    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("new source")
    receipt = tmp_path / "receipt.json"
    digest = file_digest(source / "data.txt")
    receipt.write_text(
        json.dumps(
            {"assets": {"data.txt": {"sha256": digest["sha256"], "size": digest["size_bytes"]}}}
        )
    )

    class Adapter:
        schema_version = 1
        source_id = "fixture"
        asset_receipt_path = receipt
        support_files = {}

        def write(self, directory):
            (directory / "typed").mkdir()
            pq.write_table(pa.table({"id": ["source:1"]}), directory / "typed/rows.parquet")
            return {"n_rows": 1}

        def verify(self, directory, reconciliation):
            assert (
                pq.read_metadata(directory / "typed/rows.parquet").num_rows
                == reconciliation["n_rows"]
            )

        def registry(self, reconciliation):
            return {"n_source_rows": reconciliation["n_rows"]}

    monkeypatch.setattr(successor, "verify_reference_evidence_package", verify_parent)
    monkeypatch.setattr(successor, "verify_assay_adjudication_package", verify_parent)
    monkeypatch.setattr(successor, "_adapters", lambda: {"fixture": Adapter()})
    return parent, source, calls


@pytest.mark.parametrize("adjudication", [False, True])
def test_successor_preserves_and_isolates_predecessor(tmp_path, monkeypatch, adjudication):
    parent, source, calls = _fixture(tmp_path, monkeypatch, adjudication=adjudication)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    manifest = successor.verify_external_successor(output)
    assert manifest["schema_version"] == 9
    assert manifest["catalog_schema_version"] == 14
    assert (output / "newer-modality.txt").read_bytes() == (
        parent / "newer-modality.txt"
    ).read_bytes()
    assert (output / "newer-modality.txt").stat().st_ino != (
        parent / "newer-modality.txt"
    ).stat().st_ino
    kind = "adjudication" if adjudication else "reference"
    assert (output, f"manifests/{kind}.json") in calls
    assert manifest["parent_kind"] == kind
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    for relative in parent_manifest["files"]:
        assert (output / relative).read_bytes() == (parent / relative).read_bytes()
        assert (output / relative).stat().st_ino != (parent / relative).stat().st_ino
    assert manifest["parent_manifest_sha256"] == file_digest(parent / "manifest.json")["sha256"]
    (parent / "newer-modality.txt").write_text("edited after snapshot")
    successor.verify_external_successor(output)
    with pytest.raises(FileExistsError):
        successor.write_external_successor(parent, output, source_roots={"fixture": source})
    (output / "external/fixture/assets/data.txt").write_text("tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        successor.verify_external_successor(output)


def test_successor_rejects_replacing_sources_and_unknown_ids(tmp_path, monkeypatch):
    parent, source, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unregistered"):
        successor.write_external_successor(parent, output, source_roots={"../escape": source})
    (parent / "external/fixture").mkdir(parents=True)
    with pytest.raises(ValueError, match="replaces predecessor"):
        successor.write_external_successor(parent, output, source_roots={"fixture": source})
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["reference", "registry", "unowned", "traversal"])
def test_successor_rejects_changed_semantics_even_with_updated_hashes(
    tmp_path, monkeypatch, mutation
):
    parent, source, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "reference":
        manifest["reference"] = {"identity": "changed"}
    elif mutation == "registry":
        manifest["appended_sources"]["fixture"]["registry"] = {"n_source_rows": 2}
    elif mutation == "unowned":
        (output / "unowned.txt").write_text("unregistered input")
        manifest["files"]["unowned.txt"] = file_digest(output / "unowned.txt")
    else:
        manifest["files"]["../escape"] = manifest["files"].pop("newer-modality.txt")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        successor.verify_external_successor(output)


def test_bindings_do_not_claim_absent_mapp_or_mouse_sources():
    six = materialized_bindings(6)
    seven = materialized_bindings(7)
    eight = materialized_bindings(8, appended_source_ids=["pxd009092"])
    assert "external_reference_occurrence" not in six
    assert "external_reference_occurrence" in seven
    assert "mapp_peptide" not in seven
    assert "mapp_peptide" not in materialized_bindings(8)
    assert set(seven) < set(eight)
    assert "mapp_peptide" in eight and "external_peptide_parent_occurrence" in eight
    assert "assay_adjudication" not in eight
    mapp_only = materialized_bindings(
        9, appended_source_ids=["pxd009092"], include_adjudications=True
    )
    assert "mapp_peptide" in mapp_only
    assert "calibration_s4_measurement" not in mapp_only
    combined = materialized_bindings(
        9, appended_source_ids=["pxd009092", "pxd004886"], include_adjudications=True
    )
    assert {"mapp_peptide", "assay_adjudication", "calibration_s4_measurement"} <= combined.keys()


@pytest.mark.parametrize("catalog", [10, 11])
def test_legacy_external_eight_remains_readable(tmp_path, monkeypatch, catalog):
    parent, source, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(schema_version=8, catalog_schema_version=catalog)
    del manifest["parent_kind"], manifest["parent_manifest_path"]
    path.write_text(json.dumps(manifest))
    assert successor.verify_external_successor(output) == manifest


@pytest.mark.parametrize("catalog", [11, 12, 13])
def test_released_schema_nine_packages_remain_readable(tmp_path, monkeypatch, catalog):
    parent, source, _ = _fixture(tmp_path, monkeypatch, adjudication=True)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["catalog_schema_version"] = catalog
    path.write_text(json.dumps(manifest))
    assert successor.verify_external_successor(output) == manifest


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_kind": "external"},
        {"parent_manifest_path": "manifests/adjudication.json"},
        {"parent_manifest_path": "../escape"},
        {"review_revision": "ambiguous"},
        {"catalog_schema_version": 15},
    ],
)
def test_successor_rejects_false_family_or_predecessor(tmp_path, monkeypatch, changes):
    parent, source, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        successor.verify_external_successor(output)


def test_adjudication_bytes_are_verified_under_updated_outer_hash(tmp_path, monkeypatch):
    parent, source, _ = _fixture(tmp_path, monkeypatch, adjudication=True)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    (output / "reviews.txt").write_text("lost a review")
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["reviews.txt"] = file_digest(output / "reviews.txt")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predecessor mismatch"):
        successor.verify_external_successor(output)
