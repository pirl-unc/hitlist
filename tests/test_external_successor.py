"""Successors preserve newer modalities and isolate their predecessor bytes."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import external_successor as successor
from hitlist.evidence_bindings import materialized_bindings
from hitlist.provenance import file_digest


def _fixture(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "newer-modality.txt").write_text("a later T-cell or numeric measurement")
    manifest = {
        "schema_version": 7,
        "reference": {"identity": "pinned"},
        "files": {"newer-modality.txt": file_digest(parent / "newer-modality.txt")},
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
    monkeypatch.setattr(successor, "_adapters", lambda: {"fixture": Adapter()})
    return parent, source, calls


def test_successor_preserves_and_isolates_predecessor(tmp_path, monkeypatch):
    parent, source, calls = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    successor.write_external_successor(parent, output, source_roots={"fixture": source})
    manifest = successor.verify_external_successor(output)
    assert manifest["schema_version"] == 8
    assert (output / "newer-modality.txt").read_bytes() == (
        parent / "newer-modality.txt"
    ).read_bytes()
    assert (output / "newer-modality.txt").stat().st_ino != (
        parent / "newer-modality.txt"
    ).stat().st_ino
    assert (output, "manifests/reference.json") in calls
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
