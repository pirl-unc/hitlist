"""Successors preserve newer modalities and isolate their predecessor bytes."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import external_successor as successor
from hitlist import goldberg_mapp
from hitlist.evidence_bindings import materialized_bindings
from hitlist.mapp_controlled_evidence import write_controlled_tables
from hitlist.mapp_evidence import write_mapp_tables
from hitlist.mapp_pbmc_evidence import write_pbmc_tables
from hitlist.provenance import file_digest
from tests.test_goldberg_mapp import original as goldberg_original_fixture
from tests.test_mapp_controlled_evidence import _fixture as controlled_fixture
from tests.test_mapp_evidence import _source as pxd009092_fixture
from tests.test_mapp_pbmc_evidence import _fixture as pxd009438_fixture


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
    assert manifest["schema_version"] == 27
    assert manifest["catalog_schema_version"] == 17
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
    assert set(seven) == set(eight)
    assert "mapp_peptide" not in eight
    assert "assay_adjudication" not in eight
    mapp_only = materialized_bindings(
        9, include_adjudications=True, package_features=["canonical_mapp"]
    )
    assert "mapp_peptide" in mapp_only and "mapp_quantity_result" in mapp_only
    assert "calibration_s4_measurement" not in mapp_only
    combined = materialized_bindings(
        9,
        appended_source_ids=["pxd009092", "pxd004886"],
        include_adjudications=True,
        package_features=["canonical_mapp"],
    )
    assert {"mapp_peptide", "assay_adjudication", "calibration_s4_measurement"} <= combined.keys()


def test_all_mapp_adapters_publish_and_verify_one_canonical_projection(tmp_path, monkeypatch):
    parent, _source, _calls = _fixture(tmp_path, monkeypatch)

    def adapter(source_id, prepare, write_tables):
        root = tmp_path / f"{source_id}-assets"
        root.mkdir()
        marker = root / "marker.txt"
        marker.write_text(source_id)
        receipt = tmp_path / f"{source_id}-receipt.json"
        digest = file_digest(marker)
        receipt.write_text(
            json.dumps(
                {
                    "assets": {
                        "marker.txt": {
                            "sha256": digest["sha256"],
                            "size": digest["size_bytes"],
                        }
                    }
                }
            )
        )

        class Adapter:
            schema_version = 1
            asset_receipt_path = receipt
            support_files = {}

            def write(self, directory):
                directory = Path(directory)
                source_receipt = (directory / "contracts/source_assets.json").read_bytes()
                shutil.rmtree(directory / "assets")
                shutil.rmtree(directory / "contracts")
                prepare(directory)
                (directory / "contracts/source_assets.json").write_bytes(source_receipt)
                (directory / "assets/marker.txt").write_text(source_id)
                return write_tables(directory)

            def verify(self, directory, reconciliation):
                typed = Path(directory) / "typed"
                assert typed.is_dir() and any(typed.glob("*.parquet"))

            def registry(self, reconciliation):
                return {"source_id": source_id, "fixture_reconciliation": reconciliation}

        return root, Adapter()

    pxd009092_root, pxd009092 = adapter("pxd009092", pxd009092_fixture, write_mapp_tables)
    controlled_root, controlled = adapter(
        "wolflevy2018_controlled", controlled_fixture, write_controlled_tables
    )
    pxd009438_root, pxd009438 = adapter("pxd009438", pxd009438_fixture, write_pbmc_tables)

    def write_goldberg_tables(directory):
        original = goldberg_original_fixture.__wrapped__(directory, monkeypatch)
        return goldberg_mapp._compile(
            original, Path(directory) / "typed", json.loads(goldberg_mapp.CONTRACT.read_text())
        )

    def prepare_goldberg(directory):
        for name in ("assets", "contracts"):
            (Path(directory) / name).mkdir()

    goldberg_root, goldberg = adapter("goldberg2025", prepare_goldberg, write_goldberg_tables)
    monkeypatch.setattr(
        successor,
        "_adapters",
        lambda: {
            "pxd009092": pxd009092,
            "wolflevy2018_controlled": controlled,
            "pxd009438": pxd009438,
            "goldberg2025": goldberg,
        },
    )

    output = tmp_path / "output"
    successor.write_external_successor(
        parent,
        output,
        source_roots={
            "pxd009092": pxd009092_root,
            "wolflevy2018_controlled": controlled_root,
            "pxd009438": pxd009438_root,
            "goldberg2025": goldberg_root,
        },
    )
    manifest = successor.verify_external_successor(output)
    assert manifest["mapp_canonical"]["source_datasets"] == [
        "PXD007263",
        "PXD009092",
        "PXD009438",
        "PXD009508",
        "PXD052314",
    ]
    assert manifest["mapp_canonical"]["relation_rows"]["mapp_acquisition"] == 64

    path = output / "mapp/typed/channels.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["condition"] = "drift"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.ParquetFile(path).schema_arrow), path)
    manifest_path = output / "manifest.json"
    changed = json.loads(manifest_path.read_text())
    changed["files"][str(path.relative_to(output))] = file_digest(path)
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Canonical MAPP relation drift"):
        successor.verify_external_successor(output)


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
    manifest.update(schema_version=9, catalog_schema_version=catalog)
    path.write_text(json.dumps(manifest))
    assert successor.verify_external_successor(output) == manifest


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_kind": "external"},
        {"parent_manifest_path": "manifests/adjudication.json"},
        {"parent_manifest_path": "../escape"},
        {"review_revision": "ambiguous"},
        {"catalog_schema_version": 18},
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
