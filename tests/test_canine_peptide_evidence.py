"""Offline consumer acceptance of released-format native canine evidence."""

import copy
import hashlib
import json
import math
import socket
from pathlib import Path

import pytest

pytest.importorskip("orfdata", minversion="0.40.0")
from orfdata import DatasetRegistry

from hitlist import canine_bulk
from hitlist.canine_orfdata import _registry_identity
from hitlist.canine_peptide_evidence import open_canine_peptide_evidence
from hitlist.provenance import file_digest

FIXTURE = Path(__file__).parent / "data/canine_peptide_evidence"


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    digest = json.loads((FIXTURE / "expected.json").read_text())["artifact"]
    compressed = FIXTURE / "dataset.sqlite.gz"
    download = {
        "distribution": {
            "access_terms": "Explicit regression slice; see fixture README",
            "redistributable": True,
            "download_size": compressed.stat().st_size,
            "download_sha256": file_digest(compressed)["sha256"],
        },
        "assets": {
            "dataset.sqlite": {
                "url": "https://example.org/dataset.sqlite.gz",
                "sha256": digest["sha256"],
                "size": digest["size_bytes"],
                "decompress": True,
            }
        },
    }
    name, release = manifest["dataset_id"], manifest["version"]
    definitions = {
        name: {
            "default_version": release,
            "versions": {release: {"manifest": manifest, **download}},
        }
    }
    registry = DatasetRegistry(definitions, cache_root=tmp_path / "cache")
    registry.download(name, release, source_paths={"dataset.sqlite": compressed})
    monkeypatch.setattr("orfdata.DatasetRegistry", lambda **kwargs: registry)
    catalog = copy.deepcopy(canine_bulk.canine_bulk_catalog())
    source = catalog["sources"]["pfizer_2026_03"]
    source["orfdata_peptides"] = {
        "dataset_id": name,
        "version": release,
        "registry_sha256": _registry_identity(manifest, download),
    }
    baseline = manifest["processing"]["protein_summary"]
    source["orfdata"].update(dataset_id=baseline["dataset_id"], version=baseline["version"])
    assets = {asset["asset_id"]: asset for asset in manifest["sources"]}
    for role, name in (("table", "protein.tsv"), ("reference", "search.fasta")):
        source["assets"][role].update(
            sha256=assets[name]["sha256"], size_bytes=assets[name]["size"]
        )
    monkeypatch.setattr(canine_bulk, "canine_bulk_catalog", lambda: catalog)
    return registry, source


def _curate(tmp_path, **kwargs):
    output = tmp_path / "bundle"
    canine_bulk.curate_canine_bulk("pfizer_2026_03", output, provider="orfdata_peptides", **kwargs)
    return output


def test_complete_snapshot_preserves_every_measurement_and_native_fact(
    upstream, tmp_path, monkeypatch
):
    registry, source = upstream

    def no_network(*args, **kwargs):
        pytest.fail("An offline consumer attempted acquisition")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(registry, "download", no_network)
    monkeypatch.setattr(registry, "download_reference", no_network)
    output = _curate(tmp_path)
    reader = open_canine_peptide_evidence(output)
    expected = json.loads((FIXTURE / "expected.json").read_text())
    assert file_digest(output / "dataset.sqlite") == expected["artifact"]
    values = [value.to_dict() for value in reader.dataset.measurements()]
    assert (
        hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == expected["measurements_sha256"]
    )
    assert reader.dataset.counts == expected["counts"]
    assert reader.manifest["presentation_coverage"] == "missing"
    assert reader.manifest["source"]["source_id"] == "pfizer_2026_03"
    assert reader.manifest["source"]["instrument_reports"] == source["instrument_reports"]
    assert len(list(reader.native.records("DecoyPsms"))) == 1
    decoy = next(reader.native.records("DecoyPsms"))
    assert decoy["Confidence"] == 3
    assert decoy["Modifications"]
    acquisition = list(reader.native.records("AcquisitionFixture"))
    assert any(isinstance(row["value"], float) and math.isinf(row["value"]) for row in acquisition)
    assert any(row["opaque"] == b"\xff\0" for row in acquisition)
    monkeypatch.setattr("orfdata.DatasetRegistry", no_network)
    monkeypatch.setattr(canine_bulk, "canine_bulk_catalog", no_network)
    assert canine_bulk.verify_bulk_evidence(output) == reader.manifest


def test_bounded_queries_keep_controls_missingness_and_shared_candidates(upstream, tmp_path):
    output = _curate(tmp_path)
    reader = open_canine_peptide_evidence(output, verify=False)
    frame = canine_bulk.load_bulk_evidence(output)
    assert len(frame) == reader.dataset.counts["measurement"]
    assert frame.status.eq("missing").any()
    assert frame.value.isna().any()
    assert frame.status.eq("measured").any()
    assert frame.peptide_modifications_raw.notna().any()
    assert set(frame.evidence_kind) == {"bulk"}
    assert not frame.is_ms_observation.any()
    assert frame.mhc_restriction.eq("").all()
    assert frame.donor_id.isna().all()
    assert {"bridge", "tissue_pool", "empty"} <= set(frame.sample_role)
    tissue = frame.source_tissue.dropna().iloc[0]
    selected = canine_bulk.load_bulk_evidence(output, tissue=tissue.upper())
    assert selected.source_tissue.eq(tissue).all()
    sequence = frame.peptide.iloc[0]
    assert (
        canine_bulk.load_bulk_evidence(output, peptide=sequence.lower()).peptide.eq(sequence).all()
    )
    mappings = canine_bulk.load_bulk_evidence(output, table="mappings")
    assert {row["basis"] for row in mappings.evidence} == {"reported", "sequence_compatibility"}
    shared = mappings[mappings.targets.map(len).gt(1)]
    assert not shared.empty
    assert shared.status.eq("unresolved").any()
    raw = canine_bulk.load_bulk_evidence(output, table="raw_DecoyPsms")
    assert raw.Confidence.tolist() == [3]
    assert "is_ms_observation" not in raw  # Exact native cells, with no added interpretation.
    groups = canine_bulk.load_bulk_evidence(output, table="groups")
    assert len(groups) == reader.dataset.counts["target_relation"]
    assert {row["protein_summary"]["dataset_id"] for row in groups.evidence} == {
        reader.dataset.manifest.processing["protein_summary"]["dataset_id"]
    }
    with pytest.raises(ValueError, match="max_rows"):
        canine_bulk.load_bulk_evidence(output, max_rows=1)
    with pytest.raises(ValueError, match="tissue filter"):
        canine_bulk.load_bulk_evidence(output, table="mappings", tissue=tissue)
    with pytest.raises(ValueError, match="peptide filter"):
        canine_bulk.load_bulk_evidence(output, table="samples", peptide=sequence)
    assert canine_bulk.load_bulk_evidence(output, peptide="NOTINTHEDATA").empty


@pytest.mark.parametrize("failure", ["registry", "reference", "summary"])
def test_mismatched_source_release_rejected_before_acquisition(
    upstream, tmp_path, monkeypatch, failure
):
    registry, source = upstream
    if failure == "registry":
        source["orfdata_peptides"]["registry_sha256"] = "0" * 64
    else:
        source["assets"]["reference" if failure == "reference" else "table"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        registry, "download", lambda *a, **k: pytest.fail("download before validation")
    )
    with pytest.raises(ValueError):
        _curate(tmp_path, allow_download=True)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "limit", ["max_input_bytes", "max_upstream_bytes", "max_output_bytes", "max_records"]
)
def test_bounds_reject_before_acquisition(upstream, tmp_path, monkeypatch, limit):
    registry, _ = upstream
    monkeypatch.setattr(
        registry, "download", lambda *a, **k: pytest.fail("download before limit check")
    )
    with pytest.raises(ValueError, match=limit):
        _curate(tmp_path, allow_download=True, limits={limit: 1})
    assert not (tmp_path / "bundle").exists()


def test_explicit_acquisition_only_requests_pinned_shared_artifact(upstream, tmp_path, monkeypatch):
    registry, source = upstream
    calls = []
    monkeypatch.setattr(registry, "download", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        registry, "download_reference", lambda *a, **k: pytest.fail("Separate FASTA acquisition")
    )
    _curate(tmp_path, allow_download=True)
    pin = source["orfdata_peptides"]
    assert calls == [((pin["dataset_id"], pin["version"]), {})]


def test_frozen_snapshot_survives_mutated_upstream_and_detects_corruption(upstream, tmp_path):
    registry, source = upstream
    output = _curate(tmp_path)
    original = (output / "dataset.sqlite").read_bytes()
    pin = source["orfdata_peptides"]
    cache = registry.open(pin["dataset_id"], pin["version"]).path
    assert cache.stat().st_ino != (output / "dataset.sqlite").stat().st_ino
    with cache.open("ab") as stream:
        stream.write(b"later cache corruption")
    assert (output / "dataset.sqlite").read_bytes() == original
    canine_bulk.verify_bulk_evidence(output)
    with pytest.raises((ValueError, OSError)):
        canine_bulk.curate_canine_bulk(
            "pfizer_2026_03", tmp_path / "corrupt-cache", provider="orfdata_peptides"
        )
    assert not (tmp_path / "corrupt-cache").exists()
    with (output / "dataset.sqlite").open("ab") as stream:
        stream.write(b"bundle corruption")
    with pytest.raises(ValueError, match="artifact mismatch"):
        canine_bulk.verify_bulk_evidence(output)


def test_partial_snapshot_failure_does_not_publish(upstream, tmp_path, monkeypatch):
    def interrupted(source, target):
        target.write_bytes(b"partial")
        raise OSError("interrupted snapshot")

    monkeypatch.setattr("hitlist.canine_peptide_evidence.snapshot_file", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        _curate(tmp_path)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_output_created_during_snapshot_is_preserved(upstream, tmp_path, monkeypatch):
    from hitlist.evidence_snapshot import snapshot_file

    def racing_output(source, target):
        snapshot_file(source, target)
        output = tmp_path / "bundle"
        output.mkdir()
        (output / "existing.txt").write_text("keep this")

    monkeypatch.setattr("hitlist.canine_peptide_evidence.snapshot_file", racing_output)
    with pytest.raises(FileExistsError):
        _curate(tmp_path)
    assert (tmp_path / "bundle" / "existing.txt").read_text() == "keep this"
    assert not list(tmp_path.glob(".bundle-*"))


@pytest.mark.parametrize("field", ["presentation_coverage", "schema_version", "n_records_by_table"])
def test_changed_frozen_semantics_or_counts_rejected(upstream, tmp_path, field):
    output = _curate(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = {"peptides": 0} if field == "n_records_by_table" else "incorrect"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        canine_bulk.verify_bulk_evidence(output)
