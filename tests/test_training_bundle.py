import json
import os
import sys

import pandas as pd
import pytest

from hitlist.export import generate_training_table
from hitlist.provenance import contributors_path, file_digest, load_contributors
from hitlist.training_bundle import (
    audit_training_bundles,
    verify_training_bundle,
    write_training_bundle,
)
from tests.test_provenance import _row
from tests.test_scanner import _write_tiny_iedb_csv


@pytest.fixture
def built_index(tmp_path, monkeypatch, _isolated_curation_root, request):
    from hitlist import builder, downloads, supplement
    from hitlist.parquet_io import atomic_write_parquet

    indexes = tmp_path / "indexes"
    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    monkeypatch.setattr(downloads, "_override_data_dir", indexes)
    source = tmp_path / "iedb.csv"
    extra_rows = []
    if getattr(request, "param", False):
        structural = _row("http://iedb.org/assay/3")
        structural[22], structural[23] = "x-ray crystallography", "3D structure"
        extra_rows.append(structural)
    _write_tiny_iedb_csv(
        source,
        [
            _row("http://iedb.org/assay/1"),
            _row("http://iedb.org/assay/1", "source copy"),
            _row("http://iedb.org/assay/2", "independent"),
            *extra_rows,
        ],
    )
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kwargs: pd.DataFrame())

    def empty(path):
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder, "build_bulk_proteomics", lambda **kw: empty(builder._bulk_proteomics_path())
    )
    monkeypatch.setattr(
        builder, "build_line_expression", lambda **kw: empty(builder._line_expression_path())
    )
    builder.build_observations(build_mappings=False)
    monkeypatch.setattr(
        "hitlist.export._load_training_mappings_for_peptides",
        lambda *a, **kw: pd.DataFrame({"peptide": ["SLYNTVATL"] * 2, "protein_id": ["P1", "P2"]}),
    )
    return indexes


def test_real_build_bundle_projection_and_mapping_parity(built_index, tmp_path):
    compact = generate_training_table(include_evidence="ms")
    assert len(compact) == 2
    assert set(compact.provenance_status) == {"indexed"}
    assert len(load_contributors(compact.provenance_id)) == 3
    directory = tmp_path / "bundle"
    write_training_bundle(
        directory,
        include_evidence="ms",
        columns=["peptide"],
        map_source_proteins=True,
        split_policy="specimen_disjoint",
    )
    manifest = verify_training_bundle(directory)
    projected = pd.read_parquet(directory / "training.parquet")
    ordinary = generate_training_table(
        include_evidence="ms", columns=["peptide"], map_source_proteins=True
    )
    pd.testing.assert_frame_equal(projected, ordinary)
    assert manifest["n_rows"] == 4
    assert manifest["coverage"]["n_observations"] == 2
    assert manifest["coverage"]["n_contributor_links"] == 3
    assert manifest["training_options"]["exclude_non_peptide_ligand"] is True
    assert manifest["randomness"] == {"used": False, "seed": None}
    assert manifest["inputs"]["build_metadata"]["provenance"]["sources"]["iedb"]["sha256"]
    with pytest.raises(FileExistsError):
        write_training_bundle(directory)
    second = tmp_path / "second"
    write_training_bundle(
        second,
        include_evidence="ms",
        columns=["peptide"],
        map_source_proteins=True,
        split_policy="specimen_disjoint",
    )
    assert (directory / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    report = audit_training_bundles(
        {"train": directory, "test": second}, policy="independent_experiments"
    )
    assert report["verdict"] == "fail"
    assert report["pairs"][0]["overlaps"]["observation"]["n_shared_values"] == 2
    assert report["partitions"]["train"]["n_mapping_alternative_rows"] == 2


def test_empty_and_legacy_bundles_are_explicit(built_index, tmp_path):
    empty = tmp_path / "empty"
    write_training_bundle(empty, include_evidence="ms", peptide="NOTPRESENT")
    assert verify_training_bundle(empty)["coverage"]["n_observations"] == 0
    path = built_index / "observations.parquet"
    pd.read_parquet(path).drop(columns="provenance_id").to_parquet(path, index=False)
    (built_index / "observations_meta.json").unlink()
    contributors_path().unlink()
    legacy = tmp_path / "legacy"
    write_training_bundle(legacy, include_evidence="ms")
    manifest = verify_training_bundle(legacy)
    assert manifest["coverage"]["n_observations_with_contributors"] == 0
    assert set(pd.read_parquet(legacy / "identities.parquet").provenance_status) == {
        "legacy_missing"
    }


@pytest.mark.parametrize("artifact", ["observation_contributors.parquet", "observations.parquet"])
def test_inconsistent_build_artifacts_require_rebuild(built_index, tmp_path, artifact):
    path = built_index / artifact
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Provenance artifact mismatch"):
        load_contributors()
    assert not (tmp_path / "bundle").exists()


def test_bundle_tampering_and_failed_export_cleanup(built_index, tmp_path, monkeypatch):
    directory = tmp_path / "bundle"
    write_training_bundle(directory, include_evidence="ms")
    (directory / "contributors.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_training_bundle(directory)
    import hitlist.training_bundle as module

    original = module._inputs
    calls = []

    def changed_inputs():
        calls.append(True)
        value = original()
        if len(calls) > 1:
            value["changed"] = True
        return value

    monkeypatch.setattr(module, "_inputs", changed_inputs)
    with pytest.raises(ValueError, match="inputs changed"):
        write_training_bundle(tmp_path / "failed", include_evidence="ms")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_ordinary_projection_rejects_missing_claimed_sidecar(built_index):
    contributors_path().unlink()
    with pytest.raises(ValueError, match="Provenance artifact mismatch"):
        generate_training_table(include_evidence="ms", columns=["peptide"])


def test_copied_artifacts_accept_new_mtimes_but_hash_checks_detect_same_size_tampering(built_index):
    for name in ("observations.parquet", "binding.parquet", "observation_contributors.parquet"):
        path = built_index / name
        stat = path.stat()
        os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    assert len(generate_training_table(include_evidence="ms", columns=["peptide"])) == 2
    path = contributors_path()
    stat = path.stat()
    content = bytearray(path.read_bytes())
    content[20] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="Provenance artifact mismatch"):
        load_contributors()


def test_bundle_verification_rejects_conflicting_projected_identity_fields(built_index, tmp_path):
    directory = tmp_path / "bundle"
    write_training_bundle(directory, include_evidence="ms", columns=["peptide", "pmid"])
    path = directory / "training.parquet"
    frame = pd.read_parquet(path)
    frame["pmid"] = 12345678
    frame.to_parquet(path, index=False)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["training"].update(file_digest(path))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Training/identity pmid mismatch"):
        verify_training_bundle(directory)


def test_cache_rejects_missing_contributor_contract(built_index):
    from hitlist.builder import _cache_is_valid, _source_paths

    path = built_index / "observations_meta.json"
    metadata = json.loads(path.read_text())
    del metadata["provenance"]
    path.write_text(json.dumps(metadata))
    assert not _cache_is_valid(_source_paths())


def test_cli_bundle_and_split_audit(built_index, tmp_path, monkeypatch):
    from hitlist.cli import main

    directory = tmp_path / "bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "export",
            "training",
            "--include-evidence",
            "ms",
            "--bundle",
            str(directory),
            "--columns",
            "peptide",
        ],
    )
    main()
    assert verify_training_bundle(directory)["n_rows"] == 2
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "audit-splits",
            "--partition",
            f"train={directory}",
            "--partition",
            f"test={directory}",
            "--policy",
            "peptide_disjoint",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert json.loads(output.read_text())["verdict"] == "fail"
