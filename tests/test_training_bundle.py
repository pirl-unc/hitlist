import json
import os
import shutil
import sys

import pandas as pd
import pytest

from hitlist.export import generate_training_table
from hitlist.proteome import DEFAULT_FLANK
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
    if getattr(request, "param", False) is True:
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
    generation = builder.build_observations(build_mappings=False).parent
    from hitlist import mappings
    from hitlist.dataset import update_dataset

    # Most cases intentionally exercise editable legacy indexes and their
    # corruption checks. A separate parameter exercises immutable generations.
    use_generation = getattr(request, "param", False) == "generation"
    if not use_generation:
        legacy = tmp_path / "legacy-indexes"
        legacy.mkdir()
        shutil.copy2(indexes / "manifest.json", legacy / "manifest.json")
        for path in generation.iterdir():
            if path.name.endswith((".parquet", ".json")) and not path.name.startswith("."):
                shutil.copy2(path, legacy / path.name)
        monkeypatch.setattr(downloads, "_override_data_dir", legacy)
        indexes = legacy

    def write_mappings():
        pd.DataFrame({"peptide": ["SLYNTVATL"] * 2, "protein_id": ["P1", "P2"]}).to_parquet(
            mappings.mappings_path(), index=False
        )
        mappings.mappings_meta_path().write_text(
            json.dumps(
                {
                    "observations": mappings._obs_fingerprint(),
                    "contract": mappings._mapping_artifact_contract(
                        release=112, use_uniprot=False, fetch_missing=True, flank=DEFAULT_FLANK
                    ),
                }
            )
        )
        return mappings.mappings_path()

    installed = update_dataset(
        ("peptide_mappings.parquet", "peptide_mappings_meta.json"), write_mappings
    )
    if use_generation:
        indexes = installed.parent
    monkeypatch.setattr(
        "hitlist.export._load_training_mappings_for_peptides",
        lambda *a, **kw: pd.DataFrame({"peptide": ["SLYNTVATL"] * 2, "protein_id": ["P1", "P2"]}),
    )
    return indexes


@pytest.mark.parametrize("built_index", [False, "generation"], indirect=True)
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
    assert manifest["input_validation"]["require_current_indexes"] is True
    assert manifest["input_validation"]["observations"] == "current"
    assert manifest["input_validation"]["mappings"] == "current"
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
    assert "contributor_pmids" in pd.read_parquet(empty / "identities.parquet")
    path = built_index / "observations.parquet"
    pd.read_parquet(path).drop(columns="provenance_id").to_parquet(path, index=False)
    (built_index / "observations_meta.json").unlink()
    contributors_path().unlink()
    legacy = tmp_path / "legacy"
    with pytest.raises(ValueError, match="current input indexes"):
        write_training_bundle(legacy, include_evidence="ms")
    write_training_bundle(legacy, include_evidence="ms", require_current_indexes=False)
    manifest = verify_training_bundle(legacy)
    assert manifest["coverage"]["n_observations_with_contributors"] == 0
    assert manifest["input_validation"]["require_current_indexes"] is False
    assert manifest["input_validation"]["observations"] == "stale"
    assert set(pd.read_parquet(legacy / "identities.parquet").provenance_status) == {
        "legacy_missing"
    }
    assert set(pd.read_parquet(legacy / "identities.parquet").contributor_pmids) == {"[]"}


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


def test_frozen_bundle_verification_does_not_recheck_mutable_local_inputs(
    built_index, tmp_path, monkeypatch
):
    from hitlist import observations

    directory = tmp_path / "bundle"
    write_training_bundle(directory, include_evidence="ms")
    monkeypatch.setattr(observations, "observations_cache_is_current", lambda: False)
    assert verify_training_bundle(directory)["input_validation"]["observations"] == "current"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["input_validation"]  # Existing schema-v1 bundles predate preflight.
    manifest_path.write_text(json.dumps(manifest))
    assert "input_validation" not in verify_training_bundle(directory)


def test_freshness_change_during_export_cleans_up(built_index, tmp_path, monkeypatch):
    from hitlist import observations

    statuses = iter([True, False])
    monkeypatch.setattr(observations, "observations_cache_is_current", lambda: next(statuses))
    directory = tmp_path / "bundle"
    with pytest.raises(ValueError, match="current input indexes"):
        write_training_bundle(directory, include_evidence="ms")
    assert not directory.exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_cli_archival_opt_out_requires_bundle(tmp_path, monkeypatch, capsys):
    from hitlist.cli import main

    output = tmp_path / "training.csv"
    monkeypatch.setattr(
        sys, "argv", ["hitlist", "export", "training", "--allow-stale-indexes", "-o", str(output)]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "--allow-stale-indexes requires --bundle" in capsys.readouterr().err
    assert not output.exists()


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


@pytest.mark.parametrize("built_index", ["generation"], indirect=True)
def test_export_remains_bound_to_generation_published_at_start(built_index, tmp_path, monkeypatch):
    from hitlist import builder, training_bundle
    from hitlist.dataset import dataset_path

    original = training_bundle._inputs
    calls = []
    published = []

    def inputs(**kwargs):
        value = original(**kwargs)
        calls.append(value)
        if len(calls) == 1:
            published.append(builder._write_bulk_index(pd.DataFrame()))
        return value

    monkeypatch.setattr(training_bundle, "_inputs", inputs)
    directory = tmp_path / "bundle"
    write_training_bundle(directory, include_evidence="ms")
    assert published[0].parent != built_index
    assert dataset_path("observations.parquet").parent == published[0].parent
    manifest = verify_training_bundle(directory)
    assert manifest["inputs"]["files"]["observations.parquet"]["path"] == str(
        built_index / "observations.parquet"
    )
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("built_index", ["generation"], indirect=True)
def test_bundle_checks_mapping_bytes_against_generation_receipt(built_index, tmp_path):
    path = built_index / "peptide_mappings.parquet"
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[20] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="Dataset generation hash mismatch: peptide_mappings"):
        write_training_bundle(tmp_path / "bundle", include_evidence="ms", map_source_proteins=True)
    assert not (tmp_path / "bundle").exists()
