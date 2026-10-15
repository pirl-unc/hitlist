"""Public corpus provenance must describe the actual files and original build."""

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_dataset_generations import tiny_dataset as _tiny_dataset

tiny_dataset = _tiny_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_corpus.py"
SPEC = importlib.util.spec_from_file_location("ci_corpus", SCRIPT)
ci_corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci_corpus)


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "private"
    source.mkdir()
    for name in ci_corpus.PARQUETS:
        (source / name).write_bytes(f"source bytes for {name}".encode())
    metadata = {
        "artifact_version": 5,
        "mhcgnomes_version": "3.64.4",
        "parquets": ci_corpus.build_fingerprints(source),
        "sources": {"/private/source/path.csv": {"size": 123}},
    }
    (source / ci_corpus.METADATA).write_text(json.dumps(metadata))
    public = tmp_path / "public"
    public.mkdir()
    return source, public


def snapshot(source, public):
    ci_corpus.write_manifest(source, public / ci_corpus.METADATA)
    for name in ci_corpus.PARQUETS:
        shutil.copyfile(source / name, public / name)


@pytest.mark.parametrize("version", [5, 6])
def test_public_snapshot_preserves_original_contract_and_hides_private_paths(corpus, version):
    source, public = corpus
    metadata_path = source / ci_corpus.METADATA
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_version"] = version
    original = json.dumps(metadata)
    metadata_path.write_text(original)
    snapshot(source, public)
    ci_corpus.verify_manifest(public)
    exported = json.loads((public / ci_corpus.METADATA).read_text())
    assert exported["artifact_version"] == version
    assert exported["mhcgnomes_version"] == metadata["mhcgnomes_version"]
    assert set(exported) == {
        "ci_corpus_manifest_version",
        "artifact_version",
        "mhcgnomes_version",
        "files",
    }
    assert "/private" not in json.dumps(exported)
    assert metadata_path.read_text() == original


def test_original_build_metadata_cannot_be_overwritten(corpus):
    source, _ = corpus
    metadata = source / ci_corpus.METADATA
    original = metadata.read_bytes()
    with pytest.raises(ValueError, match="must not replace"):
        ci_corpus.write_manifest(source, metadata)
    assert metadata.read_bytes() == original


def test_stale_build_metadata_cannot_authorize_a_different_corpus(corpus):
    source, public = corpus
    (source / "observations.parquet").write_bytes(b"rebuilt, no matching metadata")
    with pytest.raises(ValueError, match="fingerprints"):
        snapshot(source, public)
    assert not (public / ci_corpus.METADATA).exists()


@pytest.mark.parametrize("version", [None, 4, "5", True])
def test_absent_or_old_build_contract_cannot_claim_exclusion_coverage(corpus, version):
    source, public = corpus
    metadata_path = source / ci_corpus.METADATA
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_version"] = version
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="original artifact_version"):
        snapshot(source, public)


@pytest.mark.parametrize("damage", ["modify", "remove", "empty"])
def test_changed_or_incomplete_download_is_rejected(corpus, damage):
    source, public = corpus
    snapshot(source, public)
    target = public / "observations.parquet"
    if damage == "remove":
        target.unlink()
    else:
        target.write_bytes(b"wrong corpus" if damage == "modify" else b"")
    with pytest.raises((ValueError, FileNotFoundError)):
        ci_corpus.verify_manifest(public)


@pytest.mark.parametrize("damage", ["missing", "private", "old", "extra_file", "missing_file"])
def test_missing_or_incorrect_public_provenance_is_rejected(corpus, damage):
    source, public = corpus
    snapshot(source, public)
    metadata_path = public / ci_corpus.METADATA
    metadata = json.loads(metadata_path.read_text())
    if damage == "missing":
        metadata_path.unlink()
    elif damage == "private":
        shutil.copyfile(source / ci_corpus.METADATA, metadata_path)
    else:
        if damage == "old":
            metadata["artifact_version"] = 4
        elif damage == "extra_file":
            metadata["files"]["other.parquet"] = {}
        else:
            del metadata["files"]["observations.parquet"]
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        ci_corpus.verify_manifest(public)


@pytest.mark.parametrize("condition", ["valid", "current_valid", "stale", "existing_release"])
def test_publisher_uploads_only_a_verified_snapshot_and_never_clobbers(corpus, condition, request):
    source, public = corpus
    metadata_directory = source
    if condition == "current_valid":
        source, metadata_directory = request.getfixturevalue("current_corpus")
    tools = source.parent / "tools"
    tools.mkdir()
    (tools / "python").write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    (tools / "python").chmod(0o755)
    fake_gh = tools / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env python\n"
        "import os, pathlib, shutil, sys\n"
        "if sys.argv[1:3] == ['release', 'view']:\n"
        "    sys.exit(0 if os.environ['RELEASE_EXISTS'] == 'yes' else 1)\n"
        "assert sys.argv[1:3] == ['release', 'create'], sys.argv\n"
        "assert '--clobber' not in sys.argv\n"
        "assert '--latest=false' in sys.argv\n"
        "for argument in sys.argv[4:]:\n"
        "    if argument.endswith(('.parquet', '.json', '.md')):\n"
        "        path = pathlib.Path(argument)\n"
        "        assert path.parent != pathlib.Path(os.environ['HITLIST_DATA_DIR'])\n"
        "        shutil.copyfile(path, pathlib.Path(os.environ['ASSET_DIR']) / path.name)\n"
    )
    fake_gh.chmod(0o755)
    original_metadata = (metadata_directory / ci_corpus.METADATA).read_bytes()
    if condition == "stale":
        (source / "binding.parquet").write_bytes(b"a different build")
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("publish_ci_corpus.sh")), "test"],
        env=dict(
            os.environ,
            PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
            HITLIST_DATA_DIR=str(source),
            ASSET_DIR=str(public),
            RELEASE_EXISTS="yes" if condition == "existing_release" else "no",
        ),
        capture_output=True,
        text=True,
    )
    assert (metadata_directory / ci_corpus.METADATA).read_bytes() == original_metadata
    if condition in {"valid", "current_valid"}:
        assert result.returncode == 0, result.stderr
        ci_corpus.verify_manifest(public)
        assert "sources" not in json.loads((public / ci_corpus.METADATA).read_text())
    else:
        assert result.returncode != 0
        assert not list(public.iterdir())


@pytest.fixture
def current_corpus(tiny_dataset, monkeypatch):
    from hitlist import builder, downloads, mappings
    from tests.test_dataset_generations import _REAL_BUILD_MAPPINGS

    builder.build_observations(build_mappings=False)
    monkeypatch.setattr(mappings, "build_peptide_mappings", _REAL_BUILD_MAPPINGS)
    monkeypatch.setattr(downloads, "lookup_proteome", lambda *args, **kw: None)
    monkeypatch.setattr(builder, "_collect_pmid_extra_proteomes", lambda: {})
    mappings.build_peptide_mappings(fetch_missing=False, verbose=False)
    return downloads.data_dir(), builder._observations_path().parent


def test_actual_current_builder_publishes_complete_portable_provenance(
    current_corpus, tmp_path, monkeypatch
):
    from hitlist import downloads, load_contributors

    root, generation = current_corpus
    published = ci_corpus.snapshot_corpus(root, cache_dir=tmp_path / "public-cache")
    assert ci_corpus.snapshot_corpus(root, cache_dir=tmp_path / "public-cache") == published
    manifest = ci_corpus.verify_manifest(published)
    assert manifest["artifact_version"] == 10
    assert manifest["ci_corpus_manifest_version"] == 2
    assert set(manifest["files"]) == set(ci_corpus.CURRENT_FILES)
    assert manifest["provenance"]["sources"]["iedb"]["sha256"]
    assert manifest["provenance"]["n_contributor_links"] > 0
    for name in (ci_corpus.METADATA, "peptide_mappings_meta.json", "README.md"):
        assert str(tmp_path) not in (published / name).read_text()
    for name in ci_corpus.CURRENT_PARQUETS:
        assert (published / name).read_bytes() == (generation / name).read_bytes()
    mapping = json.loads((published / "peptide_mappings_meta.json").read_text())
    assert mapping["contract"]["flank"] == 15
    assert mapping["unmapped_organisms"]
    assert (
        mapping["observation_hashes"]["other_assays"] == manifest["files"]["other_assays.parquet"]
    )
    # The public receipt remains usable by the real contributor reader.
    monkeypatch.setattr(downloads, "_override_data_dir", published)
    assert not load_contributors().empty


@pytest.mark.parametrize(
    "name",
    ["other_assays.parquet", "observation_contributors.parquet", "peptide_mappings_meta.json"],
)
def test_current_publication_never_fills_missing_files_from_flat_cache(
    current_corpus, tmp_path, name
):
    root, generation = current_corpus
    shutil.copy2(generation / name, root / name)
    (generation / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        ci_corpus.snapshot_corpus(root, cache_dir=tmp_path / "public-cache")
    assert not list((tmp_path / "public-cache").glob("*/bundles/*"))


@pytest.mark.parametrize("damage", ["index", "contributor_hash", "mapping", "source_identity"])
def test_current_snapshot_rejects_stale_or_inconsistent_claims(current_corpus, tmp_path, damage):
    _, generation = current_corpus
    metadata = json.loads((generation / ci_corpus.METADATA).read_text())
    if damage == "index":
        (generation / "other_assays.parquet").write_bytes(b"changed")
    elif damage == "contributor_hash":
        metadata["provenance"]["contributors"]["sha256"] = "0" * 64
    elif damage == "source_identity":
        metadata["provenance"]["sources"]["iedb"].pop("sha256")
    else:
        path = generation / "peptide_mappings_meta.json"
        mapping = json.loads(path.read_text())
        mapping["observations"]["other_assays"]["size"] += 1
        path.write_text(json.dumps(mapping))
    (generation / ci_corpus.METADATA).write_text(json.dumps(metadata))
    with pytest.raises(
        ValueError, match=r"fingerprints|content hash|source index|content identity"
    ):
        ci_corpus.snapshot_corpus(generation, cache_dir=tmp_path / "public-cache")


def test_snapshot_pins_one_generation_across_a_concurrent_publication(
    current_corpus, tmp_path, monkeypatch
):
    import datacache

    from hitlist import builder

    root, first = current_corpus
    materialize = datacache.materialize
    switched = []

    def switch(*args, **kwargs):
        # Called only after publication has resolved and inspected its inputs.
        second = builder.build_observations(force=True, build_mappings=False).parent
        switched.append(second)
        return materialize(*args, **kwargs)

    monkeypatch.setattr(datacache, "materialize", switch)
    published = ci_corpus.snapshot_corpus(root, cache_dir=tmp_path / "public-cache")
    assert switched and switched[0] != first
    assert (published / "peptide_mappings.parquet").read_bytes() == (
        first / "peptide_mappings.parquet"
    ).read_bytes()
    assert not (switched[0] / "peptide_mappings.parquet").exists()
    ci_corpus.verify_manifest(published)


def test_interrupted_portable_copy_preserves_the_existing_snapshot(
    current_corpus, tmp_path, monkeypatch
):
    root, _ = current_corpus
    cache = tmp_path / "public-cache"
    published = ci_corpus.snapshot_corpus(root, cache_dir=cache)
    before = (published / ci_corpus.METADATA).read_bytes()
    copyfile = ci_corpus.shutil.copyfile

    def fail(source, destination, *args, **kwargs):
        if Path(destination).name == "binding.parquet":
            raise OSError("injected portable copy failure")
        return copyfile(source, destination, *args, **kwargs)

    # A different recipe identity forces a new transform without modifying data.
    original_digest = ci_corpus.file_digest

    def digest(path):
        result = original_digest(path)
        return {**result, "sha256": "1" * 64} if path == Path(ci_corpus.__file__) else result

    monkeypatch.setattr(ci_corpus, "file_digest", digest)
    monkeypatch.setattr(ci_corpus.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="injected portable"):
        ci_corpus.snapshot_corpus(root, cache_dir=cache)
    assert (published / ci_corpus.METADATA).read_bytes() == before
    ci_corpus.verify_manifest(published)


def test_same_size_mapping_corruption_cannot_be_republished(current_corpus, tmp_path):
    root, generation = current_corpus
    path = generation / "peptide_mappings.parquet"
    stamp = path.stat()
    damaged = bytearray(path.read_bytes())
    damaged[8] ^= 1
    path.write_bytes(damaged)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="content hash mismatch: peptide_mappings"):
        ci_corpus.snapshot_corpus(root, cache_dir=tmp_path / "public-cache")


def test_pinned_download_reuses_verified_public_artifacts(current_corpus, tmp_path):
    from datacache import FileValidationError

    root, _ = current_corpus
    published = ci_corpus.snapshot_corpus(root, cache_dir=tmp_path / "publisher")
    manifest = published / ci_corpus.METADATA
    digest = ci_corpus.file_digest(manifest)["sha256"]
    installed = ci_corpus.install_corpus(manifest.as_uri(), digest, cache_dir=tmp_path / "consumer")
    assert installed != published
    assert ci_corpus.verify_manifest(installed) == ci_corpus.verify_manifest(published)
    # Reuse requires no reachable source files once both pinned bundles exist.
    manifest.rename(manifest.with_suffix(".offline"))
    assert (
        ci_corpus.install_corpus(manifest.as_uri(), digest, cache_dir=tmp_path / "consumer")
        == installed
    )
    (installed / "observation_contributors.parquet").write_bytes(b"damaged")
    with pytest.raises(FileValidationError, match=r"damaged|size"):
        ci_corpus.install_corpus(manifest.as_uri(), digest, cache_dir=tmp_path / "consumer")
