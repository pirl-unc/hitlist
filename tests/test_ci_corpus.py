"""Public corpus provenance must describe the actual files and original build."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("condition", ["valid", "stale", "existing_release"])
def test_publisher_uploads_only_a_verified_snapshot_and_never_clobbers(corpus, condition):
    source, public = corpus
    tools = source.parent / "tools"
    tools.mkdir()
    (tools / "python").symlink_to(sys.executable)
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
        "    if argument.endswith(('.parquet', '.json')):\n"
        "        path = pathlib.Path(argument)\n"
        "        assert path.parent != pathlib.Path(os.environ['HITLIST_DATA_DIR'])\n"
        "        shutil.copyfile(path, pathlib.Path(os.environ['ASSET_DIR']) / path.name)\n"
    )
    fake_gh.chmod(0o755)
    original_metadata = (source / ci_corpus.METADATA).read_bytes()
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
    assert (source / ci_corpus.METADATA).read_bytes() == original_metadata
    if condition == "valid":
        assert result.returncode == 0, result.stderr
        ci_corpus.verify_manifest(public)
        assert "sources" not in json.loads((public / ci_corpus.METADATA).read_text())
    else:
        assert result.returncode != 0
        assert not list(public.iterdir())
