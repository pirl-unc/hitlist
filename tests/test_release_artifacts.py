"""Reject stale, modified or unapproved release-build artifacts before publication."""

import importlib.util
import json
import subprocess
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("release_artifacts", SCRIPT)
release_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_artifacts)
CORPUS_FILES = release_artifacts.CORPUS_FILES
verify_manifest = release_artifacts.verify_manifest
write_manifest = release_artifacts.write_manifest


@pytest.fixture
def release_bundle(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "hitlist").mkdir()
    (root / "hitlist" / "version.py").write_text('__version__ = "1.2.3"\n')
    for command in (
        ["init", "-b", "main"],
        ["add", "."],
        [
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-m",
            "base",
        ],
    ):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    dist = tmp_path / "artifacts"
    dist.mkdir()
    for filename in ("hitlist-1.2.3-py3-none-any.whl", "hitlist-1.2.3.tar.gz"):
        (dist / filename).write_bytes(b"original release bytes")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for filename in CORPUS_FILES:
        (corpus / filename).write_bytes(b"corpus")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    write_manifest(dist, corpus, root)
    manifest = json.loads((dist / "release.json").read_text())
    archive = BytesIO()
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("release.json", json.dumps(manifest))
    archive_bytes = archive.getvalue()
    run = {
        "workflow_id": 456,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": manifest["source_commit"],
        "status": "completed",
        "conclusion": "success",
    }
    original_output = subprocess.check_output

    def command_output(command, **kwargs):
        if command[:2] == ["gh", "api"]:
            if command[-1].endswith("/workflows/release-build.yml"):
                return json.dumps({"id": 456})
            if command[-1].endswith("/runs/123/artifacts"):
                return json.dumps(
                    {
                        "artifacts": [
                            {
                                "id": 789,
                                "expired": False,
                                "name": f"hitlist-release-{manifest['source_commit']}",
                            }
                        ]
                    }
                )
            assert command[-1].endswith("/runs/123")
            return json.dumps(run)
        return original_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", command_output)
    original_run = subprocess.run

    def run_command(command, **kwargs):
        if command[:2] == ["gh", "api"]:
            assert command[-1].endswith("/artifacts/789/zip")
            kwargs["stdout"].write(archive_bytes)
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_command)
    return root, dist, corpus, run


def test_verified_main_release_matches_both_artifacts(release_bundle):
    root, dist, _, _ = release_bundle
    verify_manifest(dist, root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "pull_request"),
        ("head_branch", "feature"),
        ("head_sha", "0" * 40),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("workflow_id", 789),
    ],
)
def test_only_successful_manual_main_workflow_can_publish(release_bundle, field, value):
    root, dist, _, run = release_bundle
    run[field] = value
    with pytest.raises(ValueError, match="successful manual release build"):
        verify_manifest(dist, root)


def test_modified_distribution_is_rejected(release_bundle):
    root, dist, _, _ = release_bundle
    (dist / "hitlist-1.2.3.tar.gz").write_bytes(b"changed release bytes")
    with pytest.raises(ValueError, match="checksum or size"):
        verify_manifest(dist, root)


def test_forged_local_manifest_cannot_claim_an_existing_successful_run(release_bundle):
    root, dist, _, _ = release_bundle
    archive = dist / "hitlist-1.2.3.tar.gz"
    archive.write_bytes(b"changed release bytes")
    manifest_path = dist / "release.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][archive.name] = release_artifacts.file_digest(archive)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from the verified GitHub artifact"):
        verify_manifest(dist, root)


def test_unexpected_distribution_cannot_be_uploaded(release_bundle):
    root, dist, _, _ = release_bundle
    (dist / "other.whl").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="Expected distributions"):
        verify_manifest(dist, root)


def test_stale_source_commit_is_rejected(release_bundle):
    root, dist, _, _ = release_bundle
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-m",
            "new main",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="current main commit"):
        verify_manifest(dist, root)


def test_dirty_main_cannot_verify_artifacts(release_bundle):
    root, dist, _, _ = release_bundle
    (root / "hitlist" / "version.py").write_text('__version__ = "1.2.4"\n')
    with pytest.raises(ValueError, match="tracked changes"):
        verify_manifest(dist, root)


def test_feature_checkout_cannot_publish(release_bundle):
    root, dist, _, _ = release_bundle
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=root, check=True, capture_output=True)
    with pytest.raises(ValueError, match="clean main checkout"):
        verify_manifest(dist, root)


def test_missing_corpus_cannot_produce_release_manifest(release_bundle):
    root, dist, corpus, _ = release_bundle
    (corpus / "observations.parquet").unlink()
    (dist / "release.json").unlink()
    with pytest.raises(FileNotFoundError):
        write_manifest(dist, corpus, root)
    assert not (dist / "release.json").exists()
