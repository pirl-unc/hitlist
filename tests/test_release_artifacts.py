"""Reject stale, modified or unapproved release-build artifacts before publication."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("release_artifacts", SCRIPT)
release_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_artifacts)
CORPUS_FILES = release_artifacts.CORPUS_FILES
verify_manifest = release_artifacts.verify_manifest
write_manifest = release_artifacts.write_manifest


def development_distributions():
    return [
        SimpleNamespace(
            metadata={"Name": name},
            version="1.2.3",
            read_text=lambda filename: json.dumps({"vcs_info": {"commit_id": "a" * 40}}),
        )
        for name in release_artifacts.DEVELOPMENT_DEPENDENCIES
    ]


@pytest.fixture
def release_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        release_artifacts.importlib.metadata, "distributions", development_distributions
    )
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
    (corpus / "observations_meta.json").write_text(
        json.dumps(
            {
                "ci_corpus_manifest_version": 1,
                "artifact_version": 5,
                "mhcgnomes_version": "3.64.4",
                "files": {
                    name: release_artifacts.file_digest(corpus / name)
                    for name in CORPUS_FILES
                    if name != "observations_meta.json"
                },
            }
        )
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    write_manifest(dist, corpus, root)
    manifest = json.loads((dist / "release.json").read_text())
    trusted_files = {path.name: path.read_bytes() for path in dist.iterdir()}
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
        if command[:3] == ["gh", "run", "download"]:
            assert command[3] == "123"
            assert command[command.index("--name") + 1] == (
                f"hitlist-release-{manifest['source_commit']}"
            )
            assert command[command.index("--repo") + 1] == "pirl-unc/hitlist"
            destination = Path(command[command.index("--dir") + 1])
            for name, data in trusted_files.items():
                (destination / name).write_bytes(data)
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_command)
    return root, dist, corpus, run


def test_verified_main_release_matches_both_artifacts(release_bundle):
    root, dist, _, _ = release_bundle
    verify_manifest(dist, root)


def test_trusted_artifact_download_is_bounded_and_run_scoped(release_bundle, monkeypatch):
    root, dist, _, _ = release_bundle
    commit = json.loads((dist / "release.json").read_text())["source_commit"]
    original_run = subprocess.run

    def download(command, **kwargs):
        if command[:2] == ["gh", "api"] and command[-1].endswith("/zip"):
            pytest.fail("Direct artifact ZIP request has no bounded download contract")
        if command[:3] == ["gh", "run", "download"]:
            assert command[3] == "123"
            assert command[command.index("--name") + 1] == f"hitlist-release-{commit}"
            assert command[command.index("--repo") + 1] == "pirl-unc/hitlist"
            assert kwargs["timeout"] == 120
            target = Path(command[command.index("--dir") + 1])
            for file in dist.iterdir():
                shutil.copy2(file, target / file.name)
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", download)
    verify_manifest(dist, root)


def test_release_metadata_timeout_fails_closed(release_bundle, monkeypatch):
    root, dist, _, _ = release_bundle
    original_output = subprocess.check_output

    def timed_out(command, **kwargs):
        if command[:2] == ["gh", "api"]:
            assert kwargs["timeout"] == 30
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", timed_out)
    with pytest.raises(ValueError, match="Timed out querying GitHub release provenance"):
        verify_manifest(dist, root)


def test_partial_download_timeout_fails_closed_and_cleans_up(release_bundle, monkeypatch):
    root, dist, _, _ = release_bundle
    original_run = subprocess.run
    downloaded_to = []

    def timed_out(command, **kwargs):
        if command[:3] == ["gh", "run", "download"]:
            destination = Path(command[command.index("--dir") + 1])
            (destination / "release.json").write_text("partial")
            downloaded_to.append(destination)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(ValueError, match="Timed out downloading"):
        verify_manifest(dist, root)
    assert downloaded_to and not downloaded_to[0].exists()


def test_failed_artifact_download_fails_closed(release_bundle, monkeypatch):
    root, dist, _, _ = release_bundle
    original_run = subprocess.run

    def failed(command, **kwargs):
        if command[:3] == ["gh", "run", "download"]:
            raise subprocess.CalledProcessError(1, command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ValueError, match="artifact download failed"):
        verify_manifest(dist, root)


@pytest.mark.parametrize(
    "change", ["missing_receipt", "corrupt_receipt", "modified_wheel", "symlinked_receipt"]
)
def test_incomplete_or_corrupt_artifact_fails_closed(release_bundle, monkeypatch, change):
    root, dist, _, _ = release_bundle
    original_run = subprocess.run

    def changed_download(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[:3] == ["gh", "run", "download"]:
            destination = Path(command[command.index("--dir") + 1])
            if change == "missing_receipt":
                (destination / "release.json").unlink()
            elif change == "corrupt_receipt":
                (destination / "release.json").write_text("{invalid JSON")
            elif change == "symlinked_receipt":
                (destination / "release.json").unlink()
                (destination / "release.json").symlink_to(dist / "release.json")
            else:
                (destination / "hitlist-1.2.3-py3-none-any.whl").write_bytes(b"changed bytes")
        return result

    monkeypatch.setattr(subprocess, "run", changed_download)
    message = {
        "missing_receipt": "missing or unexpected files",
        "corrupt_receipt": "unreadable release manifest",
        "modified_wheel": "distributions disagree with its manifest",
        "symlinked_receipt": "missing or unexpected files",
    }[change]
    with pytest.raises(ValueError, match=message):
        verify_manifest(dist, root)


def test_artifact_replacement_during_download_fails_closed(release_bundle, monkeypatch):
    root, dist, _, _ = release_bundle
    original_output = subprocess.check_output
    inventory_calls = 0

    def replaced(command, **kwargs):
        nonlocal inventory_calls
        response = original_output(command, **kwargs)
        if command[:2] == ["gh", "api"] and command[-1].endswith("/runs/123/artifacts"):
            inventory_calls += 1
            if inventory_calls == 2:
                inventory = json.loads(response)
                inventory["artifacts"][0]["id"] = 790
                return json.dumps(inventory)
        return response

    monkeypatch.setattr(subprocess, "check_output", replaced)
    with pytest.raises(ValueError, match="artifact identity changed"):
        verify_manifest(dist, root)
    assert inventory_calls == 2


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


@pytest.mark.parametrize("filename", ["observations.parquet", "observations_meta.json"])
def test_missing_corpus_cannot_produce_release_manifest(release_bundle, filename):
    root, dist, corpus, _ = release_bundle
    (corpus / filename).unlink()
    (dist / "release.json").unlink()
    with pytest.raises(FileNotFoundError):
        write_manifest(dist, corpus, root)
    assert not (dist / "release.json").exists()


@pytest.mark.parametrize("missing_installation", [False, True])
def test_pypi_version_without_git_revision_cannot_authorize_release(
    release_bundle, monkeypatch, missing_installation
):
    root, dist, corpus, _ = release_bundle
    distributions = development_distributions()
    if missing_installation:
        distributions.pop()
    else:
        # Same version as the Git install, but installed from the package index.
        distributions[-1].read_text = lambda filename: None
    monkeypatch.setattr(
        release_artifacts.importlib.metadata, "distributions", lambda: distributions
    )
    (dist / "release.json").unlink()
    with pytest.raises(ValueError, match=r"lack installed Git revisions.*mhcgnomes"):
        write_manifest(dist, corpus, root)
    assert not (dist / "release.json").exists()


@pytest.fixture
def publication(tmp_path, monkeypatch):
    import hashlib
    import io

    directory = tmp_path / "dist"
    directory.mkdir()
    payload = b"precisely the tested release bytes"
    files = {}
    for name in ("hitlist-1.2.3-py3-none-any.whl", "hitlist-1.2.3.tar.gz"):
        (directory / name).write_bytes(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    manifest = {
        "version": "1.2.3",
        "source_commit": "abc123",
        "workflow_run_id": "123",
        "files": files,
    }
    public = {}
    calls = []

    def item(name):
        return {
            "filename": name,
            "digests": {"sha256": files[name]["sha256"]},
            "size": len(payload),
            "url": f"https://files.pythonhosted.org/{name}",
            "yanked": False,
        }

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["twine", "upload"]:
            for path in command[3:]:
                assert Path(path).read_bytes() == payload
                public[Path(path).name] = item(Path(path).name)

    monkeypatch.setattr(release_artifacts, "public_files", lambda version: public.copy())
    monkeypatch.setattr(
        release_artifacts.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    monkeypatch.setattr(release_artifacts.subprocess, "run", run)
    return directory, manifest, public, calls, item


def test_publication_uploads_only_tested_pair_and_verifies_downloads(publication):
    directory, manifest, public, calls, _ = publication
    release_artifacts.publish_distributions(directory, manifest)
    assert set(public) == set(manifest["files"])
    assert [command[:2] for command in calls] == [["twine", "check"], ["twine", "upload"]]
    receipt = json.loads((directory.parent / "pypi-1.2.3-verified.json").read_text())
    assert receipt["public_files"] == manifest["files"]
    # Repeating a successful publication does not upload anything again.
    calls.clear()
    release_artifacts.publish_distributions(directory, manifest)
    assert calls == []


def test_matching_partial_publication_can_resume_but_is_never_complete(publication):
    directory, manifest, public, calls, item = publication
    name = next(iter(manifest["files"]))
    public[name] = item(name)
    with pytest.raises(ValueError, match="inventory"):
        release_artifacts.verify_public_files(manifest)
    release_artifacts.publish_distributions(directory, manifest)
    assert len(calls[-1][3:]) == 1
    assert Path(calls[-1][-1]).name != name


@pytest.mark.parametrize("damage", ["hash", "size", "yanked", "unexpected", "download", "host"])
def test_publication_conflicts_fail_before_upload(publication, monkeypatch, damage):
    import io

    directory, manifest, public, calls, item = publication
    name = next(iter(manifest["files"]))
    public[name] = item(name)
    if damage == "hash":
        public[name]["digests"]["sha256"] = "0" * 64
    elif damage == "size":
        public[name]["size"] += 1
    elif damage == "yanked":
        public[name]["yanked"] = True
    elif damage == "unexpected":
        public["unexpected.whl"] = item(name)
    elif damage == "host":
        public[name]["url"] = "https://example.org/distribution"
    else:
        monkeypatch.setattr(
            release_artifacts.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong")
        )
    with pytest.raises(ValueError):
        release_artifacts.publish_distributions(directory, manifest)
    assert calls == []
    assert not (directory.parent / "pypi-1.2.3-verified.json").exists()


def test_changed_local_distribution_cannot_be_uploaded(publication):
    directory, manifest, _, calls, _ = publication
    (directory / next(iter(manifest["files"]))).write_bytes(b"changed after verification")
    with pytest.raises(ValueError, match="changed after"):
        release_artifacts.publish_distributions(directory, manifest)
    assert calls == []


@pytest.mark.parametrize("status", [404, 403, 429, 500])
def test_only_pypi_not_found_means_unpublished(monkeypatch, status):
    import urllib.error

    def unavailable(*args, **kwargs):
        raise urllib.error.HTTPError("https://pypi.org", status, "failure", {}, None)

    monkeypatch.setattr(release_artifacts.urllib.request, "urlopen", unavailable)
    if status == 404:
        assert release_artifacts.public_files("1.2.3") == {}
    else:
        with pytest.raises(urllib.error.HTTPError):
            release_artifacts.public_files("1.2.3")
