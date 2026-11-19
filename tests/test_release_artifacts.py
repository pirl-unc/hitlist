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

    def simple_item(name):
        entry = public[name]
        return {
            "filename": name,
            "hashes": {"sha256": entry["digests"]["sha256"]},
            "size": entry["size"],
            "url": entry["url"],
            "yanked": entry["yanked"],
        }

    def installer_report(version):
        wheel_name = f"hitlist-{version}-py3-none-any.whl"
        if wheel_name not in public:
            raise release_artifacts.IncompleteInventoryError(
                f"Uncached public installer cannot yet resolve hitlist=={version}"
            )
        entry = public[wheel_name]
        return {
            "version": "1",
            "pip_version": "24.2",
            "install": [
                {
                    "is_yanked": bool(entry["yanked"]),
                    "metadata": {"name": "hitlist", "version": version},
                    "download_info": {
                        "url": entry["url"],
                        "archive_info": {
                            "hashes": {"sha256": entry["digests"]["sha256"]},
                        },
                    },
                }
            ],
        }

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["twine", "upload"]:
            for path in command[3:]:
                assert Path(path).read_bytes() == payload
                public[Path(path).name] = item(Path(path).name)

    monkeypatch.setattr(release_artifacts, "public_files", lambda version: public.copy())
    monkeypatch.setattr(
        release_artifacts,
        "simple_files",
        lambda version: {name: simple_item(name) for name in public},
    )
    monkeypatch.setattr(release_artifacts, "run_installer_dry_run", installer_report)
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
    assert set(receipt["visibility"]) == {
        "project_json",
        "simple_json",
        "installer_dry_run",
        "scope",
    }
    assert (
        receipt["visibility"]["installer_dry_run"]["resolved_filename"]
        == "hitlist-1.2.3-py3-none-any.whl"
    )
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


def test_delayed_visibility_across_project_simple_and_installer_waits_without_reupload(
    publication, monkeypatch, capsys
):
    import io

    directory, manifest, _public, calls, item = publication
    wheel = "hitlist-1.2.3-py3-none-any.whl"
    sdist = "hitlist-1.2.3.tar.gz"
    payload = (directory / wheel).read_bytes()

    uploaded = False

    def run_upload(command, **kwargs):
        nonlocal uploaded
        calls.append(command)
        if command[:2] == ["twine", "upload"]:
            uploaded = True

    monkeypatch.setattr(release_artifacts.subprocess, "run", run_upload)

    project_states = iter(
        [
            {},  # pre-upload check
            {},  # post-upload poll 1: 404
            {wheel: item(wheel)},  # poll 2: partial (wheel only)
            {wheel: item(wheel), sdist: item(sdist)},  # poll 3: complete
        ]
    )
    monkeypatch.setattr(release_artifacts, "public_files", lambda version: next(project_states))

    def to_simple(entry):
        return {
            "filename": entry["filename"],
            "hashes": {"sha256": entry["digests"]["sha256"]},
            "size": entry["size"],
            "url": entry["url"],
            "yanked": False,
        }

    simple_states = iter(
        [
            {wheel: to_simple(item(wheel))},  # poll 3: simple still missing sdist
            {wheel: to_simple(item(wheel)), sdist: to_simple(item(sdist))},  # poll 4: complete
        ]
    )
    monkeypatch.setattr(release_artifacts, "simple_files", lambda version: next(simple_states))

    installer_calls = 0

    def delayed_installer(version):
        nonlocal installer_calls
        installer_calls += 1
        if installer_calls == 1:
            raise release_artifacts.IncompleteInventoryError(
                f"Uncached public installer cannot yet resolve hitlist=={version}"
            )
        return {
            "version": "1",
            "pip_version": "26.3",
            "install": [
                {
                    "is_yanked": False,
                    "metadata": {"name": "hitlist", "version": version},
                    "download_info": {
                        "url": item(wheel)["url"],
                        "archive_info": {
                            "hashes": {"sha256": item(wheel)["digests"]["sha256"]},
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(release_artifacts, "run_installer_dry_run", delayed_installer)

    downloaded_urls = []

    def track_downloads(url_or_req, *args, **kwargs):
        downloaded_urls.append(str(url_or_req))
        return io.BytesIO(payload)

    monkeypatch.setattr(release_artifacts.urllib.request, "urlopen", track_downloads)

    now = 100.0
    slept = []

    def fake_sleep(seconds):
        nonlocal now
        slept.append(seconds)
        now += seconds

    release_artifacts.publish_distributions(
        directory,
        manifest,
        timeout_seconds=60.0,
        poll_interval_seconds=5.0,
        sleep_fn=fake_sleep,
        monotonic_fn=lambda: now,
    )
    assert uploaded
    assert [command[:2] for command in calls] == [["twine", "check"], ["twine", "upload"]]
    # Wheel downloaded once during partial poll and not re-downloaded on complete poll; sdist once.
    assert downloaded_urls == [item(wheel)["url"], item(sdist)["url"]]
    assert slept == [5.0, 5.0, 5.0, 5.0]
    err = capsys.readouterr().err
    assert "project_json" in err and "simple_json" in err and "installer_dry_run" in err
    receipt = json.loads((directory.parent / "pypi-1.2.3-verified.json").read_text())
    assert receipt["visibility"]["installer_dry_run"]["pip_version"] == "26.3"


def test_visibility_wait_respects_bounded_deadline_and_leaves_no_receipt(publication, monkeypatch):
    directory, manifest, _, calls, _ = publication
    monkeypatch.setattr(release_artifacts, "public_files", lambda version: {})
    now = 0.0
    slept = []

    def fake_sleep(seconds):
        nonlocal now
        slept.append(seconds)
        now += seconds

    with pytest.raises(ValueError, match=r"Timed out after 12\.0s.*project_json"):
        release_artifacts.publish_distributions(
            directory,
            manifest,
            timeout_seconds=12.0,
            poll_interval_seconds=5.0,
            sleep_fn=fake_sleep,
            monotonic_fn=lambda: now,
        )
    assert [command[:2] for command in calls] == [["twine", "check"], ["twine", "upload"]]
    assert slept == [5.0, 5.0, 2.0]
    assert not (directory.parent / "pypi-1.2.3-verified.json").exists()


@pytest.mark.parametrize("conflict", ["yanked_partial", "bad_hash_partial", "http_503"])
def test_conflicts_and_server_errors_during_wait_fail_immediately_without_sleeping(
    publication, monkeypatch, conflict
):
    import urllib.error

    directory, manifest, _, calls, item = publication
    wheel = "hitlist-1.2.3-py3-none-any.whl"
    poll_count = 0

    def post_upload_conflict(version):
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return {}
        if conflict == "http_503":
            raise urllib.error.HTTPError("https://pypi.org", 503, "unavailable", {}, None)
        entry = item(wheel)
        if conflict == "yanked_partial":
            entry["yanked"] = True
        else:
            entry["digests"]["sha256"] = "f" * 64
        return {wheel: entry}

    monkeypatch.setattr(release_artifacts, "public_files", post_upload_conflict)
    slept = []
    expected_exc = urllib.error.HTTPError if conflict == "http_503" else ValueError
    with pytest.raises(expected_exc):
        release_artifacts.publish_distributions(
            directory,
            manifest,
            timeout_seconds=60.0,
            poll_interval_seconds=5.0,
            sleep_fn=slept.append,
        )
    assert slept == []
    assert [command[:2] for command in calls] == [["twine", "check"], ["twine", "upload"]]
    assert not (directory.parent / "pypi-1.2.3-verified.json").exists()


def test_simple_files_and_installer_dry_run_enforce_version_isolation_and_integrity(monkeypatch):
    import io
    import urllib.error

    wheel = "hitlist-1.2.3-py3-none-any.whl"
    sdist = "hitlist-1.2.3.tar.gz"
    manifest = {
        "version": "1.2.3",
        "files": {
            wheel: {"sha256": "a" * 64, "size_bytes": 100},
            sdist: {"sha256": "b" * 64, "size_bytes": 200},
        },
    }

    class FakeSimpleResponse(io.StringIO):
        headers = {"Content-Type": "application/vnd.pypi.simple.v1+json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    seen_requests = []

    def fake_urlopen(req, timeout=30):
        seen_requests.append(req)
        payload = {
            "files": [
                # Neighbouring versions (including prefix-overlapping 1.2.30) must be ignored.
                {
                    "filename": "hitlist-1.2.30-py3-none-any.whl",
                    "hashes": {"sha256": "c" * 64},
                    "size": 999,
                    "url": "https://files.pythonhosted.org/hitlist-1.2.30-py3-none-any.whl",
                    "yanked": False,
                },
                {
                    "filename": wheel,
                    "hashes": {"sha256": "a" * 64},
                    "size": 100,
                    "url": f"https://files.pythonhosted.org/{wheel}",
                    "yanked": False,
                },
                {
                    "filename": sdist,
                    "hashes": {"sha256": "b" * 64},
                    "size": 200,
                    "url": f"https://files.pythonhosted.org/{sdist}",
                    "yanked": False,
                },
            ]
        }
        return FakeSimpleResponse(json.dumps(payload))

    monkeypatch.setattr(release_artifacts.urllib.request, "urlopen", fake_urlopen)
    verified_simple = release_artifacts.verify_simple_files(manifest)
    assert set(verified_simple) == {wheel, sdist}
    assert seen_requests[0].headers["Accept"] == "application/vnd.pypi.simple.v1+json"
    assert seen_requests[0].headers["Cache-control"] == "no-cache"

    # HTML fallback must be rejected rather than silently parsed.
    class HTMLResponse(FakeSimpleResponse):
        headers = {"Content-Type": "text/html"}

    monkeypatch.setattr(
        release_artifacts.urllib.request, "urlopen", lambda *a, **k: HTMLResponse("<html></html>")
    )
    with pytest.raises(ValueError, match="Unexpected Simple index content type"):
        release_artifacts.simple_files("1.2.3")

    # HTTP 404 -> {}, HTTP 429 -> HTTPError.
    monkeypatch.setattr(
        release_artifacts.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            urllib.error.HTTPError("https://pypi.org/simple/hitlist/", 404, "missing", {}, None)
        ),
    )
    assert release_artifacts.simple_files("1.2.3") == {}

    # Real run_installer_dry_run command shape & error classification:
    seen_commands = []

    def fake_pip_missing(command, **kwargs):
        seen_commands.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="ERROR: No matching distribution found for hitlist==1.2.3",
        )

    monkeypatch.setattr(release_artifacts.subprocess, "run", fake_pip_missing)
    with pytest.raises(release_artifacts.IncompleteInventoryError):
        release_artifacts.run_installer_dry_run("1.2.3")
    cmd = seen_commands[0]
    for required_flag in ("-I", "--dry-run", "--ignore-installed", "--no-deps", "--no-cache-dir"):
        assert required_flag in cmd
    assert cmd[-1] == "hitlist==1.2.3"

    # HTTP 403 inside pip output must NOT be misclassified as IncompleteInventoryError.
    monkeypatch.setattr(
        release_artifacts.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="WARNING: HTTP error 403\nERROR: No matching distribution found for hitlist==1.2.3",
        ),
    )
    with pytest.raises(ValueError) as exc_info:
        release_artifacts.run_installer_dry_run("1.2.3")
    assert not isinstance(exc_info.value, release_artifacts.IncompleteInventoryError)

    # Hash mismatch in pip report must fail closed.
    monkeypatch.setattr(
        release_artifacts.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "version": "1",
                    "pip_version": "26.3",
                    "install": [
                        {
                            "is_yanked": False,
                            "metadata": {"name": "hitlist", "version": "1.2.3"},
                            "download_info": {
                                "url": f"https://files.pythonhosted.org/{wheel}",
                                "archive_info": {"hashes": {"sha256": "0" * 64}},
                            },
                        }
                    ],
                }
            ),
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="conflicting sha256"):
        release_artifacts.verify_installer_resolution(manifest)
