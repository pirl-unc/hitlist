"""Record tested release artifacts and verify them on clean main before upload (#538)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
GITHUB_API_TIMEOUT_SECONDS = 30
ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS = 120
INSTALLER_TIMEOUT_SECONDS = 60
PUBLICATION_VISIBILITY_TIMEOUT_SECONDS = float(
    os.environ.get("HITLIST_PYPI_VISIBILITY_TIMEOUT_SECONDS", "300")
)
PUBLICATION_POLL_INTERVAL_SECONDS = float(os.environ.get("HITLIST_PYPI_POLL_INTERVAL_SECONDS", "5"))
SIMPLE_INDEX_URL = "https://pypi.org/simple/hitlist/"
SIMPLE_INDEX_ROOT_URL = "https://pypi.org/simple/"
CORPUS_FILES = (
    "observations.parquet",
    "peptide_mappings.parquet",
    "binding.parquet",
    "bulk_proteomics.parquet",
    "line_expression.parquet",
    "observations_meta.json",
)
DEVELOPMENT_DEPENDENCIES = (
    "datacache",
    "gtfparse",
    "serializable",
    "mhcflurry",
    "pyensembl",
    "mhcgnomes",
)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def source_state(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Release source has tracked changes")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "version": runpy.run_path(str(root / "hitlist" / "version.py"))["__version__"],
    }


def distribution_files(directory, version):
    expected = {f"hitlist-{version}-py3-none-any.whl", f"hitlist-{version}.tar.gz"}
    actual = {path.name for pattern in ("*.whl", "*.tar.gz") for path in directory.glob(pattern)}
    if actual != expected:
        raise ValueError(f"Expected distributions {sorted(expected)}, found {sorted(actual)}")
    return {name: file_digest(directory / name) for name in sorted(expected)}


def installed_dependencies():
    """Require actual Git installations, including when PyPI has the same version."""
    dependencies = {}
    for distribution in importlib.metadata.distributions():
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        details = {"version": distribution.version}
        revision = direct.get("vcs_info", {}).get("commit_id")
        if revision:
            details["commit"] = revision
        dependencies[distribution.metadata["Name"]] = details
    missing = [
        name for name in DEVELOPMENT_DEPENDENCIES if not dependencies.get(name, {}).get("commit")
    ]
    if missing:
        raise ValueError(f"Development dependencies lack installed Git revisions: {missing}")
    return dependencies


def write_manifest(directory, corpus_dir, root=ROOT):
    source = source_state(root)
    corpus_tools = runpy.run_path(str(ROOT / "scripts" / "ci_corpus.py"))
    corpus_manifest = corpus_tools["verify_manifest"](corpus_dir)
    names = (*corpus_tools["public_files"](corpus_manifest), "observations_meta.json")
    corpus = {name: file_digest(corpus_dir / name) for name in names}
    manifest = {
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "source_commit": source["commit"],
        "version": source["version"],
        "files": distribution_files(directory, source["version"]),
        "corpus": corpus,
        "corpus_manifest": corpus_manifest,
        "dependencies": installed_dependencies(),
    }
    (directory / "release.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_manifest(directory, root=ROOT):
    source = source_state(root)
    if source["branch"] != "main":
        raise ValueError("Publication requires a clean main checkout")
    manifest = json.loads((directory / "release.json").read_text())
    if manifest["source_commit"] != source["commit"] or manifest["version"] != source["version"]:
        raise ValueError("Artifacts do not match the current main commit and version")
    if manifest["files"] != distribution_files(directory, source["version"]):
        raise ValueError("Distribution checksum or size differs from the tested artifact")
    if "corpus_manifest" in manifest:
        corpus_tools = runpy.run_path(str(ROOT / "scripts" / "ci_corpus.py"))
        public = manifest["corpus_manifest"]
        names = (*corpus_tools["public_files"](public), "observations_meta.json")
        if any(manifest["corpus"].get(name) != digest for name, digest in public["files"].items()):
            raise ValueError("Release corpus hashes disagree with the public corpus manifest")
    else:
        names = CORPUS_FILES  # Existing release receipts retain their original v1 inventory.
    if set(manifest["corpus"]) != set(names):
        raise ValueError("Release provenance is missing required corpus files")
    run_id = str(manifest["workflow_run_id"])
    if not run_id.isdigit():
        raise ValueError("Release provenance has no GitHub workflow run")

    def github(path):
        try:
            response = subprocess.check_output(
                ["gh", "api", f"repos/pirl-unc/hitlist/actions/{path}"],
                text=True,
                timeout=GITHUB_API_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("Timed out querying GitHub release provenance") from error
        return json.loads(response)

    workflow = github("workflows/release-build.yml")
    run = github(f"runs/{run_id}")
    expected = {
        "workflow_id": workflow["id"],
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": source["commit"],
        "status": "completed",
        "conclusion": "success",
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Artifacts require a successful manual release build on current main")
    artifact_name = f"hitlist-release-{source['commit']}"

    def matching_artifact():
        artifacts = github(f"runs/{run_id}/artifacts")["artifacts"]
        matches = [
            artifact
            for artifact in artifacts
            if artifact["name"] == artifact_name and not artifact["expired"]
        ]
        if len(matches) != 1:
            raise ValueError("Expected one retained release artifact on the verified workflow run")
        return matches[0]

    artifact_id = matching_artifact()["id"]
    # Bind the supplied manifest to the actual Actions artifact, rather than
    # trusting a locally editable JSON file's claim about a successful run.
    with tempfile.TemporaryDirectory(prefix="hitlist-release-") as temporary:
        download_dir = Path(temporary)
        try:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    run_id,
                    "--name",
                    artifact_name,
                    "--repo",
                    "pirl-unc/hitlist",
                    "--dir",
                    str(download_dir),
                ],
                check=True,
                timeout=ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("Timed out downloading the verified GitHub artifact") from error
        except subprocess.CalledProcessError as error:
            raise ValueError("Verified GitHub artifact download failed") from error
        if matching_artifact()["id"] != artifact_id:
            raise ValueError("Release artifact identity changed during download")
        expected_files = {"release.json", *manifest["files"]}
        downloaded_files = list(download_dir.iterdir())
        if {path.name for path in downloaded_files} != expected_files or any(
            not path.is_file() or path.is_symlink() for path in downloaded_files
        ):
            raise ValueError("Verified GitHub artifact has missing or unexpected files")
        try:
            trusted_manifest = json.loads((download_dir / "release.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Verified GitHub artifact has an unreadable release manifest"
            ) from error
        if manifest != trusted_manifest:
            raise ValueError("Release manifest differs from the verified GitHub artifact")
        if trusted_manifest["files"] != distribution_files(download_dir, source["version"]):
            raise ValueError("Verified GitHub artifact distributions disagree with its manifest")
    print(f"Both distributions match clean main {source['commit']} ({source['version']}).")
    return manifest


class IncompleteInventoryError(ValueError):
    """Expected release files have not yet propagated to a public PyPI endpoint."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _no_cache_request(url: str, *, accept: str | None = None) -> urllib.request.Request:
    headers = {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def _check_download_host(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org":
        raise ValueError("Unexpected public distribution download host")
    return parsed


def public_files(version):
    """Only a genuine PyPI 404 means the release has not been published."""
    url = f"https://pypi.org/pypi/hitlist/{urllib.parse.quote(version, safe='')}/json"
    try:
        with urllib.request.urlopen(_no_cache_request(url), timeout=30) as response:
            metadata = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise
    files = metadata["urls"]
    if len({item["filename"] for item in files}) != len(files):
        raise ValueError("PyPI returned duplicate distribution names")
    return {item["filename"]: item for item in files}


def _matches_release_version(filename: str, version: str) -> bool:
    target = Version(version)
    try:
        if filename.endswith(".whl"):
            name, parsed_version, _, _ = parse_wheel_filename(filename)
            return canonicalize_name(name) == "hitlist" and parsed_version == target
        if filename.endswith((".tar.gz", ".zip")):
            name, parsed_version = parse_sdist_filename(filename)
            return canonicalize_name(name) == "hitlist" and parsed_version == target
    except (InvalidWheelFilename, InvalidSdistFilename):
        pass
    return filename.startswith(f"hitlist-{version}-") or filename.startswith(f"hitlist-{version}.")


def simple_files(version):
    """Read PEP 691 Simple JSON entries for ``version`` without local caching."""
    request = _no_cache_request(SIMPLE_INDEX_URL, accept="application/vnd.pypi.simple.v1+json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = getattr(response, "headers", None)
            if headers is not None:
                content_type = headers.get("Content-Type", "")
                if content_type and not content_type.startswith(
                    "application/vnd.pypi.simple.v1+json"
                ):
                    raise ValueError(f"Unexpected Simple index content type: {content_type}")
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise ValueError("Malformed PyPI Simple index response")
    matched = [
        item
        for item in entries
        if isinstance(item, dict)
        and isinstance(item.get("filename"), str)
        and _matches_release_version(item["filename"], version)
    ]
    if len({item["filename"] for item in matched}) != len(matched):
        raise ValueError("PyPI Simple index returned duplicate distribution names")
    return {item["filename"]: item for item in matched}


def verify_public_files(manifest, *, require_complete=True, verified_downloads=None):
    """Check the public inventory, metadata and actual downloaded bytes.

    A matching partial upload may be resumed, but never counts as a release.
    Every visible file is validated immediately so unexpected names, yanked
    entries, digest/size conflicts, invalid hosts or corrupt bytes fail closed
    without waiting for remaining files (#867).
    """
    expected = manifest["files"]
    public = public_files(manifest["version"])
    if set(public) - set(expected):
        raise ValueError("Public PyPI distribution inventory differs from the tested pair")
    for name, item in public.items():
        pin = expected[name]
        if (
            bool(item.get("yanked"))
            or {"sha256": item["digests"]["sha256"], "size_bytes": item["size"]} != pin
        ):
            raise ValueError(f"PyPI already contains conflicting or yanked bytes: {name}")
        _check_download_host(item["url"])
        cache_key = (name, item["url"])
        if verified_downloads is not None and cache_key in verified_downloads:
            continue
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(item["url"], timeout=30) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > pin["size_bytes"]:
                    raise ValueError(f"Public download exceeds the tested file size: {name}")
                digest.update(block)
        if {"sha256": digest.hexdigest(), "size_bytes": size} != pin:
            raise ValueError(f"Public downloaded bytes differ from the tested file: {name}")
        if verified_downloads is not None:
            verified_downloads.add(cache_key)
    if require_complete and set(public) != set(expected):
        missing = sorted(set(expected) - set(public))
        raise IncompleteInventoryError(
            f"Public PyPI distribution inventory differs from the tested pair (missing: {missing})"
        )
    return public


def verify_simple_files(manifest, *, require_complete=True):
    """Check PEP 691 Simple-index entries for exact hashes, sizes and non-yanked state."""
    expected = manifest["files"]
    simple = simple_files(manifest["version"])
    if set(simple) - set(expected):
        raise ValueError("Simple PyPI distribution inventory differs from the tested pair")
    for name, item in simple.items():
        pin = expected[name]
        sha256 = (item.get("hashes") or {}).get("sha256")
        size = item.get("size")
        if (
            bool(item.get("yanked"))
            or sha256 != pin["sha256"]
            or (size is not None and size != pin["size_bytes"])
        ):
            raise ValueError(f"Simple PyPI index contains conflicting or yanked metadata: {name}")
        _check_download_host(str(item.get("url", "")))
    if require_complete and set(simple) != set(expected):
        missing = sorted(set(expected) - set(simple))
        raise IncompleteInventoryError(
            f"Simple PyPI distribution inventory incomplete for {manifest['version']} (missing: {missing})"
        )
    return simple


def run_installer_dry_run(version: str) -> dict:
    """Resolve ``hitlist==version`` uncached via isolated ``pip install --dry-run``."""
    command = [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--no-deps",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--index-url",
        SIMPLE_INDEX_ROOT_URL,
        "--quiet",
        "--report",
        "-",
        f"hitlist=={version}",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=INSTALLER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("Timed out running uncached public installer resolution") from error
    if completed.returncode != 0:
        detail = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip()
        )
        fatal_markers = (
            "HTTP error 401",
            "HTTP error 403",
            "HTTP error 429",
            "HTTP error 5",
            "SSLError",
            "ReadTimeoutError",
            "ConnectTimeoutError",
        )
        if not any(marker in detail for marker in fatal_markers) and (
            f"No matching distribution found for hitlist=={version}" in detail
            or f"Could not find a version that satisfies the requirement hitlist=={version}"
            in detail
        ):
            raise IncompleteInventoryError(
                f"Uncached public installer cannot yet resolve hitlist=={version}"
            )
        raise ValueError(f"Uncached public installer resolution failed: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Uncached public installer returned unreadable report JSON") from error


def verify_installer_resolution(manifest) -> dict:
    """Verify ordinary uncached public-index installer resolution against tested hashes."""
    expected = manifest["files"]
    report = run_installer_dry_run(manifest["version"])
    installs = report.get("install")
    if not isinstance(installs, list) or len(installs) != 1 or not isinstance(installs[0], dict):
        raise ValueError("Installer dry-run did not resolve exactly one distribution")
    entry = installs[0]
    metadata = entry.get("metadata") or {}
    if (
        canonicalize_name(str(metadata.get("name", ""))) != "hitlist"
        or metadata.get("version") != manifest["version"]
        or bool(entry.get("is_yanked"))
    ):
        raise ValueError("Installer dry-run resolved unexpected or yanked package metadata")
    download_info = entry.get("download_info") or {}
    url = str(download_info.get("url", ""))
    parsed = _check_download_host(url)
    filename = urllib.parse.unquote(Path(parsed.path).name)
    if filename not in expected:
        raise ValueError(f"Installer dry-run resolved unexpected filename: {filename}")
    sha256 = ((download_info.get("archive_info") or {}).get("hashes") or {}).get("sha256")
    if sha256 != expected[filename]["sha256"]:
        raise ValueError(f"Installer dry-run resolved conflicting sha256 for {filename}")
    return {
        "index_url": SIMPLE_INDEX_ROOT_URL,
        "python_version": platform.python_version(),
        "pip_version": str(report.get("pip_version", "")),
        "resolved_filename": filename,
    }


def wait_for_verified_public_release(
    manifest,
    *,
    timeout_seconds=None,
    poll_interval_seconds=None,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
) -> dict:
    """Wait within a bounded deadline for Project JSON, Simple JSON and installer visibility."""
    timeout = (
        PUBLICATION_VISIBILITY_TIMEOUT_SECONDS
        if timeout_seconds is None
        else float(timeout_seconds)
    )
    interval = (
        PUBLICATION_POLL_INTERVAL_SECONDS
        if poll_interval_seconds is None
        else float(poll_interval_seconds)
    )
    if timeout < 0 or interval < 0:
        raise ValueError("Visibility timeout and poll interval must be non-negative")
    started = monotonic_fn()
    deadline = started + timeout
    verified_downloads = set()
    project_verified_at = None
    simple_verified_at = None
    installer_details = None
    project_url = (
        f"https://pypi.org/pypi/hitlist/{urllib.parse.quote(manifest['version'], safe='')}/json"
    )
    while True:
        stage = "project_json"
        try:
            if project_verified_at is None:
                verify_public_files(
                    manifest, require_complete=True, verified_downloads=verified_downloads
                )
                project_verified_at = _utc_now()
            stage = "simple_json"
            if simple_verified_at is None:
                verify_simple_files(manifest, require_complete=True)
                simple_verified_at = _utc_now()
            stage = "installer_dry_run"
            if installer_details is None:
                installer_details = {
                    **verify_installer_resolution(manifest),
                    "verified_at_utc": _utc_now(),
                }
            break
        except IncompleteInventoryError as pending_error:
            now = monotonic_fn()
            remaining = deadline - now
            if remaining <= 0:
                elapsed = now - started
                raise ValueError(
                    f"Timed out after {elapsed:.1f}s waiting for public PyPI visibility "
                    f"at stage {stage}: {pending_error}"
                ) from pending_error
            wait_seconds = min(interval, remaining) if interval > 0 else 0.0
            print(
                f"Waiting {wait_seconds:.1f}s for PyPI {stage} visibility "
                f"({pending_error}; {remaining:.1f}s remaining)...",
                file=sys.stderr,
                flush=True,
            )
            sleep_fn(wait_seconds)
    return {
        "project_json": {
            "endpoint": project_url,
            "verified_at_utc": project_verified_at,
        },
        "simple_json": {
            "endpoint": SIMPLE_INDEX_URL,
            "verified_at_utc": simple_verified_at,
        },
        "installer_dry_run": installer_details,
        "scope": (
            "Observed visibility from this publisher endpoint only; "
            "not a global CDN propagation guarantee."
        ),
    }


def publish_distributions(
    directory,
    manifest,
    *,
    timeout_seconds=None,
    poll_interval_seconds=None,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
):
    """Upload an immutable copy of a tested pair, then verify public downloads and index visibility."""
    # Snapshot exactly the bytes whose digests were validated. Never use a glob
    # or --skip-existing: either can hide a wrong version or a PyPI collision.
    with tempfile.TemporaryDirectory(prefix="hitlist-publish-") as temporary:
        snapshot = Path(temporary)
        for name in manifest["files"]:
            if Path(name).name != name:
                raise ValueError("Distribution name must be a basename")
            shutil.copyfile(directory / name, snapshot / name)
        if distribution_files(snapshot, manifest["version"]) != manifest["files"]:
            raise ValueError("Distribution changed after release verification")
        present = verify_public_files(manifest, require_complete=False)
        pending = [
            str(snapshot / name) for name in sorted(manifest["files"]) if name not in present
        ]
        if pending:
            subprocess.run(["twine", "check", *pending], check=True)
            subprocess.run(["twine", "upload", "--non-interactive", *pending], check=True)
        visibility = wait_for_verified_public_release(
            manifest,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
    receipt = {
        "version": manifest["version"],
        "source_commit": manifest["source_commit"],
        "workflow_run_id": manifest.get("workflow_run_id"),
        "public_files": manifest["files"],
        "visibility": visibility,
    }
    path = directory.parent / f"pypi-{manifest['version']}-verified.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Verified both public PyPI downloads: {path}")


def local_manifest(directory, root=ROOT):
    """Bind deploy.sh's freshly tested local build to its clean-main source."""
    source = source_state(root)
    if source["branch"] != "main":
        raise ValueError("Publication requires a clean main checkout")
    return {
        "source_commit": source["commit"],
        "version": source["version"],
        "files": distribution_files(directory, source["version"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("write", "verify", "publish", "publish-local", "verify-public")
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--corpus-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "write":
        if args.corpus_dir is None:
            parser.error("write requires --corpus-dir")
        write_manifest(args.directory, args.corpus_dir)
    elif args.mode == "verify":
        verify_manifest(args.directory)
    elif args.mode == "publish":
        publish_distributions(args.directory, verify_manifest(args.directory))
    elif args.mode == "publish-local":
        publish_distributions(args.directory, local_manifest(args.directory))
    else:
        wait_for_verified_public_release(verify_manifest(args.directory), timeout_seconds=0)
