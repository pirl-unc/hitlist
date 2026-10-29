"""Record tested release artifacts and verify them on clean main before upload (#538)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB_API_TIMEOUT_SECONDS = 30
ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS = 120
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


def public_files(version):
    """Only a genuine PyPI 404 means the release has not been published."""
    url = f"https://pypi.org/pypi/hitlist/{urllib.parse.quote(version, safe='')}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise
    files = metadata["urls"]
    if len({item["filename"] for item in files}) != len(files):
        raise ValueError("PyPI returned duplicate distribution names")
    return {item["filename"]: item for item in files}


def verify_public_files(manifest, *, require_complete=True):
    """Check the public inventory, metadata and actual downloaded bytes.

    A matching partial upload may be resumed, but never counts as a release.
    Conflicting or unexpected files fail before any additional upload.
    """
    expected = manifest["files"]
    public = public_files(manifest["version"])
    if set(public) - set(expected) or (require_complete and set(public) != set(expected)):
        raise ValueError("Public PyPI distribution inventory differs from the tested pair")
    for name, item in public.items():
        pin = expected[name]
        if (
            item.get("yanked")
            or {"sha256": item["digests"]["sha256"], "size_bytes": item["size"]} != pin
        ):
            raise ValueError(f"PyPI already contains conflicting or yanked bytes: {name}")
        parsed = urllib.parse.urlparse(item["url"])
        if parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org":
            raise ValueError("Unexpected public distribution download host")
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
    return public


def publish_distributions(directory, manifest):
    """Upload an immutable copy of a tested pair, then verify public downloads."""
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
        verify_public_files(manifest)
    receipt = {
        "version": manifest["version"],
        "source_commit": manifest["source_commit"],
        "workflow_run_id": manifest.get("workflow_run_id"),
        "public_files": manifest["files"],
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
        verify_public_files(verify_manifest(args.directory))
