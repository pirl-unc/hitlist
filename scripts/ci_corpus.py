"""Export and verify public CI corpus provenance without private build paths (#540)."""

import argparse
import hashlib
import json
from pathlib import Path

METADATA = "observations_meta.json"
PARQUETS = (
    "observations.parquet",
    "peptide_mappings.parquet",
    "binding.parquet",
    "bulk_proteomics.parquet",
    "line_expression.parquet",
)
BUILD_FINGERPRINTS = ("observations", "binding", "bulk_proteomics", "line_expression")
MIN_ARTIFACT_VERSION = 5  # #444's build-time exclusion of non-MS studies.


def check_artifact_version(metadata):
    version = metadata.get("artifact_version")
    if type(version) is not int or version < MIN_ARTIFACT_VERSION:
        raise ValueError("Corpus must record its original artifact_version >= 5")


def file_digest(path):
    before = path.stat()
    if not before.st_size:
        raise ValueError(f"Empty corpus file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"Corpus file changed while hashing: {path.name}")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def build_fingerprints(directory):
    result = {}
    for label in BUILD_FINGERPRINTS:
        stat = (directory / f"{label}.parquet").stat()
        result[label] = {"size": stat.st_size, "mtime": stat.st_mtime}
    return result


def write_manifest(directory, output):
    original = directory / METADATA
    if output.resolve() == original.resolve():
        raise ValueError("Public manifest must not replace original build metadata")
    metadata = json.loads(original.read_text())
    check_artifact_version(metadata)
    if metadata.get("parquets") != build_fingerprints(directory):
        raise ValueError("Build metadata does not match the current parquet fingerprints")
    files = {name: file_digest(directory / name) for name in PARQUETS}
    if metadata["parquets"] != build_fingerprints(directory):
        raise ValueError("Corpus changed since its build metadata was checked")
    manifest = {
        "ci_corpus_manifest_version": 1,
        "artifact_version": metadata["artifact_version"],
        "mhcgnomes_version": metadata["mhcgnomes_version"],
        "files": files,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_manifest(directory):
    manifest = json.loads((directory / METADATA).read_text())
    if manifest.get("ci_corpus_manifest_version") != 1:
        raise ValueError("Missing or unsupported public corpus manifest version")
    check_artifact_version(manifest)
    if set(manifest.get("files", {})) != set(PARQUETS):
        raise ValueError("Public manifest must name exactly the five required parquets")
    for name in PARQUETS:
        if manifest["files"][name] != file_digest(directory / name):
            raise ValueError(f"Corpus checksum or size mismatch: {name}")
    print(f"Verified five corpus files; original artifact contract {manifest['artifact_version']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "write":
        if args.output is None:
            parser.error("write requires --output outside the original metadata")
        write_manifest(args.directory, args.output)
    else:
        verify_manifest(args.directory)
