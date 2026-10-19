"""Export and verify public CI corpus provenance without private build paths (#540)."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urljoin

METADATA = "observations_meta.json"
PARQUETS = (
    "observations.parquet",
    "peptide_mappings.parquet",
    "binding.parquet",
    "bulk_proteomics.parquet",
    "line_expression.parquet",
)
BUILD_FINGERPRINTS = ("observations", "binding", "bulk_proteomics", "line_expression")
CURRENT_PARQUETS = (*PARQUETS, "other_assays.parquet", "observation_contributors.parquet")
CURRENT_FILES = (*CURRENT_PARQUETS, "peptide_mappings_meta.json", "README.md")
CURRENT_FINGERPRINTS = (*BUILD_FINGERPRINTS, "other_assays", "contributors")
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


def build_fingerprints(directory, labels=BUILD_FINGERPRINTS):
    result = {}
    for label in labels:
        filename = (
            "observation_contributors.parquet" if label == "contributors" else f"{label}.parquet"
        )
        stat = (directory / filename).stat()
        result[label] = {"size": stat.st_size, "mtime": stat.st_mtime}
    return result


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _bytes_digest(value):
    return {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}


def _current_provenance(metadata, files):
    provenance = metadata.get("provenance", {})
    if provenance.get("schema_version") != 1:
        raise ValueError("Current corpus requires supported contributor provenance")
    for label in ("observations", "binding", "other_assays"):
        if provenance.get("indexes", {}).get(label) != files[f"{label}.parquet"]:
            raise ValueError(f"Provenance content hash mismatch: {label}")
    expected = {
        "file": "observation_contributors.parquet",
        **files["observation_contributors.parquet"],
    }
    if provenance.get("contributors") != expected:
        raise ValueError("Provenance content hash mismatch: contributors")
    sources = {}
    for name, source in provenance.get("sources", {}).items():
        digest = source.get("sha256", "")
        size = source.get("size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError(f"Source snapshot requires a content identity: {name}")
        sources[name] = {
            key: source[key]
            for key in ("sha256", "size_bytes", "locator_basis", "upstream_record_locator_status")
            if key in source
        }
    if not sources:
        raise ValueError("Current corpus requires captured source snapshots")
    return {
        "schema_version": 1,
        "sources": sources,
        "indexes": {
            name: files[f"{name}.parquet"] for name in ("observations", "binding", "other_assays")
        },
        "contributors": expected,
        "n_contributor_links": provenance["n_contributor_links"],
    }


def _portable_mapping_metadata(directory, files):
    original = json.loads((directory / "peptide_mappings_meta.json").read_text())
    fingerprints = {}
    for label in ("observations", "binding", "other_assays"):
        path = directory / f"{label}.parquet"
        stat = path.stat()
        expected = original.get("observations", {}).get(label, {})
        if expected.get("size") != stat.st_size or expected.get("mtime") != stat.st_mtime:
            raise ValueError(f"Mappings do not match their source index: {label}")
        fingerprints[label] = files[path.name]
    from hitlist.mappings import _mapping_artifact_contract

    contract = original.get("contract", {})
    required = {"release", "use_uniprot", "fetch_missing", "flank"}
    if not required <= set(contract) or contract != _mapping_artifact_contract(
        **{key: contract[key] for key in required}
    ):
        raise ValueError("Current corpus mapping settings do not match the supported contract")
    return {
        "ci_mapping_manifest_version": 1,
        "contract": contract,
        "observation_hashes": fingerprints,
        "original_metadata_sha256": file_digest(directory / "peptide_mappings_meta.json")["sha256"],
        **{
            key: original[key]
            for key in (
                "unavailable_proteomes",
                "n_rows",
                "n_peptides",
                "n_proteomes",
                "per_proteome",
                "unmapped_organisms",
                "built_at",
            )
            if key in original
        },
    }


def _definition(directory):
    original = directory / METADATA
    metadata = json.loads(original.read_text())
    check_artifact_version(metadata)
    current = metadata["artifact_version"] >= 9
    labels = CURRENT_FINGERPRINTS if current else BUILD_FINGERPRINTS
    if metadata.get("parquets") != build_fingerprints(directory, labels):
        raise ValueError("Build metadata does not match the current parquet fingerprints")
    files = {
        name: file_digest(directory / name) for name in (CURRENT_PARQUETS if current else PARQUETS)
    }
    receipt_path = directory / ".datacache-manifest.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        expected = {*CURRENT_PARQUETS, METADATA, "peptide_mappings_meta.json", "README.md"}
        if receipt.get("format") != 1 or set(receipt.get("outputs", {})) != expected:
            raise ValueError("Datacache generation has an incomplete publication inventory")
        for name, record in receipt["outputs"].items():
            actual = files.get(name) or file_digest(directory / name)
            if actual != {"sha256": record.get("sha256"), "size_bytes": record.get("size")}:
                raise ValueError(f"Datacache generation content hash mismatch: {name}")
    extras = {}
    provenance = None
    if current:
        provenance = _current_provenance(metadata, files)
        extras["peptide_mappings_meta.json"] = _json_bytes(
            _portable_mapping_metadata(directory, files)
        )
        extras["README.md"] = (
            b"# Hitlist portable CI corpus\n\n"
            b"This is a content-verified snapshot of one dataset generation. "
            b"The original artifact contract and source hashes are retained. "
            b"It is a software regression corpus, not a qualified training split.\n\n"
            b"| File | Meaning |\n| --- | --- |\n"
            b"| observations.parquet | Positive MHC-MS evidence |\n"
            b"| binding.parquet | Binding endpoints, units and censoring |\n"
            b"| other_assays.parquet | Other, negative and unknown assay evidence |\n"
            b"| observation_contributors.parquet | Every retained source contributor and reference |\n"
            b"| peptide_mappings.parquet | Alternative protein occurrences, not independent observations |\n"
            b"| peptide_mappings_meta.json | Mapping contract, source index hashes and reference limitations |\n"
            b"| bulk_proteomics.parquet | Bulk observations; fraction counts do not establish detectability |\n"
            b"| line_expression.parquet | Cell-line expression evidence |\n"
            b"| observations_meta.json | Original build contract, source hashes and complete file inventory |\n\n"
            b"Cell types, conditions, MHC typing and assay restrictions retain their distinct roles. "
            b"Private source paths are omitted. Mapping input hashes replace machine-specific "
            b"mtime/path comparisons in the portable receipt; this does not make the downloaded "
            b"snapshot a fresh local build. Actual cohort coverage, partitions, biological "
            b"independence and training performance require separate qualification.\n"
        )
        files.update({name: _bytes_digest(value) for name, value in extras.items()})
    if metadata["parquets"] != build_fingerprints(directory, labels):
        raise ValueError("Corpus changed since its build metadata was checked")
    manifest = {
        "ci_corpus_manifest_version": 2 if current else 1,
        "artifact_version": metadata["artifact_version"],
        "mhcgnomes_version": metadata["mhcgnomes_version"],
        "files": files,
    }
    if current:
        manifest["provenance"] = provenance
        manifest["counts"] = {
            k: v for k, v in metadata.items() if k.startswith("n_") and type(v) is int
        }
        manifest["curation_inputs"] = {
            name: {key: value[key] for key in ("size", "sha256") if key in value}
            for name, value in metadata.get("sources", {}).items()
            if name not in provenance["sources"]
        }
    return manifest, extras


def write_manifest(directory, output):
    """Low-level export; the public publisher uses snapshot_corpus atomically."""
    from hitlist.dataset import dataset_snapshot

    with dataset_snapshot(root=directory) as pinned:
        if output.resolve() == (pinned / METADATA).resolve():
            raise ValueError("Public manifest must not replace original build metadata")
        manifest, extras = _definition(pinned)
    for name, content in extras.items():
        with (output.parent / name).open("xb") as stream:
            stream.write(content)
    with output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def public_files(manifest):
    """Validate the versioned closed inventory before reading any named path."""
    version = manifest.get("ci_corpus_manifest_version")
    if version not in (1, 2):
        raise ValueError("Missing or unsupported public corpus manifest version")
    check_artifact_version(manifest)
    expected = CURRENT_FILES if version == 2 else PARQUETS
    if (version == 2) != (manifest["artifact_version"] >= 9):
        raise ValueError("Public manifest version does not preserve the original artifact contract")
    if set(manifest.get("files", {})) != set(expected):
        raise ValueError("Public manifest must name exactly its required corpus files")
    return expected


def verify_manifest(directory):
    manifest = json.loads((directory / METADATA).read_text())
    expected = public_files(manifest)
    for name in expected:
        if manifest["files"][name] != file_digest(directory / name):
            raise ValueError(f"Corpus checksum or size mismatch: {name}")
    if manifest["ci_corpus_manifest_version"] == 2:
        if manifest["provenance"] != _current_provenance(manifest, manifest["files"]):
            raise ValueError("Public contributor provenance contains unsupported private fields")
        mappings = json.loads((directory / "peptide_mappings_meta.json").read_text())
        if mappings.get("ci_mapping_manifest_version") != 1 or mappings.get(
            "observation_hashes"
        ) != {
            label: manifest["files"][f"{label}.parquet"]
            for label in ("observations", "binding", "other_assays")
        }:
            raise ValueError("Portable mapping metadata disagrees with source index hashes")
    return manifest


def snapshot_corpus(directory, *, cache_dir=None):
    """Pin a generation and publish a complete datacache-managed portable copy."""
    from datacache import materialize

    from hitlist.dataset import dataset_snapshot
    from hitlist.downloads import data_dir

    with dataset_snapshot(root=directory) as pinned:
        manifest, extras = _definition(pinned)
        manifest_bytes = _json_bytes(manifest)
        source_names = [name for name in manifest["files"] if name not in extras]
        source_names += [METADATA]
        if extras:
            source_names += ["peptide_mappings_meta.json"]
        sources = {
            name: {
                "path": str(pinned / name),
                "sha256": (value := file_digest(pinned / name))["sha256"],
                "size": value["size_bytes"],
            }
            for name in source_names
        }
        options = {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "recipe_sha256": file_digest(Path(__file__))["sha256"],
        }

        def build(inputs, outputs):
            for name in manifest["files"]:
                if name in extras:
                    Path(outputs[name]).write_bytes(extras[name])
                else:
                    shutil.copyfile(inputs[name], outputs[name])
            Path(outputs[METADATA]).write_bytes(manifest_bytes)
            # A changed legacy source must not authorize a mixed portable copy.
            refreshed, _ = _definition(pinned)
            if refreshed != manifest:
                raise ValueError("Source corpus changed during portable publication")
            verify_manifest(Path(outputs[METADATA]).parent)

        root = Path(cache_dir) if cache_dir is not None else data_dir() / "ci-corpora"
        paths = materialize(
            root / hashlib.sha256(_json_bytes(options)).hexdigest(),
            sources,
            transform={"version": "hitlist-portable-ci-v2", "options": options},
            outputs={name: {} for name in (*manifest["files"], METADATA)},
            builder=build,
        )
    result = Path(paths[METADATA]).parent
    verify_manifest(result)
    return result


def install_corpus(manifest_url, manifest_sha256, *, cache_dir=None):
    """Install a hash-pinned public inventory and all its artifacts via datacache."""
    from datacache import fetch_file, install_bundle

    from hitlist.downloads import data_dir

    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in manifest_sha256)
    ):
        raise ValueError("Public corpus manifest requires a SHA-256 digest")
    manifest_sha256 = manifest_sha256.lower()
    root = Path(cache_dir) if cache_dir is not None else data_dir() / "ci-corpus-downloads"
    # The initial JSON must be pinned before any artifact URLs are interpreted.
    manifest_path = Path(
        fetch_file(
            manifest_url,
            destination=root / "manifests" / manifest_sha256 / METADATA,
            expected_sha256=manifest_sha256,
            max_bytes=10 * 1024**2,
            raw=True,
            record_provenance=True,
        )
    )
    manifest = json.loads(manifest_path.read_text())
    names = public_files(manifest)
    assets = {
        name: {
            "url": urljoin(manifest_url, name),
            "sha256": manifest["files"][name]["sha256"],
            "size": manifest["files"][name]["size_bytes"],
        }
        for name in names
    }
    assets[METADATA] = {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "size": manifest_path.stat().st_size,
    }
    paths = install_bundle(root / "corpora" / manifest_sha256, assets)
    result = Path(paths[METADATA]).parent
    verify_manifest(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "verify", "snapshot", "install"))
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--manifest-url")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "install":
        if args.directory is not None or not args.manifest_url or not args.manifest_sha256:
            parser.error("install requires --manifest-url and --manifest-sha256, without directory")
        print(install_corpus(args.manifest_url, args.manifest_sha256, cache_dir=args.cache_dir))
    elif args.directory is None:
        parser.error(f"{args.mode} requires a directory")
    elif args.mode == "write":
        if args.output is None:
            parser.error("write requires --output outside the original metadata")
        write_manifest(args.directory, args.output)
    elif args.mode == "snapshot":
        print(snapshot_corpus(args.directory, cache_dir=args.cache_dir))
    else:
        result = verify_manifest(args.directory)
        print(
            f"Verified {len(result['files'])} corpus files; original artifact contract {result['artifact_version']}."
        )
