"""Verify full benchmark outputs, allowing only named location/time metadata.

Parquet bytes must match exactly: this checks every value, row order, null,
curation flag, contributor payload and schema without loading the corpus in RAM.
Every ignored JSON difference is retained in the comparison receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metadata_differences(left, right, filename, path=()):
    """Fail closed on any difference outside specific build-location stamps."""
    if type(left) is not type(right):
        raise ValueError(f"Metadata type changed: {filename}:{path}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Metadata fields changed: {filename}:{path}")
        return [
            change
            for key in left
            for change in metadata_differences(left[key], right[key], filename, (*path, key))
        ]
    if left == right:
        return []
    allowed = False
    reason = None
    if filename == "observations_meta.json":
        if len(path) == 3 and path[0] == "sources" and path[-1] in ("path", "mtime"):
            allowed = True
            reason = "source location or filesystem timestamp"
        elif len(path) == 3 and path[0] == "parquets" and path[-1] == "mtime":
            allowed = True
            reason = "new output filesystem timestamp"
        elif len(path) == 4 and path[:2] == ("provenance", "sources") and path[-1] == "path":
            allowed = True
            reason = "source location; source hashes and sizes compared separately"
    elif filename == "peptide_mappings_meta.json":
        if path == ("built_at",):
            allowed = True
            reason = "new build timestamp"
        elif len(path) == 3 and path[0] == "observations" and path[-1] in ("path", "mtime"):
            allowed = True
            reason = "new output location or filesystem timestamp"
    elif filename == ".datacache-manifest.json":
        if path == ("created_at",):
            allowed, reason = True, "new generation timestamp"
        elif path == ("definition", "transform", "version"):
            allowed, reason = True, "producer version, checked against the build report"
        elif (
            len(path) == 4
            and path[:2] == ("definition", "sources")
            and path[-1] in ("origin", "identity")
        ) or (
            len(path) == 3
            and path[0] == "sources"
            and path[-1] in ("origin", "identity", "fetched_at")
        ):
            allowed, reason = True, "source location/time; identity and content pins validated"
        elif (
            len(path) == 3
            and path[0] == "outputs"
            and path[1] in ("README.md", "observations_meta.json", "peptide_mappings_meta.json")
            and path[-1] in ("size", "sha256")
        ):
            allowed, reason = (
                True,
                "validated receipt for separately checked metadata/documentation",
            )
    if allowed and path[-1] in ("path", "origin") and Path(left).name != Path(right).name:
        allowed = False  # A different source filename is not just a moved checkout.
    if not allowed:
        raise ValueError(
            f"Unexpected metadata difference: {filename}:{path}: {left!r} != {right!r}"
        )
    return [{"path": list(path), "before": left, "after": right, "reason": reason}]


def validate_generation(root, inventory):
    """Validate generation receipts before allowing their derived differences.

    The benchmark retained every Parquet/JSON file. Reconstruct the deterministic
    README with its recorded producer version and verify its hash too; this is
    explicitly recorded rather than claiming the original README was retained.
    """
    from hitlist.dataset import _write_readme

    output = root / "output"
    manifest = json.loads((output / ".datacache-manifest.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    report = json.loads((root / "report.json").read_text())
    version = report["versions"]["hitlist"]
    definition = manifest["definition"]
    if manifest["format"] != 1 or definition["transform"]["version"] != f"hitlist-{version}":
        raise ValueError("Generation producer differs from its successful build report")
    if report["status"] != "success":
        raise ValueError("Incomplete build cannot support an output comparison")
    datetime.fromisoformat(manifest["created_at"])
    if definition["sources"].keys() != inputs["sources"].keys():
        raise ValueError("Generation source inventory changed")
    if manifest["sources"].keys() != definition["sources"].keys():
        raise ValueError("Generation source receipts are incomplete")
    for name, source in definition["sources"].items():
        receipt = manifest["sources"][name]
        pin = inputs["sources"][name]
        identity = hashlib.sha256(json.dumps(source["origin"], sort_keys=True).encode()).hexdigest()
        if source["identity"] != identity:
            raise ValueError("Generation source location identity is invalid")
        for item in (source, receipt):
            if item["sha256"] != pin["sha256"] or item["size"] != pin["size_bytes"]:
                raise ValueError("Generation source bytes differ from the pinned inputs")
            if item["origin"] != source["origin"] or item["identity"] != identity:
                raise ValueError("Generation source receipt identity differs")
        if receipt["verified"] is not True:
            raise ValueError("Generation source was not checked against its input pin")
        datetime.fromisoformat(receipt["fetched_at"])
    expected = (set(inventory) - {".datacache-manifest.json"}) | {"README.md"}
    if set(manifest["outputs"]) != expected or set(definition["outputs"]) != expected:
        raise ValueError("Generation output inventory changed")
    for name in expected - {"README.md"}:
        record = manifest["outputs"][name]
        if record != {"sha256": inventory[name]["sha256"], "size": inventory[name]["size_bytes"]}:
            raise ValueError(f"Generation output receipt differs: {name}")
    with tempfile.TemporaryDirectory(prefix="hitlist-readme-check-") as temporary:
        directory = Path(temporary)
        for name in expected - {"README.md"}:
            (directory / name).touch()
        shutil.copyfile(output / "observations_meta.json", directory / "observations_meta.json")
        with patch("hitlist.version.__version__", version):
            _write_readme(directory)
        readme = directory / "README.md"
        if manifest["outputs"]["README.md"] != {
            "sha256": digest(readme),
            "size": readme.stat().st_size,
        }:
            raise ValueError("Reconstructed generation README differs from its recorded bytes")
    return {
        "producer_version": version,
        "readme": "reconstructed and hash-verified",
        "sources": "all pins match",
    }


def compare(left, right):
    manifests = [json.loads((root / "outputs.json").read_text()) for root in (left, right)]
    if manifests[0].keys() != manifests[1].keys():
        raise ValueError("Output file inventory changed")
    if not manifests[0]:
        raise ValueError("No completed output inventory")
    if any(Path(name).name != name for name in manifests[0]):
        raise ValueError("Invalid output filename")
    result = {"status": "success", "before": str(left), "after": str(right), "files": {}}
    if ".datacache-manifest.json" in manifests[0]:
        result["generation_validation"] = {
            "before": validate_generation(left, manifests[0]),
            "after": validate_generation(right, manifests[1]),
        }
    for name in sorted(manifests[0]):
        paths = [root / "output" / name for root in (left, right)]
        identities = []
        for path, manifest in zip(paths, manifests, strict=True):
            identity = {"sha256": digest(path), "size_bytes": path.stat().st_size}
            if identity != manifest[name]:
                raise ValueError(f"Retained output differs from its build receipt: {path}")
            identities.append(identity)
        entry = {"before": identities[0], "after": identities[1]}
        if name.endswith(".parquet"):
            if identities[0] != identities[1]:
                raise ValueError(f"Parquet bytes differ; requires investigation: {name}")
            metadata = pq.ParquetFile(paths[0])
            entry.update(
                comparison="byte-identical complete artifact",
                n_rows=metadata.metadata.num_rows,
                n_columns=len(metadata.schema_arrow),
                columns=metadata.schema_arrow.names,
                schema_sha256=hashlib.sha256(metadata.schema_arrow.serialize()).hexdigest(),
            )
        elif name.endswith(".json"):
            entry.update(
                comparison="all metadata values equal except the recorded location/time fields",
                expected_differences=metadata_differences(
                    json.loads(paths[0].read_text()), json.loads(paths[1].read_text()), name
                ),
            )
        else:
            raise ValueError(f"Unrecognized output: {name}")
        result["files"][name] = entry
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    args.report.write_text(json.dumps(compare(args.before, args.after), indent=2) + "\n")
