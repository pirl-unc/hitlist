"""Verify full benchmark outputs, allowing only named location/time metadata.

Parquet bytes must match exactly: this checks every value, row order, null,
curation flag, contributor payload and schema without loading the corpus in RAM.
Every ignored JSON difference is retained in the comparison receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    if allowed and path[-1] == "path" and Path(left).name != Path(right).name:
        allowed = False  # A different source filename is not just a moved checkout.
    if not allowed:
        raise ValueError(
            f"Unexpected metadata difference: {filename}:{path}: {left!r} != {right!r}"
        )
    return [{"path": list(path), "before": left, "after": right, "reason": reason}]


def compare(left, right):
    manifests = [json.loads((root / "outputs.json").read_text()) for root in (left, right)]
    if manifests[0].keys() != manifests[1].keys():
        raise ValueError("Output file inventory changed")
    if not manifests[0]:
        raise ValueError("No completed output inventory")
    result = {"status": "success", "before": str(left), "after": str(right), "files": {}}
    for name in sorted(manifests[0]):
        if Path(name).name != name:
            raise ValueError("Invalid output filename")
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
