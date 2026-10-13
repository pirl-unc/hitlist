"""Consume pinned shared protein facts without changing historical Hitlist views."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path

from .provenance import file_digest

# The pinned source's serialization, not an alternative biological schema.
_COLUMNS = (
    "Run",
    "Protein",
    "Abundance",
    "Channel",
    "BioReplicate",
    "Condition",
    "TechRepMixture",
    "Mixture",
    "sample.type",
    "ProteinGroup",
    "sumIBAQ",
    "IBAQ",
    "rIBAQ",
)
_NUMERIC = {"Abundance", "TechRepMixture", "sumIBAQ", "IBAQ", "rIBAQ"}
_SAMPLE = {
    "Run",
    "Channel",
    "BioReplicate",
    "Condition",
    "TechRepMixture",
    "Mixture",
    "sample.type",
}


def _registry_identity(manifest, download):
    return hashlib.sha256(
        json.dumps(
            {"manifest": manifest, "download": download}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _quote(value):
    return '"' + value.replace('"', '""') + '"'


def _native_row(left, right, samples, assays, manifest):
    """Require exactly two native series from the same original source record."""
    (reference,) = manifest.references
    if left is None or right is None:
        raise ValueError("orfdata source row is missing a native unit")
    for measurement, unit, column in ((left, "iBAQ", "IBAQ"), (right, "rIBAQ", "rIBAQ")):
        p = measurement.provenance
        line = p.get("line")
        if (
            measurement.dataset_id != manifest.dataset_id
            or measurement.dataset_version != manifest.version
            or measurement.target.reference != reference
            or measurement.target.kind != "protein_group"
            or measurement.target.namespace != manifest.coverage["accession"] + ":ProteinGroup"
            or measurement.quantity != "protein_abundance"
            or measurement.unit != unit
            or measurement.scale != "linear"
            or measurement.origin != "observed"
            or p.get("source_asset") != "protein.tsv"
            or p.get("column") != column
            or type(line) is not int
            or line < 2
            or measurement.measurement_id != f"row:{line}:{column}"
        ):
            raise ValueError("Unexpected orfdata native measurement scope")
        native = p.get("native_value")
        if not isinstance(native, str):
            raise ValueError("orfdata native value text is missing")
        expected = None if native == "NA" else float(native)
        if (expected is not None and (not math.isfinite(expected) or expected < 0)) or (
            measurement.value != expected
            or measurement.status != ("missing" if expected is None else "measured")
        ):
            raise ValueError("orfdata native value/status disagrees with source text")
    if (
        left.target != right.target
        or left.sample_id != right.sample_id
        or left.assay_id != right.assay_id
        or any(
            left.provenance.get(k) != right.provenance.get(k)
            for k in ("line", "representative", "native_Abundance", "native_sumIBAQ")
        )
    ):
        raise ValueError("orfdata native units refer to different source rows")
    sample, assay = samples[left.sample_id], assays[left.assay_id]
    attrs = dict(sample.attributes["source_attributes"])
    if (
        set(attrs) != _SAMPLE
        or attrs != dict(assay.attributes["source_attributes"])
        or assay.sample_id != sample.sample_id
        or assay.kind != "digest_ms"
        or assay.reference != reference
        or sample.reference != reference
        or sample.tissue != attrs["Condition"]
        or attrs["sample.type"] != "sample"
    ):
        raise ValueError("orfdata source sample/assay context disagrees")
    row = dict(
        attrs,
        Protein=left.provenance["representative"],
        ProteinGroup=left.target.identifier,
        Abundance=left.provenance["native_Abundance"],
        sumIBAQ=left.provenance["native_sumIBAQ"],
        IBAQ=left.provenance["native_value"],
        rIBAQ=right.provenance["native_value"],
    )
    if any(not isinstance(value, str) or not value for value in row.values()):
        raise ValueError("Incomplete orfdata native source row")
    return left.provenance["line"], row


def _restore_table(dataset, destination, scratch, bounds):
    """Restore native order on disk; the final source hash verifies every cell."""
    manifest = dataset.manifest
    samples = {s.sample_id: s for s in dataset.samples()}
    assays = {a.assay_id: a for a in dataset.assays()}
    with closing(sqlite3.connect(scratch)) as db, db:
        db.execute("PRAGMA page_size=4096")
        db.execute(f"PRAGMA max_page_count={max(1, bounds['max_output_bytes'] // 4096)}")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("CREATE TABLE source_row (line INTEGER PRIMARY KEY, record TEXT NOT NULL)")
        queries = [
            dataset.measurements(quantity="protein_abundance", unit=unit, scale="linear")
            for unit in ("iBAQ", "rIBAQ")
        ]
        n_rows = 0
        for left, right in zip_longest(*queries):
            n_rows += 1
            if n_rows > bounds["max_records"]:
                raise ValueError("orfdata source exceeds max_records")
            line, row = _native_row(left, right, samples, assays, manifest)
            db.execute("INSERT INTO source_row VALUES (?,?)", (line, json.dumps(row)))
        expected = manifest.coverage["source_counts"]
        if (
            n_rows != expected["rows"]
            or len(samples) != expected["assays"]
            or len(assays) != expected["assays"]
            or dataset.counts["measurement"] != 2 * n_rows
        ):
            raise ValueError("orfdata source row/sample counts disagree")
        with destination.open("w", encoding="utf-8", newline="") as output:
            output.write("\t".join(map(_quote, _COLUMNS)) + "\n")
            for expected_line, (line, record) in enumerate(
                db.execute("SELECT line,record FROM source_row ORDER BY line"), 2
            ):
                if line != expected_line:
                    raise ValueError("orfdata source lines are not contiguous")
                row = json.loads(record)
                output.write(
                    "\t".join(row[c] if c in _NUMERIC else _quote(row[c]) for c in _COLUMNS) + "\n"
                )
                if output.tell() > bounds["max_input_bytes"]:
                    raise ValueError("Reconstituted source exceeds max_input_bytes")


def curate_orfdata(source, target, bounds, allow_download, cache_root):
    from orfdata import DatasetRegistry

    from .canine_bulk import _check_asset, _write_bundle

    pin = source.get("orfdata")
    if source["adapter"] != "pfizer" or not pin:
        raise ValueError("Source has no reviewed orfdata provider")
    registry = DatasetRegistry(cache_root=cache_root)
    name, release = pin["dataset_id"], pin["version"]
    manifest = registry.info(name, release)
    download = registry.download_metadata(name, release)
    if _registry_identity(manifest.to_dict(), download) != pin["registry_sha256"]:
        raise ValueError("orfdata registry disagrees with the pinned release")
    if len(manifest.references) != 1 or manifest.references[0].taxon_id != source["source_taxon"]:
        raise ValueError("orfdata reference/species disagrees with source")
    assets = {a.asset_id: a for a in manifest.sources}
    for role, asset_id in (("table", "protein.tsv"), ("reference", "search.fasta")):
        spec = source["assets"][role]
        asset = assets[asset_id]
        if (asset.sha256, asset.size) != (spec["sha256"], spec["size_bytes"]):
            raise ValueError("orfdata source asset disagrees with reviewed source")
        if asset.size > bounds["max_input_bytes"]:
            raise ValueError("Source exceeds max_input_bytes")
    artifact = download["assets"]["dataset.sqlite"]
    if artifact["size"] > bounds["max_upstream_bytes"]:
        raise ValueError("orfdata installed artifact exceeds max_upstream_bytes")
    if download["distribution"]["download_size"] > bounds["max_input_bytes"]:
        raise ValueError("orfdata download exceeds max_input_bytes")
    (spec,) = registry.search_references(name, release)
    if spec.reference != manifest.references[0] or spec.source != assets["search.fasta"]:
        raise ValueError("orfdata search reference disagrees with quantitative source")
    if allow_download:
        registry.download(name, release)
        registry.download_reference(name, release)
    dataset = registry.open(name, release)
    search = registry.open_reference(name, release)
    digest = file_digest(dataset.path)
    if (digest["sha256"], digest["size_bytes"]) != (artifact["sha256"], artifact["size"]):
        raise ValueError("orfdata installed artifact checksum/size mismatch")
    receipt = {
        "provider": "orfdata",
        "package_version": version("orfdata"),
        "registry_sha256": pin["registry_sha256"],
        "dataset_manifest": manifest.to_dict(),
        "download": download,
        "dataset_artifact": digest,
        "search_reference": spec.to_dict(),
        "table_reconstruction": "pfizer-native-tsv-v1; original source checksum verified",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".orfdata-pfizer-", dir=target.parent) as directory:
            root = Path(directory)
            table = root / "protein.tsv"
            _restore_table(dataset, table, root / "rows.sqlite", bounds)
            paths = {"table": table, "reference": search.path}
            for role, path in paths.items():
                _check_asset(path, source["assets"][role])
            return _write_bundle(source, target, bounds, paths, upstream=receipt)
    except (sqlite3.Error, KeyError) as error:
        raise ValueError(f"Invalid orfdata source records: {error}") from error
