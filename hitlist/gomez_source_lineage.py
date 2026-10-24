"""Original Gomez-Zepeda S3/S5 rows and source-run peptide memberships."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .binding_assay_context import verify_binding_assay_context_package
from .evidence_snapshot import snapshot_file
from .external_evidence import _safe_relative
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 11
CATALOG_SCHEMA_VERSION = 16
REVISION = "gomez_zepeda_2024_original_results_v1"
PARENT_SHA256 = "58ac75009b6c6452cf19b317908f6043f1be414c81dc8ae9e9423a185467721d"
PARENT_MANIFEST = Path("manifests/gomez_parent.json")
ROOT = Path("gomez_source")
RESULTS = ROOT / "results.parquet"
MEMBERSHIPS = ROOT / "memberships.parquet"
REGISTRY = ROOT / "registry.json"
SOURCES = (
    {
        "id": "s3_method_comparison",
        "archive": "MOESM7.zip",
        "archive_url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-46380-y/MediaObjects/41467_2024_46380_MOESM7_ESM.zip",
        "archive_sha256": "ab6db7e77d04881b8ee191b4d6b2025e117870a995089836d34be02ef2ca04af",
        "member": "S4_pep_all_06154.csv",
        "member_sha256": "2baa1a205db58a5a054dbd53dcc4635b8929b1fe18e3f40c98cf1b4e93d5bb17",
        "n_rows": 135_688,
    },
    {
        "id": "s5_spike_experiment",
        "archive": "MOESM9.zip",
        "archive_url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-46380-y/MediaObjects/41467_2024_46380_MOESM9_ESM.zip",
        "archive_sha256": "00ceccb43126c8e10ec97a540df1bdf3b7738dc76fed83256b19d27ebe2a047c",
        "member": "S6_pep_all_06133b.csv",
        "member_sha256": "4c5baa1ed8f915a895521ec183a1cb0d45091efef5a9cfd9024c93be95854c71",
        "n_rows": 459_228,
    },
)
CELL_FILES = {
    "JY": "gomez_zepeda_2024_jy.csv",
    "Raji": "gomez_zepeda_2024_raji.csv",
    "HeLa": "gomez_zepeda_2024_hela.csv",
    "SK-MEL-37": "gomez_zepeda_2024_skmel37.csv",
    "Plasma": "gomez_zepeda_2024_plasma.csv",
}
EXPECTED_UNIONS = {
    "JY": 20_983,
    "Raji": 24_325,
    "HeLa": 10_356,
    "SK-MEL-37": 11_491,
    "Plasma": 1_397,
}
COMMON = (
    "source_record_id",
    "source_row_index",
    "source_phase",
    "source_asset_id",
    "source_pmid",
    "source_dataset",
    "peptide_sequence",
    "cell_line_name",
    "condition_text",
    "biological_replicate_label",
    "sample_label_raw",
    "technical_replicate_label",
    "acquisition_id",
    "peptide_modifications_raw",
    "reported_peptidoform_raw",
    "charge_raw",
    "identification_method_raw",
    "identification_confidence_raw",
    "identification_q_value_raw",
    "tims_ramp_raw",
    "polygon_raw",
    "quantitative_raw",
    "log2_area_raw",
    "measurement_unit",
    "membership_id",
    "derived_observation_id",
    "derivation_link_status",
)
MEMBERSHIP_COLUMNS = (
    "membership_id",
    "source_phase",
    "source_asset_id",
    "peptide_sequence",
    "cell_line_name",
    "condition_text",
    "biological_replicate_label",
    "sample_label_raw",
    "acquisition_id",
    "first_source_record_id",
    "n_source_result_rows",
    "derived_observation_id",
    "derivation_link_status",
)


def _source_reader(path, source):
    if file_digest(path)["sha256"] != source["archive_sha256"]:
        raise ValueError(f"Changed Gomez-Zepeda archive: {source['archive']}")
    archive = zipfile.ZipFile(path)
    if archive.namelist() != [source["member"]]:
        archive.close()
        raise ValueError("Unexpected Gomez-Zepeda ZIP members")
    digest = hashlib.sha256()
    with archive.open(source["member"]) as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != source["member_sha256"]:
        archive.close()
        raise ValueError(f"Changed Gomez-Zepeda CSV member: {source['member']}")
    return archive


def _derived_index(directory):
    table = ds.dataset(directory / "native/observations.parquet").to_table(
        columns=["provenance_id", "peptide", "supplementary_file"],
        filter=ds.field("pmid") == 38480730,
    )
    result = {}
    for row in table.to_pylist():
        key = row["supplementary_file"], row["peptide"]
        if key in result:
            raise ValueError(f"Duplicate derived Gomez-Zepeda peptide: {key}")
        result[key] = row["provenance_id"]
    if len(result) != sum(EXPECTED_UNIONS.values()):
        raise ValueError("Derived Gomez-Zepeda peptide inventory changed")
    return result


def _membership_key(source, row):
    return (
        source["id"],
        row["Cell.Line"],
        row["Condition"],
        row["BioReplicate"],
        row["Sample"],
        row["Source_File_ID"],
        row["Sequence"],
    )


def _membership_id(key):
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return "gomez:membership:" + hashlib.sha256(payload.encode()).hexdigest()


def _derived_link(row, index):
    sequence = row["Sequence"]
    if not 8 <= len(sequence) <= 15:
        return None, "outside_derived_length_8_15"
    key = CELL_FILES[row["Cell.Line"]], sequence
    if key not in index:
        raise ValueError(f"Original Gomez-Zepeda sequence missing derived row: {key}")
    return index[key], "linked_exact_sequence_and_cell_source"


def _schemas(sources):
    columns = set()
    for source, archive in sources:
        with archive.open(source["member"]) as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
            columns.update(next(reader))
    if columns & set(COMMON):
        raise ValueError("Original Gomez-Zepeda column collides with typed field")
    result_schema = pa.schema(
        [(name, pa.int64() if name == "source_row_index" else pa.string()) for name in COMMON]
        + [(name, pa.string()) for name in sorted(columns)]
    )
    membership_schema = pa.schema(
        [
            (name, pa.int64() if name == "n_source_result_rows" else pa.string())
            for name in MEMBERSHIP_COLUMNS
        ]
    )
    return result_schema, membership_schema


def _compile(directory, *, verify=False):
    index = _derived_index(directory)
    sources = [
        (source, _source_reader(directory / ROOT / source["archive"], source)) for source in SOURCES
    ]
    result_schema, membership_schema = _schemas(sources)
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE memberships (membership_id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
        "first_source_record_id TEXT NOT NULL, n_source_result_rows INTEGER NOT NULL)"
    )
    results_path = directory / RESULTS
    output_path = results_path if not verify else None
    writer = (
        pq.ParquetWriter(output_path, result_schema, compression="zstd") if output_path else None
    )
    actual_reader = pq.ParquetFile(results_path).iter_batches(batch_size=8192) if verify else None
    source_rows = Counter()
    unions = defaultdict(set)
    linked_ids = set()
    pending = []

    def flush():
        if not pending:
            return
        expected = pa.Table.from_pylist(pending, schema=result_schema)
        if verify:
            actual = pa.Table.from_batches([next(actual_reader)], schema=result_schema)
            if not actual.equals(expected, check_metadata=False):
                raise ValueError("Gomez-Zepeda original source result rows changed")
        else:
            writer.write_table(expected)
        pending.clear()

    try:
        for source, archive in sources:
            with archive.open(source["member"]) as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
                if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError("Invalid Gomez-Zepeda original CSV header")
                for row_index, raw in enumerate(reader, 1):
                    if None in raw or any(value is None for value in raw.values()):
                        raise ValueError("Malformed Gomez-Zepeda original CSV row")
                    if raw["Cell.Line"] not in CELL_FILES:
                        raise ValueError("Unknown Gomez-Zepeda source material")
                    source_rows[source["id"]] += 1
                    key = _membership_key(source, raw)
                    membership_id = _membership_id(key)
                    record_id = f"gomez:{source['id']}:{row_index}"
                    derived_id, status = _derived_link(raw, index)
                    if derived_id:
                        unions[raw["Cell.Line"]].add(raw["Sequence"])
                        linked_ids.add(derived_id)
                    row = dict.fromkeys(result_schema.names)
                    row.update(raw)
                    row.update(
                        source_record_id=record_id,
                        source_row_index=row_index,
                        source_phase=source["id"],
                        source_asset_id=source["archive"],
                        source_pmid="38480730",
                        source_dataset="gomez_zepeda_2024_original",
                        peptide_sequence=raw["Sequence"],
                        cell_line_name=raw["Cell.Line"],
                        condition_text=raw["Condition"],
                        biological_replicate_label=raw["BioReplicate"],
                        sample_label_raw=raw["Sample"],
                        technical_replicate_label=raw["TechReplicate"],
                        acquisition_id=raw["Source_File_ID"],
                        peptide_modifications_raw=raw["PTM"],
                        reported_peptidoform_raw=raw["Peptide_charge"],
                        charge_raw=raw["Charge"],
                        identification_method_raw=raw["Found_By"],
                        identification_confidence_raw=raw["inv10lgP"],
                        identification_q_value_raw=raw.get("mokapot_q_value"),
                        tims_ramp_raw=raw.get("TIMS_ramp"),
                        polygon_raw=raw.get("Polygon"),
                        quantitative_raw=raw["Area"],
                        log2_area_raw=raw["Log2Area"],
                        measurement_unit="source_peak_area_arbitrary_units",
                        membership_id=membership_id,
                        derived_observation_id=derived_id,
                        derivation_link_status=status,
                    )
                    pending.append(row)
                    payload = json.dumps((key, derived_id, status), ensure_ascii=False)
                    db.execute(
                        "INSERT INTO memberships VALUES (?, ?, ?, 1) "
                        "ON CONFLICT(membership_id) DO UPDATE SET "
                        "n_source_result_rows=n_source_result_rows+1",
                        (membership_id, payload, record_id),
                    )
                    if len(pending) == 8192:
                        flush()
            if source_rows[source["id"]] != source["n_rows"]:
                raise ValueError("Gomez-Zepeda source row count changed")
        flush()
        if verify:
            try:
                next(actual_reader)
            except StopIteration:
                pass
            else:
                raise ValueError("Gomez-Zepeda result table has extra rows")
        if {name: len(union) for name, union in unions.items()} != EXPECTED_UNIONS:
            raise ValueError("Gomez-Zepeda five-source peptide union changed")
        if linked_ids != set(index.values()):
            raise ValueError("Not all original derived Gomez-Zepeda rows have source support")
        membership_path = directory / MEMBERSHIPS
        membership_writer = (
            pq.ParquetWriter(membership_path, membership_schema, compression="zstd")
            if not verify
            else None
        )
        actual_memberships = (
            pq.ParquetFile(membership_path).iter_batches(batch_size=8192) if verify else None
        )
        membership_rows = []
        n_memberships = 0

        def flush_memberships():
            if not membership_rows:
                return
            expected = pa.Table.from_pylist(membership_rows, schema=membership_schema)
            if verify:
                actual = pa.Table.from_batches([next(actual_memberships)], schema=membership_schema)
                if not actual.equals(expected, check_metadata=False):
                    raise ValueError("Gomez-Zepeda peptide memberships changed")
            else:
                membership_writer.write_table(expected)
            membership_rows.clear()

        for membership_id, payload, first_id, count in db.execute(
            "SELECT * FROM memberships ORDER BY membership_id"
        ):
            key, derived_id, status = json.loads(payload)
            phase, cell_line, condition, biological, sample, acquisition, sequence = key
            membership_rows.append(
                {
                    "membership_id": membership_id,
                    "source_phase": phase,
                    "source_asset_id": next(s["archive"] for s in SOURCES if s["id"] == phase),
                    "peptide_sequence": sequence,
                    "cell_line_name": cell_line,
                    "condition_text": condition,
                    "biological_replicate_label": biological,
                    "sample_label_raw": sample,
                    "acquisition_id": acquisition,
                    "first_source_record_id": first_id,
                    "n_source_result_rows": count,
                    "derived_observation_id": derived_id,
                    "derivation_link_status": status,
                }
            )
            n_memberships += 1
            if len(membership_rows) == 8192:
                flush_memberships()
        flush_memberships()
        if verify:
            try:
                next(actual_memberships)
            except StopIteration:
                pass
            else:
                raise ValueError("Gomez-Zepeda membership table has extra rows")
        else:
            membership_writer.close()
        return {
            "n_source_result_rows": sum(source_rows.values()),
            "n_source_memberships": n_memberships,
            "n_derived_observations_linked": len(linked_ids),
            "source_rows": dict(source_rows),
            "derived_sequence_unions": {name: len(unions[name]) for name in sorted(unions)},
        }
    finally:
        if writer:
            writer.close()
        db.close()
        for _, archive in sources:
            archive.close()


def verify_gomez_source_package(
    directory, *, manifest_path="manifest.json", allow_additional_files=False
):
    """Check every inherited byte and replay the original publisher source rows."""
    directory = Path(directory)
    manifest_path = Path(manifest_path)
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        raise ValueError("Invalid Gomez-Zepeda source package manifest path")
    manifest = json.loads((directory / manifest_path).read_text())
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("revision") != REVISION
        or manifest.get("parent_manifest_path") != str(PARENT_MANIFEST)
        or manifest.get("parent_manifest_sha256") != PARENT_SHA256
    ):
        raise ValueError("Unsupported Gomez-Zepeda source package")
    files = manifest.get("files", {})
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / manifest_path
    }
    if not allow_additional_files and set(files) != actual:
        raise ValueError("Gomez-Zepeda source package inventory mismatch")
    if allow_additional_files and not set(files).issubset(actual):
        raise ValueError("Gomez-Zepeda source package inventory mismatch")
    for name, digest in files.items():
        if file_digest(directory / _safe_relative(name)) != digest:
            raise ValueError(f"Changed Gomez-Zepeda package artifact: {name}")
    if files[str(PARENT_MANIFEST)]["sha256"] != PARENT_SHA256:
        raise ValueError("Changed Gomez-Zepeda package predecessor")
    parent = verify_binding_assay_context_package(
        directory, manifest_path=PARENT_MANIFEST, allow_additional_files=True
    )
    additions = {str(PARENT_MANIFEST), str(RESULTS), str(MEMBERSHIPS), str(REGISTRY)} | {
        str(ROOT / source["archive"]) for source in SOURCES
    }
    expected_files = set(parent["files"]) | additions
    if not allow_additional_files and set(files) != expected_files:
        raise ValueError("Gomez-Zepeda package has unowned files")
    if allow_additional_files and not expected_files.issubset(set(files)):
        raise ValueError("Gomez-Zepeda package has unowned files")
    if any(files[name] != digest for name, digest in parent["files"].items()):
        raise ValueError("Gomez-Zepeda package changed predecessor artifacts")
    if any(
        manifest.get(name) != parent.get(name)
        for name in (
            "reference",
            "appended_sources",
            "review_revision",
            "adjudication_manifest_path",
        )
    ):
        raise ValueError("Gomez-Zepeda package changed predecessor metadata")
    registry = json.loads((directory / REGISTRY).read_text())
    if registry != _registry():
        raise ValueError("Gomez-Zepeda source registry changed")
    if manifest.get("reconciliation") != _compile(directory, verify=True):
        raise ValueError("Gomez-Zepeda source reconciliation changed")
    return manifest


def _registry():
    return {
        "revision": REVISION,
        "article_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10937930/",
        "doi": "10.1038/s41467-024-46380-y",
        "pmid": "38480730",
        "license": "CC BY 4.0",
        "sources": list(SOURCES),
        "source_result_grain": "one_original_CSV_data_row_not_independent_donor_or_assay",
        "membership_grain": "sequence_by_phase_condition_sample_and_acquisition_file",
        "quantitative_unit": "source_peak_area_arbitrary_units",
        "non_detection_policy": "not_encoded_as_negative",
    }


def write_gomez_source_package(predecessor_package, archives_directory, destination):
    """Append both publisher archives and queryable source rows to schema 10."""
    predecessor_package = Path(predecessor_package)
    archives_directory = Path(archives_directory)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    parent = verify_binding_assay_context_package(predecessor_package)
    if (parent["schema_version"], parent["catalog_schema_version"]) != (10, 15):
        raise ValueError("Gomez-Zepeda source requires schema-10 predecessor")
    if file_digest(predecessor_package / "manifest.json")["sha256"] != PARENT_SHA256:
        raise ValueError("Gomez-Zepeda source requires pinned 2.0.10 predecessor")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in parent["files"]:
            snapshot_file(predecessor_package / name, staging / name)
        snapshot_file(predecessor_package / "manifest.json", staging / PARENT_MANIFEST)
        for source in SOURCES:
            snapshot_file(
                archives_directory / source["archive"], staging / ROOT / source["archive"]
            )
        (staging / REGISTRY).write_text(json.dumps(_registry(), indent=2, sort_keys=True) + "\n")
        counts = _compile(staging)
        from .version import __version__

        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "parent_kind": "binding_review",
            "parent_manifest_path": str(PARENT_MANIFEST),
            "parent_manifest_sha256": PARENT_SHA256,
            "adjudication_manifest_path": parent["adjudication_manifest_path"],
            "reference": parent["reference"],
            "appended_sources": parent["appended_sources"],
            "review_revision": parent["review_revision"],
            "reconciliation": counts,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_gomez_source_package(staging)
        if file_digest(predecessor_package / "manifest.json")["sha256"] != PARENT_SHA256:
            raise ValueError("Gomez-Zepeda predecessor changed during publication")
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
