"""Source-faithful numeric representations for bulk protein and gene expression.

Each populated native value becomes a measurement linked to an existing typed
result. Bulk peptide rows remain detection evidence because their source numeric
cells are blank. The successor generation independently snapshots all parent
bytes before adding these relations.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import file_digest
from .tcell_evidence import verify_tcell_evidence_package

PACKAGE_SCHEMA_VERSION = 6
CATALOG_SCHEMA_VERSION = 7
BULK_FIELDS = ("log2_intensity", "abundance_log2_normalized", "abundance_percentile")
EXPRESSION_FIELDS = ("tpm", "log2_tpm")
EXPECTED_SOURCE_ROWS = {
    "bulk": {
        "Bekker-Jensen_2017:peptide": 2_047_003,
        "Bekker-Jensen_2017:protein": 156_211,
        "CCLE_Nusinow_2020:protein": 66_074,
    },
    "expression": {
        "ENCODE_GM12878_polyA_rnaseq:gene": 23_716,
        "DepMap_24Q4_HAP1_gene:gene": 19_193,
    },
}
EXPECTED_MEASUREMENTS = {
    "bulk": {
        "Bekker-Jensen_2017": {
            "log2_intensity": 156_211,
            "abundance_percentile": 156_211,
        },
        "CCLE_Nusinow_2020": {
            "abundance_log2_normalized": 66_074,
            "abundance_percentile": 66_074,
        },
    },
    "expression": {
        "ENCODE_GM12878_polyA_rnaseq": {"tpm": 23_716, "log2_tpm": 23_716},
        "DepMap_24Q4_HAP1_gene": {"tpm": 19_193, "log2_tpm": 19_193},
    },
}
MEASUREMENT_SCHEMA = pa.schema(
    [
        pa.field("measurement_id", pa.string()),
        pa.field("observation_id", pa.string()),
        pa.field("source_record_id", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("native_row_index", pa.int64()),
        pa.field("endpoint", pa.string()),
        pa.field("source_value_field", pa.string()),
        pa.field("quantity_scale", pa.string()),
        pa.field("quantitative_value", pa.float64()),
        pa.field("value_status", pa.string()),
        pa.field("measurement_unit", pa.string()),
        pa.field("unit_status", pa.string()),
    ]
)
_MISSING = object()


class _Sink:
    def __init__(self, path, *, batch_size=16_384):
        self.writer = pq.ParquetWriter(path, MEASUREMENT_SCHEMA, compression="zstd")
        self.batch_size = batch_size
        self.rows = []
        self.n_measurements = 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=MEASUREMENT_SCHEMA))
            self.n_measurements += len(self.rows)
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()


def _rows(path, columns):
    """Stream columns together without assuming two Parquet row groups align."""
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=16_384):
        values = batch.to_pydict()
        for row in zip(*(values[column] for column in columns)):
            yield dict(zip(columns, row))


def _write_one(native_path, typed_path, output_path, *, kind):
    fields = BULK_FIELDS if kind == "bulk" else EXPRESSION_FIELDS
    native_columns = (
        ("source", "granularity", *fields)
        if kind == "bulk"
        else ("source_id", "granularity", "gene_id", "profile_id", *fields)
    )
    typed_columns = (
        ("native_row_index", "source_record_id", "observation_id", "source_dataset", "grain")
        if kind == "bulk"
        else (
            "native_row_index",
            "source_record_id",
            "observation_id",
            "source_dataset",
            "grain",
            "gene_id",
            "profile_id_raw",
        )
    )
    source_rows = Counter()
    measurements = Counter()
    n_peptide_rows_with_numeric_value = 0
    n_zero_tpm_measurements = 0
    sink = _Sink(output_path)
    try:
        native_rows = _rows(native_path, native_columns)
        typed_rows = _rows(typed_path, typed_columns)
        for index, (native, typed) in enumerate(
            zip_longest(native_rows, typed_rows, fillvalue=_MISSING)
        ):
            if native is _MISSING or typed is _MISSING:
                raise ValueError(f"{kind} native/typed result count differs")
            source = native["source" if kind == "bulk" else "source_id"]
            grain = native["granularity"]
            typed_grain = grain if kind == "bulk" else "gene_profile"
            native_file = "bulk_proteomics.parquet" if kind == "bulk" else "line_expression.parquet"
            observation_id = f"bulk:v4:{index}" if kind == "bulk" else f"expression:v4:{index}"
            if (
                typed["native_row_index"] != index
                or typed["source_record_id"] != f"{native_file}:{index}"
                or typed["observation_id"] != observation_id
                or typed["source_dataset"] != source
                or typed["grain"] != typed_grain
            ):
                raise ValueError(f"{kind} source-row alignment mismatch: {index}")
            if kind == "expression" and (
                grain != "gene"
                or typed["gene_id"] != native["gene_id"]
                or typed["profile_id_raw"] != native["profile_id"]
            ):
                raise ValueError(f"Expression gene/profile alignment mismatch: {index}")
            source_rows[f"{source}:{grain}"] += 1
            populated = [field for field in fields if native[field] is not None]
            if kind == "bulk" and grain == "peptide":
                n_peptide_rows_with_numeric_value += bool(populated)
                continue
            if kind == "bulk" and grain != "protein":
                raise ValueError(f"Unexpected bulk result grain: {grain}")
            for field in populated:
                value = native[field]
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {kind} source value: {index}:{field}")
                if field == "tpm" and value == 0:
                    n_zero_tpm_measurements += 1
                measurements[source, field] += 1
                sink.add(
                    {
                        "measurement_id": f"{observation_id}:{field}",
                        "observation_id": observation_id,
                        "source_record_id": typed["source_record_id"],
                        "source_dataset": source,
                        "native_row_index": index,
                        "endpoint": "protein_abundance" if kind == "bulk" else "gene_expression",
                        "source_value_field": field,
                        "quantity_scale": field,
                        "quantitative_value": value,
                        "value_status": "source_numeric_cell",
                        "measurement_unit": "TPM" if field == "tpm" else None,
                        "unit_status": "source_named_unit" if field == "tpm" else "unreported",
                    }
                )
    finally:
        sink.close()
    return {
        "n_source_rows": sum(source_rows.values()),
        "n_measurements": sink.n_measurements,
        "n_peptide_rows_with_numeric_value": n_peptide_rows_with_numeric_value,
        "n_zero_tpm_measurements": n_zero_tpm_measurements,
        "source_rows": dict(sorted(source_rows.items())),
        "measurements_by_source_field": {
            source: {
                field: count
                for (item_source, field), count in sorted(measurements.items())
                if item_source == source
            }
            for source in sorted({source for source, _ in measurements})
        },
    }


def _validate_reconciliation(reconciliation):
    for kind, expected_rows in EXPECTED_SOURCE_ROWS.items():
        actual = reconciliation[kind]
        if actual["source_rows"] != expected_rows:
            raise ValueError(f"Pinned {kind} source-row count changed")
        if actual["measurements_by_source_field"] != EXPECTED_MEASUREMENTS[kind]:
            raise ValueError(f"Pinned {kind} numeric representations changed")
        expected_measurements = sum(
            count for source in EXPECTED_MEASUREMENTS[kind].values() for count in source.values()
        )
        if actual["n_measurements"] != expected_measurements:
            raise ValueError(f"Pinned {kind} measurement count changed")
    if reconciliation["bulk"]["n_peptide_rows_with_numeric_value"] != 0:
        raise ValueError("Bulk peptide detection acquired a numeric source value")
    if reconciliation["expression"]["n_zero_tpm_measurements"] != 3_771:
        raise ValueError("Pinned measured-zero expression count changed")


def _field_disposition():
    return {
        "bulk_proteomics.parquet": {
            field: {
                "typed_artifact": "typed/bulk_numeric_measurements.parquet",
                "source_grain": "protein",
                "quantity_scale": field,
                "unit_status": "unreported",
            }
            for field in BULK_FIELDS
        },
        "line_expression.parquet": {
            field: {
                "typed_artifact": "typed/line_expression_measurements.parquet",
                "source_grain": "gene",
                "quantity_scale": field,
                "unit_status": "source_named_unit" if field == "tpm" else "unreported",
            }
            for field in EXPRESSION_FIELDS
        },
    }


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid evidence artifact path: {value}")
    return path


def _snapshot_file(source, target):
    """Copy a package byte stream without sharing the source inode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_stat, target_stat = source.stat(), target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise ValueError(f"Supporting numeric snapshot shares source inode: {source}")


def verify_supporting_numeric_evidence_package(directory, *, manifest_path="manifest.json"):
    """Verify an appended numeric generation and its complete T-cell parent."""
    directory = Path(directory)
    location = _safe_relative(manifest_path)
    manifest = json.loads((directory / location).read_text())
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported supporting numeric package schema")
    if manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported supporting numeric catalog schema")
    files = manifest.get("files", {})
    if location == Path("manifest.json"):
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"
        }
        if set(files) != actual:
            raise ValueError("Supporting numeric package file inventory mismatch")
    for relative, expected in files.items():
        if file_digest(directory / _safe_relative(relative)) != expected:
            raise ValueError(f"Supporting numeric artifact mismatch: {relative}")
    parent_path = "manifests/tcell.json"
    if files[parent_path]["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("Supporting numeric predecessor manifest mismatch")
    parent = verify_tcell_evidence_package(directory, manifest_path=parent_path)
    if parent["catalog_schema_version"] != 6:
        raise ValueError("Unsupported T-cell predecessor catalog")
    reconciliation = manifest["reconciliation"]
    _validate_reconciliation(reconciliation)
    if manifest.get("field_disposition") != _field_disposition():
        raise ValueError("Supporting numeric field disposition changed")
    for kind, name in (
        ("bulk", "bulk_numeric_measurements.parquet"),
        ("expression", "line_expression_measurements.parquet"),
    ):
        parquet = pq.ParquetFile(directory / "typed" / name)
        if (
            parquet.schema_arrow != MEASUREMENT_SCHEMA
            or parquet.metadata.num_rows != reconciliation[kind]["n_measurements"]
        ):
            raise ValueError(f"Supporting numeric typed relation mismatch: {kind}")
    return manifest


def write_supporting_numeric_evidence_package(tcell_package, destination):
    """Append source numeric cells without modifying the complete predecessor."""
    tcell_package = Path(tcell_package)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Supporting numeric package already exists: {destination}")
    parent = verify_tcell_evidence_package(tcell_package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent["files"]:
            source = tcell_package / _safe_relative(relative)
            target = staging / relative
            _snapshot_file(source, target)
        _snapshot_file(tcell_package / "manifest.json", staging / "manifests" / "tcell.json")
        typed = staging / "typed"
        reconciliation = {
            "bulk": _write_one(
                staging / "native" / "bulk_proteomics.parquet",
                typed / "bulk_results.parquet",
                typed / "bulk_numeric_measurements.parquet",
                kind="bulk",
            ),
            "expression": _write_one(
                staging / "native" / "line_expression.parquet",
                typed / "expression_results.parquet",
                typed / "line_expression_measurements.parquet",
                kind="expression",
            ),
        }
        _validate_reconciliation(reconciliation)
        verify_tcell_evidence_package(tcell_package)
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_manifest_sha256": files["manifests/tcell.json"]["sha256"],
            "field_disposition": _field_disposition(),
            "reconciliation": reconciliation,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_supporting_numeric_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
