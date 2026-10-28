"""Content-bound, lossless MHC assay evidence package.

The typed tables add roles to a pinned native corpus. Native rows and source
contributors remain in the same portable package, so a typed projection never
becomes the only surviving representation of an assay or source claim.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .provenance import CONTRIBUTOR_COLUMNS, file_digest

PACKAGE_SCHEMA_VERSION = 2
# Freeze each MHC package's catalog contract independently of the live catalog.
# Schema 1 / catalog 2 predates required positive-MS core validation.
MHC_CATALOG_SCHEMA_VERSION = 3
MHC_PARTITIONS = {
    "observations.parquet": "mhc_positive_ms",
    "binding.parquet": "mhc_binding",
    "other_assays.parquet": "mhc_other",
}
PINNED_V4_MANIFEST_SHA256 = "2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7"
PINNED_V4_COUNTS = {
    "mhc_positive_ms": 4_486_991,
    "mhc_binding": 814_268,
    "mhc_other": 26_537,
}

# Exact v4 schema. New native columns require an explicit disposition before
# this projection can run. The original files are packaged regardless of role.
NATIVE_FIELDS = (
    "peptide",
    "peptide_modifications",
    "has_ptm",
    "peptide_extended",
    "mhc_restriction",
    "mhc_species",
    "mhc_species_source",
    "mhc_species_context_disagrees",
    "mhc_class",
    "mhc_class_reported",
    "mhc_class_source",
    "mhc_class_corrected",
    "allele_resolution",
    "serotype",
    "serotypes",
    "serotype_source",
    "assay_iri",
    "reference_iri",
    "pmid",
    "submission_id",
    "reference_title",
    "source_organism",
    "species",
    "host",
    "host_age",
    "host_mhc_types",
    "process_type",
    "disease",
    "disease_stage",
    "source_tissue",
    "cell_name",
    "culture_condition",
    "antigen_processing_comments",
    "assay_comments",
    "qualitative_measurement",
    "assay_method",
    "response_measured",
    "measurement_units",
    "measurement_inequality",
    "quantitative_measurement",
    "quantitative_value",
    "assay_modality",
    "assay_modality_source",
    "is_binding_assay",
    "is_ms_observation",
    "ms_outcome_status",
    "is_non_peptide_ligand",
    "src_cancer",
    "src_adjacent_to_tumor",
    "src_activated_apc",
    "src_healthy_tissue",
    "src_healthy_thymus",
    "src_healthy_reproductive",
    "src_healthy_reproductive_female",
    "src_healthy_reproductive_male",
    "src_cell_line",
    "src_ebv_lcl",
    "src_ex_vivo",
    "cell_line_name",
    "cell_type",
    "is_monoallelic",
    "monoallelic_host",
    "restriction_evidence",
    "mhc_allele_set",
    "mhc_allele_provenance",
    "mhc_allele_set_size",
    "attributed_sample_label",
    "provenance_id",
    "source",
    "supplementary_file",
    "is_potential_contaminant",
)

TYPED_FIELDS = {
    "peptide": "peptide_sequence",
    "peptide_modifications": "peptide_modifications_raw",
    "has_ptm": "has_ptm",
    "peptide_extended": "peptide_extended",
    "mhc_restriction": ("reported_mhc", "tested_mhc"),
    "assay_iri": "source_assay_id",
    "reference_iri": "source_reference_iri",
    "pmid": "source_pmid",
    "qualitative_measurement": ("qualitative_result", "qualitative_status"),
    "assay_method": "assay_method",
    "response_measured": ("assay_readout", "endpoint_reported", "endpoint"),
    "measurement_units": ("measurement_unit", "unit_status"),
    "measurement_inequality": ("measurement_inequality", "inequality_status"),
    "quantitative_measurement": ("quantitative_raw", "quantitative_status"),
    "quantitative_value": ("quantitative_value", "quantitative_status"),
    "assay_modality": "assay_modality",
    "assay_modality_source": "assay_modality_basis",
    "ms_outcome_status": ("outcome_status", "endpoint_basis"),
    "is_non_peptide_ligand": "is_non_peptide_ligand",
    "restriction_evidence": "restriction_evidence",
    "mhc_allele_set": "candidate_mhc_set_raw",
    "provenance_id": "observation_id",
    "source": "source_dataset",
    "supplementary_file": "source_file",
    "is_potential_contaminant": "is_potential_contaminant",
}

DEFERRED_FIELDS = frozenset(
    {
        "mhc_species",
        "mhc_species_source",
        "mhc_species_context_disagrees",
        "mhc_class",
        "mhc_class_reported",
        "mhc_class_source",
        "mhc_class_corrected",
        "allele_resolution",
        "serotype",
        "serotypes",
        "serotype_source",
        "source_organism",
        "species",
        "host",
        "host_age",
        "host_mhc_types",
        "process_type",
        "disease",
        "disease_stage",
        "source_tissue",
        "cell_name",
        "culture_condition",
        "antigen_processing_comments",
        "assay_comments",
        "src_cancer",
        "src_adjacent_to_tumor",
        "src_activated_apc",
        "src_healthy_tissue",
        "src_healthy_thymus",
        "src_healthy_reproductive",
        "src_healthy_reproductive_female",
        "src_healthy_reproductive_male",
        "src_cell_line",
        "src_ebv_lcl",
        "src_ex_vivo",
        "cell_line_name",
        "cell_type",
        "is_monoallelic",
        "monoallelic_host",
        "mhc_allele_provenance",
        "mhc_allele_set_size",
        "attributed_sample_label",
    }
)

SOURCE_ONLY_FIELDS = frozenset(
    {
        "submission_id",
        "reference_title",
        "is_binding_assay",
        "is_ms_observation",
    }
)

_all_dispositions = set(TYPED_FIELDS) | DEFERRED_FIELDS | SOURCE_ONLY_FIELDS
if _all_dispositions != set(NATIVE_FIELDS) or (
    set(TYPED_FIELDS) & DEFERRED_FIELDS
    or set(TYPED_FIELDS) & SOURCE_ONLY_FIELDS
    or DEFERRED_FIELDS & SOURCE_ONLY_FIELDS
):
    raise RuntimeError("MHC native field dispositions must cover the exact schema once")

_READ_FIELDS = (
    "provenance_id",
    "source",
    "supplementary_file",
    "assay_iri",
    "reference_iri",
    "pmid",
    "peptide",
    "peptide_modifications",
    "has_ptm",
    "peptide_extended",
    "mhc_restriction",
    "mhc_allele_set",
    "restriction_evidence",
    "assay_method",
    "response_measured",
    "assay_modality",
    "assay_modality_source",
    "ms_outcome_status",
    "is_non_peptide_ligand",
    "is_potential_contaminant",
    "qualitative_measurement",
    "quantitative_measurement",
    "quantitative_value",
    "measurement_units",
    "measurement_inequality",
)

_OBSERVATION_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("evidence_type", pa.string()),
        ("source_population", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_assay_id", pa.string()),
        ("source_reference_iri", pa.string()),
        ("source_pmid", pa.string()),
        ("peptide_sequence", pa.string()),
        ("peptide_modifications_raw", pa.string()),
        ("has_ptm", pa.bool_()),
        ("peptide_extended", pa.string()),
        ("is_non_peptide_ligand", pa.bool_()),
        ("is_potential_contaminant", pa.bool_()),
        ("reported_mhc", pa.string()),
        ("tested_mhc", pa.string()),
        ("candidate_mhc_set_raw", pa.string()),
        ("peptide_mhc_restriction", pa.string()),
        ("peptide_restriction_status", pa.string()),
        ("restriction_evidence", pa.string()),
        ("assay_method", pa.string()),
        ("assay_readout", pa.string()),
        ("assay_modality", pa.string()),
        ("assay_modality_basis", pa.string()),
        ("outcome_status", pa.string()),
        ("native_partition", pa.string()),
        ("native_row_index", pa.int64()),
    ]
)

_MEASUREMENT_SCHEMA = pa.schema(
    [
        ("measurement_id", pa.string()),
        ("observation_id", pa.string()),
        ("endpoint", pa.string()),
        ("endpoint_reported", pa.string()),
        ("endpoint_basis", pa.string()),
        ("qualitative_result", pa.string()),
        ("qualitative_status", pa.string()),
        ("quantitative_raw", pa.string()),
        ("quantitative_value", pa.float64()),
        ("quantitative_status", pa.string()),
        ("measurement_unit", pa.string()),
        ("unit_status", pa.string()),
        ("measurement_inequality", pa.string()),
        ("inequality_status", pa.string()),
    ]
)

# A positive ligand-MS result has a uniform two-relation contract.  It is
# intentionally small: neither a donor/specimen nor instrument/protocol fact
# is fabricated when a source did not report one.  Status/value pairs make
# every optional measurement axis explicit.
MHC_POSITIVE_MS_CORE_CONTRACT = {
    "observation_required": [
        "observation_id",
        "evidence_type",
        "source_population",
        "source_dataset",
        "source_assay_id",
        "peptide_sequence",
        "assay_modality",
        "assay_modality_basis",
        "outcome_status",
        "native_partition",
        "native_row_index",
    ],
    "measurement_required": [
        "measurement_id",
        "observation_id",
        "endpoint",
        "endpoint_basis",
        "qualitative_status",
        "quantitative_status",
        "unit_status",
        "inequality_status",
    ],
    "conditional_fields": {
        "qualitative_status=reported": ["qualitative_result"],
        "quantitative_status=reported": ["quantitative_raw", "quantitative_value"],
        "quantitative_status=unresolved": ["quantitative_raw"],
        "unit_status=reported": ["measurement_unit"],
        "inequality_status=reported": ["measurement_inequality"],
    },
    "unavailable_without_source_relation": [
        "specimen_id",
        "aliquot_id",
        "instrument",
        "acquisition_mode",
        "fractionation",
    ],
}


def _reported(value):
    return value if value is not None and value != "" else None


def _measurement_status(raw, parsed):
    if parsed is not None:
        return "reported"
    return "unresolved" if raw is not None else "unreported"


def _validate_positive_ms_core(observation, measurement):
    """Reject an incomplete or self-contradictory positive-MS projection."""
    for field in MHC_POSITIVE_MS_CORE_CONTRACT["observation_required"]:
        if observation[field] is None or observation[field] == "":
            raise ValueError(f"Positive MHC ligand MS lacks core observation field: {field}")
    if observation["evidence_type"] != "mhc_positive_ms":
        raise ValueError("Positive MHC ligand MS has wrong evidence type")
    if observation["assay_modality"] != "ms":
        raise ValueError("Positive MHC ligand MS has non-MS modality")
    if observation["outcome_status"] not in ("positive_result", "curated_ligand_table"):
        raise ValueError("Positive MHC ligand MS has non-positive outcome")
    if observation["native_partition"] != "observations.parquet":
        raise ValueError("Positive MHC ligand MS has wrong native partition")
    for field in MHC_POSITIVE_MS_CORE_CONTRACT["measurement_required"]:
        if measurement[field] is None or measurement[field] == "":
            raise ValueError(f"Positive MHC ligand MS lacks core measurement field: {field}")
    if measurement["observation_id"] != observation["observation_id"]:
        raise ValueError("Positive MHC ligand MS measurement points to another observation")
    for status, value, reported in (
        ("qualitative_status", "qualitative_result", "reported"),
        ("unit_status", "measurement_unit", "reported"),
        ("inequality_status", "measurement_inequality", "reported"),
    ):
        if (measurement[status] == reported) != (measurement[value] is not None):
            raise ValueError(f"Positive MHC ligand MS has inconsistent {status}")
    quantitative = measurement["quantitative_status"]
    raw, value = measurement["quantitative_raw"], measurement["quantitative_value"]
    if quantitative == "reported" and (raw is None or value is None):
        raise ValueError("Positive MHC ligand MS reported quantity lacks raw or parsed value")
    if quantitative == "unresolved" and (raw is None or value is not None):
        raise ValueError("Positive MHC ligand MS unresolved quantity is inconsistent")
    if quantitative == "unreported" and (raw is not None or value is not None):
        raise ValueError("Positive MHC ligand MS unreported quantity is inconsistent")
    if quantitative not in ("reported", "unresolved", "unreported"):
        raise ValueError("Positive MHC ligand MS has unknown quantitative status")
    for status in ("unit_status", "inequality_status"):
        if quantitative == "unreported" and measurement[status] != "not_applicable":
            raise ValueError(f"Positive MHC ligand MS has {status} without a quantity")
        if quantitative != "unreported" and measurement[status] == "not_applicable":
            raise ValueError(f"Positive MHC ligand MS omits applicable {status}")


def _result_rows(batch, evidence_type, partition, offset):
    columns = batch.to_pydict()
    observations = []
    measurements = []
    for index, values in enumerate(zip(*(columns[name] for name in _READ_FIELDS))):
        row = dict(zip(_READ_FIELDS, values))
        observation_id = row["provenance_id"]
        source_assay_id = row["assay_iri"]
        peptide = row["peptide"]
        modality = row["assay_modality"]
        outcome = row["ms_outcome_status"]
        if not observation_id or not source_assay_id or not peptide:
            raise ValueError(f"Missing MHC result identity at {partition} row {offset + index}")
        if evidence_type == "mhc_positive_ms":
            valid = modality == "ms" and outcome in ("positive_result", "curated_ligand_table")
        elif evidence_type == "mhc_binding":
            valid = modality == "binding"
        else:
            valid = modality in ("ms", "structural", "non_ms_ligand") and not (
                modality == "ms" and outcome in ("positive_result", "curated_ligand_table")
            )
        if not valid:
            raise ValueError(f"MHC family mismatch at {partition} row {offset + index}")
        reported_mhc = _reported(row["mhc_restriction"])
        tested_mhc = (
            reported_mhc
            if evidence_type == "mhc_binding" and row["restriction_evidence"] == "experimental"
            else None
        )
        observation = {
            "observation_id": observation_id,
            "evidence_type": evidence_type,
            "source_population": "mhc_v4",
            "source_dataset": row["source"],
            "source_file": _reported(row["supplementary_file"]),
            "source_assay_id": source_assay_id,
            "source_reference_iri": _reported(row["reference_iri"]),
            "source_pmid": (str(row["pmid"]) if _reported(row["pmid"]) is not None else None),
            "peptide_sequence": peptide,
            "peptide_modifications_raw": _reported(row["peptide_modifications"]),
            "has_ptm": row["has_ptm"],
            "peptide_extended": _reported(row["peptide_extended"]),
            "is_non_peptide_ligand": row["is_non_peptide_ligand"],
            "is_potential_contaminant": row["is_potential_contaminant"],
            "reported_mhc": reported_mhc,
            "tested_mhc": tested_mhc,
            "candidate_mhc_set_raw": _reported(row["mhc_allele_set"]),
            "peptide_mhc_restriction": None,
            "peptide_restriction_status": "unreported",
            "restriction_evidence": row["restriction_evidence"],
            "assay_method": _reported(row["assay_method"]),
            "assay_readout": _reported(row["response_measured"]),
            "assay_modality": modality,
            "assay_modality_basis": row["assay_modality_source"],
            "outcome_status": outcome,
            "native_partition": partition,
            "native_row_index": offset + index,
        }
        endpoint_reported = _reported(row["response_measured"])
        derived_endpoint = endpoint_reported is None and outcome == "curated_ligand_table"
        endpoint = "positive_mhc_ligand_ms" if derived_endpoint else endpoint_reported
        raw = _reported(row["quantitative_measurement"])
        parsed = row["quantitative_value"]
        unit = _reported(row["measurement_units"])
        inequality = _reported(row["measurement_inequality"])
        has_quantitative = raw is not None or parsed is not None
        qualitative = _reported(row["qualitative_measurement"])
        measurement = {
            "measurement_id": f"{observation_id}:reported",
            "observation_id": observation_id,
            "endpoint": endpoint,
            "endpoint_reported": endpoint_reported,
            "endpoint_basis": (
                "curated_ligand_table"
                if derived_endpoint
                else "reported"
                if endpoint_reported is not None
                else "unreported"
            ),
            "qualitative_result": qualitative,
            "qualitative_status": "reported" if qualitative is not None else "unreported",
            "quantitative_raw": raw,
            "quantitative_value": parsed,
            "quantitative_status": _measurement_status(raw, parsed),
            "measurement_unit": unit,
            "unit_status": (
                "reported"
                if unit is not None
                else "unreported"
                if has_quantitative
                else "not_applicable"
            ),
            "measurement_inequality": inequality,
            "inequality_status": (
                "reported"
                if inequality is not None
                else "unreported"
                if has_quantitative
                else "not_applicable"
            ),
        }
        if evidence_type == "mhc_positive_ms":
            _validate_positive_ms_core(observation, measurement)
        observations.append(observation)
        measurements.append(measurement)
    return observations, measurements


def _native_disposition():
    return {
        field: (
            {
                "status": "typed",
                "logical_fields": list(
                    TYPED_FIELDS[field]
                    if isinstance(TYPED_FIELDS[field], tuple)
                    else (TYPED_FIELDS[field],)
                ),
            }
            if field in TYPED_FIELDS
            else {"status": "deferred_to_context_protocol", "retained_in": "native"}
            if field in DEFERRED_FIELDS
            else {"status": "source_only", "retained_in": "native"}
        )
        for field in NATIVE_FIELDS
    }


def _source_registry(source_manifest):
    return {
        name: {
            "snapshot": snapshot,
            "row_payload": "native/observation_contributors.parquet",
            "access": "source_row_payload_in_package",
            "upstream_file_access": "not_established",
            "license_status": "unverified",
        }
        for name, snapshot in sorted(
            source_manifest.get("provenance", {}).get("sources", {}).items()
        )
    }


def _source_manifest(source_dir):
    path = source_dir / "observations_meta.json"
    manifest = json.loads(path.read_text())
    required = set(MHC_PARTITIONS) | {"observation_contributors.parquet"}
    if manifest.get("ci_corpus_manifest_version") != 2 or not required <= set(
        manifest.get("files", {})
    ):
        raise ValueError("A complete portable v4 corpus manifest is required")
    if not manifest.get("provenance", {}).get("sources"):
        raise ValueError("Portable MHC corpus lacks its source registry")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_digest(source_dir / name) != expected:
            raise ValueError(f"Source corpus artifact mismatch: {name}")
    return manifest, file_digest(path)


def _write_typed(source_dir, typed_dir, audit_path, known_sources):
    typed_dir.mkdir()
    counts = Counter()
    source_counts = Counter()
    modality_counts = Counter()
    audit = sqlite3.connect(audit_path)
    audit.execute("PRAGMA journal_mode=OFF")
    audit.execute("PRAGMA synchronous=OFF")
    audit.execute("CREATE TABLE results (id TEXT PRIMARY KEY) WITHOUT ROWID")
    audit.execute("CREATE TABLE links (id TEXT PRIMARY KEY) WITHOUT ROWID")
    try:
        with (
            pq.ParquetWriter(
                typed_dir / "mhc_observations.parquet", _OBSERVATION_SCHEMA, compression="zstd"
            ) as observation_writer,
            pq.ParquetWriter(
                typed_dir / "mhc_measurements.parquet", _MEASUREMENT_SCHEMA, compression="zstd"
            ) as measurement_writer,
        ):
            for partition, evidence_type in MHC_PARTITIONS.items():
                parquet = pq.ParquetFile(source_dir / partition)
                if tuple(parquet.schema_arrow.names) != NATIVE_FIELDS:
                    raise ValueError(f"Unreviewed MHC source schema: {partition}")
                offset = 0
                for batch in parquet.iter_batches(batch_size=16384, columns=list(_READ_FIELDS)):
                    observations, measurements = _result_rows(
                        batch, evidence_type, partition, offset
                    )
                    try:
                        audit.executemany(
                            "INSERT INTO results VALUES (?)",
                            ((row["observation_id"],) for row in observations),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            "Duplicate MHC observation ID across partitions"
                        ) from error
                    observation_writer.write_table(
                        pa.Table.from_pylist(observations, schema=_OBSERVATION_SCHEMA)
                    )
                    measurement_writer.write_table(
                        pa.Table.from_pylist(measurements, schema=_MEASUREMENT_SCHEMA)
                    )
                    counts[evidence_type] += len(observations)
                    source_counts.update(
                        (evidence_type, row["source_dataset"]) for row in observations
                    )
                    modality_counts.update(
                        (evidence_type, row["assay_modality"], row["outcome_status"])
                        for row in observations
                    )
                    offset += len(observations)
        n_links = 0
        contributor = pq.ParquetFile(source_dir / "observation_contributors.parquet")
        if tuple(contributor.schema_arrow.names) != tuple(CONTRIBUTOR_COLUMNS):
            raise ValueError("Unreviewed MHC contributor schema")
        contributor_fields = (
            "provenance_id",
            "source_record_id",
            "source_dataset",
            "source_row",
            "original_fields",
            "source_row_values",
        )
        for batch in contributor.iter_batches(batch_size=65536, columns=contributor_fields):
            for index in (0, 1, 2, 4, 5):
                values = batch.column(index)
                if pc.any(pc.is_null(values)).as_py() or pc.any(pc.equal(values, "")).as_py():
                    raise ValueError(f"MHC contributor lacks {contributor_fields[index]}")
            rows = batch.column(3)
            if pc.any(pc.is_null(rows)).as_py() or pc.min(rows).as_py() < 1:
                raise ValueError("MHC contributor lacks a positive source row locator")
            if not set(batch.column(2).to_pylist()) <= known_sources:
                raise ValueError("MHC contributor uses an unregistered source dataset")
            values = batch.column(0).to_pylist()
            audit.executemany(
                "INSERT OR IGNORE INTO links VALUES (?)", ((value,) for value in values)
            )
            n_links += len(values)
        n_orphan_links = audit.execute(
            "SELECT count(*) FROM links LEFT JOIN results ON links.id=results.id WHERE results.id IS NULL"
        ).fetchone()[0]
        n_unlinked_results = audit.execute(
            "SELECT count(*) FROM results LEFT JOIN links ON results.id=links.id WHERE links.id IS NULL"
        ).fetchone()[0]
        if n_orphan_links or n_unlinked_results:
            raise ValueError(
                f"MHC contributor coverage mismatch: {n_orphan_links} orphan link IDs, "
                f"{n_unlinked_results} results without links"
            )
        by_source = {evidence_type: {} for evidence_type in MHC_PARTITIONS.values()}
        for (evidence_type, source), count in sorted(source_counts.items()):
            by_source[evidence_type][source] = count
        return {
            "n_observations": sum(counts.values()),
            "n_measurements": sum(counts.values()),
            "n_contributor_links": n_links,
            "n_orphan_link_ids": n_orphan_links,
            "n_results_without_links": n_unlinked_results,
            "by_evidence_type": dict(sorted(counts.items())),
            "by_source": by_source,
            "by_modality_and_outcome": [
                {
                    "evidence_type": evidence_type,
                    "modality": modality,
                    "outcome": outcome,
                    "n_rows": count,
                }
                for (evidence_type, modality, outcome), count in sorted(modality_counts.items())
            ],
        }
    finally:
        audit.close()


def verify_mhc_evidence_package(directory, *, manifest_path="manifest.json"):
    """Verify an MHC manifest, including one preserved inside a later package."""
    directory = Path(directory)
    if Path(manifest_path).is_absolute() or ".." in Path(manifest_path).parts:
        raise ValueError("Invalid MHC package manifest path")
    manifest = json.loads((directory / manifest_path).read_text())
    schema_version = manifest.get("schema_version")
    if schema_version not in (1, PACKAGE_SCHEMA_VERSION):
        raise ValueError("Unsupported MHC evidence package schema")
    expected_catalog = 2 if schema_version == 1 else MHC_CATALOG_SCHEMA_VERSION
    if manifest.get("catalog_schema_version") != expected_catalog:
        raise ValueError("MHC evidence package catalog schema mismatch")
    if manifest.get("native_field_disposition") != _native_disposition():
        raise ValueError("MHC native field disposition mismatch")
    if schema_version == 2 and (
        manifest.get("positive_ms_core_contract") != MHC_POSITIVE_MS_CORE_CONTRACT
    ):
        raise ValueError("MHC positive-MS core contract mismatch")
    native_manifest_path = directory / "native" / "observations_meta.json"
    if file_digest(native_manifest_path) != manifest["source_manifest"]:
        raise ValueError("MHC source manifest mismatch")
    native_manifest = json.loads(native_manifest_path.read_text())
    if manifest.get("source_artifact_version") != native_manifest.get("artifact_version"):
        raise ValueError("MHC source artifact version mismatch")
    if manifest.get("source_registry") != _source_registry(native_manifest):
        raise ValueError("MHC source registry mismatch")
    expected_files = {
        *(f"native/{name}" for name in native_manifest["files"]),
        "native/observations_meta.json",
        "typed/mhc_observations.parquet",
        "typed/mhc_measurements.parquet",
    }
    if set(manifest["files"]) != expected_files:
        raise ValueError("MHC package inventory mismatch")
    for name, expected in native_manifest["files"].items():
        if manifest["files"][f"native/{name}"] != expected:
            raise ValueError(f"MHC source file inventory mismatch: {name}")
    for relative, expected in manifest["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid MHC package artifact path")
        if file_digest(directory / relative) != expected:
            raise ValueError(f"MHC package artifact mismatch: {relative}")
    inventory = manifest["reconciliation"]
    for name in MHC_PARTITIONS:
        if (
            pq.ParquetFile(directory / "native" / name).metadata.num_rows
            != inventory["by_evidence_type"][MHC_PARTITIONS[name]]
        ):
            raise ValueError(f"MHC native row count mismatch: {name}")
    for name, count_key in (
        ("mhc_observations.parquet", "n_observations"),
        ("mhc_measurements.parquet", "n_measurements"),
    ):
        if pq.ParquetFile(directory / "typed" / name).metadata.num_rows != inventory[count_key]:
            raise ValueError(f"MHC typed row count mismatch: {name}")
    if (
        pq.ParquetFile(directory / "typed" / "mhc_observations.parquet").schema_arrow
        != _OBSERVATION_SCHEMA
    ):
        raise ValueError("MHC typed observation schema mismatch")
    if (
        pq.ParquetFile(directory / "typed" / "mhc_measurements.parquet").schema_arrow
        != _MEASUREMENT_SCHEMA
    ):
        raise ValueError("MHC typed measurement schema mismatch")
    if (
        pq.ParquetFile(directory / "native" / "observation_contributors.parquet").metadata.num_rows
        != inventory["n_contributor_links"]
    ):
        raise ValueError("MHC contributor row count mismatch")
    return manifest


def write_mhc_evidence_package(source_dir, destination):
    """Stream a portable v4 source into one atomic, source-faithful v2 package."""
    source_dir = Path(source_dir)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"MHC evidence package already exists: {destination}")
    source_manifest, source_manifest_digest = _source_manifest(source_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        native = staging / "native"
        native.mkdir()
        for name in (*source_manifest["files"], "observations_meta.json"):
            snapshot_file(source_dir / name, native / name)
        reconciliation = _write_typed(
            native,
            staging / "typed",
            staging / "audit.sqlite",
            set(source_manifest["provenance"]["sources"]),
        )
        (staging / "audit.sqlite").unlink()
        if (
            source_manifest_digest["sha256"] == PINNED_V4_MANIFEST_SHA256
            and reconciliation["by_evidence_type"] != PINNED_V4_COUNTS
        ):
            raise ValueError("Pinned v4 MHC result counts changed")
        # A pinned source file can be replaced during a long conversion.
        _source_manifest(source_dir)
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted((*native.iterdir(), *(staging / "typed").iterdir()))
        }
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": MHC_CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "source_manifest": source_manifest_digest,
            "source_artifact_version": source_manifest["artifact_version"],
            "source_access": (
                "public_pinned_release"
                if source_manifest_digest["sha256"] == PINNED_V4_MANIFEST_SHA256
                else "provided_snapshot"
            ),
            "source_license_status": "unverified",
            "source_registry": _source_registry(source_manifest),
            "native_field_disposition": _native_disposition(),
            "positive_ms_core_contract": MHC_POSITIVE_MS_CORE_CONTRACT,
            "relation_keys": {
                "native_observation_id": "provenance_id",
                "typed_observation_id": "observation_id",
                "native_contributor_observation_id": "provenance_id",
                "native_source_record_id": "source_record_id",
            },
            "reconciliation": reconciliation,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_mhc_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)
