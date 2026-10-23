"""Lossless typed bulk MS, line expression, and peptide mapping evidence.

All source rows remain in the parent package. This module adds typed results,
measurement representations, source-scoped protocol claims, and explicit
mapping coverage without promoting a cell-line label to a physical specimen.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .context_evidence import verify_context_evidence_package
from .curation_yaml import load_curation_yaml
from .evidence_snapshot import snapshot_file
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 3
CATALOG_SCHEMA_VERSION = 4
PINNED_COUNTS = {
    "n_bulk_results": 2_269_288,
    "n_bulk_peptide_results": 2_047_003,
    "n_bulk_protein_results": 222_285,
    "n_expression_results": 42_909,
    "n_mapping_occurrences": 6_205_013,
    "n_eligible_input_peptides": 1_506_664,
    "n_mapped_input_peptides": 1_145_721,
}
BULK_FIELDS = (
    "evidence_kind",
    "granularity",
    "source",
    "reference",
    "pmid",
    "study_label",
    "species",
    "cell_line_name",
    "sample_label",
    "perturbation",
    "gene_symbol",
    "uniprot_acc",
    "peptide",
    "length",
    "start_position",
    "end_position",
    "log2_intensity",
    "abundance_log2_normalized",
    "abundance_percentile",
    "n_peptides",
    "n_replicates_detected",
    "instrument",
    "instrument_type",
    "fragmentation",
    "acquisition_mode",
    "labeling",
    "search_engine",
    "fdr",
    "digestion",
    "digestion_enzyme",
    "fractionation",
    "n_fractions",
    "n_fractions_in_run",
    "fractionation_ph",
    "enrichment",
    "modifications",
    "quantification",
)
EXPRESSION_FIELDS = (
    "backend",
    "source_id",
    "pmid",
    "reference",
    "study_label",
    "line_key",
    "parent_line_key",
    "granularity",
    "gene_id",
    "gene_name",
    "transcript_id",
    "profile_id",
    "tpm",
    "log2_tpm",
    "normalization",
    "quantifier",
    "species",
    "license",
)
MAPPING_FIELDS = (
    "peptide",
    "protein_id",
    "gene_name",
    "gene_id",
    "gene_biotype",
    "transcript_id",
    "is_canonical_transcript",
    "position",
    "n_flank",
    "c_flank",
    "proteome",
    "proteome_source",
)
ASSET_RECEIPT = Path(__file__).parent / "data" / "evidence_revisions" / "supporting_v4_assets.json"

_BULK_RESULT_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("source_record_id", pa.string()),
        ("native_row_index", pa.int64()),
        ("source_dataset", pa.string()),
        ("source_reference_id", pa.string()),
        ("source_pmid", pa.int64()),
        ("grain", pa.string()),
        ("endpoint", pa.string()),
        ("sample_label_raw", pa.string()),
        ("cell_line_name_raw", pa.string()),
        ("material_link_status", pa.string()),
        ("species_raw", pa.string()),
        ("peptide_sequence", pa.string()),
        ("protein_id", pa.string()),
        ("gene_symbol", pa.string()),
        ("protocol_scope_id", pa.string()),
        ("digestion_enzyme_raw", pa.string()),
        ("search_enzyme", pa.string()),
        ("search_enzyme_status", pa.string()),
        ("n_fractions_in_run", pa.int64()),
        ("enrichment_raw", pa.string()),
        ("modifications_raw", pa.string()),
        ("instrument_raw", pa.string()),
        ("acquisition_mode_raw", pa.string()),
        ("fragmentation_raw", pa.string()),
        ("n_replicates_detected", pa.int64()),
        ("n_peptides", pa.int64()),
    ]
)
_BULK_MEASUREMENT_SCHEMA = pa.schema(
    [
        ("measurement_id", pa.string()),
        ("observation_id", pa.string()),
        ("endpoint", pa.string()),
        ("representation", pa.string()),
        ("quantitative_value", pa.float64()),
        ("value_status", pa.string()),
        ("measurement_unit", pa.string()),
        ("quantity_scale", pa.string()),
        ("source_field", pa.string()),
        ("derivation", pa.string()),
    ]
)
_EXPRESSION_RESULT_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("source_record_id", pa.string()),
        ("native_row_index", pa.int64()),
        ("source_dataset", pa.string()),
        ("source_reference_id", pa.string()),
        ("grain", pa.string()),
        ("line_key", pa.string()),
        ("parent_line_key", pa.string()),
        ("profile_id_raw", pa.string()),
        ("backend", pa.string()),
        ("gene_id", pa.string()),
        ("gene_name", pa.string()),
        ("transcript_id_raw", pa.string()),
        ("normalization", pa.string()),
        ("quantifier", pa.string()),
        ("material_link_status", pa.string()),
        ("license_raw", pa.string()),
    ]
)
_EXPRESSION_MEASUREMENT_SCHEMA = pa.schema(
    [
        ("measurement_id", pa.string()),
        ("observation_id", pa.string()),
        ("endpoint", pa.string()),
        ("representation", pa.string()),
        ("quantitative_value", pa.float64()),
        ("value_status", pa.string()),
        ("measurement_unit", pa.string()),
        ("quantity_scale", pa.string()),
        ("source_field", pa.string()),
        ("derivation", pa.string()),
    ]
)
_PROTOCOL_SCHEMA = pa.schema(
    [
        ("protocol_scope_id", pa.string()),
        ("source_dataset", pa.string()),
        ("digestion_enzyme_raw", pa.string()),
        ("search_enzyme", pa.string()),
        ("search_enzyme_status", pa.string()),
        ("source_scope", pa.string()),
        ("source_locator", pa.string()),
        ("instrument_raw", pa.string()),
        ("acquisition_mode_raw", pa.string()),
        ("fragmentation_raw", pa.string()),
        ("fractionation_raw", pa.string()),
        ("quantification_raw", pa.string()),
    ]
)
_EXPRESSION_SOURCE_SCHEMA = pa.schema(
    [
        ("source_id", pa.string()),
        ("source_reference_id", pa.string()),
        ("backend", pa.string()),
        ("granularity", pa.string()),
        ("normalization", pa.string()),
        ("quantifier", pa.string()),
        ("build_status", pa.string()),
        ("license_raw", pa.string()),
        ("source_locator", pa.string()),
        ("n_materialized_rows", pa.int64()),
    ]
)
_ANCHOR_SCHEMA = pa.schema(
    [
        ("line_name", pa.string()),
        ("aliases_json", pa.string()),
        ("parent_line", pa.string()),
        ("line_family", pa.string()),
        ("expression_backend", pa.string()),
        ("expression_key", pa.string()),
        ("source_ids_json", pa.string()),
        ("material_link_precision", pa.string()),
        ("has_v4_measurement", pa.bool_()),
        ("source_locator", pa.string()),
    ]
)
_COVERAGE_SCHEMA = pa.schema(
    [
        ("peptide_sequence", pa.string()),
        ("n_mapping_occurrences", pa.int64()),
        ("mapping_status", pa.string()),
    ]
)


def _source_assets(source_root, receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("schema_version") != 1:
        raise ValueError("Unsupported supporting source asset receipt")
    if not receipt.get("assets"):
        raise ValueError("Supporting source asset receipt is empty")
    source_root = Path(source_root) if source_root is not None else None
    paths = {}
    for relative, expected in receipt["assets"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid supporting source asset path")
        if source_root is not None:
            path = source_root / path
        else:
            packaged = Path(__file__).parent / "data" / path
            if packaged.is_file():
                path = packaged
            else:
                from .downloads import packaged_or_cached

                path = packaged_or_cached(packaged, path.name)
                if path is None:
                    raise FileNotFoundError(
                        f"Pinned supporting source asset unavailable: {relative}; "
                        "fetch the Hitlist data asset or pass source_root"
                    )
        if file_digest(path) != expected:
            raise ValueError(f"Supporting source asset mismatch: {relative}")
        paths[relative] = path
    return receipt, paths


def _check_native_schemas(directory):
    native = directory / "native"
    for name, expected in (
        ("bulk_proteomics.parquet", BULK_FIELDS),
        ("line_expression.parquet", EXPRESSION_FIELDS),
        ("peptide_mappings.parquet", MAPPING_FIELDS),
    ):
        if tuple(pq.ParquetFile(native / name).schema_arrow.names) != expected:
            raise ValueError(f"Unreviewed supporting source schema: {name}")


def _field_disposition():
    return {
        "bulk_proteomics": {
            field: {
                "status": "typed"
                if field
                in {
                    "source",
                    "reference",
                    "pmid",
                    "granularity",
                    "species",
                    "cell_line_name",
                    "sample_label",
                    "gene_symbol",
                    "uniprot_acc",
                    "peptide",
                    "log2_intensity",
                    "abundance_log2_normalized",
                    "abundance_percentile",
                    "n_peptides",
                    "n_replicates_detected",
                    "digestion_enzyme",
                    "n_fractions_in_run",
                    "enrichment",
                    "modifications",
                    "instrument",
                    "acquisition_mode",
                    "fragmentation",
                }
                else "native_source_preserved",
                "native_relation": "native/bulk_proteomics.parquet",
            }
            for field in BULK_FIELDS
        },
        "line_expression": {
            field: {
                "status": "typed" if field != "study_label" else "native_source_preserved",
                "native_relation": "native/line_expression.parquet",
            }
            for field in EXPRESSION_FIELDS
        },
        "peptide_mappings": {
            field: {
                "status": "typed",
                "native_relation": "native/peptide_mappings.parquet",
            }
            for field in MAPPING_FIELDS
        },
    }


def _string(value):
    return None if value is None else str(value)


def _protocol_scope(source, enzyme):
    return f"bulk:{source}:enzyme:{enzyme}"


def _write_registries(source_dir, typed_dir, expression_counts, expression_profiles):
    bulk_sources = load_curation_yaml(source_dir / "bulk_proteomics" / "sources.yaml")["sources"]
    protocols = []
    protocol_by_key = {}
    for source in bulk_sources:
        source_id = source["source_id"]
        methods = [
            (
                source["digestion_enzyme"],
                source.get("search_enzyme"),
                "source_default",
                f"bulk_proteomics/sources.yaml::sources/{source_id}",
            ),
            *(
                (
                    arm["digestion_enzyme"],
                    arm.get("maxquant_spec"),
                    "reviewed_ancillary_digest",
                    f"bulk_proteomics/sources.yaml::sources/{source_id}/ancillary_digests/{index}",
                )
                for index, arm in enumerate(source.get("ancillary_digests", []))
            ),
        ]
        for enzyme, search_enzyme, scope, locator in methods:
            key = (source_id, enzyme)
            if key in protocol_by_key:
                raise ValueError(f"Duplicate bulk source/enzyme protocol: {key}")
            row = {
                "protocol_scope_id": _protocol_scope(*key),
                "source_dataset": source_id,
                "digestion_enzyme_raw": enzyme,
                "search_enzyme": _string(search_enzyme),
                "search_enzyme_status": "reviewed_search_spec" if search_enzyme else "unreported",
                "source_scope": scope,
                "source_locator": locator,
                "instrument_raw": _string(source.get("instrument")),
                "acquisition_mode_raw": _string(source.get("acquisition_mode")),
                "fragmentation_raw": _string(source.get("fragmentation")),
                "fractionation_raw": _string(source.get("fractionation")),
                "quantification_raw": _string(source.get("quantification")),
            }
            protocols.append(row)
            protocol_by_key[key] = row
    pq.write_table(
        pa.Table.from_pylist(protocols, schema=_PROTOCOL_SCHEMA),
        typed_dir / "bulk_protocol_scopes.parquet",
        compression="zstd",
    )

    expression_sources = load_curation_yaml(source_dir / "line_expression" / "sources.yaml")[
        "sources"
    ]
    source_rows = []
    seen_source_ids = set()
    for source in expression_sources:
        source_id = source["source_id"]
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate expression source ID: {source_id}")
        seen_source_ids.add(source_id)
        source_rows.append(
            {
                "source_id": source_id,
                "source_reference_id": _string(source.get("reference")),
                "backend": _string(source.get("backend")),
                "granularity": _string(source.get("granularity")),
                "normalization": _string(source.get("normalization")),
                "quantifier": _string(source.get("quantifier")),
                "build_status": _string(source.get("build_status")),
                "license_raw": _string(source.get("license")),
                "source_locator": f"line_expression/sources.yaml::sources/{source_id}",
                "n_materialized_rows": expression_counts.get(source_id, 0),
            }
        )
    if set(expression_counts) - seen_source_ids:
        raise ValueError("Expression source rows lack source-registry entries")
    pq.write_table(
        pa.Table.from_pylist(source_rows, schema=_EXPRESSION_SOURCE_SCHEMA),
        typed_dir / "expression_sources.parquet",
        compression="zstd",
    )

    anchors = load_curation_yaml(source_dir / "line_expression_anchors.yaml")["lines"]
    anchor_rows = []
    seen_names = set()
    for index, entry in enumerate(anchors):
        name = entry["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate line anchor: {name}")
        seen_names.add(name)
        source_ids = entry.get("source_ids") or []
        key = entry.get("expression_key")
        has_measurement = any((source_id, key) in expression_profiles for source_id in source_ids)
        anchor_rows.append(
            {
                "line_name": name,
                "aliases_json": json.dumps(entry.get("aliases") or [], sort_keys=True),
                "parent_line": _string(entry.get("parent_line")),
                "line_family": _string(entry.get("line_family")),
                "expression_backend": _string(entry.get("expression_backend")),
                "expression_key": _string(key),
                "source_ids_json": json.dumps(source_ids, sort_keys=True),
                "material_link_precision": (
                    "parent_line_surrogate" if entry.get("parent_line") else "same_named_line"
                ),
                "has_v4_measurement": has_measurement,
                "source_locator": f"line_expression_anchors.yaml::lines/{index}",
            }
        )
    pq.write_table(
        pa.Table.from_pylist(anchor_rows, schema=_ANCHOR_SCHEMA),
        typed_dir / "line_anchor_claims.parquet",
        compression="zstd",
    )
    return protocol_by_key, {
        "n_bulk_protocol_scopes": len(protocols),
        "n_expression_sources": len(source_rows),
        "n_line_anchor_claims": len(anchor_rows),
        "n_line_anchors_with_v4_measurement": sum(row["has_v4_measurement"] for row in anchor_rows),
    }


def _bulk_measurement(
    observation_id,
    endpoint,
    representation,
    value,
    unit,
    scale,
    field,
    *,
    value_status="source_reported",
    derivation="source_index_reported",
):
    return {
        "measurement_id": f"{observation_id}:{representation}",
        "observation_id": observation_id,
        "endpoint": endpoint,
        "representation": representation,
        "quantitative_value": float(value),
        "value_status": value_status,
        "measurement_unit": unit,
        "quantity_scale": scale,
        "source_field": field,
        "derivation": derivation,
    }


def _write_bulk(native, typed_dir, protocols):
    source = pq.ParquetFile(native / "bulk_proteomics.parquet")
    result_writer = pq.ParquetWriter(
        typed_dir / "bulk_results.parquet", _BULK_RESULT_SCHEMA, compression="zstd"
    )
    measurement_writer = pq.ParquetWriter(
        typed_dir / "bulk_measurements.parquet", _BULK_MEASUREMENT_SCHEMA, compression="zstd"
    )
    counts = Counter()
    by_source_grain = Counter()
    by_protocol = Counter()
    try:
        for batch in source.iter_batches(batch_size=16384):
            results = []
            measurements = []
            for row in batch.to_pylist():
                index = counts["n_bulk_results"]
                counts["n_bulk_results"] += 1
                if row["evidence_kind"] != "bulk_proteomics":
                    raise ValueError("Unexpected bulk evidence kind")
                source_id = row["source"]
                grain = row["granularity"]
                protocol = protocols.get((source_id, row["digestion_enzyme"]))
                if protocol is None:
                    raise ValueError("Bulk result lacks a reviewed source/enzyme protocol")
                observation_id = f"bulk:v4:{index}"
                if grain == "peptide":
                    if (
                        not row["peptide"]
                        or row["log2_intensity"] is not None
                        or row["abundance_log2_normalized"] is not None
                        or not row["n_replicates_detected"]
                    ):
                        raise ValueError("Invalid bulk peptide-detection source row")
                    endpoint = "bulk_peptide_detection"
                    counts["n_bulk_peptide_results"] += 1
                    measurements.append(
                        _bulk_measurement(
                            observation_id,
                            endpoint,
                            "observed_within_search",
                            1,
                            "binary",
                            "scoped_detection",
                            "peptide",
                        )
                    )
                elif grain == "protein":
                    if not row["uniprot_acc"]:
                        raise ValueError("Bulk protein row lacks source protein key")
                    endpoint = "bulk_protein_abundance"
                    counts["n_bulk_protein_results"] += 1
                    if source_id == "Bekker-Jensen_2017":
                        field = "log2_intensity"
                        representation = "bekker_jensen_log2_intensity"
                        unit = "log2_intensity"
                        scale = "label_free_source_index"
                    elif source_id == "CCLE_Nusinow_2020":
                        field = "abundance_log2_normalized"
                        representation = "ccle_log2_normalized_abundance"
                        unit = "log2_relative_panel_median"
                        scale = "tmt_panel_relative"
                    else:
                        raise ValueError(f"Unreviewed bulk protein source: {source_id}")
                    if row[field] is None:
                        raise ValueError("Bulk protein row lacks its source-specific quantity")
                    measurements.append(
                        _bulk_measurement(
                            observation_id, endpoint, representation, row[field], unit, scale, field
                        )
                    )
                    if row["abundance_percentile"] is not None:
                        measurements.append(
                            _bulk_measurement(
                                observation_id,
                                "protein_abundance_percentile",
                                "source_derived_percentile",
                                row["abundance_percentile"],
                                "percentile",
                                "within_source_rank",
                                "abundance_percentile",
                                value_status="derived_in_source_index",
                                derivation="source_index_percentile",
                            )
                        )
                else:
                    raise ValueError(f"Unreviewed bulk grain: {grain}")
                results.append(
                    {
                        "observation_id": observation_id,
                        "source_record_id": f"bulk_proteomics.parquet:{index}",
                        "native_row_index": index,
                        "source_dataset": source_id,
                        "source_reference_id": _string(row["reference"]),
                        "source_pmid": row["pmid"],
                        "grain": grain,
                        "endpoint": endpoint,
                        "sample_label_raw": _string(row["sample_label"]),
                        "cell_line_name_raw": _string(row["cell_line_name"]),
                        "material_link_status": "cell_line_label_only",
                        "species_raw": _string(row["species"]),
                        "peptide_sequence": _string(row["peptide"]),
                        "protein_id": _string(row["uniprot_acc"]),
                        "gene_symbol": _string(row["gene_symbol"]),
                        "protocol_scope_id": protocol["protocol_scope_id"],
                        "digestion_enzyme_raw": _string(row["digestion_enzyme"]),
                        "search_enzyme": protocol["search_enzyme"],
                        "search_enzyme_status": protocol["search_enzyme_status"],
                        "n_fractions_in_run": row["n_fractions_in_run"],
                        "enrichment_raw": _string(row["enrichment"]),
                        "modifications_raw": _string(row["modifications"]),
                        "instrument_raw": _string(row["instrument"]),
                        "acquisition_mode_raw": _string(row["acquisition_mode"]),
                        "fragmentation_raw": _string(row["fragmentation"]),
                        "n_replicates_detected": row["n_replicates_detected"],
                        "n_peptides": row["n_peptides"],
                    }
                )
                by_source_grain[(source_id, grain)] += 1
                by_protocol[
                    (
                        source_id,
                        grain,
                        row["digestion_enzyme"],
                        row["n_fractions_in_run"],
                        row["enrichment"],
                    )
                ] += 1
            result_writer.write_table(pa.Table.from_pylist(results, schema=_BULK_RESULT_SCHEMA))
            measurement_writer.write_table(
                pa.Table.from_pylist(measurements, schema=_BULK_MEASUREMENT_SCHEMA)
            )
            counts["n_bulk_measurements"] += len(measurements)
    finally:
        result_writer.close()
        measurement_writer.close()
    return {
        **dict(counts),
        "bulk_by_source_grain": [
            {"source": source, "grain": grain, "n_results": n_rows}
            for (source, grain), n_rows in sorted(by_source_grain.items())
        ],
        "bulk_by_protocol_scope": [
            {
                "source": source,
                "grain": grain,
                "digestion_enzyme_raw": enzyme,
                "n_fractions_in_run": n_fractions,
                "enrichment_raw": enrichment,
                "n_results": n_rows,
            }
            for (source, grain, enzyme, n_fractions, enrichment), n_rows in sorted(
                by_protocol.items(), key=lambda item: str(item[0])
            )
        ],
    }


def _expression_measurement(observation_id, representation, value, status, field, derivation):
    return {
        "measurement_id": f"{observation_id}:{representation}",
        "observation_id": observation_id,
        "endpoint": "gene_expression",
        "representation": representation,
        "quantitative_value": float(value),
        "value_status": status,
        "measurement_unit": "TPM" if representation == "tpm" else "log2(TPM+1)",
        "quantity_scale": "linear_tpm" if representation == "tpm" else "log2_tpm_plus_1",
        "source_field": field,
        "derivation": derivation,
    }


def _write_expression(native, typed_dir):
    source = pq.ParquetFile(native / "line_expression.parquet")
    result_writer = pq.ParquetWriter(
        typed_dir / "expression_results.parquet", _EXPRESSION_RESULT_SCHEMA, compression="zstd"
    )
    measurement_writer = pq.ParquetWriter(
        typed_dir / "expression_measurements.parquet",
        _EXPRESSION_MEASUREMENT_SCHEMA,
        compression="zstd",
    )
    counts = Counter()
    by_source = Counter()
    profiles = set()
    try:
        for batch in source.iter_batches(batch_size=16384):
            results = []
            measurements = []
            for row in batch.to_pylist():
                index = counts["n_expression_results"]
                counts["n_expression_results"] += 1
                if row["granularity"] != "gene" or row["transcript_id"] is not None:
                    raise ValueError("Unreviewed v4 expression grain")
                if (
                    not row["source_id"]
                    or not row["line_key"]
                    or not (row["gene_id"] or row["gene_name"])
                ):
                    raise ValueError("Expression result lacks required source or molecular key")
                if row["tpm"] is None or row["log2_tpm"] is None:
                    raise ValueError("Expression result lacks the retained representations")
                normalization = row["normalization"]
                if normalization == "tpm":
                    reported = "tpm"
                elif normalization == "log2_tpm_plus_1":
                    reported = "log2_tpm"
                else:
                    raise ValueError(f"Unreviewed v4 expression normalization: {normalization}")
                observation_id = f"expression:v4:{index}"
                results.append(
                    {
                        "observation_id": observation_id,
                        "source_record_id": f"line_expression.parquet:{index}",
                        "native_row_index": index,
                        "source_dataset": _string(row["source_id"]),
                        "source_reference_id": _string(row["reference"]),
                        "grain": "gene_profile",
                        "line_key": _string(row["line_key"]),
                        "parent_line_key": _string(row["parent_line_key"]),
                        "profile_id_raw": _string(row["profile_id"]),
                        "backend": _string(row["backend"]),
                        "gene_id": _string(row["gene_id"]),
                        "gene_name": _string(row["gene_name"]),
                        "transcript_id_raw": _string(row["transcript_id"]),
                        "normalization": normalization,
                        "quantifier": _string(row["quantifier"]),
                        "material_link_status": "line_profile_not_paired_specimen",
                        "license_raw": _string(row["license"]),
                    }
                )
                for field, derivation in (
                    ("tpm", "source_index_reported" if reported == "tpm" else "inverse_log2_tpm"),
                    (
                        "log2_tpm",
                        "source_index_reported" if reported == "log2_tpm" else "log2_tpm_plus_1",
                    ),
                ):
                    measurements.append(
                        _expression_measurement(
                            observation_id,
                            field,
                            row[field],
                            "source_reported" if field == reported else "derived_in_source_index",
                            field,
                            derivation,
                        )
                    )
                by_source[row["source_id"]] += 1
                profiles.add((row["source_id"], row["line_key"]))
            result_writer.write_table(
                pa.Table.from_pylist(results, schema=_EXPRESSION_RESULT_SCHEMA)
            )
            measurement_writer.write_table(
                pa.Table.from_pylist(measurements, schema=_EXPRESSION_MEASUREMENT_SCHEMA)
            )
            counts["n_expression_measurements"] += len(measurements)
    finally:
        result_writer.close()
        measurement_writer.close()
    return (
        {
            **dict(counts),
            "expression_by_source": [
                {"source": source_id, "n_results": n_rows}
                for source_id, n_rows in sorted(by_source.items())
            ],
            "n_expression_profiles": len(profiles),
            "n_expression_transcript_results": 0,
        },
        by_source,
        profiles,
    )


def _insert_mapping_counts(connection, peptides):
    counts = Counter(peptides)
    connection.executemany(
        "INSERT INTO peptides (peptide, n_mappings, seen_any, seen_eligible) "
        "VALUES (?, ?, 0, 0) ON CONFLICT(peptide) DO UPDATE SET "
        "n_mappings = n_mappings + excluded.n_mappings",
        counts.items(),
    )


def _write_mappings(native, typed_dir, audit_path):
    source = pq.ParquetFile(native / "peptide_mappings.parquet")
    schema = pa.schema(
        [
            pa.field("mapping_id", pa.string()),
            pa.field("native_row_index", pa.int64()),
            *source.schema_arrow,
        ]
    )
    writer = pq.ParquetWriter(
        typed_dir / "peptide_protein_occurrences.parquet", schema, compression="zstd"
    )
    connection = sqlite3.connect(audit_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE peptides (peptide TEXT PRIMARY KEY, n_mappings INTEGER NOT NULL, "
        "seen_any INTEGER NOT NULL, seen_eligible INTEGER NOT NULL) WITHOUT ROWID"
    )
    n_mappings = 0
    try:
        try:
            for batch in source.iter_batches(batch_size=65536):
                sequences = batch.column(batch.schema.get_field_index("peptide")).to_pylist()
                if any(not peptide for peptide in sequences):
                    raise ValueError("Mapping occurrence lacks a peptide sequence")
                _insert_mapping_counts(connection, sequences)
                count = batch.num_rows
                table = pa.Table.from_batches([batch])
                table = table.add_column(
                    0,
                    "mapping_id",
                    pa.array(f"mapping:v4:{n_mappings + i}" for i in range(count)),
                )
                table = table.add_column(
                    1,
                    "native_row_index",
                    pa.array(range(n_mappings, n_mappings + count), type=pa.int64()),
                )
                writer.write_table(table.cast(schema))
                n_mappings += count
        finally:
            writer.close()
        for name in ("observations.parquet", "binding.parquet", "other_assays.parquet"):
            assay_source = pq.ParquetFile(native / name)
            for batch in assay_source.iter_batches(
                batch_size=65536, columns=["peptide", "is_non_peptide_ligand"]
            ):
                peptide_col = batch.column(0).to_pylist()
                non_peptide_col = batch.column(1).to_pylist()
                statuses = {}
                for peptide, non_peptide in zip(peptide_col, non_peptide_col):
                    if peptide:
                        statuses[peptide] = statuses.get(peptide, False) or not non_peptide
                connection.executemany(
                    "INSERT INTO peptides (peptide, n_mappings, seen_any, seen_eligible) "
                    "VALUES (?, 0, 1, ?) ON CONFLICT(peptide) DO UPDATE SET "
                    "seen_any = 1, seen_eligible = max(seen_eligible, excluded.seen_eligible)",
                    ((peptide, int(eligible)) for peptide, eligible in statuses.items()),
                )
        counts = dict(
            zip(
                (
                    "n_raw_input_peptides",
                    "n_eligible_input_peptides",
                    "n_mapped_input_peptides",
                    "n_unmapped_input_peptides",
                    "n_mapping_peptides_without_eligible_input",
                ),
                connection.execute(
                    "SELECT sum(seen_any), sum(seen_eligible), "
                    "sum(seen_eligible AND n_mappings > 0), "
                    "sum(seen_eligible AND n_mappings = 0), "
                    "sum(n_mappings > 0 AND NOT seen_eligible) FROM peptides"
                ).fetchone(),
            )
        )
        if counts["n_mapping_peptides_without_eligible_input"]:
            raise ValueError("Mapping occurrence lacks eligible source peptide")
        coverage_writer = pq.ParquetWriter(
            typed_dir / "peptide_mapping_coverage.parquet", _COVERAGE_SCHEMA, compression="zstd"
        )
        try:
            cursor = connection.execute(
                "SELECT peptide, n_mappings FROM peptides WHERE seen_eligible = 1 ORDER BY peptide"
            )
            while rows := cursor.fetchmany(65536):
                coverage_writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "peptide_sequence": peptide,
                                "n_mapping_occurrences": n_rows,
                                "mapping_status": "mapped" if n_rows else "unmapped",
                            }
                            for peptide, n_rows in rows
                        ],
                        schema=_COVERAGE_SCHEMA,
                    )
                )
        finally:
            coverage_writer.close()
        return {
            "n_mapping_occurrences": n_mappings,
            **counts,
            "mapping_occurrence_schema": schema.to_string(),
        }
    finally:
        connection.close()


def _source_registry(reconciliation, files):
    return {
        "bulk_v4": {
            "native_artifact": "native/bulk_proteomics.parquet",
            "n_results": reconciliation["n_bulk_results"],
            "source_asset_receipt_sha256": files["assets/receipt.json"]["sha256"],
            "license_status": "unverified",
        },
        "line_expression_v4": {
            "native_artifact": "native/line_expression.parquet",
            "n_results": reconciliation["n_expression_results"],
            "source_asset_receipt_sha256": files["assets/receipt.json"]["sha256"],
            "license_status": "source_specific_in_bundled_registry",
        },
        "peptide_mapping_v4": {
            "native_artifact": "native/peptide_mappings.parquet",
            "n_relations": reconciliation["n_mapping_occurrences"],
            "reference_contract": "native/peptide_mappings_meta.json",
            "license_status": "unverified",
        },
    }


def verify_supporting_evidence_package(directory, *, manifest_path="manifest.json"):
    """Verify a supporting generation, including one embedded in its successor."""
    directory = Path(directory)
    manifest_location = Path(manifest_path)
    if manifest_location.is_absolute() or ".." in manifest_location.parts:
        raise ValueError("Invalid supporting package manifest path")
    manifest = json.loads((directory / manifest_location).read_text())
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported supporting evidence package schema")
    if manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported supporting evidence catalog schema")
    if manifest.get("field_disposition") != _field_disposition():
        raise ValueError("Supporting source field disposition mismatch")
    files = manifest.get("files", {})
    if manifest_location == Path("manifest.json"):
        actual_files = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"
        }
        if set(files) != actual_files:
            raise ValueError("Supporting package file inventory mismatch")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_digest(directory / path) != expected:
            raise ValueError(f"Supporting package artifact mismatch: {relative}")
    if files["manifests/context.json"]["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("Supporting parent context manifest mismatch")
    parent = verify_context_evidence_package(directory, manifest_path="manifests/context.json")
    if parent["catalog_schema_version"] != 3:
        raise ValueError("Unsupported parent context catalog")
    receipt, _ = _source_assets(directory / "assets", directory / "assets" / "receipt.json")
    if manifest.get("source_asset_receipt_sha256") != files["assets/receipt.json"]["sha256"]:
        raise ValueError("Supporting source asset receipt mismatch")
    if receipt.get("generation") != manifest.get("source_asset_generation"):
        raise ValueError("Supporting source asset generation mismatch")
    _check_native_schemas(directory)
    inventory = manifest["reconciliation"]
    if manifest.get("source_registry") != _source_registry(inventory, files):
        raise ValueError("Supporting source registry mismatch")
    native = directory / "native"
    for name, key in (
        ("bulk_proteomics.parquet", "n_bulk_results"),
        ("line_expression.parquet", "n_expression_results"),
        ("peptide_mappings.parquet", "n_mapping_occurrences"),
    ):
        if pq.ParquetFile(native / name).metadata.num_rows != inventory[key]:
            raise ValueError(f"Supporting native row count mismatch: {name}")
    required = {
        "bulk_results.parquet": (inventory["n_bulk_results"], _BULK_RESULT_SCHEMA),
        "bulk_measurements.parquet": (
            inventory["n_bulk_measurements"],
            _BULK_MEASUREMENT_SCHEMA,
        ),
        "expression_results.parquet": (
            inventory["n_expression_results"],
            _EXPRESSION_RESULT_SCHEMA,
        ),
        "expression_measurements.parquet": (
            inventory["n_expression_measurements"],
            _EXPRESSION_MEASUREMENT_SCHEMA,
        ),
        "bulk_protocol_scopes.parquet": (
            inventory["n_bulk_protocol_scopes"],
            _PROTOCOL_SCHEMA,
        ),
        "expression_sources.parquet": (
            inventory["n_expression_sources"],
            _EXPRESSION_SOURCE_SCHEMA,
        ),
        "line_anchor_claims.parquet": (
            inventory["n_line_anchor_claims"],
            _ANCHOR_SCHEMA,
        ),
        "peptide_mapping_coverage.parquet": (
            inventory["n_eligible_input_peptides"],
            _COVERAGE_SCHEMA,
        ),
    }
    mapping_schema = pa.schema(
        [
            pa.field("mapping_id", pa.string()),
            pa.field("native_row_index", pa.int64()),
            *pq.ParquetFile(native / "peptide_mappings.parquet").schema_arrow,
        ]
    )
    required["peptide_protein_occurrences.parquet"] = (
        inventory["n_mapping_occurrences"],
        mapping_schema,
    )
    for name, (n_rows, schema) in required.items():
        typed = pq.ParquetFile(directory / "typed" / name)
        if typed.metadata.num_rows != n_rows or typed.schema_arrow != schema:
            raise ValueError(f"Supporting typed relation mismatch: {name}")
    if (
        inventory["n_bulk_peptide_results"] + inventory["n_bulk_protein_results"]
        != inventory["n_bulk_results"]
        or inventory["n_mapped_input_peptides"] + inventory["n_unmapped_input_peptides"]
        != inventory["n_eligible_input_peptides"]
        or inventory["n_mapping_peptides_without_eligible_input"]
    ):
        raise ValueError("Supporting result or mapping conservation mismatch")
    return manifest


def write_supporting_evidence_package(
    context_package,
    destination,
    *,
    source_root=None,
    source_asset_receipt=None,
):
    """Atomically add typed non-MHC measurements and mappings to one generation."""
    context_package = Path(context_package)
    destination = Path(destination)
    source_root = Path(source_root) if source_root is not None else None
    source_asset_receipt = Path(source_asset_receipt or ASSET_RECEIPT)
    if destination.exists():
        raise FileExistsError(f"Supporting evidence package already exists: {destination}")
    parent = verify_context_evidence_package(context_package)
    _check_native_schemas(context_package)
    receipt, source_paths = _source_assets(source_root, source_asset_receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent["files"]:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(context_package / relative, target)
        (staging / "manifests").mkdir(exist_ok=True)
        snapshot_file(context_package / "manifest.json", staging / "manifests" / "context.json")
        asset_dir = staging / "assets"
        asset_dir.mkdir()
        snapshot_file(source_asset_receipt, asset_dir / "receipt.json")
        for relative in receipt["assets"]:
            target = asset_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(source_paths[relative], target)
        typed = staging / "typed"
        native = staging / "native"
        expression, expression_counts, expression_profiles = _write_expression(native, typed)
        protocols, registry = _write_registries(
            asset_dir, typed, expression_counts, expression_profiles
        )
        bulk = _write_bulk(native, typed, protocols)
        audit_path = staging / "mapping_audit.sqlite"
        mappings = _write_mappings(native, typed, audit_path)
        audit_path.unlink()
        reconciliation = {**bulk, **expression, **registry, **mappings}
        if (
            parent["cohort_manifest_sha256"]
            == "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"
        ):
            for key, expected in PINNED_COUNTS.items():
                if reconciliation[key] != expected:
                    raise ValueError(f"Pinned supporting source count changed: {key}")
        verify_context_evidence_package(context_package)
        _source_assets(source_root, source_asset_receipt)
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
            "parent_manifest_sha256": files["manifests/context.json"]["sha256"],
            "source_asset_receipt_sha256": files["assets/receipt.json"]["sha256"],
            "source_asset_generation": receipt.get("generation"),
            "source_registry": _source_registry(reconciliation, files),
            "field_disposition": _field_disposition(),
            "relation_keys": {
                "bulk_observation_id": "observation_id",
                "expression_observation_id": "observation_id",
                "mapping_id": "mapping_id",
                "mapping_to_mhc": "peptide_sequence",
                "mapping_to_bulk_peptide": "peptide_sequence",
                "physical_material_join": "unresolved",
                "surrogate_line_anchor": "line_name",
            },
            "reconciliation": reconciliation,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_supporting_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)
