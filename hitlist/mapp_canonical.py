"""One source-conserving public package for captured-fragment MAPP evidence.

Adapters deliberately retain their original typed relations below
``external/<source>/typed``.  This module projects those relations into a
single layout below ``mapp/typed`` without deleting or rewriting the sources.
The projection is intentionally strict: source identity is mandatory, rows are
not deduplicated, and source-specific context lives in explicit extensions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .mapp_controlled_evidence import SCHEMAS as CONTROLLED_SCHEMAS
from .mapp_evidence import SCHEMAS as PXD009092_SCHEMAS
from .sequence_reference_evidence import SCHEMAS as REFERENCE_SCHEMAS

MAPP_SOURCE_IDS = ("pxd009092", "wolflevy2018_controlled", "pxd009438")
SOURCE_DATASETS = {
    "pxd009092": ("PXD009092",),
    "wolflevy2018_controlled": ("PXD009508", "PXD007263"),
    "pxd009438": ("PXD009438",),
}
ROOT = Path("mapp")
TYPED_ROOT = ROOT / "typed"

# The stable base layout is intentionally no wider than the first source. The
# controlled/PBMC additions are represented below as same-grain extensions.
_CORE_SOURCE_TABLES = {
    "mapp_source_record": "source_records",
    "mapp_peptide": "peptide_rows",
    "mapp_group": "groups",
    "mapp_acquisition": "channels",
    "mapp_identification": "evidence",
    "mapp_quantity_result": "quantity_results",
    "mapp_quantity_measurement": "quantity_measurements",
    "mapp_derivation_input": "derivation_inputs",
    "mapp_source_assertion": "source_assertions",
    "mapp_material_context": "material_contexts",
    "mapp_material_link": "material_links",
}
_PUBLISHER_SOURCE_TABLES = {
    "mapp_publisher_cell": "publisher_cells",
    "mapp_publisher_quantity": "publisher_quantities",
}
_REFERENCE_SOURCE_TABLES = {
    "mapp_sequence_reference": "sequence_references",
    "mapp_reference_protein": "reference_proteins",
    "mapp_reference_accession": "reference_accessions",
    "mapp_peptide_parent_occurrence": "peptide_reference_occurrences",
    "mapp_source_parent_claim": "source_parent_claims",
    "mapp_source_parent_resolution": "source_parent_resolutions",
    "mapp_peptide_reference_coverage": "peptide_reference_coverage",
}
_CHANNEL_EXTENSION_FIELDS = tuple(
    field.name
    for field in CONTROLLED_SCHEMAS["channels"]
    if field.name not in set(PXD009092_SCHEMAS["channels"].names) | {"source_dataset"}
)
_RESULT_EXTENSION_FIELDS = tuple(
    field.name
    for field in CONTROLLED_SCHEMAS["quantity_results"]
    if field.name not in set(PXD009092_SCHEMAS["quantity_results"].names) | {"source_dataset"}
)


def _schema(fields):
    """Prepend the invariant source identity to one source-native schema."""
    result = [
        pa.field("source_dataset", pa.string()),
        pa.field("source_dataset_precision", pa.string()),
    ]
    seen = {field.name for field in result}
    for field in fields:
        if field.name not in seen:
            result.append(field)
            seen.add(field.name)
    return pa.schema(result)


SCHEMAS = {
    **{
        relation: _schema(PXD009092_SCHEMAS[table])
        for relation, table in _CORE_SOURCE_TABLES.items()
    },
    **{
        relation: _schema(PXD009092_SCHEMAS[table])
        for relation, table in _PUBLISHER_SOURCE_TABLES.items()
    },
    **{
        relation: _schema(REFERENCE_SCHEMAS[table])
        for relation, table in _REFERENCE_SOURCE_TABLES.items()
    },
    "mapp_label_alias": _schema(CONTROLLED_SCHEMAS["label_aliases"]),
    "mapp_acquisition_context": _schema(
        [
            pa.field("channel_id", pa.string()),
            *[CONTROLLED_SCHEMAS["channels"].field(name) for name in _CHANNEL_EXTENSION_FIELDS],
            pa.field("extension_status", pa.string()),
        ]
    ),
    "mapp_quantity_evidence_summary": _schema(
        [
            pa.field("result_id", pa.string()),
            *[
                CONTROLLED_SCHEMAS["quantity_results"].field(name)
                for name in _RESULT_EXTENSION_FIELDS
            ],
            pa.field("extension_status", pa.string()),
        ]
    ),
    "mapp_source_table": pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_dataset_precision", pa.string()),
            ("source_adapter_id", pa.string()),
            ("canonical_relation", pa.string()),
            ("source_typed_table", pa.string()),
            ("n_source_rows", pa.int64()),
            ("source_schema_fields_json", pa.string()),
        ]
    ),
}
ARTIFACT_NAMES = {
    **_CORE_SOURCE_TABLES,
    **_PUBLISHER_SOURCE_TABLES,
    **_REFERENCE_SOURCE_TABLES,
    "mapp_label_alias": "label_aliases",
    "mapp_acquisition_context": "acquisition_context",
    "mapp_quantity_evidence_summary": "quantity_evidence_summary",
    "mapp_source_table": "source_table",
}


def _source_dataset(source_id, row):
    """Return an explicit source scope without inventing a project assignment."""
    value = row.get("source_dataset")
    if value:
        if value not in SOURCE_DATASETS[source_id]:
            raise ValueError(f"Unexpected MAPP source dataset: {source_id}:{value}")
        return value, "exact_project"
    datasets = SOURCE_DATASETS[source_id]
    if len(datasets) == 1:
        return datasets[0], "exact_project"
    # Shared controlled-panel reference tables do not carry a project key. They
    # remain queryable at their actual adapter scope rather than being copied to
    # both PXD projects or arbitrarily assigned to one.
    return source_id, "source_adapter_scope_multiple_projects"


def _source_table_path(directory, source_id, table):
    return Path(directory) / "external" / source_id / "typed" / f"{table}.parquet"


def _project_rows(directory, source_id, source_table, schema):
    path = _source_table_path(directory, source_id, source_table)
    if not path.exists():
        return []
    rows = []
    for raw in pq.read_table(path).to_pylist():
        source_dataset, precision = _source_dataset(source_id, raw)
        row = {
            "source_dataset": source_dataset,
            "source_dataset_precision": precision,
        }
        row.update({field.name: raw.get(field.name) for field in list(schema)[2:]})
        rows.append(row)
    return rows


def _extension_rows(directory, source_id, source_table, key, fields):
    path = _source_table_path(directory, source_id, source_table)
    if not path.exists():
        return []
    rows = []
    for raw in pq.read_table(path).to_pylist():
        source_dataset, precision = _source_dataset(source_id, raw)
        has_extension = any(raw.get(field) is not None for field in fields)
        rows.append(
            {
                "source_dataset": source_dataset,
                "source_dataset_precision": precision,
                key: raw[key],
                **{field: raw.get(field) for field in fields},
                "extension_status": (
                    "source_adapter_context" if has_extension else "not_reported_by_source_layout"
                ),
            }
        )
    return rows


def _write_table(directory, relation, rows):
    schema = SCHEMAS[relation]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        Path(directory) / TYPED_ROOT / f"{ARTIFACT_NAMES[relation]}.parquet",
        compression="zstd",
    )


def _specifications():
    yield from _CORE_SOURCE_TABLES.items()
    yield from _PUBLISHER_SOURCE_TABLES.items()
    yield from _REFERENCE_SOURCE_TABLES.items()


def _source_table_specifications():
    """Expose every native typed-table-to-canonical projection explicitly."""
    yield from _specifications()
    yield "mapp_label_alias", "label_aliases"
    # One native table can intentionally supply a narrow shared base relation
    # and a same-grain source-specific extension relation.
    yield "mapp_acquisition_context", "channels"
    yield "mapp_quantity_evidence_summary", "quantity_results"


def _expected_rows(directory, source_ids):
    """Project every retained source table; used by both write and verify."""
    expected = {}
    for relation, source_table in _specifications():
        expected[relation] = [
            row
            for source_id in source_ids
            for row in _project_rows(directory, source_id, source_table, SCHEMAS[relation])
        ]
    expected["mapp_label_alias"] = [
        row
        for source_id in source_ids
        for row in _project_rows(directory, source_id, "label_aliases", SCHEMAS["mapp_label_alias"])
    ]
    expected["mapp_acquisition_context"] = [
        row
        for source_id in source_ids
        for row in _extension_rows(
            directory, source_id, "channels", "channel_id", _CHANNEL_EXTENSION_FIELDS
        )
    ]
    expected["mapp_quantity_evidence_summary"] = [
        row
        for source_id in source_ids
        for row in _extension_rows(
            directory, source_id, "quantity_results", "result_id", _RESULT_EXTENSION_FIELDS
        )
    ]
    source_tables = []
    for relation, source_table in _source_table_specifications():
        for source_id in source_ids:
            path = _source_table_path(directory, source_id, source_table)
            if not path.exists():
                continue
            raw = pq.read_table(path)
            datasets = Counter(_source_dataset(source_id, row) for row in raw.to_pylist())
            if not datasets:
                datasets[_source_dataset(source_id, {})] = 0
            for (source_dataset, precision), n_source_rows in sorted(datasets.items()):
                source_tables.append(
                    {
                        "source_dataset": source_dataset,
                        "source_dataset_precision": precision,
                        "source_adapter_id": source_id,
                        "canonical_relation": relation,
                        "source_typed_table": source_table,
                        "n_source_rows": n_source_rows,
                        "source_schema_fields_json": json.dumps(raw.schema.names),
                    }
                )
    expected["mapp_source_table"] = source_tables
    return expected


def _verify_source_datasets(rows, relation):
    identities = [row["source_dataset"] for row in rows]
    if any(not value for value in identities):
        raise ValueError(f"Canonical MAPP relation lacks source dataset: {relation}")
    if relation == "mapp_quantity_measurement" and len(rows) != len(
        {row["measurement_id"] for row in rows}
    ):
        raise ValueError("Canonical MAPP measurement identity is not unique")


def write_canonical_mapp_tables(directory, *, source_ids=MAPP_SOURCE_IDS):
    """Write exact canonical projections after all source adapters have run."""
    directory = Path(directory)
    source_ids = tuple(source_ids)
    if tuple(sorted(source_ids)) != tuple(sorted(MAPP_SOURCE_IDS)):
        raise ValueError("Canonical MAPP requires all admitted captured-fragment source adapters")
    if (directory / ROOT).exists():
        raise FileExistsError(directory / ROOT)
    (directory / TYPED_ROOT).mkdir(parents=True)
    expected = _expected_rows(directory, source_ids)
    for relation, rows in expected.items():
        _verify_source_datasets(rows, relation)
        _write_table(directory, relation, rows)
    return {
        "revision": "captured_fragment_mapp_canonical_v1",
        "source_adapter_ids": list(source_ids),
        "source_datasets": sorted(
            {
                row["source_dataset"]
                for relation, rows in expected.items()
                if relation != "mapp_source_table"
                for row in rows
            }
        ),
        "relation_rows": {relation: len(rows) for relation, rows in expected.items()},
        "source_relation_rows": {
            relation: dict(sorted(Counter(row["source_dataset"] for row in rows).items()))
            for relation, rows in expected.items()
        },
    }


def verify_canonical_mapp_tables(directory, reconciliation):
    """Replay the canonical projection and reject row, field, or source drift."""
    directory = Path(directory)
    if reconciliation.get("revision") != "captured_fragment_mapp_canonical_v1":
        raise ValueError("Unsupported canonical MAPP revision")
    source_ids = tuple(reconciliation.get("source_adapter_ids", ()))
    if tuple(sorted(source_ids)) != tuple(sorted(MAPP_SOURCE_IDS)):
        raise ValueError("Canonical MAPP source adapters differ")
    expected = _expected_rows(directory, source_ids)
    for relation, rows in expected.items():
        path = directory / TYPED_ROOT / f"{ARTIFACT_NAMES[relation]}.parquet"
        if not path.is_file():
            raise ValueError(f"Missing canonical MAPP relation: {relation}")
        table = pq.read_table(path)
        if table.schema != SCHEMAS[relation] or table.to_pylist() != rows:
            raise ValueError(f"Canonical MAPP relation drift: {relation}")
        _verify_source_datasets(rows, relation)
    observed = {relation: len(rows) for relation, rows in expected.items()}
    if reconciliation.get("relation_rows") != observed:
        raise ValueError("Canonical MAPP row conservation differs")
    actual_files = {path.stem for path in (directory / TYPED_ROOT).glob("*.parquet")}
    if actual_files != set(ARTIFACT_NAMES.values()):
        raise ValueError("Canonical MAPP relation inventory differs")
    return reconciliation
