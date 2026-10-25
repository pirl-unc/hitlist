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

from .goldberg_mapp import SCHEMAS as GOLDBERG_SCHEMAS
from .mapp_controlled_evidence import SCHEMAS as CONTROLLED_SCHEMAS
from .mapp_evidence import SCHEMAS as PXD009092_SCHEMAS
from .sequence_reference_evidence import SCHEMAS as REFERENCE_SCHEMAS

MAPP_SOURCE_IDS = ("pxd009092", "wolflevy2018_controlled", "pxd009438", "goldberg2025")
V1_MAPP_SOURCE_IDS = ("pxd009092", "wolflevy2018_controlled", "pxd009438")
SOURCE_DATASETS = {
    "pxd009092": ("PXD009092",),
    "wolflevy2018_controlled": ("PXD009508", "PXD007263"),
    "pxd009438": ("PXD009438",),
    "goldberg2025": ("PXD052314",),
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
_GOLDBERG_SOURCE_TABLES = {
    "mapp_source_record": "source_records",
    "mapp_peptide": "peptides",
    "mapp_group": "groups",
    "mapp_acquisition": "acquisitions",
    "mapp_identification": "identifications",
    "mapp_quantity_result": "quantities",
    "mapp_quantity_measurement": "quantities",
    "mapp_source_assertion": "assertions",
    "mapp_material_context": "material_contexts",
    "mapp_material_link": "material_links",
    "mapp_publisher_cell": "publisher_cells",
    "mapp_publisher_quantity": "publisher_quantities",
}
_GOLDBERG_EXTENSION_SOURCE_TABLES = {
    "mapp_source_asset": "assets",
    "mapp_source_record_source_detail": "source_records",
    "mapp_acquisition_source_detail": "acquisitions",
    "mapp_group_source_detail": "groups",
    "mapp_source_reference": "references",
    "mapp_peptide_source_detail": "peptides",
    "mapp_peptide_form": "forms",
    "mapp_identification_source_detail": "identifications",
    "mapp_search_quantity_detail": "quantities",
    "mapp_source_assertion_source_detail": "assertions",
    "mapp_publisher_cell_source_detail": "publisher_cells",
    "mapp_publisher_quantity_source_detail": "publisher_quantities",
    "mapp_publisher_reconciliation": "publisher_links",
    "mapp_pooled_quantity_slot": "pooled_slots",
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

# Revision one is retained for source-package verification. The expanded
# revision keeps those base relation schemas intact and adds source-qualified
# Goldberg extensions rather than rewriting historical source packages.
V1_SCHEMAS = dict(SCHEMAS)
V1_ARTIFACT_NAMES = dict(ARTIFACT_NAMES)

SCHEMAS.update(
    {
        relation: _schema(GOLDBERG_SCHEMAS[source_table])
        for relation, source_table in _GOLDBERG_EXTENSION_SOURCE_TABLES.items()
    }
)
ARTIFACT_NAMES.update(
    {
        "mapp_source_asset": "source_asset",
        "mapp_source_record_source_detail": "source_record_source_detail",
        "mapp_acquisition_source_detail": "acquisition_source_detail",
        "mapp_group_source_detail": "group_source_detail",
        "mapp_source_reference": "source_reference",
        "mapp_peptide_source_detail": "peptide_source_detail",
        "mapp_peptide_form": "peptide_form",
        "mapp_identification_source_detail": "identification_source_detail",
        "mapp_search_quantity_detail": "search_quantity_detail",
        "mapp_source_assertion_source_detail": "source_assertion_source_detail",
        "mapp_publisher_cell_source_detail": "publisher_cell_source_detail",
        "mapp_publisher_quantity_source_detail": "publisher_quantity_source_detail",
        "mapp_publisher_reconciliation": "publisher_reconciliation",
        "mapp_pooled_quantity_slot": "pooled_quantity_slot",
    }
)


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


def _source_table_for(source_id, relation):
    if source_id == "goldberg2025":
        return (_GOLDBERG_SOURCE_TABLES | _GOLDBERG_EXTENSION_SOURCE_TABLES).get(relation)
    return (_CORE_SOURCE_TABLES | _PUBLISHER_SOURCE_TABLES | _REFERENCE_SOURCE_TABLES).get(relation)


def _source_integer(value):
    """Expose a source integer only when the literal is exactly integral."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _goldberg_peptide_id(native_id):
    from .goldberg_mapp import _id

    return _id("peptide", str(native_id)) if native_id is not None else None


def _goldberg_core_row(relation, raw):
    """Project matching-grain Goldberg facts into the existing base relations.

    Any column not represented by a matching base concept is retained verbatim
    in a source-detail extension. In particular, no source literal is coerced
    into a specimen, aliquot, inferred condition or quantitative aggregate.
    """
    if relation in ("mapp_material_context", "mapp_material_link"):
        # The material graph is already shared verbatim by both source layouts.
        return raw
    if relation == "mapp_source_record":
        return {
            "source_record_id": raw["source_record_id"],
            "asset_path": raw["source_file"],
            "fields_json": raw["source_row_json"],
            "row_1based": raw["source_row_index"],
        }
    if relation == "mapp_peptide":
        return {
            "peptide_row_id": raw["peptide_id"],
            "source_record_id": raw["source_record_id"],
            "peptide_id_raw": raw["native_id"],
            "sequence_raw": raw["peptide_sequence"],
            "pep_raw": raw["confidence_raw"],
            "pep_status": raw["confidence_status"],
            "parent_labels_raw": raw["parent_label_raw"],
            "leading_parent_raw": None,
            "group_ids_raw": raw["source_group_ids_raw"],
            "n_flank_raw": raw["flank_n_raw"],
            "c_flank_raw": raw["flank_c_raw"],
            "measured_pool": None,
            "material_link_precision": None,
            "start_1based_reported": _source_integer(raw["start_raw"]),
            "end_1based_reported": _source_integer(raw["end_raw"]),
            "pep": raw["posterior_error_probability"],
            "reverse": None,
            "potential_contaminant": None,
        }
    if relation == "mapp_group":
        return {
            "group_record_id": raw["group_id"],
            "source_record_id": raw["source_record_id"],
            "group_id_raw": raw["native_id"],
            "protein_ids_raw": raw["parent_label_raw"],
            "peptide_ids_raw": raw["source_peptide_ids_raw"],
            "status": None,
            "reverse": None,
            "potential_contaminant": None,
        }
    if relation == "mapp_acquisition":
        return {
            "channel_id": raw["acquisition_id"],
            "source_record_id": raw["source_record_id"],
            "experiment_raw": raw["assay_name_raw"],
            "raw_file": raw["raw_file"],
            "condition": raw["sample_label_raw"],
            "condition_role": raw["condition_status"],
            "measured_species": None,
            "measured_cell": raw["cell_line_name"],
            "measured_pool": raw["measured_pool"],
            "material_link_precision": raw["material_link_precision"],
            "search_enzyme_mode_raw": None,
            "replicate_label": None,
        }
    if relation == "mapp_identification":
        return {
            "evidence_id": raw["identification_id"],
            "source_record_id": raw["source_record_id"],
            "peptide_row_id": raw["peptide_id"],
            "channel_id": raw["acquisition_id"],
            "sequence_raw": raw["peptide_sequence"],
            "modified_sequence_raw": raw["modified_sequence_raw"],
            "modifications_raw": raw["peptide_modifications_raw"],
            "type_raw": raw["identification_type"],
            "group_ids_raw": raw["source_group_ids_raw"],
            "pep_raw": raw["confidence_raw"],
            "pep_status": raw["confidence_status"],
            "intensity_raw": None,
            "intensity_status": "separate_source_quantity_relation",
            "msms_count": None,
            "intensity": None,
            "pep": raw["posterior_error_probability"],
            "reverse": None,
            "potential_contaminant": None,
        }
    if relation == "mapp_quantity_result":
        return {
            "result_id": raw["measurement_id"],
            "peptide_row_id": raw["peptide_id"],
            "channel_id": raw["acquisition_id"],
            "measured_pool": None,
            "endpoint": raw["quantity_kind"],
            "material_link_precision": None,
        }
    if relation == "mapp_quantity_measurement":
        return {
            "measurement_id": raw["measurement_id"],
            "result_id": raw["measurement_id"],
            "peptide_row_id": raw["peptide_id"],
            "representation": raw["quantity_kind"],
            "value_status": raw["value_status"],
            "unit": raw["measurement_unit"],
            "quantity_scale": raw["quantity_scale"],
            "source_locator": raw["source_cell"],
            "derivation_rule": None,
            "value": raw["quantitative_value"],
            "recomputed_value": None,
            "absolute_replay_error": None,
        }
    if relation == "mapp_source_assertion":
        return {
            "assertion_id": raw["claim_id"],
            "subject_id": raw["subject_id"],
            "predicate": raw["assertion_predicate"],
            "value_raw": raw["assertion_value"],
            "normalized_value": None,
            "assertion_status": raw["assertion_status"],
            "source_locator": raw["source_locator"],
            "scope": raw["source_scope"],
        }
    if relation == "mapp_publisher_cell":
        return {
            "cell_id": raw["cell_id"],
            "asset_path": raw["source_file"],
            "sheet_name": raw["sheet_name"],
            "cell_ref": raw["source_cell"],
            "excel_type": raw["cell_type_raw"],
            "value_text": raw["cell_value_raw"],
            "row_1based": raw["source_row_index"],
            "column_1based": raw["source_column_index"],
            "value_number": None,
        }
    if relation == "mapp_publisher_quantity":
        return {
            "measurement_id": raw["measurement_id"],
            "peptide_row_id": _goldberg_peptide_id(raw["native_id"]),
            "source_cell": raw["source_cell"],
            "representation": raw["quantity_kind"],
            "condition_header_raw": raw["source_field"],
            "condition": None,
            "value_status": raw["value_status"],
            "unit": raw["measurement_unit"],
            "derivation_rule": None,
            "value": raw["quantitative_value"],
            "recomputed_value": None,
            "absolute_replay_error": None,
        }
    raise ValueError(f"No Goldberg core projection for {relation}")


def _source_table_path(directory, source_id, table):
    return Path(directory) / "external" / source_id / "typed" / f"{table}.parquet"


def _project_rows(directory, source_id, relation, source_table, schema):
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
        values = (
            _goldberg_core_row(relation, raw)
            if source_id == "goldberg2025" and relation in _GOLDBERG_SOURCE_TABLES
            else raw
        )
        row.update({field.name: values.get(field.name) for field in list(schema)[2:]})
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


def _unreported_extension_rows(directory, source_table, source_key, key, fields):
    """Make a same-grain extension joinable while preserving its absence."""
    rows = []
    path = _source_table_path(directory, "goldberg2025", source_table)
    if not path.exists():
        return rows
    for raw in pq.read_table(path).to_pylist():
        source_dataset, precision = _source_dataset("goldberg2025", raw)
        rows.append(
            {
                "source_dataset": source_dataset,
                "source_dataset_precision": precision,
                key: raw[source_key],
                **dict.fromkeys(fields),
                "extension_status": "not_reported_by_source_layout",
            }
        )
    return rows


def _write_table(directory, relation, rows, schemas=SCHEMAS, artifact_names=ARTIFACT_NAMES):
    schema = schemas[relation]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        Path(directory) / TYPED_ROOT / f"{artifact_names[relation]}.parquet",
        compression="zstd",
    )


def _base_specifications():
    yield from _CORE_SOURCE_TABLES.items()
    yield from _PUBLISHER_SOURCE_TABLES.items()
    yield from _REFERENCE_SOURCE_TABLES.items()


def _is_v1(source_ids):
    return tuple(sorted(source_ids)) == tuple(sorted(V1_MAPP_SOURCE_IDS))


def _projected_source_tables(source_ids, *, include_extensions):
    """Yield every canonical-relation/native-table projection for its source."""
    for source_id in source_ids:
        for relation, default_table in _base_specifications():
            source_table = _source_table_for(source_id, relation) or default_table
            yield source_id, relation, source_table
        if source_id != "goldberg2025":
            yield source_id, "mapp_label_alias", "label_aliases"
            # One native table can intentionally supply a narrow shared base
            # relation and a same-grain source-specific extension relation.
            yield source_id, "mapp_acquisition_context", "channels"
            yield source_id, "mapp_quantity_evidence_summary", "quantity_results"
        else:
            yield source_id, "mapp_acquisition_context", "acquisitions"
            yield source_id, "mapp_quantity_evidence_summary", "quantities"
        if include_extensions and source_id == "goldberg2025":
            for relation, source_table in _GOLDBERG_EXTENSION_SOURCE_TABLES.items():
                yield source_id, relation, source_table


def _expected_rows(directory, source_ids):
    """Project every retained source table; used by both write and verify."""
    v1 = _is_v1(source_ids)
    schemas = V1_SCHEMAS if v1 else SCHEMAS
    expected = {}
    for relation, _source_table in _base_specifications():
        expected[relation] = [
            row
            for source_id in source_ids
            for row in _project_rows(
                directory,
                source_id,
                relation,
                _source_table_for(source_id, relation) or _source_table,
                schemas[relation],
            )
        ]
    expected["mapp_label_alias"] = [
        row
        for source_id in source_ids
        if source_id != "goldberg2025"
        for row in _project_rows(
            directory,
            source_id,
            "mapp_label_alias",
            "label_aliases",
            schemas["mapp_label_alias"],
        )
    ]
    expected["mapp_acquisition_context"] = [
        row
        for source_id in source_ids
        if source_id != "goldberg2025"
        for row in _extension_rows(
            directory, source_id, "channels", "channel_id", _CHANNEL_EXTENSION_FIELDS
        )
    ]
    if not v1:
        expected["mapp_acquisition_context"].extend(
            _unreported_extension_rows(
                directory,
                "acquisitions",
                "acquisition_id",
                "channel_id",
                _CHANNEL_EXTENSION_FIELDS,
            )
        )
    expected["mapp_quantity_evidence_summary"] = [
        row
        for source_id in source_ids
        if source_id != "goldberg2025"
        for row in _extension_rows(
            directory, source_id, "quantity_results", "result_id", _RESULT_EXTENSION_FIELDS
        )
    ]
    if not v1:
        expected["mapp_quantity_evidence_summary"].extend(
            _unreported_extension_rows(
                directory,
                "quantities",
                "measurement_id",
                "result_id",
                _RESULT_EXTENSION_FIELDS,
            )
        )
    if not v1:
        for relation, source_table in _GOLDBERG_EXTENSION_SOURCE_TABLES.items():
            expected[relation] = _project_rows(
                directory, "goldberg2025", relation, source_table, schemas[relation]
            )
    source_tables = []
    for source_id, relation, source_table in _projected_source_tables(
        source_ids, include_extensions=not v1
    ):
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
        "revision": "captured_fragment_mapp_canonical_v2",
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
    revision = reconciliation.get("revision")
    if revision not in (
        "captured_fragment_mapp_canonical_v1",
        "captured_fragment_mapp_canonical_v2",
    ):
        raise ValueError("Unsupported canonical MAPP revision")
    source_ids = tuple(reconciliation.get("source_adapter_ids", ()))
    expected_source_ids = V1_MAPP_SOURCE_IDS if revision.endswith("_v1") else MAPP_SOURCE_IDS
    if tuple(sorted(source_ids)) != tuple(sorted(expected_source_ids)):
        raise ValueError("Canonical MAPP source adapters differ")
    schemas = V1_SCHEMAS if revision.endswith("_v1") else SCHEMAS
    artifact_names = V1_ARTIFACT_NAMES if revision.endswith("_v1") else ARTIFACT_NAMES
    expected = _expected_rows(directory, source_ids)
    for relation, rows in expected.items():
        path = directory / TYPED_ROOT / f"{artifact_names[relation]}.parquet"
        if not path.is_file():
            raise ValueError(f"Missing canonical MAPP relation: {relation}")
        table = pq.read_table(path)
        if table.schema != schemas[relation] or table.to_pylist() != rows:
            raise ValueError(f"Canonical MAPP relation drift: {relation}")
        _verify_source_datasets(rows, relation)
    observed = {relation: len(rows) for relation, rows in expected.items()}
    if reconciliation.get("relation_rows") != observed:
        raise ValueError("Canonical MAPP row conservation differs")
    actual_files = {path.stem for path in (directory / TYPED_ROOT).glob("*.parquet")}
    if actual_files != set(artifact_names.values()):
        raise ValueError("Canonical MAPP relation inventory differs")
    return reconciliation
