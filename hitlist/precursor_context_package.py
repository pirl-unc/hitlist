"""Standalone source-contract package for the PXD009360 precursor context."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .material_context import MATERIAL_CONTEXT_SCHEMA, MATERIAL_LINK_SCHEMA, validate_material_graph
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 23
CATALOG_SCHEMA_VERSION = 36
REVISION = "pxd009360_precursor_context_v4"
PACKAGE_FAMILY = "precursor_context"
SOURCE_ID = "pxd009360_precursor"
SOURCE_PREFIX = f"external/{SOURCE_ID}/"
ASSET_RECEIPT = Path(__file__).parent / "data/evidence_revisions/precursor_context_assets.json"

_PREVIOUS_PACKAGE_SCHEMA_VERSION = 22
_PREVIOUS_CATALOG_SCHEMA_VERSION = 27
_PREVIOUS_REVISION = "pxd009360_precursor_context_v3"

QUERY_ALIASES = {
    "precursor_asset": "precursor_source_asset",
    "precursor_run": "precursor_acquisition",
    "precursor_table": "precursor_table_inventory",
    "precursor_pairing": "precursor_pairing_assertion",
    "precursor_evidence": "precursor_evidence_row",
    "precursor_peptide": "precursor_peptide_row",
    "precursor_all_peptide": "precursor_all_peptide_row",
    "precursor_protein_group": "precursor_protein_group_row",
    "precursor_recipe": "precursor_aggregation_recipe",
    "precursor_qualification": "precursor_aggregation_qualification",
    "precursor_pairing_qualification": "precursor_pairing_qualification",
    "precursor_assertion": "precursor_source_assertion",
    "precursor_material_context": "precursor_material_context",
    "precursor_material_link": "precursor_material_link",
}
_V3_QUERY_ALIASES = {
    key: value
    for key, value in QUERY_ALIASES.items()
    if key not in {"precursor_assertion", "precursor_material_context", "precursor_material_link"}
}
MATERIALIZED_EVIDENCE_TYPES = frozenset(
    {
        "precursor_context_family",
        "precursor_context_availability",
        "precursor_source_asset",
        "precursor_acquisition",
        "precursor_table_inventory",
        "precursor_pairing_assertion",
        "precursor_source_assertion",
        "precursor_material_context",
        "precursor_material_link",
        "precursor_evidence_row",
        "precursor_peptide_row",
        "precursor_all_peptide_row",
        "precursor_protein_group_row",
        "precursor_aggregation_recipe",
        "precursor_aggregation_qualification",
        "precursor_pairing_qualification",
    }
)

FAMILY_SCHEMA = pa.schema(
    [
        ("family_id", pa.string()),
        ("evidence_family", pa.string()),
        ("availability_state", pa.string()),
        ("source_scope", pa.string()),
        ("n_source_rows", pa.int64()),
        ("n_measurements", pa.int64()),
        ("n_training_examples", pa.int64()),
        ("training_status", pa.string()),
        ("evidence_types_json", pa.string()),
        ("notes", pa.string()),
    ]
)
AVAILABILITY_SCHEMA = pa.schema(
    [
        ("family_id", pa.string()),
        ("source_id", pa.string()),
        ("availability_state", pa.string()),
        ("n_source_rows", pa.int64()),
        ("n_measurements", pa.int64()),
        ("n_training_examples", pa.int64()),
        ("count_unit", pa.string()),
        ("evidence_basis", pa.string()),
        ("source_locator_status", pa.string()),
        ("training_status", pa.string()),
    ]
)
SOURCE_ASSET_SCHEMA = pa.schema(
    [
        ("source_asset_id", pa.string()),
        ("source_file", pa.string()),
        ("source_asset_role", pa.string()),
        ("asset_sha256", pa.string()),
        ("asset_size_bytes", pa.int64()),
        ("source_locator", pa.string()),
        ("source_locator_status", pa.string()),
        ("availability_state", pa.string()),
        ("notes", pa.string()),
    ]
)
ACQUISITION_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("acquisition_id", pa.string()),
        ("sample_label_raw", pa.string()),
        ("source_file", pa.string()),
        ("source_dataset", pa.string()),
        ("condition_status", pa.string()),
        ("condition_evidence", pa.string()),
        ("specimen_status", pa.string()),
        ("material_link_precision", pa.string()),
        ("source_locator", pa.string()),
        ("source_row_disposition", pa.string()),
        ("notes", pa.string()),
    ]
)
TABLE_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_file", pa.string()),
        ("table_name", pa.string()),
        ("archive_member_size_bytes", pa.int64()),
        ("source_dataset", pa.string()),
        ("availability_state", pa.string()),
        ("table_status", pa.string()),
        ("materialized_relation", pa.string()),
        ("n_source_rows", pa.int64()),
        ("source_member_sha256", pa.string()),
        ("source_member_size_bytes", pa.int64()),
        ("relation_sha256", pa.string()),
        ("source_locator", pa.string()),
        ("source_row_disposition", pa.string()),
        ("notes", pa.string()),
    ]
)

QUANTITY_ROW_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_file", pa.string()),
        ("source_dataset", pa.string()),
        ("source_row_index", pa.int64()),
        ("source_row_sha256", pa.string()),
        ("raw_row", pa.string()),
        ("column_names_json", pa.string()),
        ("values_json", pa.string()),
        ("sequence_raw", pa.string()),
        ("modified_sequence_raw", pa.string()),
        ("modifications_raw", pa.string()),
        ("raw_file_raw", pa.string()),
        ("experiment_raw", pa.string()),
        ("charge_raw", pa.string()),
        ("identification_type_raw", pa.string()),
        ("intensity_raw", pa.string()),
        ("lfq_intensity_raw", pa.string()),
        ("match_between_runs_raw", pa.string()),
        ("contributors_json", pa.string()),
        ("status_flags_json", pa.string()),
    ]
)
RECIPE_SCHEMA = pa.schema(
    [
        ("recipe_id", pa.string()),
        ("recipe_revision", pa.string()),
        ("source_dataset", pa.string()),
        ("source_relations_json", pa.string()),
        ("grouping_policy_json", pa.string()),
        ("quantity_policy_json", pa.string()),
        ("missingness_policy_json", pa.string()),
        ("pairing_policy_json", pa.string()),
        ("training_status", pa.string()),
        ("notes", pa.string()),
    ]
)
QUALIFICATION_SCHEMA = pa.schema(
    [
        ("qualification_id", pa.string()),
        ("recipe_id", pa.string()),
        ("source_relation", pa.string()),
        ("source_file", pa.string()),
        ("n_source_rows", pa.int64()),
        ("n_rows_with_sequence", pa.int64()),
        ("n_rows_with_contributors", pa.int64()),
        ("n_rows_with_intensity", pa.int64()),
        ("n_rows_missing_intensity", pa.int64()),
        ("n_rows_ambiguous", pa.int64()),
        ("n_contributor_links", pa.int64()),
        ("source_relation_sha256", pa.string()),
        ("qualification_status", pa.string()),
        ("training_status", pa.string()),
        ("notes", pa.string()),
    ]
)
PAIRING_QUALIFICATION_SCHEMA = pa.schema(
    [
        ("qualification_id", pa.string()),
        ("claim_id", pa.string()),
        ("pairing_scope", pa.string()),
        ("material_link_precision", pa.string()),
        ("pairing_status", pa.string()),
        ("source_backed", pa.bool_()),
        ("training_status", pa.string()),
        ("split_policy", pa.string()),
        ("evidence_basis", pa.string()),
        ("notes", pa.string()),
    ]
)
PAIRING_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("source_dataset", pa.string()),
        ("assertion_predicate", pa.string()),
        ("assertion_value", pa.string()),
        ("assertion_status", pa.string()),
        ("pairing_scope", pa.string()),
        ("pairing_status", pa.string()),
        ("pairing_evidence", pa.string()),
        ("source_locator", pa.string()),
        ("notes", pa.string()),
    ]
)


def _source_scoped_schema(schema):
    """Give shared context vocabularies the canonical source scope columns."""
    return pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_dataset_precision", pa.string()),
            *schema,
        ]
    )


SOURCE_ASSERTION_SCHEMA = pa.schema(
    [
        ("source_dataset", pa.string()),
        ("source_dataset_precision", pa.string()),
        ("assertion_id", pa.string()),
        ("subject_id", pa.string()),
        ("predicate", pa.string()),
        ("value_raw", pa.string()),
        ("normalized_value", pa.string()),
        ("assertion_status", pa.string()),
        ("source_locator", pa.string()),
        ("scope", pa.string()),
    ]
)
PRECURSOR_MATERIAL_CONTEXT_SCHEMA = _source_scoped_schema(MATERIAL_CONTEXT_SCHEMA)
PRECURSOR_MATERIAL_LINK_SCHEMA = _source_scoped_schema(MATERIAL_LINK_SCHEMA)


def _receipt():
    return json.loads(ASSET_RECEIPT.read_text())


def _manifest(directory):
    path = Path(directory)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads((path / "manifest.json").read_text())


def _source_paths(source_root, receipt):
    source_root = Path(source_root)
    paths = {}
    for relative, expected in receipt["assets"].items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned PXD009360 source asset: {relative}")
        actual = file_digest(path)
        if actual != {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}:
            raise ValueError(f"PXD009360 source asset mismatch: {relative}")
        paths[relative] = path
    return paths


QUANTITY_MEMBERS = {
    "txt/evidence.txt": "precursor_evidence_row",
    "txt/peptides.txt": "precursor_peptide_row",
    "txt/allPeptides.txt": "precursor_all_peptide_row",
    "txt/proteinGroups.txt": "precursor_protein_group_row",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _first(headers, values, *names):
    for name in names:
        try:
            return values[headers.index(name)]
        except ValueError:
            continue
    return None


def _matching(headers, values, prefix):
    return {
        header: values[index]
        for index, header in enumerate(headers)
        if header == prefix or header.startswith(prefix + " ")
    }


def _quantity_row(relative, row_index, raw_line, headers, values, source_dataset):
    row_sha256 = hashlib.sha256(raw_line).hexdigest()
    raw_row = raw_line.rstrip(b"\r\n").decode("utf-8")
    contributors = {
        header: values[index]
        for index, header in enumerate(headers)
        if any(
            token in header.lower()
            for token in ("protein", "peptide ids", "evidence ids", "ms/ms ids")
        )
    }
    status_flags = {
        header: values[index]
        for index, header in enumerate(headers)
        if header in {"Reverse", "Potential contaminant", "Identified", "Type"}
        or header.startswith("Identification type")
    }
    return {
        "source_record_id": f"{SOURCE_ID}:{relative}:{row_index}",
        "source_file": relative,
        "source_dataset": source_dataset,
        "source_row_index": row_index,
        "source_row_sha256": row_sha256,
        "raw_row": raw_row,
        "column_names_json": _json(headers),
        "values_json": _json(values),
        "sequence_raw": _first(headers, values, "Sequence"),
        "modified_sequence_raw": _first(headers, values, "Modified sequence"),
        "modifications_raw": _first(headers, values, "Modifications"),
        "raw_file_raw": _first(headers, values, "Raw file"),
        "experiment_raw": _first(headers, values, "Experiment"),
        "charge_raw": _first(headers, values, "Charge", "Charges"),
        "identification_type_raw": _json(_matching(headers, values, "Identification type")),
        "intensity_raw": _first(headers, values, "Intensity"),
        "lfq_intensity_raw": _json(_matching(headers, values, "LFQ intensity")),
        "match_between_runs_raw": _json(_matching(headers, values, "Match")),
        "contributors_json": _json(contributors),
        "status_flags_json": _json(status_flags),
    }


def _extract_quantity_tables(archive_path, destination, source_dataset):
    """Stream selected MaxQuant members into typed Parquet relations."""
    destination = Path(destination)
    writers = {}
    buffers = {relation: [] for relation in QUANTITY_MEMBERS.values()}
    metadata = {}
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                relation = QUANTITY_MEMBERS.get(member.name)
                if relation is None:
                    continue
                expected_size = None
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        if stream is None:
                            raise ValueError(f"Archive member cannot be read: {member.name}")
                        member_digest = hashlib.sha256()
                        member_size = 0
                        row_count = 0
                        relation_digest = hashlib.sha256()
                        headers = None
                        for raw_line in stream:
                            member_digest.update(raw_line)
                            member_size += len(raw_line)
                            if headers is None:
                                headers = raw_line.rstrip(b"\r\n").decode("utf-8").split("\t")
                                continue
                            row_count += 1
                            values = raw_line.rstrip(b"\r\n").decode("utf-8").split("\t")
                            if len(values) != len(headers):
                                raise ValueError(
                                    f"Column count mismatch in {member.name} row {row_count}"
                                )
                            row = _quantity_row(
                                member.name, row_count, raw_line, headers, values, source_dataset
                            )
                            relation_digest.update(row["source_row_sha256"].encode("ascii"))
                            relation_digest.update(b"\n")
                            buffers[relation].append(row)
                            if len(buffers[relation]) >= 1024:
                                table = pa.Table.from_pylist(
                                    buffers[relation], schema=QUANTITY_ROW_SCHEMA
                                )
                                if relation not in writers:
                                    writers[relation] = pq.ParquetWriter(
                                        destination / f"{relation}.parquet", QUANTITY_ROW_SCHEMA
                                    )
                                writers[relation].write_table(table)
                                buffers[relation].clear()
                        if headers is None:
                            raise ValueError(f"Archive member has no header: {member.name}")
                        expected_size = member.size
                        if member_size != expected_size:
                            raise ValueError(
                                f"Archive member size mismatch for {member.name}: "
                                f"{member_size} != {expected_size}"
                            )
                        metadata[member.name] = {
                            "materialized_relation": relation,
                            "n_source_rows": row_count,
                            "source_member_sha256": member_digest.hexdigest(),
                            "source_member_size_bytes": member_size,
                            "relation_sha256": relation_digest.hexdigest(),
                        }
    finally:
        for relation, values in buffers.items():
            if values and relation not in writers:
                writers[relation] = pq.ParquetWriter(
                    destination / f"{relation}.parquet", QUANTITY_ROW_SCHEMA
                )
            if values:
                writers[relation].write_table(
                    pa.Table.from_pylist(values, schema=QUANTITY_ROW_SCHEMA)
                )
                values.clear()
        for _relation, writer in writers.items():
            writer.close()
    for relation in QUANTITY_MEMBERS.values():
        path = destination / f"{relation}.parquet"
        if not path.exists():
            pq.write_table(pa.Table.from_pylist([], schema=QUANTITY_ROW_SCHEMA), path)
    return metadata


def _recipe_rows(receipt):
    from .precursor_recipe import CONTRACT, RECIPE_ID

    return [
        {
            "recipe_id": RECIPE_ID,
            "recipe_revision": REVISION,
            "source_dataset": receipt["dataset_accession"],
            "source_relations_json": _json(CONTRACT["source_relations"]),
            "grouping_policy_json": _json(CONTRACT["grouping"]),
            "quantity_policy_json": _json(CONTRACT["quantity_policy"]),
            "missingness_policy_json": _json(CONTRACT["missingness_policy"]),
            "pairing_policy_json": _json(CONTRACT["pairing_policy"]),
            "training_status": CONTRACT["training_status"],
            "notes": "Qualification only; no precursor rows are admitted as training examples.",
        }
    ]


def _qualify_quantity_relations(directory, materialized_tables):
    from .precursor_recipe import RECIPE_ID

    rows = []
    for source_file, metadata in sorted(materialized_tables.items()):
        relation = metadata["materialized_relation"]
        counts = {
            "n_source_rows": 0,
            "n_rows_with_sequence": 0,
            "n_rows_with_contributors": 0,
            "n_rows_with_intensity": 0,
            "n_rows_missing_intensity": 0,
            "n_rows_ambiguous": 0,
            "n_contributor_links": 0,
        }
        parquet = pq.ParquetFile(Path(directory) / f"{relation}.parquet")
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=["sequence_raw", "contributors_json", "intensity_raw"],
        ):
            for sequence, contributor_text, intensity in zip(
                batch.column("sequence_raw").to_pylist(),
                batch.column("contributors_json").to_pylist(),
                batch.column("intensity_raw").to_pylist(),
            ):
                counts["n_source_rows"] += 1
                if sequence not in (None, ""):
                    counts["n_rows_with_sequence"] += 1
                contributor_values = []
                if contributor_text:
                    contributor_values = list(json.loads(contributor_text).values())
                if contributor_values:
                    counts["n_rows_with_contributors"] += 1
                    counts["n_contributor_links"] += sum(
                        1 for value in contributor_values for item in value.split(";") if item
                    )
                    if any(";" in value for value in contributor_values):
                        counts["n_rows_ambiguous"] += 1
                if intensity in (None, "", "NaN", "nan", "#N/A"):
                    counts["n_rows_missing_intensity"] += 1
                else:
                    counts["n_rows_with_intensity"] += 1
        rows.append(
            {
                "qualification_id": f"{RECIPE_ID}:{relation}",
                "recipe_id": RECIPE_ID,
                "source_relation": relation,
                "source_file": source_file,
                **counts,
                "source_relation_sha256": metadata["relation_sha256"],
                "qualification_status": "qualified_for_nontraining_only",
                "training_status": "unavailable",
                "notes": "Counts are source-row qualifications; no abundance or pairing inference.",
            }
        )
    return rows


def _pairing_qualification_rows(receipt):
    from .precursor_recipe import RECIPE_ID

    return [
        {
            "qualification_id": f"{RECIPE_ID}:pairing:{row['claim_id']}",
            "claim_id": row["claim_id"],
            "pairing_scope": row["pairing_scope"],
            "material_link_precision": "unresolved",
            "pairing_status": row["pairing_status"],
            "source_backed": row["assertion_status"] == "established",
            "training_status": "unavailable",
            "split_policy": "no_biological_split_until_source_backed_crosswalk",
            "evidence_basis": row["pairing_evidence"],
            "notes": "Run labels, ordering and abundance correlation do not establish pairing.",
        }
        for row in receipt["pairings"]
    ]


def _material_graph_rows(receipt):
    """Return only the source-backed project/run graph and unresolved claims.

    A deposited raw-file record establishes an acquisition's project membership.
    It does not establish a condition, study arm, specimen, aliquot, or link to
    a captured-fragment MAPP acquisition. The rows deliberately use the same
    source-scoped assertion/material layout as canonical MAPP.
    """
    source_dataset = receipt["dataset_accession"]
    project_id = f"{source_dataset.lower()}:project"
    contexts = [
        {
            "source_dataset": source_dataset,
            "source_dataset_precision": "exact_project",
            "context_id": project_id,
            "context_kind": "source_project",
            "sample_label_raw": None,
            "condition_id": None,
            "condition_text": None,
            "condition_status": "not_applicable_source_project",
            "specimen_id": None,
            "aliquot_id": None,
            "material_link_precision": "project_context",
            "context_status": "source_project_context",
            "source_locator": f"{source_dataset}/project.json",
            "notes": "Project identity does not establish condition or physical material.",
        }
    ]
    links = []
    for acquisition in receipt["acquisitions"]:
        acquisition_id = acquisition["acquisition_id"]
        raw_file = acquisition["source_file"]
        locator = f"{source_dataset}/files.json::fileName={raw_file}"
        contexts.append(
            {
                "source_dataset": source_dataset,
                "source_dataset_precision": "exact_project",
                "context_id": acquisition_id,
                "context_kind": "source_acquisition",
                "sample_label_raw": acquisition["sample_label_raw"],
                "condition_id": None,
                "condition_text": None,
                "condition_status": "unreported_in_admitted_assets",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": "source_acquisition",
                "context_status": "source_acquisition_without_condition_or_physical_material",
                "source_locator": locator,
                "notes": "The raw-file label does not establish condition, specimen, or aliquot.",
            }
        )
        links.append(
            {
                "source_dataset": source_dataset,
                "source_dataset_precision": "exact_project",
                "claim_id": f"{acquisition_id}:in_project",
                "source_context_id": acquisition_id,
                "target_context_id": project_id,
                "assertion_predicate": "source_acquisition_in_project",
                "assertion_status": "source_reported",
                "material_link_precision": "project_context",
                "source_locator": locator,
                "notes": "The file inventory establishes project membership only.",
            }
        )
    assertions = [
        {
            "source_dataset": source_dataset,
            "source_dataset_precision": "exact_project",
            "assertion_id": f"{source_dataset.lower()}:acquisition_physical_material_map",
            "subject_id": project_id,
            "predicate": "acquisition_to_specimen_or_aliquot",
            "value_raw": "not supplied in admitted project/file metadata",
            "normalized_value": None,
            "assertion_status": "unreported_in_admitted_assets",
            "source_locator": f"{source_dataset}/project.json;{source_dataset}/files.json",
            "scope": "acquisition_to_physical_material",
        }
    ]
    assertions.extend(
        {
            "source_dataset": source_dataset,
            "source_dataset_precision": "exact_project",
            "assertion_id": row["claim_id"],
            "subject_id": project_id,
            "predicate": row["assertion_predicate"],
            "value_raw": row["assertion_value"],
            "normalized_value": None,
            "assertion_status": row["assertion_status"],
            "source_locator": f"{source_dataset}/project.json;PXD009092 source contract",
            "scope": row["pairing_scope"],
        }
        for row in receipt["pairings"]
    )
    validate_material_graph(
        [{field.name: row[field.name] for field in MATERIAL_CONTEXT_SCHEMA} for row in contexts],
        [{field.name: row[field.name] for field in MATERIAL_LINK_SCHEMA} for row in links],
    )
    return assertions, contexts, links


def _rows(
    receipt,
    materialized_tables=None,
    recipe_rows=None,
    qualification_rows=None,
    pairing_qualification_rows=None,
):
    materialized_tables = materialized_tables or {}
    recipe_rows = recipe_rows or []
    qualification_rows = qualification_rows or []
    pairing_qualification_rows = pairing_qualification_rows or []
    assets = [
        {
            "source_asset_id": f"{SOURCE_ID}:asset:{relative}",
            "source_file": relative,
            "source_asset_role": expected["asset_kind"],
            "asset_sha256": expected["sha256"],
            "asset_size_bytes": expected["size_bytes"],
            "source_locator": expected["source_locator"],
            "source_locator_status": "hash_pinned",
            "availability_state": "materialized",
            "notes": "Caller-supplied source bytes copied into this immutable package.",
        }
        for relative, expected in receipt["assets"].items()
    ]
    acquisitions = [
        {
            "source_record_id": row["acquisition_id"],
            "acquisition_id": row["acquisition_id"],
            "sample_label_raw": row["sample_label_raw"],
            "source_file": row["source_file"],
            "source_dataset": receipt["dataset_accession"],
            "condition_status": "unresolved",
            "condition_evidence": "no_source_backed_condition_map",
            "specimen_status": "unresolved",
            "material_link_precision": "assay_label_only",
            "source_locator": f"{receipt['dataset_accession']}/files.json",
            "source_row_disposition": "source_metadata_only",
            "notes": "Run label does not establish condition, specimen or MAPP pairing.",
        }
        for row in receipt["acquisitions"]
    ]
    tables = [
        {
            "source_record_id": f"{SOURCE_ID}:archive_member:{row['path']}",
            "source_file": row["path"],
            "table_name": Path(row["path"]).stem,
            "archive_member_size_bytes": row["size_bytes"],
            "source_dataset": receipt["dataset_accession"],
            "availability_state": (
                "materialized_typed_rows"
                if row["path"] in materialized_tables
                else "materialized_source_archive"
            ),
            "table_status": (
                "typed_materialized"
                if row["path"] in materialized_tables
                else "deferred_typed_materialization"
            ),
            "materialized_relation": materialized_tables.get(row["path"], {}).get(
                "materialized_relation"
            ),
            "n_source_rows": materialized_tables.get(row["path"], {}).get("n_source_rows", 0),
            "source_member_sha256": materialized_tables.get(row["path"], {}).get(
                "source_member_sha256"
            ),
            "source_member_size_bytes": materialized_tables.get(row["path"], {}).get(
                "source_member_size_bytes", row["size_bytes"]
            ),
            "relation_sha256": materialized_tables.get(row["path"], {}).get("relation_sha256"),
            "source_locator": "search/pxd009360-search.tar.gz",
            "source_row_disposition": (
                "typed_source_rows" if row["path"] in materialized_tables else "source_asset_only"
            ),
            "notes": (
                "Archive member and lossless typed quantity rows are retained."
                if row["path"] in materialized_tables
                else "Archive member is retained; typed quantity rows are child PR 2."
            ),
        }
        for row in receipt["archive_members"]
    ]
    pairings = [
        {
            **row,
            "source_dataset": receipt["dataset_accession"],
            "source_locator": "PXD009360/project.json;PXD009092 source contract",
            "notes": "No run order, abundance correlation or shared label is used as evidence.",
        }
        for row in receipt["pairings"]
    ]
    families = [
        {
            "family_id": "precursor_source_metadata",
            "evidence_family": "source_provenance",
            "availability_state": "materialized",
            "source_scope": "PXD009360_project_and_deposited_search_assets",
            "n_source_rows": len(assets),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_training_data",
            "evidence_types_json": json.dumps(["precursor_source_asset"], separators=(",", ":")),
            "notes": "Hash-pinned project, archive, parameter and summary assets.",
        },
        {
            "family_id": "precursor_acquisition_inventory",
            "evidence_family": "assay_context",
            "availability_state": "materialized",
            "source_scope": "PXD009360_tryptic_acquisition_labels",
            "n_source_rows": len(acquisitions),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "not_training_data",
            "evidence_types_json": json.dumps(["precursor_acquisition"], separators=(",", ":")),
            "notes": "Nine acquisition labels are retained without inferred biological pairing.",
        },
        {
            "family_id": "precursor_quantity_tables",
            "evidence_family": "precursor_measurement",
            "availability_state": (
                "typed_rows_materialized"
                if materialized_tables
                else "source_assets_available_rows_deferred"
            ),
            "source_scope": "PXD009360_maxquant_search_archive",
            "n_source_rows": sum(
                row.get("n_source_rows", 0) for row in materialized_tables.values()
            ),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": (
                "not_training_data" if materialized_tables else "deferred_typed_materialization"
            ),
            "evidence_types_json": json.dumps(["precursor_table_inventory"], separators=(",", ":")),
            "notes": (
                "Evidence, peptide, all-peptide and protein-group rows are retained losslessly."
                if materialized_tables
                else "Evidence, peptide, all-peptide and protein-group rows await child PR 2."
            ),
        },
        {
            "family_id": "mapp_precursor_pairing",
            "evidence_family": "lineage_context",
            "availability_state": "unresolved_in_pinned_sources",
            "source_scope": "PXD009360_to_PXD009092_cross_project_links",
            "n_source_rows": len(pairings),
            "n_measurements": 0,
            "n_training_examples": 0,
            "training_status": "unavailable",
            "evidence_types_json": json.dumps(
                ["precursor_pairing_assertion"], separators=(",", ":")
            ),
            "notes": "Condition, specimen and aliquot links remain unresolved claims.",
        },
    ]
    if recipe_rows:
        families.append(
            {
                "family_id": "precursor_recipe_qualification",
                "evidence_family": "recipe_qualification",
                "availability_state": "materialized",
                "source_scope": "PXD009360_lossless_quantity_relations",
                "n_source_rows": len(qualification_rows),
                "n_measurements": 0,
                "n_training_examples": 0,
                "training_status": "unavailable",
                "evidence_types_json": _json(
                    [
                        "precursor_aggregation_recipe",
                        "precursor_aggregation_qualification",
                        "precursor_pairing_qualification",
                    ]
                ),
                "notes": "Recipe arithmetic and pairing scope are qualified without training admission.",
            }
        )
    availability = [
        {
            "family_id": row["family_id"],
            "source_id": SOURCE_ID,
            "availability_state": row["availability_state"],
            "n_source_rows": row["n_source_rows"],
            "n_measurements": row["n_measurements"],
            "n_training_examples": row["n_training_examples"],
            "count_unit": "typed_relation_rows_or_training_examples",
            "evidence_basis": "pinned_PXD009360_source_contract",
            "source_locator_status": "hash_pinned",
            "training_status": row["training_status"],
        }
        for row in families
    ]
    return families, availability, assets, acquisitions, tables, pairings


def _write_table(directory, name, rows, schema):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), directory / f"{name}.parquet")


def _verify_assets(directory, receipt):
    for relative, expected in receipt["assets"].items():
        path = directory / SOURCE_PREFIX / "assets" / relative
        actual = file_digest(path)
        if actual != {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}:
            raise ValueError(f"PXD009360 package asset mismatch: {relative}")


def _relation_digest(path):
    table = pq.read_table(path, columns=["source_row_sha256"])
    digest = hashlib.sha256()
    for value in table.column("source_row_sha256").to_pylist():
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return table.num_rows, digest.hexdigest()


def verify_precursor_context_package(directory):
    directory = Path(directory)
    manifest = _manifest(directory)
    current = (
        manifest.get("schema_version") == PACKAGE_SCHEMA_VERSION
        and manifest.get("catalog_schema_version") == CATALOG_SCHEMA_VERSION
        and manifest.get("revision") == REVISION
    )
    previous = (
        manifest.get("schema_version") == _PREVIOUS_PACKAGE_SCHEMA_VERSION
        and manifest.get("catalog_schema_version") == _PREVIOUS_CATALOG_SCHEMA_VERSION
        and manifest.get("revision") == _PREVIOUS_REVISION
    )
    if manifest.get("package_family") != PACKAGE_FAMILY or not (current or previous):
        raise ValueError("Unsupported precursor context package envelope")
    receipt_path = directory / SOURCE_PREFIX / "contracts/source_assets.json"
    if file_digest(receipt_path) != file_digest(ASSET_RECEIPT):
        raise ValueError("PXD009360 source receipt differs from the pinned contract")
    receipt = json.loads(receipt_path.read_text())
    _verify_assets(directory, receipt)
    if manifest.get("query_aliases") != (QUERY_ALIASES if current else _V3_QUERY_ALIASES):
        raise ValueError("Precursor context query aliases changed")
    stored_tables = pq.read_table(directory / "precursor_context" / "tables.parquet").to_pylist()
    materialized_tables = {
        row["source_file"]: row for row in stored_tables if row.get("materialized_relation")
    }
    for relative, expected in receipt.get("materialized_tables", {}).items():
        stored = materialized_tables.get(relative)
        if stored is None:
            raise ValueError(f"Missing pinned materialized table ledger row: {relative}")
        for field in ("n_source_rows", "source_member_size_bytes", "source_member_sha256"):
            if stored[field] != expected[field]:
                raise ValueError(f"Pinned materialized table ledger changed: {relative}.{field}")
    recipe_rows = pq.read_table(directory / "precursor_context" / "recipes.parquet").to_pylist()
    qualification_rows = pq.read_table(
        directory / "precursor_context" / "qualifications.parquet"
    ).to_pylist()
    pairing_qualification_rows = pq.read_table(
        directory / "precursor_context" / "pairing_qualifications.parquet"
    ).to_pylist()
    assertion_rows, material_context_rows, material_link_rows = _material_graph_rows(receipt)
    families, availability, assets, acquisitions, tables, pairings = _rows(
        receipt,
        materialized_tables,
        recipe_rows,
        qualification_rows,
        pairing_qualification_rows,
    )
    expected = {
        "families": (FAMILY_SCHEMA, families),
        "availability": (AVAILABILITY_SCHEMA, availability),
        "source_assets": (SOURCE_ASSET_SCHEMA, assets),
        "acquisitions": (ACQUISITION_SCHEMA, acquisitions),
        "tables": (TABLE_SCHEMA, tables),
        "pairings": (PAIRING_SCHEMA, pairings),
        "recipes": (RECIPE_SCHEMA, recipe_rows),
        "qualifications": (QUALIFICATION_SCHEMA, qualification_rows),
        "pairing_qualifications": (
            PAIRING_QUALIFICATION_SCHEMA,
            pairing_qualification_rows,
        ),
    }
    if current:
        expected.update(
            {
                "source_assertions": (SOURCE_ASSERTION_SCHEMA, assertion_rows),
                "material_contexts": (PRECURSOR_MATERIAL_CONTEXT_SCHEMA, material_context_rows),
                "material_links": (PRECURSOR_MATERIAL_LINK_SCHEMA, material_link_rows),
            }
        )
    for name, (schema, rows) in expected.items():
        table = pq.read_table(directory / "precursor_context" / f"{name}.parquet")
        if table.schema != schema or table.to_pylist() != rows:
            raise ValueError(f"Precursor context relation changed: {name}")
    if qualification_rows != _qualify_quantity_relations(
        directory / "precursor_context", materialized_tables
    ):
        raise ValueError("Precursor quantity qualification changed")
    for _source_file, row in materialized_tables.items():
        relation = row["materialized_relation"]
        path = directory / "precursor_context" / f"{relation}.parquet"
        if not path.is_file():
            raise ValueError(f"Missing materialized precursor relation: {relation}")
        table = pq.read_table(path)
        if table.schema != QUANTITY_ROW_SCHEMA:
            raise ValueError(f"Precursor quantity relation schema changed: {relation}")
        n_rows, relation_sha256 = _relation_digest(path)
        if n_rows != row["n_source_rows"] or relation_sha256 != row["relation_sha256"]:
            raise ValueError(f"Precursor quantity relation digest changed: {relation}")
    for name, digest in manifest["files"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Precursor context package artifact mismatch: {name}")
    return manifest


def write_precursor_context_package(source_root, destination):
    """Write a standalone source-contract package from verified PXD009360 bytes."""
    source_root, destination = Path(source_root), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    receipt = _receipt()
    paths = _source_paths(source_root, receipt)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        source_dir = staging / SOURCE_PREFIX
        (source_dir / "assets").mkdir(parents=True)
        (source_dir / "contracts").mkdir()
        snapshot_file(ASSET_RECEIPT, source_dir / "contracts/source_assets.json")
        for relative, path in paths.items():
            snapshot_file(path, source_dir / "assets" / relative)
        typed = staging / "precursor_context"
        typed.mkdir()
        archive_relative = next(
            relative
            for relative, expected in receipt["assets"].items()
            if expected["asset_kind"] in {"search_archive", "maxquant_search_archive"}
            or relative.endswith(".tar.gz")
        )
        materialized_paths = {
            row["path"] for row in receipt["archive_members"] if row["path"] in QUANTITY_MEMBERS
        }
        materialized_tables = _extract_quantity_tables(
            paths[archive_relative], typed, receipt["dataset_accession"]
        )
        if materialized_paths - set(materialized_tables):
            missing = sorted(materialized_paths - set(materialized_tables))
            raise ValueError(f"Missing required quantity archive members: {missing}")
        for relative, expected in receipt.get("materialized_tables", {}).items():
            actual = materialized_tables.get(relative)
            if actual is None:
                raise ValueError(f"Missing pinned materialized table: {relative}")
            for field in ("n_source_rows", "source_member_size_bytes", "source_member_sha256"):
                if actual[field] != expected[field]:
                    raise ValueError(f"Pinned materialized table mismatch: {relative}.{field}")
        recipe_rows = _recipe_rows(receipt)
        qualification_rows = _qualify_quantity_relations(typed, materialized_tables)
        pairing_qualification_rows = _pairing_qualification_rows(receipt)
        assertion_rows, material_context_rows, material_link_rows = _material_graph_rows(receipt)
        families, availability, assets, acquisitions, tables, pairings = _rows(
            receipt,
            materialized_tables,
            recipe_rows,
            qualification_rows,
            pairing_qualification_rows,
        )
        _write_table(typed, "families", families, FAMILY_SCHEMA)
        _write_table(typed, "availability", availability, AVAILABILITY_SCHEMA)
        _write_table(typed, "source_assets", assets, SOURCE_ASSET_SCHEMA)
        _write_table(typed, "acquisitions", acquisitions, ACQUISITION_SCHEMA)
        _write_table(typed, "tables", tables, TABLE_SCHEMA)
        _write_table(typed, "pairings", pairings, PAIRING_SCHEMA)
        _write_table(typed, "source_assertions", assertion_rows, SOURCE_ASSERTION_SCHEMA)
        _write_table(
            typed,
            "material_contexts",
            material_context_rows,
            PRECURSOR_MATERIAL_CONTEXT_SCHEMA,
        )
        _write_table(typed, "material_links", material_link_rows, PRECURSOR_MATERIAL_LINK_SCHEMA)
        _write_table(typed, "recipes", recipe_rows, RECIPE_SCHEMA)
        _write_table(typed, "qualifications", qualification_rows, QUALIFICATION_SCHEMA)
        _write_table(
            typed,
            "pairing_qualifications",
            pairing_qualification_rows,
            PAIRING_QUALIFICATION_SCHEMA,
        )
        from .version import __version__

        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        registry = {
            "source_id": SOURCE_ID,
            "dataset_accession": receipt["dataset_accession"],
            "n_assets": len(assets),
            "n_archive_members": len(receipt["archive_members"]),
            "n_acquisitions": len(acquisitions),
            "condition_map_status": "unresolved",
            "specimen_pairing_status": "unresolved",
            "quantity_materialization_status": "materialized_typed_relations",
            "n_materialized_quantity_rows": sum(
                row["n_source_rows"] for row in materialized_tables.values()
            ),
            "n_recipe_rows": len(recipe_rows),
            "n_qualification_rows": len(qualification_rows),
            "n_pairing_qualification_rows": len(pairing_qualification_rows),
            "n_source_assertion_rows": len(assertion_rows),
            "n_material_context_rows": len(material_context_rows),
            "n_material_link_rows": len(material_link_rows),
        }
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "package_family": PACKAGE_FAMILY,
            "query_aliases": QUERY_ALIASES,
            "materialized_evidence_types": sorted(MATERIALIZED_EVIDENCE_TYPES),
            "appended_sources": {SOURCE_ID: {}},
            "source_registry": {SOURCE_ID: registry},
            "reconciliation": {
                "n_source_assets": len(assets),
                "n_archive_members": len(receipt["archive_members"]),
                "n_acquisitions": len(acquisitions),
                "n_pairing_assertions": len(pairings),
                "n_precursor_quantity_rows": sum(
                    row["n_source_rows"] for row in materialized_tables.values()
                ),
                "n_training_examples": 0,
                "n_recipe_rows": len(recipe_rows),
                "n_qualification_rows": len(qualification_rows),
                "n_pairing_qualification_rows": len(pairing_qualification_rows),
                "n_source_assertion_rows": len(assertion_rows),
                "n_material_context_rows": len(material_context_rows),
                "n_material_link_rows": len(material_link_rows),
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_precursor_context_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
