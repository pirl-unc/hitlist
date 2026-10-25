"""The captured-fragment MAPP API is one lossless, source-qualified layout."""

from __future__ import annotations

import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_store import EvidenceStore
from hitlist.mapp_canonical import (
    ARTIFACT_NAMES,
    SCHEMAS,
    TYPED_ROOT,
    _source_table_specifications,
    verify_canonical_mapp_tables,
    write_canonical_mapp_tables,
)
from hitlist.mapp_controlled_evidence import write_controlled_tables
from hitlist.mapp_evidence import write_mapp_tables
from hitlist.mapp_pbmc_evidence import write_pbmc_tables
from tests.test_mapp_controlled_evidence import _fixture as controlled_fixture
from tests.test_mapp_evidence import _source as pxd009092_fixture
from tests.test_mapp_pbmc_evidence import _fixture as pxd009438_fixture


def _package(root):
    pxd009092 = root / "external" / "pxd009092"
    pxd009092.mkdir(parents=True)
    pxd009092_fixture(pxd009092)
    write_mapp_tables(pxd009092)

    controlled = root / "external" / "wolflevy2018_controlled"
    controlled.mkdir(parents=True)
    controlled_fixture(controlled)
    write_controlled_tables(controlled)

    pxd009438 = root / "external" / "pxd009438"
    pxd009438.mkdir(parents=True)
    pxd009438_fixture(pxd009438)
    write_pbmc_tables(pxd009438)
    return write_canonical_mapp_tables(root)


def _store(root):
    bindings = materialized_bindings(9, package_features=("canonical_mapp",))
    store = EvidenceStore.__new__(EvidenceStore)
    (root / "manifest.json").write_text(json.dumps({"schema_version": 9, "mapp_canonical": {}}))
    store.directory = root
    store.manifest = {"schema_version": 9, "mapp_canonical": {}}
    store.bindings = {
        name: binding for name, binding in bindings.items() if name.startswith("mapp_")
    }
    store.schemas = {
        name: pq.ParquetFile(root / binding[0].materialized_artifact).schema_arrow
        for name, binding in store.bindings.items()
    }
    return store


def test_all_admitted_sources_share_one_schema_and_preserve_rows(tmp_path):
    receipt = _package(tmp_path)
    assert receipt["source_datasets"] == ["PXD007263", "PXD009092", "PXD009438", "PXD009508"]
    assert receipt["relation_rows"]["mapp_acquisition"] == 62
    assert receipt["relation_rows"]["mapp_quantity_result"] == 94
    assert receipt["source_relation_rows"]["mapp_acquisition"] == {
        "PXD007263": 12,
        "PXD009092": 18,
        "PXD009438": 14,
        "PXD009508": 18,
    }
    for relation, schema in SCHEMAS.items():
        table = pq.read_table(tmp_path / TYPED_ROOT / f"{ARTIFACT_NAMES[relation]}.parquet")
        # Relation artifacts deliberately omit the public prefix; all have the
        # same required source identity columns before source-specific fields.
        assert table.schema == schema
        assert table.schema.names[:2] == ["source_dataset", "source_dataset_precision"]
    verify_canonical_mapp_tables(tmp_path, receipt)

    source_tables = pq.read_table(
        tmp_path / TYPED_ROOT / f"{ARTIFACT_NAMES['mapp_source_table']}.parquet"
    ).to_pylist()
    mapped = {
        (row["source_dataset"], row["canonical_relation"], row["source_typed_table"])
        for row in source_tables
    }
    assert ("PXD009508", "mapp_label_alias", "label_aliases") in mapped
    assert ("PXD007263", "mapp_acquisition_context", "channels") in mapped
    assert ("PXD009438", "mapp_quantity_evidence_summary", "quantity_results") in mapped


def test_every_native_mapp_field_has_a_canonical_base_or_extension_home(tmp_path):
    _package(tmp_path)
    fields_by_source_table = defaultdict(set)
    for relation, source_table in _source_table_specifications():
        fields_by_source_table[source_table].update(SCHEMAS[relation].names)

    for source_id in ("pxd009092", "wolflevy2018_controlled", "pxd009438"):
        typed = tmp_path / "external" / source_id / "typed"
        for source_table, projected_fields in fields_by_source_table.items():
            path = typed / f"{source_table}.parquet"
            if not path.exists():
                continue
            native_fields = set(pq.ParquetFile(path).schema_arrow.names)
            # Native source_dataset is represented by the canonical invariant
            # plus its precision rather than copied as a second raw field.
            assert native_fields - {"source_dataset"} <= projected_fields


def test_uniform_query_spans_sources_and_joins_context_without_physical_inference(tmp_path):
    _package(tmp_path)
    store = _store(tmp_path)
    rows = store.query(
        "mapp_quantity_result",
        select=[
            "source_dataset",
            "result_id",
            "acquisition.condition_text",
            "evidence_summary.n_direct_msms_evidence",
        ],
        joins=["acquisition", "evidence_summary"],
        limit=None,
    ).table.to_pylist()
    assert {row["source_dataset"] for row in rows} == {
        "PXD007263",
        "PXD009092",
        "PXD009438",
        "PXD009508",
    }
    assert any(row["evidence_summary.n_direct_msms_evidence"] is None for row in rows)
    acquisitions = store.query(
        "mapp_acquisition",
        select=["source_dataset", "channel_id", "acquisition_context.capture_role"],
        joins=["acquisition_context"],
        limit=None,
    ).table.to_pylist()
    assert any(
        row["acquisition_context.capture_role"] == "candidate_proteasome_capture"
        for row in acquisitions
    )
    contexts = store.query(
        "mapp_material_context",
        select=["source_dataset", "context_kind", "specimen_id", "aliquot_id"],
        limit=None,
    ).table.to_pylist()
    assert all(row["specimen_id"] is row["aliquot_id"] is None for row in contexts)


def test_canonical_projection_rejects_any_source_row_drift(tmp_path):
    receipt = _package(tmp_path)
    path = tmp_path / "external" / "pxd009438" / "typed" / "peptide_rows.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["sequence_raw"] = "DRIFT"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.ParquetFile(path).schema_arrow), path)
    with pytest.raises(ValueError, match="Canonical MAPP relation drift"):
        verify_canonical_mapp_tables(tmp_path, receipt)


def test_legacy_source_prefixed_public_relations_are_not_bindable():
    bindings = materialized_bindings(9, package_features=("canonical_mapp",))
    assert {"mapp_acquisition", "mapp_quantity_result", "mapp_label_alias"} <= set(bindings)
    assert not any(
        name.startswith("processing_") or name.startswith("pbmc_mapp_") for name in bindings
    )
