"""The captured-fragment MAPP API is one lossless, source-qualified layout."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import goldberg_mapp
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_store import EvidenceStore
from hitlist.mapp_canonical import (
    ARTIFACT_NAMES,
    SCHEMAS,
    TYPED_ROOT,
    V1_ARTIFACT_NAMES,
    V1_MAPP_SOURCE_IDS,
    V1_SCHEMAS,
    _expected_rows,
    _write_table,
    verify_canonical_mapp_tables,
    write_canonical_mapp_tables,
)
from hitlist.mapp_controlled_evidence import write_controlled_tables
from hitlist.mapp_evidence import write_mapp_tables
from hitlist.mapp_pbmc_evidence import write_pbmc_tables
from tests.test_goldberg_mapp import original as goldberg_original_fixture
from tests.test_mapp_controlled_evidence import _fixture as controlled_fixture
from tests.test_mapp_evidence import _source as pxd009092_fixture
from tests.test_mapp_pbmc_evidence import _fixture as pxd009438_fixture


def _package(root, monkeypatch):
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

    original = goldberg_original_fixture.__wrapped__(root, monkeypatch)
    goldberg = root / "external" / "goldberg2025"
    goldberg.mkdir(parents=True)
    contract = json.loads(goldberg_mapp.CONTRACT.read_text())
    goldberg_mapp._compile(original, goldberg / "typed", contract)
    return write_canonical_mapp_tables(root)


def _legacy_package(root):
    """Build the prior three-source projection to lock its replay contract."""
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

    expected = _expected_rows(root, V1_MAPP_SOURCE_IDS)
    (root / TYPED_ROOT).mkdir(parents=True)
    for relation, rows in expected.items():
        _write_table(
            root,
            relation,
            rows,
            schemas=V1_SCHEMAS,
            artifact_names=V1_ARTIFACT_NAMES,
        )
    return {
        "revision": "captured_fragment_mapp_canonical_v1",
        "source_adapter_ids": list(V1_MAPP_SOURCE_IDS),
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


def _store(root, *, v2=False):
    package_features = ("canonical_mapp", "canonical_mapp_v2") if v2 else ("canonical_mapp",)
    bindings = materialized_bindings(9, package_features=package_features)
    store = EvidenceStore.__new__(EvidenceStore)
    manifest = {
        "schema_version": 9,
        "mapp_canonical": {"revision": "captured_fragment_mapp_canonical_v2"} if v2 else {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    store.directory = root
    store.manifest = manifest
    store.bindings = {
        name: binding for name, binding in bindings.items() if name.startswith("mapp_")
    }
    store.schemas = {
        name: pq.ParquetFile(root / binding[0].materialized_artifact).schema_arrow
        for name, binding in store.bindings.items()
    }
    return store


def test_all_admitted_sources_share_one_schema_and_preserve_rows(tmp_path, monkeypatch):
    receipt = _package(tmp_path, monkeypatch)
    assert receipt["source_datasets"] == [
        "PXD007263",
        "PXD009092",
        "PXD009438",
        "PXD009508",
        "PXD052314",
    ]
    assert receipt["relation_rows"]["mapp_acquisition"] == 64
    assert receipt["relation_rows"]["mapp_quantity_result"] == 107
    assert receipt["source_relation_rows"]["mapp_acquisition"] == {
        "PXD007263": 12,
        "PXD009092": 18,
        "PXD009438": 14,
        "PXD009508": 18,
        "PXD052314": 2,
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
    assert ("PXD052314", "mapp_search_quantity_detail", "quantities") in mapped
    assert ("PXD052314", "mapp_pooled_quantity_slot", "pooled_slots") in mapped


def test_every_native_mapp_field_has_a_canonical_base_or_extension_home(tmp_path, monkeypatch):
    _package(tmp_path, monkeypatch)
    fields_by_source_table = defaultdict(set)
    ledger = pq.read_table(
        tmp_path / TYPED_ROOT / f"{ARTIFACT_NAMES['mapp_source_table']}.parquet"
    ).to_pylist()
    for row in ledger:
        fields_by_source_table[(row["source_dataset"], row["source_typed_table"])].update(
            SCHEMAS[row["canonical_relation"]].names
        )

    for source_id, dataset in (
        ("pxd009092", "PXD009092"),
        ("wolflevy2018_controlled", "PXD009508"),
        ("pxd009438", "PXD009438"),
        ("goldberg2025", "PXD052314"),
    ):
        typed = tmp_path / "external" / source_id / "typed"
        for (_dataset, source_table), projected_fields in fields_by_source_table.items():
            if _dataset != dataset:
                continue
            path = typed / f"{source_table}.parquet"
            if not path.exists():
                continue
            native_fields = set(pq.ParquetFile(path).schema_arrow.names)
            # Native source_dataset is represented by the canonical invariant
            # plus its precision rather than copied as a second raw field.
            assert native_fields - {"source_dataset"} <= projected_fields


def test_uniform_query_spans_sources_and_joins_context_without_physical_inference(
    tmp_path, monkeypatch
):
    _package(tmp_path, monkeypatch)
    store = _store(tmp_path)
    rows = store.query(
        "mapp_quantity_result",
        select=["source_dataset", "result_id", "channel_id"],
        limit=None,
    ).table.to_pylist()
    assert {row["source_dataset"] for row in rows} == {
        "PXD007263",
        "PXD009092",
        "PXD009438",
        "PXD009508",
        "PXD052314",
    }
    # Aggregate Goldberg source quantities retain their unresolved acquisition
    # relation rather than being assigned a run merely to satisfy a join.
    assert any(row["source_dataset"] == "PXD052314" and row["channel_id"] is None for row in rows)
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
        select=["source_dataset", "context_kind", "sample_label_raw", "specimen_id", "aliquot_id"],
        limit=None,
    ).table.to_pylist()
    assert all(row["specimen_id"] is row["aliquot_id"] is None for row in contexts)
    assert {
        row["sample_label_raw"]
        for row in contexts
        if row["source_dataset"] == "PXD052314" and row["context_kind"] == "source_sample"
    } == {"Ctrl_1", "Ctrl4_1"}
    source_sample_links = store.query(
        "mapp_material_link",
        select=[
            "source_dataset",
            "material_link_precision",
            "source_material_context.context_kind",
            "target_material_context.context_kind",
        ],
        joins=["source_material_context", "target_material_context"],
        limit=None,
    ).table.to_pylist()
    assert any(
        row["source_dataset"] == "PXD052314"
        and row["material_link_precision"] == "source_acquisition"
        and row["source_material_context.context_kind"] == "source_acquisition"
        and row["target_material_context.context_kind"] == "source_sample"
        for row in source_sample_links
    )
    acquisition_assertions = store.query(
        "mapp_source_assertion",
        select=["source_dataset", "source_scope", "subject_acquisition.channel_id"],
        joins=["subject_acquisition"],
        limit=None,
    ).table.to_pylist()
    assert any(
        row["source_dataset"] == "PXD052314"
        and row["source_scope"] == "original_deposition_acquisition"
        and row["subject_acquisition.channel_id"] is not None
        for row in acquisition_assertions
    )


def test_goldberg_extensions_are_uniformly_queryable_without_identity_inference(
    tmp_path, monkeypatch
):
    _package(tmp_path, monkeypatch)
    store = _store(tmp_path, v2=True)
    assets = store.query(
        "mapp_source_asset",
        select=["source_dataset", "source_asset_id", "asset_sha256"],
        limit=None,
    ).table.to_pylist()
    assert {row["source_dataset"] for row in assets} == {"PXD052314"}
    assert len(assets) == 6
    details = store.query(
        "mapp_search_quantity_detail",
        select=["measurement_id", "value_status", "quantitative_raw", "acquisition_id"],
        limit=None,
    ).table.to_pylist()
    assert {row["value_status"] for row in details} == {
        "positive",
        "zero",
        "missing",
        "nonfinite",
        "invalid_literal",
    }
    slots = store.query(
        "mapp_pooled_quantity_slot",
        select=["slot_id", "sample_label_raw", "acquisition_id", "specimen_id", "aliquot_id"],
        limit=None,
    ).table.to_pylist()
    assert [row["sample_label_raw"] for row in slots] == ["repeat", "repeat", "unknown"]
    assert all(
        row["acquisition_id"] is row["specimen_id"] is row["aliquot_id"] is None for row in slots
    )
    linked = store.query(
        "mapp_search_quantity_detail",
        select=[
            "measurement_id",
            "value_status",
            "canonical_measurement.quantitative_value",
            "canonical_measurement.measurement_unit",
        ],
        joins=["canonical_measurement"],
        limit=None,
    ).table.to_pylist()
    assert {row["value_status"] for row in linked} == {
        "positive",
        "zero",
        "missing",
        "nonfinite",
        "invalid_literal",
    }
    assert {row["canonical_measurement.measurement_unit"] for row in linked} == {"search_intensity"}
    source_rows = store.query(
        "mapp_pooled_quantity_slot",
        select=["slot_id", "canonical_source_record.source_file"],
        joins=["canonical_source_record"],
        limit=None,
    ).table.to_pylist()
    assert {row["canonical_source_record.source_file"] for row in source_rows} == {
        "supplement.zip!2024-04-08499C-s5/"
        "2024-04-08499C-Supplementary Table 3- Data from multiple MAPP experiments_correct.xlsx"
    }


def test_canonical_projection_rejects_any_source_row_drift(tmp_path, monkeypatch):
    receipt = _package(tmp_path, monkeypatch)
    path = tmp_path / "external" / "pxd009438" / "typed" / "peptide_rows.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["sequence_raw"] = "DRIFT"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.ParquetFile(path).schema_arrow), path)
    with pytest.raises(ValueError, match="Canonical MAPP relation drift"):
        verify_canonical_mapp_tables(tmp_path, receipt)


def test_historical_three_source_canonical_mapp_package_still_replays(tmp_path):
    receipt = _legacy_package(tmp_path)
    assert receipt["source_adapter_ids"] == list(V1_MAPP_SOURCE_IDS)
    verify_canonical_mapp_tables(tmp_path, receipt)


def test_legacy_source_prefixed_public_relations_are_not_bindable():
    bindings = materialized_bindings(9, package_features=("canonical_mapp",))
    assert {"mapp_acquisition", "mapp_quantity_result", "mapp_label_alias"} <= set(bindings)
    assert not any(
        name.startswith("processing_") or name.startswith("pbmc_mapp_") for name in bindings
    )
