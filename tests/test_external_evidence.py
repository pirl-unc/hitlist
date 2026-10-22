"""External quantitative sources retain assay scope, derivation and raw claims."""

import hashlib
import json
import math

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import external_evidence
from hitlist.pxd057588_evidence import (
    SCHEMAS,
    SUPPORT_FILES,
    _Sink,
    _write_assertions,
    _write_table,
)
from tests.test_supporting_evidence import _package


def _mini_workbook(root, *, table, samples, intensities, reported, extra_sequence=None):
    parent_column, fc_column = {9: ("K", "M"), 10: ("N", "P"), 11: ("L", "O")}[table]
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = f"Supplementary Table {table}"
    sheet["A5"] = "Sequence"
    sheet[f"{parent_column}5"] = "Protein"
    sheet[f"{fc_column}5"] = samples["treated"]
    for column, name in zip(("B", "C"), samples["order"]):
        sheet[f"{column}4"] = "128N" if column == "B" else "128C"
        sheet[f"{column}5"] = name
    for row_number, (sequence, parent, values, fold_change) in enumerate(
        zip(("PEPTIDE", "SEQUENCE"), ("literal-A", "literal-B"), intensities, reported),
        start=6,
    ):
        sheet[f"A{row_number}"] = sequence
        sheet[f"{parent_column}{row_number}"] = parent
        sheet[f"B{row_number}"] = values[0]
        sheet[f"C{row_number}"] = values[1]
        sheet[f"{fc_column}{row_number}"] = fold_change
    panels = [{"sequence_header_cell": "A5", "populated_sequence_rows": 2}]
    if extra_sequence:
        sheet["Z5"] = "Sequence"
        sheet["Z6"] = extra_sequence
        panels.append({"sequence_header_cell": "Z5", "populated_sequence_rows": 1})
    asset = root / "assets" / "PXD057588"
    asset.mkdir(parents=True, exist_ok=True)
    workbook.save(asset / f"table{table}.xlsx")
    spec = {
        "table": table,
        "n_rows": 2,
        "parent_column": parent_column,
        "policy": "total_signal" if table == 11 else "unscaled",
        "contrasts": [
            {
                "sample": samples["treated"],
                "fc_column": fc_column,
                "control_samples": [samples["control"]],
                "expected_fc_header": samples["treated"],
            }
        ],
    }
    inventory = {
        "primary_rows": 2,
        "channels": [
            {
                "column": column,
                "source_channel_label": "128N" if column == "B" else "128C",
                "source_sample_label": name,
            }
            for column, name in zip(("B", "C"), samples["order"])
        ],
        "independently_keyed_sequence_blocks": panels,
    }
    return spec, inventory


def _convert_mini(root, cases):
    typed = root / "typed"
    typed.mkdir()
    sinks = {name: _Sink(typed, name) for name in SCHEMAS}
    try:
        counts = [
            _write_table(number, root, spec, inventory, sinks, 1e-8)
            for number, spec, inventory in cases
        ]
    finally:
        for sink in sinks.values():
            sink.close()
    return counts, typed


def test_channel_identity_pool_and_normalization_are_assay_scoped(tmp_path):
    table9 = _mini_workbook(
        tmp_path,
        table=9,
        samples={"order": ("C2+", "R_ctrl_1"), "treated": "C2+", "control": "R_ctrl_1"},
        intensities=((8, 2), (4, 2)),
        reported=(2.0, 1.0),
        extra_sequence="OTHER-PANEL-PEPTIDE",
    )
    table11 = _mini_workbook(
        tmp_path,
        table=11,
        samples={"order": ("R_ctrl_1", "C2+"), "treated": "C2+", "control": "R_ctrl_1"},
        intensities=((2, 8), (8, 4)),
        reported=(math.log2((8 / 12) / (2 / 10)), math.log2((4 / 12) / (8 / 10))),
    )
    counts, typed = _convert_mini(tmp_path, [(9, *table9), (11, *table11)])
    assert [row["n_intensity_results"] for row in counts] == [4, 4]
    channels = pq.read_table(typed / "channels.parquet").to_pylist()
    channel128n = [row for row in channels if row["channel_label_raw"] == "128N"]
    assert {row["sample_label_raw"] for row in channel128n} == {"C2+", "R_ctrl_1"}
    assert len({row["channel_id"] for row in channel128n}) == 2
    results = pq.read_table(typed / "quantity_results.parquet").to_pylist()
    assert {row["measured_pool"] for row in results if row["table_number"] == 9} == {"mhc_bound"}
    assert {row["measured_pool"] for row in results if row["table_number"] == 11} == {
        "bulk_digest_peptide"
    }
    panels = pq.read_table(typed / "panel_records.parquet").to_pylist()
    assert any(row["sequence_raw"] == "OTHER-PANEL-PEPTIDE" for row in panels)
    assert not any(
        row["sequence_raw"] == "OTHER-PANEL-PEPTIDE"
        for row in pq.read_table(typed / "peptide_rows.parquet").to_pylist()
    )
    reported = [
        row
        for row in pq.read_table(typed / "quantity_measurements.parquet").to_pylist()
        if row["representation"] == "reported_log2_fold_change"
        and row["result_id"].startswith("pxd057588:t11:")
    ]
    assert len(reported) == 2
    assert all(row["normalization_policy"] == "total_signal" for row in reported)
    assert all(row["absolute_replay_error"] <= 1e-8 for row in reported)


def test_wrong_fold_change_policy_fails_before_publication(tmp_path):
    spec, inventory = _mini_workbook(
        tmp_path,
        table=11,
        samples={"order": ("R_ctrl_1", "C2+"), "treated": "C2+", "control": "R_ctrl_1"},
        intensities=((2, 8), (8, 4)),
        reported=(math.log2(8 / 2), math.log2(4 / 8)),
    )
    with pytest.raises(ValueError, match="fails arithmetic replay"):
        _convert_mini(tmp_path, [(11, spec, inventory)])


def test_partner_loss_is_not_assigned_to_measured_macrophage(tmp_path):
    spec, inventory = _mini_workbook(
        tmp_path,
        table=10,
        samples={"order": ("G1", "U1"), "treated": "G1", "control": "U1"},
        intensities=((8, 2), (4, 2)),
        reported=(2.0, 1.0),
    )
    _, typed = _convert_mini(tmp_path, [(10, spec, inventory)])
    channels = pq.read_table(typed / "channels.parquet").to_pylist()
    assert all(row["measured_population"] == "macrophage" for row in channels)
    treated = next(row for row in channels if row["sample_label_raw"] == "G1")
    assert treated["partner_loss_status"] == "reported_partner_mhc_i_loss"
    assert treated["partner_population"] == "mhc_i_loss_tumor_cell"
    assert all(row["material_link_precision"] != "exact_specimen" for row in channels)


def test_conflicting_source_claims_and_project_instrument_keep_their_scope(tmp_path):
    project = tmp_path / "assets" / "PXD057588"
    project.mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps({"instruments": [{"name": "Orbitrap Exploris 480"}]})
    )
    inventory = {
        entry["table"]: entry
        for entry in json.loads(SUPPORT_FILES["workbook_inventory.json"].read_text())
    }
    typed = tmp_path / "typed"
    typed.mkdir()
    sink = _Sink(typed, "source_assertions")
    try:
        _write_assertions(tmp_path, inventory, {"source_assertions": sink})
    finally:
        sink.close()
    claims = pq.read_table(typed / "source_assertions.parquet").to_pylist()
    assert len(claims) == 36
    table8 = [
        row
        for row in claims
        if row["subject_id"] == "pxd057588:t8" and row["predicate"] == "polarization_class"
    ]
    assert {row["normalized_value"] for row in table8} == {
        "anti_inflammatory",
        "pro_inflammatory",
    }
    table10 = [
        row
        for row in claims
        if row["subject_id"] == "pxd057588:t10" and row["predicate"] == "replication_description"
    ]
    assert {row["normalized_value"] for row in table10} == {
        "reported_n_3",
        "four_labeled_channels_per_partner_arm_not_donor_count",
    }
    instrument = next(row for row in claims if row["predicate"] == "instrument")
    assert instrument["scope"] == "project_not_individual_run"


def test_external_generation_embeds_parent_and_rejects_asset_tampering(tmp_path, monkeypatch):
    parent, _, _, _ = _package(tmp_path)
    source = tmp_path / "external-source"
    source.mkdir()
    (source / "data.txt").write_text("one source row\n")
    receipt = tmp_path / "external-receipt.json"
    original = (source / "data.txt").read_bytes()
    receipt.write_text(
        json.dumps(
            {
                "assets": {
                    "data.txt": {
                        "sha256": hashlib.sha256(original).hexdigest(),
                        "size": len(original),
                    }
                }
            }
        )
    )

    class Adapter:
        source_id = "fixture"
        schema_version = 1
        asset_receipt_path = receipt
        support_files = {}

        def write(self, directory):
            typed = directory / "typed"
            typed.mkdir()
            pq.write_table(pa.table({"source_row_id": ["fixture:1"]}), typed / "rows.parquet")
            return {"n_source_rows": 1}

        def verify(self, directory, reconciliation):
            assert (
                pq.read_metadata(directory / "typed" / "rows.parquet").num_rows
                == (reconciliation["n_source_rows"])
            )

        def registry(self, reconciliation):
            return {"n_source_rows": reconciliation["n_source_rows"]}

    monkeypatch.setattr(external_evidence, "_adapters", lambda: {"fixture": Adapter()})
    output = tmp_path / "external"
    external_evidence.write_external_evidence_package(
        parent, output, source_roots={"fixture": source}, source_ids=("fixture",)
    )
    manifest = external_evidence.verify_external_evidence_package(output)
    assert manifest["source_ids"] == ["fixture"]
    assert manifest["sources"]["fixture"]["reconciliation"]["n_source_rows"] == 1
    (source / "data.txt").write_text("source edited after ingest\n")
    external_evidence.verify_external_evidence_package(output)
    with (output / "external" / "fixture" / "assets" / "data.txt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact mismatch"):
        external_evidence.verify_external_evidence_package(output)
