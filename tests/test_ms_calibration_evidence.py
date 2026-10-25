"""Calibration evidence preserves identity, missingness and source meaning."""

import json
import xml.etree.ElementTree as ET
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import ms_calibration_evidence as calibration
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_store import EvidenceRelationshipError, EvidenceStore


def test_reporter_sequence_membership_does_not_erase_form_identity():
    labels = ["RUXF.C[CAM]NNVLYIR[+10]", "RUXF.C[PCm]NNVLYIR[+10]"]
    assert {calibration.canonical_sequence(x) for x in labels} == {"CNNVLYIR"}
    assert calibration._id("reporter", labels[0]) != calibration._id("reporter", labels[1])
    assert calibration.canonical_sequence("ROA2.GGNFGFGDSR[+10].sum3") == "GGNFGFGDSR"
    assert calibration.canonical_sequence("CNNVLYIR[+10]", panel=True) == "CNNVLYIR"
    with pytest.raises(ValueError, match="Invalid calibration"):
        calibration.canonical_sequence("protein.INVALIDX")


def test_curve_reporter_aggregation_is_not_an_s4_identity_alias():
    s4 = "SMD2.NNTQVLINC[CAM]R[+10]"
    curve = s4 + ".sum4"
    assert calibration.canonical_sequence(s4) == calibration.canonical_sequence(curve)
    assert calibration._id("reporter", s4) != calibration._id("reporter", curve)
    labels = {curve} | {f"other:{i}" for i in range(30)}
    blocks = {
        (str(site), mode): set(labels) for site in range(1, 12) for mode in ("raw", "normalized")
    }
    calibration._validate_curve_reporters(blocks)
    blocks["5", "normalized"].remove(curve)
    blocks["5", "normalized"].add(s4)
    with pytest.raises(ValueError, match="within source table"):
        calibration._validate_curve_reporters(blocks)
    blocks.pop(("5", "normalized"))
    with pytest.raises(ValueError, match="site/representation"):
        calibration._validate_curve_reporters(blocks)


@pytest.mark.parametrize(
    "value,present,formula,status",
    [
        (None, False, False, "absent_source_cell"),
        ("", True, False, "explicit_empty_string"),
        (None, True, True, "empty_formula_cache"),
        (0.0, True, False, "reported_zero"),
        (1.0, True, False, "reported_positive"),
        ("#N/A", True, True, "reported_nonnumeric"),
    ],
)
def test_missingness_states_are_not_quantitative_negatives(value, present, formula, status):
    assert calibration._value_status(value, present=present, formula=formula) == status


def test_actual_rsq_arguments_define_curve_positions_not_layout_offsets():
    header, response, pairs = calibration.curve_range(
        '=IF(ISERROR(RSQ($E$364:$I$364,E367:I367)),"",RSQ($E$364:$I$364,E367:I367))'
    )
    assert header == "$E$364:$I$364"
    assert response == "E367:I367"
    assert pairs == [(f"{c}364", f"{c}367") for c in "EFGHI"]
    for invalid in ["=RSQ(A1:A4,B2:B5)", "=RSQ(A1:E1,A2:F2)", "=SUM(A1:A5)"]:
        with pytest.raises(ValueError, match="RSQ"):
            calibration.curve_range(invalid)


def test_original_cells_preserve_empty_strings_formula_cache_and_merged_headers(tmp_path):
    import openpyxl

    assets = tmp_path / "assets"
    assets.mkdir()
    for number in (1, 2, 4, 8, 13, 14, 15):
        path = assets / f"supplementary-data-{number}.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = "original title"
        sheet["A2"] = "placeholder"
        sheet["A3"] = "=SUM(1,2)"
        sheet["A4"].number_format = "0.00"  # Style only: no reported value.
        sheet["A5"] = "=SUM(A3)"  # Deliberately absent formula cache.
        sheet["C1"] = "merged header"
        sheet.merge_cells("C1:D1")
        workbook.save(path)
        with zipfile.ZipFile(path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        xml = ET.fromstring(members["xl/worksheets/sheet1.xml"])
        namespace = calibration.NS["s"]
        for node in xml.findall(".//s:c", calibration.NS):
            if node.attrib["r"] == "A2":
                node.find("s:is/s:t", calibration.NS).text = ""
            if node.attrib["r"] == "A3":
                node.find(f"{{{namespace}}}v").text = "3"
        members["xl/worksheets/sheet1.xml"] = ET.tostring(xml)
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
    tables = {name: [] for name in calibration.SCHEMAS}
    books = calibration._workbooks(tmp_path, tables)
    first = books[1]
    assert first["A2"]["value_json"] == '""'
    assert first["A2"]["value_status"] == "explicit_empty_string"
    assert first["A3"]["formula"] == "=SUM(1,2)"
    assert first["A3"]["value"] == 3
    assert first["A5"]["value_status"] == "empty_formula_cache"
    assert "A4" not in first
    assert json.loads(tables["source_sheets"][0]["merged_ranges_json"]) == ["C1:D1"]
    assert tables["source_sheets"][3]["title_status"] == "source_label_conflict"


def _query_store(tmp_path, rows):
    bindings = materialized_bindings()
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory = tmp_path
    store.manifest = {"schema_version": 8}
    (tmp_path / "manifest.json").write_text(json.dumps(store.manifest))
    store.bindings = {name: bindings[name] for name in rows}
    store.schemas = {}
    for name, records in rows.items():
        path = tmp_path / bindings[name][0].materialized_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records)
        pq.write_table(table, path)
        store.schemas[name] = table.schema
    return store


def test_nominal_load_and_reporter_joins_do_not_claim_form_molarity(tmp_path):
    store = _query_store(
        tmp_path,
        {
            "calibration_s4_measurement": [
                {
                    "measurement_id": "raw:1",
                    "reporter_id": "form:cam",
                    "nominal_load_id": "load:B:S4",
                    "value": 100.0,
                    "representation": "raw",
                    "unit": "arbitrary_peak_area",
                },
                {
                    "measurement_id": "normalized:1",
                    "reporter_id": "form:cam",
                    "nominal_load_id": "load:B:S4",
                    "value": 220.0,
                    "representation": "normalized",
                    "unit": "arbitrary_peak_area",
                },
            ],
            "calibration_nominal_load": [
                {"nominal_load_id": "load:B:S4", "value": 10 / 3, "unit": "fmol_on_column"}
            ],
            "calibration_reporter_form": [
                {
                    "reporter_id": "form:cam",
                    "standard_id": "standard:1",
                    "peptide_sequence": "CNNVLYIR",
                },
                {
                    "reporter_id": "form:pcm",
                    "standard_id": "standard:1",
                    "peptide_sequence": "CNNVLYIR",
                },
            ],
            "calibration_standard": [{"standard_id": "standard:1", "peptide_sequence": "CNNVLYIR"}],
        },
    )
    result = store.query(
        "calibration_s4_measurement",
        select=[
            "measurement_id",
            "representation",
            "quantitative_value",
            "measurement_unit",
            "nominal_load.quantitative_value",
            "nominal_load.measurement_unit",
        ],
        joins=["nominal_load"],
        limit=None,
    ).table.to_pylist()
    assert len(result) == 2
    assert [r["quantitative_value"] for r in result] == [100, 220]
    assert all(
        r["measurement_unit"] == "arbitrary_peak_area"
        and r["nominal_load.measurement_unit"] == "fmol_on_column"
        for r in result
    )
    with pytest.raises(EvidenceRelationshipError):
        store.query(
            "calibration_standard",
            select=["peptide_sequence", "reporter_forms.native.reporter_id"],
            joins=["reporter_forms"],
            limit=None,
        )


def test_calibration_bindings_are_scoped_to_actual_source_generations():
    assert "calibration_s4_measurement" not in materialized_bindings(7)
    assert "calibration_s4_measurement" not in materialized_bindings(
        8, appended_source_ids=["pxd009092"]
    )
    both = materialized_bindings(
        9,
        appended_source_ids=["pxd004886"],
        package_features=["canonical_mapp"],
    )
    assert "calibration_s4_measurement" in both and "mapp_peptide" in both
    assert sum(name.startswith("calibration_") for name in both) == 13


def test_verifier_rejects_rewritten_quantities_even_with_reconciliation_counts(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    (source / "contracts").mkdir(parents=True)
    (source / "contracts/contract.json").write_text(json.dumps({"expected_reconciliation": {}}))
    tables = {name: [] for name in calibration.SCHEMAS}
    tables["s4_measurements"] = [
        {
            "measurement_id": "one",
            "representation": "raw",
            "value_status": "reported_positive",
            "value": 2.0,
        }
    ]
    monkeypatch.setattr(calibration, "_build_tables", lambda directory: tables)
    adapter = calibration.MSCalibrationAdapter()
    reconciliation = adapter.write(source)
    adapter.verify(source, reconciliation)
    path = source / "typed/s4_measurements.parquet"
    changed = pq.read_table(path).to_pylist()
    changed[0]["value"] = 20.0
    pq.write_table(
        pa.Table.from_pylist(changed, schema=calibration.SCHEMAS["s4_measurements"]), path
    )
    with pytest.raises(ValueError, match="differs from source"):
        adapter.verify(source, reconciliation)
