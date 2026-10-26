"""Synthetic source layouts and corruptions; never biological training data."""

import copy
import json
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pyarrow.parquet as pq
import pytest

from hitlist import nontryptic_calibration as package
from hitlist.evidence_store import EvidenceStore
from hitlist.nontryptic_calibration_compiler import (
    compile_nontryptic_source,
    numeric,
    write_nontryptic_tables,
)
from hitlist.provenance import file_digest

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
HEADERS = [
    "Peptide",
    "Modified Sequence",
    "Replicate",
    "Precursor Mz",
    "Precursor Charge",
    "Product Mz",
    "Product Charge",
    "Fragment Ion",
    "Retention Time",
    "Area",
    "Peak Rank",
    "Concat",
]


def workbook(path, sheets):
    with ZipFile(path, "w") as archive:
        names, relations = [], []
        for i, (name, cells) in enumerate(sheets.items(), 1):
            names.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
            relations.append(
                f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>'
            )
            body = []
            for address, value in cells.items():
                if value is None:
                    body.append(f'<c r="{address}" s="1"/>')
                elif isinstance(value, tuple):
                    body.append(f'<c r="{address}"><f>{escape(value[0])}</f><v>{value[1]}</v></c>')
                elif isinstance(value, (int, float)):
                    body.append(f'<c r="{address}"><v>{value}</v></c>')
                else:
                    body.append(
                        f'<c r="{address}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
                    )
            archive.writestr(
                f"xl/worksheets/sheet{i}.xml",
                f'<worksheet xmlns="{NS}"><sheetData><row r="1">'
                + "".join(body)
                + "</row></sheetData></worksheet>",
            )
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets>'
            + "".join(names)
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(relations)
            + "</Relationships>",
        )


def transition(cells, row, form, label, area, *, fragment="y2", concat="y2_1"):
    values = ["ALV", form, label, 300.123, 2, 200.234, 1, fragment, 10.0, area, 1, concat]
    cells.update(
        {f"{column}{row}": value for column, value in zip("ABCDEFGHIJKL", values, strict=True)}
    )


@pytest.fixture
def source_layout(tmp_path):
    panel = {
        f"{c}3": v
        for c, v in zip(
            "BCDEF",
            ["Peptide", "Gene", "Annotated Gene", "Length", "Predicted affinity [nM]"],
            strict=True,
        )
    }
    panel.update({"B4": "ALV", "C4": "GENE", "D4": "GENE_1", "E4": 3, "F4": 42.5, "XFD21": None})
    direct = {
        "A1": "Free 1H response",
        "A3": "Peptide",
        "B3": "fmol",
        "C3": "Summed intensity",
        "A4": "ALV[+6]",
        "B4": 1,
        "C4": 12,
        "D4": 0,
        "E4": None,
        "B5": 100,
        "C5": 123,
        "D5": 124,
        "E5": 125,
    }
    embedded = {
        "A1": "Embedded 2H response",
        "A3": "Peptide",
        "B3": "Fmol",
        "C3": "Summed Intensity",
        "A4": "ALV",
        "B4": 100,
        "C4": 24,
        "D4": 36,
    }
    counts = {"B2": "Sequence", "B3": "ALV"}
    for column, dose in zip(
        "CDEFGH", ["10 amol", "100 amol", "1 fmol", "10 fmol", "100 fmol", "1000 fmol"], strict=True
    ):
        counts[column + "2"] = dose
        counts[column + "3"] = 2
    residues = {
        "A5": "L",
        "B5": "13C6,15N1",
        "C5": 7.017,
        "A6": "V",
        "B6": "13C3,15N1",
        "C6": 4.007,
        "A7": "A",
        "B7": "13C3,15N1",
        "C7": 4.007,
    }
    forms = {
        "A6": "1H",
        "B6": "AL(V)",
        "D6": 6,
        "E6": "V:13C3,15N1",
        "A7": "2H",
        "B7": "A(L)(V)",
        "D7": 13,
        "E7": "L:13C6,15N1",
        "A8": "3H",
        "B8": "(A)(L)(V)",
        "D8": 17,
        "E8": "A:13C3,15N1",
    }
    direct_export = {c + "1": v for c, v in zip("ABCDEFGHIJKL", HEADERS, strict=True)}
    direct_export.update(
        {
            "N1": "Replicate ",
            "N2": "0.1 = 100 amol",
            "N3": "1 = 1 fmol",
            "N4": "10 = 10 fmol",
            "N5": "100 = 100 fmol",
        }
    )
    transition(direct_export, 2, "ALV[+6]", "1_1", 5, concat=None)
    transition(direct_export, 3, "ALV[+6]", "1_1", 7, fragment="y1", concat="y1_1")
    embedded_export = {c + "1": v for c, v in zip("ABCDEFGHIJKL", HEADERS, strict=True)}
    transition(embedded_export, 2, "AL[+7]V[+6]", "100fmol_1", 24)
    transition(
        embedded_export,
        3,
        "AL[+7]V[+6]",
        "100fmol_1",
        1000,
        fragment="precursor",
        concat="precursor_1",
    )
    transition(embedded_export, 4, "AL[+7]V[+6]", "100fmol_3", 36)
    transition(
        embedded_export,
        5,
        "A[+4]L[+7]V[+6]",
        "1fmol_1",
        3000,
        fragment="precursor",
        concat="precursor_1",
    )
    layouts = {
        "panel.xlsx": {"DatasetS1": panel},
        "responses.xlsx": {"DatasetS3A": direct, "DatasetS3B": embedded, "DatasetS3C": counts},
        "isotopes.xlsx": {"DatasetS6A": residues, "DatasetS6B": forms},
        "direct.xlsx": {"Sheet1": direct_export},
        "embedded.xlsx": {"Sheet1": embedded_export},
    }
    contract = {
        "source_dataset": "PXD024917",
        "workbooks": {
            "panel": "panel.xlsx",
            "responses": "responses.xlsx",
            "isotopes": "isotopes.xlsx",
            "direct_export": "direct.xlsx",
            "embedded_export": "embedded.xlsx",
        },
        "context_locators": {
            "project": "fixture",
            "direct": "responses.xlsx#DatasetS3A!A1",
            "embedded": "responses.xlsx#DatasetS3B!A1",
        },
        "protocol_assertions": [
            {
                "source_file": "responses.xlsx",
                "context": "direct",
                "predicate": "spike_stage",
                "value_raw": "after_purification",
                "status": "source_reported",
                "scope": "reference_protocol",
                "source_locator": "responses.xlsx#DatasetS3A!A1",
            }
        ],
    }

    def materialize():
        for filename, sheets in layouts.items():
            workbook(tmp_path / filename, sheets)
        contract["assets"] = [
            {
                "filename": name,
                **file_digest(tmp_path / name),
                "source_url": "https://example.invalid/" + name,
                "role": "synthetic_software_fixture",
                "license_status": "synthetic_fixture",
            }
            for name in layouts
        ]
        return tmp_path, copy.deepcopy(contract)

    return layouts, materialize


def rows(tables, name):
    return tables[name].to_pylist()


def test_source_quantities_and_prediction_metadata_stay_separate(source_layout):
    _, materialize = source_layout
    tables = compile_nontryptic_source(*materialize())
    measured = rows(tables, "measurements")
    assert len(measured) == 9  # Only three reported dose rows; no absent low-dose rows.
    states = {
        r["source_locator"].split("!")[1]: r
        for r in measured
        if "DatasetS3A" in r["source_locator"]
    }
    assert states["D4"]["value_status"] == "reported_zero"
    assert states["D4"]["quantitative_value"] == 0
    assert states["E4"]["value_status"] == "not_reported"
    assert states["E4"]["cell_id"] is not None  # Stored formatted blank.
    blank = next(r for r in measured if "DatasetS3B!E4" in r["source_locator"])
    assert blank["cell_id"] is None  # Truly absent cell; slot remains explicit.
    assert blank["quantitative_value"] is None
    assert rows(tables, "standards")[0]["predicted_affinity_nm"] == 42.5
    assert rows(tables, "standards")[0]["prediction_status"].endswith("prediction_not_assay")
    assert all(
        r["measurement_unit"] == "selected_transition_count"
        for r in rows(tables, "transition_counts")
    )
    assert all(
        r["acquisition_id"] is None and r["specimen_id"] is None
        for r in rows(tables, "source_slots")
    )
    assert all(
        r["s6a_consistency_status"] == "conflicting_source_assertions"
        for r in rows(tables, "reported_forms")
    )


@pytest.mark.parametrize(
    "reported,parsed,status,consistency",
    [
        (3, 3.0, "reported_positive", "agrees_with_sequence"),
        (2, 2.0, "reported_positive", "conflicts_with_sequence"),
        (None, None, "not_reported", "no_valid_reported_length"),
        ("3", None, "non_numeric_cell", "no_valid_reported_length"),
        (("LEN(B4)", "3"), None, "formula_not_measurement", "no_valid_reported_length"),
        (2.5, 2.5, "reported_positive", "no_valid_reported_length"),
        (0, 0.0, "reported_zero", "no_valid_reported_length"),
    ],
)
def test_panel_length_assertion_is_preserved_without_discarding_data(
    source_layout, reported, parsed, status, consistency
):
    layouts, materialize = source_layout
    layouts["panel.xlsx"]["DatasetS1"]["E4"] = reported
    tables = compile_nontryptic_source(*materialize())
    standard = rows(tables, "standards")[0]
    assert standard["peptide_sequence"] == "ALV"
    assert standard["peptide_length"] == 3
    raw = reported[1] if isinstance(reported, tuple) else reported
    assert standard["reported_length_raw"] == (str(raw) if raw is not None else None)
    assert standard["reported_length_value"] == parsed
    assert standard["reported_length_status"] == status
    assert standard["length_consistency_status"] == consistency
    assert len(rows(tables, "measurements")) == 9


def test_precursors_and_reported_sums_are_distinct(source_layout):
    _, materialize = source_layout
    tables = compile_nontryptic_source(*materialize())
    groups = {g["transition_group_id"]: g for g in rows(tables, "transition_groups")}
    aggregates = rows(tables, "aggregate_signals")
    sums = {
        a["aggregation_mode"]: a["quantitative_value"]
        for a in aggregates
        if groups[a["transition_group_id"]]["source_run_label"] == "100fmol_1"
    }
    assert sums == {"product_only": 24, "all_transitions": 1024}
    precursor_only = next(
        a
        for a in aggregates
        if a["aggregation_mode"] == "product_only"
        and groups[a["transition_group_id"]]["source_run_label"] == "1fmol_1"
    )
    assert precursor_only["quantitative_value"] is None
    assert precursor_only["value_status"] == "no_transitions_of_requested_kind"
    assert precursor_only["n_transition_rows"] == 0
    assert rows(tables, "transitions")[0]["concat_status"] == "not_reported"


def test_ordinal_similarity_does_not_assign_acquisitions(source_layout):
    _, materialize = source_layout
    tables = compile_nontryptic_source(*materialize())
    measurements = {m["measurement_id"]: m for m in rows(tables, "measurements")}
    matches = [
        r
        for r in rows(tables, "measurement_candidates")
        if measurements[r["measurement_id"]]["source_locator"].endswith("DatasetS3B!D4")
    ]
    assert len(matches) == 2
    assert all(not r["ordinal_suffix_equal"] for r in matches)
    assert all(
        not r["acquisition_join_established"] for r in rows(tables, "measurement_candidates")
    )
    assert all(g["acquisition_id"] is None for g in rows(tables, "transition_groups"))


@pytest.mark.parametrize("target", ["response", "transition"])
def test_formula_cache_never_becomes_a_measurement(source_layout, target):
    layouts, materialize = source_layout
    if target == "response":
        layouts["responses.xlsx"]["DatasetS3A"]["C4"] = ("1+1", "999")
    else:
        layouts["direct.xlsx"]["Sheet1"]["J2"] = ("1+1", "999")
    tables = compile_nontryptic_source(*materialize())
    table = "measurements" if target == "response" else "transitions"
    value = next(r for r in rows(tables, table) if r["quantitative_raw"] == "999")
    assert value["value_status"] == "formula_not_measurement"
    assert value["quantitative_value"] is None
    if target == "transition":
        sums = [
            r
            for r in rows(tables, "aggregate_signals")
            if r["transition_group_id"] == value["transition_group_id"]
        ]
        assert all(
            r["value_status"] == "incomplete_sum" and r["quantitative_value"] is None for r in sums
        )


def test_duplicates_are_retained_and_sums_disqualified(source_layout):
    layouts, materialize = source_layout
    transition(layouts["direct.xlsx"]["Sheet1"], 4, "ALV[+6]", "1_1", 5, concat=None)
    tables = compile_nontryptic_source(*materialize())
    duplicates = [r for r in rows(tables, "transitions") if r["n_duplicate_identity_rows"] == 2]
    assert len(duplicates) == 2
    sums = [
        r
        for r in rows(tables, "aggregate_signals")
        if r["transition_group_id"] == duplicates[0]["transition_group_id"]
    ]
    assert all(
        r["quantitative_value"] == 17 and r["value_status"] == "duplicate_identity_sum_unqualified"
        for r in sums
    )


def test_ambiguous_form_assertions_are_not_selected_by_row_order(source_layout):
    layouts, materialize = source_layout
    form = layouts["isotopes.xlsx"]["DatasetS6B"]
    for column in "ABDE":
        form[column + "9"] = form[column + "7"]
    tables = compile_nontryptic_source(*materialize())
    ambiguous = [
        r for r in rows(tables, "form_correspondences") if r["correspondence_status"] == "ambiguous"
    ]
    assert len(ambiguous) == 2
    assert {r["n_form_candidates"] for r in ambiguous} == {2}
    assert len({r["form_id"] for r in ambiguous}) == 2


def test_pinned_source_change_fails_before_interpretation(source_layout):
    _, materialize = source_layout
    source, contract = materialize()
    with (source / "panel.xlsx").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Pinned source asset differs"):
        compile_nontryptic_source(source, contract)


def test_parquet_replay_is_deterministic_and_nonoverwriting(source_layout):
    _, materialize = source_layout
    source, contract = materialize()
    expected = compile_nontryptic_source(source, contract)
    destination = source / "typed"
    counts = write_nontryptic_tables(source, destination, contract)
    for name, table in expected.items():
        assert pq.read_table(destination / (name + ".parquet")).equals(table)
        assert counts["n_" + name + "_rows"] == table.num_rows
    with pytest.raises(FileExistsError):
        write_nontryptic_tables(source, destination, contract)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "1e1000", "1e-1000", "bad"])
def test_nonphysical_numeric_states_remain_unusable(value):
    assert numeric(value) == (None, "invalid_numeric")


@pytest.fixture
def source_package(source_layout, monkeypatch):
    _, materialize = source_layout
    source, contract = materialize()
    contract["expected_reconciliation"] = {}
    contract["source_registry"] = {"PXD024917": {"scope": "synthetic_software_fixture"}}
    contract_path = source / "fixture-contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(package, "CONTRACT", contract_path)
    destination = source / "package"
    package.write_nontryptic_calibration_package(source, destination)
    return destination


def test_public_package_queries_preserve_missing_slots_and_scope(source_package):
    store = EvidenceStore.open(source_package)
    assert len(store.bindings) == 19
    assert all(name.startswith("nontryptic_") for name in store.bindings)
    result = store.query(
        "nontryptic_measurements", select=["measurement_id", "quantitative_value", "value_status"]
    )
    assert result.table.num_rows == 9
    assert sum(r["quantitative_value"] is None for r in result.table.to_pylist()) == 2
    assert store.catalog()["source_registry"]["PXD024917"]["scope"] == "synthetic_software_fixture"
    assert all(
        not r["acquisition_join_established"]
        for r in store.query(
            "nontryptic_measurement_candidates", select=["acquisition_join_established"]
        ).table.to_pylist()
    )
    # Verify that package assets are snapshots, not links to the caller's source.
    original = source_package.parent / "panel.xlsx"
    original.write_bytes(b"changed after package creation")
    assert EvidenceStore.open(source_package).query(
        "nontryptic_standards", select=["peptide_sequence"]
    ).table.to_pylist() == [{"peptide_sequence": "ALV"}]


def test_typed_tampering_is_rejected_even_after_rehash(source_package):
    path = source_package / "nontryptic_calibration/typed/measurements.parquet"
    table = pq.read_table(path)
    data = table.to_pylist()
    data[0]["quantitative_value"] = 999
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(data, schema=table.schema), path)
    manifest_path = source_package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][str(path.relative_to(source_package))] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original-source replay differs: measurements"):
        EvidenceStore.open(source_package)
