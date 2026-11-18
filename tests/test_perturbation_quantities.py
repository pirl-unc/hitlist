"""Synthetic layouts exercise source conservation, not biological evidence."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import perturbation_quantities as package
from hitlist.evidence_recipe import verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.nontryptic_workbooks import read_source_workbook
from hitlist.perturbation_contrast import RECIPE_ID
from hitlist.perturbation_quantity_compiler import Compiler, numeric, peptide_form
from hitlist.provenance import file_digest


@pytest.fixture
def original(tmp_path, monkeypatch):
    source = tmp_path / "source.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "comparison"
    header = [
        "Sequence",
        "Protein",
        "Gene",
        "Length",
        "DMSO 1",
        "DMSO 2",
        "DMSO 3",
        "1uM 1",
        "1uM 2",
        "1uM 3",
        "Log2(FC)",
        "pval",
    ]
    sheet.append(header)
    sheet.append(
        ["AAAGSGAEF", "protein A;protein B", "GENE_A;GENE_B", 9, 1, 2, 3, 2, 4, 6, 1, 0.04]
    )
    sheet.append(["RVA[S]PTSGVK", "IRS2", "IRS2", 10, 1, 1, 1, 2, 2, 2, 1, 0.001])
    sheet.append(["TSTSTSTST", "protein C", "GENE_C", 9, 0, 1, 1, 2, 2, 2, 1.585, 0.2])
    sheet.append(["Grand Total", None, "?", 11, 99, 99, 99, 99, 99, 99])
    sheet.cell(6, 2, "source note, not a peptide")
    book.save(source)
    # Keep a formula and its original cache without evaluating Excel formulas.
    with ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    xml = ET.fromstring(files["xl/worksheets/sheet1.xml"])
    cell = xml.find('.//s:c[@r="E2"]', namespace)
    formula = ET.Element("{" + namespace["s"] + "}f")
    formula.text = "2/2"
    cell.insert(0, formula)
    files["xl/worksheets/sheet1.xml"] = ET.tostring(xml)
    with ZipFile(source, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    contract = {
        "revision": package.REVISION,
        "source_dataset": "PXD017407",
        "source_reference_id": "synthetic_fixture_not_training_evidence",
        "article_url": "https://example.org/synthetic-fixture",
        "assets": [
            {
                "filename": source.name,
                "role": "data3",
                "source_url": "https://example.org/source.xlsx",
                **file_digest(source),
                "license_status": "synthetic_fixture",
            }
        ],
        "conditions": [
            {
                "condition_id": "control",
                "sample_group": "fixture_line",
                "condition": "DMSO",
                "mhc": "HLA-A*02:01 HLA-B*07:02",
            },
            {
                "condition_id": "treated",
                "sample_group": "fixture_line",
                "condition": "palbociclib 10 uM",
                "mhc": "HLA-A*02:01 HLA-B*07:02",
            },
        ],
        "comparisons": [
            {
                "asset": source.name,
                "sheet": "comparison",
                "header_row": 1,
                "cell_line_name": "fixture_line",
                "control_condition_id": "control",
                "treated_condition_id": "treated",
                "dose": 10,
                "dose_unit": "uM",
                "treatment": "palbociclib",
                "dose_qualification": "source_header_conflict_resolved_by_title_and_filemap",
                "expected_headers": {
                    openpyxl.utils.get_column_letter(i): value for i, value in enumerate(header, 1)
                },
                "gene_column": "C",
                "length_column": "D",
                "n_forms": 3,
                "quantities": [
                    {
                        "quantity_kind": "hipmhc_normalized_reporter",
                        "columns": list("EFGHIJ"),
                        "slots": list(range(6)),
                    }
                ],
                "statistics": {"log2_fold_change": "K", "p_value": "L"},
            }
        ],
        "filemaps": [],
        "protocol_assertions": [["fixture_scope", "software fixture only", "curation policy"]],
        "source_registry": {"PXD017407": {"scope": "synthetic_software_fixture"}},
        "expected_reconciliation": {
            "n_forms_rows": 3,
            "n_plexes_rows": 1,
            "n_samples_rows": 6,
            "n_conditions_rows": 2,
            "n_measurements_rows": 18,
            "n_normalized_measurements": 18,
            "n_excluded_aggregate_rows": 1,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(package, "CONTRACT", contract_path)
    return source, contract


@pytest.fixture
def source_package(original):
    source, _ = original
    destination = source.parent / "package"
    package.write_perturbation_quantity_package(source.parent, destination)
    return destination


def recipe(**kwargs):
    return {
        "schema_version": 1,
        "name": "fixture contrast",
        "registered_recipe": RECIPE_ID,
        **kwargs,
    }


def rows(directory, name):
    return pq.read_table(directory / package.ROOT / "typed" / f"{name}.parquet").to_pylist()


def rehash(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    for relative in manifest["files"]:
        manifest["files"][relative] = file_digest(directory / relative)
    path.write_text(json.dumps(manifest))


def test_source_cells_and_conflicting_headers_are_preserved(source_package):
    records = rows(source_package, "source_records")
    source_row = next(row for row in records if row["source_row_ordinal"] == 2)
    cell = next(
        cell for cell in json.loads(source_row["source_cells_json"]) if cell["coordinate"] == "E2"
    )
    assert cell["formula"] == "2/2"
    assert cell["raw_cached_value"] == "1"
    assert cell["has_formula"] and cell["value_present"]
    assert len([r for r in records if r["row_disposition"] == "excluded_aggregate"]) == 1
    forms = rows(source_package, "forms")
    assert [r["peptide_sequence"] for r in forms] == ["AAAGSGAEF", "RVASPTSGVK", "TSTSTSTST"]
    assert forms[0]["protein_label_raw"] == "protein A;protein B"
    assert json.loads(forms[1]["modifications_json"]) == [
        {"position": 4, "residue": "S", "modification": "phosphorylation"}
    ]
    assert rows(source_package, "conditions")[1]["treatment_dose"] == 10
    assert "1uM 1" in {r["source_header_raw"] for r in rows(source_package, "measurements")}
    assert all(
        r["specimen_id"] is None and r["aliquot_id"] is None
        for r in rows(source_package, "samples")
    )


def test_shared_queries_and_empty_joins(source_package):
    store = EvidenceStore.open(source_package)
    assert len(store.bindings) == 13
    assert all(name.startswith("perturbation_") for name in store.bindings)
    result = store.query(
        "perturbation_measurements",
        select=["quantitative_value", "sample.arm_role", "form.peptide_sequence"],
        joins=["sample", "form"],
        where={"sample.arm_role": "treated"},
        limit=None,
    )
    assert result.table.num_rows == 9
    assert set(result.table.column("sample.arm_role").to_pylist()) == {"treated"}
    empty = store.query(
        "perturbation_measurements",
        select=["quantitative_value", "sample.arm_role"],
        joins=["sample"],
        where={"measurement_id": {"op": "in", "value": []}},
    )
    assert empty.table.num_rows == 0
    assert empty.table.schema.field("quantitative_value").type == pa.float64()
    assert store.catalog()["source_registry"]["PXD017407"]["scope"] == "synthetic_software_fixture"


def test_unpaired_contrasts_exclude_zero_and_aggregate(source_package, tmp_path):
    store = EvidenceStore.open(source_package)
    destination = tmp_path / "contrasts"
    store.materialize(recipe(), destination)
    manifest = verify_recipe_package(destination)
    assert manifest["coverage"] == {
        "n_source_candidates": 4,
        "n_contrasts": 2,
        "n_excluded_candidates": 2,
        "n_selected_plexes": 1,
        "n_automatic_training_admissions": 0,
    }
    contrast = pq.read_table(destination / "typed/contrasts.parquet").to_pylist()[0]
    assert contrast["target_log2_fold_change"] == pytest.approx(1)
    assert contrast["mean_control_intensity"] == 2
    assert contrast["sd_control_intensity"] == 1
    assert contrast["mean_treated_intensity"] == 4
    assert contrast["sd_treated_intensity"] == 2
    assert contrast["reported_p_value"] == 0.04
    assert contrast["n_control_samples"] == contrast["n_treated_samples"] == 3
    assert len(json.loads(contrast["measurement_ids_json"])) == 6
    assert contrast["training_admission"].startswith("pending_external_")
    assert contrast["split_group_id"] == "PXD017407"
    assert "peptide_mhc_restriction" not in contrast
    assert contrast["cellular_mhc_genotype"] is None
    assert contrast["candidate_mhc_set"] == "HLA-A*02:01;HLA-B*07:02"
    assert "not_independent_cellular_typing" in contrast["mhc_context_evidence"]


def test_empty_plex_selection_preserves_all_exclusions(source_package, tmp_path):
    destination = tmp_path / "empty_selection"
    EvidenceStore.open(source_package).materialize(recipe(plex_ids=[]), destination)
    manifest = verify_recipe_package(destination)
    assert manifest["coverage"]["n_contrasts"] == 0
    assert manifest["coverage"]["n_source_candidates"] == 4
    assert (
        pq.read_table(destination / "typed/contrasts.parquet")
        .schema.field("target_log2_fold_change")
        .type
        == pa.float64()
    )
    with pytest.raises(ValueError, match="Unknown selected plex"):
        EvidenceStore.open(source_package).materialize(
            recipe(plex_ids=["missing"]), tmp_path / "unknown"
        )


def test_rehash_does_not_hide_quantity_tampering(source_package):
    path = source_package / package.ROOT / "typed/measurements.parquet"
    table = pq.read_table(path)
    data = table.to_pylist()
    data[0]["quantitative_value"] = 500.0
    pq.write_table(pa.Table.from_pylist(data, schema=table.schema), path)
    rehash(source_package)
    with pytest.raises(ValueError, match="original-source replay"):
        EvidenceStore.open(source_package)


def test_rehash_does_not_hide_recipe_target_tampering(source_package, tmp_path):
    destination = tmp_path / "contrast"
    EvidenceStore.open(source_package).materialize(recipe(), destination)
    path = destination / "typed/contrasts.parquet"
    table = pq.read_table(path)
    data = table.to_pylist()
    data[0]["target_log2_fold_change"] = 500.0
    pq.write_table(pa.Table.from_pylist(data, schema=table.schema), path)
    rehash(destination)
    with pytest.raises(ValueError, match="source replay"):
        verify_recipe_package(destination)


def test_original_is_snapshotted_and_overwrite_is_rejected(original, source_package):
    source, _ = original
    source.write_bytes(b"changed original")
    assert (
        package.verify_perturbation_quantity_package(source_package)["reconciliation"][
            "n_forms_rows"
        ]
        == 3
    )
    with pytest.raises(FileExistsError):
        package.write_perturbation_quantity_package(source.parent, source_package)


def test_wrong_original_and_unsafe_asset_names_fail(original, tmp_path):
    source, contract = original
    source.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="Pinned perturbation source differs"):
        package.write_perturbation_quantity_package(source.parent, tmp_path / "wrong")
    contract["assets"][0]["filename"] = "../source.xlsx"
    package.CONTRACT.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="safe basenames"):
        package.source_contract()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "not_reported"),
        ("", "not_reported"),
        ("0", "reported_zero"),
        ("1.25", "reported_finite"),
        ("-1", "invalid_numeric"),
        ("nan", "invalid_numeric"),
        ("inf", "invalid_numeric"),
        ("1e999", "invalid_numeric"),
        ("1e-999", "invalid_numeric"),
        ("text", "invalid_numeric"),
    ],
)
def test_numeric_status_never_imputes(raw, expected):
    assert numeric(raw)[1] == expected
    value, _status = numeric(raw)
    assert (
        value is None if expected in {"not_reported", "invalid_numeric"} else math.isfinite(value)
    )
    assert numeric("-1", signed=True) == (-1.0, "reported_finite")


@pytest.mark.parametrize("raw", ["AA[X]AA", "A[+80]AAA", "AAXAAA", "aAAAA", "Grand Total"])
def test_unknown_form_syntax_is_not_stripped(raw):
    with pytest.raises(ValueError, match="Unsupported source peptide form"):
        peptide_form(raw)


def test_fetch_uses_pinned_managed_downloads(original, tmp_path, monkeypatch):
    _, contract = original
    calls = []

    def download(url, destination, **kwargs):
        calls.append((url, destination, kwargs))
        return destination

    monkeypatch.setattr("hitlist.downloads.download_to_file", download)
    result = package.fetch_perturbation_sources(tmp_path / "cache", verbose=False)
    assert set(result) == {"source.xlsx"}
    assert calls[0][2]["expected_sha256"] == contract["assets"][0]["sha256"]
    assert calls[0][2]["expected_size"] == contract["assets"][0]["size_bytes"]


def test_default_contract_retains_every_admitted_comparison():
    contract = json.loads(
        (
            Path(__file__).parents[1]
            / "hitlist/data/evidence_revisions/perturbation_quantities.json"
        ).read_text()
    )
    assert len(contract["comparisons"]) == 12
    assert sum(c["n_forms"] for c in contract["comparisons"]) == 17763
    assert len({c["treated_condition_id"] for c in contract["comparisons"]}) == 12
    assert len({c["control_condition_id"] for c in contract["comparisons"]}) == 4
    assert (
        sum(
            c["dose_qualification"].startswith("source_header_conflict")
            for c in contract["comparisons"]
        )
        == 4
    )


def test_filemap_preserves_literal_label_and_checks_line_dose(tmp_path):
    source = tmp_path / "map.xlsx"
    book = openpyxl.Workbook()
    book.active.append(["line", "10 uM", "2.2000000000000002", "run.raw", "raw"])
    book.save(source)
    sheet = read_source_workbook(source)[0]
    comparison = {
        "asset": source.name,
        "sheet": "comparison",
        "line_raw": "line",
        "filemap_condition_raw": "10 uM",
    }
    contract = {"source_dataset": "fixture", "comparisons": [comparison]}
    recorded = []

    class Sink:
        def add(self, name, row):
            recorded.append((name, row))

    compiler = Compiler(contract, Sink())
    asset = {"filename": source.name, "role": "data3"}
    spec = {
        "line_column": "A",
        "condition_column": "B",
        "table_column": "C",
        "filename_column": "D",
        "type_column": "E",
        "blocks": [
            {
                "start_row": 1,
                "rows": [1],
                "line_raw": "line",
                "condition_raw": "10 uM",
                "table_raw": "2.2000000000000002",
                "comparison_sheet": "comparison",
            }
        ],
    }
    compiler.filemap(asset, sheet, spec)
    assert recorded[0][1]["source_table_label"] == "2.2000000000000002"
    comparison["filemap_condition_raw"] = "1 uM"
    with pytest.raises(ValueError, match="line/dose"):
        compiler.filemap(asset, sheet, spec)
    comparison["filemap_condition_raw"] = "10 uM"
    spec["blocks"][0]["table_raw"] = "2.2"
    with pytest.raises(ValueError, match="file-map block changed"):
        compiler.filemap(asset, sheet, spec)
