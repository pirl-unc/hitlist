"""Conditional response admission cannot turn ambiguous forms into molar labels."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import ms_calibration_response as recipe
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_recipe import verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.ms_calibration_evidence import SCHEMAS
from hitlist.provenance import file_digest


def _tables():
    tables = {name: [] for name in recipe.TABLE_BINDINGS}
    for index, sequence in enumerate(("ACDEFGHIK", "LMNPQRSTK", "FGHIKLMNK", "QRSTVWYAK")):
        standard_id = f"standard:{index}"
        tables["nominal_standards"].append(
            {
                "standard_id": standard_id,
                "peptide_sequence": sequence,
                "form_label_raw": sequence + "[+08]",
                "group_label": "A",
            }
        )
        n_forms = 2 if index == 3 else 1
        for form in range(n_forms):
            reporter_id = f"reporter:{index}:{form}"
            label = "protein." + sequence + ("" if index == 1 else "[+08]")
            aggregation = "sum4" if index == 2 else None
            if aggregation:
                label += ".sum4"
            if index == 3:
                label += f"[form{form}]"
            tables["reporter_forms"].append(
                {
                    "reporter_id": reporter_id,
                    "standard_id": standard_id,
                    "peptide_sequence": sequence,
                    "label_raw": label,
                    "aggregation_label_raw": aggregation,
                    "annotations_json": json.dumps(["+08"]),
                }
            )
            for slot, value in enumerate((100.0, 10000.0, None)):
                row = {
                    "measurement_id": f"measurement:{index}:{form}:{slot}",
                    "reporter_id": reporter_id,
                    "slot_id": f"slot:{slot}",
                    "nominal_load_id": "load:A:S4",
                    "value": value,
                    "representation": "raw",
                    "unit": "arbitrary_peak_area",
                    "value_status": "reported_positive" if value else "absent_source_cell",
                    "source_cell": f"cell:{index}:{form}:{slot}" if value else None,
                    "source_locator": f"source.xlsx#{index}:{form}:{slot}",
                }
                tables["s4_measurements"].append(row)
                tables["s4_measurements"].append(
                    {
                        **row,
                        "measurement_id": row["measurement_id"] + ":normalized",
                        "representation": "normalized",
                        "value": 1e9,
                    }
                )
    tables["reporter_forms"].append(
        {
            **tables["reporter_forms"][0],
            "reporter_id": "curve-only",
            "label_raw": "protein.ACDEFGHIK[+08].sum4",
            "aggregation_label_raw": "sum4",
        }
    )
    tables["nominal_loads"] = [
        {
            "nominal_load_id": "load:A:S4",
            "sample_label": "S4",
            "group_label": "A",
            "value": 10.0,
            "unit": "fmol_on_column",
            "source_cell": "load-source",
        }
    ]
    tables["source_slots"] = [
        {
            "slot_id": f"slot:{i}",
            "site_label": "1",
            "sample_label": "S4",
            "source_slot_ordinal": i + 1,
        }
        for i in range(3)
    ]
    return tables


def _store(directory, tables):
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory = directory
    store.bindings = {}
    store.schemas = {}
    all_bindings = materialized_bindings(8, appended_source_ids=["pxd004886"])
    files = {}
    for table, rows in tables.items():
        name = recipe.TABLE_BINDINGS[table]
        binding = all_bindings[name]
        path = directory / binding[0].materialized_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[table]), path)
        store.bindings[name] = binding
        store.schemas[name] = SCHEMAS[table]
        files[str(path.relative_to(directory))] = file_digest(path)
    store.manifest = {"schema_version": 8, "files": files, "appended_sources": {"pxd004886": {}}}
    (directory / "manifest.json").write_text(json.dumps(store.manifest))
    return store


def test_source_grains_and_ambiguities_are_not_silent_molar_assumptions():
    tables = _tables()
    result = recipe.compile_response(tables)
    assert len(result["dispositions"]) == 15
    assert result["summary"]["n_out_of_scope_normalized_rows"] == 15
    assert len(result["examples"]) == 1
    example = result["examples"][0]
    assert example["target_log10_response"] == 3.0  # Median of logs, not log of median.
    assert example["nominal_log10_load"] == 1.0  # Not subtracted from the target.
    assert example["n_positive_source_slots"] == 2
    assert example["min_log10_response"] == 2.0 and example["max_log10_response"] == 4.0
    assert len(result["contributors"]) == 2
    reasons = result["summary"]["reason_counts"]
    assert reasons == {
        "multiple_raw_reporters_for_nominal_standard": 6,
        "unresolved_reporter_aggregation": 3,
        "unresolved_nominal_isotope_annotation": 3,
        "response_not_reported_positive_finite": 5,
    }
    excluded = [row for row in result["dispositions"] if len(row["reasons"]) == 2]
    assert excluded and all(row["example_id"] is None for row in excluded)
    assert all(
        row["source_value_status"] == "absent_source_cell"
        for row in result["dispositions"]
        if row["source_value"] is None
    )


def test_group_assignments_ignore_response_values_and_source_order():
    tables = _tables()
    before = recipe.compile_response(tables)
    for row in tables["s4_measurements"]:
        if row["value"] is not None:
            row["value"] *= 123
    for rows in tables.values():
        rows.reverse()
    after = recipe.compile_response(tables)
    assert [(r["measurement_id"], r["fold_id"]) for r in before["dispositions"]] == [
        (r["measurement_id"], r["fold_id"]) for r in after["dispositions"]
    ]
    folds = {}
    for row in after["dispositions"]:
        folds.setdefault(row["sequence"], set()).add(row["fold_id"])
    assert all(len(x) == 1 for x in folds.values())


def test_invalid_nominal_units_or_duplicate_source_slots_fail():
    tables = _tables()
    tables["nominal_loads"][0]["unit"] = "nM"
    with pytest.raises(ValueError, match="unit"):
        recipe.compile_response(tables)
    tables = _tables()
    tables["s4_measurements"].append({**tables["s4_measurements"][0], "measurement_id": "another"})
    with pytest.raises(ValueError, match="source-slot"):
        recipe.compile_response(tables)


def test_canonical_export_checks_original_inputs_and_recomputed_targets(tmp_path):
    store = _store(tmp_path / "source", _tables())
    contract = {"schema_version": 1, "name": "fixture", "registered_recipe": recipe.RECIPE_ID}
    output = tmp_path / "output"
    assert recipe.describe(store)["status"] == "materialized"
    with pytest.raises(ValueError, match="Invalid"):
        store.materialize({**contract, "limit": 1}, output)
    store.materialize(contract, output)
    assert (
        verify_recipe_package(output, source_package=store.directory)["output_kind"]
        == "registered_response"
    )
    with pytest.raises(FileExistsError):
        store.materialize(contract, output)
    manifest = json.loads((output / "manifest.json").read_text())
    examples = json.loads((output / "examples.json").read_text())
    examples[0]["target_log10_response"] = 99.0
    (output / "examples.json").write_text(json.dumps(examples))
    manifest["files"]["examples.json"] = file_digest(output / "examples.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="derived contents"):
        verify_recipe_package(output)
    # Even internally consistent rehashed output cannot substitute changed source values.
    tables = json.loads((output / "source_tables.json").read_text())
    tables["s4_measurements"][0]["value"] = 123456.0
    compiled = recipe.compile_response(tables)
    for name, value in {"source_tables": tables, **compiled}.items():
        (output / f"{name}.json").write_text(json.dumps(value))
        manifest["files"][f"{name}.json"] = file_digest(output / f"{name}.json")
    manifest["coverage"] = compiled["summary"]
    (output / "manifest.json").write_text(json.dumps(manifest))
    verify_recipe_package(output)  # Standalone checks prove internal consistency only.
    with pytest.raises(ValueError, match="source rows changed"):
        verify_recipe_package(output, source_package=store.directory)


def test_missing_source_cannot_advertise_or_materialize_recipe(tmp_path):
    tables = _tables()
    del tables["source_assertions"]
    store = _store(tmp_path / "source", tables)
    assert recipe.describe(store)["status"] == "not_in_generation"
    with pytest.raises(ValueError, match="not materialized"):
        store.materialize(
            {"schema_version": 1, "name": "fixture", "registered_recipe": recipe.RECIPE_ID},
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()
