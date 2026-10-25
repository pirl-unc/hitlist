"""Temporal eligibility, denominator identity, label separation and source replay."""

import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import intact_degradomics as source
from hitlist import intact_kinetics as kinetics
from hitlist.evidence_recipe import RecipeError, materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest
from tests import test_intact_degradomics as source_tests

original_source = source_tests.original


@pytest.fixture
def temporal(original_source):
    assets, package, contract = original_source
    sheets, inventory = {}, []
    for time_index, minutes in enumerate((0, 40, 100, 120)):
        for replicate in (1, 2):
            label = f"20S_AlphaSyn_{minutes}min_R{replicate}"
            rows = [source_tests.HEADERS]
            products = [("ACDEFGHIK", 1, 9, 10), ("ACDE", 1, 4, 100 * (time_index + 1))]
            if replicate == 1:
                products.append(("CDEF", 2, 5, 20 * (time_index + 1)))
            if minutes >= 100:
                products.append(("GHIK", 6, 9, 25))  # No fitting-time observation.
            for mass_id, (sequence, start, end, value) in enumerate(products, 1):
                intensity = value * replicate
                shifted = minutes == 40 and replicate == 1
                row = source_tests.search_rows(intensity=intensity, shift=shifted)[1]
                row[0] = row[6] = mass_id
                offset = int(shifted)
                row[14 + offset], row[15 + offset], row[17 + offset] = start, end, sequence
                rows.append(row)
            sheets[label] = rows
            inventory.append(
                {
                    "fileName": f"F2_AlphaSyn_20S_{minutes}min_R{replicate}.RAW",
                    "fileCategory": {"value": "RAW"},
                }
            )
    inventory.append(
        {"fileName": "F3_AlphaSyn_Brain20S_0min_R2.RAW", "fileCategory": {"value": "RAW"}}
    )
    source_tests.write_workbook(assets / "F2_SF6_SearchResults.xlsx", sheets)
    (assets / "files-project-route.json").write_text(json.dumps(inventory))
    for asset in contract["assets"]:
        asset.update(file_digest(assets / asset["filename"]))
    source.CONTRACT.write_text(json.dumps(contract))
    source.write_intact_degradomics_package(assets, package)
    return EvidenceStore.open(package)


def recipe():
    return {"schema_version": 1, "name": "temporal_test", "registered_recipe": kinetics.RECIPE_ID}


def read(output, name):
    return pq.read_table(output / f"{name}.parquet").to_pylist()


def test_public_recipe_ratios_denominators_weights_and_label_free_covariates(temporal, tmp_path):
    assert any(
        r["contract"]["recipe_id"] == kinetics.RECIPE_ID and r["status"] == "materialized"
        for r in temporal.catalog()["registered_recipes"]
    )
    output = tmp_path / "recipe"
    materialize_recipe(temporal, recipe(), output)
    manifest = verify_recipe_package(output, source_package=temporal.directory)
    assert not manifest["coverage"]["support_passed"]
    rows = read(output / "typed", "examples")
    assert len(rows) == 12
    assert {r["peptide_sequence"] for r in rows} == {"ACDE", "CDEF"}
    source_rows = {
        r["assignment_id"]: r
        for r in temporal.query("intact_assignments", limit=None).table.to_pylist()
    }
    for row in rows:
        parent = source_rows[row["normalizer_assignment_id"]]
        assert parent["acquisition_id"] == row["acquisition_id"]
        assert parent["peptide_sequence"] == "ACDEFGHIK"
        assert row["target_log10_ratio"] == pytest.approx(
            math.log10(row["quantitative_value"] / row["normalizer_value"])
        )
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split]
        assert sum(r["weight"] for r in selected) == pytest.approx(1)
        assert sum(
            r["weight"] for r in selected if r["peptide_sequence"] == "ACDE"
        ) == pytest.approx(0.5)
    train = [r for r in rows if r["split"] == "train"]
    assert {r["weight"] for r in train if r["peptide_sequence"] == "ACDE"} == {1 / 8}
    assert {r["weight"] for r in train if r["peptide_sequence"] == "CDEF"} == {1 / 4}
    covariates = pq.read_table(output / "typed/covariates.parquet")
    assert not set(covariates.column_names) & kinetics.LABEL_FIELDS
    decisions = read(output / "typed", "decisions")
    late = [r for r in decisions if r["peptide_sequence"] == "GHIK"]
    assert len(late) == 4 and all(
        "fewer_than_two_fitting_times" in json.loads(r["reasons_json"]) for r in late
    )
    assert sum(r["status"] == "normalizer_only" for r in decisions) == 8
    assert any("outside_kinetics_view" in json.loads(r["reasons_json"]) for r in decisions)
    with pytest.raises(FileExistsError):
        materialize_recipe(temporal, recipe(), output)


@pytest.mark.parametrize("mutation", ["no_denominator", "duplicate_product", "late_values"])
def test_source_conflicts_and_future_values_do_not_create_training_support(
    temporal, tmp_path, monkeypatch, mutation
):
    original = kinetics._rows
    tables = temporal.directory / source.ROOT / "typed"
    assignments = list(original(tables, "assignments"))
    views = {r["view_id"]: r for r in original(tables, "source_views")}
    acquisitions = {r["acquisition_id"]: r for r in original(tables, "acquisitions")}
    denominators = {
        r["measurement_id"] for r in assignments if r["peptide_sequence"] == "ACDEFGHIK"
    }
    late_ids = {
        r["measurement_id"]
        for r in assignments
        if acquisitions.get(r["acquisition_id"], {}).get("time_minutes", 0) >= 100
    }

    def changed(directory, name):
        for row in original(directory, name):
            if (
                name == "quantities"
                and mutation == "no_denominator"
                and row["measurement_id"] in denominators
            ):
                row = {**row, "quantitative_value": 0}
            if (
                name == "quantities"
                and mutation == "late_values"
                and row["measurement_id"] in late_ids
            ):
                row = {**row, "quantitative_value": row["quantitative_value"] * 10000}
            yield row
            if (
                name == "assignments"
                and mutation == "duplicate_product"
                and row["peptide_sequence"] == "ACDE"
                and views[row["view_id"]]["source_view_role"] == "kinetics"
            ):
                yield {**row, "assignment_id": row["assignment_id"] + "duplicate"}

    kinetics.compile_kinetics(tables, tmp_path / "before")
    monkeypatch.setattr(kinetics, "_rows", changed)
    summary = kinetics.compile_kinetics(tables, tmp_path / "after")
    before, after = read(tmp_path / "before", "examples"), read(tmp_path / "after", "examples")
    if mutation == "no_denominator":
        assert not after and summary["reason_counts"]["no_unique_admitted_intact_substrate"] == 16
    elif mutation == "duplicate_product":
        assert {r["peptide_sequence"] for r in after} == {"CDEF"}
        assert summary["reason_counts"]["multiple_masses_for_sequence_acquisition"] == 16
    else:
        assert [r for r in before if r["split"] == "train"] == [
            r for r in after if r["split"] == "train"
        ]
        assert {r["assignment_id"] for r in before} == {r["assignment_id"] for r in after}


def test_rehashed_target_tampering_fails_source_replay(temporal, tmp_path):
    output = tmp_path / "recipe"
    materialize_recipe(temporal, recipe(), output)
    path = output / "typed/examples.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["target_log10_ratio"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["typed/examples.parquet"] = file_digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="rows differ"):
        verify_recipe_package(output, source_package=temporal.directory)
