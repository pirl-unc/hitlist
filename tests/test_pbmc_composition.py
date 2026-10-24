"""PBMC admission includes excluded bridges and refuses ambiguous quantity aggregation."""

import csv
import hashlib
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.evidence_recipe import RecipeError, materialize_recipe, verify_recipe_package
from hitlist.mapp_pbmc_evidence import write_pbmc_tables
from hitlist.pbmc_composition import (
    CONTRACT,
    LABEL_FIELDS,
    RECIPE_ID,
    ROOT,
    TABLE_BINDINGS,
    compile_composition,
    dependence_components,
    describe,
)
from hitlist.provenance import file_digest
from tests.test_mapp_pbmc_evidence import _fixture, _write


def read(root, name):
    return pq.read_table(root / f"{name}.parquet").to_pylist()


@pytest.fixture
def source(tmp_path):
    package = tmp_path / "source"
    directory = _fixture(package / ROOT)
    with (directory / "peptides.txt").open() as stream:
        template = next(csv.DictReader(stream, delimiter="\t"))
    with (directory / "evidence.txt").open() as stream:
        evidence_template = list(csv.DictReader(stream, delimiter="\t"))[:14]
    sequences = [
        "ACDEFGH",
        "CCDEFGH",
        "DCDEFGH",
        "ECDEFGH",
        "ECDEFGH",
        "FCDEFGH",
        "GCDEFGH",
        "HCDEFGH",
    ]
    peptides, evidence = [], []
    for i, sequence in enumerate(sequences):
        peptide = {
            **template,
            "id": str(i),
            "Sequence": sequence,
            "Evidence IDs": ";".join(str(14 * i + j) for j in range(14)),
        }
        if i == 5:
            peptide["Proteins"] = "P1;P2"  # Excluded bridge remains in dependence graph.
        if i == 6:
            peptide["Reverse"] = "?"  # Unknown is not explicitly blank.
        if i == 7:
            peptide["PEP"] = "0.01"
        for j, original in enumerate(evidence_template):
            value = str(10 ** (i + 1))
            experiment = original["Experiment"]
            peptide["Intensity " + experiment] = value
            peptide["Identification type " + experiment] = "By MS/MS"
            evidence.append(
                {
                    **original,
                    "id": str(14 * i + j),
                    "Peptide ID": str(i),
                    "Sequence": sequence,
                    "Intensity": value,
                }
            )
        peptides.append(peptide)
    _write(directory / "peptides.txt", peptides)
    _write(directory / "evidence.txt", evidence)
    with (directory / "proteinGroups.txt").open() as stream:
        protein_groups = list(csv.DictReader(stream, delimiter="\t"))
    protein_groups[0]["Peptide IDs"] = ";".join(str(i) for i in range(len(sequences)))
    _write(directory / "proteinGroups.txt", protein_groups)
    write_pbmc_tables(package / ROOT)
    manifest = {
        "schema_version": 9,
        "files": {
            str(p.relative_to(package)): file_digest(p)
            for p in sorted(package.rglob("*"))
            if p.is_file()
        },
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    return package


def store(package):
    # Recipe-envelope unit fixture; full source/store validation is exercised on the real package.
    return SimpleNamespace(
        directory=package,
        bindings=dict.fromkeys(TABLE_BINDINGS.values()),
        manifest=json.loads((package / "manifest.json").read_text()),
        _software_version=lambda: "fixture",
        _manifest_hash=lambda: file_digest(package / "manifest.json")["sha256"],
    )


def recipe():
    return {"schema_version": 1, "name": "test", "registered_recipe": RECIPE_ID}


def test_quantities_weights_duplication_and_source_replay(source, tmp_path):
    output = tmp_path / "recipe"
    assert describe(store(source))["status"] == "materialized"
    materialize_recipe(store(source), recipe(), output)
    manifest = verify_recipe_package(output, source_package=source)
    summary = manifest["coverage"]
    assert summary["n_eligible_before_group"] == 70
    assert summary["n_duplicate_measurements"] == 28
    assert summary["support_passed"] == {"CTRL": False, "LUPUS": False}
    assert summary["cohort_support"]["CTRL"]["n_observations"] == 18
    assert summary["cohort_support"]["LUPUS"]["n_observations"] == 24
    rows = read(output / "composition/typed", "examples")
    for cohort in CONTRACT["cohorts"]:
        selected = [r for r in rows if r["cohort"] == cohort]
        assert sum(r["weight"] for r in selected) == pytest.approx(1)
        for group in {r["group_id"] for r in selected}:
            group_rows = sorted(
                [r for r in selected if r["group_id"] == group],
                key=lambda r: r["quantitative_value"],
            )
            assert [r["target_centered_log10_intensity"] for r in group_rows] == [-1, 0, 1]
            assert [r["target_intensity_fraction"] for r in group_rows] == pytest.approx(
                [1 / 111, 10 / 111, 100 / 111]
            )
    nodes = read(output / "composition/typed", "components")
    assert len({r["component_id"] for r in nodes}) == 1
    assert "parent:P2" in {r["node_id"] for r in nodes}
    decisions = read(output / "composition/typed", "decisions")
    assert len(decisions) == 112
    assert sum(r["status"] == "admitted" for r in decisions) == 42
    assert (
        not set(pq.read_schema(output / "composition/typed/covariates.parquet").names)
        & LABEL_FIELDS
    )
    path = output / "composition/typed/examples.parquet"
    rows[0]["target_intensity_fraction"] = 0.999
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    manifest["files"][str(path.relative_to(output))] = file_digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="rows or schema differ"):
        verify_recipe_package(output, source_package=source)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("n_positive_direct_msms_evidence", 0, "no_positive_direct_MS_MS_evidence"),
        ("n_numeric_evidence", 0, "incomplete_evidence_intensities"),
    ],
)
def test_evidence_support_exclusions(source, tmp_path, field, value, reason):
    path = source / ROOT / "typed/quantity_results.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if row["peptide_row_id"] == "pxd009438:peptide:0":
            row[field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    summary = compile_composition(path.parent, tmp_path / "out")
    assert summary["reason_counts"][reason] == 14
    assert summary["output_counts"]["n_examples_rows"] == 0
    assert summary["reason_counts"]["fewer_than_three_distinct_sequences_in_group"] == 28


def test_per_run_identification_is_required(source, tmp_path):
    path = source / ROOT / "typed/source_records.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        raw = json.loads(row["fields_json"])
        if "Identification type 985_CTRL_1" in raw:
            raw["Identification type 985_CTRL_1"] = "By matching"
            row["fields_json"] = json.dumps(raw)
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    summary = compile_composition(path.parent, tmp_path / "out")
    assert summary["reason_counts"]["not_direct_run_MS_MS_identification"] == 8
    assert summary["cohort_support"]["CTRL"]["n_groups"] == 5
    assert summary["cohort_support"]["LUPUS"]["n_groups"] == 8


def test_order_independent_graph_and_unavailable_recipe():
    rows = [
        {"sequence_raw": "AAA", "parent_labels_raw": "P1"},
        {"sequence_raw": "CCC", "parent_labels_raw": "P2"},
        {"sequence_raw": "DDD", "parent_labels_raw": "P1;P2"},
    ]
    result = dependence_components(rows)
    assert result == dependence_components(reversed(rows))
    nodes = ["parent:P1", "parent:P2", "peptide:AAA", "peptide:CCC", "peptide:DDD"]
    digest = hashlib.sha256(
        ("pxd009438_composition_v1:" + json.dumps(nodes, separators=(",", ":"))).encode()
    ).hexdigest()
    assert set(result.values()) == {digest}
    assert describe(SimpleNamespace(bindings={}))["status"] == "not_in_generation"
