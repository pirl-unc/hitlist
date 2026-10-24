"""Observed-set targets, excluded bridges, support refusal and source replay."""

import csv
import hashlib
import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import goldberg_mapp as source
from hitlist.evidence_recipe import RecipeError, materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.goldberg_composition import LABEL_FIELDS, RECIPE_ID, dependence_components
from hitlist.provenance import file_digest
from tests import test_goldberg_mapp as source_tests

original_source = source_tests.original
tsv = source_tests.tsv


@pytest.fixture
def populated(original_source):
    original = original_source
    contract = json.loads(source.CONTRACT.read_text())
    archive_path = original / contract["archive"]
    with zipfile.ZipFile(archive_path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    rows = list(csv.DictReader(io.StringIO(files["peptides.txt"].decode()), delimiter="\t"))
    for native, sequence, intensity in [("3", "CCDEFGH", "100"), ("4", "DCDEFGH", "1000")]:
        row = dict(rows[0])
        row.update(
            {
                "id": native,
                "Sequence": sequence,
                "Intensity": intensity,
                "Intensity sourceA_Ctrl_1": intensity,
                "Evidence IDs": "",
                "Mod. peptide IDs": "",
            }
        )
        rows.append(row)
    files["peptides.txt"] = tsv(rows).encode()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    for asset in contract["assets"]:
        asset.update(file_digest(original / asset["filename"]))
    source.CONTRACT.write_text(json.dumps(contract))
    return original


def recipe():
    return {"schema_version": 1, "name": "test", "registered_recipe": RECIPE_ID}


def test_targets_weights_label_free_export_and_rehashed_tamper(populated, tmp_path):
    package, output = tmp_path / "package", tmp_path / "output"
    source.write_goldberg_mapp_package(populated, package)
    store = EvidenceStore.open(package)
    assert any(
        r["contract"]["recipe_id"] == RECIPE_ID and r["status"] == "materialized"
        for r in store.catalog()["registered_recipes"]
    )
    materialize_recipe(store, recipe(), output)
    manifest = verify_recipe_package(output, source_package=package)
    assert not manifest["coverage"]["support_passed"]
    path = output / "composition/typed/examples.parquet"
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda r: r["quantitative_value"])
    assert len(rows) == 3
    assert [r["target_centered_log10_intensity"] for r in rows] == [-1, 0, 1]
    assert [r["target_intensity_fraction"] for r in rows] == pytest.approx(
        [1 / 111, 10 / 111, 100 / 111]
    )
    assert [r["weight"] for r in rows] == [1 / 3] * 3
    assert len({r["component_id"] for r in rows}) == len({r["split"] for r in rows}) == 1
    assert all(
        r["composition_eligibility_status"] == "conditional_observed_signal_only" for r in rows
    )
    assert all(
        r["training_eligibility_status"] == "not_fully_qualified_processing_supervision"
        for r in rows
    )
    covariates = pq.read_table(output / "composition/typed/covariates.parquet")
    assert not (set(covariates.column_names) & LABEL_FIELDS)
    assert covariates.num_rows == 3
    decisions = pq.read_table(output / "composition/typed/decisions.parquet").to_pylist()
    assert len(decisions) == 15
    assert sum(r["status"] == "admitted" for r in decisions) == 3
    # An excluded, ambiguous source row connects the P1 and P2 origins.
    nodes = pq.read_table(output / "composition/typed/components.parquet").to_pylist()
    assert len({r["component_id"] for r in nodes}) == 1
    assert {r["node_id"] for r in nodes} >= {"parent:P1", "parent:P2"}
    rows[0]["target_centered_log10_intensity"] = -999
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    manifest["files"][str(path.relative_to(output))] = file_digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="rows differ"):
        verify_recipe_package(output, source_package=package)


def test_small_groups_retained_as_exclusions(original_source, tmp_path):
    original = original_source
    package, output = tmp_path / "package", tmp_path / "output"
    source.write_goldberg_mapp_package(original, package)
    materialize_recipe(EvidenceStore.open(package), recipe(), output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_eligible_before_group"] == summary["n_small_groups"] == 1
    assert summary["output_counts"]["n_examples_rows"] == 0
    assert summary["reason_counts"]["fewer_than_three_distinct_sequences_in_group"] == 1
    assert not summary["support_passed"]


def test_components_are_order_independent_and_include_excluded_bridges():
    rows = {
        0: {"peptide_sequence": "AAA", "parent_label_raw": "P1"},
        1: {"peptide_sequence": "CCC", "parent_label_raw": "P2"},
        2: {"peptide_sequence": "DDD", "parent_label_raw": "P1;P2"},
        3: {"peptide_sequence": "EEE", "parent_label_raw": "P3"},
    }
    actual = dependence_components(rows)
    assert actual == dependence_components(dict(reversed(list(rows.items()))))
    members = ["parent:P1", "parent:P2", "peptide:AAA", "peptide:CCC", "peptide:DDD"]
    digest = hashlib.sha256(
        ("goldberg2025_baseline_v1:" + json.dumps(members, separators=(",", ":"))).encode()
    ).hexdigest()
    assert actual["parent:P1"][0] == actual["parent:P2"][0] == digest
    assert actual["parent:P3"][0] != digest


def test_duplicate_sequence_aggregation_refused(populated, tmp_path, monkeypatch):
    from hitlist import goldberg_composition as composition

    package = tmp_path / "package"
    source.write_goldberg_mapp_package(populated, package)
    original_rows = composition._rows

    def duplicate(directory, name):
        rows = list(original_rows(directory, name))
        if name == "decisions":
            rows.append(next(dict(row) for row in rows if row["status"] == "candidate"))
        return iter(rows)

    monkeypatch.setattr(composition, "_rows", duplicate)
    with pytest.raises(ValueError, match="Duplicate peptide"):
        materialize_recipe(EvidenceStore.open(package), recipe(), tmp_path / "output")
    assert not (tmp_path / "output").exists()
