"""Synthetic selection/graph fixtures and registered-package semantic replay."""

import json
from copy import deepcopy

import pytest

from hitlist import lymph_composition as recipe
from hitlist import nanaware_lymph as source
from hitlist.evidence_recipe import RecipeError, materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest
from tests import test_nanaware_lymph as source_tests


@pytest.fixture
def original(tmp_path, monkeypatch):
    return source_tests.original.__wrapped__(tmp_path, monkeypatch)


def fixture():
    tables = {name: [] for name in recipe.TABLE_BINDINGS}
    tables["references"] = [
        {
            "reference_record_id": "ref",
            "source_file": "PXD051024.mzid",
            "accession_raw": "P00001|HOST",
            "reference_sequence": "MADEFGHIKLMN",
        }
    ]
    tables["samples"] = [{"sample_id": "sample", "context_id": "context"}]
    tables["contexts"] = [{"context_id": "context"}]
    for index, (seq, confidence, area) in enumerate(
        [("ADE", 20, 10), ("DEF", 25, 100), ("EFG", 30, 1000), ("ADE", 10, 1e9)]
    ):
        key = str(index)
        tables["observations"].append(
            {
                "observation_id": key,
                "source_record_id": key,
                "block_id": "human_native",
                "peptide_sequence": seq,
                "peptide_raw": "K." + seq + ".R",
                "peptide_modifications_raw": "",
                "parent_label_raw": "sp|P00001|HOST",
                "confidence_json": json.dumps({"minus_10lgP": confidence}),
                "charge_raw": "2",
                "length_status": "conflicting",
                "start_raw": "999",
                "end_raw": "999",
            }
        )
        tables["source_records"].append({"source_record_id": key})
        tables["quantities"].append(
            {
                "measurement_id": key,
                "observation_id": key,
                "source_record_id": key,
                "sample_id": "sample",
                "context_id": "context",
                "quantity_kind": "area",
                "quantitative_value": area,
                "value_status": "positive",
                "source_cell": "K" + key,
            }
        )
    return tables


def test_selection_targets_and_real_positions_ignore_bad_annotation():
    data = fixture()
    result = recipe.compile_composition(data)
    assert [r["observation_id"] for r in result["examples"]] == ["0", "1", "2"]
    assert [r["target_centered_log10_area"] for r in result["examples"]] == [-1, 0, 1]
    assert sum(r["target_area_fraction"] for r in result["examples"]) == pytest.approx(1)
    assert result["examples"][0]["occurrences_0based_halfopen"] == [[1, 4]]
    data["quantities"][3]["quantitative_value"] = 1e30
    assert recipe.compile_composition(data)["examples"] == result["examples"]
    assert len(result["dependence_groups"]) == 1


def test_missing_selected_feature_does_not_fall_back_to_other_charge():
    data = fixture()
    data["quantities"][0].update(quantitative_value=None, value_status="missing")
    result = recipe.compile_composition(data)
    assert result["examples"] == []
    assert result["summary"]["reason_counts"]["not_selected_representative_feature"] == 1
    assert result["summary"]["reason_counts"]["fewer_than_three_positive_selected_peptides"] == 2


def test_original_alternative_parent_and_excluded_bridge_remain_in_graph():
    data = fixture()
    data["references"].append(
        {
            "reference_record_id": "ref2",
            "source_file": "PXD051024.mzid",
            "accession_raw": "P00002",
            "reference_sequence": "MADEFGHIKLMNA",
        }
    )
    data["identifications"].append(
        {
            "reference_record_id": "ref2",
            "source_file": "PXD051024.mzid",
            "peptide_sequence": "ADE",
            "is_decoy_raw": "false",
        }
    )
    result = recipe.compile_composition(data)
    assert result["examples"] == []
    assert len(result["dependence_groups"]) == 1
    assert (
        len(
            [node for node in result["dependence_groups"][0]["nodes"] if node.startswith("parent:")]
        )
        == 2
    )
    assert result["summary"]["reason_counts"]["missing_or_ambiguous_original_parent"] == 2


@pytest.mark.parametrize(
    "raw,seq,annotation,expected",
    [
        ("K.ADE.R", "ADE", "", True),
        ("AC(+57.02)D", "ACD", "Carbamidomethylation (C)", True),
        ("ACD", "ACD", "", False),
        ("AM(+15.99)D", "AMD", "Oxidation (M)", False),
        ("AC(+57.02)C", "ACC", "", False),
        ("ADE", "ADE", "Unknown", False),
    ],
)
def test_observer_chemistry(raw, seq, annotation, expected):
    assert (
        recipe._chemistry(
            {"peptide_raw": raw, "peptide_sequence": seq, "peptide_modifications_raw": annotation}
        )
        is expected
    )


def test_permutation_does_not_change_membership_or_splits():
    data = fixture()
    expected = recipe.compile_composition(data)
    shuffled = deepcopy(data)
    for rows in shuffled.values():
        rows.reverse()
    assert recipe.compile_composition(shuffled) == expected


def test_registered_api_empty_scope_and_rehashed_semantic_tamper(original, tmp_path):
    package, output = tmp_path / "package", tmp_path / "recipe"
    source.write_nanaware_lymph_package(original, package)
    store = EvidenceStore.open(package)
    spec = {"schema_version": 1, "name": "fixture", "registered_recipe": recipe.RECIPE_ID}
    materialize_recipe(store, spec, output)
    assert (
        verify_recipe_package(output, source_package=package)["coverage"]["n_admitted_measurements"]
        == 0
    )
    with pytest.raises(FileExistsError):
        materialize_recipe(store, spec, output)
    path = output / "examples.json"
    path.write_text('[{"invented_target":1}]')
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["examples.json"] = file_digest(output / "examples.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="replay differs"):
        verify_recipe_package(output, source_package=package)
