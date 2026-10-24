"""Native response grouping, reference alternatives and source-preserving targets."""

import copy
import gzip
import hashlib
import json

import pytest

from hitlist.evidence_recipe import RecipeError, verify_recipe_package
from hitlist.tmab_response_recipe import RECIPE_ID, _load_tables, _proteins, compile_responses
from tests import test_tmab_semantics as source_fixtures

source_package = source_fixtures.source_package


def read(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "human.fasta"
    path.write_text(
        ">sp|P1|ONE_HUMAN protein OS=Homo sapiens OX=9606 GN=ONE SV=1\n"
        "AALGDGPKLVCCPEPTIDEYYMAGM\n"
        ">sp|P2|TWO_HUMAN protein OS=Homo sapiens OX=9606 GN=TWO SV=1\n"
        "TTLGDGPKLVGGLGDGPKLV\n"
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_url": "https://example.invalid/human.fasta",
        "generation_label": "explicit_test_reference",
        "license": "test_fixture",
        "role": "model_reference_not_original_search",
    }


def recipe(reference):
    return {
        "schema_version": 1,
        "name": "native",
        "registered_recipe": RECIPE_ID,
        "reference": reference,
    }


def test_public_recipe_preserves_sources_and_reuse_contributors(
    source_package, reference, tmp_path
):
    source, store = source_package
    path = tmp_path / "responses"
    manifest_path = store.materialize(recipe(reference), path)
    manifest = verify_recipe_package(path, source_package=source)
    assert manifest_path == path / "manifest.json"
    assert manifest["output_kind"] == "registered_native_response"
    summary = manifest["coverage"]
    assert summary["n_source_measurement_slots"] == 72
    assert summary["n_admitted_source_slots"] == 12
    assert summary["n_effective_observations"] == 6
    assert summary["n_response_groups"] == 3
    assert len(read(path / "dispositions.json.gz")) == 72
    observations = [o for g in read(path / "examples.json.gz") for o in g["observations"]]
    assert all(o["n_source_slots"] == 2 and len(o["measurement_ids"]) == 2 for o in observations)
    assert {o["quantitative_value"] for o in observations} == {1.2, 0.8}
    assert {o["independent_specimen_status"] for o in observations} == {"unknown"}
    contracts = store.catalog()["registered_recipes"]
    assert any(c["contract"]["recipe_id"] == RECIPE_ID for c in contracts)


def test_all_exact_parent_alternatives_and_overlapping_positions_remain(source_package, reference):
    _, store = source_package
    result = compile_responses(_load_tables(store), _proteins(reference["path"]))
    origins = [r for r in result["occurrences"] if r["peptide_sequence"] == "LGDGPKLV"]
    assert {(r["reference_accession"], r["start_1based"]) for r in origins} == {
        ("P1", 3),
        ("P2", 3),
        ("P2", 13),
    }
    assert {r["origin_status"] for r in origins} == {"candidate_not_measured_origin"}
    # The source's description 'parent' is preserved, never interpreted as P1.
    assert {
        a["parent_description_raw"]
        for a in result["assertions"]
        if a["source_row_disposition"] == "primary_ratio_panel"
    } == {"parent"}
    mapped_rows = {
        a["source_record_id"]
        for a in result["assertions"]
        if a["peptide_sequence"] in {"LGDGPKLV", "PEPTIDE", "MAGM"}
    }
    assert (
        len({c["split"] for c in result["components"] if c["source_record_id"] in mapped_rows}) == 1
    )


def test_condition_form_and_contributor_weights_do_not_count_copies_twice(
    source_package, reference
):
    _, store = source_package
    result = compile_responses(_load_tables(store), _proteins(reference["path"]))
    examples = result["examples"]
    assert sum(g["group_weight"] for g in examples) == pytest.approx(1)
    for condition in {g["condition_id"] for g in examples}:
        groups = [g for g in examples if g["condition_id"] == condition]
        assert sum(g["group_weight"] for g in groups) == pytest.approx(0.5)
        assert sum(o["weight"] for g in groups for o in g["observations"]) == pytest.approx(0.5)
        for group in groups:
            assert sum(o["weight"] for o in group["observations"]) == pytest.approx(
                group["group_weight"]
            )


def test_bounds_are_not_averaged_into_exact_labels(source_package, reference):
    _, store = source_package
    tables = _load_tables(store)
    for m in tables["measurements"]:
        if m["eligibility_status"] == "candidate_exact" and m["replicate_label"] == "Rep2":
            m.update(
                quantitative_raw=">5",
                quantitative_value=5.0,
                measurement_inequality="gt",
                censoring_status="source_declared_bound",
                eligibility_status="candidate_censored",
            )
    result = compile_responses(tables, _proteins(reference["path"]))
    for group in result["examples"]:
        assert {o["censoring_status"] for o in group["observations"]} == {
            "exact",
            "source_declared_bound",
        }
        bound = next(o for o in group["observations"] if o["inequality"] == "gt")
        assert bound["quantitative_value"] == 5
        assert bound["target_log10_ratio"] == pytest.approx(0.6989700043360189)


def test_possible_reuse_with_different_labels_is_deferred_without_split_leakage(
    source_package, reference
):
    _, store = source_package
    tables = _load_tables(store)
    target = next(m for m in tables["measurements"] if m["peptide_sequence"] == "LGDGPKLV")
    candidate = next(
        r["candidate_reuse_group_id"]
        for r in tables["lineage"]
        if r["measurement_id"] == target["measurement_id"]
    )
    for line in tables["lineage"]:
        if line["candidate_reuse_group_id"] == candidate:
            line["experiment_label_agreement"] = "different_source_labels"
    result = compile_responses(tables, _proteins(reference["path"]))
    assert "LGDGPKLV" not in {g["peptide_sequence"] for g in result["examples"]}
    assert result["summary"]["n_conflicting_reuse_groups"] == 1
    assert (
        result["summary"]["defer_reason_counts"]["deferred_conflicting_possible_reuse_context"] == 4
    )
    assert len(result["components"]) == len(tables["source_rows"])


def test_deferred_identity_can_still_link_parent_components(source_package, reference):
    _, store = source_package
    tables = _load_tables(store)
    # Remove name/gene links so only sequence/reference relationships can connect.
    for m in tables["measurements"]:
        m["parent_label_raw"] = m["source_record_id"]
        m["gene_name"] = None
        if m["peptide_sequence"] == "PEPTIDE":
            m["eligibility_status"] = "deferred_tentative"
            m["sequence_identity_status"] = "tentative"
    result = compile_responses(tables, _proteins(reference["path"]))
    components = {c["source_record_id"]: c["component_id"] for c in result["components"]}
    connected = [
        a for a in result["assertions"] if a["peptide_sequence"] in {"PEPTIDE", "LGDGPKLV", "MAGM"}
    ]
    assert len({components[a["source_record_id"]] for a in connected}) == 1
    assert "PEPTIDE" not in {g["peptide_sequence"] for g in result["examples"]}


def test_source_tables_are_not_mutated(source_package, reference):
    _, store = source_package
    tables = _load_tables(store)
    before = copy.deepcopy(tables)
    compile_responses(tables, _proteins(reference["path"]))
    assert tables == before


def test_reference_hash_and_output_tampering_fail(source_package, reference, tmp_path):
    source, store = source_package
    wrong = {**reference, "sha256": "0" * 64}
    with pytest.raises(RecipeError, match="hash changed"):
        store.materialize(recipe(wrong), tmp_path / "wrong")
    path = tmp_path / "native"
    store.materialize(recipe(reference), path)
    with (path / "model_reference.fasta").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(RecipeError, match="artifact mismatch"):
        verify_recipe_package(path, source_package=source)


def test_nonhuman_reference_is_not_silently_used(source_package, reference, tmp_path):
    _, store = source_package
    path = tmp_path / "mouse.fasta"
    path.write_text(">sp|P|P_MOUSE protein OX=10090 SV=1\nPEPTIDE\n")
    wrong = {
        **reference,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="explicitly human"):
        store.materialize(recipe(wrong), tmp_path / "wrong-species")
