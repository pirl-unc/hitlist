"""Time-course admission preserves provenance and fails closed on unsupported curves."""

import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.digestion_recipe import RECIPE_ID, TABLE_BINDINGS, compile_timecourses


def fixture():
    tables = {name: [] for name in TABLE_BINDINGS}
    tables["constructs"] = [
        {"construct_id": "p1", "sequence": "ACDIKL", "sequence_sha256": "literal"}
    ]
    tables["contexts"] = [
        {
            "condition_id": "c1",
            "construct_id": "p1",
            "training_status": "pending_registered_recipe_and_independent_replay",
            "fields_json": "{}",
            "enzyme_species": "Homo sapiens",
        }
    ]
    for bio in ("1", "2"):
        tables["raw_quantities"].append(
            {
                "source_record_id": "raw" + bio,
                "condition_id": "c1",
                "peptide_sequence": "CDI",
                "biological_replicate_raw": bio,
                "run_id": "r" + bio,
                "quantity_time_hours": 0.0,
                "metadata_time_hours": 0.0,
                "time_status": "source_agreement",
                "intensity": 0.0,
            }
        )
        for tech in ("1", "2"):
            tables["filtered_vectors"].append(
                {
                    "source_record_id": "filtered" + bio + tech,
                    "condition_id": "c1",
                    "peptide_sequence": "CDI",
                    "biological_replicate_raw": bio,
                    "technical_replicate_raw": tech,
                    "times_raw": "0;1;2;4",
                }
            )
        for peptide in ("CDI", "CDL"):
            tables["final_vectors"].append(
                {
                    "source_record_id": "final" + bio + peptide,
                    "condition_id": "c1",
                    "construct_id": "p1",
                    "peptide_sequence": peptide,
                    "biological_replicate_raw": bio,
                    "times_raw": "0;1;2;4;0",
                    "intensities_raw": "0;10;20;15;0",
                    "product_type_raw": "PCP",
                    "splice_type_raw": "PCP",
                    "positions_raw": "2_4",
                    "row_status": "reported_vector",
                    "fields_json": "{}",
                }
            )
    return tables


def test_aliases_and_repeated_zero_slots_do_not_multiply_targets():
    compiled = compile_timecourses(fixture())
    assert compiled["summary"]["n_source_final_vectors"] == 4
    assert len(compiled["examples"]) == 2
    for curve in compiled["examples"]:
        assert curve["common_positive_times"] == [1.0, 2.0, 4.0]
        assert [r["target_value"] for r in curve["observations"]] == [10.0, 20.0, 15.0]
        assert all(len(r["source_slots"]) == 2 for r in curve["observations"])
        assert curve["peptide_sequence"] == "CDI" and curve["peptide_IL_sequence"] == "CDL"
        assert curve["start_0based"] == 1 and curve["end_0based"] == 4
        assert curve["actual_zero_raw_quantity_ids"]
        assert curve["context"]["enzyme_species"] == "Homo sapiens"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("mock_zero", "no_actual_zero_baseline_for_every_technical_label"),
        ("time_conflict", "unresolved_condition_time_conflict"),
        ("alias_conflict", "conflicting_IL_alias_or_repeated_slot_values"),
        ("mapping", "not_one_IL_equivalent_contiguous_span"),
        ("splice", "not_exclusively_source_contiguous"),
        ("missing", "fewer_than_three_positive_nonzero_times"),
    ],
)
def test_unsupported_measurements_exclude_with_exact_reason(mutation, reason):
    tables = fixture()
    if mutation == "mock_zero":
        tables["filtered_vectors"][0]["times_raw"] = "1;2;4"
    elif mutation == "time_conflict":
        tables["raw_quantities"][0]["time_status"] = "source_metadata_conflict"
        tables["raw_quantities"][0]["quantity_time_hours"] = 1.0
    elif mutation == "alias_conflict":
        tables["final_vectors"][0]["intensities_raw"] = "0;11;20;15;0"
    elif mutation == "mapping":
        tables["constructs"][0]["sequence"] = "ACDIKCDIKL"
    elif mutation == "splice":
        tables["final_vectors"][0]["splice_type_raw"] = "cis"
    elif mutation == "missing":
        for r in tables["final_vectors"]:
            r["intensities_raw"] = "0;NaN;20;15;0"
    compiled = compile_timecourses(tables)
    assert not compiled["examples"]
    assert reason in compiled["summary"]["reason_counts"]


def test_excluded_annotation_bridges_define_groups_before_outcomes():
    tables = fixture()
    tables["constructs"].append(
        {"construct_id": "p2", "sequence": "GCDLK", "sequence_sha256": "other"}
    )
    tables["contexts"].append(
        {
            "condition_id": "c2",
            "construct_id": "p2",
            "training_status": "pending",
            "fields_json": "{}",
        }
    )
    row = copy.deepcopy(tables["final_vectors"][0])
    row.update(
        {
            "source_record_id": "annotation",
            "condition_id": "c2",
            "construct_id": "p2",
            "row_status": "annotation_only",
            "biological_replicate_raw": "NA",
            "times_raw": "NA",
            "intensities_raw": "NA",
            "product_type_raw": "PSP",
        }
    )
    tables["final_vectors"].append(row)
    before = compile_timecourses(tables)
    assert len({r["component_id"] for r in before["components"]}) == 1
    for r in tables["final_vectors"][:-1]:
        r["intensities_raw"] = "0;100;200;150;0"
    assert compile_timecourses(tables)["components"] == before["components"]


def test_portable_roundtrip_rejects_changed_target_even_with_updated_hash(tmp_path):
    from hitlist.digestion_evidence import SCHEMAS
    from hitlist.evidence_bindings import materialized_bindings
    from hitlist.evidence_recipe import RecipeError, verify_recipe_package
    from hitlist.evidence_store import EvidenceStore
    from hitlist.provenance import file_digest

    package = tmp_path / "package"
    source = package / "external/soh2024_digestion/typed"
    source.mkdir(parents=True)
    for name, rows in fixture().items():
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]), source / f"{name}.parquet")
    envelope = {
        "schema_version": 9,
        "appended_sources": {"soh2024_digestion": {}},
        "files": {
            str(p.relative_to(package)): file_digest(p) for p in package.rglob("*") if p.is_file()
        },
    }
    (package / "manifest.json").write_text(json.dumps(envelope))
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory, store.manifest = package, envelope
    bindings = materialized_bindings(9, appended_source_ids=["soh2024_digestion"])
    store.bindings = {name: bindings[name] for name in TABLE_BINDINGS.values()}
    store.schemas = {
        name: pq.ParquetFile(package / binding[0].materialized_artifact).schema_arrow
        for name, binding in store.bindings.items()
    }
    output = tmp_path / "recipe"
    store.materialize(
        {"schema_version": 1, "name": "fixture", "registered_recipe": RECIPE_ID}, output
    )
    verify_recipe_package(output, source_package=package)
    p = output / "examples.json"
    examples = json.loads(p.read_text())
    examples[0]["observations"][0]["target_value"] += 1
    p.write_text(json.dumps(examples))
    p = output / "manifest.json"
    manifest = json.loads(p.read_text())
    manifest["files"]["examples.json"] = file_digest(output / "examples.json")
    p.write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="derived contents changed: examples"):
        verify_recipe_package(output, source_package=package)
