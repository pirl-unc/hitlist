"""Positive conditional composition cannot identify unobserved product mass."""

import copy
import hashlib
import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import mapp_composition as recipes


def _tables():
    tables = {name: [] for name in recipes.TABLES}
    for condition in recipes.CONDITIONS:
        for run in range(3):
            tables["channels"].append(
                {
                    "channel_id": f"{condition}:{run}",
                    "condition": condition,
                    "measured_species": "Homo sapiens",
                    "measured_cell": "HEK293",
                    "measured_pool": "proteasome_associated_fragment",
                    "material_link_precision": "acquisition_run_unresolved_specimen",
                }
            )
    for index, sequence in enumerate(("ACDEFGHIKLMNPQRSTVWY", "YWVTSRQPNMLKIHGFEDCA")):
        checksum = hashlib.sha256(sequence.encode()).hexdigest()
        rid = recipes.REFERENCE_ID + f":P{index}"
        tables["reference_proteins"].append(
            {
                "reference_id": recipes.REFERENCE_ID,
                "reference_record_id": rid,
                "sequence": sequence,
                "sequence_sha256": checksum,
            }
        )
        for position in range(3):
            pid = f"p{index}:{position}"
            start = position * 4
            tables["peptide_rows"].append(
                {
                    "peptide_row_id": pid,
                    "source_record_id": "source:" + pid,
                    "sequence_raw": sequence[start : start + 6],
                    "group_ids_raw": f"g{index}",
                    "pep": 0.001,
                    "reverse": False,
                    "potential_contaminant": False,
                }
            )
            tables["peptide_reference_coverage"].append(
                {
                    "reference_id": recipes.REFERENCE_ID,
                    "peptide_row_id": pid,
                    "shared_sequence_component_id": f"component:{index}",
                    "leading_coordinate_flanks_match": True,
                }
            )
            tables["peptide_reference_occurrences"].append(
                {
                    "reference_id": recipes.REFERENCE_ID,
                    "peptide_row_id": pid,
                    "reference_record_id": rid,
                    "start_1based": start + 1,
                }
            )
            for condition in recipes.CONDITIONS:
                intensity = 10000 * (position + 1 if condition == "UT" else 3 - position)
                pub = f"publisher:{pid}:{condition}"
                tables["publisher_quantities"].append(
                    {
                        "peptide_row_id": pid,
                        "measurement_id": pub,
                        "source_cell": pub,
                        "representation": "log10_median_filtered",
                        "condition": condition,
                        "value_status": "reported",
                        "value": math.log10(intensity),
                    }
                )
                for run in range(3):
                    channel = f"{condition}:{run}"
                    result = f"{pid}:{channel}"
                    mid = result + ":filtered"
                    tables["quantity_results"].append(
                        {"result_id": result, "peptide_row_id": pid, "channel_id": channel}
                    )
                    tables["quantity_measurements"].append(
                        {
                            "measurement_id": mid,
                            "result_id": result,
                            "peptide_row_id": pid,
                            "representation": "bead_filtered",
                            "value": intensity if run < 2 else 0,
                        }
                    )
                    tables["derivation_inputs"].append(
                        {
                            "derived_measurement_id": pub,
                            "input_measurement_id": mid,
                            "input_role": "median_run",
                        }
                    )
                    tables["evidence"].append(
                        {
                            "evidence_id": "e:" + result,
                            "peptide_row_id": pid,
                            "channel_id": channel,
                            "msms_count": int(run == 0),
                        }
                    )
    return tables


def test_conditional_quantities_keep_zero_run_contributors_and_ignore_parent_scale():
    tables = _tables()
    output = recipes.compile_mapp_recipe(tables)
    assert output["summary"]["n_admitted_peptides"] == 6
    assert output["summary"]["n_examples"] == 4
    assert len(output["contributors"]) == 12
    assert all(len(c["run_measurement_ids"]) == 3 for c in output["contributors"])
    assert all(len(c["direct_msms_channel_ids"]) == 1 for c in output["contributors"])
    for example in output["examples"]:
        probs = [r["target_probability"] for r in example["candidates"]]
        assert sum(probs) == pytest.approx(1)
        expected = [1 / 6, 2 / 6, 3 / 6] if example["condition"] == "UT" else [3 / 6, 2 / 6, 1 / 6]
        assert probs == pytest.approx(expected)
    for q in tables["publisher_quantities"]:
        if q["condition"] == "UT":
            q["value"] += 2
    scaled = recipes.compile_mapp_recipe(tables)
    for old, new in zip(output["examples"], scaled["examples"]):
        assert [r["target_probability"] for r in old["candidates"]] == pytest.approx(
            [r["target_probability"] for r in new["candidates"]]
        )


def test_alias_occurrences_do_not_duplicate_labels_or_assign_extra_weight():
    tables = _tables()
    duplicate = copy.deepcopy(tables["reference_proteins"][0])
    duplicate["reference_record_id"] += "-2"
    tables["reference_proteins"].append(duplicate)
    for occurrence in list(tables["peptide_reference_occurrences"]):
        if occurrence["reference_record_id"].endswith(":P0"):
            tables["peptide_reference_occurrences"].append(
                {**occurrence, "reference_record_id": duplicate["reference_record_id"]}
            )
    output = recipes.compile_mapp_recipe(tables)
    assert output["summary"]["n_admitted_peptides"] == 6
    assert len(output["contributors"]) == 12
    assert max(len(p["reference_record_ids"]) for p in output["parents"]) == 2
    assert all(
        len(c["start_positions_0based"]) == 1 for e in output["examples"] for c in e["candidates"]
    )


def test_excluded_bridges_still_constrain_splits_and_exclusions_are_explicit():
    tables = _tables()
    bridge = {
        **tables["peptide_rows"][0],
        "peptide_row_id": "bridge",
        "group_ids_raw": "g0;g1",
        "pep": None,
        "potential_contaminant": True,
    }
    tables["peptide_rows"].append(bridge)
    tables["peptide_reference_coverage"].append(
        {
            "reference_id": recipes.REFERENCE_ID,
            "peptide_row_id": "bridge",
            "shared_sequence_component_id": "unmatched",
            "leading_coordinate_flanks_match": False,
        }
    )
    output = recipes.compile_mapp_recipe(tables)
    assert len({r["component_id"] for r in output["peptide_dispositions"]}) == 1
    assert len({r["split"] for r in output["examples"]}) == 1
    row = next(r for r in output["peptide_dispositions"] if r["peptide_row_id"] == "bridge")
    assert row["status"] == "excluded"
    assert {
        "source_contaminant",
        "ambiguous_source_group",
        "peptide_confidence_unreported",
        "reference_unmapped",
    } <= set(row["reasons"])
    tables["peptide_rows"][0]["pep"] = 0.02
    output = recipes.compile_mapp_recipe(tables)
    excluded = {r["peptide_row_id"]: r["reasons"] for r in output["peptide_dispositions"]}
    assert "peptide_pep_above_threshold" in excluded["p0:0"]
    assert "fewer_than_three_eligible_peptides_per_parent" in excluded["p0:1"]


def test_bad_relationships_and_context_fail():
    tables = _tables()
    tables["derivation_inputs"][0]["input_measurement_id"] = tables["derivation_inputs"][1][
        "input_measurement_id"
    ]
    with pytest.raises(ValueError, match="three distinct"):
        recipes.compile_mapp_recipe(tables)


def test_two_peptide_extension_keeps_existing_splits_and_quantities():
    tables = _tables()
    tables["peptide_rows"][0]["pep"] = 0.02
    old = recipes.compile_mapp_recipe(tables)
    new = recipes.compile_mapp_recipe(tables, recipe_id=recipes.PAIR_RECIPE_ID)
    assert old["summary"]["n_admitted_peptides"] == 3
    assert new["summary"]["n_admitted_peptides"] == 5
    assert [
        (r["peptide_row_id"], r["component_id"], r["split"]) for r in old["peptide_dispositions"]
    ] == [(r["peptide_row_id"], r["component_id"], r["split"]) for r in new["peptide_dispositions"]]
    new_examples = {(r["parent_sequence_sha256"], r["condition"]): r for r in new["examples"]}
    for row in old["examples"]:
        extended = new_examples[row["parent_sequence_sha256"], row["condition"]]
        assert row["candidates"] == extended["candidates"]
        assert row["split"] == extended["split"]
    tables = _tables()
    tables["channels"][0]["measured_cell"] = "different cell"
    with pytest.raises(ValueError, match="Source contexts differ"):
        recipes.compile_mapp_recipe(tables)


def _store_tables(tmp_path, tables):
    from hitlist.distribution_recipe import TABLE_BINDINGS
    from hitlist.evidence_bindings import materialized_bindings
    from hitlist.evidence_store import EvidenceStore

    store = EvidenceStore.__new__(EvidenceStore)
    store.directory = tmp_path
    store.manifest = {"schema_version": 9, "mapp_canonical": {}}
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(json.dumps(store.manifest))
    bindings = materialized_bindings(9, package_features=["canonical_mapp"])
    store.bindings, store.schemas = {}, {}
    for table, rows in tables.items():
        name = TABLE_BINDINGS[table]
        path = tmp_path / bindings[name][0].materialized_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        store.bindings[name] = bindings[name]
        store.schemas[name] = pq.ParquetFile(path).schema_arrow
    return store


@pytest.mark.parametrize("recipe_id", list(recipes.RECIPES))
def test_registered_export_is_immutable_and_hash_bound(tmp_path, recipe_id):
    from hitlist.evidence_recipe import verify_recipe_package

    tables = _complete_tables() if recipe_id == recipes.RUN_RECIPE_ID else _tables()
    store = _store_tables(tmp_path / "source", tables)
    output = tmp_path / "output"
    recipe = {"schema_version": 1, "name": "fixture", "registered_recipe": recipe_id}
    with pytest.raises(ValueError, match="unknown"):
        store.materialize({**recipe, "registered_recipe": "invented"}, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="Invalid"):
        store.materialize({**recipe, "limit": 1}, output)
    store.materialize(recipe, output)
    verified = verify_recipe_package(output, source_package=store.directory)
    assert verified["output_kind"] == "registered_distribution"
    compiled = recipes.compile_mapp_recipe(tables, recipe_id=recipe_id)
    for name, values in compiled.items():
        if name == "summary":
            actual = json.loads((output / "summary.json").read_text())
        else:
            actual = [
                json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()
            ]
        assert actual == values
    with pytest.raises(FileExistsError):
        store.materialize(recipe, output)
    (output / "examples.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_recipe_package(output)
    contract = recipes.registered_contracts()[0]
    contract["selection"]["max_reported_peptide_pep"] = 1
    assert recipes.registered_contracts()[0]["selection"]["max_reported_peptide_pep"] == 0.01


def test_registered_recipe_missing_sources_fail_without_output(tmp_path):
    tables = _tables()
    del tables["evidence"]
    store = _store_tables(tmp_path / "source", tables)
    with pytest.raises(ValueError, match="not materialized"):
        store.materialize(
            {"schema_version": 1, "name": "fixture", "registered_recipe": recipes.RECIPE_ID},
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_mapp_query_keeps_result_measurement_and_parent_grains_separate(tmp_path):
    from hitlist.evidence_store import EvidenceRelationshipError

    store = _store_tables(tmp_path / "source", _tables())
    result = store.query(
        "mapp_quantity_result",
        select=["result_id", "peptide.peptide_sequence", "acquisition.organism_name"],
        joins=["peptide", "acquisition"],
        limit=None,
    ).table.to_pylist()
    assert len(result) == 36
    assert {row["acquisition.organism_name"] for row in result} == {"Homo sapiens"}
    with pytest.raises(EvidenceRelationshipError, match="many cardinality"):
        store.query(
            "mapp_quantity_result",
            select=["measurements.quantitative_value"],
            joins=["measurements"],
        )


def test_registered_verifier_recomputes_targets_after_rehashed_tampering(tmp_path):
    from hitlist.evidence_recipe import verify_recipe_package
    from hitlist.provenance import file_digest

    store = _store_tables(tmp_path / "source", _tables())
    output = tmp_path / "output"
    store.materialize(
        {"schema_version": 1, "name": "fixture", "registered_recipe": recipes.RECIPE_ID}, output
    )
    path = output / "examples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["candidates"][0]["target_probability"] = 0.999
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        verify_recipe_package(output)


def _complete_tables():
    tables = _tables()
    for channel in tables["channels"]:
        channel.update(
            raw_file="original:" + channel["channel_id"],
            replicate_label=int(channel["channel_id"][-1]) + 1,
        )
    for quantity in tables["quantity_measurements"]:
        if quantity["value"] == 0:
            quantity["value"] = 10000.0
        quantity["value_status"] = "passed_background_filter"
        quantity["unit"] = "arbitrary_intensity"
    return tables


def test_complete_run_recipe_preserves_observations_and_base_partitions():
    tables = _complete_tables()
    old = recipes.compile_mapp_recipe(tables, recipe_id=recipes.PAIR_RECIPE_ID)
    result = recipes.compile_mapp_recipe(tables, recipe_id=recipes.RUN_RECIPE_ID)
    assert result["summary"]["n_admitted_peptides"] == 6
    assert len(result["run_observations"]) == 36
    assert result["parents"] == old["parents"]
    assert result["peptide_dispositions"] == old["peptide_dispositions"]
    for a, b in zip(old["examples"], result["examples"]):
        assert a["candidates"] == b["candidates"] and a["split"] == b["split"]
    assert all(r["channel"]["raw_file"].startswith("original:") for r in result["run_observations"])
    assert all(
        r["measurement"]["unit"] == "arbitrary_intensity" for r in result["run_observations"]
    )
    shuffled = copy.deepcopy(tables)
    for values in shuffled.values():
        values.reverse()
    assert result == recipes.compile_mapp_recipe(shuffled, recipe_id=recipes.RUN_RECIPE_ID)
    assert old == recipes.compile_mapp_recipe(tables, recipe_id=recipes.PAIR_RECIPE_ID)


@pytest.mark.parametrize("missing", [0.0, None, float("nan")])
def test_complete_runs_keep_ineligible_values_and_recompute_candidate_normalization(missing):
    tables = _complete_tables()
    tables["quantity_measurements"][0]["value"] = missing
    output = recipes.compile_mapp_recipe(tables, recipe_id=recipes.RUN_RECIPE_ID)
    assert output["summary"]["n_admitted_peptides"] == 5
    assert output["summary"]["n_nonpositive_or_missing_runs"] == 1
    excluded = next(r for r in output["peptide_dispositions"] if r["peptide_row_id"] == "p0:0")
    assert excluded["reasons"] == ["incomplete_positive_filtered_runs"]
    observations = [r for r in output["run_observations"] if r["peptide_row_id"] == "p0:0"]
    assert len(observations) == 6 and all(not r["complete_peptide"] for r in observations)
    for example in output["examples"]:
        values = [10 ** c["source_log10_intensity"] for c in example["candidates"]]
        assert [c["target_probability"] for c in example["candidates"]] == pytest.approx(
            [v / sum(values) for v in values]
        )
    tables["quantity_measurements"][6]["value"] = 0
    output = recipes.compile_mapp_recipe(tables, recipe_id=recipes.RUN_RECIPE_ID)
    assert output["summary"]["n_admitted_peptides"] == 3
    survivor = next(r for r in output["peptide_dispositions"] if r["peptide_row_id"] == "p0:2")
    assert survivor["reasons"] == ["fewer_than_two_complete_peptides_per_parent"]
    assert len(output["run_observations"]) == 36


def test_duplicate_run_evidence_fails():
    tables = _complete_tables()
    tables["evidence"].append(copy.deepcopy(tables["evidence"][0]))
    with pytest.raises(ValueError, match="Duplicate run evidence"):
        recipes.compile_mapp_recipe(tables, recipe_id=recipes.RUN_RECIPE_ID)
