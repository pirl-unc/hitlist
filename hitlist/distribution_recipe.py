"""Registered distribution tasks behind the canonical recipe interface.

Row-wise and distribution targets have different grains. Both use the same
verified store and recipe envelope; distributions retain their observed sets.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .evidence_catalog import SCHEMA_VERSION
from .mapp_composition import RECIPES, RUN_RECIPE_ID, compile_mapp_recipe, registered_contracts
from .provenance import file_digest

TABLE_BINDINGS = {
    "peptide_rows": "mapp_peptide",
    "channels": "mapp_channel",
    "evidence": "mapp_identification",
    "quantity_results": "external_proteasome_fragment_quantity",
    "quantity_measurements": "mapp_quantity_measurement",
    "publisher_quantities": "mapp_publisher_quantity",
    "derivation_inputs": "mapp_derivation_input",
    "reference_proteins": "external_reference_protein",
    "peptide_reference_occurrences": "external_peptide_parent_occurrence",
    "peptide_reference_coverage": "external_peptide_reference_coverage",
}


def describe_registered_recipes(store):
    from .digestion_recipe import describe as describe_digestion
    from .hap1_response_recipe import describe as describe_hap1_response
    from .mapp_controlled_recipe import describe as describe_controlled
    from .ms_calibration_response import describe
    from .precursor_recipe import describe as describe_precursor
    from .proteometools_response import describe as describe_proteometools
    from .tmab_response_recipe import describe as describe_native_response

    available = set(TABLE_BINDINGS.values()) <= store.bindings.keys()
    return [
        {"contract": contract, "status": "materialized" if available else "not_in_generation"}
        for contract in registered_contracts()
    ] + [
        describe(store),
        describe_controlled(store),
        describe_controlled(store, contrast=True),
        describe_digestion(store),
        describe_native_response(store),
        describe_precursor(store),
        describe_hap1_response(store),
        describe_proteometools(store),
    ]


def validate_registered_recipe(recipe):
    from .evidence_recipe import RecipeError

    if (
        set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe["schema_version"] != 1
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
        or not isinstance(recipe["registered_recipe"], str)
        or recipe["registered_recipe"] not in RECIPES
    ):
        raise RecipeError("Invalid or unknown registered distribution recipe")


def _coverage(compiled):
    dispositions = compiled["peptide_dispositions"]
    counts = Counter(row["status"] for row in dispositions)
    if set(counts) - {"admitted", "excluded"}:
        raise ValueError("Unknown distribution admission status")
    return {
        "mapp_peptide": {
            "grain": "source_peptide_row",
            "n_source_rows": len(dispositions),
            "n_query_matches": len(dispositions),
            "n_filtered_by_query": 0,
            "n_limit_omitted": 0,
            "n_candidates": len(dispositions),
            "n_accepted": counts["admitted"],
            "n_excluded": counts["excluded"],
            "n_ambiguous": 0,
            "n_examples": len(compiled["examples"]),
            "n_parent_sequences": len(compiled["parents"]),
            "n_contributor_links": len(compiled["contributors"]),
            "n_run_observations": len(compiled.get("run_observations", [])),
            "reason_counts": dict(
                sorted(Counter(reason for row in dispositions for reason in row["reasons"]).items())
            ),
        }
    }


def materialize_distribution(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    validate_registered_recipe(recipe)
    missing = set(TABLE_BINDINGS.values()) - store.bindings.keys()
    if missing:
        raise RecipeError(f"Registered distribution source is not materialized: {sorted(missing)}")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tables = {}
    for table, name in TABLE_BINDINGS.items():
        result = store.query(
            name, select=[f"native.{field}" for field in store.schemas[name].names], limit=None
        )
        tables[table] = [
            {key.removeprefix("native."): value for key, value in row.items()}
            for row in result.table.to_pylist()
        ]
    recipe_id = recipe["registered_recipe"]
    compiled = compile_mapp_recipe(tables, recipe_id=recipe_id)
    coverage = _coverage(compiled)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copyfile(store.directory / "manifest.json", staging / "source_manifest.json")
        for name, value in (
            ("recipe", RECIPES[recipe_id]),
            ("summary", compiled["summary"]),
            ("coverage", coverage),
        ):
            (staging / f"{name}.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
        for name, rows in compiled.items():
            if name == "summary":
                continue
            with (staging / f"{name}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_distribution",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": coverage,
            "files": {path.name: file_digest(path) for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_distribution_contents(directory, manifest):
    """Check the task contract, source binding, grains and probability arithmetic."""
    from .evidence_recipe import RecipeError

    validate_registered_recipe(manifest["recipe"])
    directory = Path(directory)
    recipe_id = manifest["recipe"]["registered_recipe"]
    relations = {"parents", "examples", "peptide_dispositions", "contributors"}
    if recipe_id == RUN_RECIPE_ID:
        relations.add("run_observations")
    expected = {f"{name}.jsonl" for name in relations} | {
        "source_manifest.json",
        "recipe.json",
        "summary.json",
        "coverage.json",
    }
    if set(manifest["files"]) != expected:
        raise RecipeError("Registered distribution inventory mismatch")
    if json.loads((directory / "recipe.json").read_text()) != RECIPES[recipe_id]:
        raise RecipeError("Registered distribution contract mismatch")
    source = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Registered distribution source binding mismatch")
    records = {
        name: [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]
        for name in relations
    }
    if (
        _coverage(records) != manifest["coverage"]
        or json.loads((directory / "coverage.json").read_text()) != manifest["coverage"]
    ):
        raise RecipeError("Registered distribution coverage mismatch")
    parents = {row["parent_sequence_sha256"]: row for row in records["parents"]}
    dispositions = {row["peptide_row_id"]: row for row in records["peptide_dispositions"]}
    contributors = {row["measurement_id"]: row for row in records["contributors"]}
    if any(
        len(values) != len(records[name])
        for values, name in (
            (parents, "parents"),
            (dispositions, "peptide_dispositions"),
            (contributors, "contributors"),
        )
    ):
        raise RecipeError("Registered distribution duplicate identity")
    for key, parent in parents.items():
        if hashlib.sha256(parent["sequence"].encode()).hexdigest() != key:
            raise RecipeError("Registered distribution parent sequence mismatch")
    example_ids, used_measurements, used_peptides = set(), set(), set()
    for example in records["examples"]:
        if example["example_id"] in example_ids:
            raise RecipeError("Registered distribution duplicate example")
        example_ids.add(example["example_id"])
        parent = parents[example["parent_sequence_sha256"]]
        if any(example[key] != parent[key] for key in ("component_id", "split")):
            raise RecipeError("Registered distribution split mismatch")
        candidates = example["candidates"]
        if len(candidates) < RECIPES[recipe_id]["selection"]["min_eligible_peptides_per_parent"]:
            raise RecipeError("Registered distribution parent gate mismatch")
        raw = [row["source_log10_intensity"] for row in candidates]
        if not all(math.isfinite(value) for value in raw):
            raise RecipeError("Registered distribution nonfinite intensity")
        weights = [10 ** (value - max(raw)) for value in raw]
        for row, weight in zip(candidates, weights):
            p = weight / sum(weights)
            if (
                not math.isclose(row["target_probability"], p, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(row["target_log_probability"], math.log(p), abs_tol=1e-10)
                or not math.isclose(
                    row["target_centered_log10"],
                    row["source_log10_intensity"] - sum(raw) / len(raw),
                    abs_tol=1e-10,
                )
            ):
                raise RecipeError("Registered distribution target arithmetic mismatch")
            disposition = dispositions[row["peptide_row_id"]]
            contributor = contributors[row["measurement_id"]]
            if (
                disposition["status"] != "admitted"
                or any(
                    disposition[key] != example[key]
                    for key in ("parent_sequence_sha256", "component_id", "split")
                )
                or contributor["condition"] != example["condition"]
                or contributor["peptide_row_id"] != row["peptide_row_id"]
                or len(set(contributor["run_measurement_ids"])) != 3
                or len(set(contributor["channel_ids"])) != 3
                or not row["start_positions_0based"]
                or any(
                    parent["sequence"][start : start + len(row["sequence"])] != row["sequence"]
                    for start in row["start_positions_0based"]
                )
                or row["measurement_id"] in used_measurements
            ):
                raise RecipeError("Registered distribution contributor or mapping mismatch")
            used_measurements.add(row["measurement_id"])
            used_peptides.add(row["peptide_row_id"])
    if used_measurements != contributors.keys() or used_peptides != {
        key for key, row in dispositions.items() if row["status"] == "admitted"
    }:
        raise RecipeError("Registered distribution unused contributors or admissions")
    summary = json.loads((directory / "summary.json").read_text())
    for field, expected_count in {
        "n_source_peptides": len(dispositions),
        "n_admitted_peptides": len(used_peptides),
        "n_parent_sequences": len(parents),
        "n_examples": len(example_ids),
        "n_contributor_rows": len(contributors),
    }.items():
        if summary.get(field) != expected_count:
            raise RecipeError(f"Registered distribution summary mismatch: {field}")
    if recipe_id == RUN_RECIPE_ID:
        by_peptide = {}
        for row in records["run_observations"]:
            pid = row["peptide_row_id"]
            channel = row["channel"]
            measurement = row["measurement"]
            value = measurement["value"]
            positive = value is not None and math.isfinite(value) and value > 0
            channels = by_peptide.setdefault(pid, {})
            if (
                channel["channel_id"] in channels
                or measurement["peptide_row_id"] != pid
                or row["condition"] != channel["condition"]
                or row["positive_finite"] != positive
                or row["admitted_complete_parent_set"] != (pid in used_peptides)
                or any(row[key] != dispositions[pid][key] for key in ("component_id", "split"))
                or not set(row["direct_msms_evidence_ids"]) <= set(row["evidence_ids"])
            ):
                raise RecipeError("Registered distribution run relationship mismatch")
            channels[channel["channel_id"]] = row
        for channels in by_peptide.values():
            rows = list(channels.values())
            complete = all(row["positive_finite"] for row in rows)
            if (
                len(rows) != 6
                or Counter(row["condition"] for row in rows) != {"UT": 3, "TNF+IFN": 3}
                or any(row["complete_peptide"] != complete for row in rows)
                or any(row["admitted_complete_parent_set"] and not complete for row in rows)
            ):
                raise RecipeError("Registered distribution complete-run gate mismatch")
        if not used_peptides <= by_peptide.keys():
            raise RecipeError("Registered distribution missing admitted runs")
    return manifest
