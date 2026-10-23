"""Declared captured-run observations; no implicit concentration or negative label."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_catalog import SCHEMA_VERSION
from .provenance import file_digest

RECIPE_ID = "mapp_controlled_run_intensity_v1"
TABLE_BINDINGS = {
    "peptide_rows": "processing_peptide",
    "channels": "processing_channel",
    "quantity_results": "processing_quantity_result",
    "quantity_measurements": "processing_quantity_measurement",
    "evidence": "processing_identification",
    "derivation_inputs": "processing_derivation_input",
    "source_assertions": "processing_source_assertion",
    "label_aliases": "processing_label_alias",
    "reference_proteins": "processing_reference_protein",
    "peptide_reference_occurrences": "processing_peptide_parent_occurrence",
}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "source_id": "wolflevy2018_controlled",
    "endpoint": "reported_positive_captured_fragment_MS_run_intensity",
    "target": "log10_reported_arbitrary_intensity",
    "admission": "nonreverse_noncontaminant_one_source_group_peptide_PEP_le_0.01_positive_reported_and_replayed_intensity",
    "zero_policy": "exclude_from_positive_response_targets;retain_source_zero_not_absence",
    "MSMS_policy": "retain_direct_and_matching_counts;do_not_require_direct_per_run",
    "parent_policy": "all_exact_reference_alternatives_retained;no_unique_parent_assignment",
    "split": "all_source_peptide_components_union_identical_sequence_and_shared_full_reference_sequence;sha256(recipe_id+component_id)_mod_20:0-13_train,14-16_val,17-19_test",
    "limitations": [
        "captured_MS_signal_not_molar_concentration",
        "positive_response_selection_not_detection_probability",
        "unresolved_biological_specimens_and_pairing",
        "same_study_all_splits;not_external_biological_validation",
        "exact_parent_sequence_components_not_homology_clusters",
        "original_2016_search_reference_unavailable",
        "no_independent_precursor_or_capture_MS_response_calibration",
        "cytokine_exposure_time_not_time_since_substrate_synthesis",
    ],
}


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe["schema_version"] != 1
        or recipe["registered_recipe"] != RECIPE_ID
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
    ):
        raise RecipeError("Invalid registered controlled MAPP recipe")


def describe(store):
    return {
        "contract": CONTRACT,
        "status": "materialized"
        if set(TABLE_BINDINGS.values()) <= store.bindings.keys()
        else "not_in_generation",
    }


def _components(tables, peptides):
    roots = {key: key for key in peptides}

    def root(key):
        while key != roots[key]:
            roots[key] = roots[roots[key]]
            key = roots[key]
        return key

    def join(left, right):
        left, right = root(left), root(right)
        roots[max(left, right)] = min(left, right)

    owner = {}
    for key, peptide in sorted(peptides.items()):
        sequence = peptide["sequence_raw"]
        if sequence in owner:
            join(key, owner[sequence])
        owner[sequence] = key
    proteins = {
        row["reference_record_id"]: row["sequence_sha256"] for row in tables["reference_proteins"]
    }
    owner = {}
    for row in tables["peptide_reference_occurrences"]:
        key, parent = row["peptide_row_id"], proteins[row["reference_record_id"]]
        if parent in owner:
            join(key, owner[parent])
        owner[parent] = key
    groups = defaultdict(list)
    for key in sorted(peptides):
        groups[root(key)].append(key)
    records = []
    for members in groups.values():
        sequences = sorted({peptides[key]["sequence_raw"] for key in members})
        component = hashlib.sha256("\n".join(sequences).encode()).hexdigest()
        bucket = int(hashlib.sha256((RECIPE_ID + ":" + component).encode()).hexdigest(), 16) % 20
        split = "train" if bucket < 14 else "val" if bucket < 17 else "test"
        records.extend(
            {"peptide_row_id": key, "component_id": component, "split": split} for key in members
        )
    return sorted(records, key=lambda row: row["peptide_row_id"])


def compile_controlled(tables):
    peptides = {row["peptide_row_id"]: row for row in tables["peptide_rows"]}
    channels = {row["channel_id"]: row for row in tables["channels"]}
    reported = {
        row["result_id"]: row
        for row in tables["quantity_measurements"]
        if row["representation"] == "reported_run_intensity"
    }
    if len(peptides) != len(tables["peptide_rows"]) or len(reported) != len(
        tables["quantity_results"]
    ):
        raise ValueError("Controlled recipe identities or quantity coverage changed")
    components = _components(tables, peptides)
    assignments = {row["peptide_row_id"]: row for row in components}
    dispositions, examples, reasons = [], [], Counter()
    for result in sorted(tables["quantity_results"], key=lambda row: row["result_id"]):
        peptide = peptides[result["peptide_row_id"]]
        channel = channels[result["channel_id"]]
        quantity = reported[result["result_id"]]
        errors = []
        if peptide["reverse"]:
            errors.append("source_reverse")
        if peptide["potential_contaminant"]:
            errors.append("source_contaminant")
        if peptide["pep"] is None or not math.isfinite(peptide["pep"]) or peptide["pep"] > 0.01:
            errors.append("peptide_PEP_unreported_or_above_0.01")
        if len(set(filter(None, peptide["group_ids_raw"].split(";")))) != 1:
            errors.append("not_one_source_group")
        if not peptide["sequence_raw"] or set(peptide["sequence_raw"]) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            errors.append("unsupported_sequence")
        value, replay = quantity["value"], quantity["recomputed_value"]
        if value is None or not math.isfinite(value) or value <= 0:
            errors.append("no_reported_positive_quantity")
        if replay is None or not math.isfinite(replay) or replay <= 0:
            errors.append("no_positive_evidence_sum")
        common = {
            "result_id": result["result_id"],
            "peptide_row_id": result["peptide_row_id"],
            "channel_id": result["channel_id"],
            "source_dataset": result["source_dataset"],
            "measurement_id": quantity["measurement_id"],
            **assignments[result["peptide_row_id"]],
        }
        dispositions.append(
            {
                **common,
                "status": "excluded" if errors else "admitted",
                "reasons": errors,
                "reported_value": value,
                "reported_value_status": quantity["value_status"],
            }
        )
        reasons.update(errors)
        if not errors:
            examples.append(
                {
                    **common,
                    "peptide_sequence": peptide["sequence_raw"],
                    "source_group_ids_raw": peptide["group_ids_raw"],
                    "source_parent_labels_raw": peptide["parent_labels_raw"],
                    "source_peptide_PEP": peptide["pep"],
                    "target_log10_intensity": math.log10(value),
                    "reported_intensity": value,
                    "recomputed_intensity": replay,
                    "context": channel,
                    "evidence_counts": {
                        name: result[name] for name in result if name.startswith("n_")
                    },
                }
            )
    summary = {
        "n_source_results": len(dispositions),
        "n_examples": len(examples),
        "n_excluded_results": len(dispositions) - len(examples),
        "n_source_peptides": len(peptides),
        "n_components": len({row["component_id"] for row in components}),
        "reason_counts": dict(sorted(reasons.items())),
        "examples_by_split": dict(sorted(Counter(row["split"] for row in examples).items())),
        "examples_by_project": dict(
            sorted(Counter(row["source_dataset"] for row in examples).items())
        ),
    }
    return {
        "components": components,
        "examples": examples,
        "dispositions": dispositions,
        "summary": summary,
    }


def materialize_controlled(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not set(TABLE_BINDINGS.values()) <= store.bindings.keys():
        raise RecipeError("Controlled MAPP source is not in this generation")
    tables = {
        table: [
            {key.removeprefix("native."): value for key, value in row.items()}
            for row in store.query(
                name, select=["native." + field for field in store.schemas[name].names], limit=None
            ).table.to_pylist()
        ]
        for table, name in TABLE_BINDINGS.items()
    }
    compiled = compile_controlled(tables)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copyfile(store.directory / "manifest.json", staging / "source_manifest.json")
        for name, value in {"recipe": CONTRACT, "source_tables": tables, **compiled}.items():
            with (staging / f"{name}.json").open("w") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_processing_runs",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": compiled["summary"],
            "files": {p.name: file_digest(p) for p in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_controlled_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    _validate(manifest["recipe"])
    directory = Path(directory)
    expected = {
        "recipe",
        "source_tables",
        "components",
        "examples",
        "dispositions",
        "summary",
        "source_manifest",
    }
    if set(manifest["files"]) != {name + ".json" for name in expected}:
        raise RecipeError("Controlled recipe inventory changed")
    records = {name: json.loads((directory / f"{name}.json").read_text()) for name in expected}
    if records["recipe"] != CONTRACT:
        raise RecipeError("Controlled recipe contract changed")
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or records["source_manifest"]["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Controlled recipe source binding changed")
    if source_package is not None:
        for table in TABLE_BINDINGS:
            relative = f"external/wolflevy2018_controlled/typed/{table}.parquet"
            path = Path(source_package) / relative
            if (
                file_digest(path) != records["source_manifest"]["files"][relative]
                or pq.read_table(path).to_pylist() != records["source_tables"][table]
            ):
                raise RecipeError(f"Controlled recipe source rows changed: {table}")
    compiled = compile_controlled(records["source_tables"])
    for name, value in compiled.items():
        if records[name] != value:
            raise RecipeError(f"Controlled recipe derived contents changed: {name}")
    if manifest["coverage"] != compiled["summary"]:
        raise RecipeError("Controlled recipe coverage changed")
    return manifest
