"""Conditional nominal-load response recipe; no intracellular concentration target."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_catalog import SCHEMA_VERSION
from .provenance import file_digest

RECIPE_ID = "ms_calibration_raw_s4_nominal_response_v1"
TABLE_BINDINGS = {
    "s4_measurements": "calibration_s4_measurement",
    "nominal_standards": "calibration_standard",
    "nominal_loads": "calibration_nominal_load",
    "reporter_forms": "calibration_reporter_form",
    "source_slots": "calibration_source_slot",
    "protocol_parameters": "calibration_protocol_parameter",
    "source_assertions": "calibration_source_assertion",
}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "source_id": "pxd004886",
    "endpoint": "conditional_log10_raw_MS_response_at_nominal_S4_load",
    "quantity_unit": "log10_arbitrary_peak_area",
    "input_load_unit": "fmol_on_column",
    "aggregation": "median_log10_positive_raw_response_per_reporter_and_site",
    "min_positive_source_slots": 1,
    "exclusions": [
        "multiple_raw_reporters_for_nominal_standard",
        "unresolved_reporter_aggregation",
        "unresolved_nominal_isotope_annotation",
        "response_not_reported_positive_finite",
    ],
    "evaluation_group": "canonical_peptide_sequence_across_all_sites",
    "fold_rule": "sort_all_panel_sequences_by_sha256(recipe_id+':'+sequence),rank_modulo_5",
    "n_folds": 5,
    "limitations": [
        "conditional_on_reported_positive_response_not_detection_probability",
        "nominal_standard_load_not_certified_form_molarity",
        "response_selected_panel_and_group_assignment",
        "run_day_specimen_identity_unreported",
        "censoring_and_saturation_mapping_unresolved",
        "synthetic_standard_spiked_after_digestion_not_processing_supervision",
        "no_intracellular_concentration_or_MHC_capture_target",
        "no_established_transfer_from_SWATH_to_MAPP_or_MHC_elution",
    ],
}


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe["schema_version"] != 1
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
        or recipe["registered_recipe"] != RECIPE_ID
    ):
        raise RecipeError("Invalid registered response recipe")


def describe(store):
    return {
        "contract": json.loads(json.dumps(CONTRACT)),
        "status": "materialized"
        if set(TABLE_BINDINGS.values()) <= store.bindings.keys()
        else "not_in_generation",
    }


def _unique(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate response source identity: {key}")
    return result


def _identity(kind, value):
    return f"{RECIPE_ID}:{kind}:{hashlib.sha256(value.encode()).hexdigest()}"


def compile_response(tables):
    """Derive bounded examples while retaining every original raw-slot decision."""
    if set(tables) != set(TABLE_BINDINGS):
        raise ValueError("Incomplete response source tables")
    measurements = _unique(tables["s4_measurements"], "measurement_id")
    standards = _unique(tables["nominal_standards"], "standard_id")
    loads = _unique(tables["nominal_loads"], "nominal_load_id")
    reporters = _unique(tables["reporter_forms"], "reporter_id")
    slots = _unique(tables["source_slots"], "slot_id")
    if any(row["representation"] not in {"raw", "normalized"} for row in measurements.values()):
        raise ValueError("Unknown S4 representation")
    raw = [row for row in measurements.values() if row["representation"] == "raw"]
    sequences = [row["peptide_sequence"] for row in standards.values()]
    if len(set(sequences)) != len(sequences) or any(
        not re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", sequence) for sequence in sequences
    ):
        raise ValueError("Invalid or duplicate nominal panel sequence")
    folds = {
        sequence: rank % CONTRACT["n_folds"]
        for rank, sequence in enumerate(
            sorted(
                sequences,
                key=lambda sequence: hashlib.sha256(f"{RECIPE_ID}:{sequence}".encode()).hexdigest(),
            )
        )
    }
    forms_by_standard = defaultdict(set)
    for row in raw:
        reporter = reporters[row["reporter_id"]]
        forms_by_standard[reporter["standard_id"]].add(reporter["reporter_id"])
    dispositions, contributors, groups = [], [], defaultdict(list)
    seen_slots = set()
    for row in sorted(raw, key=lambda row: row["measurement_id"]):
        reporter, slot, load = (
            reporters[row["reporter_id"]],
            slots[row["slot_id"]],
            loads[row["nominal_load_id"]],
        )
        standard = standards[reporter["standard_id"]]
        sequence = reporter["peptide_sequence"]
        value, amount = row["value"], load["value"]
        key = (row["reporter_id"], row["slot_id"])
        if key in seen_slots:
            raise ValueError("Duplicate raw reporter/source-slot pair")
        seen_slots.add(key)
        if (
            sequence != standard["peptide_sequence"]
            or load["group_label"] != standard["group_label"]
            or load["sample_label"] != slot["sample_label"]
            or load["sample_label"] != "S4"
            or load["unit"] != "fmol_on_column"
            or row["unit"] != "arbitrary_peak_area"
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            raise ValueError("Incoherent nominal response source join or unit")
        reasons = []
        if len(forms_by_standard[reporter["standard_id"]]) != 1:
            reasons.append("multiple_raw_reporters_for_nominal_standard")
        if reporter["aggregation_label_raw"] is not None:
            reasons.append("unresolved_reporter_aggregation")
        nominal_isotopes = re.findall(r"\[([+]\d+)\]", standard["form_label_raw"])
        reported_isotopes = re.findall(r"\[([+]\d+)\]", reporter["label_raw"])
        if not nominal_isotopes or sorted(nominal_isotopes) != sorted(reported_isotopes):
            reasons.append("unresolved_nominal_isotope_annotation")
        positive = isinstance(value, (float, int)) and math.isfinite(value) and value > 0
        if row["value_status"] != "reported_positive" or not positive:
            reasons.append("response_not_reported_positive_finite")
        example_id = _identity("example", f"{reporter['reporter_id']}:{slot['site_label']}")
        decision = {
            "measurement_id": row["measurement_id"],
            "reporter_id": reporter["reporter_id"],
            "standard_id": standard["standard_id"],
            "slot_id": slot["slot_id"],
            "nominal_load_id": load["nominal_load_id"],
            "sequence": sequence,
            "site_label": slot["site_label"],
            "fold_id": folds[sequence],
            "source_cell": row["source_cell"],
            "source_locator": row["source_locator"],
            "source_value": value,
            "source_value_status": row["value_status"],
            "status": "excluded" if reasons else "admitted",
            "reasons": reasons,
            "example_id": None if reasons else example_id,
        }
        dispositions.append(decision)
        if not reasons:
            if not row["source_cell"]:
                raise ValueError("Positive response lacks original source cell")
            contributor = {
                "measurement_id": row["measurement_id"],
                "example_id": example_id,
                "source_cell": row["source_cell"],
                "source_locator": row["source_locator"],
                "slot_id": slot["slot_id"],
                "source_slot_ordinal": slot["source_slot_ordinal"],
                "source_value": value,
                "log10_response": math.log10(value),
            }
            contributors.append(contributor)
            groups[reporter["reporter_id"], slot["site_label"]].append((decision, contributor))
    examples = []
    for (reporter_id, site), members in sorted(groups.items()):
        decision = members[0][0]
        reporter = reporters[reporter_id]
        standard, load = standards[reporter["standard_id"]], loads[decision["nominal_load_id"]]
        values = [contributor["log10_response"] for _, contributor in members]
        examples.append(
            {
                "example_id": decision["example_id"],
                "sequence": decision["sequence"],
                "fold_id": decision["fold_id"],
                "reporter_id": reporter_id,
                "reporter_label_raw": reporter["label_raw"],
                "reporter_annotations_json": reporter["annotations_json"],
                "standard_id": standard["standard_id"],
                "standard_label_raw": standard["form_label_raw"],
                "nominal_load_id": load["nominal_load_id"],
                "nominal_load_fmol": load["value"],
                "nominal_log10_load": math.log10(load["value"]),
                "nominal_load_source_cell": load["source_cell"],
                "group_label_audit_only": standard["group_label"],
                "site_label": site,
                "target_log10_response": statistics.median(values),
                "min_log10_response": min(values),
                "max_log10_response": max(values),
                "n_positive_source_slots": len(values),
                "measurement_ids": sorted(
                    contributor["measurement_id"] for _, contributor in members
                ),
                "interpretation": "conditional_raw_signal_at_nominal_standard_load_not_form_molarity",
            }
        )
    summary = {
        "n_source_measurements": len(measurements),
        "n_raw_source_slots": len(raw),
        "n_out_of_scope_normalized_rows": len(measurements) - len(raw),
        "n_admitted_measurements": len(contributors),
        "n_excluded_measurements": len(raw) - len(contributors),
        "n_examples": len(examples),
        "n_admitted_sequences": len({row["sequence"] for row in examples}),
        "reason_counts": dict(
            sorted(Counter(reason for row in dispositions for reason in row["reasons"]).items())
        ),
        "examples_per_fold": dict(sorted(Counter(str(row["fold_id"]) for row in examples).items())),
    }
    return {
        "dispositions": dispositions,
        "contributors": contributors,
        "examples": examples,
        "summary": summary,
    }


def _source_tables(store):
    from .evidence_recipe import RecipeError

    missing = set(TABLE_BINDINGS.values()) - store.bindings.keys()
    if missing:
        raise RecipeError(f"Registered response source is not materialized: {sorted(missing)}")
    return {
        table: [
            {key.removeprefix("native."): value for key, value in row.items()}
            for row in store.query(
                name, select=["native." + field for field in store.schemas[name].names], limit=None
            ).table.to_pylist()
        ]
        for table, name in TABLE_BINDINGS.items()
    }


def materialize_response(store, recipe, destination):
    from .evidence_recipe import verify_recipe_package

    _validate(recipe)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tables = _source_tables(store)
    compiled = compile_response(tables)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copyfile(store.directory / "manifest.json", staging / "source_manifest.json")
        values = {"recipe": CONTRACT, "source_tables": tables, **compiled}
        for name, value in values.items():
            (staging / f"{name}.json").write_text(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_response",
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


def verify_response_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    _validate(manifest["recipe"])
    directory = Path(directory)
    expected = {
        "recipe",
        "source_tables",
        "dispositions",
        "contributors",
        "examples",
        "summary",
        "source_manifest",
    }
    if set(manifest["files"]) != {name + ".json" for name in expected}:
        raise RecipeError("Registered response inventory mismatch")
    records = {name: json.loads((directory / f"{name}.json").read_text()) for name in expected}
    if records["recipe"] != CONTRACT:
        raise RecipeError("Registered response contract mismatch")
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or records["source_manifest"]["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Registered response source binding mismatch")
    if source_package is not None:
        for table in TABLE_BINDINGS:
            relative = f"external/pxd004886/typed/{table}.parquet"
            path = Path(source_package) / relative
            if file_digest(path) != records["source_manifest"]["files"][relative]:
                raise RecipeError("Registered response source file changed")
            if pq.read_table(path).to_pylist() != records["source_tables"][table]:
                raise RecipeError("Registered response source rows changed")
    compiled = compile_response(records["source_tables"])
    for name, value in compiled.items():
        if records[name] != value:
            raise RecipeError(f"Registered response derived contents mismatch: {name}")
    if manifest["coverage"] != compiled["summary"]:
        raise RecipeError("Registered response coverage mismatch")
    return manifest
