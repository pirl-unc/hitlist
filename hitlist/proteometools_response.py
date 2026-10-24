"""Conditional measured MS1 response with explicit synthesis and selection limits."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_catalog import SCHEMA_VERSION
from .evidence_snapshot import snapshot_file
from .proteometools_ms1 import ROOT, _id
from .provenance import file_digest

RECIPE_ID = "pxd021013_intended_unmodified_ms1_v1"
TABLE_BINDINGS = {
    name: "proteometools_" + name
    for name in ("planned_products", "runs", "peptides", "forms", "contexts")
}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "source_id": "pxd021013",
    "endpoint": "conditional_within_run_centered_log10_MS1_signal",
    "unit": "log10_arbitrary_MS1_signal_relative_to_run_geometric_mean",
    "quality": "both_sequence_and_form_positive_PEP_in_[0,0.01)_nonreverse_noncontaminant",
    "form_policy": "unique_Unmodified_variable_form_with_explicit_fixed_modifications",
    "target_normalization": "subtract_mean_log10_over_admitted_observed_products_in_same_run",
    "prediction_normalization": "center_predictions_over_the_same_conditional_candidate_set",
    "dependence": "connect_selected_pools_sharing_any_intended_sequence_including_unobserved_products_before_outcome_filtering",
    "fold_assignment": "deferred_to_prospective_experiment_no_guarantee_of_independent_group_count",
    "limitations": [
        "conditional_on_identification_and_positive_signal_not_detection_probability",
        "crude_synthesis_unknown_individual_purity_and_molar_amount",
        "estimated_200_or_500_fmol_not_run_resolved",
        "synthesis_handling_LC_MS_confounding",
        "unmodified_variable_form_retains_search_fixed_modifications",
        "library_origin_not_measured_MHC_restriction",
        "no_absolute_detector_or_intracellular_concentration_calibration",
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
        raise RecipeError("Invalid registered ProteomeTools response recipe")


def describe(store):
    return {
        "contract": json.loads(json.dumps(CONTRACT)),
        "status": "materialized"
        if set(TABLE_BINDINGS.values()) <= store.bindings.keys()
        else "not_in_generation",
    }


def _unique(rows, field):
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate source identity: {field}")
    return result


def compile_response(tables):
    """Keep every form decision and planned product; never synthesize missing zeros."""
    runs = _unique(tables["runs"], "run_id")
    peptides = _unique(tables["peptides"], "measurement_id")
    _unique(tables["forms"], "measurement_id")
    _unique(tables["planned_products"], "planned_id")
    contexts = _unique(tables["contexts"], "context_id")
    pool_names = {row["pool_name"] for row in runs.values()}
    planned = defaultdict(list)
    for row in tables["planned_products"]:
        if row["pool_name"] in pool_names:
            planned[row["pool_name"], row["peptide_sequence"]].append(row["planned_id"])
    parents = {pool: pool for pool in pool_names}

    def find(pool):
        while parents[pool] != pool:
            parents[pool] = parents[parents[pool]]
            pool = parents[pool]
        return pool

    sequence_pool = {}
    # Off-planned identifications can never enter this endpoint. Grouping them
    # would connect unrelated intended panels through shared background/QC IDs.
    # All intended products, including unobserved ones, determine dependence.
    for pool, seq in sorted(planned):
        if seq in sequence_pool:
            left, right = sorted((find(pool), find(sequence_pool[seq])))
            parents[right] = left
        else:
            sequence_pool[seq] = pool
    components = defaultdict(list)
    for pool in sorted(pool_names):
        components[find(pool)].append(pool)
    group_ids = {
        pool: _id("pool_component", members) for members in components.values() for pool in members
    }
    forms_by_run_sequence = defaultdict(list)
    for row in tables["forms"]:
        if row["peptide_modifications_raw"] == "Unmodified":
            forms_by_run_sequence[row["run_id"], row["peptide_sequence"]].append(
                row["measurement_id"]
            )
    dispositions, examples = [], []
    for row in sorted(tables["forms"], key=lambda value: value["measurement_id"]):
        run = runs[row["run_id"]]
        peptide = peptides[row["peptide_measurement_id"]]
        context = contexts[run["context_id"]]
        if (
            peptide["run_id"] != row["run_id"]
            or peptide["peptide_sequence"] != row["peptide_sequence"]
        ):
            raise ValueError("Form and peptide join disagree")
        seq = row["peptide_sequence"]
        reasons = []
        if (run["pool_name"], seq) not in planned:
            reasons.append("not_intended_product_in_this_pool")
        if re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", seq) is None:
            reasons.append("noncanonical_sequence")
        for kind, value in (("sequence", peptide), ("form", row)):
            if (
                value["value_status"] != "positive"
                or value["quantitative_value"] is None
                or not math.isfinite(value["quantitative_value"])
                or value["quantitative_value"] <= 0
            ):
                reasons.append(kind + "_signal_not_positive_finite")
            if value["pep"] is None or not 0 <= value["pep"] < 0.01:
                reasons.append(kind + "_PEP_not_in_range")
            if value["reverse_raw"] or value["contaminant_raw"]:
                reasons.append(kind + "_reverse_or_contaminant")
        if row["peptide_modifications_raw"] != "Unmodified":
            reasons.append("variable_modification_out_of_scope")
        elif len(forms_by_run_sequence[row["run_id"], seq]) != 1:
            reasons.append("ambiguous_multiple_unmodified_forms")
        if run["fixed_modifications_raw"] is None:
            reasons.append("fixed_modifications_unresolved")
        disposition = {
            "measurement_id": row["measurement_id"],
            "source_record_id": row["source_record_id"],
            "peptide_measurement_id": row["peptide_measurement_id"],
            "run_id": row["run_id"],
            "pool_name": run["pool_name"],
            "sequence": seq,
            "reasons": reasons,
            "admitted": not reasons,
        }
        dispositions.append(disposition)
        if not reasons:
            examples.append(
                {
                    **{
                        k: disposition[k]
                        for k in (
                            "measurement_id",
                            "source_record_id",
                            "peptide_measurement_id",
                            "run_id",
                            "pool_name",
                            "sequence",
                        )
                    },
                    "planned_ids": sorted(planned[run["pool_name"], seq]),
                    "dependence_component": group_ids[run["pool_name"]],
                    "library_origin_audit_only": run["library_origin"],
                    "fixed_modifications_raw": run["fixed_modifications_raw"],
                    "variable_modifications_raw": row["peptide_modifications_raw"],
                    "quantitative_raw": row["quantitative_raw"],
                    "log10_signal": math.log10(row["quantitative_value"]),
                    "individual_molar_amount_status": context["individual_molar_amount_status"],
                }
            )
    per_run = defaultdict(list)
    for example in examples:
        per_run[example["run_id"]].append(example)
    for members in per_run.values():
        center = math.fsum(row["log10_signal"] for row in members) / len(members)
        for row in members:
            row["target_centered_log10_signal"] = row["log10_signal"] - center
            row["run_target_center_audit_only"] = center
            row["n_conditional_products"] = len(members)
    admitted = {(row["run_id"], row["sequence"]) for row in examples}
    observed = {(row["run_id"], row["peptide_sequence"]) for row in peptides.values()}
    planned_dispositions = []
    for run_id, run in sorted(runs.items()):
        for (pool, seq), ids in sorted(planned.items()):
            if pool == run["pool_name"]:
                status = (
                    "admitted"
                    if (run_id, seq) in admitted
                    else "observed_not_admitted"
                    if (run_id, seq) in observed
                    else "unobserved_unknown_cause_not_negative"
                )
                planned_dispositions.append(
                    {
                        "run_id": run_id,
                        "pool_name": pool,
                        "sequence": seq,
                        "planned_ids": sorted(ids),
                        "status": status,
                    }
                )
    groups = [
        {"component_id": group_ids[members[0]], "pools": members}
        for members in sorted(components.values())
    ]
    return {
        "examples": examples,
        "dispositions": dispositions,
        "planned_dispositions": planned_dispositions,
        "dependence_groups": groups,
        "summary": {
            "n_source_forms": len(dispositions),
            "n_admitted_forms": len(examples),
            "n_excluded_forms": len(dispositions) - len(examples),
            "n_selected_pools": len(pool_names),
            "n_runs": len(runs),
            "n_dependence_components": len(groups),
            "n_planned_product_run_pairs": len(planned_dispositions),
            "planned_status_counts": dict(
                sorted(Counter(row["status"] for row in planned_dispositions).items())
            ),
            "reason_counts": dict(
                sorted(Counter(reason for row in dispositions for reason in row["reasons"]).items())
            ),
            "examples_per_run": dict(sorted((run, len(rows)) for run, rows in per_run.items())),
        },
    }


def materialize_response(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if not set(TABLE_BINDINGS.values()) <= store.bindings.keys():
        raise RecipeError("ProteomeTools source is not materialized")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tables = {
        table: [
            {k.removeprefix("native."): v for k, v in row.items()}
            for row in store.query(
                name, select=["native." + field for field in store.schemas[name].names], limit=None
            ).table.to_pylist()
        ]
        for table, name in TABLE_BINDINGS.items()
    }
    compiled = compile_response(tables)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        for table in TABLE_BINDINGS:
            snapshot_file(
                store.directory / ROOT / "typed" / f"{table}.parquet",
                staging / "source_tables" / f"{table}.parquet",
            )
        for name, value in {"contract": CONTRACT, **compiled}.items():
            (staging / f"{name}.json").write_text(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_proteometools_response",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": compiled["summary"],
            "files": {
                str(p.relative_to(staging)): file_digest(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
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


def verify_response_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    _validate(manifest["recipe"])
    directory = Path(directory)
    source = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("ProteomeTools recipe source binding differs")
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("ProteomeTools response contract differs")
    tables = {}
    for table in TABLE_BINDINGS:
        path = directory / "source_tables" / f"{table}.parquet"
        relative = str(ROOT / "typed" / f"{table}.parquet")
        if file_digest(path) != source["files"][relative]:
            raise RecipeError("ProteomeTools recipe source table differs")
        if (
            source_package is not None
            and file_digest(Path(source_package) / relative) != source["files"][relative]
        ):
            raise RecipeError("ProteomeTools source table changed")
        tables[table] = pq.read_table(path).to_pylist()
    compiled = compile_response(tables)
    expected = (
        {"source_manifest.json", "contract.json"}
        | {f"source_tables/{table}.parquet" for table in TABLE_BINDINGS}
        | {name + ".json" for name in compiled}
    )
    if set(manifest["files"]) != expected:
        raise RecipeError("ProteomeTools response inventory differs")
    for name, value in compiled.items():
        if json.loads((directory / f"{name}.json").read_text()) != value:
            raise RecipeError(f"ProteomeTools response replay differs: {name}")
    if manifest["coverage"] != compiled["summary"]:
        raise RecipeError("ProteomeTools response coverage differs")
    return manifest
