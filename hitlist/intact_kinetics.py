"""Measured product/substrate temporal forecasts; no concentration calibration."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_catalog import SCHEMA_VERSION
from .evidence_snapshot import snapshot_file
from .intact_degradomics import ROOT, _json
from .mapp_evidence import _schema, _Sink
from .provenance import file_digest

RECIPE_ID = "pxd058121_product_substrate_forecast_v1"
TABLES = (
    "assignments",
    "quantities",
    "source_views",
    "acquisitions",
    "assertions",
    "material_contexts",
    "material_links",
    "reconciliations",
)
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "endpoint": "observed_product_to_intact_substrate_log10_intensity_ratio",
    "decision_universe": "all_original_assignments_including_outside_kinetics_views",
    "view": "deposited_F2_kinetics_once_not_reused_F1_or_publisher_selection",
    "admission": [
        "positive_finite",
        "valid_substrate_span",
        "one_reported_and_replayed_assignment",
        "absolute_mass_error_at_most_10ppm",
        "explicit_unmodified",
        "input_mass_identity_and_intensity_agree",
    ],
    "duplicates": "exclude_all_remaining_multiple_mass_sequence_acquisition_rows",
    "normalizer": "unique_admitted_intact_substrate_same_acquisition",
    "eligibility": "at_least_two_distinct_admitted_fitting_times_never_late_support",
    "train_minutes": [0, 20, 40, 60, 80],
    "val_minutes": [100],
    "test_minutes": [120],
    "weights": "equal_peptide_then_time_then_replicate_cell_within_each_split",
    "support": {
        "val_sequences": 20,
        "val_observations": 40,
        "test_sequences": 20,
        "test_observations": 40,
    },
    "missingness": "unknown_cause_no_imputation_negatives_or_inferred_censoring",
    "training_scope": "same_substrate_study_future_time_forecast_conditional_on_observed_positives",
    "limitations": [
        "unknown_peptide_MS_response_requires_training_time_anchor",
        "source_mass_match_not_general_MS_MS_confirmation",
        "not_molar_concentration",
        "purified_mixture_not_APC",
        "one_substrate_and_source_animal",
        "replicate_independence_unresolved",
        "nominal_zero_contains_products",
    ],
}
DECISIONS = _schema(
    (
        "assignment_id",
        "source_record_id",
        "measurement_id",
        "acquisition_id",
        "context_id",
        "view_id",
        "source_view_role",
        "peptide_sequence",
        "parent_sequence",
        "parent_accession",
        "fragment_type",
        "raw_file",
        "replicate_label_raw",
        "organ_label",
        "substrate_species",
        "enzyme_species",
        "material_link_precision",
        "confidence_status",
        "mass_link_status",
        "status",
        "reasons_json",
        "split",
        "normalizer_assignment_id",
        "normalizer_measurement_id",
    ),
    integers=("start_1based", "end_1based", "n_assignments_reported", "n_assignments_replayed"),
    numbers=(
        "time_minutes",
        "quantitative_value",
        "normalizer_value",
        "target_log10_ratio",
        "weight",
    ),
)
LABEL_FIELDS = {"quantitative_value", "normalizer_value", "target_log10_ratio"}
COVARIATES = pa.schema([field for field in DECISIONS if field.name not in LABEL_FIELDS])
TRAJECTORIES = _schema(
    ("peptide_sequence", "eligibility_status", "train_times_json"),
    integers=("n_train_times", "n_train_observations", "n_val_observations", "n_test_observations"),
)
OUTPUT_SCHEMAS = {
    "decisions": DECISIONS,
    "examples": DECISIONS,
    "covariates": COVARIATES,
    "trajectories": TRAJECTORIES,
}


def _rows(directory, name):
    for batch in pq.ParquetFile(directory / f"{name}.parquet").iter_batches():
        yield from batch.to_pylist()


def _split(time):
    for name in ("train", "val", "test"):
        if time in CONTRACT[name + "_minutes"]:
            return name
    return None


def compile_kinetics(source, destination):
    views = {r["view_id"]: r for r in _rows(source, "source_views")}
    acquisitions = {r["acquisition_id"]: r for r in _rows(source, "acquisitions")}
    quantities = {r["measurement_id"]: r for r in _rows(source, "quantities")}
    rows, groups = [], defaultdict(list)
    for assignment in _rows(source, "assignments"):
        view = views[assignment["view_id"]]
        q = quantities[assignment["measurement_id"]]
        acquisition = acquisitions.get(assignment["acquisition_id"], {})
        reasons = []
        value = q["quantitative_value"]
        if view["source_view_role"] != "kinetics":
            reasons.append("outside_kinetics_view")
        if value is None or not math.isfinite(value) or value <= 0:
            reasons.append("not_positive_finite")
        if assignment["span_status"] != "matches_source_substrate":
            reasons.append("span_conflict")
        if assignment["mass_link_status"] != "input_id_and_intensity_agree":
            reasons.append("input_link_or_quantity_conflict")
        if assignment["n_assignments_reported"] != 1 or assignment["n_assignments_replayed"] != 1:
            reasons.append("ambiguous_mass_assignment")
        error = assignment["mass_error_ppm"]
        if error is None or not math.isfinite(error) or abs(error) > 10:
            reasons.append("outside_reported_mass_tolerance")
        if json.loads(assignment["modifications_json"]) != {
            "Localized Mod": "0",
            "Unlocalized Mod": "None",
            "Term Mod": "0",
        }:
            reasons.append("not_explicit_unmodified")
        split = _split(acquisition.get("time_minutes"))
        if split is None:
            reasons.append("outside_declared_times")
        row = {k: assignment.get(k) for k in DECISIONS.names}
        row.update(
            {
                k: acquisition.get(k)
                for k in (
                    "context_id",
                    "raw_file",
                    "time_minutes",
                    "replicate_label_raw",
                    "organ_label",
                    "substrate_species",
                    "enzyme_species",
                    "material_link_precision",
                )
            }
        )
        row.update(
            source_view_role=view["source_view_role"],
            quantitative_value=value,
            split=split,
            reasons_json=reasons,
        )
        rows.append(row)
        if not reasons:
            groups[row["peptide_sequence"], row["acquisition_id"]].append(row)
    for group in groups.values():
        if len(group) > 1:
            for row in group:
                row["reasons_json"].append("multiple_masses_for_sequence_acquisition")
    denominators = defaultdict(list)
    for row in rows:
        if (
            not row["reasons_json"]
            and row["start_1based"] == 1
            and row["end_1based"] == len(row["parent_sequence"])
        ):
            denominators[row["acquisition_id"]].append(row)
    products = []
    for row in rows:
        if row["reasons_json"]:
            continue
        if row["start_1based"] == 1 and row["end_1based"] == len(row["parent_sequence"]):
            row["status"] = "normalizer_only"
            continue
        matches = denominators[row["acquisition_id"]]
        if len(matches) != 1:
            row["reasons_json"].append("no_unique_admitted_intact_substrate")
            continue
        parent = matches[0]
        row.update(
            normalizer_assignment_id=parent["assignment_id"],
            normalizer_measurement_id=parent["measurement_id"],
            normalizer_value=parent["quantitative_value"],
        )
        products.append(row)
    by_peptide = defaultdict(list)
    for row in products:
        by_peptide[row["peptide_sequence"]].append(row)
    trajectories = []
    for sequence, observations in sorted(by_peptide.items()):
        times = sorted({r["time_minutes"] for r in observations if r["split"] == "train"})
        eligible = len(times) >= 2
        trajectories.append(
            {
                "peptide_sequence": sequence,
                "eligibility_status": "train_eligible" if eligible else "insufficient_train_times",
                "train_times_json": _json(times),
                "n_train_times": len(times),
                **{
                    "n_" + s + "_observations": sum(r["split"] == s for r in observations)
                    for s in ("train", "val", "test")
                },
            }
        )
        for row in observations:
            if not eligible:
                row["reasons_json"].append("fewer_than_two_fitting_times")
            else:
                row["status"] = "admitted"
                row["target_log10_ratio"] = math.log10(row["quantitative_value"]) - math.log10(
                    row["normalizer_value"]
                )
    support = {}
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["status"] == "admitted" and r["split"] == split]
        sequences = {r["peptide_sequence"] for r in selected}
        times = defaultdict(set)
        cells = Counter((r["peptide_sequence"], r["time_minutes"]) for r in selected)
        for r in selected:
            times[r["peptide_sequence"]].add(r["time_minutes"])
        for r in selected:
            r["weight"] = 1 / (
                len(sequences)
                * len(times[r["peptide_sequence"]])
                * cells[r["peptide_sequence"], r["time_minutes"]]
            )
        support[split] = {
            "n_sequences": len(sequences),
            "n_observations": len(selected),
            "n_acquisitions": len({r["acquisition_id"] for r in selected}),
        }
    destination.mkdir(parents=True)
    sinks = {n: _Sink(destination, n, schema=s) for n, s in OUTPUT_SCHEMAS.items()}
    reasons_count = Counter()
    try:
        for row in rows:
            reasons_count.update(row["reasons_json"])
            row["status"] = row["status"] or "excluded"
            row["reasons_json"] = _json(row["reasons_json"])
            sinks["decisions"].add(row)
            if row["status"] == "admitted":
                sinks["examples"].add(row)
                sinks["covariates"].add({k: v for k, v in row.items() if k not in LABEL_FIELDS})
        for row in trajectories:
            sinks["trajectories"].add(row)
        return {
            "cohort_support": support,
            "support_passed": all(
                support[s]["n_sequences"] >= 20 and support[s]["n_observations"] >= 40
                for s in ("val", "test")
            ),
            "reason_counts": dict(sorted(reasons_count.items())),
            "output_counts": {"n_" + name + "_rows": sink.n_rows for name, sink in sinks.items()},
        }
    finally:
        for sink in sinks.values():
            sink.close()


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe["schema_version"] != 1
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
        or recipe["registered_recipe"] != RECIPE_ID
    ):
        raise RecipeError("Invalid intact kinetic forecast recipe")


def describe(store):
    return {
        "contract": json.loads(_json(CONTRACT)),
        "status": "materialized"
        if {"intact_" + n for n in TABLES} <= store.bindings.keys()
        else "not_in_generation",
    }


def materialize_kinetics(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if describe(store)["status"] != "materialized":
        raise RecipeError("Intact-degradomics source not materialized")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        for name in TABLES:
            snapshot_file(
                store.directory / ROOT / "typed" / f"{name}.parquet",
                staging / "source_tables" / f"{name}.parquet",
            )
        summary = compile_kinetics(staging / "source_tables", staging / "typed")
        (staging / "contract.json").write_text(_json(CONTRACT) + "\n")
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_intact_kinetics",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(_json(recipe).encode()).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": summary,
            "files": {
                str(p.relative_to(staging)): file_digest(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            _json({"destination": str(destination)}) + "\n"
        )
        raise


def verify_kinetic_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    _validate(manifest["recipe"])
    source = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source.get("package_family") != "intact_degradomics"
        or source["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Intact kinetics source binding differs")
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("Intact kinetics contract differs")
    expected = (
        {"source_manifest.json", "contract.json", "summary.json"}
        | {f"typed/{name}.parquet" for name in OUTPUT_SCHEMAS}
        | {f"source_tables/{n}.parquet" for n in TABLES}
    )
    if set(manifest["files"]) != expected:
        raise RecipeError("Intact kinetics inventory differs")
    for name in TABLES:
        relative = str(ROOT / "typed" / f"{name}.parquet")
        if (
            file_digest(directory / "source_tables" / f"{name}.parquet")
            != source["files"][relative]
        ):
            raise RecipeError("Intact kinetics source table differs")
        if (
            source_package is not None
            and file_digest(Path(source_package) / relative) != source["files"][relative]
        ):
            raise RecipeError("Intact kinetics source table changed")
    temporary = Path(tempfile.mkdtemp(prefix="hitlist-intact-kinetic-verify-"))
    try:
        summary = compile_kinetics(directory / "source_tables", temporary / "typed")
        if summary != manifest["coverage"] or summary != json.loads(
            (directory / "summary.json").read_text()
        ):
            raise RecipeError("Intact kinetics coverage differs")
        for name, schema in OUTPUT_SCHEMAS.items():
            left = pq.ParquetFile(temporary / "typed" / f"{name}.parquet")
            right = pq.ParquetFile(directory / "typed" / f"{name}.parquet")
            if not right.schema_arrow.equals(schema):
                raise RecipeError("Intact kinetics schema differs")
            for expected_batch, actual in zip(
                left.iter_batches(), right.iter_batches(), strict=True
            ):
                if not expected_batch.equals(actual):
                    raise RecipeError("Intact kinetics rows differ from source")
    except Exception:
        (temporary / "failure-context.json").write_text(_json({"package": str(directory)}) + "\n")
        raise
    else:
        shutil.rmtree(temporary)
    return manifest
