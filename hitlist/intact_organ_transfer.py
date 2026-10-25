"""Source-bound organ time-course changes; supplied baseline, no concentration claim."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.dataset as ds

from . import intact_kinetics as kinetics
from .intact_degradomics import _json
from .mapp_evidence import _schema, _Sink

RECIPE_ID = "pxd058121_organ_baseline_change_v1"
OUTPUT_KIND = "registered_intact_organ_transfer"
TABLES = kinetics.TABLES
ORGANS = ("Brain", "Liver", "Spleen")
TIMES = {0: "baseline", 30: "test", 60: "test", 90: "extrapolation", 120: "extrapolation"}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "endpoint": "log10_product_intact_ratio_minus_organ_nominal_zero_mean_log_ratio",
    "view": "original_F3_organs_excluding_recorded_F2_acquisition_reuse",
    "decision_universe": "all_assignments_without_interpreting_non_F3_quantities",
    "admission": kinetics.CONTRACT["admission"],
    "duplicates": kinetics.CONTRACT["duplicates"],
    "normalizer": kinetics.CONTRACT["normalizer"],
    "baseline": "mean_eligible_zero_time_log_ratios_same_organ_parent_span_sequence",
    "required_organs": list(ORGANS),
    "baseline_minutes": [0],
    "test_minutes": [30, 60],
    "extrapolation_minutes": [90, 120],
    "weights": "equal_represented_organ_then_peptide_curve_then_time_then_replicate_cell",
    "support": {"per_organ_test_sequences": 20, "per_organ_test_observations": 40},
    "missingness": kinetics.CONTRACT["missingness"],
    "limitations": [
        "observed_positive_baseline_supported_products_only",
        "fixed_response_and_shared_run_gain_cancellation_assumptions",
        "time_dependent_response_and_matrix_effects_unresolved",
        "supplied_baseline_not_an_unconditional_forecast",
        "not_molar_concentration_or_APC",
        "same_parent_and_one_source_animal",
        "replicate_physical_independence_unresolved",
        "nominal_zero_contains_products",
    ],
}
DECISIONS = pa.schema(
    list(kinetics.DECISIONS)
    + list(
        _schema(
            (
                "curve_id",
                "baseline_assignment_ids_json",
                "baseline_measurement_ids_json",
                "baseline_normalizer_assignment_ids_json",
            ),
            integers=("n_baseline_observations",),
            numbers=("baseline_log10_ratio", "target_change"),
        )
    )
)
LABEL_FIELDS = kinetics.LABEL_FIELDS | {"target_change"}
COVARIATES = pa.schema([f for f in DECISIONS if f.name not in LABEL_FIELDS])
TRAJECTORIES = _schema(
    ("curve_id", "organ_label", "peptide_sequence", "parent_sequence", "eligibility_status"),
    integers=(
        "start_1based",
        "end_1based",
        "n_baseline_observations",
        "n_test_observations",
        "n_extrapolation_observations",
    ),
    numbers=("baseline_log10_ratio",),
)
OUTPUT_SCHEMAS = {
    "decisions": DECISIONS,
    "examples": DECISIONS,
    "covariates": COVARIATES,
    "baselines": DECISIONS,
    "trajectories": TRAJECTORIES,
}
_rows = kinetics._rows


def acquisition_scope(views, acquisitions, reconciliations):
    """Metadata only: never infer preparation independence from distinct filenames."""
    old_ids = {
        v["acquisition_id"]
        for v in views.values()
        if v["source_view_role"] == "kinetics" and v["acquisition_id"]
    }
    old_raw = {
        acquisitions[i]["raw_file"]
        for i in old_ids
        if i in acquisitions and acquisitions[i].get("raw_file")
    }
    reused = set()
    for claim in reconciliations:
        if claim["assertion_predicate"] != "repeated_acquisition_source_views":
            continue
        pair = [views.get(claim[k], {}) for k in ("source_subject_id", "target_subject_id")]
        if {v.get("source_view_role") for v in pair} == {"kinetics", "organs"}:
            reused.update(v["view_id"] for v in pair if v["source_view_role"] == "organs")
    result = {}
    for key, view in views.items():
        reasons = []
        a = acquisitions.get(view["acquisition_id"], {})
        if view["source_view_role"] != "organs":
            reasons.append("outside_organs_view")
        else:
            if not a.get("raw_file"):
                reasons.append("unresolved_acquisition")
            if view["acquisition_id"] in old_ids or a.get("raw_file") in old_raw:
                reasons.append("F2_acquisition_overlap")
            if key in reused:
                reasons.append("source_backed_reused_view")
            if a.get("organ_label") not in ORGANS:
                reasons.append("outside_fixed_organ_scope")
            if a.get("time_minutes") not in TIMES:
                reasons.append("outside_declared_times")
        result[key] = reasons
    return result


def compile_kinetics(source, destination):
    views = {r["view_id"]: r for r in _rows(source, "source_views")}
    acquisitions = {r["acquisition_id"]: r for r in _rows(source, "acquisitions")}
    scope = acquisition_scope(views, acquisitions, _rows(source, "reconciliations"))
    assignments = list(_rows(source, "assignments"))
    ids = {
        r["measurement_id"]
        for r in assignments
        if views[r["view_id"]]["source_view_role"] == "organs"
    }
    # Predicate pushdown prevents materializing F2/publisher quantity values here.
    table = ds.dataset(source / "quantities.parquet", format="parquet").to_table(
        filter=ds.field("measurement_id").isin(sorted(ids))
    )
    quantities = {r["measurement_id"]: r for r in table.to_pylist()}
    rows, groups = [], defaultdict(list)
    for assignment in assignments:
        view = views[assignment["view_id"]]
        acquisition = acquisitions.get(assignment["acquisition_id"], {})
        reasons = list(scope[view["view_id"]])
        q = quantities.get(assignment["measurement_id"], {})
        value = q.get("quantitative_value") if view["source_view_role"] == "organs" else None
        if view["source_view_role"] == "organs":
            if value is None or not math.isfinite(value) or value <= 0:
                reasons.append("not_positive_finite")
            if assignment["span_status"] != "matches_source_substrate":
                reasons.append("span_conflict")
            if assignment["mass_link_status"] != "input_id_and_intensity_agree":
                reasons.append("input_link_or_quantity_conflict")
            if (
                assignment["n_assignments_reported"] != 1
                or assignment["n_assignments_replayed"] != 1
            ):
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
            split=TIMES.get(acquisition.get("time_minutes")),
            reasons_json=reasons,
        )
        rows.append(row)
        if not reasons:
            groups[row["peptide_sequence"], row["acquisition_id"]].append(row)
    for group in groups.values():
        if len(group) > 1:
            for row in group:
                row["reasons_json"].append("multiple_masses_for_sequence_acquisition")
    parents = defaultdict(list)
    for row in rows:
        if (
            not row["reasons_json"]
            and row["start_1based"] == 1
            and row["end_1based"] == len(row["parent_sequence"])
        ):
            parents[row["acquisition_id"]].append(row)
    curves = defaultdict(list)
    for row in rows:
        if row["reasons_json"]:
            continue
        if row["start_1based"] == 1 and row["end_1based"] == len(row["parent_sequence"]):
            row["status"] = "normalizer_only"
            continue
        matches = parents[row["acquisition_id"]]
        if len(matches) != 1:
            row["reasons_json"].append("no_unique_admitted_intact_substrate")
            continue
        parent = matches[0]
        curve = [
            row[k]
            for k in (
                "organ_label",
                "parent_sequence",
                "start_1based",
                "end_1based",
                "peptide_sequence",
            )
        ]
        row.update(
            normalizer_assignment_id=parent["assignment_id"],
            normalizer_measurement_id=parent["measurement_id"],
            normalizer_value=parent["quantitative_value"],
            target_log10_ratio=math.log10(row["quantitative_value"])
            - math.log10(parent["quantitative_value"]),
            curve_id=hashlib.sha256(_json(curve).encode()).hexdigest(),
        )
        curves[row["curve_id"]].append(row)
    trajectories = []
    for curve_id, records in sorted(curves.items()):
        baseline = [r for r in records if r["split"] == "baseline"]
        mean = (
            math.fsum(r["target_log10_ratio"] for r in baseline) / len(baseline)
            if baseline
            else None
        )
        trajectories.append(
            {
                **{
                    k: records[0][k]
                    for k in (
                        "organ_label",
                        "peptide_sequence",
                        "parent_sequence",
                        "start_1based",
                        "end_1based",
                    )
                },
                "curve_id": curve_id,
                "eligibility_status": "baseline_supported"
                if baseline
                else "no_eligible_zero_baseline",
                "baseline_log10_ratio": mean,
                **{
                    "n_" + s + "_observations": sum(r["split"] == s for r in records)
                    for s in TIMES.values()
                },
            }
        )
        for row in records:
            if not baseline:
                row["reasons_json"].append("no_eligible_zero_baseline")
                continue
            row.update(
                baseline_log10_ratio=mean,
                n_baseline_observations=len(baseline),
                baseline_assignment_ids_json=_json([r["assignment_id"] for r in baseline]),
                baseline_measurement_ids_json=_json([r["measurement_id"] for r in baseline]),
                baseline_normalizer_assignment_ids_json=_json(
                    [r["normalizer_assignment_id"] for r in baseline]
                ),
                status="baseline_only" if row["split"] == "baseline" else "admitted",
            )
            if row["status"] == "admitted":
                row["target_change"] = row["target_log10_ratio"] - mean
    support = {}
    for split in ("test", "extrapolation"):
        selected = [r for r in rows if r["status"] == "admitted" and r["split"] == split]
        by_organ, times = defaultdict(set), defaultdict(set)
        counts = Counter((r["curve_id"], r["time_minutes"]) for r in selected)
        for row in selected:
            by_organ[row["organ_label"]].add(row["curve_id"])
            times[row["curve_id"]].add(row["time_minutes"])
        for row in selected:
            row["weight"] = 1 / (
                len(by_organ)
                * len(by_organ[row["organ_label"]])
                * len(times[row["curve_id"]])
                * counts[row["curve_id"], row["time_minutes"]]
            )
        support[split] = {
            organ: {
                "n_sequences": len(
                    {r["peptide_sequence"] for r in selected if r["organ_label"] == organ}
                ),
                "n_observations": sum(r["organ_label"] == organ for r in selected),
                "n_acquisitions": len(
                    {r["acquisition_id"] for r in selected if r["organ_label"] == organ}
                ),
            }
            for organ in ORGANS
        }
    destination.mkdir(parents=True)
    sinks = {n: _Sink(destination, n, schema=s) for n, s in OUTPUT_SCHEMAS.items()}
    reasons = Counter()
    try:
        for row in rows:
            reasons.update(row["reasons_json"])
            row["status"] = row["status"] or "excluded"
            row["reasons_json"] = _json(row["reasons_json"])
            sinks["decisions"].add(row)
            if row["status"] == "admitted":
                sinks["examples"].add(row)
                sinks["covariates"].add({k: v for k, v in row.items() if k not in LABEL_FIELDS})
            elif row["status"] == "baseline_only":
                sinks["baselines"].add(row)
        for row in trajectories:
            sinks["trajectories"].add(row)
        return {
            "cohort_support": support,
            "support_passed": all(
                support["test"][o]["n_sequences"] >= 20
                and support["test"][o]["n_observations"] >= 40
                for o in ORGANS
            ),
            "reason_counts": dict(sorted(reasons.items())),
            "output_counts": {"n_" + n + "_rows": s.n_rows for n, s in sinks.items()},
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
        raise RecipeError("Invalid intact organ baseline-change recipe")


def describe(store):
    return {
        "contract": json.loads(_json(CONTRACT)),
        "status": "materialized"
        if {"intact_" + n for n in TABLES} <= store.bindings.keys()
        else "not_in_generation",
    }


def materialize_organ_transfer(store, recipe, destination):
    return kinetics._materialize(store, recipe, destination, sys.modules[__name__])


def verify_organ_contents(directory, manifest, source_package=None):
    return kinetics._verify_contents(directory, manifest, source_package, sys.modules[__name__])
