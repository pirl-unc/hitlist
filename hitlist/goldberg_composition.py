"""Conditional observed MAPP signal, with all-source dependence before filtering."""

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
from .goldberg_candidates import DECISION_SCHEMA, TABLES, _rows
from .goldberg_candidates import _compile as compile_baseline
from .goldberg_mapp import ROOT, _id, _json
from .mapp_evidence import _schema, _Sink
from .provenance import file_digest

RECIPE_ID = "goldberg2025_observed_baseline_composition_v1"
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "baseline_recipe": "goldberg2025_baseline_candidates_v1",
    "endpoint": "conditional_observed_source_accession_acquisition_MAPP_intensity_composition",
    "confidence": "source_peptide_PEP_in_[0,.01)_and_run_identification_type_By_MS/MS",
    "origin": "one_distinct_supplied_parent_accession_not_verified_full_reference",
    "minimum_distinct_sequences_per_group": 3,
    "duplicate_sequence_within_eligible_group": "refuse_ambiguous_aggregation",
    "dependence": "all_original_peptide_sequence_and_parent_label_nodes_including_excluded_bridges",
    "component_id": "sha256(goldberg2025_baseline_v1:+compact_sorted_node_JSON)",
    "split": "int(component_id,16)%10_train0to5_val6to7_test8to9",
    "weights": "equal_component_then_accession_acquisition_group_then_peptide_within_split",
    "targets": "centered_log10_intensity_and_observed_set_intensity_fraction",
    "support": {
        "train_components": 100,
        "val_components": 20,
        "test_components": 20,
        "val_groups": 20,
        "test_groups": 20,
    },
    "training_status": "conditional_captured_signal_only_not_full_processing_eligibility",
    "limitations": [
        "not_molarity_or_complete_free_peptide_distribution",
        "A549_not_professional_APC",
        "positive_observed_set_conditioning",
        "sequence_totals_may_aggregate_forms",
        "source_parent_annotations_not_verified_origins",
        "no_full_search_reference",
        "specimen_homology_and_exact_precursor_pairing_unresolved",
        "source_confidence_not_new_FDR_guarantee",
    ],
}
DECISIONS = _schema(
    (
        "candidate_id",
        "measurement_id",
        "peptide_id",
        "component_id",
        "split",
        "group_id",
        "baseline_status",
        "status",
        "reasons_json",
    ),
)
EXAMPLES = pa.schema(
    list(DECISION_SCHEMA)
    + list(
        _schema(
            (
                "group_id",
                "component_id",
                "split",
                "parent_accession",
                "composition_eligibility_status",
            ),
            numbers=(
                "target_centered_log10_intensity",
                "target_intensity_fraction",
                "weight",
                "group_weight",
            ),
        )
    )
)
LABEL_FIELDS = {
    "quantitative_raw",
    "quantitative_value",
    "log10_intensity",
    "target_centered_log10_intensity",
    "target_intensity_fraction",
}
COVARIATES = pa.schema([field for field in EXAMPLES if field.name not in LABEL_FIELDS])
COMPONENTS = _schema(("node_id", "component_id", "split"))
GROUPS = _schema(
    ("group_id", "component_id", "split", "parent_accession", "acquisition_id"),
    integers=("n_peptides",),
    numbers=("group_weight",),
)
OUTPUT_SCHEMAS = {
    "decisions": DECISIONS,
    "examples": EXAMPLES,
    "covariates": COVARIATES,
    "components": COMPONENTS,
    "groups": GROUPS,
}


def _parents(row):
    return sorted({p for p in (row["parent_label_raw"] or "").split(";") if p})


def dependence_components(peptides):
    union = {}

    def find(node):
        union.setdefault(node, node)
        while union[node] != node:
            union[node] = union[union[node]]
            node = union[node]
        return node

    for row in peptides.values():
        nodes = ["peptide:" + row["peptide_sequence"]] + ["parent:" + p for p in _parents(row)]
        roots = sorted({find(node) for node in nodes})
        for root in roots[1:]:
            union[root] = roots[0]
    members = defaultdict(list)
    for node in sorted(union):
        members[find(node)].append(node)
    result = {}
    for nodes in members.values():
        digest = hashlib.sha256(
            ("goldberg2025_baseline_v1:" + _json(sorted(nodes))).encode()
        ).hexdigest()
        bucket = int(digest, 16) % 10
        split = "train" if bucket < 6 else "val" if bucket < 8 else "test"
        for node in nodes:
            result[node] = (digest, split)
    return result


def _eligibility(row):
    reasons = []
    if row["status"] != "candidate":
        reasons.append("baseline_" + row["status"])
    pep = row["posterior_error_probability"]
    if pep is None or not math.isfinite(pep) or not 0 <= pep < 0.01:
        reasons.append("peptide_PEP_not_in_[0,.01)")
    if row["identification_type"] != "By MS/MS":
        reasons.append("not_direct_run_MS_MS_support")
    parents = _parents(row)
    if len(parents) != 1:
        reasons.append("not_one_distinct_supplied_parent")
    group = (
        _id("baseline_composition_group", parents[0], row["acquisition_id"])
        if len(parents) == 1 and row["acquisition_id"]
        else None
    )
    return reasons, group, parents


def compile_composition(source, destination):
    destination.mkdir(parents=True)
    baseline = compile_baseline(source, destination / "baseline")
    peptides = {r["peptide_id"]: r for r in _rows(source, "peptides")}
    components = dependence_components(peptides)
    groups = defaultdict(list)
    for row in _rows(destination / "baseline", "decisions"):
        reasons, group, parents = _eligibility(row)
        if not reasons:
            component, split = components["peptide:" + row["peptide_sequence"]]
            groups[group].append(
                {
                    **row,
                    "group_id": group,
                    "parent_accession": parents[0],
                    "component_id": component,
                    "split": split,
                }
            )
    small_groups = set()
    n_before_group = sum(len(rows) for rows in groups.values())
    for group, rows in groups.items():
        sequences = [r["peptide_sequence"] for r in rows]
        if len(set(sequences)) != len(sequences):
            raise ValueError("Duplicate peptide sequence within eligible MAPP composition group")
        if len(sequences) < CONTRACT["minimum_distinct_sequences_per_group"]:
            small_groups.add(group)
    groups = {key: rows for key, rows in groups.items() if key not in small_groups}
    group_count = Counter(rows[0]["component_id"] for rows in groups.values())
    support = {}
    for split in ("train", "val", "test"):
        selected = [rows for rows in groups.values() if rows[0]["split"] == split]
        support[split] = {
            "n_components": len({r[0]["component_id"] for r in selected}),
            "n_groups": len(selected),
            "n_observations": sum(len(r) for r in selected),
            "n_peptides": len({r["peptide_sequence"] for rows in selected for r in rows}),
        }
    (destination / "typed").mkdir()
    sinks = {
        name: _Sink(destination / "typed", name, schema=s) for name, s in OUTPUT_SCHEMAS.items()
    }
    reasons_count = Counter()
    try:
        for node, (component, split) in sorted(components.items()):
            sinks["components"].add({"node_id": node, "component_id": component, "split": split})
        for row in _rows(destination / "baseline", "decisions"):
            reasons, group, _ = _eligibility(row)
            if not reasons and group in small_groups:
                reasons.append("fewer_than_three_distinct_sequences_in_group")
            component, split = components["peptide:" + row["peptide_sequence"]]
            sinks["decisions"].add(
                {
                    "candidate_id": row["candidate_id"],
                    "measurement_id": row["measurement_id"],
                    "peptide_id": row["peptide_id"],
                    "component_id": component,
                    "split": split,
                    "group_id": group,
                    "baseline_status": row["status"],
                    "status": "excluded" if reasons else "admitted",
                    "reasons_json": _json(reasons),
                }
            )
            reasons_count.update(reasons)
        for group, rows in sorted(groups.items()):
            component, split = rows[0]["component_id"], rows[0]["split"]
            weight = 1 / (support[split]["n_components"] * group_count[component])
            center = math.fsum(r["log10_intensity"] for r in rows) / len(rows)
            total = math.fsum(r["quantitative_value"] for r in rows)
            sinks["groups"].add(
                {
                    "group_id": group,
                    "component_id": component,
                    "split": split,
                    "parent_accession": rows[0]["parent_accession"],
                    "acquisition_id": rows[0]["acquisition_id"],
                    "n_peptides": len(rows),
                    "group_weight": weight,
                }
            )
            for row in sorted(rows, key=lambda r: r["measurement_id"]):
                example = {
                    **row,
                    "target_centered_log10_intensity": row["log10_intensity"] - center,
                    "target_intensity_fraction": row["quantitative_value"] / total,
                    "weight": weight / len(rows),
                    "group_weight": weight,
                    "composition_eligibility_status": "conditional_observed_signal_only",
                }
                sinks["examples"].add(example)
                sinks["covariates"].add({k: v for k, v in example.items() if k not in LABEL_FIELDS})
        passed = (
            support["train"]["n_components"] >= 100
            and support["val"]["n_components"] >= 20
            and support["test"]["n_components"] >= 20
            and support["val"]["n_groups"] >= 20
            and support["test"]["n_groups"] >= 20
        )
        return {
            "baseline": baseline,
            "n_eligible_before_group": n_before_group,
            "n_small_groups": len(small_groups),
            "cohort_support": support,
            "support_passed": passed,
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
        raise RecipeError("Invalid Goldberg baseline composition recipe")


def describe(store):
    return {
        "contract": json.loads(json.dumps(CONTRACT)),
        "status": "materialized"
        if {"goldberg_" + n for n in TABLES} <= store.bindings.keys()
        else "not_in_generation",
    }


def materialize_composition(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if describe(store)["status"] != "materialized":
        raise RecipeError("Goldberg source is not materialized")
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
        summary = compile_composition(staging / "source_tables", staging / "composition")
        (staging / "contract.json").write_text(_json(CONTRACT) + "\n")
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_goldberg_composition",
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
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_composition_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    _validate(manifest["recipe"])
    source = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Goldberg composition source binding differs")
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("Goldberg composition contract differs")
    schemas = {
        "composition/typed/" + name + ".parquet": schema for name, schema in OUTPUT_SCHEMAS.items()
    }
    schemas.update(
        {
            "composition/baseline/" + name + ".parquet": DECISION_SCHEMA
            for name in ("decisions", "candidates")
        }
    )
    expected = (
        {"source_manifest.json", "contract.json", "summary.json"}
        | set(schemas)
        | {f"source_tables/{name}.parquet" for name in TABLES}
    )
    if set(manifest["files"]) != expected:
        raise RecipeError("Goldberg composition file inventory differs")
    for name in TABLES:
        relative = str(ROOT / "typed" / f"{name}.parquet")
        if (
            file_digest(directory / "source_tables" / f"{name}.parquet")
            != source["files"][relative]
        ):
            raise RecipeError("Goldberg composition source table differs")
        if (
            source_package is not None
            and file_digest(Path(source_package) / relative) != source["files"][relative]
        ):
            raise RecipeError("Goldberg composition source table changed")
    with tempfile.TemporaryDirectory(prefix="hitlist-goldberg-composition-") as temp:
        replay = Path(temp)
        summary = compile_composition(directory / "source_tables", replay / "composition")
        if summary != manifest["coverage"] or summary != json.loads(
            (directory / "summary.json").read_text()
        ):
            raise RecipeError("Goldberg composition coverage differs")
        for name, schema in schemas.items():
            left, right = pq.ParquetFile(replay / name), pq.ParquetFile(directory / name)
            if not right.schema_arrow.equals(schema):
                raise RecipeError("Goldberg composition schema differs")
            for expected_batch, actual in zip(
                left.iter_batches(), right.iter_batches(), strict=True
            ):
                if not expected_batch.equals(actual):
                    raise RecipeError("Goldberg composition rows differ from source decisions")
    return manifest
