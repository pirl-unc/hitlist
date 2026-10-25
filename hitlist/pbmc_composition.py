"""Observed PBMC MAPP composition, retaining source and material uncertainty."""

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
from .mapp_evidence import _schema, _Sink
from .provenance import file_digest

RECIPE_ID = "pxd009438_observed_baseline_composition_v1"
ROOT = Path("external/pxd009438")
TABLE_BINDINGS = {
    "source_records": "mapp_source_record",
    "peptide_rows": "mapp_peptide",
    "groups": "mapp_group",
    "channels": "mapp_acquisition",
    "evidence": "mapp_identification",
    "quantity_results": "mapp_quantity_result",
    "quantity_measurements": "mapp_quantity_measurement",
    "derivation_inputs": "mapp_derivation_input",
    "source_assertions": "mapp_source_assertion",
    "material_contexts": "mapp_material_context",
    "material_links": "mapp_material_link",
    "label_aliases": "mapp_label_alias",
}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "endpoint": "conditional_observed_source_accession_acquisition_MAPP_intensity_composition",
    "cohorts": {"CTRL": "primary", "LUPUS": "secondary_descriptive"},
    "split": "external_test_only_no_training_or_validation_partition",
    "confidence": "peptide_PEP_in_[0,.01)_and_per_run_By_MS/MS_and_positive_direct_evidence",
    "intensity": "positive_finite_reported_run_intensity_with_all_evidence_intensities_reported",
    "flags": "original_peptide_reverse_and_contaminant_explicitly_blank",
    "origin": "one_distinct_supplied_parent_accession_not_verified_full_reference",
    "species": "human_project_search_scope_not_verified_per_parent",
    "minimum_distinct_sequences_per_group": 3,
    "duplicate_sequence_within_eligible_group": "exclude_all_ambiguous_rows_no_sum_or_selection",
    "dependence": "all_original_sequence_and_parent_nodes_including_excluded_bridges",
    "component_id": "sha256(pxd009438_composition_v1:+compact_sorted_node_JSON)",
    "group_id": "pxd009438:composition-group:+sha256(compact_JSON_[parent,channel_id])",
    "weights": "equal_component_then_group_then_peptide_separately_within_each_cohort",
    "targets": "centered_log10_intensity_and_observed_set_intensity_fraction",
    "support_per_cohort": {"n_components": 20, "n_groups": 20, "n_observations": 100},
    "training_status": "conditional_captured_signal_only_not_full_processing_eligibility",
    "limitations": [
        "not_molarity_or_complete_free_peptide_distribution",
        "PBMC_mixed_immune_population_not_purified_APC",
        "publication_PBMC_conflicts_with_repository_blood_serum",
        "positive_observed_set_conditioning",
        "sequence_totals_may_aggregate_forms",
        "no_full_original_search_reference_or_verified_parent_species",
        "specimen_aliquot_homology_and_exact_precursor_pairing_unresolved",
        "source_confidence_not_new_FDR_guarantee",
    ],
}
DECISIONS = _schema(
    (
        "measurement_id",
        "peptide_row_id",
        "channel_id",
        "cohort",
        "component_id",
        "group_id",
        "status",
        "reasons_json",
    )
)
EXAMPLES = _schema(
    (
        "measurement_id",
        "source_record_id",
        "peptide_row_id",
        "channel_id",
        "acquisition_id",
        "cohort",
        "split",
        "component_id",
        "group_id",
        "peptide_sequence",
        "parent_accession",
        "identification_type",
        "measured_pool",
        "material_link_precision",
        "species_scope",
        "parent_origin_status",
        "composition_eligibility_status",
    ),
    numbers=(
        "quantitative_value",
        "log10_intensity",
        "target_centered_log10_intensity",
        "target_intensity_fraction",
        "weight",
        "group_weight",
    ),
)
LABEL_FIELDS = {
    "quantitative_value",
    "log10_intensity",
    "target_centered_log10_intensity",
    "target_intensity_fraction",
}
COVARIATES = pa.schema([field for field in EXAMPLES if field.name not in LABEL_FIELDS])
COMPONENTS = _schema(("node_id", "component_id"))
GROUPS = _schema(
    ("group_id", "component_id", "cohort", "parent_accession", "acquisition_id"),
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


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rows(source, table):
    return pq.read_table(source / f"{table}.parquet").to_pylist()


def _parents(row):
    return sorted({p for p in (row["parent_labels_raw"] or "").split(";") if p})


def dependence_components(peptides):
    union = {}

    def find(node):
        union.setdefault(node, node)
        while union[node] != node:
            union[node] = union[union[node]]
            node = union[node]
        return node

    for row in peptides:
        nodes = ["peptide:" + row["sequence_raw"]] + ["parent:" + p for p in _parents(row)]
        roots = sorted({find(node) for node in nodes})
        for root in roots[1:]:
            union[root] = roots[0]
    members = defaultdict(list)
    for node in sorted(union):
        members[find(node)].append(node)
    result = {}
    for nodes in members.values():
        digest = hashlib.sha256(("pxd009438_composition_v1:" + _json(nodes)).encode()).hexdigest()
        result.update(dict.fromkeys(nodes, digest))
    return result


def compile_composition(source, destination):
    peptides = {r["peptide_row_id"]: r for r in _rows(source, "peptide_rows")}
    records = {
        r["source_record_id"]: json.loads(r["fields_json"]) for r in _rows(source, "source_records")
    }
    channels = {r["channel_id"]: r for r in _rows(source, "channels")}
    results = {r["result_id"]: r for r in _rows(source, "quantity_results")}
    components = dependence_components(peptides.values())
    decisions, candidates = [], defaultdict(list)
    measurements = [
        r
        for r in _rows(source, "quantity_measurements")
        if r["representation"] == "reported_run_intensity"
    ]
    if len({r["result_id"] for r in measurements}) != len(results) or len(measurements) != len(
        results
    ):
        raise ValueError("PBMC composition requires one reported measurement per result")
    for measurement in sorted(measurements, key=lambda r: r["measurement_id"]):
        result = results[measurement["result_id"]]
        peptide, channel = peptides[result["peptide_row_id"]], channels[result["channel_id"]]
        raw = records[peptide["source_record_id"]]
        value, sequence, parents = measurement["value"], peptide["sequence_raw"], _parents(peptide)
        cohort = channel["condition"]
        identification = raw.get("Identification type " + channel["experiment_raw"])
        reasons = []
        if cohort not in CONTRACT["cohorts"]:
            reasons.append("outside_registered_source_cohorts")
        if value is None or not math.isfinite(value) or value <= 0:
            reasons.append("not_positive_finite_original_run_intensity")
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            reasons.append("noncanonical_sequence")
        if raw.get("Reverse") != "" or raw.get("Potential contaminant") != "":
            reasons.append("peptide_flags_not_explicitly_blank")
        pep = peptide["pep"]
        if pep is None or not math.isfinite(pep) or not 0 <= pep < 0.01:
            reasons.append("peptide_PEP_not_in_[0,.01)")
        if identification != "By MS/MS":
            reasons.append("not_direct_run_MS_MS_identification")
        if not result["n_positive_direct_msms_evidence"]:
            reasons.append("no_positive_direct_MS_MS_evidence")
        if result["n_numeric_evidence"] != result["n_source_evidence"]:
            reasons.append("incomplete_evidence_intensities")
        if len(parents) != 1:
            reasons.append("not_one_distinct_supplied_parent")
        group = (
            "pxd009438:composition-group:"
            + hashlib.sha256(_json([parents[0], result["channel_id"]]).encode()).hexdigest()
            if len(parents) == 1
            else None
        )
        decision = {
            "measurement_id": measurement["measurement_id"],
            "peptide_row_id": peptide["peptide_row_id"],
            "channel_id": result["channel_id"],
            "cohort": cohort,
            "component_id": components["peptide:" + sequence],
            "group_id": group,
            "reasons": reasons,
        }
        decisions.append(decision)
        if not reasons:
            candidates[group].append(
                (
                    decision,
                    {
                        "measurement_id": measurement["measurement_id"],
                        "source_record_id": peptide["source_record_id"],
                        "peptide_row_id": peptide["peptide_row_id"],
                        "channel_id": result["channel_id"],
                        "acquisition_id": channel["experiment_raw"],
                        "cohort": cohort,
                        "split": "test",
                        "component_id": decision["component_id"],
                        "group_id": group,
                        "peptide_sequence": sequence,
                        "parent_accession": parents[0],
                        "identification_type": identification,
                        "measured_pool": channel["measured_pool"],
                        "material_link_precision": channel["material_link_precision"],
                        "species_scope": "human_project_search_scope_not_verified_per_parent",
                        "parent_origin_status": "supplied_accession_full_original_reference_unverified",
                        "composition_eligibility_status": "conditional_observed_signal_only",
                        "quantitative_value": value,
                        "log10_intensity": math.log10(value),
                    },
                )
            )
    n_before_group = sum(len(rows) for rows in candidates.values())
    groups, n_duplicates, n_small_groups = {}, 0, 0
    for group, pairs in candidates.items():
        counts = Counter(row["peptide_sequence"] for _, row in pairs)
        kept = []
        for decision, row in pairs:
            if counts[row["peptide_sequence"]] > 1:
                decision["reasons"].append("duplicate_eligible_sequence_in_group")
                n_duplicates += 1
            else:
                kept.append((decision, row))
        if len(kept) < 3:
            n_small_groups += 1
            for decision, _ in kept:
                decision["reasons"].append("fewer_than_three_distinct_sequences_in_group")
        else:
            groups[group] = [row for _, row in kept]
    support, group_counts = {}, Counter()
    for rows in groups.values():
        group_counts[(rows[0]["cohort"], rows[0]["component_id"])] += 1
    for cohort in CONTRACT["cohorts"]:
        selected = [rows for rows in groups.values() if rows[0]["cohort"] == cohort]
        support[cohort] = {
            "n_components": len({rows[0]["component_id"] for rows in selected}),
            "n_groups": len(selected),
            "n_observations": sum(map(len, selected)),
            "n_peptides": len({r["peptide_sequence"] for rows in selected for r in rows}),
        }
    destination.mkdir(parents=True)
    (destination / "typed").mkdir()
    sinks = {
        name: _Sink(destination / "typed", name, schema=s) for name, s in OUTPUT_SCHEMAS.items()
    }
    reason_counts = Counter()
    try:
        for node, component in sorted(components.items()):
            sinks["components"].add({"node_id": node, "component_id": component})
        for decision in decisions:
            reasons = decision.pop("reasons")
            reason_counts.update(reasons)
            sinks["decisions"].add(
                {
                    **decision,
                    "status": "excluded" if reasons else "admitted",
                    "reasons_json": _json(reasons),
                }
            )
        for group, rows in sorted(groups.items()):
            first = rows[0]
            cohort, component = first["cohort"], first["component_id"]
            gw = 1 / (support[cohort]["n_components"] * group_counts[(cohort, component)])
            center = math.fsum(r["log10_intensity"] for r in rows) / len(rows)
            total = math.fsum(r["quantitative_value"] for r in rows)
            sinks["groups"].add(
                {
                    "group_id": group,
                    "component_id": component,
                    "cohort": cohort,
                    "parent_accession": first["parent_accession"],
                    "acquisition_id": first["acquisition_id"],
                    "n_peptides": len(rows),
                    "group_weight": gw,
                }
            )
            for row in rows:
                example = {
                    **row,
                    "target_centered_log10_intensity": row["log10_intensity"] - center,
                    "target_intensity_fraction": row["quantitative_value"] / total,
                    "weight": gw / len(rows),
                    "group_weight": gw,
                }
                sinks["examples"].add(example)
                sinks["covariates"].add({k: v for k, v in example.items() if k not in LABEL_FIELDS})
        return {
            "n_eligible_before_group": n_before_group,
            "n_duplicate_measurements": n_duplicates,
            "n_small_groups": n_small_groups,
            "cohort_support": support,
            "support_passed": {
                c: all(counts[k] >= v for k, v in CONTRACT["support_per_cohort"].items())
                for c, counts in support.items()
            },
            "reason_counts": dict(sorted(reason_counts.items())),
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
        raise RecipeError("Invalid PBMC composition recipe")


def describe(store):
    return {
        "contract": json.loads(_json(CONTRACT)),
        "status": "materialized"
        if set(TABLE_BINDINGS.values()) <= store.bindings.keys()
        else "not_in_generation",
    }


def materialize_composition(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if describe(store)["status"] != "materialized":
        raise RecipeError("PBMC source is not materialized")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        for name in TABLE_BINDINGS:
            snapshot_file(
                store.directory / ROOT / "typed" / f"{name}.parquet",
                staging / "source_tables" / f"{name}.parquet",
            )
        summary = compile_composition(staging / "source_tables", staging / "composition")
        (staging / "contract.json").write_text(_json(CONTRACT) + "\n")
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_pbmc_composition",
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
        raise RecipeError("PBMC composition source binding differs")
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("PBMC composition contract differs")
    schemas = {
        "composition/typed/" + name + ".parquet": schema for name, schema in OUTPUT_SCHEMAS.items()
    }
    expected = (
        {"source_manifest.json", "contract.json", "summary.json"}
        | set(schemas)
        | {f"source_tables/{name}.parquet" for name in TABLE_BINDINGS}
    )
    if set(manifest["files"]) != expected:
        raise RecipeError("PBMC composition file inventory differs")
    for name in TABLE_BINDINGS:
        relative = str(ROOT / "typed" / f"{name}.parquet")
        if (
            file_digest(directory / "source_tables" / f"{name}.parquet")
            != source["files"][relative]
        ):
            raise RecipeError("PBMC composition source table differs")
        if (
            source_package is not None
            and file_digest(Path(source_package) / relative) != source["files"][relative]
        ):
            raise RecipeError("PBMC composition source table changed")
    with tempfile.TemporaryDirectory(prefix="hitlist-pbmc-composition-") as temp:
        replay = Path(temp)
        summary = compile_composition(directory / "source_tables", replay / "composition")
        if summary != manifest["coverage"] or summary != json.loads(
            (directory / "summary.json").read_text()
        ):
            raise RecipeError("PBMC composition coverage differs")
        for name, schema in schemas.items():
            left, right = pq.read_table(replay / name), pq.read_table(directory / name)
            if not right.schema.equals(schema) or not left.equals(right):
                raise RecipeError("PBMC composition rows or schema differ from source")
    return manifest
