"""Conditional representative-feature composition, never intracellular concentration."""

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
from .nanaware_lymph import ROOT, _id
from .provenance import file_digest

RECIPE_ID = "nanaware2024_observed_composition_v1"
TABLE_BINDINGS = {
    name: "lymph_" + name
    for name in (
        "observations",
        "quantities",
        "references",
        "identifications",
        "samples",
        "contexts",
        "source_records",
    )
}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "endpoint": "conditional_within_parent_sample_representative_feature_area_composition",
    "quantity": "positive_finite_original_PEAKS_area",
    "feature_choice": "highest_finite_minus_10lgP_then_numeric_charge_then_observation_id_before_area",
    "confidence_policy": "published_identifications_no_new_FDR_claim_or_threshold",
    "chemistry": "canonical_no_modifications_except_explicit_CAM_57.02_plus_minus_0.02_on_every_C",
    "origin_policy": "one_distinct_full_original_parent_sequence_all_exact_occurrences",
    "reference_candidates": "supplied_accession_and_same_study_deposited_nondecoy_peptide_evidence",
    "minimum_distinct_peptides_per_group": 3,
    "target": "centered_log10_area_and_area_fraction",
    "normalization": "within_block_sample_parent_over_admitted_positive_selected_features",
    "dependence": "all_source_parsed_peptides_accessions_candidate_parent_sequences_before_filtering",
    "split": "sha256(nanaware2024_composition_v1:component_id)_mod10_train0to5_val6to7_test8to9",
    "limitations": [
        "representative_charge_feature_not_total_peptide_molar_quantity",
        "positive_observed_set_conditioning_not_detection_probability",
        "human_extracellular_lymph_not_intracellular_APC",
        "mouse_DC_MHC_II_selected_not_free_peptide_pool",
        "database_and_library_searches_reuse_acquisitions",
        "unknown_precursor_abundance_kinetic_units_and_observation_response",
        "reported_length_start_conflicts_preserved_positions_derived_from_sequence",
        "same_study_components_not_independent_specimens_or_external_studies",
    ],
}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def accession_token(value):
    if not isinstance(value, str):
        return None
    value = value.removeprefix("sp|").removeprefix("tr|").split("|", 1)[0]
    return value if re.fullmatch(r"(?:[A-Z0-9]{6}|[A-Z0-9]{10})(?:-[0-9]+)?", value) else None


def _study(filename):
    return "human" if filename.startswith("PXD051024") else "dc"


def _occurrences(parent, peptide):
    result = []
    if not peptide:
        return result
    start = parent.find(peptide)
    while start >= 0:
        result.append([start, start + len(peptide)])
        start = parent.find(peptide, start + 1)
    return result


def _chemistry(row):
    raw = row["peptide_raw"] or ""
    flanked = re.fullmatch(r"[A-Z-]\.(.*)\.[A-Z-]", raw)
    if flanked:
        raw = flanked[1]
    pattern = r"([ACDEFGHIKLMNPQRSTVWY])(?:\(([+-]?[0-9]+(?:\.[0-9]+)?)\))?"
    parts = list(re.finditer(pattern, raw))
    if "".join(match[0] for match in parts) != raw or not parts:
        return False
    if "".join(match[1] for match in parts) != row["peptide_sequence"]:
        return False
    for match in parts:
        aa, delta = match.groups()
        if aa == "C":
            if delta is None or abs(float(delta) - 57.02) > 0.020000001:
                return False
        elif delta is not None:
            return False
    annotation = (row["peptide_modifications_raw"] or "").strip().casefold()
    return annotation in ("", "carbamidomethylation (c)", "carbamidomethyl (c)")


def _score(row):
    value = json.loads(row["confidence_json"]).get("minus_10lgP")
    return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None


def _choice(row):
    try:
        charge = float(row["charge_raw"])
        if not math.isfinite(charge):
            charge = math.inf
    except (TypeError, ValueError):
        charge = math.inf
    return (-_score(row), charge, row["observation_id"])


def _unique(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate source identity: {key}")
    return result


def compile_composition(tables):
    observations = _unique(tables["observations"], "observation_id")
    references = _unique(tables["references"], "reference_record_id")
    _unique(tables["quantities"], "measurement_id")
    samples = _unique(tables["samples"], "sample_id")
    contexts = _unique(tables["contexts"], "context_id")
    records = _unique(tables["source_records"], "source_record_id")
    by_accession, by_peptide = defaultdict(set), defaultdict(set)
    for key, row in references.items():
        token = accession_token(row["accession_raw"])
        if token and row["reference_sequence"]:
            by_accession[_study(row["source_file"]), token].add(key)
    for row in tables["identifications"]:
        if (row["is_decoy_raw"] or "").lower() in ("true", "1"):
            continue
        if row["reference_record_id"] not in references:
            raise ValueError("Unknown deposited reference")
        by_peptide[_study(row["source_file"]), row["peptide_sequence"]].add(
            row["reference_record_id"]
        )
    union = {}

    def find(node):
        union.setdefault(node, node)
        while union[node] != node:
            union[node] = union[union[node]]
            node = union[node]
        return node

    def connect(nodes):
        roots = sorted({find(node) for node in nodes})
        for root in roots[1:]:
            union[root] = roots[0]

    origin_rows, observation_nodes, parents = {}, {}, {}
    for key, row in sorted(observations.items()):
        study = "human" if row["block_id"] == "human_native" else "dc"
        peptide = row["peptide_sequence"]
        token = accession_token(row["parent_label_raw"])
        candidates = by_accession[study, token] | by_peptide[study, peptide]
        mappings = defaultdict(list)
        for reference_id in sorted(candidates):
            ref = references[reference_id]
            sequence = ref["reference_sequence"]
            if sequence and _occurrences(sequence, peptide):
                parent_id = hashlib.sha256(sequence.encode()).hexdigest()
                parents[parent_id] = sequence
                mappings[parent_id].append(reference_id)
        origins = [
            {
                "parent_id": parent_id,
                "reference_record_ids": ids,
                "occurrences_0based_halfopen": _occurrences(parents[parent_id], peptide),
            }
            for parent_id, ids in sorted(mappings.items())
        ]
        origin_rows[key] = origins
        nodes = [
            "observation:" + key,
            "accession:" + study + ":" + (token or row["parent_label_raw"] or key),
        ]
        if peptide:
            nodes.append("peptide:" + peptide)
        nodes.extend("parent:" + origin["parent_id"] for origin in origins)
        observation_nodes[key] = nodes[0]
        connect(nodes)
    members = defaultdict(list)
    for node in sorted(union):
        members[find(node)].append(node)
    component_ids = {root: _id("component", nodes) for root, nodes in members.items()}
    components = [
        {"component_id": component_ids[root], "nodes": nodes}
        for root, nodes in sorted(members.items())
    ]
    component = {key: component_ids[find(node)] for key, node in observation_nodes.items()}
    splits = {}
    for group in components:
        key = group["component_id"]
        bucket = (
            int(hashlib.sha256(("nanaware2024_composition_v1:" + key).encode()).hexdigest(), 16)
            % 10
        )
        splits[key] = "train" if bucket < 6 else "val" if bucket < 8 else "test"
        group["split"] = splits[key]
    reasons_by_obs, choice_groups = {}, defaultdict(list)
    for key, row in sorted(observations.items()):
        reasons = []
        if not row["peptide_sequence"] or not set(row["peptide_sequence"]) <= AA:
            reasons.append("noncanonical_or_unparsed_peptide")
        if not _chemistry(row):
            reasons.append("observer_chemistry_out_of_scope")
        origins = origin_rows[key]
        if len(origins) != 1:
            reasons.append("missing_or_ambiguous_original_parent")
        elif not set(parents[origins[0]["parent_id"]]) <= AA:
            reasons.append("noncanonical_original_parent")
        if _score(row) is None:
            reasons.append("missing_finite_source_confidence")
        reasons_by_obs[key] = reasons
        if not reasons:
            choice_groups[row["block_id"], row["peptide_sequence"], origins[0]["parent_id"]].append(
                row
            )
    selected = {min(rows, key=_choice)["observation_id"] for rows in choice_groups.values()}
    for rows in choice_groups.values():
        for row in rows:
            if row["observation_id"] not in selected:
                reasons_by_obs[row["observation_id"]].append("not_selected_representative_feature")
    dispositions, candidate_groups = [], defaultdict(list)
    for quantity in sorted(tables["quantities"], key=lambda row: row["measurement_id"]):
        key = quantity["observation_id"]
        row = observations[key]
        sample = samples[quantity["sample_id"]]
        if (
            quantity["source_record_id"] != row["source_record_id"]
            or sample["context_id"] != quantity["context_id"]
        ):
            raise ValueError("Inconsistent source join")
        if sample["context_id"] not in contexts or row["source_record_id"] not in records:
            raise ValueError("Missing context or original row")
        reasons = list(reasons_by_obs[key])
        if quantity["quantity_kind"] != "area":
            reasons.append("not_area")
        value = quantity["quantitative_value"]
        if (
            quantity["value_status"] != "positive"
            or value is None
            or not math.isfinite(value)
            or value <= 0
        ):
            reasons.append("not_positive_finite_area")
        disposition = {
            "measurement_id": quantity["measurement_id"],
            "observation_id": key,
            "source_record_id": row["source_record_id"],
            "source_cell": quantity["source_cell"],
            "block_id": row["block_id"],
            "sample_id": sample["sample_id"],
            "context_id": sample["context_id"],
            "component_id": component[key],
            "split": splits[component[key]],
            "sequence": row["peptide_sequence"],
            "reasons": reasons,
            "admitted": False,
        }
        dispositions.append(disposition)
        if not reasons:
            parent_id = origin_rows[key][0]["parent_id"]
            candidate_groups[row["block_id"], sample["sample_id"], parent_id].append(
                (disposition, quantity, row)
            )
    examples = []
    for group_key, rows in sorted(candidate_groups.items()):
        if len({row[0]["sequence"] for row in rows}) != len(rows):
            raise ValueError("Repeated selected sequence within sample/parent")
        if len(rows) < CONTRACT["minimum_distinct_peptides_per_group"]:
            for disposition, _, _ in rows:
                disposition["reasons"].append("fewer_than_three_positive_selected_peptides")
            continue
        group_id = _id("composition", *group_key)
        mean_log = math.fsum(math.log10(row[1]["quantitative_value"]) for row in rows) / len(rows)
        total = math.fsum(row[1]["quantitative_value"] for row in rows)
        for disposition, quantity, row in rows:
            disposition["admitted"] = True
            examples.append(
                {
                    **{k: v for k, v in disposition.items() if k not in ("reasons", "admitted")},
                    "group_id": group_id,
                    "parent_id": group_key[2],
                    "reference_record_ids": origin_rows[row["observation_id"]][0][
                        "reference_record_ids"
                    ],
                    "occurrences_0based_halfopen": origin_rows[row["observation_id"]][0][
                        "occurrences_0based_halfopen"
                    ],
                    "area": quantity["quantitative_value"],
                    "log10_area": math.log10(quantity["quantitative_value"]),
                    "target_centered_log10_area": math.log10(quantity["quantitative_value"])
                    - mean_log,
                    "target_area_fraction": quantity["quantitative_value"] / total,
                    "confidence_json": row["confidence_json"],
                    "charge_raw": row["charge_raw"],
                    "peptide_raw": row["peptide_raw"],
                    "reported_length_status": row["length_status"],
                    "reported_start_raw": row["start_raw"],
                    "reported_end_raw": row["end_raw"],
                }
            )
    coverage = []
    for block in sorted({row["block_id"] for row in observations.values()}):
        for split in ("train", "val", "test"):
            rows = [row for row in examples if row["block_id"] == block and row["split"] == split]
            coverage.append(
                {
                    "block_id": block,
                    "split": split,
                    "n_measurements": len(rows),
                    **{
                        "n_" + label: len({r[key] for r in rows})
                        for label, key in (
                            ("groups", "group_id"),
                            ("components", "component_id"),
                            ("parents", "parent_id"),
                            ("peptides", "sequence"),
                        )
                    },
                }
            )
    return {
        "examples": sorted(examples, key=lambda row: row["measurement_id"]),
        "dispositions": dispositions,
        "dependence_groups": components,
        "origins": [
            {"observation_id": key, "component_id": component[key], "candidates": value}
            for key, value in sorted(origin_rows.items())
        ],
        "parents": [
            {"parent_id": key, "sequence": value} for key, value in sorted(parents.items())
        ],
        "summary": {
            "n_source_observations": len(observations),
            "n_quantity_cells": len(dispositions),
            "n_admitted_measurements": len(examples),
            "coverage": coverage,
            "reason_counts": dict(
                sorted(Counter(r for row in dispositions for r in row["reasons"]).items())
            ),
        },
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
        raise RecipeError("Invalid registered lymph composition recipe")


def describe(store):
    return {
        "contract": json.loads(json.dumps(CONTRACT)),
        "status": "materialized"
        if set(TABLE_BINDINGS.values()) <= store.bindings.keys()
        else "not_in_generation",
    }


def materialize_response(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if not set(TABLE_BINDINGS.values()) <= store.bindings.keys():
        raise RecipeError("Lymph composition source is not materialized")
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
    compiled = compile_composition(tables)
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
            "output_kind": "registered_lymph_composition",
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
        raise RecipeError("Lymph composition recipe source binding differs")
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("Lymph composition response contract differs")
    tables = {}
    for table in TABLE_BINDINGS:
        path = directory / "source_tables" / f"{table}.parquet"
        relative = str(ROOT / "typed" / f"{table}.parquet")
        if file_digest(path) != source["files"][relative]:
            raise RecipeError("Lymph composition recipe source table differs")
        if (
            source_package is not None
            and file_digest(Path(source_package) / relative) != source["files"][relative]
        ):
            raise RecipeError("Lymph composition source table changed")
        tables[table] = pq.read_table(path).to_pylist()
    compiled = compile_composition(tables)
    expected = (
        {"source_manifest.json", "contract.json"}
        | {f"source_tables/{table}.parquet" for table in TABLE_BINDINGS}
        | {name + ".json" for name in compiled}
    )
    if set(manifest["files"]) != expected:
        raise RecipeError("Lymph composition response inventory differs")
    for name, value in compiled.items():
        if json.loads((directory / f"{name}.json").read_text()) != value:
            raise RecipeError(f"Lymph composition response replay differs: {name}")
    if manifest["coverage"] != compiled["summary"]:
        raise RecipeError("Lymph composition response coverage differs")
    return manifest
