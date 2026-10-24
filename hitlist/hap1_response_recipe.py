"""Source-linked HAP1 machinery responses; no imputed non-detections or origins."""

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
from .hap1_quantitative import REVISION
from .mapp_evidence import _schema, _Sink
from .provenance import file_digest

RECIPE_ID = "hap1_machinery_response_v1"
TABLES = (
    "contexts",
    "samples",
    "runs",
    "quantities",
    "source_records",
    "references",
    "occurrences",
)
AA = set("ACDEFGHIKLMNPQRSTVWY")
WT = "hap1_wildtype"
B2M = "hap1_b2m_ko"
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "source_revision": REVISION,
    "endpoint": "canonical_HLA_I_eluted_peptide_KO_vs_WT_response",
    "target": "log10_positive_KO_cell_minus_mean_log10_at_least_two_positive_WT_cells",
    "background": "B2M_control_only;exclude_sequence_positive_in_any_B2M_replicate",
    "observation": "one_supplement_biological_replicate_cell;technical_runs_already_averaged",
    "origins": "all_exact_full_sequence_groups;each_requires_human_target_support",
    "precursor": "mmc8_MaxLFQ;one_unambiguous_group_per_parent;at_least_two_positive_cells_per_condition",
    "precursor_summary": "mean_parent_log10_group_response;equal_distinct_parent_weight",
    "components": "all_canonical_and_noncanonical_peptides_claims_exact_sequences_and_protein_groups_before_filtering",
    "split": "sha256(recipe_id+component_id)_mod20:0-13_train,14-16_validation,17-19_test",
    "weights": "consumer_subset_average_condition_component_peptide_replicate",
    "missingness": "preserve_missing_zero_nonfinite_unparsed;no_imputation_or_inferred_bounds",
    "limitations": [
        "HAP1_is_not_professional_APC;elution_is_not_intracellular_concentration",
        "condition_level_protein_surrogate_not_paired_aliquot_or_molar_abundance",
        "source_reported_KO_not_measured_residual_activity",
        "shared_WT_controls;study_clone_specimen_and_homology_independence_unresolved",
        "candidate_origins_and_equal_parent_weights_not_measured_molecular_origins",
        "selection_transport_processing_and_detection_not_identified_separately",
        "positive_complete_case_endpoint_excludes_unobserved_peptides",
        "source_reported_identifications_not_new_FDR_validation",
    ],
}
SCHEMAS = {
    "components": _schema(("peptide_sequence", "component_id", "split"), ("n_component_nodes",)),
    "parents": _schema(("parent_id", "reference_sequence", "reference_ids_json", "roles_json")),
    "protein_groups": _schema(("group_id", "source_record_id", "members_json")),
    "precursor_inputs": _schema(
        (
            "parent_id",
            "condition_id",
            "status",
            "group_ids_json",
            "wt_measurement_ids_json",
            "ko_measurement_ids_json",
        ),
        numbers=("log10_response",),
    ),
    "responses": _schema(
        (
            "response_id",
            "measurement_id",
            "source_record_id",
            "sample_id",
            "peptide_sequence",
            "condition_id",
            "component_id",
            "split",
            "parent_ids_json",
            "wt_measurement_ids_json",
            "precursor_status",
        ),
        numbers=("target_log10_ratio", "precursor_log10_ratio"),
    ),
    "dispositions": _schema(("measurement_id", "source_record_id", "status", "reason")),
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path, value):
    path.write_text(_json(value) + "\n")


def _read(path):
    return json.loads(path.read_text())


def _rows(root, name, columns=None):
    for batch in pq.ParquetFile(root / f"{name}.parquet").iter_batches(
        batch_size=4096, columns=columns
    ):
        yield from batch.to_pylist()


def _tokens(value):
    return sorted(set(filter(None, re.split(r"[;,\s]+", value or ""))))


class _Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            previous = self.parent[key]
            self.parent[key] = root
            key = previous
        return root

    def union(self, first, second):
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)

    def resolved(self):
        groups = defaultdict(list)
        for key in self.parent:
            groups[self.find(key)].append(key)
        result = {}
        for nodes in groups.values():
            component = _hash(sorted(nodes))
            number = int(hashlib.sha256((RECIPE_ID + component).encode()).hexdigest(), 16) % 20
            split = "train" if number < 14 else "validation" if number < 17 else "test"
            for node in nodes:
                result[node] = component, split, len(nodes)
        return result


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe.get("schema_version") != 1
        or recipe.get("registered_recipe") != RECIPE_ID
        or not isinstance(recipe.get("name"), str)
        or not recipe["name"].strip()
    ):
        raise RecipeError("Invalid HAP1 response recipe")


def describe(store):
    return {
        "contract": CONTRACT,
        "status": "available"
        if {"hap1_" + name for name in TABLES} <= store.bindings.keys()
        else "not_in_generation",
    }


def compile_responses(source, destination):
    """Compile typed source snapshots deterministically, streaming quantity cells."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    graph = _Components()
    contexts = {row["context_id"]: row for row in _rows(source, "contexts")}
    samples = {row["sample_id"]: row for row in _rows(source, "samples")}
    references = {}
    parent_refs, parent_roles = defaultdict(list), defaultdict(set)
    parent_sequences, token_groups, groups = {}, defaultdict(set), {}
    for row in _rows(source, "references"):
        ref, parent = row["reference_record_id"], row["sequence_sha256"]
        if hashlib.sha256(row["reference_sequence"].encode()).hexdigest() != parent:
            raise ValueError("HAP1 reference sequence digest changed")
        references[ref] = row
        parent_refs[parent].append(ref)
        parent_roles[parent].add(row["reference_role"])
        parent_sequences[parent] = row["reference_sequence"]
        graph.union("parent:" + parent, "claim:" + row["accession_raw"])
    for row in _rows(source, "source_records"):
        if row["source_role"] != "deposited_protein_report":
            continue
        raw = json.loads(row["source_row_json"])
        leading = raw["Protein"]
        if leading in groups:
            raise ValueError("Ambiguous duplicate HAP1 leading protein groups")
        members = sorted({leading, *_tokens(raw.get("Indistinguishable Proteins"))})
        groups[leading] = {
            "group_id": leading,
            "source_record_id": row["source_record_id"],
            "members_json": _json(members),
        }
        for member in members:
            token_groups[member].add(leading)
            graph.union("group:" + leading, "claim:" + member)
    peptide_parents, peptide_rows = defaultdict(set), defaultdict(set)
    for row in _rows(source, "occurrences"):
        peptide = row["peptide_sequence"]
        node = "peptide:" + peptide
        graph.find(node)
        peptide_rows[peptide].add(row["source_record_id"])
        for claim in _tokens(row["parent_label_raw"]):
            graph.union(node, "claim:" + claim)
        if row["reference_record_id"] is not None:
            ref = references[row["reference_record_id"]]
            parent = ref["sequence_sha256"]
            start, end = row["start_1based"], row["end_1based"]
            if ref["reference_sequence"][start - 1 : end] != peptide:
                raise ValueError("Invalid HAP1 source occurrence")
            graph.union(node, "parent:" + parent)
            peptide_parents[peptide].add(parent)
    components = graph.resolved()
    parent_groups = {}
    for parent, refs in parent_refs.items():
        parent_groups[parent] = sorted(
            {group for ref in refs for group in token_groups[references[ref]["accession_raw"]]}
        )
    positive_peptide, positive_protein = defaultdict(list), defaultdict(list)
    canonical_rows, background = defaultdict(set), set()
    n_cells = 0
    for row in _rows(source, "quantities"):
        n_cells += 1
        kind = row["quantity_kind"]
        if kind not in {"normalized_peptide_intensity", "protein_MaxLFQ_intensity"}:
            continue
        sample = samples.get(row["sample_id"])
        context = contexts.get(row["context_id"])
        if sample is None or context is None or sample["context_id"] != row["context_id"]:
            raise ValueError("Unresolved HAP1 primary quantity context")
        condition = context["condition_id"]
        peptide = row["peptide_sequence"]
        if kind == "normalized_peptide_intensity":
            canonical_rows[peptide].add(row["source_record_id"])
        if row["value_status"] != "source_positive":
            continue
        value = row["quantitative_value"]
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid positive HAP1 quantity")
        cell = (row["measurement_id"], row["sample_id"], value)
        if kind == "normalized_peptide_intensity":
            positive_peptide[peptide, condition].append(cell)
            if condition == B2M:
                background.add(peptide)
        else:
            positive_protein[row["parent_label_raw"], condition].append(cell)
    if any(len(rows) != 1 for rows in canonical_rows.values()):
        raise ValueError("Duplicate literal canonical HAP1 peptide rows need an explicit policy")
    for cells in [*positive_peptide.values(), *positive_protein.values()]:
        if len({sample for _, sample, _ in cells}) != len(cells):
            raise ValueError("Duplicate HAP1 source biological-label quantities")
    precursor_cache, group_cache = {}, {}

    def precursor(parent, condition):
        key = parent, condition
        if key in precursor_cache:
            return precursor_cache[key]
        group_ids = parent_groups[parent]
        wt, ko, value = [], [], None
        if len(group_ids) != 1:
            status = "missing_protein_group" if not group_ids else "ambiguous_protein_groups"
        else:
            group = group_ids[0]
            group_key = group, condition
            if group_key not in group_cache:
                wt = positive_protein[group, WT]
                ko = positive_protein[group, condition]
                if len(wt) >= 2 and len(ko) >= 2:
                    value = sum(math.log10(c[2]) for c in ko) / len(ko) - sum(
                        math.log10(c[2]) for c in wt
                    ) / len(wt)
                    status = "complete"
                else:
                    status = "insufficient_positive_protein_replicates"
                group_cache[group_key] = status, wt, ko, value
            status, wt, ko, value = group_cache[group_key]
        result = {
            "parent_id": parent,
            "condition_id": condition,
            "status": status,
            "group_ids_json": _json(group_ids),
            "wt_measurement_ids_json": _json(sorted(c[0] for c in wt)),
            "ko_measurement_ids_json": _json(sorted(c[0] for c in ko)),
            "log10_response": value,
        }
        precursor_cache[key] = result
        return result

    sinks = {name: _Sink(destination, name, schema=schema) for name, schema in SCHEMAS.items()}
    reasons, response_splits, complete_splits, response_conditions = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    used_peptides, used_parents = set(), set()
    try:
        for peptide in sorted(peptide_rows):
            component, split, n_nodes = components["peptide:" + peptide]
            sinks["components"].add(
                {
                    "peptide_sequence": peptide,
                    "component_id": component,
                    "split": split,
                    "n_component_nodes": n_nodes,
                }
            )
        for group in sorted(groups):
            sinks["protein_groups"].add(groups[group])
        for row in _rows(source, "quantities"):
            kind = row["quantity_kind"]
            reason, status = "", "deferred"
            if kind != "normalized_peptide_intensity":
                reason = (
                    "protein_covariate_only"
                    if kind == "protein_MaxLFQ_intensity"
                    else "nonprimary_quantity_kind"
                )
            else:
                condition = contexts[row["context_id"]]["condition_id"]
                peptide = row["peptide_sequence"]
                parents = sorted(peptide_parents[peptide])
                wt = positive_peptide[peptide, WT]
                if condition == WT:
                    reason = "WT_reference_only"
                elif condition == B2M:
                    reason = "B2M_background_only"
                elif peptide in background:
                    reason, status = "positive_B2M_background_sequence", "excluded"
                elif not peptide or set(peptide) - AA:
                    reason = "unsupported_peptide_alphabet"
                elif not parents or any("human_target" not in parent_roles[p] for p in parents):
                    reason = "unsupported_candidate_parent_groups"
                elif row["value_status"] != "source_positive":
                    reason = row["value_status"]
                elif len(wt) < 2:
                    reason = "insufficient_positive_WT_replicates"
                else:
                    status = "admitted"
                    component, split, _ = components["peptide:" + peptide]
                    parent_inputs = [precursor(parent, condition) for parent in parents]
                    complete = all(p["status"] == "complete" for p in parent_inputs)
                    value = (
                        sum(p["log10_response"] for p in parent_inputs) / len(parents)
                        if complete
                        else None
                    )
                    sinks["responses"].add(
                        {
                            "response_id": "hap1:response:" + _hash(row["measurement_id"]),
                            "measurement_id": row["measurement_id"],
                            "source_record_id": row["source_record_id"],
                            "sample_id": row["sample_id"],
                            "peptide_sequence": peptide,
                            "condition_id": condition,
                            "component_id": component,
                            "split": split,
                            "parent_ids_json": _json(parents),
                            "wt_measurement_ids_json": _json(sorted(c[0] for c in wt)),
                            "precursor_status": "complete" if complete else "incomplete",
                            "target_log10_ratio": math.log10(row["quantitative_value"])
                            - sum(math.log10(c[2]) for c in wt) / len(wt),
                            "precursor_log10_ratio": value,
                        }
                    )
                    response_splits[split] += 1
                    if complete:
                        complete_splits[split] += 1
                    response_conditions[condition] += 1
                    used_peptides.add(peptide)
                    used_parents.update(parents)
            reasons[status + ":" + (reason or "primary_response")] += 1
            sinks["dispositions"].add(
                {
                    "measurement_id": row["measurement_id"],
                    "source_record_id": row["source_record_id"],
                    "status": status,
                    "reason": reason or None,
                }
            )
        for parent in sorted(used_parents):
            sinks["parents"].add(
                {
                    "parent_id": parent,
                    "reference_sequence": parent_sequences[parent],
                    "reference_ids_json": _json(sorted(parent_refs[parent])),
                    "roles_json": _json(sorted(parent_roles[parent])),
                }
            )
        for key in sorted(precursor_cache):
            sinks["precursor_inputs"].add(precursor_cache[key])
    finally:
        for sink in sinks.values():
            sink.close()
    summary = {
        "n_source_quantity_cells": n_cells,
        "n_source_peptide_sequences": len(peptide_rows),
        "n_background_sequences": len(background),
        "n_response_peptides": len(used_peptides),
        "n_response_parents": len(used_parents),
        "n_response_components": len({components["peptide:" + p][0] for p in used_peptides}),
        "n_rows_by_relation": {name: sink.n_rows for name, sink in sinks.items()},
        "responses_by_split": dict(sorted(response_splits.items())),
        "complete_precursor_responses_by_split": dict(sorted(complete_splits.items())),
        "responses_by_condition": dict(sorted(response_conditions.items())),
        "dispositions": dict(sorted(reasons.items())),
        "absolute_concentration_claim": False,
        "specimen_independence": "unresolved",
    }
    if sum(reasons.values()) != n_cells or sinks["responses"].n_rows != sum(
        response_splits.values()
    ):
        raise ValueError("HAP1 source conservation failed")
    _write(destination / "summary.json", summary)
    return summary


def materialize_responses(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if store.manifest.get("revision") != REVISION:
        raise RecipeError("HAP1 responses require the quantitative source revision")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in TABLES:
            relative = f"hap1/typed/{name}.parquet"
            if file_digest(store.directory / relative) != store.manifest["files"][relative]:
                raise RecipeError(f"HAP1 source relation changed: {name}")
            snapshot_file(store.directory / relative, staging / "source" / f"{name}.parquet")
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        snapshot_file(store.directory / "hap1/contract.json", staging / "source_contract.json")
        _write(staging / "contract.json", CONTRACT)
        summary = compile_responses(staging / "source", staging / "derived")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_hap1_response",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": _hash(recipe),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": summary,
            "files": {
                str(p.relative_to(staging)): file_digest(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        }
        _write(staging / "manifest.json", manifest)
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_response_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    directory = Path(directory)
    _validate(manifest["recipe"])
    expected = (
        {f"source/{name}.parquet" for name in TABLES}
        | {f"derived/{name}.parquet" for name in SCHEMAS}
        | {"contract.json", "source_manifest.json", "source_contract.json", "derived/summary.json"}
    )
    if set(manifest["files"]) != expected or _read(directory / "contract.json") != CONTRACT:
        raise RecipeError("HAP1 response contract or inventory changed")
    source_manifest = _read(directory / "source_manifest.json")
    if (
        source_manifest.get("revision") != REVISION
        or file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source_manifest["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("HAP1 response source binding changed")
    for name in TABLES:
        relative = f"hap1/typed/{name}.parquet"
        digest = source_manifest["files"][relative]
        if file_digest(directory / "source" / f"{name}.parquet") != digest or (
            source_package is not None and file_digest(Path(source_package) / relative) != digest
        ):
            raise RecipeError(f"HAP1 response source relation changed: {name}")
    if (
        file_digest(directory / "source_contract.json")
        != source_manifest["files"]["hap1/contract.json"]
    ):
        raise RecipeError("HAP1 response source contract changed")
    with tempfile.TemporaryDirectory(prefix="hitlist-hap1-verify-") as tmp:
        tmp = Path(tmp)
        summary = compile_responses(directory / "source", tmp)
        if summary != manifest["coverage"] or summary != _read(directory / "derived/summary.json"):
            raise RecipeError("HAP1 response summary changed")
        for name in SCHEMAS:
            left, right = (
                pq.ParquetFile(tmp / f"{name}.parquet"),
                pq.ParquetFile(directory / "derived" / f"{name}.parquet"),
            )
            if (
                left.metadata.num_rows != right.metadata.num_rows
                or left.schema_arrow != right.schema_arrow
            ):
                raise RecipeError(f"HAP1 response relation schema/count changed: {name}")
            for a, b in zip(
                left.iter_batches(batch_size=8192), right.iter_batches(batch_size=8192)
            ):
                if not a.equals(b):
                    raise RecipeError(f"HAP1 response derived rows changed: {name}")
    return manifest
