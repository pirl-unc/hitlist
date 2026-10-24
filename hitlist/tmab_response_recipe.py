"""Source-preserving native peptide responses with explicit model references."""

from __future__ import annotations

import gzip
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
from .provenance import file_digest
from .sequence_reference_evidence import exact_occurrences, read_reference
from .tmab_peptidomics import _PRIMARY, _sequence_identity

RECIPE_ID = "tmab_native_response_v1"
TABLES = {
    "source_rows": ("tmab_source_row", "tmab/source_rows.parquet"),
    "measurements": ("tmab_measurement", "tmab/measurements.parquet"),
    "contexts": ("tmab_protocol_context", "tmab/protocol_context.parquet"),
    "lineage": ("tmab_measurement_lineage", "tmab/lineage.parquet"),
}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "endpoint": "native_extract_treated_control_isotope_channel_ratio",
    "target": "log10_positive_ratio_or_explicit_bound",
    "admission": "corrected_candidate_exact_or_censored;defer_conflicting_possible_reuse",
    "response_group": "peptide_sequence_and_modifications_plus_literal_and_typed_condition",
    "observation": "one_lineage_slot_with_all_source_contributors",
    "weights": "within_split_average_condition_then_component_then_form_then_observation",
    "split": "all_source_sequence_parent_assertion_reference_and_reuse_components;sha256(recipe_id+component_id)_mod20:0-13_train,14-16_val,17-19_test",
    "reference": "pinned_human_reviewed_model_FASTA;all_exact_occurrences;not_original_search_reference",
    "limitations": [
        "relative_native_extract_response_not_absolute_or_APC_concentration",
        "chemical_exposure_not_genetic_loss_or_measured_residual_activity",
        "source_reported_identity_not_new_FDR_validation",
        "possible_reuse_not_adjudicated_specimen_identity",
        "charge_and_ratio_slots_not_independent_biological_replicates",
        "shared_control_label_proxies_can_cross_parent_splits;physical_denominators_unknown",
        "same_studies_in_all_splits;no_external_specimen_or_homology_independence",
        "all_reference_origins_are_candidates;no_precursor_abundance_assignment",
        "same_form_MS_response_cancellation_is_a_model_assumption",
    ],
}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normal(value):
    return " ".join(str(value or "").lower().split())


def _text(value):
    return None if value is None else str(value).strip() or None


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"schema_version", "name", "registered_recipe", "reference"}
        or recipe["schema_version"] != 1
        or recipe["registered_recipe"] != RECIPE_ID
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
    ):
        raise RecipeError("Invalid native response recipe")
    reference = recipe["reference"]
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256", "source_url", "generation_label", "license", "role"}
        or any(not isinstance(v, str) or not v.strip() for v in reference.values())
        or re.fullmatch("[0-9a-f]{64}", reference["sha256"]) is None
        or reference["role"] != "model_reference_not_original_search"
    ):
        raise RecipeError("A pinned, explicitly scoped native model reference is required")


def describe(store):
    return {
        "contract": CONTRACT,
        "status": "available_with_explicit_reference"
        if {name for name, _ in TABLES.values()} <= store.bindings.keys()
        else "not_in_generation",
    }


def _load_tables(store):
    from .evidence_recipe import RecipeError

    if store.manifest.get("revision") != "tmab_native_peptidomics_v3":
        raise RecipeError("Native responses require corrected TMAB v3 sources")
    return {
        table: [
            {key.removeprefix("native."): value for key, value in row.items()}
            for row in store.query(
                name, select=["native." + field for field in store.schemas[name].names], limit=None
            ).table.to_pylist()
        ]
        for table, (name, _) in TABLES.items()
    }


def _proteins(path):
    records = list(read_reference(path, "uniprot_fasta"))
    if not records or any(p["metadata"]["taxonomy"] != "9606" for p in records):
        raise ValueError("Native human reference must contain only explicitly human records")
    if len({p["accession"] for p in records}) != len(records):
        raise ValueError("Duplicate reference accession")
    return {p["accession"]: p for p in records}


def _condition(context):
    # No inferred compound alias, dose equivalence, specimen or original instrument.
    return [
        _normal(context["cell_line_name"]),
        _normal(context["treatment_label_raw"]),
        context["treatment_name"],
        context["treatment_dose"],
        context["treatment_dose_unit"],
        context["duration_hours"],
    ]


def compile_responses(tables, proteins):
    """Pure shared compilation; retains quantity and reference alternatives."""
    rows = {r["source_record_id"]: r for r in tables["source_rows"]}
    contexts = {c["context_id"]: c for c in tables["contexts"]}
    measurements = {m["measurement_id"]: m for m in tables["measurements"]}
    lineage = {r["measurement_id"]: r for r in tables["lineage"]}
    if (
        len(rows) != len(tables["source_rows"])
        or len(contexts) != len(tables["contexts"])
        or len(measurements) != len(tables["measurements"])
        or len(lineage) != len(tables["lineage"])
        or measurements.keys() != lineage.keys()
    ):
        raise ValueError("Duplicate identities or incomplete native lineage")
    per_row = defaultdict(list)
    for m in measurements.values():
        if m["source_record_id"] not in rows or m["context_id"] not in contexts:
            raise ValueError("Broken native row/context reference")
        per_row[m["source_record_id"]].append(m)
    assertions = []
    for key, row in sorted(rows.items()):
        values = json.loads(row["source_row_json"])
        primary = row["source_row_disposition"] == "primary_ratio_panel"
        members = per_row[key]
        if primary and not members:
            raise ValueError("Primary native row has no measurement slots")
        if primary:
            m = members[0]
            sequence = m["peptide_sequence"]
            identity = m["sequence_identity_status"]
            config = _PRIMARY[row["source_file"]]
            n_flank = _text(values[config["upstream"] - 1])
            c_flank = _text(values[config["downstream"] - 1])
            parent, gene = m["parent_label_raw"], m["gene_name"]
            if any(
                x["peptide_sequence"] != sequence or x["context_id"] != m["context_id"]
                for x in members
            ):
                raise ValueError("Inconsistent native row projections")
        else:
            parsed = _sequence_identity(row["source_sequence_raw"])
            sequence, identity = parsed["sequence"], parsed["identification"]
            n_flank = c_flank = parent = gene = None
        assertions.append(
            {
                "source_record_id": key,
                "source_id": row["source_id"],
                "source_file": row["source_file"],
                "source_row_index": row["source_row_index"],
                "source_row_disposition": row["source_row_disposition"],
                "peptide_sequence": sequence,
                "sequence_identity_status": identity,
                "parent_description_raw": parent,
                "gene_name_raw": gene,
                "n_flank_raw": n_flank,
                "c_flank_raw": c_flank,
                "origin_status": "reference_candidates_not_measured_origin",
            }
        )
    sequences = {
        a["peptide_sequence"]: a["peptide_sequence"] for a in assertions if a["peptide_sequence"]
    }
    found = exact_occurrences(proteins, sequences) if sequences else {}
    # Components are built from all source identities, before eligibility decisions.
    roots = {key: key for key in rows}

    def root(key):
        while roots[key] != key:
            roots[key] = roots[roots[key]]
            key = roots[key]
        return key

    owners = {}

    def link(key, claim):
        if claim in owners:
            a, b = root(key), root(owners[claim])
            roots[max(a, b)] = min(a, b)
        else:
            owners[claim] = key

    for a in assertions:
        key, seq = a["source_record_id"], a["peptide_sequence"]
        if seq:
            link(key, ("sequence", seq))
            for accession, _ in found.get(seq, []):
                link(key, ("reference_sequence", proteins[accession]["sequence_sha256"]))
            for field in ("gene_name_raw", "parent_description_raw"):
                value = _normal(a[field])
                if value and value not in {"unknown", "unidentified", "n/a"}:
                    link(key, (field, value))
    reuse_members = defaultdict(list)
    for line in lineage.values():
        candidate = line["candidate_reuse_group_id"]
        if candidate:
            link(line["source_record_id"], ("possible_reuse", candidate))
            reuse_members[candidate].append(line)
    grouped = defaultdict(list)
    for key in sorted(rows):
        grouped[root(key)].append(key)
    component_for = {}
    for members in grouped.values():
        component = _hash(members)
        bucket = int(hashlib.sha256((RECIPE_ID + component).encode()).hexdigest(), 16) % 20
        split = "train" if bucket < 14 else "val" if bucket < 17 else "test"
        for key in members:
            component_for[key] = {"component_id": component, "split": split}
    components = [{"source_record_id": key, **component_for[key]} for key in sorted(rows)]
    conflicting_reuse = set()
    for candidate, members in reuse_members.items():
        signatures = {
            _hash(_condition(contexts[measurements[r["measurement_id"]]["context_id"]]))
            for r in members
        }
        if len(signatures) != 1 or any(
            r["experiment_label_agreement"] != "equal_after_case_whitespace_normalization"
            for r in members
        ):
            conflicting_reuse.add(candidate)

    dispositions = []
    units = defaultdict(list)
    for identity, m in sorted(measurements.items()):
        reason = None
        if m["eligibility_status"] not in {"candidate_exact", "candidate_censored"}:
            reason = m["eligibility_status"]
        elif lineage[identity]["candidate_reuse_group_id"] in conflicting_reuse:
            reason = "deferred_conflicting_possible_reuse_context"
        decision = {
            "measurement_id": identity,
            "source_record_id": m["source_record_id"],
            **component_for[m["source_record_id"]],
            "status": "deferred" if reason else "admitted",
            "reason": reason,
            "quantitative_raw": m["quantitative_raw"],
            "quantitative_value": m["quantitative_value"],
            "measurement_inequality": m["measurement_inequality"],
            "censoring_status": m["censoring_status"],
            "response_group_id": None,
        }
        dispositions.append(decision)
        if reason is None:
            value = m["quantitative_value"]
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid source candidate value")
            if (
                m["eligibility_status"] == "candidate_exact"
                and (m["measurement_inequality"] is not None or m["censoring_status"] != "exact")
            ) or (
                m["eligibility_status"] == "candidate_censored"
                and m["measurement_inequality"] not in {"lt", "gt"}
            ):
                raise ValueError("Candidate target/qualifier disagreement")
            units[m["lineage_group_id"]].append(m)
    response_groups = {}
    measurement_group = {}
    for unit, members in sorted(units.items()):
        signatures = {
            _hash(
                [
                    m["peptide_sequence"],
                    m["peptide_modifications_json"],
                    _condition(contexts[m["context_id"]]),
                    m["quantitative_value"],
                    m["measurement_inequality"],
                    m["censoring_status"],
                ]
            )
            for m in members
        }
        if len(signatures) != 1:
            raise ValueError(
                "Possible reused observations disagree; explicit adjudication required"
            )
        m = members[0]
        seq = m["peptide_sequence"]
        condition = _condition(contexts[m["context_id"]])
        condition_id = _hash(condition)
        form = [seq, json.loads(m["peptide_modifications_json"])]
        group_id = _hash([form, condition])
        origins = found.get(seq, [])
        response_groups.setdefault(
            group_id,
            {
                "response_group_id": group_id,
                "condition_id": condition_id,
                "condition": condition,
                **component_for[m["source_record_id"]],
                "peptide_sequence": seq,
                "modifications": form[1],
                "n_reference_occurrences": len(origins),
                "n_reference_sequences": len({proteins[a]["sequence_sha256"] for a, _ in origins}),
                "full_parent_model_supported": bool(origins)
                and all(not set(proteins[a]["sequence"]) - AA for a, _ in origins),
                "observations": [],
                "context_ids": set(),
                "shared_control_proxy_ids": set(),
            },
        )
        group = response_groups[group_id]
        for x in members:
            if component_for[x["source_record_id"]] != {
                "component_id": group["component_id"],
                "split": group["split"],
            }:
                raise ValueError("A response group crosses split components")
            measurement_group[x["measurement_id"]] = group_id
            group["context_ids"].add(x["context_id"])
            group["shared_control_proxy_ids"].add(x["shared_control_group_id"])
        group["observations"].append(
            {
                "observation_id": unit,
                "target_log10_ratio": math.log10(m["quantitative_value"]),
                "quantitative_value": m["quantitative_value"],
                "inequality": m["measurement_inequality"],
                "censoring_status": m["censoring_status"],
                "measurement_ids": sorted(x["measurement_id"] for x in members),
                "source_record_ids": sorted({x["source_record_id"] for x in members}),
                "charge_labels_raw": sorted({x["charge_raw"] for x in members if x["charge_raw"]}),
                "replicate_labels": sorted({x["replicate_label"] for x in members}),
                "n_source_slots": len(members),
                "independent_specimen_status": "unknown",
            }
        )
    for d in dispositions:
        d["response_group_id"] = measurement_group.get(d["measurement_id"])
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for group in response_groups.values():
        hierarchy[group["split"]][group["condition_id"]][group["component_id"]].append(group)
    for conditions in hierarchy.values():
        for parent_groups in conditions.values():
            for forms in parent_groups.values():
                group_weight = 1 / len(conditions) / len(parent_groups) / len(forms)
                for group in forms:
                    group["group_weight"] = group_weight
                    for observation in group["observations"]:
                        observation["weight"] = group_weight / len(group["observations"])
    examples = []
    for _, group in sorted(response_groups.items()):
        group["context_ids"] = sorted(group["context_ids"])
        group["shared_control_proxy_ids"] = sorted(group["shared_control_proxy_ids"])
        group["observations"].sort(key=lambda r: r["observation_id"])
        examples.append(group)
    occurrences = []
    used_parents = set()
    for sequence in sorted(sequences):
        for accession, start in sorted(found.get(sequence, [])):
            protein = proteins[accession]
            end = start + len(sequence) - 1
            used_parents.add(accession)
            occurrences.append(
                {
                    "peptide_sequence": sequence,
                    "reference_accession": accession,
                    "reference_sequence_sha256": protein["sequence_sha256"],
                    "start_1based": start,
                    "end_1based": end,
                    "n_flank": protein["sequence"][start - 2] if start > 1 else "-",
                    "c_flank": protein["sequence"][end] if end < len(protein["sequence"]) else "-",
                    "origin_status": "candidate_not_measured_origin",
                }
            )
    parents = [proteins[accession] for accession in sorted(used_parents)]
    proxy_splits = defaultdict(set)
    for group in examples:
        for proxy in group["shared_control_proxy_ids"]:
            proxy_splits[proxy].add(group["split"])
    summary = {
        "n_source_rows": len(rows),
        "n_source_measurement_slots": len(measurements),
        "n_admitted_source_slots": len(measurement_group),
        "n_deferred_source_slots": len(dispositions) - len(measurement_group),
        "n_effective_observations": sum(len(g["observations"]) for g in examples),
        "n_response_groups": len(examples),
        "n_components": len(grouped),
        "n_reference_proteins": len(proteins),
        "n_used_reference_proteins": len(parents),
        "n_source_sequences": len(sequences),
        "n_reference_occurrences": len(occurrences),
        "n_conflicting_reuse_groups": len(conflicting_reuse),
        "n_control_proxies_crossing_splits": sum(len(s) > 1 for s in proxy_splits.values()),
        "n_full_parent_supported_groups": sum(g["full_parent_model_supported"] for g in examples),
        "groups_by_split": dict(sorted(Counter(g["split"] for g in examples).items())),
        "observations_by_censoring_status": dict(
            sorted(
                Counter(o["censoring_status"] for g in examples for o in g["observations"]).items()
            )
        ),
        "defer_reason_counts": dict(
            sorted(Counter(d["reason"] for d in dispositions if d["reason"]).items())
        ),
        "specimen_independence": "unresolved",
        "absolute_concentration_claim": False,
    }
    return {
        "assertions": assertions,
        "components": components,
        "examples": examples,
        "dispositions": dispositions,
        "parents": parents,
        "occurrences": occurrences,
        "summary": summary,
    }


def _write(path, value):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def materialize_responses(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    reference = Path(recipe["reference"]["path"])
    if file_digest(reference)["sha256"] != recipe["reference"]["sha256"]:
        raise RecipeError("Native model reference hash changed")
    tables = _load_tables(store)
    compiled = compile_responses(tables, _proteins(reference))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(reference, staging / "model_reference.fasta")
        shutil.copyfile(store.directory / "manifest.json", staging / "source_manifest.json")
        _write(staging / "contract.json", CONTRACT)
        _write(staging / "source_tables.json.gz", tables)
        for name, value in compiled.items():
            _write(staging / f"{name}.json.gz", value)
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_native_response",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": _hash(recipe),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": compiled["summary"],
            "files": {p.name: file_digest(p) for p in sorted(staging.iterdir())},
        }
        _write(staging / "manifest.json", manifest)
        del tables, compiled
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
    expected = {
        "assertions",
        "components",
        "examples",
        "dispositions",
        "parents",
        "occurrences",
        "summary",
    }
    inventory = {f"{name}.json.gz" for name in expected} | {
        "source_tables.json.gz",
        "contract.json",
        "source_manifest.json",
        "model_reference.fasta",
    }
    if set(manifest["files"]) != inventory or _read(directory / "contract.json") != CONTRACT:
        raise RecipeError("Native response inventory or contract changed")
    if (
        file_digest(directory / "model_reference.fasta")["sha256"]
        != manifest["recipe"]["reference"]["sha256"]
    ):
        raise RecipeError("Native model reference binding changed")
    source_manifest = _read(directory / "source_manifest.json")
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source_manifest.get("revision") != "tmab_native_peptidomics_v3"
        or source_manifest.get("schema_version") != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Native response source binding changed")
    tables = _read(directory / "source_tables.json.gz")
    if set(tables) != set(TABLES):
        raise RecipeError("Native source table inventory changed")
    if source_package is not None:
        for table, (_, relative) in TABLES.items():
            path = Path(source_package) / relative
            if (
                file_digest(path) != source_manifest["files"][relative]
                or pq.read_table(path).to_pylist() != tables[table]
            ):
                raise RecipeError(f"Native response source rows changed: {table}")
    compiled = compile_responses(tables, _proteins(directory / "model_reference.fasta"))
    for name, value in compiled.items():
        if _read(directory / f"{name}.json.gz") != value:
            raise RecipeError(f"Native response derived contents changed: {name}")
    if manifest["coverage"] != compiled["summary"]:
        raise RecipeError("Native response coverage changed")
    return manifest
