"""Source-linked contiguous time courses, conditional on reported positive signal."""

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
from .evidence_snapshot import snapshot_file
from .provenance import file_digest

RECIPE_ID = "digestion_contiguous_timecourses_v1"
MS1_UNIT = "source_MS1_peak_area_arbitrary_units"
QUANTITY_SCALE = "aSPIRE_background_adjusted_biological_mean"
TABLE_COLUMNS = {
    "constructs": ("construct_id", "sequence", "sequence_sha256"),
    "contexts": (
        "condition_id",
        "construct_id",
        "training_status",
        "fields_json",
        "source_locator",
        "enzyme_species",
        "enzyme_complex",
        "enzyme_preparation_cell_line",
        "substrate_species",
        "context_scope",
        "buffer_json",
        "measurement_population",
        "termination",
        "temperature_c",
        "substrate_concentration_um",
        "enzyme_substrate_denominator",
    ),
    "raw_quantities": (
        "source_record_id",
        "condition_id",
        "peptide_sequence",
        "biological_replicate_raw",
        "run_id",
        "quantity_time_hours",
        "metadata_time_hours",
        "time_status",
        "intensity",
    ),
    "filtered_vectors": (
        "source_record_id",
        "condition_id",
        "peptide_sequence",
        "biological_replicate_raw",
        "technical_replicate_raw",
        "times_raw",
    ),
    "final_vectors": (
        "source_record_id",
        "condition_id",
        "construct_id",
        "peptide_sequence",
        "biological_replicate_raw",
        "times_raw",
        "intensities_raw",
        "product_type_raw",
        "splice_type_raw",
        "positions_raw",
        "row_status",
        "fields_json",
    ),
}
TABLE_BINDINGS = {table: "digestion_" + table for table in TABLE_COLUMNS}
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "source_id": "soh2024_digestion",
    "endpoint": "source_selected_contiguous_corrected_positive_MS_timecourse",
    "target": "original_positive_background_adjusted_biological_mean_MS1_peak_area;no_normalization",
    "measurement_unit": MS1_UNIT,
    "quantity_scale": QUANTITY_SCALE,
    "mapping": "exactly_one_IL_equivalent_span_in_literal_construct;canonical20;source_PCP_without_splice_alternative",
    "baseline": "actual_filtered_zero_in_every_source_technical_label_and_raw_metadata_agreeing_zero;no_mock_zero",
    "duplicates": "IL_alias_and_repeated_slot_collapse_only_if_values_agree_relative1e-10;otherwise_exclude_curve",
    "replication": "at_least_two_source_biological_labels_share_at_least_three_positive_nonzero_times",
    "split": "all_source_final_peptide_IL_components_and_identical_constructs;sorted_component_hash_round_robin5;fold0_test,1_val,2to4_train",
    "zero_missing_policy": "retain_source_slots_and_dispositions;never_negative_or_imputed_target",
    "limitations": [
        "source_search_and_synthesis_filter_not_independently_validated",
        "observed_positive_selection_not_prospective_detection",
        "MS_response_not_molar_concentration",
        "technical_indices_not_assured_acquisition_pairs",
        "source_biological_labels_not_independent_studies",
        "exact_product_groups_not_homology_clusters",
        "no_APC_transport_clearance_or_machinery_supervision",
    ],
}


def _identity(*parts):
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _canonical(sequence):
    return sequence.replace("I", "L")


def _value(text):
    if text in ("", "NA", "NaN", "#N/A"):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def components(tables):
    parents = {r["construct_id"]: r["construct_id"] for r in tables["constructs"]}

    def root(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owners = {}
    for row in tables["constructs"]:
        owner = owners.setdefault(("parent", row["sequence"]), row["construct_id"])
        a, b = root(owner), root(row["construct_id"])
        parents[max(a, b)] = min(a, b)
    for row in tables["final_vectors"]:
        owner = owners.setdefault(
            ("peptide", _canonical(row["peptide_sequence"])), row["construct_id"]
        )
        a, b = root(owner), root(row["construct_id"])
        parents[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for key in sorted(parents):
        groups[root(key)].append(key)
    ids = {_identity(*members): members for members in groups.values()}
    return [
        {
            "construct_id": cid,
            "component_id": group,
            "fold": index % 5,
            "split": "test" if index % 5 == 0 else "val" if index % 5 == 1 else "train",
        }
        for index, (group, members) in enumerate(sorted(ids.items()))
        for cid in members
    ]


def compile_timecourses(tables):
    assignments = components(tables)
    by_construct = {r["construct_id"]: r for r in assignments}
    constructs = {r["construct_id"]: r for r in tables["constructs"]}
    contexts = {r["condition_id"]: r for r in tables["contexts"]}
    raw_zero = defaultdict(list)
    bad_conditions = {
        r["condition_id"]
        for r in tables["contexts"]
        if r["training_status"] == "unresolved_source_time_conflict"
    }
    for r in tables["raw_quantities"]:
        if r["time_status"] != "source_agreement":
            bad_conditions.add(r["condition_id"])
        if r["quantity_time_hours"] == r["metadata_time_hours"] == 0 and r["intensity"] is not None:
            raw_zero[
                r["condition_id"], _canonical(r["peptide_sequence"]), r["biological_replicate_raw"]
            ].append(r["source_record_id"])
    technical, zero_technical, filtered_ids = defaultdict(set), defaultdict(set), defaultdict(list)
    for r in tables["filtered_vectors"]:
        key = (r["condition_id"], _canonical(r["peptide_sequence"]), r["biological_replicate_raw"])
        label = r["technical_replicate_raw"]
        technical[key].add(label)
        if any(float(t) == 0 for t in r["times_raw"].split(";")):
            zero_technical[key].add(label)
        filtered_ids[key].append(r["source_record_id"])
    grouped = defaultdict(list)
    for r in tables["final_vectors"]:
        grouped[
            r["condition_id"], _canonical(r["peptide_sequence"]), r["biological_replicate_raw"]
        ].append(r)
    dispositions, candidates, maps = [], {}, {}
    for key, rows in sorted(grouped.items()):
        condition, peptide, bio = key
        cid = contexts[condition]["construct_id"]
        assert all(r["construct_id"] == cid for r in rows)
        sequence = constructs[cid]["sequence"]
        errors = []
        if not peptide or set(peptide + sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            errors.append("unsupported_sequence")
        if any(r["row_status"] != "reported_vector" for r in rows):
            errors.append("annotation_only")
        if any(
            r["product_type_raw"] != "PCP" or r["splice_type_raw"] not in (None, "", "NA", "PCP")
            for r in rows
        ):
            errors.append("not_exclusively_source_contiguous")
        if (cid, peptide) not in maps:
            parent = _canonical(sequence)
            maps[cid, peptide] = (
                [i for i in range(len(parent) - len(peptide) + 1) if parent.startswith(peptide, i)]
                if peptide
                else []
            )
        positions = maps[cid, peptide]
        if len(positions) != 1:
            errors.append("not_one_IL_equivalent_contiguous_span")
        if condition in bad_conditions:
            errors.append("unresolved_condition_time_conflict")
        if not technical[key] or technical[key] != zero_technical[key] or not raw_zero[key]:
            errors.append("no_actual_zero_baseline_for_every_technical_label")
        values = defaultdict(list)
        for row in rows:
            if row["row_status"] == "annotation_only":
                continue
            for slot, (t, v) in enumerate(
                zip(row["times_raw"].split(";"), row["intensities_raw"].split(";"), strict=True)
            ):
                time_hours, value = float(t), _value(v)
                if time_hours == 0 and value == 0:
                    value_status = "postprocessing_forced_zero"
                elif value is None:
                    value_status = "postprocessing_missing_unresolved_contributors"
                elif value > 0:
                    value_status = "source_positive"
                elif value == 0:
                    value_status = "source_zero"
                else:
                    value_status = "source_negative"
                values[time_hours].append(
                    {
                        "value": value,
                        "value_raw": v,
                        "value_status": value_status,
                        "vector_id": row["source_record_id"],
                        "slot_0based": slot,
                    }
                )
        observations = []
        source_slots = []
        for t, slots in sorted(values.items()):
            representative = slots[0]["value"]
            consistent = all(
                (s["value"] is None and representative is None)
                or (
                    s["value"] is not None
                    and representative is not None
                    and math.isclose(s["value"], representative, rel_tol=1e-10, abs_tol=1e-10)
                )
                for s in slots
            )
            if not consistent:
                errors.append("conflicting_IL_alias_or_repeated_slot_values")
            source_slots.append({"time_hours": t, "contributors": slots, "consistent": consistent})
            if consistent and t > 0 and representative is not None and representative > 0:
                observations.append(
                    {"time_hours": t, "target_value": representative, "source_slots": slots}
                )
        if len(observations) < 3:
            errors.append("fewer_than_three_positive_nonzero_times")
        curve_id = _identity(*key)
        record = {
            "curve_id": curve_id,
            "condition_id": condition,
            "construct_id": cid,
            "peptide_IL_sequence": peptide,
            "peptide_aliases": sorted({r["peptide_sequence"] for r in rows}),
            "biological_replicate_raw": bio,
            "source_vector_ids": sorted(r["source_record_id"] for r in rows),
            "candidate_start_positions_0based": positions,
            "source_slots": source_slots,
            **by_construct[cid],
            "reasons": sorted(set(errors)),
        }
        dispositions.append(record)
        if not errors:
            candidates[key] = {
                k: v for k, v in record.items() if k not in ("reasons", "source_slots")
            }
            candidates[key].update(
                {
                    "parent_sequence": sequence,
                    "start_0based": positions[0],
                    "end_0based": positions[0] + len(peptide),
                    "peptide_sequence": sequence[positions[0] : positions[0] + len(peptide)],
                    "observations": observations,
                    "target_unit": MS1_UNIT,
                    "target_quantity_scale": QUANTITY_SCALE,
                    "context": contexts[condition],
                    "actual_zero_raw_quantity_ids": sorted(raw_zero[key]),
                    "filtered_source_vector_ids": sorted(filtered_ids[key]),
                }
            )
    pairs = defaultdict(list)
    for key, candidate in candidates.items():
        pairs[key[:2]].append(candidate)
    selected = {}
    for curves in pairs.values():
        common = set.intersection(*({v["time_hours"] for v in c["observations"]} for c in curves))
        if len(curves) >= 2 and len(common) >= 3:
            for curve in curves:
                curve["common_positive_times"] = sorted(common)
                selected[curve["curve_id"]] = curve
    for record in dispositions:
        if not record["reasons"] and record["curve_id"] not in selected:
            record["reasons"].append("insufficient_repeated_common_positive_times")
        record["status"] = "excluded" if record["reasons"] else "admitted"
    examples = [selected[k] for k in sorted(selected)]
    return {
        "components": assignments,
        "examples": examples,
        "dispositions": dispositions,
        "summary": {
            "n_source_final_vectors": len(tables["final_vectors"]),
            "n_curve_candidates": len(dispositions),
            "n_examples": len(examples),
            "n_components": len({r["component_id"] for r in assignments}),
            "n_constructs": len(constructs),
            "reason_counts": dict(
                sorted(Counter(reason for r in dispositions for reason in r["reasons"]).items())
            ),
            "curves_by_split": dict(sorted(Counter(r["split"] for r in examples).items())),
            "curves_by_condition": dict(
                sorted(Counter(r["condition_id"] for r in examples).items())
            ),
        },
    }


def describe(store):
    return {
        "contract": CONTRACT,
        "status": "materialized"
        if set(TABLE_BINDINGS.values()) <= store.bindings.keys()
        else "not_in_generation",
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
        raise RecipeError("Invalid registered digestion recipe")


def materialize_timecourses(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not set(TABLE_BINDINGS.values()) <= store.bindings.keys():
        raise RecipeError("Digestion source is not in this generation")
    tables = {
        table: [
            {k.removeprefix("native."): v for k, v in row.items()}
            for row in store.query(
                binding, select=["native." + c for c in TABLE_COLUMNS[table]], limit=None
            ).table.to_pylist()
        ]
        for table, binding in TABLE_BINDINGS.items()
    }
    compiled = compile_timecourses(tables)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        for table in TABLE_BINDINGS:
            snapshot_file(
                store.directory / "external/soh2024_digestion/typed" / f"{table}.parquet",
                staging / "source_tables" / f"{table}.parquet",
            )
        for name, value in {"recipe": CONTRACT, **compiled}.items():
            (staging / f"{name}.json").write_text(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_digestion_timecourses",
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
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        del tables, compiled
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_timecourse_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    directory = Path(directory)
    _validate(manifest["recipe"])
    expected = {
        f"{n}.json"
        for n in ("recipe", "components", "examples", "dispositions", "summary", "source_manifest")
    } | {f"source_tables/{t}.parquet" for t in TABLE_BINDINGS}
    if set(manifest["files"]) != expected:
        raise RecipeError("Digestion recipe inventory changed")
    if json.loads((directory / "recipe.json").read_text()) != CONTRACT:
        raise RecipeError("Digestion recipe contract changed")
    source = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Digestion recipe source binding changed")
    for table in TABLE_BINDINGS:
        original = f"external/soh2024_digestion/typed/{table}.parquet"
        digest = file_digest(directory / "source_tables" / f"{table}.parquet")
        if digest != source["files"][original] or (
            source_package is not None and digest != file_digest(Path(source_package) / original)
        ):
            raise RecipeError(f"Digestion recipe source rows changed: {table}")
    tables = {
        t: pq.read_table(
            directory / "source_tables" / f"{t}.parquet", columns=list(columns)
        ).to_pylist()
        for t, columns in TABLE_COLUMNS.items()
    }
    compiled = compile_timecourses(tables)
    for name, value in compiled.items():
        if json.loads((directory / f"{name}.json").read_text()) != value:
            raise RecipeError(f"Digestion recipe derived contents changed: {name}")
    if manifest["coverage"] != compiled["summary"]:
        raise RecipeError("Digestion recipe coverage changed")
    return manifest
