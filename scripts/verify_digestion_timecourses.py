"""Independently replay every digestion curve decision from portable source tables.

This verifier deliberately imports no Hitlist module. The source Parquet files
must first pass the original-asset round-trip in verify_digestion_roundtrip.py.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

UNIT = "source_MS1_peak_area_arbitrary_units"
SCALE = "aSPIRE_background_adjusted_biological_mean"
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
CONTEXT_COLUMNS = (
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
)


def records(path):
    """Stream a pretty-printed JSON array without retaining the whole export."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        if stream.read(1) != "[":
            raise ValueError(f"Expected JSON array: {path}")
        buffer = ""
        while True:
            if len(buffer) < 65536:
                buffer += stream.read(65536)
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                if (buffer[1:] + stream.read()).strip():
                    raise ValueError(f"Trailing JSON data: {path}")
                return
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    more = stream.read(65536)
                    if not more:
                        raise ValueError(f"Incomplete JSON record: {path}") from None
                    buffer += more
            yield row
            buffer = buffer[end:]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return {"sha256": value.hexdigest(), "size_bytes": Path(path).stat().st_size}


def canonical(sequence):
    return sequence.replace("I", "L")


def identity(*parts):
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def value(raw):
    if raw in ("", "NA", "NaN", "#N/A"):
        return None
    result = float(raw)
    return result if math.isfinite(result) else None


def slots(rows):
    by_time = defaultdict(list)
    for row in rows:
        if row["row_status"] == "annotation_only":
            continue
        for index, (raw_time, raw_value) in enumerate(
            zip(row["times_raw"].split(";"), row["intensities_raw"].split(";"), strict=True)
        ):
            time_hours, intensity = float(raw_time), value(raw_value)
            if time_hours == 0 and intensity == 0:
                status = "postprocessing_forced_zero"
            elif intensity is None:
                status = "postprocessing_missing_unresolved_contributors"
            elif intensity > 0:
                status = "source_positive"
            elif intensity == 0:
                status = "source_zero"
            else:
                status = "source_negative"
            by_time[time_hours].append(
                {
                    "value": intensity,
                    "value_raw": raw_value,
                    "value_status": status,
                    "vector_id": row["source_record_id"],
                    "slot_0based": index,
                }
            )
    output, observed = [], []
    for time_hours, contributors in sorted(by_time.items()):
        first = contributors[0]["value"]
        consistent = all(
            (entry["value"] is None and first is None)
            or (
                entry["value"] is not None
                and first is not None
                and math.isclose(entry["value"], first, rel_tol=1e-10, abs_tol=1e-10)
            )
            for entry in contributors
        )
        output.append(
            {"time_hours": time_hours, "contributors": contributors, "consistent": consistent}
        )
        if consistent and time_hours > 0 and first is not None and first > 0:
            observed.append(
                {"time_hours": time_hours, "target_value": first, "source_slots": contributors}
            )
    return output, observed


def components(constructs, final_rows):
    parent = {row["construct_id"]: row["construct_id"] for row in constructs}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    owners = {}
    for row in constructs:
        cid = row["construct_id"]
        key = ("construct", row["sequence"])
        if key in owners:
            union(cid, owners[key])
        owners[key] = cid
    for row in final_rows:
        cid = row["construct_id"]
        key = ("peptide", canonical(row["peptide_sequence"]))
        if key in owners:
            union(cid, owners[key])
        owners[key] = cid
    groups = defaultdict(list)
    for cid in sorted(parent):
        groups[root(cid)].append(cid)
    hashed = sorted((identity(*members), members) for members in groups.values())
    output = {}
    for index, (component_id, members) in enumerate(hashed):
        fold = index % 5
        split = "test" if fold == 0 else "val" if fold == 1 else "train"
        for cid in members:
            output[cid] = {
                "construct_id": cid,
                "component_id": component_id,
                "fold": fold,
                "split": split,
            }
    return output


def replay(package, recipe):
    source_manifest = json.loads((recipe / "source_manifest.json").read_text())
    package_manifest = json.loads((package / "manifest.json").read_text())
    if source_manifest != package_manifest:
        raise ValueError("Recipe source manifest differs from the supplied package")
    tables = {}
    for name in ("constructs", "contexts", "raw_quantities", "filtered_vectors", "final_vectors"):
        relative = f"external/soh2024_digestion/typed/{name}.parquet"
        source = recipe / "source_tables" / f"{name}.parquet"
        if digest(source) != source_manifest["files"][relative] or digest(
            package / relative
        ) != digest(source):
            raise ValueError(f"Source table changed: {name}")
        tables[name] = pq.read_table(
            source, columns=list(CONTEXT_COLUMNS) if name == "contexts" else None
        ).to_pylist()

    constructs = {row["construct_id"]: row for row in tables["constructs"]}
    contexts = {row["condition_id"]: row for row in tables["contexts"]}
    assigned = components(tables["constructs"], tables["final_vectors"])
    if sorted(records(recipe / "components.json"), key=lambda r: r["construct_id"]) != [
        assigned[cid] for cid in sorted(assigned)
    ]:
        raise ValueError("Construct grouping or fold assignment changed")

    raw_zero, conflicts = defaultdict(list), set()
    for row in tables["raw_quantities"]:
        condition = row["condition_id"]
        key = (condition, canonical(row["peptide_sequence"]), row["biological_replicate_raw"])
        if row["time_status"] != "source_agreement":
            conflicts.add(condition)
        if (
            row["quantity_time_hours"] == 0
            and row["metadata_time_hours"] == 0
            and row["intensity"] is not None
        ):
            raw_zero[key].append(row["source_record_id"])
    conflicts.update(
        row["condition_id"]
        for row in tables["contexts"]
        if row["training_status"] == "unresolved_source_time_conflict"
    )
    filtered_ids, labels, zero_labels = defaultdict(list), defaultdict(set), defaultdict(set)
    literal_technical = defaultdict(lambda: defaultdict(list))
    for row in tables["filtered_vectors"]:
        key = (
            row["condition_id"],
            canonical(row["peptide_sequence"]),
            row["biological_replicate_raw"],
        )
        label = row["technical_replicate_raw"]
        filtered_ids[key].append(row["source_record_id"])
        labels[key].add(label)
        if 0 in {float(t) for t in row["times_raw"].split(";")}:
            zero_labels[key].add(label)
        if label in ("1", "2"):
            time_values = {
                float(t): value(v)
                for t, v in zip(
                    row["times_raw"].split(";"),
                    row["intensities_raw"].split(";"),
                    strict=True,
                )
                if float(t) > 0
            }
            literal_technical[
                row["condition_id"], row["peptide_sequence"], row["biological_replicate_raw"]
            ][label].append(time_values)
    grouped = defaultdict(list)
    for row in tables["final_vectors"]:
        grouped[
            row["condition_id"],
            canonical(row["peptide_sequence"]),
            row["biological_replicate_raw"],
        ].append(row)

    facts = {}
    for key, rows in grouped.items():
        condition, peptide, bio = key
        cid = contexts[condition]["construct_id"]
        if {row["construct_id"] for row in rows} != {cid}:
            raise ValueError(f"Source construct conflict: {key}")
        parent = constructs[cid]["sequence"]
        matches = (
            [
                index
                for index in range(len(parent) - len(peptide) + 1)
                if canonical(parent).startswith(peptide, index)
            ]
            if peptide
            else []
        )
        source_slots, observations = slots(rows)
        errors = set()
        if not peptide or set(peptide + parent) - ALPHABET:
            errors.add("unsupported_sequence")
        if any(row["row_status"] != "reported_vector" for row in rows):
            errors.add("annotation_only")
        if any(
            row["product_type_raw"] != "PCP"
            or row["splice_type_raw"] not in (None, "", "NA", "PCP")
            for row in rows
        ):
            errors.add("not_exclusively_source_contiguous")
        if len(matches) != 1:
            errors.add("not_one_IL_equivalent_contiguous_span")
        if condition in conflicts:
            errors.add("unresolved_condition_time_conflict")
        if (
            not labels[key]
            or labels[key] != zero_labels[key]
            or len(raw_zero[key]) != len(filtered_ids[key])
        ):
            errors.add("no_actual_zero_baseline_for_every_technical_label")
        if any(not row["consistent"] for row in source_slots):
            errors.add("conflicting_IL_alias_or_repeated_slot_values")
        if len(observations) < 3:
            errors.add("fewer_than_three_positive_nonzero_times")
        facts[key] = {
            "curve_id": identity(*key),
            "condition_id": condition,
            "construct_id": cid,
            "peptide_IL_sequence": peptide,
            "peptide_aliases": sorted({r["peptide_sequence"] for r in rows}),
            "biological_replicate_raw": bio,
            "source_vector_ids": sorted(r["source_record_id"] for r in rows),
            "candidate_start_positions_0based": matches,
            "source_slots": source_slots,
            **assigned[cid],
            "reasons": sorted(errors),
            "observations": observations,
        }

    candidates = defaultdict(list)
    for key, fact in facts.items():
        if not fact["reasons"]:
            candidates[key[:2]].append(fact)
    selected = {}
    for cohort in candidates.values():
        shared = set.intersection(
            *({obs["time_hours"] for obs in fact["observations"]} for fact in cohort)
        )
        if len(cohort) >= 2 and len(shared) >= 3:
            for fact in cohort:
                selected[fact["curve_id"]] = sorted(shared)
    seen_vectors, n_dispositions = set(), 0
    reason_counts = Counter()
    for row in records(recipe / "dispositions.json"):
        key = (
            row["condition_id"],
            row["peptide_IL_sequence"],
            row["biological_replicate_raw"],
        )
        expected = facts.pop(key, None)
        if expected is None:
            raise ValueError(f"Duplicate or invented disposition: {key}")
        reasons = set(expected["reasons"])
        if not reasons and expected["curve_id"] not in selected:
            reasons.add("insufficient_repeated_common_positive_times")
        expected["reasons"] = sorted(reasons)
        expected["status"] = "excluded" if reasons else "admitted"
        expected.pop("observations")
        if row != expected:
            raise ValueError(f"Disposition differs from source: {key}")
        overlap = seen_vectors.intersection(row["source_vector_ids"])
        if overlap:
            raise ValueError(f"Duplicated final source vectors: {overlap}")
        seen_vectors.update(row["source_vector_ids"])
        reason_counts.update(row["reasons"])
        n_dispositions += 1
    if facts or len(seen_vectors) != len(tables["final_vectors"]):
        raise ValueError("Final source vectors lack a disposition")

    # Recompute observations directly from source vector cells for every admitted curve.
    admitted = 0
    seen_examples = set()
    by_split, by_condition = Counter(), Counter()
    repeated = defaultdict(list)
    for row in records(recipe / "examples.json"):
        curve_id = row["curve_id"]
        if curve_id in seen_examples or curve_id not in selected:
            raise ValueError(f"Duplicate or ineligible example: {curve_id}")
        seen_examples.add(curve_id)
        key = (
            row["condition_id"],
            row["peptide_IL_sequence"],
            row["biological_replicate_raw"],
        )
        source_rows = grouped[key]
        condition, peptide, _ = key
        cid = contexts[condition]["construct_id"]
        parent = constructs[cid]["sequence"]
        source_slots, observations = slots(source_rows)
        start = row["start_0based"]
        expected = {
            "curve_id": curve_id,
            "condition_id": condition,
            "construct_id": cid,
            "peptide_IL_sequence": peptide,
            "peptide_aliases": sorted({r["peptide_sequence"] for r in source_rows}),
            "biological_replicate_raw": key[2],
            "source_vector_ids": sorted(r["source_record_id"] for r in source_rows),
            "candidate_start_positions_0based": [start],
            **assigned[cid],
            "parent_sequence": parent,
            "start_0based": start,
            "end_0based": start + len(peptide),
            "peptide_sequence": parent[start : start + len(peptide)],
            "observations": observations,
            "target_unit": UNIT,
            "target_quantity_scale": SCALE,
            "context": contexts[condition],
            "actual_zero_raw_quantity_ids": sorted(raw_zero[key]),
            "filtered_source_vector_ids": sorted(filtered_ids[key]),
            "common_positive_times": selected[curve_id],
        }
        if row != expected:
            raise ValueError(f"Example or target differs from source: {curve_id}")
        if len(source_slots) < len(observations):
            raise ValueError("Source slot inventory lost")
        admitted += 1
        by_split[row["split"]] += 1
        by_condition[condition] += 1
        repeated[condition, peptide].append(
            (
                row["split"],
                {obs["time_hours"]: math.log10(obs["target_value"]) for obs in observations},
            )
        )
    if seen_examples != set(selected):
        raise ValueError("Admitted curves lack examples")
    summary = json.loads((recipe / "summary.json").read_text())
    if (
        summary["n_source_final_vectors"] != len(tables["final_vectors"])
        or summary["n_curve_candidates"] != n_dispositions
        or summary["n_examples"] != admitted
        or summary["n_components"] != len({r["component_id"] for r in assigned.values()})
        or summary["n_constructs"] != len(constructs)
        or summary["reason_counts"] != dict(sorted(reason_counts.items()))
        or summary["curves_by_split"] != dict(sorted(by_split.items()))
        or summary["curves_by_condition"] != dict(sorted(by_condition.items()))
    ):
        raise ValueError("Recipe summary counts disagree with source")
    repeatability = {}
    for split in ("train", "val", "test"):
        coefficients, n_pairs, n_constant_pairs = [], 0, 0
        for curves in repeated.values():
            for index, (left_split, left) in enumerate(curves):
                for right_split, right in curves[index + 1 :]:
                    if left_split != right_split:
                        raise ValueError("Repeated source labels split across folds")
                    if left_split != split:
                        continue
                    n_pairs += 1
                    times = sorted(set(left) & set(right))
                    if len(times) < 3:
                        raise ValueError(
                            "Admitted repeated curves have fewer than three common times"
                        )
                    values_a, values_b = [left[t] for t in times], [right[t] for t in times]
                    if len(set(values_a)) < 2 or len(set(values_b)) < 2:
                        n_constant_pairs += 1
                    else:
                        coefficients.append(statistics.correlation(values_a, values_b))
        repeatability[split] = {
            "n_label_pairs": n_pairs,
            "n_constant_pairs": n_constant_pairs,
            "n_nonconstant_pairs": len(coefficients),
            "median_log10_temporal_pearson": statistics.median(coefficients)
            if coefficients
            else None,
            "fraction_positive": sum(value > 0 for value in coefficients) / len(coefficients)
            if coefficients
            else None,
        }
    technical_repeatability = {}
    for split in ("train", "val", "test"):
        coefficients, n_unpaired, n_constant = [], 0, 0
        for (condition, _, _), labels_to_rows in literal_technical.items():
            if assigned[contexts[condition]["construct_id"]]["split"] != split:
                continue
            if len(labels_to_rows.get("1", [])) != 1 or len(labels_to_rows.get("2", [])) != 1:
                n_unpaired += 1
                continue
            left, right = labels_to_rows["1"][0], labels_to_rows["2"][0]
            times = [
                t
                for t in sorted(set(left) & set(right))
                if left[t] is not None and right[t] is not None and left[t] >= 0 and right[t] >= 0
            ]
            if len(times) < 3:
                n_unpaired += 1
                continue
            values_a = [math.log10(1 + left[t]) for t in times]
            values_b = [math.log10(1 + right[t]) for t in times]
            if len(set(values_a)) < 2 or len(set(values_b)) < 2:
                n_constant += 1
            else:
                coefficients.append(statistics.correlation(values_a, values_b))
        technical_repeatability[split] = {
            "n_nonconstant_literal_pairs": len(coefficients),
            "n_unpaired_or_short_literal_groups": n_unpaired,
            "n_constant_literal_pairs": n_constant,
            "median_log10_1plus_MS1_temporal_pearson": statistics.median(coefficients)
            if coefficients
            else None,
            "fraction_positive": sum(value > 0 for value in coefficients) / len(coefficients)
            if coefficients
            else None,
        }
    return {
        "passed": True,
        "n_source_final_vectors": len(tables["final_vectors"]),
        "n_curve_dispositions": n_dispositions,
        "n_admitted_curves": admitted,
        "n_construct_components": summary["n_components"],
        "temporal_repeatability_by_split": repeatability,
        "technical_repeatability_by_split": technical_repeatability,
        "source_manifest_sha256": digest(package / "manifest.json")["sha256"],
        "recipe_manifest_sha256": digest(recipe / "manifest.json")["sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_package", type=Path)
    parser.add_argument("recipe_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.source_package, args.recipe_directory), indent=2, sort_keys=True))
