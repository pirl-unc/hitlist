"""Declared training recipes over verified shared evidence generations.

The first recipe estimates relative captured signal on an admitted observable
set. It does not assign missing-fragment negatives or intracellular concentration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

RECIPE_ID = "mapp_captured_composition_v1"
REFERENCE_ID = "reviewed_2026_03_isoforms"
CONDITIONS = ("UT", "TNF+IFN")
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
TABLES = (
    "peptide_rows",
    "channels",
    "evidence",
    "quantity_results",
    "quantity_measurements",
    "publisher_quantities",
    "derivation_inputs",
    "reference_proteins",
    "peptide_reference_occurrences",
    "peptide_reference_coverage",
)
RECIPE = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "source_id": "pxd009092",
    "reference_id": REFERENCE_ID,
    "endpoint": "relative_captured_fragment_intensity_given_admitted_parent_set",
    "quantity": "published_nonimputed_log10_median_three_filtered_runs_including_zeros",
    "conditions": list(CONDITIONS),
    "selection": {
        "canonical_peptide_and_parent": "ACDEFGHIKLMNPQRSTVWY",
        "reverse_or_contaminant": "excluded",
        "n_source_groups": 1,
        "min_positive_filtered_runs_per_condition": 2,
        "min_direct_msms_runs_per_condition": 1,
        "max_reported_peptide_pep": 0.01,
        "n_distinct_full_parent_sequences": 1,
        "source_leading_coordinate_and_flanks": "required_match",
        "min_eligible_peptides_per_parent": 3,
    },
    "alternatives": "collapse_identical_full_sequences_keep_all_accessions_and_positions",
    "split": {
        "graph": "all_source_peptides_joined_by_reference_sequence_components_and_source_groups",
        "key": "sha256(recipe_id + ':' + lexically_smallest_component_peptide_id)",
        "bucket": "first_8_digest_bytes_big_endian_modulo_100",
        "train": [0, 69],
        "validation": [70, 84],
        "test": [85, 99],
    },
    "limitations": [
        "conditional_captured_signal_not_molar_fraction_or_free_concentration",
        "peptide_specific_capture_and_ms_response_unidentified",
        "outside_admitted_set_mass_unidentified_not_negative",
        "original_search_scope_6_to_40",
        "original_2016_search_reference_unavailable",
        "peptide_PEP_selection_not_resolution_of_source_FDR_conflict",
        "source_leading_match_not_proof_of_biological_origin",
        "exact_components_not_homology_or_specimen_independence",
        "human_HEK293_not_professional_APC_validation",
        "same_peptide_response_cancellation_across_conditions_is_an_assumption",
    ],
}


PAIR_RECIPE_ID = "mapp_captured_composition_min2_v1"
PAIR_RECIPE = json.loads(json.dumps(RECIPE))
PAIR_RECIPE["recipe_id"] = PAIR_RECIPE_ID
PAIR_RECIPE["selection"]["min_eligible_peptides_per_parent"] = 2
PAIR_RECIPE["split"]["salt"] = RECIPE_ID
PAIR_RECIPE["split"]["key"] = "sha256(split_salt + ':' + lexically_smallest_component_peptide_id)"
RECIPES = {RECIPE_ID: RECIPE, PAIR_RECIPE_ID: PAIR_RECIPE}

RUN_RECIPE_ID = "mapp_complete_run_composition_v1"
RUN_RECIPE = json.loads(json.dumps(PAIR_RECIPE))
RUN_RECIPE["recipe_id"] = RUN_RECIPE_ID
RUN_RECIPE["base_recipe_id"] = PAIR_RECIPE_ID
RUN_RECIPE["selection"]["all_six_filtered_runs"] = "finite_and_strictly_positive"
RUN_RECIPE["run_observations"] = "all_base_recipe_peptides_including_ineligible_filtered_zeros"
RUN_RECIPE["limitations"].append(
    "complete_case_population_not_unbiased_replication_or_specimen_pairing"
)
RECIPES[RUN_RECIPE_ID] = RUN_RECIPE


def registered_contracts():
    """Return independent JSON-compatible public contracts, without source reads."""
    return [json.loads(json.dumps(recipe)) for recipe in RECIPES.values()]


def _unique(rows, field):
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate recipe input identity: {field}")
    return result


def _split_components(peptides, coverage):
    roots = {p: p for p in peptides}

    def root(p):
        while p != roots[p]:
            roots[p] = roots[roots[p]]
            p = roots[p]
        return p

    owners = {}
    for pid, peptide in sorted(peptides.items()):
        groups = [g for g in peptide["group_ids_raw"].split(";") if g]
        links = [("reference", coverage[pid]["shared_sequence_component_id"])] + [
            ("source_group", g) for g in groups
        ]
        for link in links:
            if link in owners:
                a, b = root(pid), root(owners[link])
                roots[max(a, b)] = min(a, b)
            else:
                owners[link] = pid
    return {p: root(p) for p in peptides}


def _partition(component):
    digest = hashlib.sha256(f"{RECIPE_ID}:{component}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 70 else ("validation" if bucket < 85 else "test")


def compile_mapp_recipe(tables, *, recipe_id=RECIPE_ID):
    """Compile typed relations; caller must verify their enclosing generation."""
    if recipe_id not in RECIPES:
        raise ValueError(f"Unknown training recipe: {recipe_id}")
    if recipe_id == RUN_RECIPE_ID:
        return _compile_complete_runs(tables)
    minimum = RECIPES[recipe_id]["selection"]["min_eligible_peptides_per_parent"]
    minimum_word = "three" if minimum == 3 else "two"
    peptides = _unique(tables["peptide_rows"], "peptide_row_id")
    channels = _unique(tables["channels"], "channel_id")
    results = _unique(tables["quantity_results"], "result_id")
    quantities = _unique(tables["quantity_measurements"], "measurement_id")
    proteins = _unique(
        [p for p in tables["reference_proteins"] if p["reference_id"] == REFERENCE_ID],
        "reference_record_id",
    )
    coverage = _unique(
        [p for p in tables["peptide_reference_coverage"] if p["reference_id"] == REFERENCE_ID],
        "peptide_row_id",
    )
    if set(coverage) != set(peptides) or not proteins:
        raise ValueError("Incomplete recipe reference coverage")
    for condition in CONDITIONS:
        if sum(c["condition"] == condition for c in channels.values()) != 3:
            raise ValueError("Expected three source runs per condition")
    context_fields = (
        "measured_species",
        "measured_cell",
        "measured_pool",
        "material_link_precision",
    )
    source_context = None
    for channel in channels.values():
        if channel["condition"] in CONDITIONS:
            observed_context = {field: channel[field] for field in context_fields}
            if source_context is not None and observed_context != source_context:
                raise ValueError("Source contexts differ within the declared comparison")
            source_context = observed_context
    components = _split_components(peptides, coverage)
    occurrences, aliases = defaultdict(list), defaultdict(list)
    for p in proteins.values():
        if hashlib.sha256(p["sequence"].encode()).hexdigest() != p["sequence_sha256"]:
            raise ValueError("Protein sequence identity mismatch")
        aliases[p["sequence_sha256"]].append(p["reference_record_id"])
    for r in tables["peptide_reference_occurrences"]:
        if r["reference_id"] == REFERENCE_ID:
            occurrences[r["peptide_row_id"]].append(r)
    positive, direct, evidence = defaultdict(set), defaultdict(set), defaultdict(list)
    for q in quantities.values():
        if q["representation"] != "bead_filtered":
            continue
        result = results[q["result_id"]]
        if result["peptide_row_id"] != q["peptide_row_id"]:
            raise ValueError("Quantity and result peptide identities disagree")
        condition = channels[result["channel_id"]]["condition"]
        if condition in CONDITIONS and q["value"] is not None and q["value"] > 0:
            positive[q["peptide_row_id"], condition].add(result["channel_id"])
    for r in tables["evidence"]:
        condition = channels[r["channel_id"]]["condition"]
        if condition in CONDITIONS:
            evidence[r["peptide_row_id"], condition].append(r["evidence_id"])
            if r["msms_count"] is not None and r["msms_count"] > 0:
                direct[r["peptide_row_id"], condition].add(r["channel_id"])
    publisher = {}
    for r in tables["publisher_quantities"]:
        if r["representation"] == "log10_median_filtered" and r["condition"] in CONDITIONS:
            key = r["peptide_row_id"], r["condition"]
            if key in publisher:
                raise ValueError("Duplicate publisher endpoint")
            publisher[key] = r
    dependencies = defaultdict(list)
    for edge in tables["derivation_inputs"]:
        if edge["input_role"] == "median_run":
            dependencies[edge["derived_measurement_id"]].append(edge["input_measurement_id"])
    ledger, grouped = [], defaultdict(list)
    for pid, p in sorted(peptides.items()):
        reasons = []
        for flag, reason in (
            (p["reverse"], "source_reverse"),
            (p["potential_contaminant"], "source_contaminant"),
            (
                not p["sequence_raw"] or not set(p["sequence_raw"]) <= CANONICAL,
                "noncanonical_peptide",
            ),
            (len(set(filter(None, p["group_ids_raw"].split(";")))) != 1, "ambiguous_source_group"),
        ):
            if flag:
                reasons.append(reason)
        for condition in CONDITIONS:
            if len(positive[pid, condition]) < 2:
                reasons.append("insufficient_positive_runs:" + condition)
            if not direct[pid, condition]:
                reasons.append("missing_direct_msms:" + condition)
        structural = not reasons
        if p["pep"] is None or not math.isfinite(p["pep"]):
            reasons.append("peptide_confidence_unreported")
        elif not 0 <= p["pep"] <= 0.01:
            reasons.append("peptide_pep_above_threshold")
        for condition in CONDITIONS:
            r = publisher.get((pid, condition))
            if (
                r is None
                or r["value_status"] != "reported"
                or r["value"] is None
                or not math.isfinite(r["value"])
            ):
                reasons.append("missing_nonimputed_quantity:" + condition)
        sequence_ids = {
            proteins[r["reference_record_id"]]["sequence_sha256"] for r in occurrences[pid]
        }
        if not sequence_ids:
            reasons.append("reference_unmapped")
        elif len(sequence_ids) != 1:
            reasons.append("multiple_parent_sequences")
        if not coverage[pid]["leading_coordinate_flanks_match"]:
            reasons.append("source_leading_coordinate_or_flank_unconfirmed")
        parent = next(iter(sequence_ids)) if len(sequence_ids) == 1 else None
        if parent is not None and not set(proteins[aliases[parent][0]]["sequence"]) <= CANONICAL:
            reasons.append("noncanonical_parent")
        row = {
            "peptide_row_id": pid,
            "source_record_id": p["source_record_id"],
            "structural_support": structural,
            "parent_sequence_sha256": parent,
            "n_reference_sequences": len(sequence_ids),
            "n_reference_occurrences": len(occurrences[pid]),
            "component_id": components[pid],
            "split": _partition(components[pid]),
            "reasons": reasons,
        }
        ledger.append(row)
        if not reasons:
            grouped[parent].append(row)
    for members in grouped.values():
        if len(members) < minimum:
            for row in members:
                row["reasons"].append(f"fewer_than_{minimum_word}_eligible_peptides_per_parent")
    for row in ledger:
        row["status"] = "excluded" if row["reasons"] else "admitted"
    parent_rows, examples, contributors = [], [], []
    for parent, members in sorted(grouped.items()):
        if len(members) < minimum:
            continue
        component_set = {m["component_id"] for m in members}
        if len(component_set) != 1:
            raise ValueError("A full parent sequence crosses split components")
        component = next(iter(component_set))
        sequence = proteins[aliases[parent][0]]["sequence"]
        parent_rows.append(
            {
                "parent_sequence_sha256": parent,
                "sequence": sequence,
                "reference_record_ids": sorted(aliases[parent]),
                "reference_id": REFERENCE_ID,
                "component_id": component,
                "split": _partition(component),
            }
        )
        for condition in CONDITIONS:
            candidates = []
            for member in members:
                pid = member["peptide_row_id"]
                p, measure = peptides[pid], publisher[pid, condition]
                inputs = sorted(dependencies[measure["measurement_id"]])
                if len(inputs) != len(set(inputs)) or len(inputs) != 3:
                    raise ValueError("Expected three distinct publisher run contributors")
                run_channels = set()
                for mid in inputs:
                    q = quantities[mid]
                    result = results[q["result_id"]]
                    c = channels[result["channel_id"]]
                    if (
                        q["peptide_row_id"] != pid
                        or c["condition"] != condition
                        or q["representation"] != "bead_filtered"
                    ):
                        raise ValueError("Publisher contributor relationship mismatch")
                    run_channels.add(result["channel_id"])
                if len(run_channels) != 3:
                    raise ValueError("Publisher contributors repeat an acquisition")
                positions = sorted({r["start_1based"] - 1 for r in occurrences[pid]})
                if any(
                    sequence[start : start + len(p["sequence_raw"])] != p["sequence_raw"]
                    for start in positions
                ):
                    raise ValueError("Literal peptide occurrence mismatch")
                candidates.append(
                    {
                        "peptide_row_id": pid,
                        "sequence": p["sequence_raw"],
                        "start_positions_0based": positions,
                        "source_log10_intensity": measure["value"],
                        "measurement_id": measure["measurement_id"],
                    }
                )
                contributors.append(
                    {
                        "peptide_row_id": pid,
                        "condition": condition,
                        "measurement_id": measure["measurement_id"],
                        "source_cell": measure["source_cell"],
                        "run_measurement_ids": inputs,
                        "channel_ids": sorted(run_channels),
                        "evidence_ids": sorted(evidence[pid, condition]),
                        "direct_msms_channel_ids": sorted(direct[pid, condition]),
                    }
                )
            log_values = [r["source_log10_intensity"] * math.log(10) for r in candidates]
            pivot = max(log_values)
            log_z = pivot + math.log(sum(math.exp(x - pivot) for x in log_values))
            mean = sum(r["source_log10_intensity"] for r in candidates) / len(candidates)
            for row, value in zip(candidates, log_values):
                row["target_log_probability"] = value - log_z
                row["target_probability"] = math.exp(value - log_z)
                row["target_centered_log10"] = row["source_log10_intensity"] - mean
            examples.append(
                {
                    "example_id": f"{recipe_id}:{parent}:{condition}",
                    "parent_sequence_sha256": parent,
                    "component_id": component,
                    "split": _partition(component),
                    "condition": condition,
                    "context": [float(condition == "TNF+IFN")],
                    **source_context,
                    "candidates": candidates,
                }
            )
    summary = {
        "n_source_peptides": len(peptides),
        "n_structurally_supported_peptides": sum(r["structural_support"] for r in ledger),
        "n_admitted_peptides": sum(r["status"] == "admitted" for r in ledger),
        "n_parent_sequences": len(parent_rows),
        "n_examples": len(examples),
        "n_contributor_rows": len(contributors),
        "exclusion_reasons": dict(
            sorted(Counter(reason for row in ledger for reason in row["reasons"]).items())
        ),
        "n_parents_by_split": dict(sorted(Counter(r["split"] for r in parent_rows).items())),
        "n_all_source_components": len(set(components.values())),
        "largest_source_component_n_peptides": max(
            Counter(components.values()).values(), default=0
        ),
    }
    return {
        "parents": parent_rows,
        "examples": examples,
        "peptide_dispositions": ledger,
        "contributors": contributors,
        "summary": summary,
    }


def _compile_complete_runs(tables):
    """Keep original admission/splits, add complete-run eligibility and lineage."""
    base = compile_mapp_recipe(tables, recipe_id=PAIR_RECIPE_ID)
    channels = _unique(tables["channels"], "channel_id")
    results = _unique(tables["quantity_results"], "result_id")
    quantities = _unique(tables["quantity_measurements"], "measurement_id")
    candidates = {
        c["peptide_row_id"]: (e, c)
        for e in base["examples"]
        if e["condition"] == "UT"
        for c in e["candidates"]
    }
    evidence = defaultdict(list)
    for row in tables["evidence"]:
        evidence[row["peptide_row_id"], row["channel_id"]].append(row)
    observations = []
    for contributor in base["contributors"]:
        pid = contributor["peptide_row_id"]
        example, _ = candidates[pid]
        for mid in contributor["run_measurement_ids"]:
            q = quantities[mid]
            result = results[q["result_id"]]
            channel = channels[result["channel_id"]]
            ev = evidence[pid, channel["channel_id"]]
            if len({r["evidence_id"] for r in ev}) != len(ev):
                raise ValueError("Duplicate run evidence identity")
            positive = q["value"] is not None and math.isfinite(q["value"]) and q["value"] > 0
            observations.append(
                {
                    "peptide_row_id": pid,
                    "parent_sequence_sha256": example["parent_sequence_sha256"],
                    "component_id": example["component_id"],
                    "split": example["split"],
                    "condition": contributor["condition"],
                    "channel": channel,
                    "measurement": q,
                    "evidence_ids": sorted(r["evidence_id"] for r in ev),
                    "direct_msms_evidence_ids": sorted(
                        r["evidence_id"]
                        for r in ev
                        if r["msms_count"] is not None and r["msms_count"] > 0
                    ),
                    "positive_finite": positive,
                }
            )
    observations.sort(key=lambda r: (r["peptide_row_id"], r["channel"]["channel_id"]))
    by_peptide = defaultdict(list)
    for row in observations:
        by_peptide[row["peptide_row_id"]].append(row)
    complete = set()
    for pid, runs in by_peptide.items():
        if len(runs) != 6 or len({r["channel"]["channel_id"] for r in runs}) != 6:
            raise ValueError("Expected six distinct source acquisitions per base peptide")
        if all(r["positive_finite"] for r in runs):
            complete.add(pid)
    counts = Counter(candidates[pid][0]["parent_sequence_sha256"] for pid in complete)
    admitted = {
        pid for pid in complete if counts[candidates[pid][0]["parent_sequence_sha256"]] >= 2
    }
    parents = {candidates[pid][0]["parent_sequence_sha256"] for pid in admitted}
    for row in observations:
        pid = row["peptide_row_id"]
        row["complete_peptide"] = pid in complete
        row["admitted_complete_parent_set"] = pid in admitted
    for row in base["peptide_dispositions"]:
        pid = row["peptide_row_id"]
        if row["status"] == "admitted":
            if pid not in complete:
                row["reasons"].append("incomplete_positive_filtered_runs")
            elif pid not in admitted:
                row["reasons"].append("fewer_than_two_complete_peptides_per_parent")
            row["status"] = "admitted" if pid in admitted else "excluded"
    examples = []
    for example in base["examples"]:
        if example["parent_sequence_sha256"] not in parents:
            continue
        example["example_id"] = example["example_id"].replace(
            PAIR_RECIPE_ID + ":", RUN_RECIPE_ID + ":", 1
        )
        example["candidates"] = [
            c for c in example["candidates"] if c["peptide_row_id"] in admitted
        ]
        values = [c["source_log10_intensity"] for c in example["candidates"]]
        pivot = max(values) * math.log(10)
        log_z = pivot + math.log(sum(math.exp(v * math.log(10) - pivot) for v in values))
        for c in example["candidates"]:
            c["target_log_probability"] = c["source_log10_intensity"] * math.log(10) - log_z
            c["target_probability"] = math.exp(c["target_log_probability"])
            c["target_centered_log10"] = c["source_log10_intensity"] - sum(values) / len(values)
        examples.append(example)
    base["examples"] = examples
    base["parents"] = [r for r in base["parents"] if r["parent_sequence_sha256"] in parents]
    base["contributors"] = [r for r in base["contributors"] if r["peptide_row_id"] in admitted]
    base["run_observations"] = observations
    base["summary"].update(
        {
            "base_recipe_id": PAIR_RECIPE_ID,
            "n_base_admitted_peptides": len(candidates),
            "n_complete_positive_peptides_before_parent_gate": len(complete),
            "n_run_observations": len(observations),
            "n_nonpositive_or_missing_runs": sum(not r["positive_finite"] for r in observations),
            "n_admitted_peptides": len(admitted),
            "n_parent_sequences": len(parents),
            "n_examples": len(examples),
            "n_contributor_rows": len(base["contributors"]),
            "n_parents_by_split": dict(
                sorted(Counter(r["split"] for r in base["parents"]).items())
            ),
            "exclusion_reasons": dict(
                sorted(
                    Counter(
                        reason for r in base["peptide_dispositions"] for reason in r["reasons"]
                    ).items()
                )
            ),
        }
    )
    return base
