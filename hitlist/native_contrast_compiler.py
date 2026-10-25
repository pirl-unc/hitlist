"""Source-wide decisions and same-parent contrasts, with no predictor fitting."""

import hashlib
import json
import math
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .mapp_evidence import _Sink
from .native_contrast_schema import CONTRACT, RECIPE_ID, SAMPLES, SCHEMAS

AA = set("ACDEFGHIKLMNPQRSTVWY")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _rows(directory, name):
    for batch in pq.ParquetFile(directory / (name + ".parquet")).iter_batches():
        yield from batch.to_pylist()


def _index(rows, key):
    result = {}
    for row in rows:
        if row[key] in result or row[key] is None:
            raise ValueError("Duplicate or missing native identity: " + key)
        result[row[key]] = row
    return result


def split_for_component(component):
    bucket = int(_hash("pxd018722-native-split-v1|" + component), 16) % 20
    return "train" if bucket < 14 else "val" if bucket < 17 else "test"


def _positive(value):
    return value is not None and math.isfinite(value) and value > 0


def compile_contrasts(source, destination):
    peptides = _index(_rows(source, "peptides"), "peptide_id")
    references = _index(_rows(source, "references"), "reference_record_id")
    components = _index(_rows(source, "dependency_groups"), "peptide_id")
    if components.keys() != peptides.keys():
        raise ValueError("Every source peptide requires its original dependency component")
    acquisitions = _index(_rows(source, "acquisitions"), "acquisition_id")
    by_label = _index(acquisitions.values(), "sample_label_raw")
    if set(by_label) != set(SAMPLES):
        raise ValueError("Native contrast requires the six original acquisition labels")
    forms = _index(_rows(source, "forms"), "form_id")
    identifications = _index(_rows(source, "identifications"), "identification_id")
    quantities, modified, feature_types = {}, defaultdict(set), defaultdict(set)
    for q in _rows(source, "quantities"):
        if q["acquisition_id"] is None:
            continue
        pid, aid = q["peptide_id"], q["acquisition_id"]
        if pid not in peptides or aid not in acquisitions:
            raise ValueError("Unknown native quantity peptide/acquisition")
        if q["observation_kind"] == "peptide_sequence" and q["quantity_kind"] == "source_intensity":
            if (pid, aid) in quantities:
                raise ValueError("Multiple original raw sequence quantities in one acquisition")
            if q["source_field"] != "Intensity " + acquisitions[aid]["sample_label_raw"]:
                raise ValueError("Raw quantity field/acquisition disagrees")
            quantities[pid, aid] = q
        elif _positive(q["quantitative_value"]):
            if q["observation_kind"] == "modification_specific_peptide":
                form = forms[q["form_id"]]
                if form["peptide_id"] != pid:
                    raise ValueError("Source form quantity disagrees with its peptide")
                if form["peptide_modifications_raw"] != "Unmodified":
                    modified[pid, aid].add(form["peptide_modifications_raw"] or "unreported")
            elif q["observation_kind"] == "feature":
                feature = identifications[q["identification_id"]]
                if feature["peptide_id"] != pid:
                    raise ValueError("Source feature quantity disagrees with its peptide")
                feature_types[pid, aid].add(feature["identification_type"])
    origins, declared = defaultdict(list), defaultdict(list)
    for origin in _rows(source, "origins"):
        origins[origin["peptide_id"]].append(origin)
    for claim in _rows(source, "declared_parents"):
        declared[claim["peptide_id"]].append(claim)
    decisions, eligible = {}, defaultdict(list)
    for pid, peptide in sorted(peptides.items()):
        reasons = []
        sequence, pep = peptide["peptide_sequence"], peptide["posterior_error_probability"]
        if not sequence or set(sequence) - AA or not 8 <= len(sequence) <= 25:
            reasons.append("outside_sequence_search_support")
        if peptide["reverse_raw"]:
            reasons.append("source_reverse_flag")
        if peptide["contaminant_raw"]:
            reasons.append("source_contaminant_flag")
        if pep is None or not math.isfinite(pep) or not 0 <= pep <= 0.01:
            reasons.append("invalid_or_above_threshold_PEP")
        alternatives = origins[pid]
        ref_ids = {r["reference_record_id"] for r in alternatives}
        hashes = {references[r]["reference_sequence_sha256"] for r in ref_ids}
        if not alternatives:
            reasons.append("no_deposited_reference_origin")
        elif len(hashes) != 1:
            reasons.append("multiple_complete_parent_sequences")
        if any(r["origin_match_type"] != "literal" for r in alternatives):
            reasons.append("nonliteral_origin_alternative")
        if any(references[r]["organism_taxon_id"] != "9606" for r in ref_ids):
            reasons.append("nonhuman_or_unresolved_taxonomy")
        if any(references[r]["reference_role"] == "engineered_construct" for r in ref_ids):
            reasons.append("engineered_origin_alternative")
        parent = next(iter(hashes)) if len(hashes) == 1 else None
        if any(c["reference_record_id"] is None for c in declared[pid]):
            reasons.append("unresolved_declared_parent")
        if parent is not None and any(
            references[c["reference_record_id"]]["reference_sequence_sha256"] != parent
            for c in declared[pid]
            if c["reference_record_id"] is not None
        ):
            reasons.append("declared_parent_disagrees_with_unique_origin")
        measurements = []
        for label in SAMPLES:
            aid = by_label[label]["acquisition_id"]
            q = quantities.get((pid, aid))
            measurements.append(
                {
                    "sample_label_raw": label,
                    "acquisition_id": aid,
                    "measurement_id": q["measurement_id"] if q else None,
                    "quantitative_raw": q["quantitative_raw"] if q else None,
                    "quantitative_value": q["quantitative_value"] if q else None,
                    "value_status": q["value_status"] if q else "missing_source_row",
                }
            )
            if q is None:
                reasons.append("missing_source_quantity_row")
            elif not _positive(q["quantitative_value"]):
                reasons.append("nonpositive_or_missing_source_signal")
        group = components[pid]["dependency_group_id"]
        decisions[pid] = {
            "peptide_id": pid,
            "source_record_id": peptide["source_record_id"],
            "native_id": peptide["native_id"],
            "peptide_sequence": sequence,
            "parent_sequence_sha256": parent,
            "dependency_group_id": group,
            "split": split_for_component(group),
            "status": "excluded" if reasons else "eligible",
            "reasons_json": _json(sorted(set(reasons))),
            "source_quantities_json": _json(measurements),
            "candidate_origins_json": _json(alternatives),
            "declared_parents_json": _json(declared[pid]),
            "feature_types_by_acquisition_json": _json(
                {
                    label: sorted(feature_types[pid, by_label[label]["acquisition_id"]])
                    for label in SAMPLES
                }
            ),
            "modified_form_support_json": _json(
                {
                    label: sorted(modified[pid, by_label[label]["acquisition_id"]])
                    for label in SAMPLES
                }
            ),
            "reference_peptide_id": None,
            "posterior_error_probability": pep,
        }
        if not reasons:
            eligible[parent].append(pid)

    for decision in decisions.values():
        parent = decision["parent_sequence_sha256"]
        if parent is not None and len(eligible[parent]) < 2:
            reasons = json.loads(decision["reasons_json"])
            reasons.append("parent_has_fewer_than_two_eligible_peptides")
            decision.update(status="excluded", reasons_json=_json(sorted(set(reasons))))
    examples, parent_rows, contributors = [], [], []
    for parent, members in sorted(eligible.items()):
        if len(members) < 2:
            continue
        if len({components[p]["dependency_group_id"] for p in members}) != 1:
            raise ValueError("Same-parent peptides cross original dependency components")
        reference = min(
            members,
            key=lambda p: _hash(
                "pxd018722-native-reference-v1|" + parent + "|" + peptides[p]["peptide_sequence"]
            ),
        )
        group = components[reference]["dependency_group_id"]
        split = split_for_component(group)
        refs = [r for r in references.values() if r["reference_sequence_sha256"] == parent]
        if len({r["reference_sequence"] for r in refs}) != 1:
            raise ValueError("Reference sequence checksum collision/inconsistency")
        parent_rows.append(
            {
                "parent_sequence_sha256": parent,
                "parent_sequence": refs[0]["reference_sequence"],
                "reference_records_json": _json(
                    [
                        {
                            k: r[k]
                            for k in ("reference_record_id", "accession_raw", "source_record_id")
                        }
                        for r in refs
                    ]
                ),
                "reference_peptide_id": reference,
                "dependency_group_id": group,
                "split": split,
            }
        )
        for pid in sorted(members):
            decisions[pid].update(
                status="reference" if pid == reference else "contrast",
                reference_peptide_id=reference,
            )
            if pid == reference:
                continue
            key = _hash(RECIPE_ID + "|" + parent + "|" + pid + "|" + reference)
            ratios, ids = [], []
            for label in SAMPLES:
                aid = by_label[label]["acquisition_id"]
                a, b = quantities[pid, aid], quantities[reference, aid]
                ratios.append(
                    math.log10(a["quantitative_value"]) - math.log10(b["quantitative_value"])
                )
                for role, q in (("peptide", a), ("reference", b)):
                    ids.append(q["measurement_id"])
                    coefficient = (
                        (1 if label in SAMPLES[3:] else -1) * (1 if role == "peptide" else -1) / 3
                    )
                    contributors.append(
                        {
                            "contributor_id": _hash(key + "|" + q["measurement_id"] + "|" + role),
                            "example_id": key,
                            "measurement_id": q["measurement_id"],
                            "peptide_id": q["peptide_id"],
                            "source_record_id": q["source_record_id"],
                            "acquisition_id": aid,
                            "sample_label_raw": label,
                            "quantity_role": role,
                            "quantitative_raw": q["quantitative_raw"],
                            "quantitative_value": q["quantitative_value"],
                            "target_coefficient": coefficient,
                        }
                    )
            examples.append(
                {
                    "example_id": key,
                    "peptide_id": pid,
                    "reference_peptide_id": reference,
                    "peptide_sequence": peptides[pid]["peptide_sequence"],
                    "reference_peptide_sequence": peptides[reference]["peptide_sequence"],
                    "parent_sequence_sha256": parent,
                    "dependency_group_id": group,
                    "split": split,
                    "measurement_ids_json": _json(ids),
                    "acquisition_log_ratios_json": _json(ratios),
                    "target_log10_ratio_change": (math.fsum(ratios[3:]) - math.fsum(ratios[:3]))
                    / 3,
                    "weight": None,
                }
            )
    counts = Counter(
        (r["split"], r["dependency_group_id"], r["parent_sequence_sha256"]) for r in examples
    )
    parents_by_group, groups_by_split = defaultdict(set), defaultdict(set)
    for split, group, parent in counts:
        parents_by_group[split, group].add(parent)
        groups_by_split[split].add(group)
    for row in examples:
        split, group, parent = (
            row["split"],
            row["dependency_group_id"],
            row["parent_sequence_sha256"],
        )
        row["weight"] = 1 / (
            len(groups_by_split[split])
            * len(parents_by_group[split, group])
            * counts[split, group, parent]
        )
    outputs = {
        "decisions": list(decisions.values()),
        "parents": parent_rows,
        "examples": examples,
        "covariates": [
            {k: v for k, v in r.items() if k in SCHEMAS["covariates"].names} for r in examples
        ],
        "contributors": contributors,
        "components": [
            {**r, "split": split_for_component(r["dependency_group_id"])}
            for _, r in sorted(components.items())
        ],
    }
    # Native source rows have common provenance fields; the output component grain
    # explicitly selects its own fields without dropping the original snapshot.
    outputs["components"] = [
        {k: r[k] for k in SCHEMAS["components"].names} for r in outputs["components"]
    ]
    destination.mkdir(parents=True)
    for name, values in outputs.items():
        sink = _Sink(destination, name, schema=SCHEMAS[name])
        try:
            for row in values:
                sink.add(row)
        finally:
            sink.close()
    split_support = {}
    for split in ("train", "val", "test"):
        members = [r for r in examples if r["split"] == split]
        observed = {
            "n_components": len(groups_by_split[split]),
            "n_contrasts": len(members),
            "n_parent_sequences": len({r["parent_sequence_sha256"] for r in members}),
        }
        observed["gate_passed"] = all(
            observed[k] >= v for k, v in CONTRACT["support_gate"][split].items()
        )
        observed["weight_sum"] = math.fsum(r["weight"] for r in members)
        split_support[split] = observed
    return {
        "n_source_peptides": len(peptides),
        "n_source_components": len({r["dependency_group_id"] for r in components.values()}),
        "n_parents": len(parent_rows),
        "n_contrasts": len(examples),
        "n_contributors": len(contributors),
        "dispositions": dict(sorted(Counter(r["status"] for r in decisions.values()).items())),
        "exclusion_reasons": dict(
            sorted(
                Counter(
                    reason for r in decisions.values() for reason in json.loads(r["reasons_json"])
                ).items()
            )
        ),
        "split_support": split_support,
        "qualified_for_screen": all(r["gate_passed"] for r in split_support.values()),
        "numeric_target_summaries_disclosed": False,
    }
