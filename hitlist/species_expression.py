"""Conservative, source-scoped expression audits for frozen species bundles.

This is an interchange consumer, not a discovery algorithm or an orthology-based
CTA definition. Candidate membership remains a caller-supplied empirical claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
REFERENCE_FIELDS = {
    "assembly_accession",
    "annotation_release",
    "source_version",
    "taxon",
    "asset_hashes",
}
NAMESPACE_FIELDS = ("reference_key", "source", "assay", "quantification", "unit")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sequence_id(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _required(record, fields, label):
    if any(not isinstance(record.get(k), str) or not record[k].strip() for k in fields):
        raise ValueError(f"{label} requires nonempty {', '.join(fields)}")


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_policy(policy):
    _required(policy, ("policy_id", "unit"), "Expression policy")
    if type(policy.get("version")) is not int or policy["version"] < 1:
        raise ValueError("Explicit expression policy version required")
    for field in ("allowed_tissues", "required_normal_tissues"):
        values = policy.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or not v.strip() for v in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Policy requires explicit, unique {field}")
    if set(policy["allowed_tissues"]) & set(policy["required_normal_tissues"]):
        raise ValueError("Allowed and required normal tissues overlap")
    if not _number(policy.get("allowed_min")) or policy["allowed_min"] <= 0:
        raise ValueError("allowed_min must be finite and positive")
    if not _number(policy.get("normal_max")) or policy["normal_max"] < 0:
        raise ValueError("normal_max must be finite and nonnegative")
    if type(policy.get("min_normal_donors")) is not int or policy["min_normal_donors"] < 1:
        raise ValueError("min_normal_donors must be positive")
    tissue = policy.get("tissue_blacklist", {})
    if tissue.get("version") != 1 or tissue.get("min_donors") != 2:
        raise ValueError("Tissue blacklist requires version 1 and two distinct donors")
    if tissue.get("tissue_status") != "nonmalignant" or set(tissue.get("tissue_groups", {})) != {
        "heart",
        "brain",
        "lung",
    }:
        raise ValueError("Tissue blacklist requires nonmalignant heart, brain and lung")
    if any(
        not isinstance(vs, list) or not vs or any(not isinstance(v, str) for v in vs)
        for vs in tissue["tissue_groups"].values()
    ):
        raise ValueError("Each tissue group requires a nonempty list of aliases")
    aliases = [v.casefold().strip() for vs in tissue["tissue_groups"].values() for v in vs]
    if not aliases or len(aliases) != len(set(aliases)) or any(not v for v in aliases):
        raise ValueError("Tissue aliases must be nonempty and unambiguous")


def validate_expression(expression, reference, candidates, policy, max_records):
    """Return validated identities; do not discard raw input fields."""
    if expression.get("schema_version") != 1 or expression.get("reference") != reference:
        raise ValueError("Expression reference or schema mismatch")
    key = json_digest(reference)
    occurrences, samples = {}, {}
    for rows, dest, identity in (
        (expression.get("occurrences", []), occurrences, "occurrence_id"),
        (expression.get("samples", []), samples, "sample_id"),
    ):
        if not isinstance(rows, list) or len(rows) > max_records:
            raise ValueError("Expression records exceed max_records")
        for row in rows:
            _required(row, (identity, "source", "reference_key"), "Expression identity")
            if row[identity] in dest:
                raise ValueError(f"Duplicate {identity}")
            if row["reference_key"] != key:
                raise ValueError("Expression identity has wrong reference")
            dest[row[identity]] = dict(row)
    for row in occurrences.values():
        _required(row, ("protein_id", "gene_id", "transcript_id", "sequence"), "Occurrence")
        sequence = "".join(row["sequence"].split()).upper().removesuffix("*")
        if not sequence or set(sequence) - AA or row.get("complete", True) is not True:
            raise ValueError("Occurrence requires complete, resolved protein sequence")
        row["sequence"] = sequence
        row["sequence_id"] = sequence_id(sequence)
    for row in samples.values():
        _required(
            row, ("study", "specimen", "library", "assay", "quantification", "unit"), "Sample"
        )
        if row.get("taxon") != reference["taxon"]:
            raise ValueError("Expression sample taxon mismatch")
        if row.get("health", "unknown") not in {"healthy", "tumor", "disease", "unknown"}:
            raise ValueError("Unrecognized sample health")
        if row.get("donor") is not None and not isinstance(row["donor"], str):
            raise ValueError("Sample donor must be a namespaced string or null")
    contributions = expression.get("contributions", [])
    if not isinstance(contributions, list) or len(contributions) > max_records:
        raise ValueError("Expression contributions exceed max_records")
    allocations, libraries, levels = {}, {}, defaultdict(set)
    for original in contributions:
        row = dict(original)
        _required(row, ("sample_id", "allocation_id"), "Contribution")
        sample = samples.get(row["sample_id"])
        ids = row.get("occurrence_ids")
        if (
            sample is None
            or not isinstance(ids, list)
            or not ids
            or not set(ids) <= occurrences.keys()
        ):
            raise ValueError("Unresolved expression allocation identity")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate occurrence in allocation")
        row["occurrence_ids"] = sorted(ids)
        for field, default in (
            ("scope", "transcript"),
            ("direct", True),
            ("coding_assignment", True),
        ):
            row.setdefault(field, default)
        if row["scope"] not in {"gene", "promoter", "transcript"}:
            raise ValueError("Unknown expression measurement scope")
        if any(type(row[field]) is not bool for field in ("direct", "coding_assignment")):
            raise ValueError("Expression direct/coding_assignment must be booleans")
        lower, upper = row.get("lower"), row.get("upper")
        if (lower is None) != (upper is None) or (
            lower is not None
            and (not _number(lower) or not _number(upper) or not 0 <= lower <= upper)
        ):
            raise ValueError("Expression bounds must be missing or finite, ordered and nonnegative")
        replicates = row.get("inferential_replicates", [])
        if any(not _number(v) or v < 0 for v in replicates):
            raise ValueError("Invalid inferential replicate")
        library = sample["library"]
        if library in libraries and libraries[library] != sample["sample_id"]:
            raise ValueError("Duplicate/reprocessed library; select one derivative")
        libraries[library] = sample["sample_id"]
        levels[sample["sample_id"]].add(row["scope"])
        if len(levels[sample["sample_id"]]) > 1:
            raise ValueError("Do not sum gene/promoter and transcript estimates")
        identity = (sample["sample_id"], row["allocation_id"])
        if identity in allocations and allocations[identity] != row:
            raise ValueError("Conflicting duplicated allocation")
        allocations[identity] = row
    if not isinstance(candidates, list) or len(candidates) > max_records:
        raise ValueError("Candidate records exceed max_records")
    groups = defaultdict(set)
    for oid, row in occurrences.items():
        groups[row["sequence_id"]].add(oid)
    seen = set()
    for candidate in candidates:
        sid = candidate.get("sequence_id")
        if sid not in groups or sid in seen:
            raise ValueError("Candidate sequence must resolve to one unique full-sequence group")
        seen.add(sid)
        if (
            candidate.get("admission_basis") != "species_expression"
            or candidate.get("policy_id") != policy["policy_id"]
        ):
            raise ValueError(
                "Candidate admission requires species expression and the declared policy"
            )
        support = candidate.get("support")
        if not isinstance(support, list) or not support:
            raise ValueError("Candidate requires supporting expression allocations")
        for link in support:
            allocation = allocations.get((link.get("sample_id"), link.get("allocation_id")))
            if allocation is None or not set(allocation["occurrence_ids"]) & groups[sid]:
                raise ValueError("Candidate support does not resolve to its sequence group")
            if allocation.get("upper") is None or allocation["upper"] <= 0:
                raise ValueError("Candidate admission needs observed expression support")
    return occurrences, samples, allocations, groups


def normal_expression_coverage(rows, samples, policy):
    """Explicit healthy donor coverage within one measurement namespace."""
    coverage = {tissue: set() for tissue in policy["required_normal_tissues"]}
    for row in rows:
        sample = samples[row["sample_id"]]
        tissue = sample.get("tissue")
        if (
            tissue in coverage
            and row["unit_matches_policy"]
            and sample.get("health") == "healthy"
            and row["upper"] is not None
            and sample.get("donor")
        ):
            coverage[tissue].add(sample["donor"])
    return coverage


def expression_audit(occurrences, samples, allocations, groups, candidates, policy, unannotated):
    """Aggregate disjoint allocations within samples; retain namespace boundaries."""
    contributions = defaultdict(list)
    for allocation in allocations.values():
        targets = {occurrences[o]["sequence_id"] for o in allocation["occurrence_ids"]}
        for sid in targets:
            contributions[allocation["sample_id"], sid].append(allocation)
    audits = []
    for (sample_id, sid), rows in sorted(contributions.items()):
        lower, upper, represented, ambiguous = [], [], set(), False
        for row in rows:
            alternatives = {occurrences[o]["sequence_id"] for o in row["occurrence_ids"]}
            certain = len(alternatives) == 1 and row["scope"] == "transcript"
            known = row["direct"] and row["coding_assignment"] and row.get("lower") is not None
            lower.append(row["lower"] if known and certain else 0.0 if known else None)
            upper.append(row["upper"] if known else None)
            represented.update(row["occurrence_ids"])
            ambiguous |= not certain or not known
        complete = groups[sid] <= represented and sid not in unannotated
        sample = samples[sample_id]
        audits.append(
            {
                "sample_id": sample_id,
                "sequence_id": sid,
                "namespace": {k: sample[k] for k in NAMESPACE_FIELDS},
                "scope": rows[0]["scope"],
                "ambiguous": ambiguous,
                "all_occurrences_represented": complete,
                "lower": sum(v for v in lower if v is not None)
                if any(v is not None for v in lower)
                else None,
                "upper": sum(upper) if complete and all(v is not None for v in upper) else None,
                "allocation_ids": sorted(row["allocation_id"] for row in rows),
                "occurrence_ids": sorted(represented),
                "unit_matches_policy": sample["unit"] == policy["unit"],
            }
        )
    by_group = defaultdict(list)
    for row in audits:
        by_group[row["sequence_id"]].append(row)
    reports = []
    for candidate in sorted(candidates, key=lambda c: c["sequence_id"]):
        sid = candidate["sequence_id"]
        namespaces = defaultdict(list)
        for row in by_group[sid]:
            namespaces[canonical_json(row["namespace"])].append(row)
        namespace_reports = []
        for namespace, rows in sorted(namespaces.items()):
            allowed, normal_evidence = [], []
            coverage = normal_expression_coverage(rows, samples, policy)
            for row in rows:
                sample = samples[row["sample_id"]]
                if not row["unit_matches_policy"] or sample.get("health") != "healthy":
                    continue
                tissue = sample.get("tissue")
                if tissue in policy["allowed_tissues"]:
                    if row["lower"] is not None and row["lower"] >= policy["allowed_min"]:
                        allowed.append(row["sample_id"])
                elif tissue and row["lower"] is not None and row["lower"] > policy["normal_max"]:
                    normal_evidence.append(row["sample_id"])
            missing = sorted(
                t for t, donors in coverage.items() if len(donors) < policy["min_normal_donors"]
            )
            # A measured upper bound above threshold remains unresolved even if
            # its lower bound does not establish positive counterevidence.
            uncertain = any(
                samples[row["sample_id"]].get("health") == "healthy"
                and samples[row["sample_id"]].get("tissue") not in policy["allowed_tissues"]
                and (row["upper"] is None or row["upper"] > policy["normal_max"])
                for row in rows
            )
            namespace_reports.append(
                {
                    "namespace": json.loads(namespace),
                    "allowed_support_samples": sorted(allowed),
                    "normal_counterevidence_samples": sorted(normal_evidence),
                    "normal_donors": {t: sorted(ds) for t, ds in coverage.items()},
                    "missing_normal_tissues": missing,
                    "restricted_in_declared_panel": bool(allowed) and not missing and not uncertain,
                }
            )
        reports.append(
            {
                "sequence_id": sid,
                "policy_id": policy["policy_id"],
                "candidate_admission_basis": candidate["admission_basis"],
                "source_occurrence_ids": sorted(groups[sid]),
                "normal_counterevidence": any(
                    r["normal_counterevidence_samples"] for r in namespace_reports
                ),
                "missing_normal_tissues": sorted(
                    {t for r in namespace_reports for t in r["missing_normal_tissues"]}
                ),
                "isoform_support_unresolved": not any(
                    r["allowed_support_samples"] for r in namespace_reports
                ),
                "unannotated_identical_protein": sid in unannotated,
                "namespaces": namespace_reports,
            }
        )
    return audits, reports
