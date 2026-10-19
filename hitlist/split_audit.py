"""Auditable partition overlap, with explicit limits on claims of independence."""

from __future__ import annotations

import itertools
import json

import pandas as pd

from .lineage import AXES, canonical_identities, load_lineage

SCHEMA_VERSION = 1
POLICIES = {
    "report_only": (),
    "peptide_disjoint": ("peptide",),
    "pmhc_disjoint": ("pmhc",),
    "independent_experiments": ("observation", "experimental_origin"),
    "specimen_disjoint": ("specimen",),
    "donor_disjoint": ("donor",),
    "whole_study": ("study",),
}


def _strings(frame, column):
    return frame.get(column, pd.Series("", index=frame.index)).astype("string").fillna("")


def _identity_values(value):
    if isinstance(value, str):
        value = json.loads(value or "[]")
    if not isinstance(value, (list, tuple)) or any(not isinstance(i, str) for i in value):
        raise ValueError("Biological identity columns must contain JSON arrays of identifiers")
    return value


def _partition_axes(frame, registry):
    canonical = canonical_identities(registry)
    entity_kinds = {e["identity_id"]: e["kind"] for e in registry["entities"]}
    evidence = _strings(frame, "evidence_row_id")
    # Count mapping alternatives once per observation. Missing IDs remain
    # separate rows rather than collapsing all unknown observations together.
    keys = evidence.where(
        evidence.ne(""),
        pd.Series([f"unresolved-row:{i}" for i in range(len(frame))], index=frame.index),
    )
    n_observations = keys.nunique()
    data = {}

    def collect(axis, values, resolved):
        members = {}
        for key, row_values in zip(keys, values):
            for value in row_values:
                members.setdefault(value, set()).add(key)
        # All mapping alternatives of one observation must agree on resolution.
        known = pd.DataFrame({"key": keys, "resolved": resolved}).groupby("key")["resolved"].all()
        data[axis] = {
            "members": members,
            "n_unresolved_observations": int((~known).sum()),
            "available": bool(n_observations == 0 or axis in available),
        }

    available = set()
    peptide = _strings(frame, "peptide")
    if "peptide" in frame:
        available.add("peptide")
    collect("peptide", [(p,) if p else () for p in peptide], peptide.ne(""))
    restriction = _strings(frame, "mhc_restriction")
    if {"peptide", "mhc_restriction"} <= set(frame.columns):
        available.add("pmhc")
    # This is exact normalized restriction equality, not inferred equivalence
    # between an unresolved allele set and its individual possible presenters.
    from .curation import classify_allele_resolution

    exact = {
        value: classify_allele_resolution(value) == "four_digit" for value in restriction.unique()
    }
    complete = peptide.ne("") & restriction.map(exact).astype(bool)
    if "has_peptide_level_allele" in frame:
        complete &= frame["has_peptide_level_allele"].fillna(False).astype(bool)
    collect(
        "pmhc",
        [((p, a),) if ok else () for p, a, ok in zip(peptide, restriction, complete)],
        complete,
    )
    # Export-relative row counters are not reusable evidence identities.
    stable = evidence.ne("") & ~evidence.str.match(
        r"^(ms|binding):(?:attributed:v1:)?row:\d+(?:$|\|)"
    )
    if "assay_iri" in frame:
        stable &= _strings(frame, "assay_iri").ne("")
    if "evidence_row_id" in frame:
        available.add("observation")
    canonical_evidence = evidence.str.replace(
        r"(?<=:)https?://(?:www\.iedb\.org|iedb\.org|cedar\.iedb\.org)(?=/assay/)", "", regex=True
    )
    collect(
        "observation", [(v,) if ok else () for v, ok in zip(canonical_evidence, stable)], stable
    )
    pmids = pd.to_numeric(_strings(frame, "pmid"), errors="coerce").astype("Int64")
    if "pmid" in frame:
        available.add("study")
    if "contributor_pmids" in frame:
        papers = [_identity_values(value) for value in _strings(frame, "contributor_pmids")]
        values = [set(ids) | ({str(p)} if pd.notna(p) else set()) for ids, p in zip(papers, pmids)]
        resolved = _strings(frame, "contributor_study_status").eq("resolved") & pmids.notna()
        collect("study", values, resolved)
    else:
        collect("study", [(str(p),) if pd.notna(p) else () for p in pmids], pmids.notna())
    for axis in AXES:
        column, status = f"{axis}_ids", f"{axis}_status"
        if {column, status} <= set(frame.columns):
            available.add(axis)
        values = [_identity_values(value) for value in _strings(frame, column)]
        known = []
        normalized = []
        for ids, resolution in zip(values, _strings(frame, status)):
            if any(i not in canonical or entity_kinds[i] != axis for i in ids):
                raise ValueError(f"Unreviewed or wrong-kind {axis} identity")
            normalized.append(tuple(canonical[i] for i in ids))
            known.append(resolution == "resolved" and bool(ids))
        if "contributor_resolution" in frame:
            known = pd.Series(known, index=frame.index) & frame.contributor_resolution.eq(
                "captured"
            )
        collect(axis, normalized, known)
    return data, {
        "n_rows": len(frame),
        "n_observations": int(n_observations),
        "n_mapping_alternative_rows": int(len(frame) - n_observations),
        "n_missing_observation_ids": int(evidence.eq("").sum()),
    }


def audit_splits(
    partitions: dict[str, pd.DataFrame], *, policy="report_only", registry=None
) -> dict:
    """Compare named partitions without constructing or changing their splits.

    A no-overlap result with unresolved required identities is inconclusive.
    ``whole_study`` is an explicitly conservative PMID grouping policy. pMHC
    overlap is exact equality of the normalized exported restriction, with
    unresolved presenters reported separately. Protein mapping alternatives
    share their observation ID and therefore cannot evade observation overlap.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown split policy: {policy}")
    if len(partitions) < 2 or any(not isinstance(name, str) or not name for name in partitions):
        raise ValueError("Provide at least two named partitions")
    registry = load_lineage() if registry is None else registry
    axes, counts = {}, {}
    for name, frame in partitions.items():
        axes[name], counts[name] = _partition_axes(frame.reset_index(drop=True), registry)
        counts[name]["coverage"] = {
            axis: {k: v for k, v in value.items() if k != "members"}
            for axis, value in axes[name].items()
        }
    pairs = []
    for left, right in itertools.combinations(sorted(partitions), 2):
        overlaps = {}
        for axis in axes[left]:
            a, b = axes[left][axis], axes[right][axis]
            common = sorted(a["members"].keys() & b["members"].keys())
            overlaps[axis] = {
                "n_shared_values": len(common),
                "shared_values": common,
                "n_left_observations": len(set().union(*(a["members"][v] for v in common))),
                "n_right_observations": len(set().union(*(b["members"][v] for v in common))),
                "coverage_complete": not (
                    a["n_unresolved_observations"] or b["n_unresolved_observations"]
                )
                and a["available"]
                and b["available"],
            }
        required = POLICIES[policy]
        if any(overlaps[axis]["n_shared_values"] for axis in required):
            verdict = "fail"
        elif any(not overlaps[axis]["coverage_complete"] for axis in required):
            verdict = "inconclusive"
        else:
            verdict = "reported" if policy == "report_only" else "pass"
        pairs.append({"left": left, "right": right, "verdict": verdict, "overlaps": overlaps})
    verdicts = {p["verdict"] for p in pairs}
    verdict = next(v for v in ("fail", "inconclusive", "pass", "reported") if v in verdicts)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "verdict": verdict,
        "partitions": counts,
        "pairs": pairs,
        "scope": "Supplied partitions only; no claim about historical training weights.",
    }
