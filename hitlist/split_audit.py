"""Auditable partition overlap, with explicit limits on claims of independence."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .lineage import (
    _CELLULAR_APPLICABLE_PREPARATIONS,
    ASSAY_PREPARATIONS,
    AXES,
    CELLULAR_DEPENDENCY_AXES,
    STATUSES,
    attach_lineage_applicability,
    canonical_identities,
    lineage_registry_sha256,
    load_lineage,
)

SCHEMA_VERSION = 1
_REUSE_AUDIT_AXES = (
    "observation",
    "experimental_origin",
    "specimen",
    "donor",
    "acquisition",
    "study",
)
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
    if "assay_preparation" in frame.columns:
        prep_series = _strings(frame, "assay_preparation").replace("", "unknown")
        invalid_preps = set(prep_series.unique()) - set(ASSAY_PREPARATIONS)
        if invalid_preps:
            raise ValueError(f"Unknown assay_preparation values: {sorted(invalid_preps)}")
    elif {"assay_method", "evidence_kind", "is_ms_observation", "generated_id"} & set(
        frame.columns
    ):
        prep_cols = [
            c
            for c in (
                "assay_method",
                "evidence_kind",
                "is_ms_observation",
                "generated_id",
                "parent_assay_contexts",
            )
            if c in frame.columns
        ]
        prep_series = (
            attach_lineage_applicability(frame[prep_cols], copy=True)["assay_preparation"]
            .astype("string")
            .fillna("unknown")
        )
    else:
        prep_series = pd.Series("unknown", index=frame.index, dtype="string")

    if n_observations:
        prep_by_key = (
            pd.DataFrame({"key": keys, "prep": prep_series})
            .groupby("key", sort=False)["prep"]
            .agg(lambda s: s.iloc[0] if s.nunique() == 1 else "unknown")
        )
    else:
        prep_by_key = pd.Series(dtype="string")

    by_assay_preparation = {
        prep: int((prep_by_key == prep).sum())
        for prep in ASSAY_PREPARATIONS
        if int((prep_by_key == prep).sum()) > 0
    }
    data = {}

    def collect(axis, values, resolved, *, status_series=None):
        members = {}
        for key, row_values in zip(keys, values):
            for value in row_values:
                members.setdefault(value, set()).add(key)
        # All mapping alternatives of one observation must agree on resolution.
        known = pd.DataFrame({"key": keys, "resolved": resolved}).groupby("key")["resolved"].all()
        n_resolved_obs = int(known.sum())
        n_unresolved_obs = int((~known).sum())
        entry = {
            "members": members,
            "n_resolved_observations": n_resolved_obs,
            "n_unresolved_observations": n_unresolved_obs,
            "coverage_fraction": (
                float(n_resolved_obs / n_observations) if n_observations else 0.0
            ),
            "available": bool(n_observations == 0 or axis in available),
        }
        if axis in AXES:
            resolved_mask = known.to_numpy(dtype=bool)
            n_app_resolved = int(resolved_mask.sum())
            if status_series is not None and n_observations:
                obs_status = (
                    pd.DataFrame({"key": keys, "status": status_series})
                    .groupby("key", sort=False)["status"]
                    .agg(lambda s: s.iloc[0] if s.nunique() == 1 else "ambiguous")
                )
                # Preserve exact status breakdown while honoring resolved_mask
                # (e.g., contributor uncertainty prevents claiming resolved).
                aligned_status = obs_status.reindex(known.index).fillna("unknown")
                aligned_status = aligned_status.where(
                    known | aligned_status.isin({"unknown", "pooled", "ambiguous"}),
                    "unknown",
                )
                status_counts = {
                    status: int((aligned_status == status).sum()) for status in sorted(STATUSES)
                }
            else:
                status_counts = {
                    "ambiguous": 0,
                    "pooled": 0,
                    "resolved": n_resolved_obs,
                    "unknown": n_unresolved_obs,
                }
            if axis not in CELLULAR_DEPENDENCY_AXES:
                n_app_unresolved = int((~resolved_mask).sum())
                n_not_app = 0
                n_app_unknown = 0
            else:
                unresolved_mask = ~resolved_mask
                aligned_preps = (
                    prep_by_key.reindex(known.index).fillna("unknown").to_numpy(dtype=object)
                )
                n_not_app = int((unresolved_mask & (aligned_preps == "purified_mhc")).sum())
                n_app_unresolved = int(
                    (
                        unresolved_mask
                        & np.isin(aligned_preps, list(_CELLULAR_APPLICABLE_PREPARATIONS))
                    ).sum()
                )
                n_app_unknown = int(n_observations - n_app_resolved - n_not_app - n_app_unresolved)
            entry.update(
                {
                    "n_unknown_observations": status_counts["unknown"],
                    "n_pooled_observations": status_counts["pooled"],
                    "n_ambiguous_observations": status_counts["ambiguous"],
                    "n_applicable_observations": n_app_resolved + n_app_unresolved,
                    "n_applicable_resolved_observations": n_app_resolved,
                    "n_applicable_unresolved_observations": n_app_unresolved,
                    "n_not_applicable_observations": n_not_app,
                    "n_applicability_unknown_observations": n_app_unknown,
                }
            )
        data[axis] = entry

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
        raw_statuses = _strings(frame, status).replace("", "unknown")
        known = []
        normalized = []
        effective_statuses = []
        for ids, resolution in zip(values, raw_statuses):
            if any(i not in canonical or entity_kinds[i] != axis for i in ids):
                raise ValueError(f"Unreviewed or wrong-kind {axis} identity")
            normalized.append(tuple(canonical[i] for i in ids))
            is_resolved = resolution == "resolved" and bool(ids)
            known.append(is_resolved)
            if resolution in STATUSES:
                effective_statuses.append(
                    resolution if (resolution != "resolved" or ids) else "unknown"
                )
            else:
                effective_statuses.append("unknown")
        status_series = pd.Series(effective_statuses, index=frame.index, dtype="string")
        if "contributor_resolution" in frame:
            captured_mask = frame.contributor_resolution.eq("captured")
            known = pd.Series(known, index=frame.index) & captured_mask
            status_series = status_series.where(
                captured_mask | status_series.isin({"unknown", "pooled", "ambiguous"}),
                "unknown",
            )
        collect(axis, normalized, known, status_series=status_series)
    return data, {
        "n_rows": len(frame),
        "n_observations": int(n_observations),
        "n_mapping_alternative_rows": int(len(frame) - n_observations),
        "n_missing_observation_ids": int(evidence.eq("").sum()),
        "by_assay_preparation": by_assay_preparation,
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
            overlap_entry = {
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
            if axis in AXES:
                if not (a["available"] and b["available"]):
                    app_status = "unavailable"
                elif common:
                    app_status = "overlap_detected"
                elif (
                    a["n_applicability_unknown_observations"]
                    or b["n_applicability_unknown_observations"]
                ):
                    app_status = "applicability_unknown"
                elif (
                    a["n_applicable_unresolved_observations"]
                    or b["n_applicable_unresolved_observations"]
                ):
                    app_status = "applicable_unresolved"
                elif (a["n_applicable_observations"] + b["n_applicable_observations"]) == 0:
                    app_status = "all_not_applicable"
                else:
                    app_status = "disjoint_where_applicable"
                overlap_entry["applicability_status"] = app_status
            overlaps[axis] = overlap_entry
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
    entity_counts_by_kind = {
        axis: sum(1 for entity in registry.get("entities", []) if entity.get("kind") == axis)
        for axis in AXES
    }
    cross_partition_reuse = []
    for pair in pairs:
        for axis in _REUSE_AUDIT_AXES:
            overlap = pair["overlaps"].get(axis)
            if overlap and overlap["n_shared_values"] > 0:
                cross_partition_reuse.append(
                    {
                        "left": pair["left"],
                        "right": pair["right"],
                        "axis": axis,
                        "n_shared_values": overlap["n_shared_values"],
                        "shared_values": overlap["shared_values"],
                        "n_left_observations": overlap["n_left_observations"],
                        "n_right_observations": overlap["n_right_observations"],
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "verdict": verdict,
        "lineage_registry_sha256": lineage_registry_sha256(registry),
        "lineage_registry_summary": {
            "n_entities": len(registry.get("entities", [])),
            "n_aliases": len(registry.get("aliases", [])),
            "n_contexts": len(registry.get("contexts", [])),
            "entity_counts_by_kind": entity_counts_by_kind,
        },
        "partitions": counts,
        "pairs": pairs,
        "cross_partition_reuse": cross_partition_reuse,
        "scope": "Supplied partitions only; no claim about historical training weights.",
    }
