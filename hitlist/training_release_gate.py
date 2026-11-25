"""Frozen training-release freshness, diff, and lineage-holdout gates (#678, #680).

Coordinates Hitlist's source-preserving evidence modules, reviewed lineage
registry, and reproducible training bundles into auditable release-readiness
checks for MHCflurry 3.x and Presto without inventing biological identities,
silently overwriting historical frozen bundles, or conflating annotation-only
curation refreshes with upstream source acquisitions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from .lineage import (
    AXES,
    LINEAGE_COLUMNS,
    attach_lineage,
    canonical_identities,
    lineage_path,
    lineage_registry_sha256,
    load_lineage,
)
from .provenance import file_digest
from .split_audit import POLICIES, audit_splits

SCHEMA_VERSION = 1
_CONTRACT_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "training_release_gate_contract.json"
)
_EVIDENCE_REVISIONS_DIR = Path(__file__).parent / "data" / "evidence_revisions"
_DEPENDENCY_PACKAGES = (
    "pandas",
    "pyarrow",
    "mhcgnomes",
    "numpy",
    "PyYAML",
    "datacache",
    "oncoref",
)


def training_release_gate_contract() -> dict[str, Any]:
    """Return a copy of the pinned training-release & lineage-gate contract (#678, #680)."""
    data = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _DEPENDENCY_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def _normalize_cell(value: Any) -> tuple[bool, str]:
    """Return ``(is_missing, canonical_text)`` for deterministic field-diff equality."""
    if value is None:
        return True, ""
    if isinstance(value, float) and pd.isna(value):
        return True, ""
    try:
        if pd.isna(value):
            return True, ""
    except (TypeError, ValueError):
        pass
    return False, str(value)


def build_training_release_freshness_receipt(
    *,
    release_tag: str,
    upstream_snapshots: dict[str, dict[str, Any]] | None = None,
    bundle_dir: str | Path | None = None,
    include_local_cache_status: bool = False,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a deterministic freshness and provenance receipt for a training release (#678).

    Enforces non-replacement of historical frozen releases (such as
    ``human-cohort-corrected-v1``), records exact software and dependency
    versions, pins the live ``specimen_lineage.yaml`` SHA-256 and integrated
    evidence-contract digests, and captures upstream raw-snapshot metadata.
    """
    from .version import __version__

    if not isinstance(release_tag, str) or not release_tag.strip():
        raise ValueError("release_tag must be a non-empty string")
    release_tag = release_tag.strip()

    contract = training_release_gate_contract()
    baseline = contract["historical_baseline"]
    if release_tag in {baseline["release_tag"], baseline["corpus_release_tag"]}:
        raise ValueError(
            f"Refreshed training releases must not reuse historical release tag {release_tag!r}; "
            "assign a new versioned release tag so frozen inputs remain immutable."
        )

    validated_registry = load_lineage() if registry is None else registry
    reg_sha256 = lineage_registry_sha256(validated_registry)
    reg_file = lineage_path()
    reg_file_digest = file_digest(reg_file) if reg_file.exists() else None

    bundle_manifest: dict[str, Any] | None = None
    extracted_snapshots: dict[str, dict[str, Any]] = {}
    cache_freshness: dict[str, Any] = {"scope": "uninspected"}
    if bundle_dir is not None:
        from .training_bundle import verify_training_bundle

        bundle_path = Path(bundle_dir)
        bundle_manifest = verify_training_bundle(bundle_path)
        manifest_digest = file_digest(bundle_path / "manifest.json")
        raw_sources = (
            bundle_manifest.get("inputs", {})
            .get("build_metadata", {})
            .get("provenance", {})
            .get("sources", {})
        )
        for src_name, src_meta in sorted(raw_sources.items()):
            extracted_snapshots[src_name] = dict(src_meta)
        cache_freshness = dict(bundle_manifest.get("input_validation", {"scope": "legacy_bundle"}))
        bundle_ref: dict[str, Any] | None = {
            "manifest_sha256": manifest_digest["sha256"],
            "manifest_size_bytes": manifest_digest["size_bytes"],
            "n_rows": int(bundle_manifest["n_rows"]),
            "n_observations": int(bundle_manifest["coverage"]["n_observations"]),
            "hitlist_version": bundle_manifest.get("hitlist_version"),
        }
    else:
        bundle_ref = None

    if upstream_snapshots is not None:
        for src_name, src_meta in sorted(upstream_snapshots.items()):
            if not isinstance(src_meta, dict):
                raise ValueError(f"upstream_snapshots[{src_name!r}] must be a dict")
            extracted_snapshots[src_name] = dict(src_meta)

    if include_local_cache_status and bundle_dir is None:
        from .observations import observations_cache_is_current

        try:
            obs_current = observations_cache_is_current()
            obs_status = {True: "current", False: "stale", None: "unverifiable"}[obs_current]
        except (OSError, ValueError):
            obs_status = "invalid"
        cache_freshness = {
            "observations": obs_status,
            "scope": "local_cache_inspection",
        }

    contract_digests: dict[str, dict[str, Any]] = {}
    for entry in contract["integrated_evidence_modules"]:
        contract_file = entry.get("contract_file")
        if contract_file:
            path = _EVIDENCE_REVISIONS_DIR / contract_file
            contract_digests[contract_file] = {
                "issue": entry["issue"],
                "module": entry["module"],
                "status": entry["status"],
                **(file_digest(path) if path.exists() else {"missing": True}),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "release_tag": release_tag,
        "hitlist_version": __version__,
        "dependency_versions": _package_versions(),
        "historical_baseline": baseline,
        "replaces_historical_baseline": False,
        "claims_predictive_improvement": False,
        "lineage_registry_sha256": reg_sha256,
        "lineage_registry_file": reg_file_digest,
        "upstream_snapshots": extracted_snapshots,
        "n_upstream_snapshots": len(extracted_snapshots),
        "cache_freshness": cache_freshness,
        "verified_bundle": bundle_ref,
        "evidence_contracts": contract_digests,
        "n_integrated_evidence_modules": len(contract["integrated_evidence_modules"]),
    }


def audit_training_release_diff(
    previous_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    id_column: str = "evidence_row_id",
    source_columns: set[str] | tuple[str, ...] | None = None,
    annotation_columns: set[str] | tuple[str, ...] | None = None,
    measurement_invariant_columns: set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Audit row and field diffs between a historical frozen export and a refreshed candidate (#678).

    Distinguishes upstream source-acquisition updates from annotation-only
    curation changes while checking whether original observation IDs and
    measured quantitative/endpoint/censoring fields remain invariant.
    """
    if id_column not in previous_df.columns:
        raise ValueError(f"previous_df missing required id_column {id_column!r}")
    if id_column not in current_df.columns:
        raise ValueError(f"current_df missing required id_column {id_column!r}")

    prev_ids = previous_df[id_column].astype("string").fillna("")
    curr_ids = current_df[id_column].astype("string").fillna("")
    if prev_ids.eq("").any() or curr_ids.eq("").any():
        raise ValueError(f"Empty {id_column!r} values are not permitted in release diff audits")
    if prev_ids.duplicated().any():
        raise ValueError(f"Duplicate {id_column!r} values in previous_df")
    if curr_ids.duplicated().any():
        raise ValueError(f"Duplicate {id_column!r} values in current_df")

    contract = training_release_gate_contract()
    col_cfg = contract["diff_column_classification"]
    source_set = (
        set(source_columns)
        if source_columns is not None
        else set(col_cfg["source_acquisition_columns"])
    )
    annotation_set = (
        set(annotation_columns)
        if annotation_columns is not None
        else set(col_cfg["annotation_curation_columns"])
    )
    invariant_set = (
        set(measurement_invariant_columns)
        if measurement_invariant_columns is not None
        else set(col_cfg["measurement_invariant_columns"])
    )

    prev_indexed = previous_df.assign(_diff_id=prev_ids).set_index("_diff_id")
    curr_indexed = current_df.assign(_diff_id=curr_ids).set_index("_diff_id")

    prev_id_set = set(prev_indexed.index)
    curr_id_set = set(curr_indexed.index)
    retained_ids = sorted(prev_id_set & curr_id_set)
    added_ids = sorted(curr_id_set - prev_id_set)
    removed_ids = sorted(prev_id_set - curr_id_set)

    shared_columns = sorted((set(previous_df.columns) & set(current_df.columns)) - {id_column})
    added_columns = sorted(set(current_df.columns) - set(previous_df.columns))
    removed_columns = sorted(set(previous_df.columns) - set(current_df.columns))

    def _col_category(col: str) -> str:
        if col in source_set:
            return "source_acquisition"
        if col in annotation_set:
            return "annotation_curation"
        return "unclassified"

    prev_retained = prev_indexed.loc[retained_ids] if retained_ids else prev_indexed.iloc[0:0]
    curr_retained = curr_indexed.loc[retained_ids] if retained_ids else curr_indexed.iloc[0:0]

    changed_fields: dict[str, dict[str, Any]] = {}
    row_source_changed = pd.Series(False, index=retained_ids, dtype=bool)
    row_annotation_changed = pd.Series(False, index=retained_ids, dtype=bool)
    row_unclassified_changed = pd.Series(False, index=retained_ids, dtype=bool)
    invariant_violations: dict[str, int] = {}

    for col in shared_columns:
        left_vals = [_normalize_cell(v) for v in prev_retained[col].tolist()]
        right_vals = [_normalize_cell(v) for v in curr_retained[col].tolist()]
        diff_mask = [a != b for a, b in zip(left_vals, right_vals)]
        n_changed = int(sum(diff_mask))
        if n_changed > 0:
            category = _col_category(col)
            changed_fields[col] = {
                "category": category,
                "n_changed_rows": n_changed,
            }
            mask_series = pd.Series(diff_mask, index=retained_ids, dtype=bool)
            if category == "source_acquisition":
                row_source_changed |= mask_series
            elif category == "annotation_curation":
                row_annotation_changed |= mask_series
            else:
                row_unclassified_changed |= mask_series
            if col in invariant_set:
                invariant_violations[col] = n_changed

    n_source_only = int(
        (row_source_changed & ~row_annotation_changed & ~row_unclassified_changed).sum()
    )
    n_annotation_only = int(
        (row_annotation_changed & ~row_source_changed & ~row_unclassified_changed).sum()
    )
    n_mixed = int((row_source_changed & row_annotation_changed).sum())
    n_unclassified_only = int(
        (row_unclassified_changed & ~row_source_changed & ~row_annotation_changed).sum()
    )
    any_changed = row_source_changed | row_annotation_changed | row_unclassified_changed
    n_changed_rows = int(any_changed.sum())
    n_unchanged_rows = int(len(retained_ids) - n_changed_rows)

    has_source_changes = bool(added_ids or removed_ids or row_source_changed.any())
    has_annotation_changes = bool(
        row_annotation_changed.any()
        or any(_col_category(col) == "annotation_curation" for col in added_columns)
    )
    if (
        not has_source_changes
        and not has_annotation_changes
        and not row_unclassified_changed.any()
        and not added_columns
        and not removed_columns
    ):
        change_classification = "identical"
    elif has_source_changes and has_annotation_changes:
        change_classification = "mixed_source_and_annotation_refresh"
    elif has_source_changes:
        change_classification = "source_acquisition_refresh"
    elif has_annotation_changes:
        change_classification = "annotation_only_refresh"
    else:
        change_classification = "unclassified_field_refresh"

    return {
        "schema_version": SCHEMA_VERSION,
        "id_column": id_column,
        "n_previous_rows": len(previous_df),
        "n_current_rows": len(current_df),
        "n_retained_ids": len(retained_ids),
        "n_added_ids": len(added_ids),
        "n_removed_ids": len(removed_ids),
        "original_ids_preserved": len(removed_ids) == 0,
        "n_unchanged_rows": n_unchanged_rows,
        "n_changed_rows": n_changed_rows,
        "n_source_acquisition_updated_rows": n_source_only,
        "n_annotation_only_changed_rows": n_annotation_only,
        "n_mixed_changed_rows": n_mixed,
        "n_unclassified_changed_rows": n_unclassified_only,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "changed_fields": changed_fields,
        "measurement_invariants_preserved": len(invariant_violations) == 0,
        "measurement_invariant_violations": invariant_violations,
        "change_classification": change_classification,
    }


def _parse_json_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text or text == "nan":
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array of identity strings, got {type(parsed).__name__}")
    return [str(v) for v in parsed if str(v)]


def assign_lineage_grouped_splits(
    frame: pd.DataFrame,
    *,
    fractions: dict[str, float] | None = None,
    group_axes: tuple[str, ...] = ("experimental_origin", "specimen", "donor"),
    include_contributor_studies: bool = True,
    policy: str = "independent_experiments",
    registry: dict[str, Any] | None = None,
    seed: str = "hitlist-lineage-holdout-v1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign candidate holdout splits keeping original datasets and reanalyses together (#680).

    Observations sharing any reviewed canonical identity on ``group_axes`` (such
    as Abelin/Sarkizova ``origin:MSV000080527:*``, Bassani-Sternberg/Liepe
    ``origin:pmid:25576301:fibroblast_immunopeptidome``, or dual-IP donor/specimen
    entities) are placed into the same connected component and assigned to the
    same partition.

    Observations whose biological lineage is ``unknown``, ``pooled``, or
    ``ambiguous`` are NEVER given synthetic biological identities; their native
    statuses remain intact so the attached :func:`audit_splits` verdict stays
    ``inconclusive`` whenever required biological identities cannot be recovered.
    """
    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    if len(fractions) < 2:
        raise ValueError("Provide at least two target partitions in fractions")
    if any(not isinstance(k, str) or not k for k in fractions):
        raise ValueError("Partition names must be non-empty strings")
    if any(float(v) <= 0 for v in fractions.values()):
        raise ValueError("All partition fractions must be positive")
    invalid_axes = set(group_axes) - set(AXES)
    if invalid_axes:
        raise ValueError(f"Unknown lineage group_axes: {sorted(invalid_axes)}")
    if policy not in POLICIES:
        raise ValueError(f"Unknown split policy: {policy}")

    validated_registry = load_lineage() if registry is None else registry
    canonical = canonical_identities(validated_registry)
    reg_sha256 = lineage_registry_sha256(validated_registry)

    work = frame.copy().reset_index(drop=True)
    if not (set(LINEAGE_COLUMNS) <= set(work.columns)):
        work = attach_lineage(work, registry=validated_registry, copy=False)

    n_rows = len(work)
    if n_rows == 0:
        work["split_partition"] = pd.Series(dtype="string")
        work["lineage_component_id"] = pd.Series(dtype="string")
        work["lineage_group_basis"] = pd.Series(dtype="string")
        return work, {
            "schema_version": SCHEMA_VERSION,
            "lineage_registry_sha256": reg_sha256,
            "group_axes": list(group_axes),
            "include_contributor_studies": include_contributor_studies,
            "n_rows": 0,
            "n_observations": 0,
            "n_components": 0,
            "n_reviewed_lineage_observations": 0,
            "n_unresolved_fallback_observations": 0,
            "partition_observation_counts": dict.fromkeys(fractions, 0),
            "audit": None,
        }

    # Build observation keys so mapping alternatives of one observation never split.
    if "evidence_row_id" in work.columns:
        raw_ev = work["evidence_row_id"].astype("string").fillna("")
        obs_keys = [ev if ev else f"unresolved-row:{idx}" for idx, ev in enumerate(raw_ev.tolist())]
    else:
        obs_keys = [f"unresolved-row:{idx}" for idx in range(n_rows)]

    # Union-find over rows
    parent = list(range(n_rows))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    token_owner: dict[str, int] = {}
    row_has_reviewed = [False] * n_rows

    def bind_token(token: str, idx: int) -> None:
        prev = token_owner.get(token)
        if prev is None:
            token_owner[token] = idx
        else:
            union(prev, idx)

    # Always keep mapping alternatives of one observation together.
    for idx, obs_key in enumerate(obs_keys):
        bind_token(f"obs:{obs_key}", idx)

    # Link by reviewed canonical lineage entities (including pooled membership so
    # pooled rows never leak across a known member's partition).
    for axis in group_axes:
        ids_col = f"{axis}_ids"
        if ids_col not in work.columns:
            continue
        for idx, raw_ids in enumerate(work[ids_col].tolist()):
            parsed_ids = _parse_json_id_list(raw_ids)
            for entity_id in parsed_ids:
                canon_id = canonical.get(entity_id, entity_id)
                bind_token(f"{axis}:{canon_id}", idx)
                row_has_reviewed[idx] = True

    # Link original/reanalysis contributors via shared provenance_id or contributor_pmids.
    if include_contributor_studies:
        if "provenance_id" in work.columns:
            for idx, prov_id in enumerate(
                work["provenance_id"].astype("string").fillna("").tolist()
            ):
                if prov_id:
                    bind_token(f"prov:{prov_id}", idx)
        if "contributor_pmids" in work.columns:
            for idx, raw_pmids in enumerate(work["contributor_pmids"].tolist()):
                for pmid_val in _parse_json_id_list(raw_pmids):
                    bind_token(f"contrib_pmid:{pmid_val}", idx)

    # Group unresolved fallback rows by PMID (without fabricating biological IDs).
    pmids_str = (
        pd.to_numeric(work["pmid"], errors="coerce").astype("Int64").astype("string").fillna("")
        if "pmid" in work.columns
        else pd.Series("", index=work.index, dtype="string")
    )
    for idx in range(n_rows):
        if not row_has_reviewed[idx]:
            pmid_val = pmids_str.iloc[idx]
            if pmid_val:
                bind_token(f"fallback_pmid:{pmid_val}", idx)

    # Collect connected components
    components: dict[int, list[int]] = {}
    for idx in range(n_rows):
        components.setdefault(find(idx), []).append(idx)

    # Build deterministic component metadata sorted by hash of stable content key
    comp_entries = []
    for root_idx, member_indices in components.items():
        comp_obs_keys = sorted({obs_keys[i] for i in member_indices})
        comp_reviewed = any(row_has_reviewed[i] for i in member_indices)
        # Prefer canonical entity tokens for component key stability across row shuffles
        comp_tokens: set[str] = set()
        for axis in group_axes:
            ids_col = f"{axis}_ids"
            if ids_col in work.columns:
                for i in member_indices:
                    for eid in _parse_json_id_list(work[ids_col].iloc[i]):
                        comp_tokens.add(f"{axis}:{canonical.get(eid, eid)}")
        if comp_tokens:
            stable_key = "|".join(sorted(comp_tokens))
        else:
            stable_key = comp_obs_keys[0]
        comp_hash = hashlib.sha256(f"{seed}:{stable_key}".encode()).hexdigest()
        comp_id = f"comp:{comp_hash[:16]}"
        comp_entries.append(
            {
                "root": root_idx,
                "component_id": comp_id,
                "hash": comp_hash,
                "indices": member_indices,
                "n_observations": len(comp_obs_keys),
                "has_reviewed_lineage": comp_reviewed,
            }
        )

    # Sort deterministically by hash, then assign greedily to maximize deficit
    # relative to target observation counts (preserving exact component atomicity).
    total_obs = sum(c["n_observations"] for c in comp_entries)
    total_weight = sum(float(v) for v in fractions.values())
    part_names = list(fractions.keys())
    targets = {name: total_obs * (float(fractions[name]) / total_weight) for name in part_names}
    assigned_obs = dict.fromkeys(part_names, 0)

    # Place largest components first within hash-tiebreak for balanced fills,
    # while ensuring every partition gets at least one component when possible.
    comp_entries.sort(key=lambda c: (-c["n_observations"], c["hash"]))

    split_col = [""] * n_rows
    comp_id_col = [""] * n_rows
    basis_col = [""] * n_rows
    n_reviewed_obs = 0
    n_fallback_obs = 0

    for entry in comp_entries:
        # Pick partition with largest normalized remaining deficit (target - current) / target
        best_part = max(
            part_names,
            key=lambda name: (
                (targets[name] - assigned_obs[name]) / max(targets[name], 1e-9),
                fractions[name],
                name,
            ),
        )
        assigned_obs[best_part] += entry["n_observations"]
        basis = (
            "reviewed_lineage_component"
            if entry["has_reviewed_lineage"]
            else "unresolved_fallback_pmid"
        )
        if entry["has_reviewed_lineage"]:
            n_reviewed_obs += entry["n_observations"]
        else:
            n_fallback_obs += entry["n_observations"]
        for i in entry["indices"]:
            split_col[i] = best_part
            comp_id_col[i] = entry["component_id"]
            basis_col[i] = basis

    work["split_partition"] = pd.Series(split_col, index=work.index, dtype="string")
    work["lineage_component_id"] = pd.Series(comp_id_col, index=work.index, dtype="string")
    work["lineage_group_basis"] = pd.Series(basis_col, index=work.index, dtype="string")

    non_empty_partitions = {
        name: work.loc[work["split_partition"] == name]
        for name in part_names
        if (work["split_partition"] == name).any()
    }
    audit_report = (
        audit_splits(non_empty_partitions, policy=policy, registry=validated_registry)
        if len(non_empty_partitions) >= 2
        else None
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "lineage_registry_sha256": reg_sha256,
        "group_axes": list(group_axes),
        "include_contributor_studies": include_contributor_studies,
        "n_rows": n_rows,
        "n_observations": int(total_obs),
        "n_components": len(comp_entries),
        "n_reviewed_lineage_observations": int(n_reviewed_obs),
        "n_unresolved_fallback_observations": int(n_fallback_obs),
        "partition_observation_counts": assigned_obs,
        "audit": audit_report,
    }
    return work, summary


def evaluate_training_release_gate(
    partitions: dict[str, pd.DataFrame],
    *,
    policy: str = "independent_experiments",
    registry: dict[str, Any] | None = None,
    freshness_receipt: dict[str, Any] | None = None,
    diff_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate combined lineage-holdout, freshness, and diff readiness gates (#678, #680).

    Never upgrades an ``inconclusive`` lineage audit to ``qualified``.
    """
    audit_report = audit_splits(partitions, policy=policy, registry=registry)
    blocking_reasons: list[str] = []

    lineage_verdict = audit_report["verdict"]
    if lineage_verdict == "fail":
        blocking_reasons.append(f"lineage_overlap_detected:{policy}")
    elif lineage_verdict == "inconclusive":
        blocking_reasons.append(f"lineage_inconclusive:{policy}")

    if freshness_receipt is not None:
        if freshness_receipt.get("replaces_historical_baseline", False):
            blocking_reasons.append("historical_baseline_replacement_forbidden")
        if (
            freshness_receipt.get("lineage_registry_sha256")
            != audit_report["lineage_registry_sha256"]
        ):
            blocking_reasons.append("lineage_registry_hash_mismatch_with_receipt")
        obs_fresh = freshness_receipt.get("cache_freshness", {}).get("observations")
        if obs_fresh in {"stale", "invalid"}:
            blocking_reasons.append(f"stale_observation_index:{obs_fresh}")

    if diff_audit is not None:
        if not diff_audit.get("original_ids_preserved", True):
            blocking_reasons.append("original_observation_ids_removed")
        if not diff_audit.get("measurement_invariants_preserved", True):
            blocking_reasons.append("measurement_invariants_mutated")

    # Aggregate observation-weighted cohort-wide lineage coverage across partitions
    total_obs = sum(part["n_observations"] for part in audit_report["partitions"].values())
    cohort_coverage: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        n_resolved = sum(
            part["coverage"][axis]["n_resolved_observations"]
            for part in audit_report["partitions"].values()
        )
        n_unknown = sum(
            part["coverage"][axis]["n_unknown_observations"]
            for part in audit_report["partitions"].values()
        )
        n_pooled = sum(
            part["coverage"][axis]["n_pooled_observations"]
            for part in audit_report["partitions"].values()
        )
        n_ambiguous = sum(
            part["coverage"][axis]["n_ambiguous_observations"]
            for part in audit_report["partitions"].values()
        )
        cohort_coverage[axis] = {
            "n_observations": int(total_obs),
            "n_resolved_observations": int(n_resolved),
            "n_unknown_observations": int(n_unknown),
            "n_pooled_observations": int(n_pooled),
            "n_ambiguous_observations": int(n_ambiguous),
            "coverage_fraction": float(n_resolved / total_obs) if total_obs else 0.0,
        }

    if lineage_verdict == "fail":
        overall_gate_status = "blocked_overlap"
    elif any(
        r.startswith(
            (
                "stale_observation_index:",
                "historical_baseline_replacement_forbidden",
                "lineage_registry_hash_mismatch",
                "original_observation_ids_removed",
                "measurement_invariants_mutated",
            )
        )
        for r in blocking_reasons
    ):
        overall_gate_status = "blocked_release_integrity"
    elif lineage_verdict == "inconclusive":
        overall_gate_status = "inconclusive_lineage"
    elif lineage_verdict == "reported":
        overall_gate_status = "reported_only"
    else:
        overall_gate_status = "qualified"

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "lineage_verdict": lineage_verdict,
        "overall_gate_status": overall_gate_status,
        "lineage_registry_sha256": audit_report["lineage_registry_sha256"],
        "n_total_observations": int(total_obs),
        "cohort_coverage": cohort_coverage,
        "cross_partition_reuse": audit_report["cross_partition_reuse"],
        "blocking_reasons": blocking_reasons,
        "split_audit": audit_report,
        "freshness_receipt": freshness_receipt,
        "diff_audit": diff_audit,
        "claims_predictive_improvement": False,
    }
