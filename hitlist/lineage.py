"""Reviewed biological identities, separate from publication and assay IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .curation_yaml import load_curation_yaml

SCHEMA_VERSION = 1
AXES = ("experimental_origin", "specimen", "donor", "acquisition")
STATUSES = {"resolved", "unknown", "pooled", "ambiguous"}
LINEAGE_COLUMNS = ["lineage_context_id", "lineage_status"] + [
    column for axis in AXES for column in (f"{axis}_ids", f"{axis}_status")
]
_RESOLVED_ARMS = {
    "curated_sample_label",
    "allele_exact",
    "single_sample_pmid",
    "elution_conditions",
    "discriminated",
    "serotype_expansion",
}


def lineage_path() -> Path:
    return Path(__file__).parent / "data" / "specimen_lineage.yaml"


def validate_lineage(registry: dict) -> dict:
    """Reject dangling references or unsupported equivalence claims."""
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported lineage schema")
    entities = {}
    names = set()
    for entity in registry.get("entities", []):
        identity = entity["identity_id"]
        name = (entity["kind"], entity["namespace"], entity["reported_id"])
        if identity in entities or name in names or entity["kind"] not in AXES:
            raise ValueError(f"Duplicate or invalid lineage identity: {identity}")
        if not all(
            entity.get(k)
            for k in (
                "identity_id",
                "namespace",
                "reported_id",
                "evidence_reference",
                "evidence_note",
            )
        ):
            raise ValueError(f"Lineage identity needs source evidence: {identity}")
        entities[identity] = entity
        names.add(name)
    aliases = set()
    for alias in registry.get("aliases", []):
        identity = alias["identity_id"]
        target = alias["canonical_id"]
        if (
            identity not in entities
            or target not in entities
            or identity == target
            or identity in aliases
            or target in aliases
            or entities[identity]["kind"] != entities[target]["kind"]
            or not alias.get("evidence_reference")
            or not alias.get("evidence_note")
        ):
            raise ValueError(f"Invalid reviewed lineage alias: {identity}")
        aliases.add(identity)
    targets = {a["canonical_id"] for a in registry.get("aliases", [])}
    if aliases & targets:
        raise ValueError(
            "Alias chains are not supported; reference the canonical identity directly"
        )
    contexts = set()
    for context in registry.get("contexts", []):
        key = (int(context["pmid"]), context["condition_id"])
        if key in contexts or not key[1] or not context.get("evidence_reference"):
            raise ValueError(f"Duplicate or unsupported lineage context: {key}")
        contexts.add(key)
        for axis in AXES:
            value = context.get(axis, {"status": "unknown", "ids": []})
            status, ids = value["status"], value.get("ids", [])
            if (
                status not in STATUSES
                or not isinstance(ids, list)
                or len(ids) != len(set(ids))
                or (status == "unknown" and ids)
                or (status == "resolved" and not ids)
            ):
                raise ValueError(f"Invalid {axis} resolution in {key}")
            if any(i not in entities or entities[i]["kind"] != axis for i in ids):
                raise ValueError(f"Invalid {axis} identity in {key}")
    return registry


def load_lineage(path=None) -> dict:
    """Read the versioned registry afresh so curation edits cannot stay cached."""
    return validate_lineage(load_curation_yaml(Path(path or lineage_path())))


def canonical_identities(registry: dict) -> dict[str, str]:
    validate_lineage(registry)
    result = {e["identity_id"]: e["identity_id"] for e in registry["entities"]}
    result.update({a["identity_id"]: a["canonical_id"] for a in registry.get("aliases", [])})
    return result


def attach_lineage(frame: pd.DataFrame, registry=None, *, copy=True) -> pd.DataFrame:
    """Attach reviewed study/arm contexts; counting fallbacks are never IDs.

    Arm attribution evidence remains in ``sample_attribution``. Even a reviewed
    context cannot resolve an observation whose arm is ambiguous or unassigned.
    Missing contexts are explicit unknowns, including in binding-only exports.
    """
    registry = load_lineage() if registry is None else validate_lineage(registry)
    aliases = canonical_identities(registry)
    rows = []
    for context in registry.get("contexts", []):
        row = {
            "_lineage_pmid": str(context["pmid"]),
            "_lineage_arm": context["condition_id"],
            "lineage_context_id": f"pmid:{context['pmid']}:arm:{context['condition_id']}",
            "lineage_status": "reviewed_context",
        }
        for axis in AXES:
            value = context.get(axis, {"status": "unknown", "ids": []})
            row[f"{axis}_ids"] = json.dumps(sorted({aliases[i] for i in value.get("ids", [])}))
            row[f"{axis}_status"] = value["status"]
        rows.append(row)
    result = frame.copy() if copy else frame
    empty = pd.Series("", index=frame.index, dtype="string")
    pmids = (
        pd.to_numeric(frame.get("pmid", empty), errors="coerce")
        .astype("Int64")
        .astype("string")
        .fillna("")
    )
    arms = frame.get("condition_id", empty).astype("string").fillna("")
    resolved = frame.get("sample_attribution", empty).isin(_RESOLVED_ARMS)
    if "evidence_kind" in frame:
        resolved &= frame["evidence_kind"].eq("ms")
    arms = arms.where(resolved, "")
    if rows:
        lookup = pd.MultiIndex.from_tuples(
            [(row["_lineage_pmid"], row["_lineage_arm"]) for row in rows]
        )
        positions = lookup.get_indexer(pd.MultiIndex.from_arrays([pmids, arms])) + 1
    else:
        import numpy as np

        positions = np.zeros(len(frame), dtype=int)
    for column in LINEAGE_COLUMNS:
        default = "[]" if column.endswith("_ids") else "unknown"
        if column == "lineage_context_id":
            default = ""
        # The registry is small; gather integer category codes, not millions
        # of repeated JSON strings or a merge/copy of the full evidence frame.
        template = pd.Categorical([default, *(row[column] for row in rows)])
        result[column] = pd.Categorical.from_codes(
            template.codes[positions], categories=template.categories
        )
    return result
