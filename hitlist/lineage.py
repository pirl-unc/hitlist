"""Reviewed biological identities, separate from publication and assay IDs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .curation_yaml import load_curation_yaml

SCHEMA_VERSION = 1
AXES = ("experimental_origin", "specimen", "donor", "acquisition")
CELLULAR_DEPENDENCY_AXES = ("specimen", "donor")
STATUSES = {"resolved", "unknown", "pooled", "ambiguous"}
ASSAY_PREPARATIONS = (
    "purified_mhc",
    "cellular_mhc",
    "lysate_mhc",
    "cell_associated_ms",
    "generated_cellular_consensus",
    "unknown",
)
APPLICABILITY_STATUSES = (
    "applicable_resolved",
    "applicable_unresolved",
    "not_applicable",
    "applicability_unknown",
)
LINEAGE_COLUMNS = ["lineage_context_id", "lineage_status"] + [
    column for axis in AXES for column in (f"{axis}_ids", f"{axis}_status")
]
APPLICABILITY_COLUMNS = [
    "assay_preparation",
    "assay_preparation_basis",
    *(f"{axis}_applicability" for axis in AXES),
]
_PREPARATION_MENTION_RE = re.compile(r"\b(?:purified mhc|cellular mhc|lysate mhc)\b")
_PREPARATION_CLAUSE_RE = re.compile(r"(?:^|[;|])\s*(purified mhc|cellular mhc|lysate mhc)(?=/|$)")
_CELLULAR_APPLICABLE_PREPARATIONS = frozenset(
    {
        "cellular_mhc",
        "lysate_mhc",
        "cell_associated_ms",
        "generated_cellular_consensus",
    }
)
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


def lineage_registry_sha256(registry: dict | None = None) -> str:
    """Return the canonical SHA-256 hex digest of a validated lineage registry (#680).

    Freezes the exact reviewed entities, aliases, and contexts independent of
    YAML formatting or in-memory key order so split-audit reports and training
    release manifests bind unambiguously to one reviewed registry state.
    """
    validated = load_lineage() if registry is None else validate_lineage(registry)
    canonical_json = json.dumps(
        validated,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def canonical_identities(registry: dict) -> dict[str, str]:
    validate_lineage(registry)
    result = {e["identity_id"]: e["identity_id"] for e in registry["entities"]}
    result.update({a["identity_id"]: a["canonical_id"] for a in registry.get("aliases", [])})
    return result


def attach_lineage(
    frame: pd.DataFrame,
    registry=None,
    *,
    copy: bool = True,
    include_applicability: bool = False,
) -> pd.DataFrame:
    """Attach reviewed study/arm contexts; counting fallbacks are never IDs.

    Arm attribution evidence remains in ``sample_attribution``. Even a reviewed
    context cannot resolve an observation whose arm is ambiguous or unassigned.
    Missing contexts are explicit unknowns, including in binding-only exports.
    When ``include_applicability=True``, also attaches assay-preparation and
    per-axis biological-independence applicability metadata (#706).
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
    if include_applicability:
        result = attach_lineage_applicability(result, copy=False)
    return result


def classify_assay_preparation(
    assay_method: object = "",
    evidence_kind: object = "",
    *,
    is_ms_observation: object = False,
) -> tuple[str, str]:
    """Classify assay preparation from explicit method and modality evidence (#706).

    Numeric endpoints (``IC50``, ``KD``, ``half life``) never imply purified MHC:
    cellular competitive and direct binding assays also report those endpoints.
    Conflicting preparation claims or ``purified MHC`` paired with MS elution
    remain ``("unknown", "conflicting_assay_preparation")``.
    """
    method = "" if assay_method is None or pd.isna(assay_method) else str(assay_method).strip()
    kind = (
        ""
        if evidence_kind is None or pd.isna(evidence_kind)
        else str(evidence_kind).strip().lower()
    )
    ms_flag = (
        False
        if is_ms_observation is None or pd.isna(is_ms_observation)
        else bool(is_ms_observation)
    )
    is_ms = ms_flag or kind == "ms"

    lowered = method.casefold()
    mentions = set(_PREPARATION_MENTION_RE.findall(lowered))
    preparations = set(_PREPARATION_CLAUSE_RE.findall(lowered))
    if len(mentions) > 1 or len(preparations) > 1 or ("purified mhc" in preparations and is_ms):
        return "unknown", "conflicting_assay_preparation"
    if "purified mhc" in preparations:
        return "purified_mhc", "explicit_purified_mhc_method"
    if "cellular mhc" in preparations:
        return "cellular_mhc", "explicit_cellular_mhc_method"
    if "lysate mhc" in preparations:
        return "lysate_mhc", "explicit_lysate_mhc_method"
    if mentions:
        # Mentioned inside free-form method text without a clean clause boundary.
        return "unknown", "unspecified_assay_preparation"
    if is_ms:
        return "cell_associated_ms", "native_elution_evidence"
    return "unknown", "unspecified_assay_preparation"


def _classify_generated_parent_contexts(raw_contexts: object) -> tuple[str, str]:
    """Resolve preparation from parent leaves of a generated negative/control row."""
    text = "" if raw_contexts is None or pd.isna(raw_contexts) else str(raw_contexts).strip()
    if not text:
        return "unknown", "generated_parent_unresolved_or_conflicting"
    parsed = json.loads(text)
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(
            not isinstance(item, dict)
            or set(item) != {"method", "evidence_kind"}
            or any(not isinstance(v, str) for v in item.values())
            for item in parsed
        )
    ):
        raise ValueError(
            "parent_assay_contexts must be a nonempty JSON array of {method, evidence_kind} strings"
        )
    preps = {
        classify_assay_preparation(item["method"], item["evidence_kind"])[0] for item in parsed
    }
    if "unknown" in preps:
        return "unknown", "generated_parent_unresolved_or_conflicting"
    if len(preps) == 1:
        return next(iter(preps)), "generated_parent_consensus"
    if preps <= _CELLULAR_APPLICABLE_PREPARATIONS:
        return "generated_cellular_consensus", "generated_parent_consensus"
    return "unknown", "generated_parent_unresolved_or_conflicting"


def axis_applicability_status(axis: str, identity_status: str, assay_preparation: str) -> str:
    """Declare whether a biological lineage axis applies to an observation (#706).

    ``experimental_origin`` and ``acquisition`` remain applicable even for
    purified recombinant MHC assays because protein preparations, experiments,
    and runs can still be shared across partitions. ``specimen`` and ``donor``
    are ``not_applicable`` for purified recombinant MHC unless an explicit
    reviewed identity is already resolved.
    """
    if axis not in AXES:
        raise ValueError(f"Unknown lineage axis: {axis!r}")
    if assay_preparation not in ASSAY_PREPARATIONS:
        raise ValueError(f"Unknown assay preparation: {assay_preparation!r}")
    if identity_status == "resolved":
        return "applicable_resolved"
    if axis not in CELLULAR_DEPENDENCY_AXES:
        return "applicable_unresolved"
    if assay_preparation == "purified_mhc":
        return "not_applicable"
    if assay_preparation in _CELLULAR_APPLICABLE_PREPARATIONS:
        return "applicable_unresolved"
    return "applicability_unknown"


def attach_lineage_applicability(frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Attach ``APPLICABILITY_COLUMNS`` without altering native lineage statuses."""
    result = frame.copy() if copy else frame
    n_rows = len(result)
    if n_rows == 0:
        for column in APPLICABILITY_COLUMNS:
            categories = (
                list(ASSAY_PREPARATIONS)
                if column == "assay_preparation"
                else list(APPLICABILITY_STATUSES)
                if column.endswith("_applicability")
                else []
            )
            result[column] = pd.Categorical([], categories=categories)
        return result

    empty = pd.Series("", index=result.index, dtype="string")
    methods = result.get("assay_method", empty).astype("string").fillna("")
    kinds = result.get("evidence_kind", empty).astype("string").fillna("")
    if "is_ms_observation" in result.columns:
        ms_flags = result["is_ms_observation"].astype("boolean").fillna(False).astype(bool)
    else:
        ms_flags = pd.Series(False, index=result.index, dtype=bool)

    unique_pairs = pd.DataFrame(
        {"method": methods, "kind": kinds, "is_ms": ms_flags}
    ).drop_duplicates()
    pair_tuples = list(unique_pairs.itertuples(index=False, name=None))
    classified = [
        classify_assay_preparation(m, k, is_ms_observation=ms) for m, k, ms in pair_tuples
    ]
    lookup = pd.MultiIndex.from_tuples(pair_tuples)
    row_index = lookup.get_indexer(pd.MultiIndex.from_arrays([methods, kinds, ms_flags]))

    prep_values = np.array([item[0] for item in classified], dtype=object)[row_index]
    basis_values = np.array([item[1] for item in classified], dtype=object)[row_index]

    if "generated_id" in result.columns:
        gen_ids = result["generated_id"].astype("string").fillna("").str.strip()
        gen_mask = gen_ids.ne("").to_numpy(dtype=bool)
        if gen_mask.any():
            raw_parents = (
                result["parent_assay_contexts"].astype("string").fillna("")
                if "parent_assay_contexts" in result.columns
                else pd.Series("", index=result.index, dtype="string")
            )
            for pos in np.flatnonzero(gen_mask):
                prep, basis = _classify_generated_parent_contexts(raw_parents.iloc[pos])
                prep_values[pos] = prep
                basis_values[pos] = basis

    result["assay_preparation"] = pd.Categorical(prep_values, categories=list(ASSAY_PREPARATIONS))
    basis_categories = sorted(set(basis_values))
    result["assay_preparation_basis"] = pd.Categorical(basis_values, categories=basis_categories)

    for axis in AXES:
        status_col = f"{axis}_status"
        statuses = (
            result[status_col].astype("string").fillna("unknown")
            if status_col in result.columns
            else pd.Series("unknown", index=result.index, dtype="string")
        )
        resolved_mask = statuses.eq("resolved").to_numpy(dtype=bool)
        if axis not in CELLULAR_DEPENDENCY_AXES:
            codes = np.where(
                resolved_mask,
                APPLICABILITY_STATUSES.index("applicable_resolved"),
                APPLICABILITY_STATUSES.index("applicable_unresolved"),
            )
        else:
            codes = np.full(
                n_rows, APPLICABILITY_STATUSES.index("applicability_unknown"), dtype=np.int8
            )
            purified_mask = prep_values == "purified_mhc"
            cellular_mask = np.isin(prep_values, list(_CELLULAR_APPLICABLE_PREPARATIONS))
            codes[purified_mask] = APPLICABILITY_STATUSES.index("not_applicable")
            codes[cellular_mask] = APPLICABILITY_STATUSES.index("applicable_unresolved")
            codes[resolved_mask] = APPLICABILITY_STATUSES.index("applicable_resolved")
        result[f"{axis}_applicability"] = pd.Categorical.from_codes(
            codes, categories=list(APPLICABILITY_STATUSES)
        )
    return result
