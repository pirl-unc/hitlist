"""Preserve source-supplied supplementary chemistry without filling unknowns."""

import hashlib
import json
from functools import lru_cache

import pandas as pd


def peptidoform_keys(frame):
    """Comparable chemistry keys without equating unknown with unmodified."""
    blank = pd.Series("", index=frame.index, dtype="string")
    flags = frame.get("has_ptm", blank).astype("string").fillna("").str.lower()
    modifications = frame.get("peptide_modifications", blank).astype("string").fillna("")
    notation = frame.get("peptide_extended", blank).astype("string").fillna("")
    notation = notation.mask(notation.eq(frame["peptide"]), "")
    return pd.Series(
        [json.dumps(row, separators=(",", ":")) for row in zip(flags, modifications, notation)],
        index=frame.index,
        dtype="string",
    )


@lru_cache(maxsize=4096)
def _state(flag, modifications):
    token = flag.strip().lower()
    if token in {"", "unknown", "not_reported", "not reported", "unspecified"}:
        reported = None
    elif token in {"true", "1"}:
        reported = True
    elif token in {"false", "0"}:
        reported = False
    else:
        raise ValueError(f"Unsupported supplementary has_ptm value: {flag!r}")
    text = modifications.strip()
    if text.lower() in {"unknown", "not_reported", "not reported", "unspecified"}:
        text = ""
    declared_unmodified = text.lower() in {"none", "unmodified", "no modifications"}
    modified = bool(text) and not declared_unmodified
    conflict = (reported is False and modified) or (reported is True and declared_unmodified)
    if conflict:
        return None, text, "conflicting"
    value = (
        reported
        if reported is not None
        else (True if modified else False if declared_unmodified else None)
    )
    status = "unknown" if value is None else "reported_modified" if value else "reported_unmodified"
    return value, "" if declared_unmodified else text, status


def supplementary_peptidoforms(frame, entry):
    """Return chemistry fields and identity suffix; raw contributor cells stay intact.

    A file-level unmodified declaration is allowed only with an explicit source
    reference and locator. It cannot override contradictory row-level chemistry.
    An empty modification cell alone is not such a declaration.
    """
    policy = entry.get("peptidoform_policy")
    if policy is not None and (
        not isinstance(policy, dict)
        or policy.get("status") != "unmodified"
        or any(
            not isinstance(policy.get(k), str) or not policy[k].strip()
            for k in ("reference", "locator")
        )
    ):
        raise ValueError(
            "Supplementary peptidoform policy requires unmodified status, reference and locator"
        )
    flags = frame.get("has_ptm", pd.Series("", index=frame.index))
    mods = frame.get("peptide_modifications", pd.Series("", index=frame.index))
    extended = frame.get("peptide_extended", pd.Series("", index=frame.index))
    records = []
    for flag, modification, notation, peptide in zip(flags, mods, extended, frame["peptide"]):
        flag, modification, notation = str(flag), str(modification), str(notation)
        value, normalized, status = _state(flag, modification)
        basis = "row_fields"
        if policy is not None:
            if value is True or status == "conflicting":
                value, status = None, "conflicting"
            elif value is None:
                value, status, basis = False, "source_policy_unmodified", "file_policy"
        # A supplied different notation may contain chemistry we cannot parse.
        # Preserve it, and never let an unmodified flag conceal that ambiguity.
        if notation and notation != peptide and value is False:
            value, status = None, "conflicting"
        identity = [value, normalized, notation, status]
        suffix = (
            ""
            if identity == [None, "", "", "unknown"]
            else ":form:"
            + hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        )
        records.append(
            {
                "has_ptm": value,
                "peptide_modifications": normalized,
                "peptide_extended": notation or peptide,
                "peptide_modifications_raw": modification,
                "has_ptm_raw": flag,
                "peptidoform_review_status": status,
                "peptidoform_evidence": json.dumps(
                    {"basis": basis, "policy": policy}, sort_keys=True
                ),
                "_peptidoform_suffix": suffix,
            }
        )
    result = pd.DataFrame(records, index=frame.index)
    result["has_ptm"] = result["has_ptm"].astype("boolean")
    return result
