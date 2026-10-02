"""Primary-literature evidence for cross-species experimental contexts.

The derived ``is_chimeric``, ``is_engineered_mhc`` and ``xenograft`` flags are
useful search leads, but they are functions of the same species labels they
would otherwise appear to validate.  This module reads an independent,
arm-scoped literature registry.  A record reaches an MS sample only through an
exact ``(pmid, condition_id)`` link; records without that link remain available
for audit and cannot be broadcast over a whole study.
"""

from __future__ import annotations

import re
from functools import lru_cache
from os.path import dirname, join
from types import MappingProxyType

import pandas as pd

from .curation_yaml import load_curation_yaml

SPECIES_CONTEXT_STATUS_VALUES = ("resolved", "partial", "unresolved")
SPECIES_CONTEXT_KIND_VALUES = (
    "culture_supplement",
    "exogenous_antigen",
    "mhc_transfectant",
    "mhc_transgenic",
    "native",
    "xenograft",
    "xenograft_derived_culture",
)

# Ordered exactly as the columns appear in sample and observation exports.
SPECIES_CONTEXT_FIELDS = MappingProxyType(
    {
        "species_context_id": "stable literature-context identity within one PMID",
        "species_context_status": (
            "review result: resolved, partial, or unresolved; independent of derived flags"
        ),
        "species_context_scope": "the exact experimental material or arm reviewed",
        "species_context_kind": (
            "sorted semicolon-separated mechanisms supported by the primary source"
        ),
        "species_context_reference": (
            "primary-source URL or identifier plus the section, table, figure, or supplement"
        ),
        "presenting_species": (
            "species of the cells or tissue carrying the immunoprecipitated MHC"
        ),
        "in_vivo_host_species": "host species containing the material when it was harvested",
        "lineage_host_species": (
            "earlier host species in the material's passage history, absent at harvest"
        ),
        "introduced_mhc_species": "species origin of an experimentally introduced MHC",
        "supported_foreign_species": (
            "foreign peptide-source species independently supported for this context"
        ),
        "reviewed_candidate_species": (
            "candidate source species reviewed, including unsupported candidates"
        ),
        "species_context_note": (
            "curator conclusion, including the unresolved boundary of the evidence"
        ),
    }
)
SPECIES_CONTEXT_COLUMNS = tuple(SPECIES_CONTEXT_FIELDS)

# These describe one review record, rather than a biological fact that can be
# retained when every candidate arm independently agrees on it.
ARM_SPECIFIC_SPECIES_CONTEXT_COLUMNS = frozenset(
    {
        "species_context_id",
        "species_context_status",
        "species_context_scope",
        "species_context_reference",
        "reviewed_candidate_species",
        "species_context_note",
    }
)

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "inventory", "records"})
_RECORD_FIELDS = frozenset({"pmid", "condition_ids", *SPECIES_CONTEXT_COLUMNS})
_CONTEXT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_MULTI_VALUE_FIELDS = (
    "species_context_kind",
    "supported_foreign_species",
    "reviewed_candidate_species",
)


def _data_path() -> str:
    return join(dirname(__file__), "data", "species_contexts.yaml")


def empty_species_context_columns() -> dict[str, str]:
    """Return the empty export record for evidence not linked to a sample."""

    return dict.fromkeys(SPECIES_CONTEXT_COLUMNS, "")


def _validate_multivalue(record: dict, field: str, where: str) -> None:
    value = record.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"{where}: {field} must be a string")
    if not value:
        return
    tokens = value.split(";")
    if any(not token or token != token.strip() for token in tokens):
        raise ValueError(f"{where}: {field} contains an empty or unstripped token")
    if tokens != sorted(set(tokens)):
        raise ValueError(f"{where}: {field} must be sorted and unique")


def _validate_record(record: dict, index: int) -> None:
    where = f"species_contexts records[{index}]"
    if not isinstance(record, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = sorted(set(record) - _RECORD_FIELDS)
    if unknown:
        raise ValueError(f"{where} has unknown field(s) {unknown}")
    missing = sorted(
        field
        for field in (
            "pmid",
            "species_context_id",
            "species_context_status",
            "species_context_scope",
            "species_context_reference",
            "species_context_note",
        )
        if not record.get(field)
    )
    if missing:
        raise ValueError(f"{where} is missing required field(s) {missing}")
    if type(record["pmid"]) is not int or record["pmid"] <= 0:
        raise ValueError(f"{where}: pmid must be a positive integer")
    if not _CONTEXT_ID_RE.fullmatch(record["species_context_id"]):
        raise ValueError(f"{where}: malformed species_context_id")
    status = record["species_context_status"]
    if status not in SPECIES_CONTEXT_STATUS_VALUES:
        raise ValueError(f"{where}: invalid species_context_status {status!r}")
    condition_ids = record.get("condition_ids", [])
    if (
        not isinstance(condition_ids, list)
        or any(
            not isinstance(value, str) or not _CONTEXT_ID_RE.fullmatch(value)
            for value in condition_ids
        )
        or condition_ids != sorted(set(condition_ids))
    ):
        raise ValueError(f"{where}: condition_ids must be sorted, unique IDs")
    for field in SPECIES_CONTEXT_COLUMNS:
        value = record.get(field, "")
        if not isinstance(value, str) or value != value.strip():
            raise ValueError(f"{where}: {field} must be a stripped string")
    for field in _MULTI_VALUE_FIELDS:
        _validate_multivalue(record, field, where)
    kinds = set(record.get("species_context_kind", "").split(";")) - {""}
    unknown_kinds = sorted(kinds - set(SPECIES_CONTEXT_KIND_VALUES))
    if unknown_kinds:
        raise ValueError(f"{where}: unknown species_context_kind token(s) {unknown_kinds}")
    if status == "resolved" and not kinds:
        raise ValueError(f"{where}: resolved context has no species_context_kind")
    if record.get("introduced_mhc_species") and not kinds.intersection(
        {"mhc_transfectant", "mhc_transgenic"}
    ):
        raise ValueError(f"{where}: introduced_mhc_species requires an MHC engineering kind")
    if record.get("lineage_host_species") and "xenograft_derived_culture" not in kinds:
        raise ValueError(f"{where}: lineage_host_species requires xenograft_derived_culture")
    if "xenograft" in kinds and not (
        record.get("presenting_species") and record.get("in_vivo_host_species")
    ):
        raise ValueError(f"{where}: xenograft requires presenting and in-vivo host species")
    if record.get("supported_foreign_species") and not kinds.intersection(
        {"culture_supplement", "exogenous_antigen"}
    ):
        raise ValueError(
            f"{where}: supported_foreign_species requires a foreign-source context kind"
        )


@lru_cache(maxsize=1)
def load_species_contexts() -> dict:
    """Load and validate the packaged literature registry and review manifest."""

    payload = load_curation_yaml(_data_path()) or {}
    if not isinstance(payload, dict):
        raise ValueError("species_contexts.yaml must contain a mapping")
    unknown = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"species_contexts.yaml has unknown top-level field(s) {unknown}")
    if payload.get("schema_version") != 1:
        raise ValueError("species_contexts.yaml schema_version must be 1")
    inventory = payload.get("inventory")
    records = payload.get("records")
    if not isinstance(inventory, dict) or not isinstance(records, list):
        raise ValueError("species_contexts.yaml requires inventory mapping and records list")
    identities: set[tuple[int, str]] = set()
    targets: set[tuple[int, str]] = set()
    for index, record in enumerate(records):
        _validate_record(record, index)
        identity = (record["pmid"], record["species_context_id"])
        if identity in identities:
            raise ValueError(f"duplicate species context identity {identity}")
        identities.add(identity)
        for condition_id in record.get("condition_ids", []):
            target = (record["pmid"], condition_id)
            if target in targets:
                raise ValueError(f"condition target {target} has multiple species contexts")
            targets.add(target)

    reviewed = inventory.get("reviewed_pmids")
    queued = inventory.get("review_queue_pmids")
    if (
        not isinstance(reviewed, list)
        or reviewed != sorted(set(reviewed))
        or not isinstance(queued, list)
        or queued != sorted(set(queued))
        or set(reviewed).intersection(queued)
    ):
        raise ValueError("inventory PMID lists must be sorted, unique, and disjoint")
    record_pmids = {record["pmid"] for record in records}
    if record_pmids != set(reviewed):
        raise ValueError("inventory reviewed_pmids must equal the PMIDs represented by records")
    expected_counts = {
        "n_reviewed_pmids": len(reviewed),
        "n_review_queue_pmids": len(queued),
    }
    for field, expected in expected_counts.items():
        if inventory.get(field) != expected:
            raise ValueError(f"inventory {field} must equal {expected}")
    return payload


def validate_species_context_condition_links(overrides: dict[int, dict]) -> None:
    """Require every declared condition target to exist in PMID curation."""

    for record in load_species_contexts()["records"]:
        if not record.get("condition_ids"):
            continue
        # Tests and downstream tools may load a deliberately small synthetic
        # PMID registry. Validate links for studies present in that registry;
        # absence of an unrelated packaged study says nothing about its links.
        if record["pmid"] not in overrides:
            continue
        known = {
            sample.get("condition_id")
            for sample in overrides.get(record["pmid"], {}).get("ms_samples", [])
        }
        missing = sorted(set(record["condition_ids"]) - known)
        if missing:
            raise ValueError(
                f"PMID {record['pmid']} species context {record['species_context_id']}: "
                f"unknown condition_id target(s) {missing}"
            )


@lru_cache(maxsize=1)
def _condition_index() -> dict[tuple[int, str], dict[str, str]]:
    index = {}
    for record in load_species_contexts()["records"]:
        exported = {column: record.get(column, "") for column in SPECIES_CONTEXT_COLUMNS}
        for condition_id in record.get("condition_ids", []):
            index[(record["pmid"], condition_id)] = exported
    return index


def species_context_for_sample(pmid: int, condition_id: str) -> dict[str, str]:
    """Return independently sourced context for an exactly linked sample."""

    return dict(_condition_index().get((pmid, condition_id), empty_species_context_columns()))


def generate_species_contexts_table() -> pd.DataFrame:
    """Return every reviewed literature context, including unattached records."""

    columns = ["pmid", "condition_ids", *SPECIES_CONTEXT_COLUMNS]
    rows = []
    for record in load_species_contexts()["records"]:
        row = {column: record.get(column, "") for column in columns}
        row["condition_ids"] = ";".join(record.get("condition_ids", []))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
