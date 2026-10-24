"""Source-scoped material contexts and qualified links between them.

The graph deliberately distinguishes a named project, study arm, acquisition
and physical material.  In particular, a run label or replicate number does
not become a specimen or aliquot identity merely because it is useful for a
comparison.
"""

from __future__ import annotations

import pyarrow as pa

CONTEXT_KINDS = frozenset(
    {
        "source_project",
        "study_arm",
        "source_acquisition",
        "specimen",
        "aliquot",
    }
)
LINK_PRECISIONS = frozenset(
    {
        "unresolved",
        "project_context",
        "study_arm",
        "source_acquisition",
        "exact_specimen",
        "exact_aliquot",
    }
)

MATERIAL_CONTEXT_SCHEMA = pa.schema(
    [
        ("context_id", pa.string()),
        ("context_kind", pa.string()),
        ("sample_label_raw", pa.string()),
        ("condition_id", pa.string()),
        ("condition_text", pa.string()),
        ("condition_status", pa.string()),
        ("specimen_id", pa.string()),
        ("aliquot_id", pa.string()),
        ("material_link_precision", pa.string()),
        ("context_status", pa.string()),
        ("source_locator", pa.string()),
        ("notes", pa.string()),
    ]
)

MATERIAL_LINK_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("source_context_id", pa.string()),
        ("target_context_id", pa.string()),
        ("assertion_predicate", pa.string()),
        ("assertion_status", pa.string()),
        ("material_link_precision", pa.string()),
        ("source_locator", pa.string()),
        ("notes", pa.string()),
    ]
)


def validate_material_graph(contexts, links):
    """Reject a graph that promotes labels into physical material identity."""
    context_ids = set()
    for row in contexts:
        context_id, kind = row["context_id"], row["context_kind"]
        if not context_id or context_id in context_ids:
            raise ValueError("Material contexts require unique IDs")
        if kind not in CONTEXT_KINDS:
            raise ValueError("Unknown material context kind")
        precision = row["material_link_precision"]
        if precision not in LINK_PRECISIONS:
            raise ValueError("Unknown material-link precision")
        has_specimen = row.get("specimen_id") is not None
        has_aliquot = row.get("aliquot_id") is not None
        if has_aliquot and precision != "exact_aliquot":
            raise ValueError("An aliquot ID requires exact aliquot precision")
        if has_specimen and precision not in {"exact_specimen", "exact_aliquot"}:
            raise ValueError("A specimen ID requires exact physical-material precision")
        if kind == "specimen" and not has_specimen:
            raise ValueError("A specimen context requires a specimen ID")
        if kind == "aliquot" and not has_aliquot:
            raise ValueError("An aliquot context requires an aliquot ID")
        context_ids.add(context_id)
    claim_ids = set()
    for row in links:
        claim_id = row["claim_id"]
        if not claim_id or claim_id in claim_ids:
            raise ValueError("Material links require unique claim IDs")
        if (
            row["source_context_id"] not in context_ids
            or row["target_context_id"] not in context_ids
        ):
            raise ValueError("Material link references an unknown context")
        if not row["assertion_predicate"] or not row["assertion_status"]:
            raise ValueError("Material links require a predicate and assertion status")
        if row["material_link_precision"] not in LINK_PRECISIONS:
            raise ValueError("Unknown material-link precision")
        claim_ids.add(claim_id)


def links_with_minimum_precision(links, minimum_precision):
    """Select qualified links without treating arm labels as physical pairs.

    The qualification sets intentionally are not a simple numeric ordering:
    a source-acquisition relation and a study-arm relation establish different
    facts.  Both are insufficient for a specimen or aliquot request.
    """
    allowed = {
        "unresolved": LINK_PRECISIONS,
        "project_context": LINK_PRECISIONS - {"unresolved"},
        "study_arm": {"study_arm", "exact_specimen", "exact_aliquot"},
        "source_acquisition": {"source_acquisition", "exact_specimen", "exact_aliquot"},
        "exact_specimen": {"exact_specimen", "exact_aliquot"},
        "exact_aliquot": {"exact_aliquot"},
    }
    if minimum_precision not in allowed:
        raise ValueError("Unknown requested material-link precision")
    return tuple(
        row for row in links if row["material_link_precision"] in allowed[minimum_precision]
    )
