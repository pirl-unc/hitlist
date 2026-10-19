"""Reviewed fractionation annotations for the legacy Bekker-Jensen summaries."""

from functools import lru_cache
from importlib.resources import files

import pandas as pd

from .curation_yaml import load_curation_yaml

FRACTIONATION_COLUMNS = (
    "legacy_fractionation_ph",
    "fractionation_protocol_id",
    "fractionation_ph_basis",
    "fractionation_ph_reference",
    "fractionation_ph_evidence_id",
)


@lru_cache(maxsize=1)
def fractionation_registry():
    return load_curation_yaml(files("hitlist.data.bulk_proteomics") / "detectability.yaml")


def experiment_fractionation_ph(experiment):
    """Return a reviewed value, including explicit unknown, for one exact name."""
    matches = [
        scope
        for scope in fractionation_registry()["scopes"]
        if experiment in scope["experiment_ids"]
    ]
    if len(matches) != 1:
        raise ValueError("No unique reviewed fractionation scope for experiment: " + experiment)
    return matches[0]["fractionation_ph"]


def curate_fractionation_ph(frame):
    """Project reviewed pH while retaining old annotations and every input row.

    Legacy summaries are matched on their stored axes; an existing preserved
    value makes repeated application idempotent. Non-Bekker-Jensen rows retain
    their values. Unknown/unmatched scopes receive no numeric pH claim.
    """
    if "source" not in frame or "fractionation_ph" not in frame:
        return frame
    selected = frame.source.eq("Bekker-Jensen_2017").fillna(False)
    if not selected.any():
        return frame
    frame = frame.copy()
    if "legacy_fractionation_ph" not in frame:
        frame["legacy_fractionation_ph"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        frame.loc[selected, "legacy_fractionation_ph"] = pd.to_numeric(
            frame.loc[selected, "fractionation_ph"], errors="coerce"
        ).astype("Float64")
    legacy = pd.to_numeric(frame.legacy_fractionation_ph, errors="coerce")
    frame["fractionation_ph"] = pd.to_numeric(frame.fractionation_ph, errors="coerce").astype(
        "Float64"
    )
    frame.loc[selected, "fractionation_ph"] = pd.NA
    for column in FRACTIONATION_COLUMNS[1:]:
        if column not in frame:
            frame[column] = pd.Series("", index=frame.index, dtype="string")
        else:
            frame[column] = frame[column].astype("string")
        frame.loc[selected, column] = ""
    frame.loc[selected, "fractionation_ph_basis"] = "unresolved_scope"
    cell_column = "cell_line_name" if "cell_line_name" in frame else "cell_line"
    required = {cell_column, "digestion_enzyme", "n_fractions_in_run", "enrichment"}
    if not required <= set(frame):
        return frame
    registry = fractionation_registry()
    matched = pd.Series(False, index=frame.index)
    for scope in registry["scopes"]:
        valid_ph = legacy.eq(scope["legacy_fractionation_ph"])
        if scope["fractionation_ph"] is None:
            valid_ph = valid_ph | legacy.isna()
        else:
            valid_ph = valid_ph | legacy.eq(scope["fractionation_ph"])
        mask = (
            selected
            & frame[cell_column]
            .astype("string")
            .str.casefold()
            .eq(scope["cell_line_name"].casefold())
            & frame.digestion_enzyme.isin(
                [scope["reported_digestion_enzyme"], scope["digestion_enzyme"]]
            )
            & pd.to_numeric(frame.n_fractions_in_run, errors="coerce").eq(
                scope["n_fractions_in_run"]
            )
            & frame.enrichment.eq(scope["enrichment"])
            & valid_ph.fillna(False)
        ).fillna(False)
        if (mask & matched).any():
            raise ValueError("Ambiguous reviewed fractionation scope")
        matched |= mask
        evidence_id = scope["fractionation_ph_evidence_id"]
        evidence = registry["fractionation_ph_evidence"][evidence_id]
        frame.loc[mask, "fractionation_ph"] = scope["fractionation_ph"]
        frame.loc[mask, "fractionation_protocol_id"] = scope["protocol_id"]
        frame.loc[mask, "fractionation_ph_evidence_id"] = evidence_id
        frame.loc[mask, "fractionation_ph_basis"] = evidence["basis"]
        frame.loc[mask, "fractionation_ph_reference"] = evidence["reference"]
    return frame
