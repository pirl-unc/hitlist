"""Source-backed experiment counts for legacy bulk peptide summaries."""

import json

import pandas as pd

from .bulk_fractionation import curate_fractionation_ph, fractionation_registry

REPLICATE_COLUMNS = (
    "reported_n_replicates_detected",
    "reported_n_replicates_possible",
    "n_experiments_detected",
    "n_experiments_possible",
    "n_replicates_possible",
    "reproducibility_fraction",
    "n_acquisition_runs",
    "n_biological_replicates",
    "replicate_protocol_id",
    "replicate_experiment_ids",
    "replicate_count_unit",
    "replicate_count_status",
    "replicate_count_reference",
    "replicate_count_source_sha256",
    "replicate_acquisition_ids_sha256",
)
INTEGER_REPLICATE_COLUMNS = (
    "n_experiments_detected",
    "n_experiments_possible",
    "n_replicates_possible",
    "n_acquisition_runs",
    "n_biological_replicates",
)


def curate_replicate_counts(frame):
    """Retain reported counts and qualify fractions against a reviewed inventory.

    ``n_replicates_*`` are compatibility names for MaxQuant experiment counts,
    never a claim of biological replication. Unknown scopes and pooled numerators
    remain nullable. Reapplying this projection preserves original reported values.
    """
    frame = frame.copy()
    original_counts = {
        name: frame[name].copy()
        for name in ("n_replicates_detected", "n_replicates_possible")
        if name in frame
    }
    for field in ("n_replicates_detected", "n_replicates_possible"):
        reported = "reported_" + field
        if reported not in frame:
            frame[reported] = frame.get(field, pd.NA)
    if "fractionation_protocol_id" not in frame:
        frame = curate_fractionation_ph(frame)
    for column in (*INTEGER_REPLICATE_COLUMNS, "n_replicates_detected"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["reproducibility_fraction"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for column in REPLICATE_COLUMNS[8:]:
        frame[column] = pd.Series("", index=frame.index, dtype="string")
    frame["replicate_count_status"] = pd.Series(
        "unresolved_scope", index=frame.index, dtype="string"
    )
    registry = fractionation_registry()
    empty = pd.Series("", index=frame.index, dtype="string")
    selected = frame.get("source", empty).eq(registry["source"]).fillna(False)
    if "peptide" in frame:
        selected &= frame.peptide.notna() & frame.peptide.ne("")
    reported = pd.to_numeric(frame.reported_n_replicates_detected, errors="coerce")
    known_protocols = set()
    for scope in registry["scopes"]:
        protocol = scope["protocol_id"]
        experiments = scope["experiment_ids"]
        n_experiments = len(experiments)
        if (
            protocol in known_protocols
            or n_experiments != len(set(experiments))
            or n_experiments != scope["n_replicates_possible"]
            or n_experiments < 1
        ):
            raise ValueError("Invalid curated experiment inventory: " + protocol)
        known_protocols.add(protocol)
        mask = selected & frame.get("fractionation_protocol_id", empty).eq(protocol).fillna(False)
        if not mask.any():
            continue
        for column, value in {
            "n_experiments_possible": n_experiments,
            "n_replicates_possible": n_experiments,
            "n_acquisition_runs": scope["n_acquisition_runs"],
            "replicate_protocol_id": protocol,
            "replicate_experiment_ids": json.dumps(experiments, separators=(",", ":")),
            "replicate_count_unit": "MaxQuant experiment",
            "replicate_acquisition_ids_sha256": scope["raw_file_ids_sha256"],
            "replicate_count_status": scope["replicate_count_status"],
            "replicate_count_reference": registry["provenance"]["archive"] + "#summary.txt",
            "replicate_count_source_sha256": registry["provenance"]["members"]["summary.txt"][
                "sha256"
            ],
        }.items():
            frame.loc[mask, column] = value
        if scope["replicate_count_status"] != "available":
            continue
        valid = reported.between(0, n_experiments) & reported.mod(1).eq(0)
        valid = (mask & valid).fillna(False)
        missing = (
            frame.reported_n_replicates_detected.isna()
            | frame.reported_n_replicates_detected.eq("")
        )
        frame.loc[mask & ~valid, "replicate_count_status"] = "invalid_reported_count"
        frame.loc[mask & missing.fillna(True), "replicate_count_status"] = "missing_reported_count"
        counts = reported.loc[valid].astype("Int64")
        frame.loc[valid, "n_experiments_detected"] = counts
        frame.loc[valid, "n_replicates_detected"] = counts
        frame.loc[valid, "reproducibility_fraction"] = counts.astype("Float64") / n_experiments
    # Other sources retain their reported API values, with no qualified fraction.
    if (~selected).any():
        for column, values in original_counts.items():
            frame[column] = frame[column].astype(object)
            frame.loc[~selected, column] = values.loc[~selected]
    return frame
