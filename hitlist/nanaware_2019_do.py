"""Source-qualified HLA-DO / H2-O contract and title-intensity parser for Nanaware 2019 (#816).

Primary source: Nanaware et al. (2019), *MCP*, PMID 30573663, PMC6398211,
DOI 10.1074/mcp.RA118.000956, MassIVE MSV000082570 / PRIDE PXD010301.

Preserves:
- Strict separation of human LG-2 WT/DO-KO clones vs mouse splenic B-cell
  WT/H2-O-KO systems.
- Explicit non-OA retrieval failure receipts for S10-S14 without manufacturing
  synthetic quantities or claiming absence.
- Outcome-selected bias guard on the 9 isotope standards in S13 (cannot be used
  as an unbiased global MS-response calibration cohort).
- Source-faithful parser for mzTab/mzIdentML spectrum-title ``intensity=...``
  values retaining exact IDs, duplicate-value flags, missingness, and
  ``unresolved_unit_and_aggregation_semantics`` (#816).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

CONTRACT_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "nanaware_2019_pxd010301.json"
)
REVISION = "nanaware2019_do_pxd010301_v1"

_TITLE_INTENSITY_RE = re.compile(r"\bintensity\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def nanaware_2019_do_source_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned Nanaware 2019 / PXD010301 (#816) source contract."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("revision") != REVISION:
        raise ValueError(f"Unsupported Nanaware 2019 contract revision: {contract.get('revision')}")
    return contract


def extract_title_encoded_intensities(
    records: list[dict[str, Any]],
    *,
    system_id: str = "human_lg2_do_clones",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse spectrum-title ``intensity=...`` tokens without summing repeated values (#816).

    Each input dict should provide ``spectrum_result_id``, ``spectrum_title``,
    and optional ``identification_item_id`` and ``peptide``. Emits a DataFrame
    with explicit ``value_state`` (``"positive"`` / ``"zero"`` / ``"missing"``),
    ``is_shared_intensity_value`` (True when another spectrum result carries the
    exact same numeric intensity), and ``unit_and_aggregation_status`` fixed to
    ``"unresolved_unit_and_aggregation_semantics"``.
    """
    contract = nanaware_2019_do_source_contract()
    systems = contract["experimental_systems"]
    if system_id not in systems:
        raise ValueError(
            f"Unknown Nanaware 2019 system_id {system_id!r}; choose from {sorted(systems)}"
        )
    sys_meta = systems[system_id]

    parsed_vals: list[float | None] = []
    for rec in records:
        title = str(rec.get("spectrum_title") or "")
        match = _TITLE_INTENSITY_RE.search(title)
        if match is None:
            parsed_vals.append(None)
        else:
            val = float(match.group(1))
            parsed_vals.append(val if math.isfinite(val) else None)

    pos_counts = Counter(v for v in parsed_vals if v is not None and v > 0.0)
    rows: list[dict[str, Any]] = []
    n_pos = n_zero = n_missing = n_shared = 0
    for rec, val in zip(records, parsed_vals, strict=True):
        if val is None:
            state = "missing"
            is_shared = False
            n_missing += 1
        elif val > 0.0:
            state = "positive"
            is_shared = pos_counts[val] > 1
            n_pos += 1
            if is_shared:
                n_shared += 1
        else:
            state = "explicit_zero"
            is_shared = False
            n_zero += 1
        rows.append(
            {
                "spectrum_result_id": str(rec.get("spectrum_result_id") or ""),
                "identification_item_id": str(rec.get("identification_item_id") or ""),
                "peptide": str(rec.get("peptide") or ""),
                "spectrum_title_raw": str(rec.get("spectrum_title") or ""),
                "title_encoded_intensity": val if val is not None else math.nan,
                "value_state": state,
                "is_shared_intensity_value": is_shared,
                "safe_to_sum_as_precursor_amount": False,
                "unit_and_aggregation_status": "unresolved_unit_and_aggregation_semantics",
                "system_id": system_id,
                "species": sys_meta["species"],
                "mhc_allele_set": sys_meta["mhc_allele_set"],
                "source_accession": contract["massive_accession"],
            }
        )

    summary = {
        "system_id": system_id,
        "n_spectrum_results": len(rows),
        "n_positive_title_intensities": n_pos,
        "n_explicit_zero_title_intensities": n_zero,
        "n_missing_title_intensities": n_missing,
        "n_distinct_positive_values": len(pos_counts),
        "n_shared_value_results": n_shared,
        "unit_and_aggregation_status": "unresolved_unit_and_aggregation_semantics",
    }
    return pd.DataFrame(rows), summary


def validate_nanaware_s13_calibration_scope(*, global_calibration: bool = False) -> dict[str, Any]:
    """Guard against using outcome-selected S13 isotope standards as global calibration (#816)."""
    contract = nanaware_2019_do_source_contract()
    s13 = contract["supplement_retrieval_receipts"]["S13"]
    if global_calibration:
        raise ValueError(
            "Nanaware 2019 S13 contains 9 outcome-selected isotope standards and cannot "
            "serve as an unbiased global MS-response calibration cohort (#816)."
        )
    return s13
