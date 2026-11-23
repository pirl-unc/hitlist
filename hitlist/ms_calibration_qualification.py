"""Non-MHC known-load MS response calibration qualification & identifiability gate (#762).

Implements #762:
- Unifies the qualification status of PXD004886 (2.0.6), PXD021013 (2.0.26), PASS01163/PASS01200
  (Michaud 2018 — stock AAA/CZE metadata verified, no separate isotope-channel areas in
  supplements), and PASS05890 (Sobsey 2025 — 48 vs 53 panel distinction, Panorama sign-in gate,
  PeptideAtlas FTPS expired-cert/hostname-mismatch receipt without disabling TLS verification).
- Enforces the critical absolute-area vs isotope-ratio identifiability gate: ratio-only curves
  (I_light / I_heavy) cancel per-peptide detector response k and are rejected for absolute
  cross-sequence response modeling.
- Wires PXD004886's 8 pinned CC-BY 4.0 source assets into the uniform ``hitlist data`` /
  ``hitlist.downloads`` surface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

_CONTRACT_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "non_mhc_calibration_qualification.json"
)
_PXD004886_ASSETS_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "ms_calibration_assets.json"
)

_ALLOWED_ABSOLUTE_ENDPOINTS = frozenset(
    {"separate_isotope_channel_area", "raw_peak_area", "raw_ms1_area"}
)
_NON_IDENTIFIABLE_ENDPOINTS = frozenset(
    {"isotope_ratio_only", "light_to_heavy_ratio", "inferred_concentration", "lloq_summary"}
)


def non_mhc_calibration_contract() -> dict[str, Any]:
    """Return the unified non-MHC MS-response calibration qualification contract (#762)."""
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


def fetch_pxd004886_calibration_sources(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    offline: bool | None = None,
) -> Path:
    """Download and SHA-256-verify all 8 pinned PXD004886 (Collins 2017) assets (~420 KB)."""
    from .downloads import data_dir, download_to_file

    manifest = json.loads(_PXD004886_ASSETS_PATH.read_text(encoding="utf-8"))
    target_dir = Path(dest_dir) if dest_dir is not None else data_dir() / "pxd004886_ms_calibration"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, meta in sorted(manifest["assets"].items()):
        download_to_file(
            meta["url"],
            target_dir / filename,
            label=f"PXD004886/{filename}",
            force=force,
            expected_sha256=meta["sha256"],
            expected_size=int(meta["size"]),
            offline=offline,
        )
    return target_dir


def guard_ms_response_identifiability(
    records_df: pd.DataFrame,
    *,
    endpoint_col: str = "endpoint_kind",
    amount_scope_col: str = "amount_scope",
) -> dict[str, Any]:
    """Enforce the #762 absolute-area vs isotope-ratio and dose-scope identifiability gate.

    Rejects ratio-only endpoints (``I_light / I_heavy`` cancels per-peptide response ``k``),
    inferred plasma concentrations, and LLOQ summaries from absolute cross-sequence detector
    response training, and separates stock concentration from on-column injected dose.
    """
    if endpoint_col not in records_df.columns:
        raise ValueError(f"records_df must include a {endpoint_col!r} column")

    out = records_df.copy()
    dispositions: list[str] = []
    reasons: list[str] = []
    for _, rec in out.iterrows():
        ep = "" if pd.isna(rec[endpoint_col]) else str(rec[endpoint_col]).strip().lower()
        amt_scope = (
            ""
            if amount_scope_col not in out.columns or pd.isna(rec.get(amount_scope_col))
            else str(rec.get(amount_scope_col)).strip().lower()
        )
        raw_val = rec.get("signal_value")
        has_pos_signal = True
        if "signal_value" in out.columns:
            if pd.isna(raw_val):
                has_pos_signal = False
            else:
                try:
                    v = float(raw_val)
                    has_pos_signal = math.isfinite(v) and v > 0.0
                except (TypeError, ValueError):
                    has_pos_signal = False

        if ep in _NON_IDENTIFIABLE_ENDPOINTS:
            dispositions.append("rejected_unidentifiable_for_absolute_response")
            reasons.append(f"endpoint_cancels_or_obscures_response_k:{ep}")
        elif ep not in _ALLOWED_ABSOLUTE_ENDPOINTS:
            dispositions.append("rejected_unknown_endpoint_kind")
            reasons.append(f"unrecognized_endpoint_kind:{ep or 'missing'}")
        elif amt_scope == "stock_solution_concentration_only":
            dispositions.append("deferred_missing_on_column_dose")
            reasons.append("stock_concentration_is_not_injected_on_column_dose")
        elif not has_pos_signal:
            dispositions.append("excluded_nonpositive_or_missing_signal")
            reasons.append("nonpositive_or_missing_not_biological_negative")
        else:
            dispositions.append("admitted_absolute_response_candidate")
            reasons.append("separate_absolute_area_with_dose_context")

    out["identifiability_disposition"] = dispositions
    out["identifiability_reason"] = reasons

    admitted = out[
        out["identifiability_disposition"] == "admitted_absolute_response_candidate"
    ].reset_index(drop=True)
    rejected_ratio = out[
        out["identifiability_disposition"] == "rejected_unidentifiable_for_absolute_response"
    ].reset_index(drop=True)
    deferred = out[
        out["identifiability_disposition"] == "deferred_missing_on_column_dose"
    ].reset_index(drop=True)

    return {
        "admitted": admitted,
        "rejected_unidentifiable": rejected_ratio,
        "deferred_dose_scope": deferred,
        "all_records": out,
        "summary": {
            "n_source_rows": len(out),
            "n_admitted_absolute_response": len(admitted),
            "n_rejected_unidentifiable": len(rejected_ratio),
            "n_deferred_stock_only": len(deferred),
            "ratio_only_identifies_absolute_response": False,
            "swath_or_mrm_transfers_to_mapp_or_apc": False,
        },
    }
