"""Qualification guards & uniform downloaders for PXD058121 (#831) and PXD018722 (#837).

Implements remaining qualification, scale/reporter guard, and catalog download scope for:
- #831 (PXD058121 intact degradomics): precursor-decay scale & AGC/accumulation-time guard
  keeping raw deconvoluted intensities distinct from TIC, normalized plots, and z-scores, and
  exposing all 14 pinned PRIDE/PMC S3 assets via ``hitlist data`` / ``hitlist.downloads``.
- #837 (PXD018722 Sahu 2021 PSMD2 KD): endogenous vs engineered-reporter (``P22222`` /
  ``NS3CCNB1_HUMAN`` tagged ``OX=9606``) origin guard, ``Intensity`` vs ``LFQ`` scale separation,
  8-25 search-length scope guard, and uniform metadata/FASTA download via ``hitlist data``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_QUAL_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "degradomics_native_qualification.json"
)
_INTACT_ASSETS_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "intact_degradomics.json"
)
_NATIVE_ASSETS_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "native_peptidomics.json"
)

_REPORTER_TOKENS = ("P22222", "NS3CCNB1_HUMAN", "NS3_CYCLIN")


def intact_degradomics_scale_qualification_contract() -> dict[str, Any]:
    """Return the pinned PXD058121 intact-degradomics scale qualification contract (#831)."""
    return json.loads(_QUAL_PATH.read_text(encoding="utf-8"))["pxd058121_intact_degradomics"]


def sahu_2021_psmd2_qualification_contract() -> dict[str, Any]:
    """Return the pinned PXD018722 Sahu 2021 native PSMD2-KD qualification contract (#837)."""
    return json.loads(_QUAL_PATH.read_text(encoding="utf-8"))["pxd018722_sahu_2021_psmd2"]


def fetch_pxd058121_intact_degradomics_sources(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    offline: bool | None = None,
) -> Path:
    """Download and SHA-256-verify all 14 pinned PXD058121 intact-degradomics assets (~4.7 MB)."""
    from .downloads import data_dir, download_to_file

    manifest = json.loads(_INTACT_ASSETS_PATH.read_text(encoding="utf-8"))
    target_dir = (
        Path(dest_dir) if dest_dir is not None else data_dir() / "pxd058121_intact_degradomics"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        download_to_file(
            asset["source_url"],
            target_dir / asset["filename"],
            label=f"PXD058121/{asset['filename']}",
            force=force,
            expected_sha256=asset["sha256"],
            expected_size=int(asset["size_bytes"]),
            offline=offline,
        )
    return target_dir


def fetch_pxd018722_sahu_metadata_sources(
    dest_dir: str | Path | None = None,
    *,
    include_fasta: bool = False,
    force: bool = False,
    offline: bool | None = None,
) -> Path:
    """Download and SHA-256-verify pinned PXD018722 (Sahu 2021) direct metadata/FASTA assets."""
    from .downloads import data_dir, download_to_file

    manifest = json.loads(_NATIVE_ASSETS_PATH.read_text(encoding="utf-8"))
    target_dir = (
        Path(dest_dir) if dest_dir is not None else data_dir() / "pxd018722_sahu_2021_psmd2"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    allowed_roles = {"original_metadata", "primary_publication"}
    if include_fasta:
        allowed_roles.add("original_reference")

    for asset in manifest["assets"]:
        if asset["role"] not in allowed_roles or "#" in asset["source_url"]:
            continue
        download_to_file(
            asset["source_url"],
            target_dir / asset["filename"],
            label=f"PXD018722/{asset['filename']}",
            force=force,
            expected_sha256=asset["sha256"],
            expected_size=int(asset["size_bytes"]),
            offline=offline,
        )
    return target_dir


def guard_intact_precursor_scale(
    measurements_df: pd.DataFrame,
    *,
    scale_col: str = "intensity_representation",
    normalizer_col: str = "normalizer_basis",
) -> pd.DataFrame:
    """Enforce PXD058121 (#831) precursor-decay scale & AGC/accumulation-time boundaries.

    Distinguishes within-run product/intact ratios (which cancel shared run gain) from
    uncalibrated cross-acquisition intact-substrate decay, rejects silently calling summed
    deconvoluted input intensities ``TIC``, and rejects dividing by protocol max accumulation
    time without actual per-scan accumulation-time receipts.
    """
    if scale_col not in measurements_df.columns:
        raise ValueError(f"measurements_df must include a {scale_col!r} column")

    out = measurements_df.copy()
    dispositions: list[str] = []
    reasons: list[str] = []
    for _, rec in out.iterrows():
        rep = "" if pd.isna(rec[scale_col]) else str(rec[scale_col]).strip().lower()
        norm = (
            ""
            if normalizer_col not in out.columns or pd.isna(rec.get(normalizer_col))
            else str(rec.get(normalizer_col)).strip().lower()
        )
        raw_receipt = rec.get("has_per_scan_accumulation_receipt", False)
        has_receipt = False if pd.isna(raw_receipt) else bool(raw_receipt)

        if norm in {"sum_deconvoluted_as_tic", "unverified_tic"}:
            dispositions.append("rejected_invalid_tic_assumption")
            reasons.append("sum_of_deconvoluted_input_intensities_is_not_tic")
        elif norm == "protocol_max_accumulation_time" and not has_receipt:
            dispositions.append("rejected_unverified_accumulation_normalization")
            reasons.append("protocol_max_accumulation_time_is_not_per_scan_receipt")
        elif rep == "within_run_product_to_intact_ratio":
            dispositions.append("admitted_within_run_gain_cancelling_ratio")
            reasons.append("shared_run_gain_cancels_within_acquisition")
        elif rep in {"raw_deconvoluted_xtract_intensity", "total_spectrum_normalized", "z_score"}:
            if norm == "independent_stable_standard_calibrated":
                dispositions.append("admitted_calibrated_cross_acquisition_scale")
                reasons.append("independent_stable_reference_calibrated")
            else:
                dispositions.append("deferred_uncalibrated_cross_acquisition_scale")
                reasons.append(f"uncalibrated_for_cross_run_intact_decay:{rep}")
        else:
            dispositions.append("rejected_unknown_intensity_representation")
            reasons.append(f"unrecognized_representation:{rep or 'missing'}")

    out["scale_disposition"] = dispositions
    out["scale_reason"] = reasons
    out["sum_deconvoluted_is_tic"] = False
    out["replicates_r1_r3_are_independent_animals"] = False
    return out


def guard_sahu_2021_endogenous_vs_reporter(
    records_df: pd.DataFrame,
    *,
    peptide_col: str = "peptide",
    protein_col: str = "protein_ids",
    intensity_scale_col: str = "intensity_scale",
) -> pd.DataFrame:
    """Enforce PXD018722 Sahu 2021 (#837) endogenous-vs-reporter, search-scope, and scale guards.

    Explicitly separates endogenous human peptides from engineered reporter ``P22222`` /
    ``NS3CCNB1_HUMAN`` (even though the reporter is tagged ``OX=9606`` in the FASTA!), keeps
    lengths < 8 flagged as outside the actual 8-25 search scope (never biological negatives),
    and keeps raw ``Intensity`` distinct from ``LFQ intensity``.
    """
    required = {peptide_col, protein_col}
    missing = sorted(required - set(records_df.columns))
    if missing:
        raise ValueError(f"records_df is missing required columns: {missing}")

    out = records_df.copy()
    origin_classes: list[str] = []
    search_scopes: list[str] = []
    scale_labels: list[str] = []
    for _, rec in out.iterrows():
        pep = "" if pd.isna(rec[peptide_col]) else str(rec[peptide_col]).strip()
        prots = "" if pd.isna(rec[protein_col]) else str(rec[protein_col]).strip()
        upper_prots = prots.upper()
        tokens = [t.strip() for t in prots.split(";") if t.strip()]

        is_reporter = any(tok in upper_prots for tok in _REPORTER_TOKENS)
        is_rev_or_con = any(t.upper().startswith(("REV__", "CON__")) for t in tokens)
        if is_rev_or_con:
            origin_classes.append("reverse_or_contaminant")
        elif is_reporter:
            has_non_reporter = any(
                not any(rt in t.upper() for rt in _REPORTER_TOKENS) for t in tokens
            )
            origin_classes.append(
                "ambiguous_endogenous_and_reporter"
                if has_non_reporter
                else "engineered_reporter_origin"
            )
        elif not tokens:
            origin_classes.append("missing_parent_annotation")
        else:
            origin_classes.append("endogenous_human_candidate")

        plen = len(pep)
        if plen < 8 or plen > 25:
            search_scopes.append("outside_hi20s_8_25_search_scope")
        else:
            search_scopes.append("within_hi20s_8_25_search_scope")

        raw_scale = (
            ""
            if intensity_scale_col not in out.columns or pd.isna(rec.get(intensity_scale_col))
            else str(rec.get(intensity_scale_col)).strip().lower()
        )
        if raw_scale in {"raw_intensity", "intensity"}:
            scale_labels.append("raw_intensity_distinct_from_lfq")
        elif raw_scale in {"lfq_intensity", "lfq"}:
            scale_labels.append("lfq_intensity_distinct_from_raw")
        else:
            scale_labels.append("unspecified_scale")

    out["origin_class"] = origin_classes
    out["hi20s_search_scope_status"] = search_scopes
    out["intensity_scale_qualified"] = scale_labels
    out["ox_9606_proves_endogenous"] = False
    out["hek293t_is_professional_apc"] = False
    out["psmd2_kd_is_complete_knockout"] = False
    out["zero_or_unsearched_is_biological_negative"] = False
    return out
