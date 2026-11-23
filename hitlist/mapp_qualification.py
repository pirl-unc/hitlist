"""Unified MAPP qualification, independent precursor recipe, and Goldberg source fetcher.

Closes remaining qualification, recipe, rights, and catalog scope across:
- #752 (Wolf-Levy MAPP PXD009092 publisher rights + reference/filter contract)
- #757 (PXD009360 parallel tryptic proteome independent precursor-estimation recipe & pairing guard)
- #759 (PXD009508/PXD007263 controlled MAPP detection-mask / yield-vs-composition summary)
- #820 (PXD009438 PBMC vs blood-serum conflict & cross-project material-link precision graph)
- #825 (Goldberg 2025 PXD052314 Table 3 pooled-slot guard & catalog-managed source fetcher)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

_UNIFIED_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "mapp_unified_qualification.json"
)
_GOLDBERG_PATH = Path(__file__).parent / "data" / "evidence_revisions" / "goldberg_mapp.json"
_EXCEL_DATE_RE = re.compile(
    r"^(?:\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{1,2}"
    r"|\d{4}-\d{2}-\d{2}|\d{5}(?:\.0+)?)$",
    re.IGNORECASE,
)


def mapp_unified_qualification_contract() -> dict[str, Any]:
    """Return the unified qualification contract across all 5 MAPP and 3 tryptic deposits."""
    return json.loads(_UNIFIED_PATH.read_text(encoding="utf-8"))


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _opt_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def guard_mapp_material_pairing(
    links_df: pd.DataFrame,
    *,
    allow_condition_surrogate: bool = False,
) -> pd.DataFrame:
    """Enforce source-backed material-link precision rules for cross-modal MAPP pairing (#757/#820).

    Explicitly rejects order-, replicate-number-, cell-line-, peptide-, or abundance-correlation
    heuristics, defers unresolved/project-only links, and admits ``condition_surrogate`` only when
    the caller explicitly opts in (never relabeling it as exact specimen/aliquot pairing).
    """
    if "material_link_precision" not in links_df.columns:
        raise ValueError("links_df must include a 'material_link_precision' column")

    rules = mapp_unified_qualification_contract()["material_pairing_rules"]
    exact_set = set(rules["allowed_exact_precisions"])
    surrogate_set = set(rules["opt_in_surrogate_precisions"])
    unresolved_set = set(rules["deferred_unresolved_precisions"])
    forbidden_set = set(rules["forbidden_heuristic_precisions"])

    out = links_df.copy()
    dispositions: list[str] = []
    reasons: list[str] = []
    exact_flags: list[bool] = []
    for raw_prec in out["material_link_precision"]:
        prec = _clean_str(raw_prec).lower() or "unresolved"
        if prec in exact_set:
            dispositions.append("admitted_exact_pairing")
            reasons.append(f"source_backed_{prec}")
            exact_flags.append(True)
        elif prec in surrogate_set:
            if allow_condition_surrogate:
                dispositions.append("admitted_condition_surrogate")
                reasons.append("caller_opted_into_condition_surrogate_not_exact_specimen")
            else:
                dispositions.append("deferred_condition_surrogate")
                reasons.append("condition_surrogate_requires_explicit_opt_in")
            exact_flags.append(False)
        elif prec in forbidden_set:
            dispositions.append("rejected_unsafe_heuristic")
            reasons.append(f"forbidden_heuristic_pairing:{prec}")
            exact_flags.append(False)
        elif prec in unresolved_set:
            dispositions.append("deferred_unresolved_material_link")
            reasons.append(f"unresolved_material_link:{prec}")
            exact_flags.append(False)
        else:
            dispositions.append("rejected_unknown_precision")
            reasons.append(f"unrecognized_material_link_precision:{prec}")
            exact_flags.append(False)

    out["pairing_disposition"] = dispositions
    out["pairing_reason"] = reasons
    out["is_exact_physical_pairing"] = exact_flags
    return out


def mapp_precursor_estimation_recipe(
    protein_groups_df: pd.DataFrame,
    *,
    material_links_df: pd.DataFrame | None = None,
    allow_condition_surrogate: bool = False,
    allow_ambiguous_protein_groups: bool = False,
) -> dict[str, Any]:
    """Declared independent tryptic precursor-abundance recipe (#757).

    Keeps MaxQuant protein-group LFQ (``source_relative_abundance_not_copies_per_cell``) strictly
    separate from raw intensity and absolute cellular concentration, preserves ambiguous multi-protein
    groups and zero/missing software outputs without imputation, and enforces material-link precision
    whenever ``material_links_df`` (keyed by ``run_id`` or ``acquisition_id``) is supplied.
    """
    required = {"protein_ids"}
    missing = sorted(required - set(protein_groups_df.columns))
    if missing:
        raise ValueError(f"protein_groups_df is missing required columns: {missing}")

    link_lookup: dict[str, tuple[str, str, bool]] = {}
    link_key_col: str | None = None
    if material_links_df is not None:
        guarded_links = guard_mapp_material_pairing(
            material_links_df, allow_condition_surrogate=allow_condition_surrogate
        )
        for cand in ("acquisition_id", "run_id", "raw_file"):
            if cand in guarded_links.columns and cand in protein_groups_df.columns:
                link_key_col = cand
                break
        if link_key_col is None:
            raise ValueError(
                "material_links_df and protein_groups_df must share 'acquisition_id', 'run_id', "
                "or 'raw_file'"
            )
        for row in guarded_links.itertuples(index=False):
            key_val = _clean_str(getattr(row, link_key_col))
            link_lookup[key_val] = (
                row.pairing_disposition,
                row.pairing_reason,
                bool(row.is_exact_physical_pairing),
            )

    rows: list[dict[str, Any]] = []
    for row_num, (_, rec) in enumerate(protein_groups_df.iterrows(), start=2):
        prot_ids = _clean_str(rec["protein_ids"])
        members = [p.strip() for p in prot_ids.split(";") if p.strip()]
        n_proteins = len(members)
        is_rev = _clean_str(rec.get("reverse")) in {"+", "True", "true", "1"} or any(
            m.startswith("REV__") for m in members
        )
        is_contam = _clean_str(rec.get("potential_contaminant")) in {
            "+",
            "True",
            "true",
            "1",
        } or any(m.startswith("CON__") for m in members)
        lfq_val = _opt_float(rec.get("lfq_intensity"))
        raw_val = _opt_float(rec.get("raw_intensity"))
        has_pos_lfq = lfq_val is not None and lfq_val > 0.0
        has_pos_raw = raw_val is not None and raw_val > 0.0

        link_disp = "no_material_link_table_supplied"
        link_reason = "standalone_tryptic_proteome_only"
        is_exact_link = False
        if link_key_col is not None:
            key_val = _clean_str(rec.get(link_key_col))
            link_disp, link_reason, is_exact_link = link_lookup.get(
                key_val,
                ("deferred_unresolved_material_link", "missing_from_material_links", False),
            )

        if is_rev or is_contam:
            disposition = "excluded"
            reason = "reverse_or_contaminant"
        elif n_proteins == 0:
            disposition = "excluded"
            reason = "empty_protein_group_id"
        elif not has_pos_lfq and not has_pos_raw:
            disposition = "excluded"
            reason = "zero_or_missing_intensity_not_imputed"
        elif link_key_col is not None and link_disp not in {
            "admitted_exact_pairing",
            "admitted_condition_surrogate",
        }:
            disposition = "deferred_unresolved_pairing"
            reason = link_reason
        elif n_proteins > 1 and not allow_ambiguous_protein_groups:
            disposition = "ambiguous_protein_group"
            reason = "multi_protein_group_retained_not_collapsed"
        elif has_pos_lfq:
            disposition = "admitted_precursor"
            reason = "positive_maxquant_group_lfq_relative_abundance"
        else:
            disposition = "excluded"
            reason = "raw_intensity_only_no_positive_lfq_do_not_mix_scales"

        rows.append(
            {
                "source_row": row_num,
                "protein_ids": prot_ids,
                "leading_protein_id": members[0] if members else "",
                "n_proteins_in_group": n_proteins,
                "is_ambiguous_protein_group": n_proteins > 1,
                "is_reverse": is_rev,
                "is_contaminant": is_contam,
                "lfq_intensity": lfq_val,
                "log2_lfq_intensity": math.log2(lfq_val)
                if has_pos_lfq and lfq_val
                else float("nan"),
                "raw_intensity": raw_val,
                "log2_raw_intensity": math.log2(raw_val)
                if has_pos_raw and raw_val
                else float("nan"),
                "abundance_unit": "maxquant_group_lfq_source_relative_not_copies_per_cell",
                "is_absolute_concentration": False,
                "pairing_disposition": link_disp,
                "pairing_reason": link_reason,
                "is_exact_physical_pairing": is_exact_link,
                "recipe_disposition": disposition,
                "recipe_reason": reason,
            }
        )

    out = pd.DataFrame(rows)
    admitted = out[out["recipe_disposition"] == "admitted_precursor"].reset_index(drop=True)
    ambiguous = out[out["recipe_disposition"] == "ambiguous_protein_group"].reset_index(drop=True)
    deferred = out[out["recipe_disposition"] == "deferred_unresolved_pairing"].reset_index(
        drop=True
    )
    exclusions = out[out["recipe_disposition"] == "excluded"].reset_index(drop=True)

    summary = {
        "n_source_rows": len(out),
        "n_admitted_precursors": len(admitted),
        "n_ambiguous_protein_groups": len(ambiguous),
        "n_deferred_unresolved_pairing": len(deferred),
        "n_excluded": len(exclusions),
        "n_exact_physical_pairings": int(out["is_exact_physical_pairing"].sum()),
        "allow_condition_surrogate": bool(allow_condition_surrogate),
        "allow_ambiguous_protein_groups": bool(allow_ambiguous_protein_groups),
        "maxquant_lfq_is_copies_per_cell": False,
        "mix_lfq_and_raw_scales_allowed": False,
    }
    return {
        "admitted_precursors": admitted,
        "ambiguous_precursors": ambiguous,
        "deferred_unresolved_pairing": deferred,
        "exclusions": exclusions,
        "all_records": out,
        "summary": summary,
    }


def mapp_detection_mask_summary(
    results_df: pd.DataFrame,
    *,
    peptide_col: str = "peptide",
    arm_col: str = "arm_label",
    run_col: str = "raw_file",
    intensity_col: str = "intensity",
) -> dict[str, Any]:
    """Compute separate total-yield and per-run detection-mask summaries for controlled MAPP (#759).

    Prevents collapsing global inhibitor-induced total yield drops into unit-sum compositions
    and keeps unobserved/zero entries distinct from biological negatives.
    """
    required = {peptide_col, arm_col, run_col, intensity_col}
    missing = sorted(required - set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    df = results_df.copy()
    vals = pd.to_numeric(df[intensity_col], errors="coerce")
    df["_is_positive"] = vals.gt(0).fillna(False)
    df["_is_zero"] = vals.eq(0).fillna(False)
    df["_is_missing"] = vals.isna()
    df["_pos_intensity"] = vals.where(df["_is_positive"], 0.0)

    arm_summaries: dict[str, dict[str, Any]] = {}
    complete_case_sets: dict[str, set[str]] = {}
    for arm, grp in df.groupby(arm_col, sort=True):
        n_runs = int(grp[run_col].nunique())
        pos_per_pep = grp.groupby(peptide_col)["_is_positive"].sum()
        complete_peps = (
            set(pos_per_pep[pos_per_pep == n_runs].index.tolist()) if n_runs > 0 else set()
        )
        any_pos_peps = set(pos_per_pep[pos_per_pep > 0].index.tolist())
        complete_case_sets[str(arm)] = complete_peps
        arm_summaries[str(arm)] = {
            "n_runs": n_runs,
            "n_total_cells": len(grp),
            "n_positive_cells": int(grp["_is_positive"].sum()),
            "n_zero_cells": int(grp["_is_zero"].sum()),
            "n_missing_cells": int(grp["_is_missing"].sum()),
            "n_peptides_any_positive": len(any_pos_peps),
            "n_peptides_complete_in_arm": len(complete_peps),
            "total_detected_signal": float(grp["_pos_intensity"].sum()),
            "mean_positive_signal": (
                float(vals[df.index.isin(grp.index) & df["_is_positive"]].mean())
                if grp["_is_positive"].any()
                else float("nan")
            ),
        }

    shared_complete = (
        set.intersection(*complete_case_sets.values()) if complete_case_sets else set()
    )
    return {
        "arms": arm_summaries,
        "n_arms": len(arm_summaries),
        "n_shared_complete_case_peptides": len(shared_complete),
        "unit_sum_composition_captures_global_yield_drop": False,
        "missing_or_zero_is_biological_negative": False,
    }


def guard_goldberg_table3_pooled_slots(
    table3_df: pd.DataFrame,
    *,
    treatment_col: str = "Treatment",
    gene_col: str = "Gene names",
) -> dict[str, Any]:
    """Audit Goldberg 2025 Table 3 pooled slots without collapsing repeated labels or guessing dates (#825).

    Preserves ordered repeated labels/tokens as lists (never collapsing to a dict), detects
    Excel-date-corrupted gene tokens, and explicitly marks the pooled table ineligible for
    supervised cohort pooling until run/unit/reference crosswalks are recovered.
    """
    n_rows = len(table3_df)
    n_rows_with_repeated_labels = 0
    n_date_styled_gene_cells = 0
    flagged_gene_rows: list[int] = []

    if treatment_col in table3_df.columns:
        for raw in table3_df[treatment_col]:
            text = _clean_str(raw)
            if not text:
                continue
            labels = [tok.split(":")[0].strip() for tok in re.split(r"[;,|]", text) if tok.strip()]
            if len(labels) != len(set(labels)):
                n_rows_with_repeated_labels += 1

    if gene_col in table3_df.columns:
        for pos, raw in enumerate(table3_df[gene_col]):
            text = _clean_str(raw)
            if not text:
                continue
            if any(_EXCEL_DATE_RE.match(tok.strip()) for tok in text.split(";") if tok.strip()):
                n_date_styled_gene_cells += 1
                flagged_gene_rows.append(pos)

    return {
        "source_dataset": "PXD052314",
        "n_source_rows": n_rows,
        "n_rows_with_repeated_slot_labels": n_rows_with_repeated_labels,
        "n_date_styled_gene_cells": n_date_styled_gene_cells,
        "flagged_gene_row_indices": flagged_gene_rows,
        "collapse_repeated_labels_to_dict_allowed": False,
        "guess_date_corrupted_gene_symbols_allowed": False,
        "eligible_for_supervised_cohort": False,
    }


def fetch_goldberg_mapp_sources(
    dest_dir: str | Path | None = None,
    *,
    include_large_archives: bool = False,
    force: bool = False,
    offline: bool | None = None,
) -> Path:
    """Download and SHA-256-verify pinned Goldberg 2025 (PXD052314) assets (#825).

    By default fetches the 4 lightweight metadata/inventory assets (~83 KB total: ``project.json``,
    ``files.json``, ``Data_Deposition_Table.xlsx``, ``PXD052314_community_annotated.sdrf.tsv``).
    Pass ``include_large_archives=True`` to also download ``supplement.zip`` (48 MB) and
    ``EC_infection_Degradome_MAPP_Peptides.zip`` (356 MB).
    """
    from .downloads import data_dir, download_to_file

    contract = json.loads(_GOLDBERG_PATH.read_text(encoding="utf-8"))
    target_dir = Path(dest_dir) if dest_dir is not None else data_dir() / "pxd052314_goldberg_mapp"
    target_dir.mkdir(parents=True, exist_ok=True)
    large_roles = {"complete_publisher_supplement", "original_mapp_search_results"}

    for asset in contract["assets"]:
        if not include_large_archives and asset["role"] in large_roles:
            continue
        url = asset["source_url"]
        if url.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            url = "https://" + url[len("ftp://") :]
        download_to_file(
            url,
            target_dir / asset["filename"],
            label=f"PXD052314/{asset['filename']}",
            force=force,
            expected_sha256=asset["sha256"],
            expected_size=int(asset["size_bytes"]),
            offline=offline,
        )
    return target_dir
