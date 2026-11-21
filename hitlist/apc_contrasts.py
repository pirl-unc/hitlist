"""Source-qualified APC/cytokine contrast catalog and DLBCL dose-conflict contracts (#904, #930).

Separates isolated cytokine and non-cytokine stimulus contrasts (with verified
same-background comparators) from constant differentiation cocktails, acute
multifactor maturation protocols, constant cytokine backgrounds, unpaired arms,
and non-APC expanded T-cell populations using Hitlist's existing ``#736``
``ms_samples`` / ``condition_*`` / ``condition_control_for`` schema.
"""

from __future__ import annotations

import json
from collections import Counter
from importlib.resources import files
from pathlib import Path

import pandas as pd

from .conditions import split_condition_tokens
from .export import generate_ms_samples_table
from .model_training import TrainingTableSelection, _load_source_frame

DLBCL_CONFLICT_REVISION = "dlbcl_35561310_treatment_conflicts_v1"
APC_PROTOCOL_REVISION = "apc_stimulus_protocol_context_v1"

DLBCL_CONFLICT_SOURCE = files("hitlist").joinpath(
    "data/evidence_revisions/dlbcl_35561310_treatment_conflicts.json"
)
APC_PROTOCOL_SOURCE = files("hitlist").joinpath(
    "data/evidence_revisions/apc_stimulus_protocol_context.json"
)

CONTRAST_ELIGIBILITY_STATUSES: frozenset[str] = frozenset(
    {
        "baseline_comparator_arm",
        "isolated_cytokine_contrast",
        "isolated_stimulus_contrast",
        "differentiation_cocktail_contrast",
        "multifactor_maturation_contrast_on_differentiated_background",
        "multifactor_combination_contrast",
        "constant_cytokine_background_other_contrast",
        "constant_cocktail_context_only",
        "expansion_protocol_non_apc_context_only",
        "unpaired_cytokine_context_only",
        "non_stimulus_arm",
    }
)

_CLASS_POOL_PROVENANCES: frozenset[str] = frozenset(
    {"pmid_class_pool", "pmid_locus_pool", "pmid_serotype_pool"}
)

_MULTIFACTOR_STATUSES: frozenset[str] = frozenset(
    {
        "differentiation_cocktail_contrast",
        "multifactor_maturation_contrast_on_differentiated_background",
        "multifactor_combination_contrast",
    }
)

_CONTEXT_ONLY_EXCLUSIONS: dict[str, str] = {
    "constant_cytokine_background_other_contrast": "constant_cytokine_background_excluded",
    "constant_cocktail_context_only": "constant_cocktail_context_only",
    "expansion_protocol_non_apc_context_only": "expansion_protocol_non_apc_context_only",
    "unpaired_cytokine_context_only": "unpaired_cytokine_context_only",
}


def dlbcl_treatment_conflict_contract(path=DLBCL_CONFLICT_SOURCE) -> dict:
    """Load and validate the pinned DLBCL (PMID 35561310) dose-unit conflict ledger (#930)."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("revision") != DLBCL_CONFLICT_REVISION:
        raise ValueError("Unsupported DLBCL treatment conflict revision")
    if contract.get("pmid") != 35561310 or contract.get("pmcid") != "PMC9327544":
        raise ValueError("Unexpected DLBCL treatment conflict PMID/PMCID")
    inv = contract.get("unit_conversion_invariants", {})
    if (
        inv.get("nmol_per_ml_equals_uM") is not True
        or inv.get("nmol_per_ml_equals_nM") is not False
        or float(inv.get("ug_per_ml_to_ng_per_ml_factor", 0.0)) != 1000.0
    ):
        raise ValueError("Invalid DLBCL unit conversion invariants")
    stmts = contract.get("treatment_statements")
    if not isinstance(stmts, list) or not stmts:
        raise ValueError("DLBCL treatment conflict ledger missing treatment_statements")
    seen: set[str] = set()
    endpoints: set[str] = set()
    for stmt in stmts:
        sid = stmt.get("statement_id")
        if not sid or sid in seen:
            raise ValueError(f"Duplicate or missing treatment statement_id: {sid!r}")
        seen.add(sid)
        endpoints.add(str(stmt.get("endpoint_kind") or ""))
    if not {"peptide_detection", "transcript_expression"} <= endpoints:
        raise ValueError(
            "DLBCL ledger must preserve both peptide_detection and transcript_expression"
        )
    return contract


def apc_stimulus_protocol_contract(path=APC_PROTOCOL_SOURCE) -> dict:
    """Load and validate the pinned APC/stimulus protocol and cell-population ledger (#904)."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("revision") != APC_PROTOCOL_REVISION:
        raise ValueError("Unsupported APC stimulus protocol revision")
    fc = contract.get("factor_classes", {})
    if not {"cytokines", "non_cytokine_stimuli", "small_molecule_drugs"} <= set(fc):
        raise ValueError("APC protocol ledger missing required factor_classes")
    if not isinstance(contract.get("arm_protocols"), dict) or not contract["arm_protocols"]:
        raise ValueError("APC protocol ledger missing arm_protocols")
    return contract


def _token_set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {tok for tok in split_condition_tokens(str(value)) if tok and tok != "none"}


def _infer_is_apc(row: pd.Series, proto: dict) -> bool:
    if "is_measured_apc" in proto:
        return bool(proto["is_measured_apc"])
    text = " ".join(
        str(row.get(c, "") or "") for c in ("sample_label", "source", "sample_group", "condition")
    ).casefold()
    if any(k in text for k in ("til", "t cell", "t-cell", "treg", "tconv", "lymphocyte")):
        return False
    return any(
        k in text
        for k in (
            "dc",
            "dendritic",
            "monocyte",
            "macrophage",
            "b-lcl",
            "b cell",
            "thp-1",
            "dlbcl",
            "b-cell",
            "db ",
            "su-dhl",
        )
    )


def build_stimulus_contrast_catalog(
    samples_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify every arm in cytokine/stimulus studies into explicit contrast vs context roles.

    Reads Hitlist's existing ``ms_samples`` curation (``condition_cytokines``,
    ``condition_stimulation``, ``condition_drugs``, ``condition_control_for``)
    combined with the pinned protocol and dose-conflict ledgers.
    """
    samples = generate_ms_samples_table() if samples_df is None else samples_df.copy()
    apc_contract = apc_stimulus_protocol_contract()
    dlbcl_contract = dlbcl_treatment_conflict_contract()
    arm_protocols: dict[str, dict] = apc_contract["arm_protocols"]

    if samples.empty:
        return pd.DataFrame()

    # Identify all studies with at least one cytokine or stimulation arm.
    has_cyt = samples["condition_cytokines"].astype("string").fillna("").str.strip().ne("")
    has_stim = samples["condition_stimulation"].astype("string").fillna("").str.strip().ne("")
    relevant_pmids = sorted(set(samples.loc[has_cyt | has_stim, "pmid"].astype(int)))

    records: list[dict] = []
    for pmid in relevant_pmids:
        study = samples[samples["pmid"].astype(int) == pmid].copy()
        by_cid: dict[str, pd.Series] = {}
        comparators_for: dict[str, list[str]] = {}
        targets_of: dict[str, list[str]] = {}
        for _, row in study.iterrows():
            cid = str(row.get("condition_id", "") or "").strip()
            if not cid:
                continue
            by_cid[cid] = row
            cfor = sorted(_token_set(row.get("condition_control_for")))
            if cfor:
                targets_of[cid] = cfor
                for target in cfor:
                    comparators_for.setdefault(target, []).append(cid)

        for cid, row in by_cid.items():
            arm_key = f"{pmid}:{cid}"
            proto = arm_protocols.get(arm_key, {})
            arm_cyt = _token_set(row.get("condition_cytokines"))
            arm_stim = _token_set(row.get("condition_stimulation"))
            arm_drugs = _token_set(row.get("condition_drugs"))
            arm_vehicle = _token_set(row.get("condition_vehicle"))
            ctrls = sorted(set(comparators_for.get(cid, [])))
            is_ctrl_for = sorted(set(targets_of.get(cid, [])))
            proto_role = str(proto.get("protocol_role") or "")

            delta_cyt: set[str] = set()
            delta_stim: set[str] = set()
            delta_drugs: set[str] = set()
            shared_bg_cyt: set[str] = set()
            shared_bg_stim: set[str] = set()

            if ctrls:
                # Compare against the first matched comparator (all listed comparators
                # share the same baseline system in curated studies).
                ctrl_row = by_cid.get(ctrls[0])
                ctrl_cyt = (
                    _token_set(ctrl_row.get("condition_cytokines"))
                    if ctrl_row is not None
                    else set()
                )
                ctrl_stim = (
                    _token_set(ctrl_row.get("condition_stimulation"))
                    if ctrl_row is not None
                    else set()
                )
                ctrl_drugs = (
                    _token_set(ctrl_row.get("condition_drugs")) if ctrl_row is not None else set()
                )
                delta_cyt = arm_cyt - ctrl_cyt
                delta_stim = arm_stim - ctrl_stim
                delta_drugs = arm_drugs - ctrl_drugs
                shared_bg_cyt = arm_cyt & ctrl_cyt
                shared_bg_stim = arm_stim & ctrl_stim

                if not arm_cyt and not arm_stim:
                    status = "non_stimulus_arm"
                elif not delta_cyt and not delta_stim:
                    status = "constant_cytokine_background_other_contrast"
                elif len(delta_cyt) == 1 and not delta_stim and not delta_drugs:
                    status = "isolated_cytokine_contrast"
                elif len(delta_stim) == 1 and not delta_cyt and not delta_drugs:
                    status = "isolated_stimulus_contrast"
                elif proto_role == "maturation_cocktail_on_differentiated_background" or bool(
                    shared_bg_cyt
                ):
                    status = "multifactor_maturation_contrast_on_differentiated_background"
                elif proto_role == "differentiation_cocktail_with_baseline_comparator":
                    status = "differentiation_cocktail_contrast"
                else:
                    status = "multifactor_combination_contrast"
            else:
                if not arm_cyt and not arm_stim:
                    if is_ctrl_for:
                        status = "baseline_comparator_arm"
                    else:
                        status = "non_stimulus_arm"
                elif proto_role == "expansion_protocol_non_apc_context_only":
                    status = "expansion_protocol_non_apc_context_only"
                elif (
                    proto_role
                    in {"constant_cocktail_no_comparator", "non_cytokine_activation_no_comparator"}
                    or len(arm_cyt | arm_stim) > 1
                ):
                    status = "constant_cocktail_context_only"
                else:
                    # Single cytokine/stimulus arm without a matched comparator (e.g.
                    # PMID 31844290 unpaired IFNG lines, 33460454 DFT1 IFNG, 33968037 Class II,
                    # 30484775 TAPBPR WT positive control carrying constant IFNG background,
                    # 34129938 MC38 IFNG background, 35051231 PMA+infection).
                    if (is_ctrl_for and pmid == 30484775) or pmid in {34129938, 35051231}:
                        status = "constant_cytokine_background_other_contrast"
                    else:
                        status = "unpaired_cytokine_context_only"

            has_dose_conflict = pmid == dlbcl_contract["pmid"] and bool(arm_cyt | arm_drugs)
            records.append(
                {
                    "pmid": pmid,
                    "condition_id": cid,
                    "sample_label": str(row.get("sample_label", "") or ""),
                    "sample_group": str(row.get("sample_group", "") or ""),
                    "mhc_class": str(row.get("mhc_class", "") or ""),
                    "condition_cytokines": ";".join(sorted(arm_cyt)),
                    "condition_stimulation": ";".join(sorted(arm_stim)),
                    "condition_drugs": ";".join(sorted(arm_drugs)),
                    "condition_vehicle": ";".join(sorted(arm_vehicle)),
                    "condition_combination": str(row.get("condition_combination", "") or ""),
                    "comparator_condition_ids": ";".join(ctrls),
                    "condition_control_for": ";".join(is_ctrl_for),
                    "is_comparator_for": ";".join(is_ctrl_for),
                    "delta_cytokines": ";".join(sorted(delta_cyt)),
                    "delta_stimuli": ";".join(sorted(delta_stim)),
                    "delta_drugs": ";".join(sorted(delta_drugs)),
                    "shared_background_cytokines": ";".join(sorted(shared_bg_cyt)),
                    "shared_background_stimuli": ";".join(sorted(shared_bg_stim)),
                    "contrast_eligibility_status": status,
                    "measured_population": str(proto.get("measured_population", "") or ""),
                    "cell_ontology_id": str(proto.get("cell_ontology_id", "") or ""),
                    "is_measured_apc": _infer_is_apc(row, proto),
                    "partner_cell_population": str(proto.get("partner_cell_population", "") or ""),
                    "differentiation_protocol": str(
                        proto.get("differentiation_protocol", "") or ""
                    ),
                    "acute_stimulus_protocol": str(proto.get("acute_stimulus_protocol", "") or ""),
                    "protocol_source_locator": str(
                        proto.get("source_locator") or row.get("condition_reference") or ""
                    ),
                    "dose_conflict_status": (
                        "source_dose_unit_conflict_documented" if has_dose_conflict else ""
                    ),
                    "dose_conflict_revision": (
                        DLBCL_CONFLICT_REVISION if has_dose_conflict else ""
                    ),
                }
            )

    return pd.DataFrame(records)


def prepare_stimulus_contrast_data(
    source: pd.DataFrame | str | Path,
    *,
    target_factor: str | None = None,
    allow_multifactor_contrasts: bool = False,
    include_context_only: bool = False,
    exclude_class_pool: bool = True,
    exclude_non_apc: bool = False,
    samples_df: pd.DataFrame | None = None,
) -> TrainingTableSelection:
    """Select source-qualified cytokine/stimulus contrast observations with 100% row conservation.

    Guarantees that:
    - Class-pool and unattributed/mixed-population rows never receive guessed treatments.
    - Multi-arm shared rows are routed to ``excluded`` with
      ``shared_multi_arm_requires_edge_expansion`` so callers use
      :func:`hitlist.export.expand_condition_memberships` rather than conflating shared
      detection with contradictory supervision.
    - Constant differentiation cocktails, constant cytokine backgrounds, unpaired arms,
      and non-APC expanded T cells are excluded from isolated contrasts by default.
    - Documented DLBCL dose-unit conflicts (``#930``) remain visible on accepted Figure 5
      rows without discarding usable detection evidence.
    """
    df, bundle_info = _load_source_frame(source, default_evidence_kind="ms")
    catalog = build_stimulus_contrast_catalog(samples_df=samples_df)
    cat_by_key = {
        (int(r["pmid"]), str(r["condition_id"])): r for r in catalog.to_dict(orient="records")
    }
    # Map each baseline comparator arm to the delta factors of its eligible targets.
    target_norm = str(target_factor).strip() if target_factor else None

    eligible_targets_by_ctrl: dict[tuple[int, str], set[str]] = {}
    for (pmid, _cid), rec in cat_by_key.items():
        status = rec["contrast_eligibility_status"]
        if status in {"isolated_cytokine_contrast", "isolated_stimulus_contrast"} or (
            allow_multifactor_contrasts and status in _MULTIFACTOR_STATUSES
        ):
            factors = _token_set(rec["delta_cytokines"]) | _token_set(rec["delta_stimuli"])
            for ctrl_cid in _token_set(rec["comparator_condition_ids"]):
                eligible_targets_by_ctrl.setdefault((pmid, ctrl_cid), set()).update(factors)

    accepted_rows: list[dict] = []
    excluded_rows: list[dict] = []

    n_attributed_single = 0
    n_shared_multi = 0
    n_ambiguous_or_unattr = 0
    n_dose_conflict = 0

    for row in df.to_dict(orient="records"):
        raw_pmid = row.get("pmid")
        pmid_int: int | None = None
        if raw_pmid is not None and not (isinstance(raw_pmid, float) and pd.isna(raw_pmid)):
            try:
                pmid_int = int(raw_pmid)
            except (TypeError, ValueError):
                pmid_int = None

        cid = str(row.get("condition_id") or "").strip()
        supp = _token_set(row.get("supported_condition_ids"))
        mhc_origin = str(row.get("sample_mhc_origin") or "").strip()
        allele_prov = str(row.get("mhc_allele_provenance") or "").strip()

        if cid:
            n_attributed_single += 1
        elif len(supp) > 1:
            n_shared_multi += 1
        else:
            n_ambiguous_or_unattr += 1

        rec = cat_by_key.get((pmid_int, cid)) if (pmid_int is not None and cid) else None
        enriched = dict(row)
        if rec is not None:
            for col in (
                "contrast_eligibility_status",
                "comparator_condition_ids",
                "condition_control_for",
                "is_comparator_for",
                "condition_vehicle",
                "delta_cytokines",
                "delta_stimuli",
                "delta_drugs",
                "shared_background_cytokines",
                "shared_background_stimuli",
                "measured_population",
                "cell_ontology_id",
                "is_measured_apc",
                "partner_cell_population",
                "differentiation_protocol",
                "acute_stimulus_protocol",
                "dose_conflict_status",
                "dose_conflict_revision",
            ):
                enriched[col] = rec[col]
        else:
            enriched.setdefault("contrast_eligibility_status", "")
            enriched.setdefault("comparator_condition_ids", "")
            enriched.setdefault("condition_control_for", "")
            enriched.setdefault("is_comparator_for", "")
            enriched.setdefault("condition_vehicle", "")
            enriched.setdefault("delta_cytokines", "")
            enriched.setdefault("delta_stimuli", "")
            enriched.setdefault("delta_drugs", "")
            enriched.setdefault("shared_background_cytokines", "")
            enriched.setdefault("shared_background_stimuli", "")
            enriched.setdefault("measured_population", "")
            enriched.setdefault("cell_ontology_id", "")
            enriched.setdefault("is_measured_apc", pd.NA)
            enriched.setdefault("partner_cell_population", "")
            enriched.setdefault("differentiation_protocol", "")
            enriched.setdefault("acute_stimulus_protocol", "")
            enriched.setdefault(
                "dose_conflict_status",
                "source_dose_unit_conflict_documented" if pmid_int == 35561310 else "",
            )
            enriched.setdefault(
                "dose_conflict_revision",
                DLBCL_CONFLICT_REVISION if pmid_int == 35561310 else "",
            )

        if enriched["dose_conflict_status"]:
            n_dose_conflict += 1

        reason = ""
        if exclude_class_pool and (
            mhc_origin == "class_pool" or allele_prov in _CLASS_POOL_PROVENANCES
        ):
            reason = "class_pool_attribution_excluded"
        elif not cid:
            if len(supp) > 1:
                reason = "shared_multi_arm_requires_edge_expansion"
            else:
                reason = "unattributed_or_mixed_population_excluded"
        elif rec is None:
            reason = "no_stimulus_context_in_study"
        elif exclude_non_apc and not bool(rec["is_measured_apc"]):
            reason = "non_apc_population_excluded"
        else:
            status = rec["contrast_eligibility_status"]
            if status == "non_stimulus_arm":
                reason = "no_stimulus_context_in_arm"
            elif status in _CONTEXT_ONLY_EXCLUSIONS and not include_context_only:
                reason = _CONTEXT_ONLY_EXCLUSIONS[status]
            elif status in _MULTIFACTOR_STATUSES and not allow_multifactor_contrasts:
                reason = "multifactor_contrast_excluded"
            elif status == "baseline_comparator_arm":
                ctrl_factors = eligible_targets_by_ctrl.get((pmid_int, cid), set())
                if not ctrl_factors:
                    reason = "no_eligible_stimulus_target_for_comparator"
                elif target_norm and target_norm not in ctrl_factors:
                    reason = "target_factor_not_present"
            elif target_norm:
                arm_factors = (
                    _token_set(rec["delta_cytokines"])
                    | _token_set(rec["delta_stimuli"])
                    | (
                        _token_set(rec["condition_cytokines"])
                        | _token_set(rec["condition_stimulation"])
                        if include_context_only
                        else set()
                    )
                )
                if target_norm not in arm_factors:
                    reason = "target_factor_not_present"

        if reason:
            enriched["exclusion_reason"] = reason
            excluded_rows.append(enriched)
        else:
            accepted_rows.append(enriched)

    accepted = pd.DataFrame(accepted_rows)
    excluded = pd.DataFrame(excluded_rows)
    summary = {
        "task": "apc_stimulus_contrast",
        "source_bundle": bundle_info,
        "target_factor": target_norm,
        "allow_multifactor_contrasts": bool(allow_multifactor_contrasts),
        "include_context_only": bool(include_context_only),
        "exclude_class_pool": bool(exclude_class_pool),
        "exclude_non_apc": bool(exclude_non_apc),
        "n_input_rows": len(df),
        "n_accepted_rows": len(accepted),
        "n_excluded_rows": len(excluded),
        "n_attributed_single_arm_rows": n_attributed_single,
        "n_shared_multi_arm_rows": n_shared_multi,
        "n_ambiguous_or_unattributed_rows": n_ambiguous_or_unattr,
        "n_dose_conflict_annotated_rows": n_dose_conflict,
        "by_contrast_eligibility_status": (
            dict(sorted(Counter(accepted["contrast_eligibility_status"].astype("string")).items()))
            if not accepted.empty
            else {}
        ),
        "by_mhc_class": (
            dict(sorted(Counter(accepted["mhc_class"].astype("string")).items()))
            if not accepted.empty and "mhc_class" in accepted.columns
            else {}
        ),
        "by_exclusion_reason": (
            dict(sorted(Counter(excluded["exclusion_reason"].astype("string")).items()))
            if not excluded.empty
            else {}
        ),
    }
    return TrainingTableSelection(accepted=accepted, excluded=excluded, summary=summary)
