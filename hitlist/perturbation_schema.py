"""Structured perturbation contrast schema and APM expression provenance (#945).

Normalizes curated ``pmid_overrides.yaml`` ``ms_samples`` arms and MS observation
frames into paired, contrast-ready perturbation records with:
- ``arm_id`` / ``control_arm_id``
- ``intervention_type`` (``knockout``, ``knockdown``, ``overexpression``,
  ``rescue``, ``cytokine``, ``small_molecule_drug``, ``infection``,
  ``maturation``, ``unperturbed_control``)
- ``target_gene`` (canonical HGNC ortholog symbol via :mod:`hitlist.apm_orthologs`)
- ``variant_or_allele``, ``dose``, ``dose_unit``, ``exposure_duration_hours``,
  ``baseline_engineering``, ``vehicle``
- Orthogonal lineage columns (``cell_lineage``, ``malignancy_status``,
  ``tumor_type``, ``activation_state``)

Also provides standardized expression provenance tags separating
``study_matched_measured`` RNA/proteomics from reference cell-line, tissue, and
primary-immune baselines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .apm_orthologs import map_apm_genes_to_canonical
from .cell_lineage import classify_cell_lineage
from .curation_yaml import load_curation_yaml

_OVERRIDES_PATH = Path(__file__).resolve().parent / "data" / "pmid_overrides.yaml"

INTERVENTION_TYPES: tuple[str, ...] = (
    "knockout",
    "knockdown",
    "overexpression",
    "rescue",
    "cytokine",
    "small_molecule_drug",
    "infection",
    "maturation",
    "unperturbed_control",
)

APM_EXPRESSION_PROVENANCE_TIERS: tuple[str, ...] = (
    "study_matched_measured",
    "reference_cell_line_ccle_depmap",
    "reference_cell_line_encode",
    "reference_tissue_gtex_hpa",
    "reference_primary_immune_dice_monaco",
    "parent_line_proxy",
    "unknown",
)

_BACKEND_TO_PROVENANCE: dict[str, str] = {
    "study_matched_measured": "study_matched_measured",
    "study_matched": "study_matched_measured",
    "depmap_rna": "reference_cell_line_ccle_depmap",
    "ccle": "reference_cell_line_ccle_depmap",
    "depmap": "reference_cell_line_ccle_depmap",
    "encode_rnaseq": "reference_cell_line_encode",
    "encode": "reference_cell_line_encode",
    "packaged_rnaseq": "reference_cell_line_encode",
    "hpa": "reference_tissue_gtex_hpa",
    "gtex": "reference_tissue_gtex_hpa",
    "gtex_hpa": "reference_tissue_gtex_hpa",
    "dice": "reference_primary_immune_dice_monaco",
    "monaco": "reference_primary_immune_dice_monaco",
    "dice_monaco": "reference_primary_immune_dice_monaco",
    "pirlygenes": "reference_tissue_gtex_hpa",
}


def _non_empty(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "none" else text


def _classify_arm_intervention(arm: dict) -> tuple[str, str]:
    """Return ``(intervention_type, target_gene)`` for a curated ``ms_samples`` arm."""
    ko = _non_empty(arm.get("condition_knockout_genes"))
    kd = _non_empty(arm.get("condition_knockdown_genes"))
    oe = _non_empty(arm.get("condition_overexpression_genes"))
    trans = _non_empty(arm.get("condition_transduction"))
    base_eng = _non_empty(arm.get("condition_baseline_engineering"))
    cyt = _non_empty(arm.get("condition_cytokines"))
    drugs = _non_empty(arm.get("condition_drugs"))
    inf = _non_empty(arm.get("condition_infection"))
    ctrl = _non_empty(arm.get("condition_control"))
    cond_text = _non_empty(arm.get("condition")).casefold()

    if ko:
        return "knockout", ";".join(map_apm_genes_to_canonical(ko))
    if kd:
        return "knockdown", ";".join(map_apm_genes_to_canonical(kd))
    if trans and base_eng and ("ko" in base_eng.casefold() or "rescue" in cond_text):
        return "rescue", ";".join(map_apm_genes_to_canonical(trans.split()[0]))
    if oe:
        return "overexpression", ";".join(map_apm_genes_to_canonical(oe))
    if drugs:
        return "small_molecule_drug", ""
    if inf:
        return "infection", ""
    if any(tok in cond_text for tok in ("lps", "maturation", "mature dc", "poly(i:c)")):
        return "maturation", ""
    if cyt:
        return "cytokine", ";".join(map_apm_genes_to_canonical(cyt))
    if trans:
        return "overexpression", ";".join(map_apm_genes_to_canonical(trans.split()[0]))
    if ctrl:
        return "unperturbed_control", ""
    return "unperturbed_control", ""


def extract_structured_perturbation_contrasts(
    pmids: list[int] | set[int] | None = None,
    *,
    overrides_path: Path | None = None,
) -> pd.DataFrame:
    """Extract normalized paired perturbation contrasts from curated ``pmid_overrides.yaml``.

    Every row represents one curated experimental arm linked to its matched
    ``control_arm_id`` (inverted from ``condition_control_for``), with canonical
    ``intervention_type``, ortholog-normalized ``target_gene``, structured
    ``dose`` / ``dose_unit`` / ``exposure_duration_hours``, and orthogonal
    lineage annotations (#945).
    """
    entries = load_curation_yaml(overrides_path or _OVERRIDES_PATH)
    target_pmids = {int(p) for p in pmids} if pmids is not None else None
    rows: list[dict] = []

    for study in entries:
        pmid = int(study["pmid"])
        if target_pmids is not None and pmid not in target_pmids:
            continue
        arms = study.get("ms_samples") or []
        if not arms:
            continue

        # Invert condition_control_for on control arms -> control_arm_id per perturbed arm.
        control_for_arm: dict[str, list[str]] = {}
        for arm in arms:
            arm_id = _non_empty(arm.get("condition_id"))
            ctrl_targets = _non_empty(arm.get("condition_control_for"))
            if arm_id and ctrl_targets:
                for target_id in ctrl_targets.split(";"):
                    target_id = target_id.strip()
                    if target_id:
                        control_for_arm.setdefault(target_id, []).append(arm_id)

        for arm in arms:
            arm_id = _non_empty(arm.get("condition_id"))
            intervention_type, target_gene = _classify_arm_intervention(arm)
            lineage = classify_cell_lineage(
                cell_name=_non_empty(arm.get("sample_label")) or _non_empty(arm.get("source")),
                condition=_non_empty(arm.get("condition")),
                condition_cytokines=_non_empty(arm.get("condition_cytokines")),
                condition_material=_non_empty(arm.get("condition_material")),
                src_cell_line=study.get("override") == "cell_line",
            )
            dose_val = arm.get("condition_dose")
            dur_val = arm.get("condition_duration_hours")
            rows.append(
                {
                    "pmid": pmid,
                    "study_label": _non_empty(study.get("study_label")),
                    "arm_id": arm_id,
                    "sample_label": _non_empty(arm.get("sample_label")),
                    "control_arm_id": ";".join(sorted(set(control_for_arm.get(arm_id, ())))),
                    "is_control_arm": bool(_non_empty(arm.get("condition_control"))),
                    "control_role": _non_empty(arm.get("condition_control")),
                    "intervention_type": intervention_type,
                    "target_gene": target_gene,
                    "variant_or_allele": _non_empty(arm.get("condition_genetic_variants")),
                    "cytokines": _non_empty(arm.get("condition_cytokines")),
                    "drugs": _non_empty(arm.get("condition_drugs")),
                    "vehicle": _non_empty(arm.get("condition_vehicle")),
                    "baseline_engineering": _non_empty(arm.get("condition_baseline_engineering")),
                    "dose": float(dose_val) if dose_val is not None else float("nan"),
                    "dose_unit": _non_empty(arm.get("condition_dose_unit")),
                    "exposure_duration_hours": (
                        float(dur_val) if dur_val is not None else float("nan")
                    ),
                    "mhc_class": _non_empty(arm.get("mhc_class")),
                    "mhc": _non_empty(arm.get("mhc")),
                    "cell_lineage": lineage.cell_lineage,
                    "malignancy_status": lineage.malignancy_status,
                    "tumor_type": lineage.tumor_type,
                    "activation_state": lineage.activation_state,
                }
            )

    return pd.DataFrame(rows)


def tag_expression_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical ``expression_provenance_tier`` to an expression or export frame (#945)."""
    out = df.copy()
    if out.empty:
        out["expression_provenance_tier"] = pd.Series(dtype="string")
        return out

    backends = (
        out["expression_backend"].astype("string").fillna("")
        if "expression_backend" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )
    tiers = (
        out["expression_match_tier"]
        if "expression_match_tier" in out.columns
        else pd.Series(pd.NA, index=out.index)
    )
    sources = (
        out["source_id"].astype("string").fillna("")
        if "source_id" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )

    resolved: list[str] = []
    for backend, tier, src in zip(backends.tolist(), tiers.tolist(), sources.tolist()):
        b_key = str(backend).strip().casefold()
        s_key = str(src).strip().casefold()
        tier_num = None
        if not pd.isna(tier):
            try:
                tier_num = int(tier)
            except (TypeError, ValueError):
                tier_num = None
        if tier_num == 2:
            resolved.append("parent_line_proxy")
        elif b_key in _BACKEND_TO_PROVENANCE:
            resolved.append(_BACKEND_TO_PROVENANCE[b_key])
        elif "depmap" in s_key or "ccle" in s_key:
            resolved.append("reference_cell_line_ccle_depmap")
        elif "encode" in s_key:
            resolved.append("reference_cell_line_encode")
        elif "gtex" in s_key or "hpa" in s_key:
            resolved.append("reference_tissue_gtex_hpa")
        elif "dice" in s_key or "monaco" in s_key:
            resolved.append("reference_primary_immune_dice_monaco")
        elif "study" in s_key or "measured" in s_key:
            resolved.append("study_matched_measured")
        else:
            resolved.append("unknown")

    out["expression_provenance_tier"] = pd.Series(resolved, index=out.index, dtype="string")
    return out
