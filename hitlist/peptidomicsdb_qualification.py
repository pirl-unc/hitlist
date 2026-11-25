"""PeptidomicsDB source qualification, capture-vs-compartment and reanalysis guards (#811).

Preserves original PeptidomicsDB category/search-enzyme assertions separately from
primary-source verified preparation semantics so HLA-IP datasets tagged as
``Intracellular peptidomics`` (PPD017/PXD021177) or generic ``Peptidomics``
(PPD070/PXD009801) can never enter native free-peptide availability cohorts.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

CONTRACT_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "peptidomicsdb_qualification.json"
)
VALID_PEAKS_METHODS = frozenset({"DB Search", "DeepNovo", "Homolog", "Sequence Variants"})
_MHC_CATEGORY_TOKENS = ("mhc-i", "mhc-ii", "immunopeptidomics", "hla")


def peptidomicsdb_source_contract() -> dict[str, Any]:
    """Return a validated deep copy of the PeptidomicsDB qualification contract (#811)."""
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported PeptidomicsDB contract schema: {data.get('schema_version')}")
    if data.get("current_n_qualified_native_apc_cohorts", 0) != 0:
        raise ValueError("PeptidomicsDB contract cannot claim qualified APC cohorts without audit")
    return copy.deepcopy(data)


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _parse_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"na", "nan", "null", "none"}:
        return None
    val = float(text)
    if not math.isfinite(val):
        return None
    return val


def qualify_peptidomicsdb_samples(
    samples_df: pd.DataFrame,
    *,
    projects_df: pd.DataFrame | None = None,
    primary_protocol_verifications: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Qualify PeptidomicsDB sample metadata rows for native APC free-peptide eligibility.

    Enforces:
    1. Orthogonal fields: ``specimen_compartment``, ``molecular_capture_enrichment``,
       ``post_extraction_digestion``, and ``ms_search_enzyme`` (``Enzyme=None`` never
       proves native extraction!).
    2. Hard primary-source exclusions for ``PPD017/PXD021177`` (PMID 34211107 HLA-IP),
       ``PPD070/PXD009801`` (PMID 24715725 dynamic SILAC HLA-I), and ``PPD016/PXD033369``
       (enteroendocrine cells, non-APC).
    3. Any row carrying an MHC/immunopeptidomics tag is excluded as MHC-selected.
    4. Any unverified non-MHC sample remains explicitly ``deferred`` until its primary
       preparation protocol is verified via ``primary_protocol_verifications``.
    5. Raw-file names and sample counts never imply specimen independence
       (``specimen_independence_status="unknown"``).
    """
    if not isinstance(samples_df, pd.DataFrame):
        raise TypeError("samples_df must be a pandas DataFrame")

    contract = peptidomicsdb_source_contract()
    reviewed = contract["primary_source_reviewed_dispositions"]
    accession_to_reviewed = {
        rec["original_accession"].upper(): rec
        for rec in reviewed.values()
        if rec.get("original_accession")
    }
    verified_overrides = primary_protocol_verifications or {}

    project_accessions: dict[str, str] = {}
    if projects_df is not None and not projects_df.empty:
        for _, prow in projects_df.iterrows():
            pid = _clean_str(prow.get("PeptidomicsDB project ID") or prow.get("project_id")).upper()
            acc = _clean_str(
                prow.get("Original project accession") or prow.get("original_accession")
            ).upper()
            if pid and acc:
                project_accessions[pid] = acc

    out_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for idx, row in samples_df.iterrows():
        project_id = _clean_str(
            row.get("PeptidomicsDB project ID") or row.get("project_id")
        ).upper()
        sample_id = _clean_str(row.get("PeptidomicsDB sample ID") or row.get("sample_id")).upper()
        original_accession = _clean_str(
            row.get("Original project accession") or row.get("original_accession")
        ).upper() or project_accessions.get(project_id, "")
        raw_file = _clean_str(row.get("Raw file") or row.get("raw_file"))
        condition = _clean_str(row.get("Condition") or row.get("condition"))
        species = _clean_str(row.get("Species") or row.get("species"))
        tissue = _clean_str(row.get("Tissue") or row.get("tissue"))
        disease = _clean_str(row.get("Diseases/status") or row.get("disease_status"))
        category_raw = _clean_str(row.get("Peptidomics category") or row.get("source_category_raw"))
        search_enzyme_raw = _clean_str(row.get("Enzyme") or row.get("ms_search_enzyme"))
        quant_type = _clean_str(row.get("Quantification type") or row.get("quantification_type"))

        # Hard rule: Enzyme=None is ONLY a search-engine parameter, never proof of native extraction!
        ms_search_enzyme = search_enzyme_raw or "unknown"
        post_extraction_digestion = "unknown"
        specimen_compartment = "unknown"
        molecular_capture_enrichment = "unknown"
        verified_endpoint_class = "unverified"
        ex_vivo_proteolysis_uncertainty = "unresolved"
        primary_citation = ""

        # 1. Check hard primary-source reviewed overrides first (by project_id or accession)
        hard = reviewed.get(project_id) or accession_to_reviewed.get(original_accession)
        if hard is not None:
            disposition = hard["disposition"]
            disposition_reason = hard["disposition_reason"]
            specimen_compartment = hard["specimen_compartment"]
            molecular_capture_enrichment = hard["molecular_capture_enrichment"]
            post_extraction_digestion = hard["post_extraction_digestion"]
            verified_endpoint_class = hard["verified_endpoint_class"]
            primary_citation = hard.get("primary_url") or hard.get("primary_pmid", "")
        elif any(tok in category_raw.casefold() for tok in _MHC_CATEGORY_TOKENS):
            disposition = "excluded"
            disposition_reason = "mhc_immunopeptidomics_category_not_free_peptide"
            molecular_capture_enrichment = "mhc_immunoaffinity_reported"
            verified_endpoint_class = "mhc_bound_eluted_ligand"
        elif "neuropeptidomics" in category_raw.casefold():
            disposition = "excluded"
            disposition_reason = "neuropeptidomics_not_apc"
            verified_endpoint_class = "neuropeptidomics"
        elif (
            "extracellular" in category_raw.casefold()
            and "intracellular" not in category_raw.casefold()
        ):
            disposition = "excluded"
            disposition_reason = "extracellular_not_intracellular_apc"
            specimen_compartment = "extracellular"
            verified_endpoint_class = "extracellular_peptidomics"
        elif project_id in verified_overrides or sample_id in verified_overrides:
            ov = verified_overrides.get(sample_id) or verified_overrides[project_id]
            specimen_compartment = _clean_str(ov.get("specimen_compartment")) or "unknown"
            molecular_capture_enrichment = (
                _clean_str(ov.get("molecular_capture_enrichment")) or "unknown"
            )
            post_extraction_digestion = _clean_str(ov.get("post_extraction_digestion")) or "unknown"
            verified_endpoint_class = _clean_str(ov.get("verified_endpoint_class")) or "unverified"
            primary_citation = _clean_str(ov.get("primary_citation"))
            is_apc = bool(ov.get("is_non_infected_human_or_mouse_apc", False))
            is_native_free = (
                molecular_capture_enrichment == "none_native_extraction"
                and post_extraction_digestion == "none"
                and verified_endpoint_class == "native_free_intracellular_peptide"
                and is_apc
            )
            if is_native_free and primary_citation:
                disposition = "eligible"
                disposition_reason = "primary_protocol_verified_native_apc_free_peptide"
                ex_vivo_proteolysis_uncertainty = (
                    _clean_str(ov.get("ex_vivo_proteolysis_uncertainty")) or "present_uncalibrated"
                )
            else:
                disposition = "excluded"
                disposition_reason = (
                    _clean_str(ov.get("exclusion_reason"))
                    or "primary_protocol_failed_native_apc_criteria"
                )
        else:
            disposition = "deferred"
            disposition_reason = "primary_protocol_unverified"

        reason_counts[disposition_reason] = reason_counts.get(disposition_reason, 0) + 1
        out_rows.append(
            {
                "source_row_index": idx + 1,
                "project_id": project_id,
                "sample_id": sample_id,
                "original_accession": original_accession,
                "raw_file": raw_file,
                "condition_raw": condition,
                "species_raw": species,
                "tissue_raw": tissue,
                "disease_status_raw": disease,
                "source_category_raw": category_raw,
                "quantification_type_raw": quant_type,
                "specimen_compartment": specimen_compartment,
                "molecular_capture_enrichment": molecular_capture_enrichment,
                "post_extraction_digestion": post_extraction_digestion,
                "ms_search_enzyme": ms_search_enzyme,
                "verified_endpoint_class": verified_endpoint_class,
                "disposition": disposition,
                "disposition_reason": disposition_reason,
                "primary_citation": primary_citation,
                "specimen_independence_status": "unknown",
                "ex_vivo_proteolysis_uncertainty": ex_vivo_proteolysis_uncertainty,
                "establishes_cellular_concentration": False,
            }
        )

    qualified_df = pd.DataFrame(out_rows)
    n_eligible = (
        int((qualified_df["disposition"] == "eligible").sum()) if not qualified_df.empty else 0
    )
    n_deferred = (
        int((qualified_df["disposition"] == "deferred").sum()) if not qualified_df.empty else 0
    )
    n_excluded = (
        int((qualified_df["disposition"] == "excluded").sum()) if not qualified_df.empty else 0
    )

    summary = {
        "tracked_issue": 811,
        "resource_name": contract["resource_name"],
        "n_rows": len(qualified_df),
        "n_projects": int(qualified_df["project_id"].nunique()) if not qualified_df.empty else 0,
        "n_unique_raw_files": (
            int(qualified_df.loc[qualified_df["raw_file"] != "", "raw_file"].nunique())
            if not qualified_df.empty
            else 0
        ),
        "n_eligible_rows": n_eligible,
        "n_deferred_rows": n_deferred,
        "n_excluded_rows": n_excluded,
        "n_by_disposition_reason": dict(sorted(reason_counts.items())),
        "establishes_specimen_independence": False,
        "establishes_cellular_concentration": False,
    }
    return qualified_df, summary


def guard_peptidomicsdb_reanalysis_rows(
    peptides_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate PeptidomicsDB PEAKS reanalysis peptide rows and preserve provenance (#811).

    Requires explicit PEAKS identification method (``DB Search``, ``DeepNovo``,
    ``Homolog``, or ``Sequence Variants``), separates missing from zero quantities,
    preserves all origin protein alternatives without collapsing to one gene, and
    explicitly flags ``cross_project_intensity_comparable=False``.
    """
    if not isinstance(peptides_df, pd.DataFrame):
        raise TypeError("peptides_df must be a pandas DataFrame")

    required = ("project_id", "sample_id", "peptide", "identification_method", "peaks_version")
    missing = [col for col in required if col not in peptides_df.columns]
    if missing:
        raise ValueError(f"PeptidomicsDB reanalysis table missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}

    for idx, row in peptides_df.iterrows():
        method = _clean_str(row["identification_method"])
        if method not in VALID_PEAKS_METHODS:
            raise ValueError(
                f"Unsupported PEAKS identification_method {method!r} at row {idx + 1}; "
                f"expected one of {sorted(VALID_PEAKS_METHODS)}"
            )
        peaks_version = _clean_str(row["peaks_version"])
        if not peaks_version:
            raise ValueError(f"Missing peaks_version at row {idx + 1}")

        qty_raw = _clean_str(row.get("quantity_raw") or row.get("area") or row.get("intensity"))
        qty_val = _parse_float(qty_raw)
        if qty_val is None:
            qty_status = "missing"
        elif qty_val == 0.0:
            qty_status = "observed_zero"
        elif qty_val > 0.0:
            qty_status = "observed_positive"
        else:
            raise ValueError(f"Negative PeptidomicsDB quantity at row {idx + 1}: {qty_val}")

        method_counts[method] = method_counts.get(method, 0) + 1
        status_counts[qty_status] = status_counts.get(qty_status, 0) + 1
        rows.append(
            {
                "project_id": _clean_str(row["project_id"]).upper(),
                "sample_id": _clean_str(row["sample_id"]).upper(),
                "peptide": _clean_str(row["peptide"]).upper(),
                "modifications_raw": _clean_str(row.get("modifications_raw")),
                "origin_alternatives_raw": _clean_str(row.get("origin_alternatives_raw")),
                "identification_lineage": "peptidomicsdb_reanalysis",
                "peaks_version": peaks_version,
                "identification_method": method,
                "confidence_raw": _clean_str(row.get("confidence_raw")),
                "quantity_raw": qty_raw,
                "quantity_value": qty_val,
                "quantity_value_status": qty_status,
                "quantity_unit": _clean_str(row.get("quantity_unit")) or "uncalibrated_peaks_area",
                "cross_project_intensity_comparable": False,
            }
        )

    out_df = pd.DataFrame(rows)
    n_projects = int(out_df["project_id"].nunique()) if not out_df.empty else 0
    summary = {
        "tracked_issue": 811,
        "n_rows": len(out_df),
        "n_projects": n_projects,
        "n_by_identification_method": dict(sorted(method_counts.items())),
        "n_by_quantity_value_status": dict(sorted(status_counts.items())),
        "cross_project_intensity_comparable": False,
        "multi_project_comparison_warning": n_projects > 1,
    }
    return out_df, summary
