"""HAP1 MHC selection-covariate provenance contract and non-filtering join (#726).

Keeps cell-line genotype (HLA-A*02:01, HLA-B*40:01, HLA-C*03:04), predicted
allele-selection covariates (e.g. MixMHCpred 2.2) and experimentally measured
MHC restriction strictly separate, records unrecovered original Shapiro 2025
MixMHCpred outputs explicitly, and verifies non-filtering 1:1 joins onto
quantitative HAP1 response tables without imputing scores or altering splits.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

CONTRACT_PATH = (
    Path(__file__).parent
    / "data"
    / "evidence_revisions"
    / "hap1_selection_covariates_contract.json"
)
_VALID_DIRECTIONS = frozenset({"lower_is_better", "higher_is_better"})


def hap1_selection_covariates_contract() -> dict[str, Any]:
    """Return a validated deep copy of the HAP1 selection-covariate contract (#726)."""
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported HAP1 covariate contract schema: {data.get('schema_version')}"
        )
    return copy.deepcopy(data)


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def validate_hap1_covariate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate provenance metadata for recovered or newly computed HAP1 covariates."""
    contract = hap1_selection_covariates_contract()
    missing = [
        field
        for field in contract["required_derivation_metadata_fields"]
        if field not in metadata or _clean_str(metadata[field]) == ""
    ]
    if missing:
        raise ValueError(f"HAP1 selection covariate metadata missing required fields: {missing}")

    origin = _clean_str(metadata["derivation_origin"])
    if origin not in contract["allowed_derivation_origins"]:
        raise ValueError(
            f"Unsupported derivation_origin {origin!r}; expected one of "
            f"{contract['allowed_derivation_origins']}"
        )

    direction = _clean_str(metadata["score_direction"])
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Unsupported score_direction {direction!r}; expected one of {sorted(_VALID_DIRECTIONS)}"
        )

    if origin == "newly_computed_pinned_derivation" and metadata.get(
        "attributed_to_original_paper", False
    ):
        raise ValueError(
            "A newly computed derivation must never be retroactively attributed to the original paper"
        )
    return copy.deepcopy(metadata)


def attach_hap1_selection_covariates(
    response_df: pd.DataFrame,
    covariate_df: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    peptide_column: str = "peptide_sequence",
    candidate_alleles: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform a non-filtering 1:1 join of multi-allele selection covariates onto HAP1 rows.

    Parameters
    ----------
    response_df
        Quantitative HAP1 response table. Every row, split assignment and alternate
        parent origin is preserved untouched (zero binding-based filtering).
    covariate_df
        Long-format per-peptide, per-allele predictor output with columns
        ``[peptide_column, 'allele', 'score']`` (optional ``'rank'``, ``'status'``,
        ``'source_row_hash'``). Must retain scores for ALL candidate alleles
        including non-best alleles.
    metadata
        Provenance dict validated by :func:`validate_hap1_covariate_metadata`.
    peptide_column
        Sequence/molecular-form join key present in both DataFrames.
    candidate_alleles
        Candidate HAP1 MHC genotype alleles (defaults to contract's 3 HAP1 alleles).
    """
    if not isinstance(response_df, pd.DataFrame) or not isinstance(covariate_df, pd.DataFrame):
        raise TypeError("response_df and covariate_df must be pandas DataFrames")

    contract = hap1_selection_covariates_contract()
    validated_meta = validate_hap1_covariate_metadata(metadata)
    alleles = tuple(candidate_alleles or contract["hap1_genotype"])
    if len(alleles) == 0:
        raise ValueError("candidate_alleles must be nonempty")
    if peptide_column not in response_df.columns:
        raise ValueError(f"response_df missing join column {peptide_column!r}")
    for req in (peptide_column, "allele", "score"):
        if req not in covariate_df.columns:
            raise ValueError(f"covariate_df missing required column {req!r}")
    if "measured_restriction" in covariate_df.columns:
        raise ValueError(
            "covariate_df must not carry measured_restriction; keep predictions distinct from measured restriction"
        )

    direction = validated_meta["score_direction"]
    higher_is_better = direction == "higher_is_better"

    # Validate uniqueness per (peptide, allele) and build multi-allele lookup
    per_peptide: dict[str, dict[str, dict[str, Any]]] = {}
    for idx, row in covariate_df.iterrows():
        pep = _clean_str(row[peptide_column]).upper()
        allele = _clean_str(row["allele"])
        if not pep or allele not in alleles:
            continue
        bucket = per_peptide.setdefault(pep, {})
        if allele in bucket:
            raise ValueError(f"Duplicate covariate row for ({pep!r}, {allele!r}) at index {idx}")
        bucket[allele] = {
            "score": _finite_float(row["score"]),
            "rank": _finite_float(row["rank"]) if "rank" in covariate_df.columns else None,
            "status": _clean_str(row.get("status")) or "scored",
            "source_row_hash": _clean_str(row.get("source_row_hash")),
        }

    # Verify that any scored peptide provides ALL candidate alleles (including non-best)
    for pep, allele_map in per_peptide.items():
        missing_alleles = [a for a in alleles if a not in allele_map]
        if missing_alleles:
            raise ValueError(
                f"Covariate table omits non-best candidate alleles {missing_alleles} for {pep!r}; "
                "all candidate alleles must be retained"
            )

    out = response_df.copy()
    genotype_str = ";".join(alleles)
    all_scores_json: list[str] = []
    best_alleles: list[str] = []
    best_scores: list[float | None] = []
    tie_alleles_list: list[str] = []
    is_tie_list: list[bool] = []
    cov_statuses: list[str] = []

    for raw_pep in out[peptide_column]:
        pep = _clean_str(raw_pep).upper()
        allele_map = per_peptide.get(pep)
        if not allele_map:
            all_scores_json.append("{}")
            best_alleles.append("")
            best_scores.append(None)
            tie_alleles_list.append("")
            is_tie_list.append(False)
            cov_statuses.append("missing_or_unsupported")
            continue

        scored_items = [
            (a, rec["score"]) for a, rec in allele_map.items() if rec["score"] is not None
        ]
        all_scores_json.append(
            json.dumps(
                {a: allele_map[a] for a in alleles},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if not scored_items:
            best_alleles.append("")
            best_scores.append(None)
            tie_alleles_list.append("")
            is_tie_list.append(False)
            cov_statuses.append("missing_or_unsupported")
            continue

        best_val = (
            max(v for _, v in scored_items) if higher_is_better else min(v for _, v in scored_items)
        )
        winners = sorted(a for a, v in scored_items if v == best_val)
        is_tie = len(winners) > 1
        best_alleles.append(winners[0] if not is_tie else "")
        best_scores.append(best_val)
        tie_alleles_list.append(";".join(winners) if is_tie else "")
        is_tie_list.append(is_tie)
        cov_statuses.append("scored_all_candidate_alleles")

    out["selection_covariate_assertion_type"] = "inferred_best_allele_prediction"
    out["selection_covariate_genotype"] = genotype_str
    out["selection_covariate_derivation_origin"] = validated_meta["derivation_origin"]
    out["selection_covariate_predictor"] = (
        f"{validated_meta['predictor_name']} {validated_meta['predictor_version']}"
    )
    out["selection_covariate_score_type"] = validated_meta["score_type"]
    out["selection_covariate_score_direction"] = direction
    out["selection_covariate_status"] = cov_statuses
    out["selection_covariate_all_alleles_json"] = all_scores_json
    out["inferred_best_allele"] = best_alleles
    out["inferred_best_score"] = best_scores
    out["inferred_best_allele_tie_set"] = tie_alleles_list
    out["inferred_best_allele_is_tie"] = is_tie_list

    n_covered = sum(1 for s in cov_statuses if s == "scored_all_candidate_alleles")
    n_missing = len(cov_statuses) - n_covered
    n_tied = sum(1 for t in is_tie_list if t)

    audit = {
        "tracked_issue": 726,
        "derivation_origin": validated_meta["derivation_origin"],
        "original_paper_outputs_status": contract["original_paper_outputs_status"],
        "candidate_alleles": list(alleles),
        "n_response_rows": len(response_df),
        "n_joined_rows": len(out),
        "n_covered_rows": n_covered,
        "n_missing_or_unsupported_rows": n_missing,
        "n_tied_rows": n_tied,
        "preserves_all_response_rows": len(out) == len(response_df),
        "applies_binding_eligibility_filter": False,
        "imputes_missing_scores": False,
    }
    return out, audit
