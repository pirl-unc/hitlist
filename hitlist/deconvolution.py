"""License-free allele pool audit, narrowing, and mono-allelic motif deconvolution (#37).

Provides three composable, source-preserving tools for ``pmid_class_pool`` and
multi-allelic candidate sets without ever overwriting native ``mhc_restriction``,
``sample_mhc``, or ``mhc_allele_set`` columns:

1. :func:`audit_pmid_class_pools`: Quantifies pool sizes and per-PMID burden.
2. :func:`narrow_candidate_pools`: Computes minimal sufficient candidate pools
   within a PMID when corroborated by sample-level genotypes (Option B).
3. :func:`build_monoallelic_motif_models` and :func:`deconvolve_allele_pools`:
   License-free position-specific log-odds (PSSM/PWM) scoring built directly
   from mono-allelic MS ligands, emitting ``predicted_allele_rank1`` and score
   margins while preserving native pools and abstaining whenever a candidate
   pool has incomplete motif coverage (#574) (Options A & C).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
_TOKEN_SPLIT_RE = re.compile(r"[;,\s]+")
_CORROBORATING_PROVENANCES = frozenset({"exact", "sample_allele_match", "peptide_attribution"})


def parse_allele_pool(raw_pool: object) -> tuple[str, ...]:
    """Split a semicolon/comma/whitespace-delimited allele pool into sorted unique tokens."""
    if raw_pool is None:
        return ()
    if isinstance(raw_pool, (list, tuple, set, frozenset)):
        tokens = [str(v).strip() for v in raw_pool if str(v).strip()]
    else:
        try:
            if pd.isna(raw_pool):
                return ()
        except (TypeError, ValueError):
            pass
        text = str(raw_pool).strip()
        if not text:
            return ()
        tokens = [tok.strip() for tok in _TOKEN_SPLIT_RE.split(text) if tok.strip()]
    return tuple(sorted(set(tokens)))


def audit_pmid_class_pools(
    obs_df: pd.DataFrame,
    *,
    pmid_col: str = "pmid",
    peptide_col: str = "peptide",
    pool_col: str = "mhc_allele_set",
    provenance_col: str = "mhc_allele_provenance",
    mhc_class_col: str = "mhc_class",
) -> dict[str, Any]:
    """Audit ``pmid_class_pool`` observations by PMID and pool cardinality (#37)."""
    if provenance_col not in obs_df.columns:
        raise ValueError(f"obs_df missing required column {provenance_col!r}")
    if peptide_col not in obs_df.columns:
        raise ValueError(f"obs_df missing required column {peptide_col!r}")

    prov = obs_df[provenance_col].astype("string").fillna("")
    pool_df = obs_df.loc[prov.eq("pmid_class_pool")].copy()
    if pool_df.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "n_total_observations": len(obs_df),
            "n_pool_observations": 0,
            "n_pool_peptides": 0,
            "n_pool_pmids": 0,
            "n_singleton_pool_observations": 0,
            "mean_pool_size": 0.0,
            "median_pool_size": 0.0,
            "by_pmid": [],
        }

    if pool_col in pool_df.columns:
        pool_sizes = pool_df[pool_col].map(lambda v: len(parse_allele_pool(v)))
    elif "mhc_allele_set_size" in pool_df.columns:
        pool_sizes = (
            pd.to_numeric(pool_df["mhc_allele_set_size"], errors="coerce").fillna(0).astype(int)
        )
    else:
        pool_sizes = pd.Series(0, index=pool_df.index, dtype=int)
    pool_df["_pool_size"] = pool_sizes

    by_pmid: list[dict[str, Any]] = []
    if pmid_col in pool_df.columns:
        for pmid_val, grp in pool_df.groupby(pmid_col, sort=False):
            classes = (
                sorted({str(v) for v in grp[mhc_class_col].dropna() if str(v)})
                if mhc_class_col in grp.columns
                else []
            )
            by_pmid.append(
                {
                    "pmid": int(pmid_val) if pd.notna(pmid_val) else None,
                    "n_observations": len(grp),
                    "n_peptides": int(grp[peptide_col].nunique()),
                    "n_singleton_observations": int((grp["_pool_size"] == 1).sum()),
                    "mean_pool_size": float(grp["_pool_size"].mean()),
                    "max_pool_size": int(grp["_pool_size"].max()),
                    "mhc_classes": classes,
                }
            )
        by_pmid.sort(key=lambda row: (-row["n_observations"], str(row["pmid"])))

    return {
        "schema_version": SCHEMA_VERSION,
        "n_total_observations": len(obs_df),
        "n_pool_observations": len(pool_df),
        "n_pool_peptides": int(pool_df[peptide_col].nunique()),
        "n_pool_pmids": len(by_pmid),
        "n_singleton_pool_observations": int((pool_df["_pool_size"] == 1).sum()),
        "mean_pool_size": float(pool_df["_pool_size"].mean()),
        "median_pool_size": float(pool_df["_pool_size"].median()),
        "by_pmid": by_pmid,
    }


def narrow_candidate_pools(
    obs_df: pd.DataFrame,
    *,
    pmid_col: str = "pmid",
    peptide_col: str = "peptide",
    pool_col: str = "mhc_allele_set",
    provenance_col: str = "mhc_allele_provenance",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute minimal sufficient candidate pools per observation without mutating native columns (#37).

    Within each ``(pmid, peptide)`` group:
    - Non-pool observations keep their existing allele set.
    - ``pmid_class_pool`` observations with a single allele in ``pool_col`` are
      marked ``singleton_pool``.
    - ``pmid_class_pool`` observations whose ``(pmid, peptide)`` also appears
      with corroborated sample-level/exact genotypes in the same PMID narrow
      their candidate pool to the non-empty intersection of the original pool
      and the corroborated genotype sets (Option B).
    """
    required = {pmid_col, peptide_col, pool_col, provenance_col}
    missing = required - set(obs_df.columns)
    if missing:
        raise ValueError(f"obs_df missing required columns: {sorted(missing)}")

    result = obs_df.copy()
    if result.empty:
        result["narrowed_allele_pool"] = pd.Series(dtype="string")
        result["n_narrowed_alleles"] = pd.Series(dtype="int64")
        result["pool_narrowing_basis"] = pd.Series(dtype="string")
        return result, {
            "schema_version": SCHEMA_VERSION,
            "n_rows": 0,
            "n_pool_rows": 0,
            "n_narrowed_rows": 0,
            "n_singleton_pool_rows": 0,
            "counts_by_basis": {},
        }

    # Build corroborated minimal intersection per (pmid, peptide) from exact /
    # sample_allele_match / peptide_attribution rows within the same study.
    corroborated: dict[tuple[Any, str], set[str]] = {}
    prov_vals = result[provenance_col].astype("string").fillna("").tolist()
    pmid_vals = result[pmid_col].tolist()
    pep_vals = result[peptide_col].astype("string").fillna("").tolist()
    parsed_pools = [parse_allele_pool(v) for v in result[pool_col].tolist()]

    for pmid_v, pep_v, prov_v, alleles in zip(pmid_vals, pep_vals, prov_vals, parsed_pools):
        if prov_v in _CORROBORATING_PROVENANCES and alleles and pep_v:
            key = (pmid_v, pep_v)
            allele_set = set(alleles)
            if key not in corroborated:
                corroborated[key] = allele_set
            else:
                inter = corroborated[key] & allele_set
                if inter:
                    corroborated[key] = inter

    narrowed_col: list[str] = []
    narrowed_sizes: list[int] = []
    basis_col: list[str] = []
    n_pool_rows = 0
    n_narrowed_rows = 0
    n_singleton_rows = 0

    for pmid_v, pep_v, prov_v, alleles in zip(pmid_vals, pep_vals, prov_vals, parsed_pools):
        if prov_v != "pmid_class_pool":
            narrowed_col.append(";".join(alleles))
            narrowed_sizes.append(len(alleles))
            basis_col.append("not_class_pool")
            continue

        n_pool_rows += 1
        if not alleles:
            narrowed_col.append("")
            narrowed_sizes.append(0)
            basis_col.append("empty_pool")
            continue
        if len(alleles) == 1:
            narrowed_col.append(alleles[0])
            narrowed_sizes.append(1)
            basis_col.append("singleton_pool")
            n_singleton_rows += 1
            continue

        corr = corroborated.get((pmid_v, pep_v))
        if corr:
            inter = sorted(set(alleles) & corr)
            if inter and len(inter) < len(alleles):
                narrowed_col.append(";".join(inter))
                narrowed_sizes.append(len(inter))
                basis_col.append("within_pmid_sample_intersection")
                n_narrowed_rows += 1
                continue

        narrowed_col.append(";".join(alleles))
        narrowed_sizes.append(len(alleles))
        basis_col.append("unchanged_pool")

    result["narrowed_allele_pool"] = pd.Series(narrowed_col, index=result.index, dtype="string")
    result["n_narrowed_alleles"] = pd.Series(narrowed_sizes, index=result.index, dtype="int64")
    result["pool_narrowing_basis"] = pd.Series(basis_col, index=result.index, dtype="string")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": len(result),
        "n_pool_rows": n_pool_rows,
        "n_narrowed_rows": n_narrowed_rows,
        "n_singleton_pool_rows": n_singleton_rows,
        "counts_by_basis": dict(Counter(basis_col)),
    }
    return result, summary


def build_monoallelic_motif_models(
    monoallelic_df: pd.DataFrame,
    *,
    peptide_col: str = "peptide",
    allele_col: str = "mhc_restriction",
    lengths: tuple[int, ...] = (8, 9, 10, 11),
    min_peptides: int = 20,
    pseudocount: float = 0.5,
) -> dict[str, Any]:
    """Build license-free log-odds PSSM/PWM motif models from mono-allelic MS ligands (#37).

    Parameters
    ----------
    monoallelic_df
        DataFrame of single-allele peptides (e.g. mono-allelic 721.221/C1R MS
        ligands). Duplicate ``(allele, peptide)`` pairs are collapsed once.
    lengths
        Peptide lengths to model.
    min_peptides
        Minimum unique canonical-AA peptides required per ``(allele, length)``.
    pseudocount
        Dirichlet pseudocount added to each of the 20 canonical amino acids.
    """
    if peptide_col not in monoallelic_df.columns or allele_col not in monoallelic_df.columns:
        raise ValueError(f"monoallelic_df must contain {peptide_col!r} and {allele_col!r}")
    if min_peptides < 1:
        raise ValueError("min_peptides must be >= 1")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be > 0")

    allowed_lengths = set(lengths)
    valid_aa = set(AMINO_ACIDS)
    dedup = (
        monoallelic_df[[allele_col, peptide_col]]
        .dropna()
        .assign(
            _allele=lambda d: d[allele_col].astype("string").str.strip(),
            _pep=lambda d: d[peptide_col].astype("string").str.strip().str.upper(),
        )
    )
    dedup = dedup.loc[
        dedup["_allele"].ne("")
        & dedup["_pep"].str.len().isin(allowed_lengths)
        & dedup["_pep"].map(lambda s: set(s) <= valid_aa)
    ][["_allele", "_pep"]].drop_duplicates()

    # Empirical background across all valid training peptides (smoothed with uniform floor).
    bg_counts = np.full(len(AMINO_ACIDS), 1.0, dtype=np.float64)
    for pep in dedup["_pep"].tolist():
        for ch in pep:
            bg_counts[_AA_INDEX[ch]] += 1.0
    bg_freqs = bg_counts / bg_counts.sum()

    models: dict[tuple[str, int], dict[str, Any]] = {}
    for (allele, pep_len), grp in dedup.groupby(
        [dedup["_allele"], dedup["_pep"].str.len()], sort=True
    ):
        peps = grp["_pep"].tolist()
        if len(peps) < min_peptides:
            continue
        length = int(pep_len)
        counts = np.full((length, len(AMINO_ACIDS)), float(pseudocount), dtype=np.float64)
        for pep in peps:
            for pos, ch in enumerate(pep):
                counts[pos, _AA_INDEX[ch]] += 1.0
        probs = counts / counts.sum(axis=1, keepdims=True)
        log_odds = np.log2(probs / bg_freqs[np.newaxis, :])
        models[(str(allele), length)] = {
            "allele": str(allele),
            "length": length,
            "n_peptides": len(peps),
            "log_odds": log_odds,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "amino_acids": AMINO_ACIDS,
        "lengths": sorted(allowed_lengths),
        "min_peptides": int(min_peptides),
        "pseudocount": float(pseudocount),
        "background_frequencies": bg_freqs.tolist(),
        "n_models": len(models),
        "n_alleles": len({allele for allele, _ in models}),
        "models": models,
    }


def _score_peptide_with_matrix(peptide: str, log_odds: np.ndarray) -> float:
    total = 0.0
    for pos, ch in enumerate(peptide):
        idx = _AA_INDEX.get(ch)
        if idx is None:
            return float("nan")
        total += float(log_odds[pos, idx])
    return total


def deconvolve_allele_pools(
    pool_df: pd.DataFrame,
    motif_models: dict[str, Any],
    *,
    pool_col: str = "mhc_allele_set",
    peptide_col: str = "peptide",
    min_log_odds_margin: float = 1.5,
    min_rank1_score: float = 0.0,
    max_pool_size: int = 24,
    require_complete_pool: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deconvolve multi-allelic pools using internal PWM motifs while preserving native pools (#37).

    Always preserves native ``pool_col`` / ``mhc_restriction`` / ``sample_mhc``
    untouched and writes computed predictions into explicit ``predicted_allele_*``
    and ``motif_score_*`` columns (Option C).

    When ``require_complete_pool=True`` (default, enforcing #574), abstains with
    ``deconvolution_status="incomplete_motif_coverage"`` whenever ANY candidate
    in a row's pool lacks a motif model for that peptide length.
    """
    if pool_col not in pool_df.columns or peptide_col not in pool_df.columns:
        raise ValueError(f"pool_df must contain {pool_col!r} and {peptide_col!r}")
    models: dict[tuple[str, int], dict[str, Any]] = motif_models.get("models", {})
    allowed_lengths = set(motif_models.get("lengths", ()))

    result = pool_df.copy()
    if result.empty:
        for col, dtype in (
            ("predicted_allele_rank1", "string"),
            ("predicted_allele_rank2", "string"),
            ("motif_score_rank1", "float64"),
            ("motif_score_rank2", "float64"),
            ("motif_score_margin", "float64"),
            ("n_alleles_scored", "int64"),
            ("deconvolution_status", "string"),
        ):
            result[col] = pd.Series(dtype=dtype)
        return result, {
            "schema_version": SCHEMA_VERSION,
            "n_rows": 0,
            "n_assigned_rows": 0,
            "n_abstained_rows": 0,
            "counts_by_status": {},
        }

    valid_aa = set(AMINO_ACIDS)
    rank1_alleles: list[str] = []
    rank2_alleles: list[str] = []
    rank1_scores: list[float] = []
    rank2_scores: list[float] = []
    margins: list[float] = []
    n_scored_list: list[int] = []
    statuses: list[str] = []

    peps = result[peptide_col].astype("string").fillna("").str.strip().str.upper().tolist()
    pools = [parse_allele_pool(v) for v in result[pool_col].tolist()]

    # Cache (peptide, pool) scoring for repeated contexts across rows.
    cache: dict[tuple[str, tuple[str, ...]], tuple[str, str, float, float, float, int, str]] = {}

    for pep, alleles in zip(peps, pools):
        key = (pep, alleles)
        cached = cache.get(key)
        if cached is None:
            if not alleles:
                cached = ("", "", math.nan, math.nan, math.nan, 0, "empty_pool")
            elif len(alleles) == 1:
                allele0 = alleles[0]
                model = models.get((allele0, len(pep)))
                s0 = (
                    _score_peptide_with_matrix(pep, model["log_odds"])
                    if model is not None and set(pep) <= valid_aa
                    else math.nan
                )
                cached = (
                    allele0,
                    "",
                    s0,
                    math.nan,
                    math.nan,
                    1 if model is not None else 0,
                    "singleton_pool",
                )
            elif len(alleles) > max_pool_size:
                cached = ("", "", math.nan, math.nan, math.nan, 0, "pool_too_large")
            elif len(pep) not in allowed_lengths or not (set(pep) <= valid_aa):
                cached = ("", "", math.nan, math.nan, math.nan, 0, "unsupported_length")
            else:
                available = [a for a in alleles if (a, len(pep)) in models]
                if require_complete_pool and len(available) != len(alleles):
                    cached = (
                        "",
                        "",
                        math.nan,
                        math.nan,
                        math.nan,
                        len(available),
                        "incomplete_motif_coverage",
                    )
                elif not available:
                    cached = ("", "", math.nan, math.nan, math.nan, 0, "incomplete_motif_coverage")
                else:
                    scored = [
                        (_score_peptide_with_matrix(pep, models[(a, len(pep))]["log_odds"]), a)
                        for a in available
                    ]
                    scored.sort(key=lambda pair: (-pair[0], pair[1]))
                    s1, a1 = scored[0]
                    if len(scored) >= 2:
                        s2, a2 = scored[1]
                        margin = s1 - s2
                    else:
                        s2, a2 = math.nan, ""
                        margin = math.inf
                    if s1 < min_rank1_score:
                        cached = ("", a2, s1, s2, margin, len(available), "below_min_score")
                    elif len(scored) >= 2 and margin < min_log_odds_margin:
                        cached = ("", a2, s1, s2, margin, len(available), "ambiguous_margin")
                    else:
                        cached = (a1, a2, s1, s2, margin, len(available), "assigned")
            cache[key] = cached

        a1, a2, s1, s2, margin, n_sc, st = cached
        rank1_alleles.append(a1)
        rank2_alleles.append(a2)
        rank1_scores.append(s1)
        rank2_scores.append(s2)
        margins.append(margin)
        n_scored_list.append(n_sc)
        statuses.append(st)

    result["predicted_allele_rank1"] = pd.Series(rank1_alleles, index=result.index, dtype="string")
    result["predicted_allele_rank2"] = pd.Series(rank2_alleles, index=result.index, dtype="string")
    result["motif_score_rank1"] = pd.Series(rank1_scores, index=result.index, dtype="float64")
    result["motif_score_rank2"] = pd.Series(rank2_scores, index=result.index, dtype="float64")
    result["motif_score_margin"] = pd.Series(margins, index=result.index, dtype="float64")
    result["n_alleles_scored"] = pd.Series(n_scored_list, index=result.index, dtype="int64")
    result["deconvolution_status"] = pd.Series(statuses, index=result.index, dtype="string")

    counts = dict(Counter(statuses))
    n_assigned = counts.get("assigned", 0) + counts.get("singleton_pool", 0)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": len(result),
        "n_assigned_rows": n_assigned,
        "n_abstained_rows": len(result) - n_assigned,
        "counts_by_status": counts,
    }
    return result, summary
