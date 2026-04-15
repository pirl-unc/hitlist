# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Retro-predictor: reassign class-only peptides to their best-scoring
HLA allele using MHCflurry (primary) or NetMHCpan (optional binary).

Class-only peptides are those annotated in IEDB with
``mhc_restriction == "HLA class I"`` (or "HLA class II") — the paper
authors knew the class but did not commit to a specific allele.  For
peptides from multi-allelic samples, we can often recover the likely
allele by running a binding predictor against the sample's curated
HLA genotype.

The TLAKFSPYL example from our audit: IEDB class-only, sample contains
A*02:01 and A*24:02 (among others).  MHCflurry gives A*02:01 a rank
of 0.03 (strong binder) vs A*24:02 at 2.65 — the peptide is almost
certainly an A*02:01 ligand.

Usage::

    hitlist reassign-alleles -o reassigned.csv
    hitlist reassign-alleles --method netmhcpan --mhc-class I
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def _class_i_alleles(mhc_field: str | None) -> list[str]:
    """Extract class I alleles from a sample's mhc string."""
    if not isinstance(mhc_field, str) or not mhc_field.strip():
        return []
    if mhc_field.startswith("HLA class") or mhc_field == "unknown":
        return []
    return [
        a for a in mhc_field.split() if a.startswith(("HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-G"))
    ]


def _class_ii_alleles(mhc_field: str | None) -> list[str]:
    if not isinstance(mhc_field, str) or not mhc_field.strip():
        return []
    if mhc_field.startswith("HLA class") or mhc_field == "unknown":
        return []
    return [a for a in mhc_field.split() if a.startswith("HLA-D")]


def _predict_mhcflurry(pairs: pd.DataFrame) -> pd.DataFrame:
    """Run MHCflurry on a (peptide, allele) cross-product DataFrame.

    Returns per-row score: affinity_nM, presentation_percentile.  Keeps
    the caller's index.
    """
    try:
        from mhcflurry import Class1PresentationPredictor
    except ImportError as e:
        raise RuntimeError(
            "mhcflurry not installed.  Install with: pip install mhcflurry && "
            "mhcflurry-downloads fetch"
        ) from e

    predictor = Class1PresentationPredictor.load()
    out = predictor.predict(
        peptides=pairs["peptide"].tolist(),
        alleles=[[a] for a in pairs["allele"]],
        verbose=0,
    )
    # MHCflurry returns one row per (peptide, allele) pair in input order.
    out = out.reset_index(drop=True)
    pairs = pairs.reset_index(drop=True).copy()
    pairs["affinity_nM"] = out["affinity"].values
    pairs["presentation_percentile"] = out["presentation_percentile"].values
    return pairs


def _netmhcpan_allele_arg(a: str) -> str:
    """NetMHCpan class-I allele string (strip the asterisk)."""
    return a.replace("*", "")


def _predict_netmhcpan(pairs: pd.DataFrame) -> pd.DataFrame:
    """Run NetMHCpan 4.x on a (peptide, allele) cross-product.

    Requires the ``netMHCpan`` binary on PATH (install separately from
    DTU; licensed for academic use).  Returns the same two score
    columns as MHCflurry so the best-allele logic is uniform.
    """
    # One netMHCpan invocation per allele is simplest; batch peptides.
    rows: list[dict] = []
    for allele, grp in pairs.groupby("allele"):
        peps = grp["peptide"].drop_duplicates().tolist()
        pep_file = Path("/tmp") / f"hitlist_netmhcpan_{hash(allele) & 0xFFFF}.txt"
        pep_file.write_text("\n".join(peps) + "\n")
        result = subprocess.run(
            ["netMHCpan", "-p", str(pep_file), "-a", _netmhcpan_allele_arg(allele), "-BA"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        for line in result.stdout.splitlines():
            if "PEPLIST" not in line:
                continue
            parts = line.split()
            if len(parts) < 16:
                continue
            try:
                rows.append(
                    {
                        "peptide": parts[2],
                        "allele": parts[1],
                        "rank_EL": float(parts[12]),
                        "affinity_nM": float(parts[15]),
                    }
                )
            except (ValueError, IndexError):
                continue
    out = pd.DataFrame(rows)
    # Normalise to the same column names MHCflurry produces.  Use rank_EL
    # as the presentation proxy (lower = better, <= 0.5 = strong, <= 2 = weak).
    out["presentation_percentile"] = out["rank_EL"]
    return out[["peptide", "allele", "affinity_nM", "presentation_percentile"]]


def reassign_class_only_alleles(
    method: str = "mhcflurry",
    mhc_class: str = "I",
    max_alleles_per_sample: int = 30,
) -> pd.DataFrame:
    """Reassign class-only peptides to their best-scoring allele.

    Loads the observations table, restricts to rows where
    ``mhc_restriction`` is class-only ("HLA class I" / "HLA class II")
    and the sample has a curated multi-allelic genotype, runs the
    requested predictor against each sample's alleles, and returns the
    best allele per peptide.

    Parameters
    ----------
    method
        ``"mhcflurry"`` (pip-installable, default) or ``"netmhcpan"``
        (requires DTU binary on PATH).
    mhc_class
        ``"I"`` or ``"II"``.  MHCflurry currently supports class I only;
        class II support via NetMHCpan requires netMHCIIpan (not yet
        wired).
    max_alleles_per_sample
        Skip samples whose genotype has more alleles than this (likely
        a pooled-donor curation artifact; see
        tasks/per_sample_allele_curation_audit.md).  Such samples can
        produce misleading "best allele" calls because the pool does
        not represent any one donor.

    Returns
    -------
    pd.DataFrame with columns:
        peptide, pmid, sample_label, sample_mhc, n_alleles_tested,
        best_allele, best_affinity_nM, best_presentation_percentile,
        is_strong_binder, is_weak_binder.
    """
    if mhc_class != "I":
        raise NotImplementedError("Only class I reassignment is supported in v1.8.0.")

    from .export import generate_observations_table

    df = generate_observations_table(mhc_class=mhc_class)
    # Keep only class-only rows in multi-allelic samples
    class_only_mask = df["mhc_restriction"].fillna("").str.startswith("HLA class")
    multi_mask = df["is_monoallelic"].fillna(False).eq(False)
    target = df[class_only_mask & multi_mask].copy()
    target["_alleles"] = target["sample_mhc"].map(_class_i_alleles)
    target = target[target["_alleles"].map(len).between(1, max_alleles_per_sample)]
    if target.empty:
        return pd.DataFrame(
            columns=[
                "peptide",
                "pmid",
                "sample_label",
                "sample_mhc",
                "n_alleles_tested",
                "best_allele",
                "best_affinity_nM",
                "best_presentation_percentile",
                "is_strong_binder",
                "is_weak_binder",
            ]
        )

    # Filter 8-12mer (class I)
    target = target[target["peptide"].str.len().between(8, 12)]

    # Build (peptide, allele) cross product de-duplicated across samples.
    pairs: list[dict] = []
    for _, row in target.iterrows():
        for a in row["_alleles"]:
            pairs.append({"peptide": row["peptide"], "allele": a})
    pair_df = pd.DataFrame(pairs).drop_duplicates(["peptide", "allele"]).reset_index(drop=True)

    if method == "mhcflurry":
        scored = _predict_mhcflurry(pair_df)
    elif method == "netmhcpan":
        scored = _predict_netmhcpan(pair_df)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Best allele per peptide (lowest presentation_percentile)
    best = scored.sort_values(["peptide", "presentation_percentile"]).drop_duplicates(
        "peptide", keep="first"
    )
    best = best.rename(
        columns={
            "allele": "best_allele",
            "affinity_nM": "best_affinity_nM",
            "presentation_percentile": "best_presentation_percentile",
        }
    )

    # Thresholds are the community conventions MHCflurry and NetMHCpan
    # both use: strong binder = rank/percentile <= 0.5, weak <= 2.0.
    best["is_strong_binder"] = best["best_presentation_percentile"] <= 0.5
    best["is_weak_binder"] = best["best_presentation_percentile"] <= 2.0

    # Attach sample context from the first occurrence of each peptide
    # in the filtered target frame.
    sample_ctx = target.drop_duplicates("peptide")[
        ["peptide", "pmid", "sample_label", "sample_mhc", "_alleles"]
    ]
    result = best.merge(sample_ctx, on="peptide", how="left")
    result["n_alleles_tested"] = result["_alleles"].map(len)
    result = result.drop(columns=["_alleles"])

    return result[
        [
            "peptide",
            "pmid",
            "sample_label",
            "sample_mhc",
            "n_alleles_tested",
            "best_allele",
            "best_affinity_nM",
            "best_presentation_percentile",
            "is_strong_binder",
            "is_weak_binder",
        ]
    ]
