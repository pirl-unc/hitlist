# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-sample and per-donor peptidome context from IEDB/CEDAR data.

The full peptidome context for each sample is critical for reasoning
about whether a peptide's presence is biologically meaningful:

- 1 CTA out of 762 peptides = 0.13% → stochastic noise
- 5 CTAs out of 100 peptides = 5% → possible occult tumor

This module provides functions to compute per-sample statistics from
a FULL (unfiltered) scan of IEDB/CEDAR data, and then overlay target
peptide sets to compute context fractions.

Typical usage::

    from hitlist.scanner import scan
    from hitlist.samples import sample_peptidomes, overlay_targets

    # First: full scan (no peptide filter) for a study
    full = scan(peptides=None, iedb_path="mhc_ligand_full.csv", mhc_class="I")

    # Per-sample peptidome stats
    samples = sample_peptidomes(full)

    # Overlay CTA peptides to get context
    context = overlay_targets(full, target_peptides=my_cta_peptides, label="cta")
"""

from __future__ import annotations

import pandas as pd


def sample_peptidomes(
    full_df: pd.DataFrame,
    sample_key: str | list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-sample peptidome statistics from a full scan.

    Parameters
    ----------
    full_df
        DataFrame from :func:`hitlist.scanner.scan` with ``peptides=None``
        (full, unfiltered scan).
    sample_key
        Column(s) to group by for defining a "sample". If None, uses
        ``["pmid", "antigen_processing_comments"]`` which gives per-donor
        per-tissue granularity when Antigen Processing Comments contains
        sample IDs (e.g. "buffy coat 25").

    Returns
    -------
    pd.DataFrame
        One row per sample with: ``total_peptides``, ``unique_peptides``,
        ``unique_alleles``, ``source_tissue``, ``disease``, ``pmid``,
        plus classification flag summaries.
    """
    if full_df.empty:
        return pd.DataFrame()

    if sample_key is None:
        sample_key = ["pmid", "antigen_processing_comments"]
    if isinstance(sample_key, str):
        sample_key = [sample_key]

    # Filter to rows that have usable sample keys
    df = full_df.copy()
    for col in sample_key:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    def _join_unique(s):
        return ";".join(sorted({str(v) for v in s if v and str(v) != "nan"}))

    def _count_unique(s):
        return len({str(v) for v in s if v and str(v) != "nan"})

    agg: dict = {
        "total_peptides": ("peptide", "size"),
        "unique_peptides": ("peptide", _count_unique),
        "unique_alleles": ("mhc_restriction", _count_unique),
        "alleles": ("mhc_restriction", _join_unique),
    }
    if "source_tissue" in df.columns:
        agg["tissues"] = ("source_tissue", _join_unique)
    if "disease" in df.columns:
        agg["diseases"] = ("disease", _join_unique)
    if "cell_name" in df.columns:
        agg["cell_names"] = ("cell_name", _join_unique)
    if "host_age" in df.columns:
        agg["host_ages"] = ("host_age", _join_unique)

    # Source classification summaries
    for flag in [
        "src_cancer",
        "src_adjacent_to_tumor",
        "src_healthy_tissue",
        "src_healthy_thymus",
        "src_healthy_reproductive",
    ]:
        if flag in df.columns:
            agg[flag] = (flag, "any")

    result = df.groupby(sample_key, as_index=False).agg(**agg)
    return result.sort_values("total_peptides", ascending=False).reset_index(drop=True)


def overlay_targets(
    full_df: pd.DataFrame,
    target_peptides: set[str],
    label: str = "target",
    sample_key: str | list[str] | None = None,
) -> pd.DataFrame:
    """Overlay a target peptide set onto per-sample peptidome context.

    For each sample, computes how many of its peptides are in the target
    set, giving the fraction context needed to interpret significance.

    Parameters
    ----------
    full_df
        Full (unfiltered) scan DataFrame.
    target_peptides
        Set of peptide sequences to count (e.g. CTA peptides).
    label
        Label for the target columns (default ``"target"``).
    sample_key
        Sample grouping columns (see :func:`sample_peptidomes`).

    Returns
    -------
    pd.DataFrame
        Per-sample stats with additional columns:
        ``{label}_peptides`` (count of target peptides in this sample),
        ``{label}_fraction`` (target peptides / total peptides),
        ``{label}_peptide_list`` (semicolon-separated target peptides found).
    """
    if full_df.empty:
        return pd.DataFrame()

    if sample_key is None:
        sample_key = ["pmid", "antigen_processing_comments"]
    if isinstance(sample_key, str):
        sample_key = [sample_key]

    df = full_df.copy()
    for col in sample_key:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["_is_target"] = df["peptide"].isin(target_peptides)

    # Get base sample stats
    samples = sample_peptidomes(full_df, sample_key=sample_key)

    # Compute target overlay per sample
    target_df = df[df["_is_target"]].copy()
    if target_df.empty:
        samples[f"{label}_peptides"] = 0
        samples[f"{label}_fraction"] = 0.0
        samples[f"{label}_peptide_list"] = ""
        return samples

    def _join_unique(s):
        return ";".join(sorted({str(v) for v in s if v and str(v) != "nan"}))

    target_agg = target_df.groupby(sample_key, as_index=False).agg(
        **{
            f"{label}_peptides": ("peptide", lambda x: len(set(x))),
            f"{label}_peptide_list": ("peptide", _join_unique),
        }
    )

    samples = samples.merge(target_agg, on=sample_key, how="left")
    samples[f"{label}_peptides"] = samples[f"{label}_peptides"].fillna(0).astype(int)
    samples[f"{label}_peptide_list"] = samples[f"{label}_peptide_list"].fillna("")
    samples[f"{label}_fraction"] = samples[f"{label}_peptides"] / samples["unique_peptides"].clip(
        lower=1
    )

    return samples.sort_values(f"{label}_peptides", ascending=False).reset_index(drop=True)
