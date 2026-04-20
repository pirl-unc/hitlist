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

"""Aggregation of per-row MS evidence into per-peptide and per-pMHC summaries."""

from __future__ import annotations

import pandas as pd

from .curation import is_cancer_specific


def _join_unique(series) -> str:
    return ";".join(sorted({str(v) for v in series if v and str(v) != "nan"}))


def _count_unique(series) -> int:
    return len({str(v) for v in series if v and str(v) != "nan"})


def _join_unique_numeric(series) -> str:
    """Like :func:`_join_unique` but sorts numerically — use for integer
    identifiers like PMIDs so a 9-digit PMID doesn't land before an 8-digit
    one under lexicographic ordering."""
    values: set[int] = set()
    for v in series:
        if not v or str(v) == "nan":
            continue
        try:
            values.add(int(v))
        except (TypeError, ValueError):
            continue
    return ";".join(str(x) for x in sorted(values))


def aggregate_per_peptide(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-row MS evidence into per-peptide summary.

    Parameters
    ----------
    hits_df
        DataFrame from :func:`hitlist.scanner.scan` with ``classify_source=True``.

    Returns
    -------
    pd.DataFrame
        One row per peptide with: hit count, ref/PMID/allele counts,
        source flags (found_in_*), tissue/disease/cell line lists,
        is_cancer_specific flag.
    """
    if hits_df.empty:
        return pd.DataFrame()

    agg: dict = {
        "ms_hit_count": ("peptide", "size"),
        "ms_alleles": ("mhc_restriction", _join_unique),
        "ms_allele_count": ("mhc_restriction", _count_unique),
    }
    if "reference_iri" in hits_df.columns:
        agg["ms_ref_count"] = ("reference_iri", _count_unique)
    if "pmid" in hits_df.columns:
        agg["ms_pmid_count"] = ("pmid", _count_unique)
        agg["ms_pmids"] = ("pmid", _join_unique)

    for flag in [
        "src_cancer",
        "src_adjacent_to_tumor",
        "src_activated_apc",
        "src_healthy_tissue",
        "src_healthy_thymus",
        "src_healthy_reproductive",
        "src_cell_line",
        "src_ebv_lcl",
    ]:
        if flag in hits_df.columns:
            agg[f"found_in_{flag.removeprefix('src_')}"] = (flag, "any")

    if "is_monoallelic" in hits_df.columns:
        agg["mono_allelic_hit_count"] = ("is_monoallelic", "sum")
        agg["has_mono_allelic_evidence"] = ("is_monoallelic", "any")

    if "source_tissue" in hits_df.columns:
        agg["ms_tissues"] = ("source_tissue", _join_unique)
    if "disease" in hits_df.columns:
        agg["ms_diseases"] = ("disease", _join_unique)
    if "cell_line_name" in hits_df.columns:
        agg["ms_cell_lines"] = ("cell_line_name", lambda x: ";".join(sorted({v for v in x if v})))

    result = hits_df.groupby("peptide", as_index=False).agg(**agg)
    result["is_cancer_specific"] = result.apply(lambda r: is_cancer_specific(r.to_dict()), axis=1)
    return result


def aggregate_per_pmhc(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-row MS evidence into per-(peptide, allele) summary.

    Parameters
    ----------
    hits_df
        DataFrame from :func:`hitlist.scanner.scan`.

    Returns
    -------
    pd.DataFrame
        One row per (peptide, allele) pair.
    """
    if hits_df.empty or "mhc_restriction" not in hits_df.columns:
        return pd.DataFrame()

    agg: dict = {
        "ms_pmhc_hit_count": ("peptide", "size"),
    }
    if "reference_iri" in hits_df.columns:
        agg["ms_pmhc_ref_count"] = ("reference_iri", _count_unique)
    if "pmid" in hits_df.columns:
        agg["ms_pmhc_pmid_count"] = ("pmid", _count_unique)
        agg["ms_pmhc_pmids"] = ("pmid", _join_unique)
    if "is_monoallelic" in hits_df.columns:
        agg["ms_pmhc_mono_hit_count"] = ("is_monoallelic", "sum")
        agg["ms_pmhc_has_mono_evidence"] = ("is_monoallelic", "any")

    result = hits_df.groupby(["peptide", "mhc_restriction"], as_index=False).agg(**agg)
    return result[result["mhc_restriction"].str.startswith("HLA-", na=False)].reset_index(drop=True)


def aggregate_per_pmhc_with_refs(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(peptide, allele) aggregation with full provenance.

    Sits between :func:`aggregate_per_peptide` (per-peptide roll-up — carries
    provenance but loses allele granularity) and :func:`aggregate_per_pmhc`
    (per-pMHC but lean).  Keeps the (peptide, allele) granularity **and**
    surfaces the provenance columns reviewers typically want on a pMHC row:
    reference count, PMID list, distinct tissues / diseases / cell lines,
    and the cancer / healthy-tissue source flags.

    Intended for therapy-prioritization workflows where a reviewer evaluates
    a specific (peptide, allele) pair for vaccine / TCR use and wants to see
    which studies and tissues support the call without rolling up across
    alleles.

    Parameters
    ----------
    hits_df
        DataFrame from :func:`hitlist.scanner.scan`.  Only ``peptide`` and
        ``mhc_restriction`` are required; optional columns (``pmid``,
        ``source_tissue``, ``disease``, ``cell_line_name``, ``src_cancer``,
        ``src_healthy_tissue``, ``is_monoallelic``) are used when present
        and silently skipped when absent so both the cached-observations
        fast path and the raw-scan slow path produce valid output.

    Returns
    -------
    pd.DataFrame
        One row per (peptide, allele).  **Output columns depend on which
        optional inputs were supplied:** only ``peptide``, ``length``,
        ``mhc_restriction``, ``ms_pmhc_hit_count`` are guaranteed.
        The rest (``ms_pmhc_ref_count``, ``ms_pmhc_pmids``,
        ``ms_pmhc_tissues``, ``ms_pmhc_diseases``, ``ms_pmhc_cell_lines``,
        ``ms_pmhc_in_cancer``, ``ms_pmhc_in_healthy_tissue``,
        ``ms_pmhc_mono_hit_count``) appear only when the corresponding
        input column was present.  The empty-frame path returns the full
        canonical column list for downstream shape stability.

        PMIDs are sorted numerically (not lexicographically), so a 9-digit
        PMID will not misrank against an 8-digit one.

        Rows are filtered to those whose ``mhc_restriction`` begins with
        ``"HLA-"`` — matches :func:`aggregate_per_pmhc` behavior.  Pass
        non-HLA data through a different path.
    """
    canonical_columns = [
        "peptide",
        "length",
        "mhc_restriction",
        "ms_pmhc_hit_count",
        "ms_pmhc_ref_count",
        "ms_pmhc_pmids",
        "ms_pmhc_tissues",
        "ms_pmhc_diseases",
        "ms_pmhc_cell_lines",
        "ms_pmhc_in_cancer",
        "ms_pmhc_in_healthy_tissue",
        "ms_pmhc_mono_hit_count",
    ]
    if hits_df.empty or "mhc_restriction" not in hits_df.columns:
        return pd.DataFrame(columns=canonical_columns)

    agg: dict = {"ms_pmhc_hit_count": ("peptide", "size")}
    if "pmid" in hits_df.columns:
        agg["ms_pmhc_ref_count"] = ("pmid", _count_unique)
        agg["ms_pmhc_pmids"] = ("pmid", _join_unique_numeric)
    if "source_tissue" in hits_df.columns:
        agg["ms_pmhc_tissues"] = ("source_tissue", _join_unique)
    if "disease" in hits_df.columns:
        agg["ms_pmhc_diseases"] = ("disease", _join_unique)
    if "cell_line_name" in hits_df.columns:
        agg["ms_pmhc_cell_lines"] = ("cell_line_name", _join_unique)
    if "src_cancer" in hits_df.columns:
        agg["ms_pmhc_in_cancer"] = ("src_cancer", "any")
    if "src_healthy_tissue" in hits_df.columns:
        agg["ms_pmhc_in_healthy_tissue"] = ("src_healthy_tissue", "any")
    if "is_monoallelic" in hits_df.columns:
        agg["ms_pmhc_mono_hit_count"] = ("is_monoallelic", "sum")

    result = hits_df.groupby(["peptide", "mhc_restriction"], as_index=False).agg(**agg)
    result.insert(1, "length", result["peptide"].str.len())
    return result[result["mhc_restriction"].str.startswith("HLA-", na=False)].reset_index(drop=True)
