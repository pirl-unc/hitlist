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

    result = hits_df.groupby(["peptide", "mhc_restriction"], as_index=False).agg(**agg)
    return result[result["mhc_restriction"].str.startswith("HLA-", na=False)].reset_index(drop=True)
