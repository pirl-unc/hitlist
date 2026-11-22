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


def _clean_tokens(series) -> list[str]:
    return [str(v).strip() for v in series if v is not None and str(v).strip() not in ("", "nan")]


def _join_unique(series) -> str:
    return ";".join(sorted(set(_clean_tokens(series))))


def _count_unique(series) -> int:
    return len(set(_clean_tokens(series)))


def _parse_allele_tokens(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "nan":
        return []
    return [tok.strip() for tok in text.split(";") if tok.strip() and tok.strip() != "nan"]


def _join_unique_alleles(series) -> str:
    """Deduplicate individual alleles parsed from single-allele and semicolon donor-set rows (#957)."""
    alleles: set[str] = set()
    for val in series:
        alleles.update(_parse_allele_tokens(val))
    return ";".join(sorted(alleles))


def _count_unique_alleles(series) -> int:
    """Count distinct individual alleles across single-allele and donor-set rows (#957)."""
    alleles: set[str] = set()
    for val in series:
        alleles.update(_parse_allele_tokens(val))
    return len(alleles)


def _join_exact_alleles(series) -> str:
    """Join distinct exact single-allele restrictions (excluding multi-allelic donor sets) (#957)."""
    exact: set[str] = set()
    for val in series:
        tokens = _parse_allele_tokens(val)
        if len(tokens) == 1:
            exact.add(tokens[0])
    return ";".join(sorted(exact))


def _join_restriction_sets(series) -> str:
    """Join distinct raw restriction strings with '|' to preserve donor-set boundaries (#957)."""
    return "|".join(sorted(set(_clean_tokens(series))))


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


def _sort_pmhc_by_evidence(result: pd.DataFrame) -> pd.DataFrame:
    """Sort pMHC summary rows within each peptide by evidence strength (#957).

    Within each peptide (ascending):
    1. Exact single-allele restriction ahead of multi-allelic donor sets (containing ';')
    2. Mono-allelic evidence ahead of poly-allelic-only rows
    3. Distinct PMID/reference count descending
    4. Hit count descending
    5. ``mhc_restriction`` ascending for deterministic tie-breaking
    """
    if result.empty:
        return result
    df = result.copy()
    df["_sort_is_donor_set"] = df["mhc_restriction"].astype(str).str.contains(";", regex=False)
    if "ms_pmhc_has_mono_evidence" in df.columns:
        df["_sort_no_mono"] = ~df["ms_pmhc_has_mono_evidence"].astype(bool)
    elif "ms_pmhc_mono_hit_count" in df.columns:
        df["_sort_no_mono"] = df["ms_pmhc_mono_hit_count"].fillna(0) <= 0
    else:
        df["_sort_no_mono"] = False

    if "ms_pmhc_pmid_count" in df.columns:
        df["_sort_neg_refs"] = -df["ms_pmhc_pmid_count"].fillna(0).astype(int)
    elif "ms_pmhc_ref_count" in df.columns:
        df["_sort_neg_refs"] = -df["ms_pmhc_ref_count"].fillna(0).astype(int)
    else:
        df["_sort_neg_refs"] = 0

    df["_sort_neg_hits"] = -df["ms_pmhc_hit_count"].fillna(0).astype(int)
    df = df.sort_values(
        by=[
            "peptide",
            "_sort_is_donor_set",
            "_sort_no_mono",
            "_sort_neg_refs",
            "_sort_neg_hits",
            "mhc_restriction",
        ],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    )
    return df.drop(
        columns=["_sort_is_donor_set", "_sort_no_mono", "_sort_neg_refs", "_sort_neg_hits"]
    ).reset_index(drop=True)


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
        deduplicated individual alleles (``ms_alleles``), exact single-allele
        restrictions (``ms_exact_alleles``), pipe-delimited raw restriction sets
        (``ms_restriction_sets``), source flags (``found_in_*``), tissue/disease/cell
        line lists, and ``is_cancer_specific`` flag.
    """
    if hits_df.empty:
        return pd.DataFrame()

    agg: dict = {
        "ms_hit_count": ("peptide", "size"),
        "ms_alleles": ("mhc_restriction", _join_unique_alleles),
        "ms_allele_count": ("mhc_restriction", _count_unique_alleles),
        "ms_exact_alleles": ("mhc_restriction", _join_exact_alleles),
        "ms_restriction_sets": ("mhc_restriction", _join_restriction_sets),
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
        "src_healthy_reproductive_female",
        "src_healthy_reproductive_male",
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

    Rows within each peptide are sorted by evidence strength so exact
    single-allele and mono-allelic hits lead multi-allelic donor sets (#957).
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
    return _sort_pmhc_by_evidence(result)


def aggregate_per_pmhc_with_refs(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(peptide, allele) aggregation with full provenance.

    Rows within each peptide are sorted by evidence strength so exact
    single-allele and mono-allelic hits lead multi-allelic donor sets (#957).
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
    return _sort_pmhc_by_evidence(result)
