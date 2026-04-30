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

"""Corpus + curation QC checks.

Three independent diagnostics, each returning a DataFrame so consumers can
print, write to CSV, or feed into a notebook:

- :func:`resolution_histogram` — count rows per (mhc_class, source,
  allele_resolution) bucket.  Tells you what fraction of the corpus is
  4-digit / 2-digit / serotype / class-only / unresolved.
- :func:`normalization_drift` — alleles in ``pmid_overrides.yaml`` whose
  ``normalize_allele`` output differs from the curated input.  Catches
  silent normalization changes.
- :func:`cross_reference` — alleles listed in YAML sample genotypes that
  never appear as ``mhc_restriction`` in the data for that PMID, and the
  reverse: data alleles for a PMID never listed in any sample.  Catches
  curation/data divergence.

Each function returns the same shape: a DataFrame with one row per
finding plus a ``severity`` column (``info`` / ``warn`` / ``error``) so
:func:`run_all` can produce a unified summary.
"""

from __future__ import annotations

import pandas as pd

from .curation import (
    _flatten_hla_alleles,
    load_pmid_overrides,
    normalize_allele,
)


def resolution_histogram(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Count observations per ``(mhc_class, source, allele_resolution)``.

    Parameters
    ----------
    mhc_class, species, source
        Optional filters passed through to ``load_observations``.

    Returns
    -------
    pd.DataFrame
        Columns: ``mhc_class``, ``source``, ``allele_resolution``,
        ``n_observations``, ``pct_within_class``.  Sorted by class then
        bucket so the most-resolved row in each class is first.
    """
    from .observations import is_built, load_observations

    if not is_built():
        raise FileNotFoundError(
            "observations.parquet has not been built. Run 'hitlist data build' first."
        )

    df = load_observations(
        mhc_class=mhc_class,
        species=species,
        source=source,
        columns=["mhc_class", "source", "allele_resolution"],
    )
    if df.empty:
        return pd.DataFrame(
            columns=[
                "mhc_class",
                "source",
                "allele_resolution",
                "n_observations",
                "pct_within_class",
            ]
        )

    counts = (
        df.groupby(["mhc_class", "source", "allele_resolution"], dropna=False)
        .size()
        .reset_index(name="n_observations")
    )
    class_totals = counts.groupby("mhc_class")["n_observations"].transform("sum")
    counts["pct_within_class"] = (counts["n_observations"] / class_totals * 100).round(2)

    # Order buckets most-resolved to least so output is readable.
    bucket_order = {
        "four_digit": 0,
        "two_digit": 1,
        "serological": 2,
        "class_only": 3,
        "unresolved": 4,
    }
    counts["_bucket_rank"] = counts["allele_resolution"].map(bucket_order).fillna(99)
    counts = counts.sort_values(["mhc_class", "_bucket_rank", "source"], kind="stable").drop(
        columns="_bucket_rank"
    )
    return counts.reset_index(drop=True)


def normalization_drift() -> pd.DataFrame:
    """Find alleles in pmid_overrides whose normalized form differs from input.

    Walks every ``hla_alleles`` value in every PMID, runs each 4-digit
    string through ``normalize_allele``, and emits a row whenever the
    output differs from the input.

    Returns
    -------
    pd.DataFrame
        Columns: ``pmid``, ``study_label``, ``allele_raw``,
        ``allele_normalized``, ``severity``.  Empty if no drift exists.
    """
    overrides = load_pmid_overrides()
    rows: list[dict] = []
    for pmid_int, entry in sorted(overrides.items()):
        study_label = entry.get("study_label", "")
        hla_alleles = entry.get("hla_alleles", {})
        if not hla_alleles:
            continue
        # _flatten_hla_alleles handles flat list, dict-of-lists, and
        # dict-of-strings shapes uniformly.
        for allele_raw in sorted(_flatten_hla_alleles(hla_alleles)):
            allele_norm = normalize_allele(allele_raw)
            if allele_norm and allele_norm != allele_raw:
                rows.append(
                    {
                        "pmid": pmid_int,
                        "study_label": study_label,
                        "allele_raw": allele_raw,
                        "allele_normalized": allele_norm,
                        "severity": "warn",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "pmid",
            "study_label",
            "allele_raw",
            "allele_normalized",
            "severity",
        ],
    )


def cross_reference(mhc_class: str | None = None) -> pd.DataFrame:
    """Find allele/data mismatches between curation YAML and observations.

    Two divergence directions:

    - **yaml_only**: an allele is listed in a YAML sample's ``mhc`` block
      but no observation row for that PMID has that ``mhc_restriction``.
      Likely a typo in curation, or genotype curated from the paper text
      but no measurements actually attributed to that allele.
    - **data_only**: a 4-digit allele appears as ``mhc_restriction`` for
      a PMID but no YAML sample for that PMID lists it.  Likely a curation
      gap (sample present in data but not yet added to YAML).

    Parameters
    ----------
    mhc_class
        Filter to ``"I"`` or ``"II"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``pmid``, ``study_label``, ``allele``, ``direction``,
        ``severity``.  ``direction`` is ``"yaml_only"`` or ``"data_only"``.
    """
    from .observations import is_built, load_observations

    if not is_built():
        raise FileNotFoundError(
            "observations.parquet has not been built. Run 'hitlist data build' first."
        )

    obs = load_observations(
        mhc_class=mhc_class,
        columns=["pmid", "mhc_restriction", "allele_resolution"],
    )
    # Only 4-digit data alleles are comparable to curated YAML alleles.
    obs_4d = obs[obs["allele_resolution"] == "four_digit"]
    data_by_pmid: dict[int, set[str]] = (
        obs_4d.groupby("pmid")["mhc_restriction"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )

    overrides = load_pmid_overrides()
    rows: list[dict] = []
    for pmid_int, entry in sorted(overrides.items()):
        study_label = entry.get("study_label", "")
        # YAML-curated alleles for this PMID (sample-level + paper-level pool).
        yaml_alleles = _flatten_hla_alleles(entry.get("hla_alleles", {}))
        ms_samples = entry.get("ms_samples", []) or []
        for sample in ms_samples:
            yaml_alleles |= _flatten_hla_alleles(sample.get("mhc", []))

        data_alleles = data_by_pmid.get(pmid_int, set())

        for allele in sorted(yaml_alleles - data_alleles):
            rows.append(
                {
                    "pmid": pmid_int,
                    "study_label": study_label,
                    "allele": allele,
                    "direction": "yaml_only",
                    "severity": "warn",
                }
            )
        for allele in sorted(data_alleles - yaml_alleles):
            rows.append(
                {
                    "pmid": pmid_int,
                    "study_label": study_label,
                    "allele": allele,
                    "direction": "data_only",
                    "severity": "info",
                }
            )

    return pd.DataFrame(
        rows,
        columns=["pmid", "study_label", "allele", "direction", "severity"],
    )


def discrepancies(
    mhc_class: str | None = None,
    min_rows: int = 50,
    by: str = "pmid",
) -> pd.DataFrame:
    """Per-PMID (or per-sample) rate of biologically suspicious patterns.

    A scan over ``observations.parquet`` that surfaces curation issues
    visible from the data alone — without re-reading any papers. One
    output row per bucket, sorted by suspect-row score so a curator
    picks targets from the top.

    Detected patterns:

    - **suspect_class_label_n / _rate** — count of rows where the
      bimodal length distribution disagrees with the curated class
      (class II ≤10aa or class I ≥18aa). Pinned by the
      ``mhc_class_label_suspect`` flag added in v1.30.0 / #182.
    - **length_p50 / _p99** — peptide length median + 99th percentile
      per bucket. Class I should have p50 ≈ 9 and a thin upper tail
      (p99 ≤ 12); class II should have p50 ≈ 14-15. Outliers on
      either tail are usually IEDB curation drift.
    - **monoallelic_class_only_n** — mono-allelic rows whose
      ``mhc_restriction`` is the class sentinel ("HLA class I/II")
      rather than a 4-digit allele. These are #45 candidates: the
      paper knows the allele, IEDB lost it.
    - **class_pool_n / _rate** — rows whose ``mhc_allele_provenance``
      came down the pmid_class_pool fallback (no per-peptide allele
      resolution, the #37 problem).
    - **nonstandard_aa_n** — peptides containing ``X`` / ``B`` /
      ``Z`` / ``U`` / ``O`` / ``*`` / lowercase / digits. Either
      ambiguous IDs from MS or upstream string corruption.

    Parameters
    ----------
    mhc_class
        Optional class filter.
    min_rows
        Drop buckets with fewer than this many rows — small buckets
        produce noisy length percentiles. Default 50.
    by
        Aggregation level. ``"pmid"`` (default) groups by
        (pmid, mhc_class) — one row per study/class. ``"sample"``
        groups by (pmid, mhc_class, cell_name) so a curator can spot
        per-sample issues like "the K562-A0201 transfectant has 30%
        suspect rows but K562-B0702 has 0%". Falls back to
        ``"(no cell_name)"`` for rows without a sample identifier.

    Returns
    -------
    pd.DataFrame
        One row per bucket. Sorted descending by
        ``suspect_class_label_n + monoallelic_class_only_n +
        nonstandard_aa_n`` — the rougher cuts, useful for picking a
        triage target.
    """
    import re

    from .observations import is_built, load_observations

    if by not in ("pmid", "sample"):
        raise ValueError(f"by must be 'pmid' or 'sample', got {by!r}")

    if not is_built():
        raise FileNotFoundError(
            "observations.parquet has not been built. Run 'hitlist data build' first."
        )

    cols = [
        "pmid",
        "peptide",
        "mhc_class",
        "mhc_restriction",
        "is_monoallelic",
        "mhc_class_label_suspect",
    ]
    if by == "sample":
        cols.append("cell_name")
    # mhc_allele_provenance is only present in indexes built since #137
    # (v1.23.0); degrade gracefully on older parquets.
    try:
        sample = load_observations(columns=["mhc_allele_provenance"])
        if "mhc_allele_provenance" in sample.columns:
            cols.append("mhc_allele_provenance")
    except Exception:
        pass

    df = load_observations(mhc_class=mhc_class, columns=cols)
    output_cols = [
        "pmid",
        "study_label",
        "mhc_class",
    ]
    if by == "sample":
        output_cols.append("cell_name")
    output_cols.extend(
        [
            "n_rows",
            "suspect_class_label_n",
            "suspect_class_label_rate",
            "length_p50",
            "length_p99",
            "monoallelic_class_only_n",
            "class_pool_n",
            "class_pool_rate",
            "nonstandard_aa_n",
            "severity",
        ]
    )
    if df.empty:
        return pd.DataFrame(columns=output_cols)

    # Compute per-row diagnostics in one pass to avoid groupby overhead
    # on the 4.4M-row corpus.
    df = df.copy()
    df["_len"] = df["peptide"].str.len()
    nonstandard_re = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")
    df["_nonstandard"] = df["peptide"].astype(str).str.contains(nonstandard_re)
    df["_class_only"] = df["mhc_restriction"].fillna("").str.startswith("HLA class")
    df["_mono_class_only"] = df["is_monoallelic"].fillna(False) & df["_class_only"]
    if "mhc_allele_provenance" in df.columns:
        df["_class_pool"] = df["mhc_allele_provenance"] == "pmid_class_pool"
    else:
        df["_class_pool"] = False

    if by == "sample":
        df["cell_name"] = df["cell_name"].fillna("").replace("", "(no cell_name)")
        group_keys = ["pmid", "mhc_class", "cell_name"]
    else:
        group_keys = ["pmid", "mhc_class"]

    grouped = df.groupby(group_keys, dropna=False)
    out = grouped.agg(
        n_rows=("peptide", "size"),
        suspect_class_label_n=("mhc_class_label_suspect", "sum"),
        length_p50=("_len", lambda s: int(s.median()) if len(s) else 0),
        length_p99=("_len", lambda s: int(s.quantile(0.99)) if len(s) else 0),
        monoallelic_class_only_n=("_mono_class_only", "sum"),
        class_pool_n=("_class_pool", "sum"),
        nonstandard_aa_n=("_nonstandard", "sum"),
    ).reset_index()
    out = out[out["n_rows"] >= min_rows].copy()
    out["suspect_class_label_rate"] = out["suspect_class_label_n"] / out["n_rows"]
    out["class_pool_rate"] = out["class_pool_n"] / out["n_rows"]

    # Attach study labels from pmid_overrides.yaml.
    overrides = load_pmid_overrides()
    out["study_label"] = out["pmid"].map(
        lambda p: overrides.get(int(p), {}).get("study_label", "") if pd.notna(p) else ""
    )

    # Severity heuristic — pick the worst signal per row.
    def _severity(row) -> str:
        if (
            row["suspect_class_label_rate"] >= 0.05
            or row["monoallelic_class_only_n"] > 0
            or row["nonstandard_aa_n"] > 0
        ):
            return "warn"
        if row["class_pool_rate"] >= 0.5:
            return "info"
        return "info"

    out["severity"] = out.apply(_severity, axis=1)
    out["_score"] = (
        out["suspect_class_label_n"] + out["monoallelic_class_only_n"] + out["nonstandard_aa_n"]
    )
    out = out.sort_values(["_score", "n_rows"], ascending=[False, False], kind="stable").drop(
        columns=["_score"]
    )

    return out[output_cols].reset_index(drop=True)


def run_all(mhc_class: str | None = None) -> dict[str, pd.DataFrame]:
    """Run every QC check and return a name → DataFrame mapping.

    The CLI ``hitlist qc`` (no subcommand) calls this and prints a summary
    table per check.
    """
    return {
        "resolution": resolution_histogram(mhc_class=mhc_class),
        "normalization": normalization_drift(),
        "cross_reference": cross_reference(mhc_class=mhc_class),
        "discrepancies": discrepancies(mhc_class=mhc_class),
    }
