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


def run_all(mhc_class: str | None = None) -> dict[str, pd.DataFrame]:
    """Run every QC check and return a name → DataFrame mapping.

    The CLI ``hitlist qc`` (no subcommand) calls this and prints a summary
    table per check.
    """
    return {
        "resolution": resolution_histogram(mhc_class=mhc_class),
        "normalization": normalization_drift(),
        "cross_reference": cross_reference(mhc_class=mhc_class),
    }
