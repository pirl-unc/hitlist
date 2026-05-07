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

"""Summary index derived from the unified observations table.

Provides aggregated study counts and allele counts from
:func:`hitlist.observations.load_observations`.  The legacy CSV-scan
fallback (which wrote a per-source allele-counts cache to
``~/.hitlist/index/``) was removed in v1.30.41 — the index is always
derived from ``observations.parquet`` so it reflects the full curated
corpus (IEDB + CEDAR + supplements + curation overrides + per-peptide
attribution) rather than the raw IEDB/CEDAR CSVs alone.

Usage::

    from hitlist.indexer import get_index

    study_df, allele_df = get_index()
    study_df, allele_df = get_index(source="iedb")
"""

from __future__ import annotations

import pandas as pd


def get_index(source: str = "merged") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get study counts and allele counts from the built observations table.

    Parameters
    ----------
    source
        ``"iedb"``, ``"cedar"``, ``"merged"`` (default), or ``"all"``.
        ``"merged"`` returns one row per (pmid, mhc_class, mhc_species)
        across all sources; ``"all"`` splits per-source.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(study_counts, allele_counts)``.  Counts include curated
        supplements + per-PMID overrides + per-peptide attribution
        because they're derived from ``observations.parquet``.

    Raises
    ------
    FileNotFoundError
        ``observations.parquet`` is not built.  Run
        ``hitlist build observations`` first.
    """
    from .observations import is_built, load_observations

    if not is_built():
        raise FileNotFoundError(
            "observations.parquet is not built.  Run: hitlist build observations"
        )

    if source in ("iedb", "cedar"):
        obs = load_observations(source=source)
    else:
        obs = load_observations()

    source_label = source if source in ("iedb", "cedar") else "iedb+cedar"

    # Study counts: group by (pmid, mhc_class, mhc_species)
    study_groups = obs.dropna(subset=["pmid"]).groupby(["pmid", "mhc_class", "mhc_species"])
    study_rows = []
    for (pmid, mhc_cls, species), group in study_groups:
        study_rows.append(
            {
                "source": source_label,
                "pmid": int(pmid),
                "mhc_class": mhc_cls,
                "mhc_species": species,
                "n_peptides": group["peptide"].nunique(),
                "n_observations": len(group),
            }
        )
    study_df = pd.DataFrame(study_rows)

    # Allele counts
    allele_counts = obs["mhc_restriction"].value_counts()
    allele_df = pd.DataFrame({"allele": allele_counts.index, "n_occurrences": allele_counts.values})

    if source == "all":
        # Split by source column
        study_dfs = []
        for src_label in obs["source"].unique():
            src_obs = obs[obs["source"] == src_label]
            src_groups = src_obs.dropna(subset=["pmid"]).groupby(
                ["pmid", "mhc_class", "mhc_species"]
            )
            for (pmid, mhc_cls, species), group in src_groups:
                study_dfs.append(
                    {
                        "source": src_label,
                        "pmid": int(pmid),
                        "mhc_class": mhc_cls,
                        "mhc_species": species,
                        "n_peptides": group["peptide"].nunique(),
                        "n_observations": len(group),
                    }
                )
        study_df = pd.DataFrame(study_dfs)

    return study_df, allele_df


def validate_alleles_from_index(allele_df: pd.DataFrame) -> pd.DataFrame:
    """Validate allele strings with mhcgnomes."""
    try:
        from mhcgnomes import parse
    except ImportError:
        allele_df = allele_df.copy()
        for col in ("parsed_name", "parsed_type", "species"):
            allele_df[col] = ""
        allele_df["valid"] = False
        return allele_df

    rows = []
    for _, row in allele_df.iterrows():
        allele_str = row["allele"]
        try:
            result = parse(allele_str)
            rows.append(
                {
                    "allele": allele_str,
                    "n_occurrences": row["n_occurrences"],
                    "parsed_name": str(result),
                    "parsed_type": type(result).__name__,
                    "species": result.species.name if hasattr(result, "species") else "",
                    "valid": True,
                }
            )
        except Exception:
            rows.append(
                {
                    "allele": allele_str,
                    "n_occurrences": row["n_occurrences"],
                    "parsed_name": "",
                    "parsed_type": "ParseError",
                    "species": "",
                    "valid": False,
                }
            )
    return pd.DataFrame(rows)
