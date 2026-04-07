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
:func:`hitlist.observations.load_observations`. Falls back to
direct CSV scanning if the observations table has not been built.

Usage::

    from hitlist.indexer import get_index

    study_df, allele_df = get_index()
    study_df, allele_df = get_index(source="iedb")
"""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _resolve_source_paths() -> dict[str, Path]:
    """Resolve registered IEDB/CEDAR paths."""
    from .downloads import get_path

    sources: dict[str, Path] = {}
    for name in ("iedb", "cedar"):
        with contextlib.suppress(KeyError, FileNotFoundError):
            p = get_path(name)
            if p.exists():
                sources[name] = p
    return sources


def _cache_dir() -> Path:
    from .downloads import data_dir

    d = data_dir() / "index"
    d.mkdir(exist_ok=True)
    return d


def _cache_key(source_path: Path) -> dict:
    stat = source_path.stat()
    return {"path": str(source_path), "size": stat.st_size, "mtime": stat.st_mtime}


def _cache_is_valid(label: str, source_path: Path) -> bool:
    meta_path = _cache_dir() / f"{label}_meta.json"
    if not meta_path.exists():
        return False
    stored = json.loads(meta_path.read_text())
    current = _cache_key(source_path)
    return stored.get("size") == current["size"] and stored.get("mtime") == current["mtime"]


def get_index(
    source: str = "merged",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get study counts and allele counts.

    Reads from the built observations table if available (fast).
    Falls back to direct CSV scanning if not built.

    Parameters
    ----------
    source
        ``"iedb"``, ``"cedar"``, ``"merged"`` (default), or ``"all"``.
    force
        Ignored when reading from observations.parquet.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(study_counts, allele_counts)``
    """
    from .observations import is_built

    if is_built():
        return _index_from_observations(source)

    # Fallback: direct CSV scan (legacy path)
    return _index_from_csv(source, force)


def _index_from_observations(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive index from the built observations table."""
    from .observations import load_observations

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


def _index_from_csv(source: str, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy fallback: scan CSV directly when observations table not built."""
    from .curation import classify_mhc_species
    from .scanner import _open_csv, _progress, _safe_col

    def _fast_species(mhc_restriction: str, _cache: dict[str, str] = {}) -> str:  # noqa: B006
        if not mhc_restriction:
            return ""
        cached = _cache.get(mhc_restriction)
        if cached is not None:
            return cached
        result = classify_mhc_species(mhc_restriction)
        _cache[mhc_restriction] = result
        return result

    paths = _resolve_source_paths()
    if not paths:
        raise FileNotFoundError(
            "No IEDB/CEDAR data found. Register with: hitlist data register iedb /path/to/file.csv"
        )

    if source in ("iedb", "cedar"):
        if source not in paths:
            raise FileNotFoundError(f"'{source}' not registered.")
        return _scan_single(paths[source], source, _fast_species, _open_csv, _progress, _safe_col)

    # Merged or all
    peptide_sets: dict[tuple, set[str]] = defaultdict(set)
    obs_counts: dict[tuple, int] = defaultdict(int)
    allele_counts: dict[str, int] = defaultdict(int)
    seen_iris: set[str] = set()

    for _label, source_path in sorted(paths.items()):
        reader, c, p = _open_csv(source_path)
        for row in _progress(reader, p, f"Indexing {p.name}"):
            iri = row[c["assay_iri"]] if row else ""
            if iri in seen_iris:
                continue
            seen_iris.add(iri)

            mhc_res = _safe_col(row, c["mhc_restriction"])
            if mhc_res:
                allele_counts[mhc_res] += 1

            pep = _safe_col(row, c["epitope_name"])
            if not pep:
                continue
            raw_pmid = _safe_col(row, c["pmid"]).strip()
            if not raw_pmid:
                continue
            try:
                pmid = int(raw_pmid)
            except ValueError:
                continue

            mhc_cls = _safe_col(row, c["mhc_class"])
            species = _fast_species(mhc_res)
            if not species:
                host = _safe_col(row, c["host"])
                species = host if host else "unknown"

            key = (pmid, mhc_cls, species)
            peptide_sets[key].add(pep)
            obs_counts[key] += 1

    study_rows = [
        {
            "source": "iedb+cedar",
            "pmid": pmid,
            "mhc_class": mhc_cls,
            "mhc_species": species,
            "n_peptides": len(peps),
            "n_observations": obs_counts[(pmid, mhc_cls, species)],
        }
        for (pmid, mhc_cls, species), peps in sorted(peptide_sets.items())
    ]
    allele_rows = [
        {"allele": a, "n_occurrences": n}
        for a, n in sorted(allele_counts.items(), key=lambda x: -x[1])
    ]

    return pd.DataFrame(study_rows), pd.DataFrame(allele_rows)


def _scan_single(path, label, _fast_species, _open_csv, _progress, _safe_col):
    """Scan a single CSV source."""
    peptide_sets: dict[tuple, set[str]] = defaultdict(set)
    obs_counts: dict[tuple, int] = defaultdict(int)
    allele_counts: dict[str, int] = defaultdict(int)

    reader, c, p = _open_csv(path)
    for row in _progress(reader, p, f"Indexing {p.name}"):
        mhc_res = _safe_col(row, c["mhc_restriction"])
        if mhc_res:
            allele_counts[mhc_res] += 1
        pep = _safe_col(row, c["epitope_name"])
        if not pep:
            continue
        raw_pmid = _safe_col(row, c["pmid"]).strip()
        if not raw_pmid:
            continue
        try:
            pmid = int(raw_pmid)
        except ValueError:
            continue
        mhc_cls = _safe_col(row, c["mhc_class"])
        species = _fast_species(mhc_res)
        if not species:
            host = _safe_col(row, c["host"])
            species = host if host else "unknown"
        key = (pmid, mhc_cls, species)
        peptide_sets[key].add(pep)
        obs_counts[key] += 1

    study_rows = [
        {
            "source": label,
            "pmid": pmid,
            "mhc_class": mhc_cls,
            "mhc_species": species,
            "n_peptides": len(peps),
            "n_observations": obs_counts[(pmid, mhc_cls, species)],
        }
        for (pmid, mhc_cls, species), peps in sorted(peptide_sets.items())
    ]
    allele_rows = [
        {"allele": a, "n_occurrences": n}
        for a, n in sorted(allele_counts.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(study_rows), pd.DataFrame(allele_rows)


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
