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

"""Single-pass indexer for IEDB/CEDAR data with caching.

Builds two cached summary tables from each source CSV:

- **study_counts**: unique peptides and observations per
  (pmid, mhc_class, mhc_species)
- **allele_counts**: occurrence counts per unique MHC restriction string

The index is stored as parquet files in ``~/.hitlist/`` and reused
when the source CSV has not changed (checked via file size + mtime).

Usage::

    from hitlist.indexer import get_index

    study_df, allele_df = get_index("iedb")
    study_df, allele_df = get_index("merged")  # IEDB+CEDAR deduped
"""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .curation import _cached_parse  # noqa: F401 — reuse the LRU cache


def _fast_species(mhc_restriction: str, _cache: dict[str, str] = {}) -> str:  # noqa: B006
    """Species lookup with local dict cache (faster than LRU for hot path)."""
    if not mhc_restriction:
        return ""
    cached = _cache.get(mhc_restriction)
    if cached is not None:
        return cached

    from .curation import classify_mhc_species

    result = classify_mhc_species(mhc_restriction)
    _cache[mhc_restriction] = result
    return result


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
    """File identity: size + mtime."""
    stat = source_path.stat()
    return {"path": str(source_path), "size": stat.st_size, "mtime": stat.st_mtime}


def _cache_is_valid(label: str, source_path: Path) -> bool:
    meta_path = _cache_dir() / f"{label}_meta.json"
    if not meta_path.exists():
        return False
    stored = json.loads(meta_path.read_text())
    current = _cache_key(source_path)
    return stored.get("size") == current["size"] and stored.get("mtime") == current["mtime"]


def _save_cache(label: str, source_path: Path, study_df: pd.DataFrame, allele_df: pd.DataFrame):
    d = _cache_dir()
    study_df.to_parquet(d / f"{label}_study_counts.parquet", index=False)
    allele_df.to_parquet(d / f"{label}_allele_counts.parquet", index=False)
    meta = _cache_key(source_path)
    (d / f"{label}_meta.json").write_text(json.dumps(meta))


def _load_cache(label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _cache_dir()
    study_df = pd.read_parquet(d / f"{label}_study_counts.parquet")
    allele_df = pd.read_parquet(d / f"{label}_allele_counts.parquet")
    return study_df, allele_df


def _index_single_source(source_path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-pass index of one CSV. Returns (study_counts, allele_counts)."""
    from .scanner import _open_csv, _progress, _safe_col

    # (pmid, mhc_class, species) -> set of peptides
    peptide_sets: dict[tuple, set[str]] = defaultdict(set)
    obs_counts: dict[tuple, int] = defaultdict(int)
    allele_counts: dict[str, int] = defaultdict(int)

    reader, c, p = _open_csv(source_path)
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

    study_rows = []
    for (pmid, mhc_cls, species), peps in sorted(peptide_sets.items()):
        study_rows.append(
            {
                "source": label,
                "pmid": pmid,
                "mhc_class": mhc_cls,
                "mhc_species": species,
                "n_peptides": len(peps),
                "n_observations": obs_counts[(pmid, mhc_cls, species)],
            }
        )

    allele_rows = [
        {"allele": a, "n_occurrences": n}
        for a, n in sorted(allele_counts.items(), key=lambda x: -x[1])
    ]

    return pd.DataFrame(study_rows), pd.DataFrame(allele_rows)


def index_source(label: str, source_path: Path | None = None, force: bool = False):
    """Index a single source (iedb or cedar), using cache if valid.

    Returns (study_counts_df, allele_counts_df).
    """
    if source_path is None:
        paths = _resolve_source_paths()
        if label not in paths:
            raise FileNotFoundError(f"'{label}' not registered.")
        source_path = paths[label]

    if not force and _cache_is_valid(label, source_path):
        return _load_cache(label)

    study_df, allele_df = _index_single_source(source_path, label)
    _save_cache(label, source_path, study_df, allele_df)
    return study_df, allele_df


def get_index(
    source: str = "merged",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get indexed study counts and allele counts.

    Parameters
    ----------
    source
        ``"iedb"``, ``"cedar"``, ``"merged"`` (deduped), or ``"all"``
        (per-source rows concatenated).
    force
        Re-index even if cache is valid.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(study_counts, allele_counts)``
    """
    paths = _resolve_source_paths()
    if not paths:
        raise FileNotFoundError(
            "No IEDB/CEDAR data found. Register with: hitlist data register iedb /path/to/file.csv"
        )

    if source in ("iedb", "cedar"):
        return index_source(source, force=force)

    if source == "all":
        study_dfs, allele_dfs = [], []
        for label in sorted(paths):
            s, a = index_source(label, force=force)
            study_dfs.append(s)
            allele_dfs.append(a)
        return (
            pd.concat(study_dfs, ignore_index=True),
            pd.concat(allele_dfs, ignore_index=True),
        )

    # Merged: need IRI-based dedup — check if merged cache is valid
    # (valid if both source caches are valid)
    merged_meta = _cache_dir() / "merged_meta.json"
    if not force and merged_meta.exists():
        stored = json.loads(merged_meta.read_text())
        all_valid = True
        for label, path in paths.items():
            current = _cache_key(path)
            if stored.get(label) != current:
                all_valid = False
                break
        if all_valid:
            return _load_cache("merged")

    # Must do a merged scan with IRI dedup
    from .scanner import _open_csv, _progress, _safe_col

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

    study_df = pd.DataFrame(study_rows)
    allele_df = pd.DataFrame(allele_rows)

    # Cache merged result
    d = _cache_dir()
    study_df.to_parquet(d / "merged_study_counts.parquet", index=False)
    allele_df.to_parquet(d / "merged_allele_counts.parquet", index=False)
    meta = {label: _cache_key(path) for label, path in paths.items()}
    merged_meta.write_text(json.dumps(meta))

    return study_df, allele_df


def validate_alleles_from_index(
    allele_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate allele strings with mhcgnomes from an index allele_counts DataFrame.

    Fast: only parses each unique string once (typically ~500-2000 unique strings).
    """
    try:
        from mhcgnomes import parse
    except ImportError:
        allele_df = allele_df.copy()
        allele_df["parsed_name"] = ""
        allele_df["parsed_type"] = ""
        allele_df["species"] = ""
        allele_df["valid"] = False
        return allele_df

    rows = []
    for _, row in allele_df.iterrows():
        allele_str = row["allele"]
        result = parse(allele_str)
        rows.append(
            {
                "allele": allele_str,
                "n_occurrences": row["n_occurrences"],
                "parsed_name": str(result),
                "parsed_type": type(result).__name__,
                "species": result.species.name if hasattr(result, "species") else "",
                "valid": type(result).__name__ not in ("ParseError", "str"),
            }
        )
    return pd.DataFrame(rows)
