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

"""Export curated study metadata as structured tables.

Reads ``pmid_overrides.yaml`` and generates per-sample, per-species,
and allele-validation reports from the ``ms_samples`` and ``hla_alleles``
metadata fields.

Can also scan local IEDB/CEDAR CSV files to count actual peptides
per study, species, and MHC class.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .curation import classify_mhc_species, load_pmid_overrides


def generate_ms_samples_table(mhc_class: str | None = None) -> pd.DataFrame:
    """Export all ms_samples entries as a flat DataFrame.

    Parameters
    ----------
    mhc_class
        Filter to ``"I"`` or ``"II"``.  Entries with ``"I+II"`` match
        either filter.  ``None`` returns all rows.

    Returns
    -------
    pd.DataFrame
        Columns: species, sample, perturbation, pmid, study, mhc_class,
        n_samples, peptides, peptides_estimated, notes.
    """
    overrides = load_pmid_overrides()
    rows: list[dict] = []

    for pmid_int, entry in sorted(overrides.items()):
        label = entry.get("label", "")
        species = entry.get("species", "Homo sapiens (human)")
        ms_samples = entry.get("ms_samples", [])

        for sample in ms_samples:
            cls = sample.get("mhc_class", "")
            if mhc_class and not _mhc_class_matches(cls, mhc_class):
                continue

            condition = sample.get("condition", "")
            if (
                not condition
                or condition.startswith("unperturbed")
                or condition == "—"
                or condition.startswith("NOT ")
            ):
                perturbation = ""
            else:
                perturbation = condition

            n = sample.get("n", "")
            if n == 0:
                continue  # skip "NOT profiled" placeholder rows

            peptides = sample.get("peptides", None)
            estimated = sample.get("peptides_estimated", False)

            rows.append(
                {
                    "species": species,
                    "sample": sample.get("type", ""),
                    "perturbation": perturbation,
                    "pmid": pmid_int,
                    "study": label,
                    "mhc_class": cls,
                    "n_samples": n if n != "" else None,
                    "peptides": peptides,
                    "peptides_estimated": estimated,
                    "notes": sample.get("classification", sample.get("reason", "")),
                }
            )

    return pd.DataFrame(rows)


def generate_species_summary(mhc_class: str | None = None) -> pd.DataFrame:
    """Summarize sample counts by species and MHC class.

    Parameters
    ----------
    mhc_class
        Optional filter (``"I"`` or ``"II"``).

    Returns
    -------
    pd.DataFrame
        Columns: species, mhc_class, n_studies, n_sample_types,
        n_samples, total_peptides.
    """
    df = generate_ms_samples_table(mhc_class=mhc_class)
    if df.empty:
        return pd.DataFrame(
            columns=["species", "mhc_class", "n_studies", "n_sample_types", "n_samples"]
        )

    # Expand I+II into separate rows for grouping
    expanded_rows: list[dict] = []
    for _, row in df.iterrows():
        cls = row["mhc_class"]
        classes = []
        parts = {p.strip() for p in cls.split("+")} if cls else {"unknown"}
        classes = sorted(parts)

        for c in classes:
            expanded_rows.append({**row.to_dict(), "mhc_class": c})

    expanded = pd.DataFrame(expanded_rows)
    if mhc_class:
        expanded = expanded[expanded["mhc_class"] == mhc_class]

    summary = (
        expanded.groupby(["species", "mhc_class"])
        .agg(
            n_studies=("pmid", "nunique"),
            n_sample_types=("sample", "count"),
            n_samples=("n_samples", lambda x: x.dropna().sum()),
            total_peptides=("peptides", lambda x: x.dropna().sum()),
        )
        .reset_index()
    )
    summary["n_samples"] = summary["n_samples"].astype(int)
    summary["total_peptides"] = summary["total_peptides"].astype(int)
    return summary


def validate_mhc_alleles() -> pd.DataFrame:
    """Parse all MHC alleles in pmid_overrides with mhcgnomes.

    Returns
    -------
    pd.DataFrame
        Columns: pmid, study, allele, parsed_name, parsed_type,
        species, valid.
    """
    try:
        from mhcgnomes import parse
    except ImportError:
        return pd.DataFrame(
            columns=["pmid", "study", "allele", "parsed_name", "parsed_type", "species", "valid"]
        )

    overrides = load_pmid_overrides()
    rows: list[dict] = []

    for pmid_int, entry in sorted(overrides.items()):
        label = entry.get("label", "")
        hla_alleles = entry.get("hla_alleles", {})
        if not hla_alleles:
            continue

        allele_strings = _extract_allele_strings(hla_alleles)
        for allele_str in sorted(set(allele_strings)):
            result = parse(allele_str)
            parsed_name = str(result)
            parsed_type = type(result).__name__
            species_name = ""
            if hasattr(result, "species"):
                species_name = result.species.name

            valid = parsed_type not in ("ParseError", "str")

            rows.append(
                {
                    "pmid": pmid_int,
                    "study": label,
                    "allele": allele_str,
                    "parsed_name": parsed_name,
                    "parsed_type": parsed_type,
                    "species": species_name,
                    "valid": valid,
                }
            )

    return pd.DataFrame(rows)


def _mhc_class_matches(sample_class: str, filter_class: str) -> bool:
    """Check if a sample's mhc_class matches a filter.

    ``"I"`` matches ``"I"`` and ``"I+II"`` but NOT ``"II"``.
    ``"II"`` matches ``"II"`` and ``"I+II"`` but NOT ``"I"``.
    """
    if not sample_class:
        return False
    parts = {p.strip() for p in sample_class.split("+")}
    return filter_class in parts


def _scan_single_source(
    source_path: Path,
    source_label: str,
) -> pd.DataFrame:
    """Count unique peptides per PMID x MHC class x species from one CSV.

    Returns DataFrame with columns: source, pmid, mhc_class, mhc_species,
    n_peptides, n_observations.
    """
    from .scanner import _open_csv, _progress, _safe_col

    peptide_sets: dict[tuple, set[str]] = defaultdict(set)
    obs_counts: dict[tuple, int] = defaultdict(int)

    reader, c, p = _open_csv(source_path)
    for row in _progress(reader, p, f"Counting {p.name}"):
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
        mhc_res = _safe_col(row, c["mhc_restriction"])
        species = classify_mhc_species(mhc_res)
        if not species:
            host = _safe_col(row, c["host"])
            species = host if host else "unknown"

        key = (pmid, mhc_cls, species)
        peptide_sets[key].add(pep)
        obs_counts[key] += 1

    rows = []
    for (pmid, mhc_cls, species), peps in sorted(peptide_sets.items()):
        rows.append(
            {
                "source": source_label,
                "pmid": pmid,
                "mhc_class": mhc_cls,
                "mhc_species": species,
                "n_peptides": len(peps),
                "n_observations": obs_counts[(pmid, mhc_cls, species)],
            }
        )
    return pd.DataFrame(rows)


def _resolve_source_paths() -> dict[str, Path]:
    """Resolve registered IEDB/CEDAR paths. Returns {label: path}."""
    from .downloads import get_path

    sources: dict[str, Path] = {}
    for name in ("iedb", "cedar"):
        with contextlib.suppress(KeyError, FileNotFoundError):
            p = get_path(name)
            if p.exists():
                sources[name] = p
    return sources


def count_peptides_by_study(
    source: str | None = None,
    iedb_path: str | Path | None = None,
    cedar_path: str | Path | None = None,
) -> pd.DataFrame:
    """Count unique peptides per PMID x MHC class x species.

    Scans local IEDB/CEDAR CSVs. Fast single-pass per file, no
    source classification overhead.

    Parameters
    ----------
    source
        Which source(s) to scan:

        - ``"iedb"`` — IEDB only
        - ``"cedar"`` — CEDAR only
        - ``"merged"`` — IEDB + CEDAR deduplicated by assay IRI (default)
        - ``"all"`` — returns per-source rows (no dedup) so you can
          compare IEDB vs CEDAR side by side
    iedb_path, cedar_path
        Override paths. If None, resolves from ``hitlist.downloads``.

    Returns
    -------
    pd.DataFrame
        Columns: source, pmid, mhc_class, mhc_species, n_peptides,
        n_observations.
    """
    from .scanner import _open_csv, _progress, _safe_col

    if source is None:
        source = "merged"

    # Resolve paths
    paths: dict[str, Path] = {}
    if iedb_path is not None:
        paths["iedb"] = Path(iedb_path)
    if cedar_path is not None:
        paths["cedar"] = Path(cedar_path)
    if not paths:
        paths = _resolve_source_paths()

    if not paths:
        raise FileNotFoundError(
            "No IEDB/CEDAR data found. Register with: hitlist data register iedb /path/to/file.csv"
        )

    # "all" mode: scan each independently, return concatenated with source labels
    if source == "all":
        dfs = []
        for label, path in sorted(paths.items()):
            dfs.append(_scan_single_source(path, label))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Single source
    if source in ("iedb", "cedar"):
        if source not in paths:
            raise FileNotFoundError(f"'{source}' not registered.")
        return _scan_single_source(paths[source], source)

    # Merged: scan all files, deduplicate by IRI across sources
    peptide_sets: dict[tuple, set[str]] = defaultdict(set)
    obs_counts: dict[tuple, int] = defaultdict(int)
    seen_iris: set[str] = set()

    for _label, source_path in sorted(paths.items()):
        reader, c, p = _open_csv(source_path)
        for row in _progress(reader, p, f"Counting {p.name}"):
            iri = row[c["assay_iri"]] if row else ""
            if iri in seen_iris:
                continue
            seen_iris.add(iri)

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
            mhc_res = _safe_col(row, c["mhc_restriction"])
            species = classify_mhc_species(mhc_res)
            if not species:
                host = _safe_col(row, c["host"])
                species = host if host else "unknown"

            key = (pmid, mhc_cls, species)
            peptide_sets[key].add(pep)
            obs_counts[key] += 1

    rows = []
    for (pmid, mhc_cls, species), peps in sorted(peptide_sets.items()):
        rows.append(
            {
                "source": "iedb+cedar",
                "pmid": pmid,
                "mhc_class": mhc_cls,
                "mhc_species": species,
                "n_peptides": len(peps),
                "n_observations": obs_counts[(pmid, mhc_cls, species)],
            }
        )
    return pd.DataFrame(rows)


def collect_alleles_from_data(
    source: str | None = None,
    iedb_path: str | Path | None = None,
    cedar_path: str | Path | None = None,
) -> pd.DataFrame:
    """Collect all unique MHC restriction strings from IEDB/CEDAR and validate with mhcgnomes.

    Parameters
    ----------
    source
        ``"iedb"``, ``"cedar"``, or ``"merged"`` (default).

    Returns
    -------
    pd.DataFrame
        Columns: allele, n_occurrences, parsed_name, parsed_type,
        species, valid.
    """
    from .scanner import _open_csv, _progress, _safe_col

    if source is None:
        source = "merged"

    paths: dict[str, Path] = {}
    if iedb_path is not None:
        paths["iedb"] = Path(iedb_path)
    if cedar_path is not None:
        paths["cedar"] = Path(cedar_path)
    if not paths:
        paths = _resolve_source_paths()

    allele_counts: dict[str, int] = defaultdict(int)
    seen_iris: set[str] = set()

    scan_paths = [paths[source]] if source in paths else list(paths.values())

    for source_path in scan_paths:
        reader, c, p = _open_csv(source_path)
        for row in _progress(reader, p, f"Alleles {p.name}"):
            if source == "merged":
                iri = row[c["assay_iri"]] if row else ""
                if iri in seen_iris:
                    continue
                seen_iris.add(iri)
            mhc_res = _safe_col(row, c["mhc_restriction"])
            if mhc_res:
                allele_counts[mhc_res] += 1

    try:
        from mhcgnomes import parse
    except ImportError:
        parse = None

    rows = []
    for allele_str, count in sorted(allele_counts.items(), key=lambda x: -x[1]):
        entry = {"allele": allele_str, "n_occurrences": count}
        if parse is not None:
            result = parse(allele_str)
            entry["parsed_name"] = str(result)
            entry["parsed_type"] = type(result).__name__
            entry["species"] = result.species.name if hasattr(result, "species") else ""
            entry["valid"] = entry["parsed_type"] not in ("ParseError", "str")
        rows.append(entry)

    return pd.DataFrame(rows)


def _extract_allele_strings(hla_alleles: dict | list | str) -> list[str]:
    """Recursively extract allele strings from the hla_alleles field."""
    results: list[str] = []
    if isinstance(hla_alleles, str):
        # Could be a description like "51 HLA-I allotypes ..." — skip non-allele text
        if ("HLA-" in hla_alleles and "*" in hla_alleles) or hla_alleles.startswith("HLA-"):
            results.append(hla_alleles)
    elif isinstance(hla_alleles, list):
        for item in hla_alleles:
            results.extend(_extract_allele_strings(item))
    elif isinstance(hla_alleles, dict):
        for value in hla_alleles.values():
            results.extend(_extract_allele_strings(value))
    return results
