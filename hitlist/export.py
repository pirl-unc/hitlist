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

"""Export curated study metadata and unified observations.

Reads ``pmid_overrides.yaml`` and generates per-sample, per-species,
and allele-validation reports from the ``ms_samples`` and ``hla_alleles``
metadata fields.

The main artifact is :func:`generate_observations_table`, which joins
per-peptide observations (from IEDB/CEDAR) with per-sample metadata
(from ``ms_samples``) to produce a single training-ready DataFrame.
"""

from __future__ import annotations

import pandas as pd

from .curation import load_pmid_overrides

# MS acquisition metadata fields.  Each may appear at the PMID level
# (study-wide default) or on individual ``ms_samples`` entries.
# Per-sample values override PMID-level defaults.
_ACQUISITION_FIELDS = (
    "ip_antibody",
    "acquisition_mode",
    "instrument",
    "fragmentation",
    "labeling",
    "search_engine",
    "fdr",
)

# Map specific instrument models → category.  Keys are matched as
# case-insensitive substrings against the ``instrument`` field.
_INSTRUMENT_TYPE_MAP = [
    ("timstof", "timsTOF"),
    ("tims tof", "timsTOF"),
    ("lumos", "Orbitrap"),
    ("fusion", "Orbitrap"),
    ("exploris", "Orbitrap"),
    ("astral", "Orbitrap"),
    ("eclipse", "Orbitrap"),
    ("q exactive", "Orbitrap"),
    ("qe ", "Orbitrap"),
    ("qe+", "Orbitrap"),
    ("orbitrap", "Orbitrap"),
    ("ltq", "Orbitrap"),
    ("velos", "Orbitrap"),
    ("elite", "Orbitrap"),
    ("triple tof", "TOF"),
    ("tripletof", "TOF"),
    ("sciex", "TOF"),
    ("synapt", "TOF"),
    ("xevo", "TOF"),
    ("qtof", "TOF"),
    ("tsq", "QqQ"),
    ("triple quad", "QqQ"),
    ("altis", "QqQ"),
    ("quantiva", "QqQ"),
    ("endura", "QqQ"),
    ("xevo tq", "QqQ"),
    ("maldi", "MALDI"),
    ("fticr", "FTICR"),
]


def _classify_instrument(instrument: str) -> str:
    """Return a broad instrument category from a specific model string."""
    if not instrument:
        return ""
    low = instrument.lower()
    for pattern, category in _INSTRUMENT_TYPE_MAP:
        if pattern in low:
            return category
    return instrument  # return raw value if no match


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
        n_samples, notes, mhc, ip_antibody, acquisition_mode, instrument,
        instrument_type, fragmentation, labeling, search_engine, fdr.
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

            row = {
                "species": species,
                "sample": sample.get("type", ""),
                "perturbation": perturbation,
                "pmid": pmid_int,
                "study": label,
                "mhc_class": cls,
                "n_samples": n if n != "" else None,
                "notes": sample.get("classification", sample.get("reason", "")),
                "mhc": sample.get("mhc") or "",
            }
            for field in _ACQUISITION_FIELDS:
                row[field] = sample.get(field) or entry.get(field) or ""
            row["instrument_type"] = _classify_instrument(row["instrument"])
            rows.append(row)

    return pd.DataFrame(rows)


def generate_observations_table(
    mhc_class: str | None = None,
    species: str | None = None,
    instrument_type: str | None = None,
    acquisition_mode: str | None = None,
    is_mono_allelic: bool | None = None,
    min_allele_resolution: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Join per-peptide observations with per-sample metadata.

    Loads the built ``observations.parquet`` and enriches each row with
    sample-level metadata (instrument, conditions, sample MHC genotype)
    from ``ms_samples`` in the YAML overrides.

    The join logic matches each peptide's ``mhc_restriction`` to the
    ``mhc`` field on ``ms_samples`` entries within the same PMID:

    - Mono-allelic samples: exact allele match
    - Multi-allelic samples: peptide allele appears in sample's genotype
    - Fallback: PMID-only match when no allele-level match is possible

    Parameters
    ----------
    mhc_class
        Filter to ``"I"`` or ``"II"``.
    species
        Filter by MHC species (e.g. ``"Homo sapiens"``).
    instrument_type
        Filter by instrument category (e.g. ``"Orbitrap"``).
    acquisition_mode
        Filter by acquisition mode (e.g. ``"DDA"``).
    is_mono_allelic
        Filter to mono-allelic (True) or multi-allelic (False) samples.
    min_allele_resolution
        Minimum allele resolution (e.g. ``"four_digit"``).
    columns
        Return only these columns.

    Returns
    -------
    pd.DataFrame
        One row per peptide observation, enriched with sample metadata.

    Raises
    ------
    FileNotFoundError
        If the observations table has not been built yet.
    """
    from .observations import load_observations

    # --- Load observations ---
    obs_filters: dict = {}
    if mhc_class:
        obs_filters["mhc_class"] = mhc_class
    if species:
        obs_filters["species"] = species
    obs = load_observations(**obs_filters)

    if min_allele_resolution:
        from .curation import allele_resolution_rank, classify_allele_resolution

        min_rank = allele_resolution_rank(min_allele_resolution)
        obs = obs[
            obs["mhc_restriction"].map(
                lambda a: allele_resolution_rank(classify_allele_resolution(a)) <= min_rank
            )
        ]

    # --- Load sample metadata ---
    samples = generate_ms_samples_table(mhc_class=mhc_class)

    # Build PMID-level metadata lookup (study label, quantification_method)
    overrides = load_pmid_overrides()
    pmid_meta = {}
    for pmid_int, entry in overrides.items():
        pmid_meta[pmid_int] = {
            "quantification_method": entry.get("quantification_method", ""),
        }

    # --- Build sample index for allele-level matching ---
    # For each PMID, build a list of (allele_set, metadata_dict) tuples
    sample_index: dict[int, list[tuple[set[str], dict]]] = {}
    meta_cols = [
        "sample",
        "perturbation",
        "mhc",
        "instrument",
        "instrument_type",
        "acquisition_mode",
        "fragmentation",
        "labeling",
        "ip_antibody",
    ]
    for _, srow in samples.iterrows():
        pmid = int(srow["pmid"])
        mhc_str = srow.get("mhc", "")
        allele_set = set(mhc_str.split()) if mhc_str and mhc_str != "unknown" else set()
        meta = {col: srow.get(col, "") for col in meta_cols}
        sample_index.setdefault(pmid, []).append((allele_set, meta))

    # --- Join ---
    enriched_rows: list[dict] = []
    for _, orow in obs.iterrows():
        record = orow.to_dict()
        pmid = record.get("pmid")
        if pd.isna(pmid):
            enriched_rows.append(record)
            continue
        pmid = int(pmid)

        # Add PMID-level metadata
        pm = pmid_meta.get(pmid, {})
        record["quantification_method"] = pm.get("quantification_method", "")

        # Match to best sample by allele
        allele = record.get("mhc_restriction", "")
        candidates = sample_index.get(pmid, [])

        matched_meta = None
        if candidates:
            # Try allele-level match first
            for allele_set, meta in candidates:
                if allele_set and allele in allele_set:
                    matched_meta = meta
                    break
            # Fallback: if only one sample for this PMID, use it
            if matched_meta is None and len(candidates) == 1:
                matched_meta = candidates[0][1]

        if matched_meta:
            record.update(matched_meta)
        else:
            for col in meta_cols:
                record.setdefault(col, "")

        enriched_rows.append(record)

    result = pd.DataFrame(enriched_rows)

    # --- Post-join filters ---
    if instrument_type:
        result = result[result.get("instrument_type", pd.Series(dtype=str)) == instrument_type]
    if acquisition_mode:
        result = result[result.get("acquisition_mode", pd.Series(dtype=str)) == acquisition_mode]
    if is_mono_allelic is not None:
        col = "is_mono_allelic" if "is_mono_allelic" in result.columns else "src_mono_allelic"
        if col in result.columns:
            result = result[result[col] == is_mono_allelic]

    if columns:
        available = [c for c in columns if c in result.columns]
        result = result[available]

    return result


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
        n_samples.

    For actual peptide counts, use ``count_peptides_by_study()``
    which scans the local IEDB/CEDAR data.
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
        )
        .reset_index()
    )
    summary["n_samples"] = summary["n_samples"].astype(int)
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


def count_peptides_by_study(
    source: str | None = None,
) -> pd.DataFrame:
    """Count unique peptides per PMID x MHC class x species.

    Uses cached index (built on first call, reused if source CSV
    unchanged). See :mod:`hitlist.indexer`.

    Parameters
    ----------
    source
        ``"iedb"``, ``"cedar"``, ``"merged"`` (default), or ``"all"``.

    Returns
    -------
    pd.DataFrame
        Columns: source, pmid, mhc_class, mhc_species, n_peptides,
        n_observations.
    """
    from .indexer import get_index

    study_df, _allele_df = get_index(source=source or "merged")
    return study_df


def collect_alleles_from_data(
    source: str | None = None,
) -> pd.DataFrame:
    """Collect all unique MHC restriction strings and validate with mhcgnomes.

    Uses cached index. See :mod:`hitlist.indexer`.

    Returns
    -------
    pd.DataFrame
        Columns: allele, n_occurrences, parsed_name, parsed_type,
        species, valid.
    """
    from .indexer import get_index, validate_alleles_from_index

    _study_df, allele_df = get_index(source=source or "merged")
    return validate_alleles_from_index(allele_df)


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
