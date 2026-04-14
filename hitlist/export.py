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

from .curation import load_pmid_overrides, normalize_allele, normalize_species

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
        species = normalize_species(entry.get("species", "Homo sapiens (human)"))
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
    mhc_allele: str | list[str] | None = None,
    gene: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
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

    # --- Resolve gene query (may require HGNC lookup) up front ---
    resolved_gene_names: set[str] = set()
    resolved_gene_ids: set[str] = set()
    if gene is not None:
        from .genes import resolve_gene_query

        for q in _to_list(gene):
            spec = resolve_gene_query(q)
            resolved_gene_names |= spec["names"]
            resolved_gene_ids |= spec["ids"]
    if gene_name is not None:
        resolved_gene_names |= set(_to_list(gene_name))
    if gene_id is not None:
        resolved_gene_ids |= set(_to_list(gene_id))

    # --- Load observations with as many filters pushed to parquet as possible ---
    obs_filters: dict = {}
    if mhc_class:
        obs_filters["mhc_class"] = mhc_class
    if species:
        obs_filters["species"] = normalize_species(species)
    if mhc_allele is not None:
        obs_filters["mhc_restriction"] = mhc_allele
    if resolved_gene_names:
        obs_filters["gene_name"] = sorted(resolved_gene_names)
    if resolved_gene_ids:
        obs_filters["gene_id"] = sorted(resolved_gene_ids)
    if serotype is not None:
        obs_filters["serotype"] = _to_list(serotype)
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

    # --- PMID-level metadata (quantification_method) ---
    overrides = load_pmid_overrides()
    pmid_meta_df = pd.DataFrame(
        [
            {"_pmid_int": int(k), "quantification_method": v.get("quantification_method", "")}
            for k, v in overrides.items()
        ]
    )

    # --- Allele-level sample lookup ---
    # Explode each sample's MHC alleles into one row per (pmid, allele)
    # so we can do a vectorized merge instead of row-by-row iteration.
    allele_rows: list[dict] = []
    for _, srow in samples.iterrows():
        pmid = int(srow["pmid"])
        mhc_str = srow.get("mhc", "")
        if mhc_str and mhc_str != "unknown":
            meta = {col: srow.get(col, "") for col in meta_cols}
            meta["_pmid_int"] = pmid
            for allele in mhc_str.split():
                allele_rows.append({**meta, "_allele": allele})

    allele_df = (
        pd.DataFrame(allele_rows)
        if allele_rows
        else pd.DataFrame(columns=["_pmid_int", "_allele", *meta_cols])
    )
    # Keep first match per (pmid, allele) — matches the break-on-first
    # behavior of the previous iterrows loop.
    allele_df = allele_df.drop_duplicates(subset=["_pmid_int", "_allele"], keep="first")

    # --- Single-sample PMID fallback ---
    # When no allele match is found and a PMID has exactly one sample,
    # use that sample's metadata.
    pmid_sample_counts = samples.groupby("pmid").size()
    single_pmids = pmid_sample_counts[pmid_sample_counts == 1].index
    single_df = samples[samples["pmid"].isin(single_pmids)][["pmid", *meta_cols]].copy()
    single_df["_pmid_int"] = single_df["pmid"].astype(int)
    single_df = single_df[["_pmid_int", *meta_cols]].rename(
        columns={c: c + "_fb" for c in meta_cols}
    )

    # --- PMID x class allele pool ---
    # For class-only observations (mhc_restriction = "HLA class I"),
    # collect the union of all alleles across all samples of that class.
    # This gives "one of X, Y, Z" even when we can't pick a specific sample.
    _class_pool: dict[tuple[int, str], str] = {}  # (pmid, mhc_class) → space-joined alleles
    for pmid_int_s, group in samples.groupby("pmid"):
        pmid_int_v = int(pmid_int_s)
        for cls in ("I", "II"):
            alleles: set[str] = set()
            for _, srow in group.iterrows():
                sample_cls = srow.get("mhc_class", "")
                if cls in str(sample_cls).split("+"):
                    mhc_str = srow.get("mhc", "")
                    if mhc_str and mhc_str != "unknown":
                        alleles.update(normalize_allele(a) for a in mhc_str.split())
            # Filter out empty strings from failed normalization
            alleles.discard("")
            if alleles:
                _class_pool[(pmid_int_v, cls)] = " ".join(sorted(alleles))

    # --- Vectorized join ---
    obs["_pmid_int"] = pd.to_numeric(obs["pmid"], errors="coerce")

    # 1) PMID-level metadata
    obs = obs.merge(pmid_meta_df, on="_pmid_int", how="left")
    obs["quantification_method"] = obs["quantification_method"].fillna("")

    # 2) Allele-level match: obs.mhc_restriction == allele_df._allele within same PMID
    obs = obs.merge(
        allele_df.rename(columns={"_allele": "mhc_restriction"}),
        on=["_pmid_int", "mhc_restriction"],
        how="left",
    )

    # 3) Single-PMID fallback for unmatched rows
    obs = obs.merge(single_df, on="_pmid_int", how="left")

    # Coalesce: allele match > single-PMID fallback > ""
    fb_cols = [col + "_fb" for col in meta_cols]
    for col, fb_col in zip(meta_cols, fb_cols):
        obs[col] = obs[col].where(obs[col].notna(), obs[fb_col])
    obs.drop(columns=fb_cols, inplace=True)
    for col in meta_cols:
        obs[col] = obs[col].fillna("")

    # 4) Class-pool fallback: for still-unmatched rows, fill sample_mhc
    #    with the union of all alleles from samples of the same class.
    still_empty = obs["mhc"] == ""
    if still_empty.any():
        obs.loc[still_empty, "mhc"] = [
            _class_pool.get((int(p), c), "") if not pd.isna(p) else ""
            for p, c in zip(
                obs.loc[still_empty, "_pmid_int"],
                obs.loc[still_empty, "mhc_class"],
            )
        ]

    # --- Provenance: how was each row matched? ---
    # Count samples per PMID for context
    _pmid_counts = samples.groupby("pmid").size().rename("matched_sample_count")
    _pmid_counts.index = _pmid_counts.index.astype(int)
    obs = obs.merge(_pmid_counts, left_on="_pmid_int", right_index=True, how="left")
    obs["matched_sample_count"] = obs["matched_sample_count"].fillna(0).astype(int)

    # Determine match type
    _matched_keys = (
        set(zip(allele_df["_pmid_int"], allele_df["_allele"])) if not allele_df.empty else set()
    )
    _single_pmid_set = set(single_df["_pmid_int"]) if not single_df.empty else set()
    _class_pool_keys = set(_class_pool.keys())

    obs["sample_match_type"] = "unmatched"

    if _matched_keys:
        allele_matched = pd.Series(
            [(p, a) in _matched_keys for p, a in zip(obs["_pmid_int"], obs["mhc_restriction"])],
            index=obs.index,
        )
        obs.loc[allele_matched, "sample_match_type"] = "allele_match"
    else:
        allele_matched = pd.Series(False, index=obs.index)

    fallback_matched = ~allele_matched & obs["_pmid_int"].isin(_single_pmid_set)
    obs.loc[fallback_matched, "sample_match_type"] = "single_sample_fallback"

    # Class pool: not allele-matched, not single-sample, but has class pool alleles
    pool_matched = (
        ~allele_matched
        & ~fallback_matched
        & pd.Series(
            [
                (int(p), c) in _class_pool_keys if not pd.isna(p) else False
                for p, c in zip(obs["_pmid_int"], obs["mhc_class"])
            ],
            index=obs.index,
        )
    )
    obs.loc[pool_matched, "sample_match_type"] = "pmid_class_pool"

    # --- Peptide-level allele evidence flag ---
    obs["has_peptide_level_allele"] = obs["mhc_restriction"].astype(str).str.strip().ne("")

    obs.drop(columns=["_pmid_int"], inplace=True)
    result = obs

    # --- Post-join filters ---
    if instrument_type and "instrument_type" in result.columns:
        result = result[result["instrument_type"] == instrument_type]
    if acquisition_mode and "acquisition_mode" in result.columns:
        result = result[result["acquisition_mode"] == acquisition_mode]
    if is_mono_allelic is not None and "is_monoallelic" in result.columns:
        result = result[result["is_monoallelic"] == is_mono_allelic]

    # Rename 'mhc' (from ms_samples join) to 'sample_mhc' to distinguish
    # from the IEDB mhc_restriction field (which may be "HLA class I").
    if "mhc" in result.columns:
        result = result.rename(columns={"mhc": "sample_mhc"})

    if columns:
        available = [c for c in columns if c in result.columns]
        result = result[available]

    return result


def generate_binding_table(
    mhc_class: str | None = None,
    species: str | None = None,
    min_allele_resolution: str | None = None,
    mhc_allele: str | list[str] | None = None,
    gene: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    source: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the binding-assay index with optional filters.

    Returns binding-assay rows (peptide microarray, refolding, MEDi,
    and quantitative-tier measurements).  No sample-metadata join is
    performed — binding assays do not carry MS sample context
    (instrument, acquisition mode, tissue), so the row schema is the
    raw binding index plus gene annotations.

    Filters parallel :func:`generate_observations_table` but omit the
    MS-only options (``--mono-allelic``, ``--instrument-type``,
    ``--acquisition-mode``).
    """
    from .observations import load_binding

    resolved_gene_names: set[str] = set()
    resolved_gene_ids: set[str] = set()
    if gene is not None:
        from .genes import resolve_gene_query

        for q in _to_list(gene):
            spec = resolve_gene_query(q)
            resolved_gene_names |= spec["names"]
            resolved_gene_ids |= spec["ids"]
    if gene_name is not None:
        resolved_gene_names |= set(_to_list(gene_name))
    if gene_id is not None:
        resolved_gene_ids |= set(_to_list(gene_id))

    bind_filters: dict = {}
    if mhc_class:
        bind_filters["mhc_class"] = mhc_class
    if species:
        bind_filters["species"] = normalize_species(species)
    if source:
        bind_filters["source"] = source
    if mhc_allele is not None:
        bind_filters["mhc_restriction"] = mhc_allele
    if resolved_gene_names:
        bind_filters["gene_name"] = sorted(resolved_gene_names)
    if resolved_gene_ids:
        bind_filters["gene_id"] = sorted(resolved_gene_ids)
    if serotype is not None:
        bind_filters["serotype"] = _to_list(serotype)

    df = load_binding(**bind_filters)

    if min_allele_resolution:
        from .curation import allele_resolution_rank, classify_allele_resolution

        min_rank = allele_resolution_rank(min_allele_resolution)
        df = df[
            df["mhc_restriction"].map(
                lambda a: allele_resolution_rank(classify_allele_resolution(a)) <= min_rank
            )
        ]

    if columns:
        available = [c for c in columns if c in df.columns]
        df = df[available]

    return df


def _to_list(v) -> list[str]:
    """Accept a string or list; split a comma-separated string."""
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [s for s in v if s]


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
