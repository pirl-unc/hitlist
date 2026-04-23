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

"""Export curated study metadata and model-facing pMHC tables.

Reads ``pmid_overrides.yaml`` and generates per-sample, per-species,
and allele-validation reports from the ``ms_samples`` and ``hla_alleles``
metadata fields.

The main artifacts are :func:`generate_observations_table`, which joins
per-peptide observations (from IEDB/CEDAR) with per-sample metadata
(from ``ms_samples``), and :func:`generate_training_table`, which
composes the built MS, binding, and peptide-mapping indexes into a
unified export surface for downstream training workflows.
"""

from __future__ import annotations

import pandas as pd

from .curation import (
    allele_to_all_serotypes,
    allele_to_serotype,
    load_pmid_overrides,
    normalize_allele,
    normalize_species,
)

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

_TRAINING_MAPPING_COLUMNS = (
    "protein_id",
    "gene_name",
    "gene_id",
    "position",
    "n_flank",
    "c_flank",
    "proteome",
    "proteome_source",
)

_TRAINING_DEFAULTS = {
    "sample_label": "",
    "perturbation": "",
    "sample_mhc": "",
    "instrument": "",
    "instrument_type": "",
    "acquisition_mode": "",
    "fragmentation": "",
    "labeling": "",
    "ip_antibody": "",
    "quantification_method": "",
    "sample_match_type": "not_applicable",
    "matched_sample_count": 0,
}


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
        Columns: species, sample_label, perturbation, pmid, study_label,
        mhc_class, n_samples, notes, mhc, ip_antibody, acquisition_mode,
        instrument, instrument_type, fragmentation, labeling, search_engine,
        fdr.
    """
    overrides = load_pmid_overrides()
    rows: list[dict] = []

    for pmid_int, entry in sorted(overrides.items()):
        study_label = entry.get("study_label", "")
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

            n = sample.get("n_samples", "")
            if n == 0:
                continue  # skip "NOT profiled" placeholder rows

            row = {
                "species": species,
                "sample_label": sample.get("sample_label", ""),
                "perturbation": perturbation,
                "pmid": pmid_int,
                "study_label": study_label,
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
    source: str | None = None,
    instrument_type: str | None = None,
    acquisition_mode: str | None = None,
    is_mono_allelic: bool | None = None,
    min_allele_resolution: str | None = None,
    mhc_allele: str | list[str] | None = None,
    gene: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
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
    resolved_gene_names, resolved_gene_ids = _resolve_gene_filters(gene, gene_name, gene_id)

    # --- Load observations with as many filters pushed to parquet as possible ---
    obs_filters: dict = {}
    if mhc_class:
        obs_filters["mhc_class"] = mhc_class
    if species:
        obs_filters["species"] = normalize_species(species)
    if source:
        obs_filters["source"] = source
    if mhc_allele is not None:
        obs_filters["mhc_restriction"] = mhc_allele
    if resolved_gene_names:
        obs_filters["gene_name"] = sorted(resolved_gene_names)
    if resolved_gene_ids:
        obs_filters["gene_id"] = sorted(resolved_gene_ids)
    if peptide is not None:
        obs_filters["peptide"] = peptide
    if serotype is not None:
        obs_filters["serotype"] = _to_list(serotype)
    if length_min is not None:
        obs_filters["length_min"] = length_min
    if length_max is not None:
        obs_filters["length_max"] = length_max
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
        "sample_label",
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
        if mhc_str and not _is_class_only_sentinel(mhc_str):
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
                    if mhc_str and not _is_class_only_sentinel(mhc_str):
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
    obs["has_peptide_level_allele"] = _compute_has_peptide_level_allele(
        obs["mhc_restriction"],
        obs["allele_resolution"] if "allele_resolution" in obs.columns else None,
    )

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
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
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

    resolved_gene_names, resolved_gene_ids = _resolve_gene_filters(gene, gene_name, gene_id)

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
    if peptide is not None:
        bind_filters["peptide"] = peptide
    if serotype is not None:
        bind_filters["serotype"] = _to_list(serotype)
    if length_min is not None:
        bind_filters["length_min"] = length_min
    if length_max is not None:
        bind_filters["length_max"] = length_max

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


def _apply_training_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the mixed MS/binding export schema."""
    result = df.copy()

    if "sample_mhc" not in result.columns and "mhc" in result.columns:
        result = result.rename(columns={"mhc": "sample_mhc"})

    for col, default in _TRAINING_DEFAULTS.items():
        if col not in result.columns:
            result[col] = default
            continue
        if isinstance(default, str):
            result[col] = result[col].fillna(default)
        elif isinstance(default, bool):
            result[col] = result[col].astype("boolean").fillna(default).astype(bool)
        else:
            result[col] = result[col].fillna(default)

    if "mhc_restriction" in result.columns:
        result["has_peptide_level_allele"] = _compute_has_peptide_level_allele(
            result["mhc_restriction"],
            result["allele_resolution"] if "allele_resolution" in result.columns else None,
        )
    elif "has_peptide_level_allele" not in result.columns:
        result["has_peptide_level_allele"] = False

    result["matched_sample_count"] = result["matched_sample_count"].astype(int)
    result["has_peptide_level_allele"] = result["has_peptide_level_allele"].astype(bool)

    if "reference_iri" in result.columns:
        ref = result["reference_iri"].fillna("").astype(str)
        result["evidence_row_id"] = [
            f"{kind}:{ref_val}" if ref_val.strip() else f"{kind}:row:{idx}"
            for idx, (kind, ref_val) in enumerate(zip(result["evidence_kind"], ref))
        ]

    return result


def _load_training_mappings_for_peptides(
    peptides: pd.Series | list[str],
    gene_name: list[str] | None = None,
    gene_id: list[str] | None = None,
) -> pd.DataFrame:
    """Load long-form mappings for a selected peptide set.

    Small peptide subsets use parquet push-down filters. Large exports fall
    back to a full mappings scan plus an in-memory peptide filter to avoid
    constructing a huge ``IN (...)`` predicate for pyarrow.
    """
    from .mappings import load_peptide_mappings

    wanted = sorted({str(p).strip() for p in peptides if str(p).strip()})
    columns = ["peptide", *_TRAINING_MAPPING_COLUMNS]
    if not wanted:
        return pd.DataFrame(columns=columns)

    filter_kwargs = {}
    if gene_name:
        filter_kwargs["gene_name"] = gene_name
    if gene_id:
        filter_kwargs["gene_id"] = gene_id

    if len(wanted) <= 10_000:
        return load_peptide_mappings(peptide=wanted, columns=columns, **filter_kwargs)

    mappings = load_peptide_mappings(columns=columns, **filter_kwargs)
    return mappings[mappings["peptide"].isin(set(wanted))]


def _project_training_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """Project the training export, always preserving evidence identity."""
    if columns is None:
        return df
    identity_cols = ["evidence_kind"]
    if "evidence_row_id" in df.columns:
        identity_cols.append("evidence_row_id")
    requested = list(dict.fromkeys([*columns, *identity_cols]))
    available = [c for c in requested if c in df.columns]
    return df[available]


def generate_training_table(
    include_evidence: str = "both",
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    instrument_type: str | None = None,
    acquisition_mode: str | None = None,
    is_mono_allelic: bool | None = None,
    min_allele_resolution: str | None = None,
    mhc_allele: str | list[str] | None = None,
    gene: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
    explode_mappings: bool = False,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Export a unified pMHC training table.

    The canonical stored indexes remain unchanged:

    - ``observations.parquet``: MS/elution observations
    - ``binding.parquet``: in-vitro binding evidence
    - ``peptide_mappings.parquet``: long-form peptide→protein mappings

    This function composes those indexes into one downstream-facing export.
    Compact mode preserves one row per evidence row. ``explode_mappings=True``
    expands the export to one row per ``(evidence row, peptide mapping)``,
    which is suitable for flank-aware model pipelines such as Presto.
    """
    mode = include_evidence.strip().lower()
    if mode not in {"ms", "binding", "both"}:
        raise ValueError("include_evidence must be one of: ms, binding, both")

    resolved_gene_names, resolved_gene_ids = _resolve_gene_filters(gene, gene_name, gene_id)

    shared_kwargs = {
        "mhc_class": mhc_class,
        "species": species,
        "source": source,
        "min_allele_resolution": min_allele_resolution,
        "mhc_allele": mhc_allele,
        "gene": gene,
        "gene_name": gene_name,
        "gene_id": gene_id,
        "peptide": peptide,
        "serotype": serotype,
        "length_min": length_min,
        "length_max": length_max,
    }

    parts: list[pd.DataFrame] = []

    if mode in {"ms", "both"}:
        ms = generate_observations_table(
            instrument_type=instrument_type,
            acquisition_mode=acquisition_mode,
            is_mono_allelic=is_mono_allelic,
            **shared_kwargs,
        ).copy()
        ms["evidence_kind"] = "ms"
        parts.append(ms)

    if mode in {"binding", "both"}:
        binding = generate_binding_table(**shared_kwargs).copy()
        binding["evidence_kind"] = "binding"
        parts.append(binding)

    if not parts:
        result = pd.DataFrame({"evidence_kind": pd.Series(dtype=str)})
    else:
        result = pd.concat(parts, ignore_index=True, sort=False)

    result = _apply_training_defaults(result)

    if explode_mappings:
        mappings = _load_training_mappings_for_peptides(
            result.get("peptide", pd.Series(dtype=str)),
            gene_name=sorted(resolved_gene_names) or None,
            gene_id=sorted(resolved_gene_ids) or None,
        )
        if mappings.empty:
            for col in _TRAINING_MAPPING_COLUMNS:
                if col not in result.columns:
                    result[col] = pd.Series(dtype=object)
        else:
            result = result.merge(mappings, on="peptide", how="left")

    return _project_training_columns(result, columns)


def _to_list(v) -> list[str]:
    """Accept a string or list; split a comma-separated string."""
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [s for s in v if s]


def _resolve_gene_filters(
    gene: str | list[str] | None,
    gene_name: str | list[str] | None,
    gene_id: str | list[str] | None,
) -> tuple[set[str], set[str]]:
    """Resolve user gene filters into exact gene_name / gene_id sets."""
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
    return resolved_gene_names, resolved_gene_ids


def _compute_has_peptide_level_allele(
    mhc_restriction: pd.Series,
    allele_resolution: pd.Series | None = None,
) -> pd.Series:
    """True when a row carries peptide-level allele evidence.

    Class-only sentinels and serological restrictions are not allele-level.
    When resolution metadata is present, it overrides the looser string
    heuristic so downstream exports can trust the flag.
    """
    restriction = mhc_restriction.fillna("").astype(str).str.strip()
    result = (
        restriction.ne("")
        & ~restriction.str.lower().str.startswith(("hla class", "mhc class"))
        & restriction.str.contains(r"\*", regex=True)
    )
    if allele_resolution is not None:
        resolution = allele_resolution.fillna("").astype(str).str.strip()
        result = result & ~resolution.isin({"class_only", "serological"})
    return result.astype(bool)


def _is_class_only_sentinel(mhc_str: str) -> bool:
    """True when the sample's mhc field carries no allele-level info.

    Recognises the legacy ``"unknown"`` sentinel and the class-only
    placeholders ``"HLA class I"`` / ``"HLA class II"`` (introduced
    in 1.7.1 to replace ``"unknown"`` when the IP antibody / mhc_class
    tells us the class but no allele genotype was reported).
    """
    s = mhc_str.strip().lower()
    if s == "unknown":
        return True
    return s.startswith("hla class") or s.startswith("mhc class")


def _join_unique_text(values) -> str:
    return ";".join(sorted({str(v) for v in values if pd.notna(v) and str(v).strip()}))


def _join_unique_numeric(values) -> str:
    nums: set[int] = set()
    for v in values:
        if pd.isna(v) or str(v).strip() == "":
            continue
        try:
            nums.add(int(v))
        except (TypeError, ValueError):
            continue
    return ";".join(str(x) for x in sorted(nums))


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _normalize_serotype_query(raw: str) -> str:
    """Normalise a user serotype query to the ``HLA-*`` display style."""
    q = str(raw).strip()
    if not q:
        return ""
    if q.upper().startswith("HLA-"):
        q = q[4:]
    low = q.lower()
    if low.startswith("bw"):
        q = "Bw" + q[2:]
    elif low.startswith(("dr", "dq", "dp", "dm", "do")):
        q = low[:2].upper() + q[2:]
    else:
        q = q[:1].upper() + q[1:]
    return f"HLA-{q}"


def _serotype_key(raw: str) -> str:
    s = str(raw).strip()
    if s.upper().startswith("HLA-"):
        s = s[4:]
    return s.lower()


def _sample_alleles(sample_mhc: str) -> list[str]:
    if (
        not isinstance(sample_mhc, str)
        or not sample_mhc.strip()
        or _is_class_only_sentinel(sample_mhc)
    ):
        return []
    return [normalize_allele(a) for a in sample_mhc.split() if normalize_allele(a)]


def _source_bucket(row: pd.Series) -> str:
    if _truthy(row.get("src_cancer", False)):
        return "cancer"
    if _truthy(row.get("src_adjacent_to_tumor", False)):
        return "adjacent"
    if (
        _truthy(row.get("src_healthy_tissue", False))
        or _truthy(row.get("src_healthy_thymus", False))
        or _truthy(row.get("src_healthy_reproductive", False))
    ):
        return "healthy"
    return "other"


def _peptide_summary_support_label(row: pd.Series) -> str:
    if _truthy(row.get("mono_exact", False)):
        return "mono_exact"
    if _truthy(row.get("multi_exact", False)):
        return "multi_exact"
    if _truthy(row.get("mono_serotype", False)):
        return "mono_serotype"
    if _truthy(row.get("multi_serotype", False)):
        return "multi_serotype"
    if _truthy(row.get("class_only_sample_allele", False)):
        return "class_only_sample_allele"
    if _truthy(row.get("class_only_sample_serotype", False)):
        return "class_only_sample_serotype"
    if _truthy(row.get("unknown_allele", False)):
        return "unknown_allele"
    return ""


def _best_support_label(summary_row: pd.Series) -> str:
    for col, label in [
        ("n_mono_exact_rows", "mono_exact"),
        ("n_multi_exact_rows", "multi_exact"),
        ("n_mono_serotype_rows", "mono_serotype"),
        ("n_multi_serotype_rows", "multi_serotype"),
        ("n_class_only_sample_allele_rows", "class_only_sample_allele"),
        ("n_class_only_sample_serotype_rows", "class_only_sample_serotype"),
        ("n_unknown_allele_rows", "unknown_allele"),
    ]:
        if int(summary_row.get(col, 0) or 0) > 0:
            return label
    return ""


def generate_ms_peptide_summary_table(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_allele: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    gene: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize per-peptide MS support for one target allele or serotype.

    This is aimed at questions like "which PRAME peptides might be
    presented on A24 in cancers?" It groups relevant observations one
    row per peptide and splits support into exact-allele, same-serotype,
    class-only sample-genotype, and unknown-allele buckets.
    """
    target_alleles = _to_list(mhc_allele) if mhc_allele is not None else []
    target_serotypes = _to_list(serotype) if serotype is not None else []
    if not target_alleles and not target_serotypes:
        raise ValueError("peptide-summary requires exactly one of --mhc-allele or --serotype")
    if target_alleles and target_serotypes:
        raise ValueError("peptide-summary accepts --mhc-allele or --serotype, not both")
    if len(target_alleles) > 1 or len(target_serotypes) > 1:
        raise ValueError("peptide-summary supports only one target allele or serotype at a time")
    scoped_filters = [gene, gene_name, gene_id, peptide]
    if not any(v is not None and _to_list(v) for v in scoped_filters):
        raise ValueError(
            "peptide-summary requires a gene or peptide filter to keep the export scoped"
        )

    target_allele = normalize_allele(target_alleles[0]) if target_alleles else ""
    if target_alleles and not target_allele:
        raise ValueError(f"Could not normalize target allele: {target_alleles[0]!r}")
    target_serotype = (
        _normalize_serotype_query(target_serotypes[0])
        if target_serotypes
        else allele_to_serotype(target_allele)
    )
    target_serotype_key = _serotype_key(target_serotype)
    query_mode = "allele" if target_allele else "serotype"

    df = generate_observations_table(
        mhc_class=mhc_class,
        species=species,
        source=source,
        gene=gene,
        gene_name=gene_name,
        gene_id=gene_id,
        peptide=peptide,
    )
    canonical_columns = [
        "query_mode",
        "target_allele",
        "target_serotype",
        "peptide",
        "n_support_rows",
        "n_pmids",
        "pmids",
        "target_peptide_alleles",
        "sample_match_types",
        "support_labels",
        "best_support",
        "n_mono_exact_rows",
        "n_mono_serotype_rows",
        "n_multi_exact_rows",
        "n_multi_serotype_rows",
        "n_class_only_sample_allele_rows",
        "n_class_only_sample_serotype_rows",
        "n_unknown_allele_rows",
        "n_cancer_rows",
        "n_healthy_rows",
        "n_adjacent_rows",
        "n_other_rows",
    ]
    if df.empty:
        result = pd.DataFrame(columns=canonical_columns)
        return result[columns] if columns else result

    work = df.copy()
    for col, default in {
        "sample_mhc": "",
        "sample_match_type": "",
        "is_monoallelic": False,
        "src_cancer": False,
        "src_adjacent_to_tumor": False,
        "src_healthy_tissue": False,
        "src_healthy_thymus": False,
        "src_healthy_reproductive": False,
    }.items():
        if col not in work.columns:
            work[col] = default
    if "has_peptide_level_allele" not in work.columns:
        work["has_peptide_level_allele"] = _compute_has_peptide_level_allele(
            work["mhc_restriction"],
            work["allele_resolution"] if "allele_resolution" in work.columns else None,
        )
    if "serotypes" not in work.columns:
        work["serotypes"] = work["mhc_restriction"].map(
            lambda a: ";".join(allele_to_all_serotypes(str(a)))
        )

    flags: list[dict] = []
    for _, row in work.iterrows():
        row_allele = normalize_allele(str(row.get("mhc_restriction", "")))
        row_serotype_keys = {
            _serotype_key(s) for s in str(row.get("serotypes", "")).split(";") if s
        }
        row_serotype_keys.update(_serotype_key(s) for s in allele_to_all_serotypes(row_allele))
        sample_allele_list = _sample_alleles(str(row.get("sample_mhc", "")))
        sample_serotype_keys = {
            _serotype_key(s)
            for allele in sample_allele_list
            for s in allele_to_all_serotypes(allele)
        }
        has_peptide_level_allele = _truthy(row.get("has_peptide_level_allele", False))
        mono = _truthy(row.get("is_monoallelic", False))

        exact = bool(target_allele and row_allele == target_allele)
        sero = bool(target_serotype_key and target_serotype_key in row_serotype_keys)
        class_only_sample_allele = (
            not has_peptide_level_allele
            and bool(target_allele)
            and target_allele in sample_allele_list
        )
        class_only_sample_serotype = (
            not has_peptide_level_allele
            and bool(target_serotype_key)
            and target_serotype_key in sample_serotype_keys
            and not class_only_sample_allele
        )
        unknown = (
            not has_peptide_level_allele
            and not class_only_sample_allele
            and not class_only_sample_serotype
            and not sample_allele_list
        )
        relevant = (
            exact or sero or class_only_sample_allele or class_only_sample_serotype or unknown
        )
        source_bucket = _source_bucket(row)
        flags.append(
            {
                "relevant_to_target": relevant,
                "target_peptide_allele": row_allele
                if relevant and has_peptide_level_allele and (exact or sero)
                else "",
                "mono_exact": mono and exact,
                "mono_serotype": mono and sero and not exact,
                "multi_exact": (not mono) and exact,
                "multi_serotype": (not mono) and sero and not exact,
                "class_only_sample_allele": class_only_sample_allele,
                "class_only_sample_serotype": class_only_sample_serotype,
                "unknown_allele": unknown,
                "source_bucket": source_bucket if relevant else "",
            }
        )

    flagged = pd.concat([work.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    flagged = flagged[flagged["relevant_to_target"]].copy()
    if flagged.empty:
        result = pd.DataFrame(columns=canonical_columns)
        return result[columns] if columns else result

    flagged["support_label"] = flagged.apply(_peptide_summary_support_label, axis=1)
    flagged["cancer_row"] = flagged["source_bucket"] == "cancer"
    flagged["healthy_row"] = flagged["source_bucket"] == "healthy"
    flagged["adjacent_row"] = flagged["source_bucket"] == "adjacent"
    flagged["other_row"] = flagged["source_bucket"] == "other"

    result = flagged.groupby("peptide", as_index=False).agg(
        n_support_rows=("peptide", "size"),
        n_pmids=("pmid", lambda x: len({int(v) for v in x if pd.notna(v)})),
        pmids=("pmid", _join_unique_numeric),
        target_peptide_alleles=("target_peptide_allele", _join_unique_text),
        sample_match_types=("sample_match_type", _join_unique_text),
        support_labels=("support_label", _join_unique_text),
        n_mono_exact_rows=("mono_exact", "sum"),
        n_mono_serotype_rows=("mono_serotype", "sum"),
        n_multi_exact_rows=("multi_exact", "sum"),
        n_multi_serotype_rows=("multi_serotype", "sum"),
        n_class_only_sample_allele_rows=("class_only_sample_allele", "sum"),
        n_class_only_sample_serotype_rows=("class_only_sample_serotype", "sum"),
        n_unknown_allele_rows=("unknown_allele", "sum"),
        n_cancer_rows=("cancer_row", "sum"),
        n_healthy_rows=("healthy_row", "sum"),
        n_adjacent_rows=("adjacent_row", "sum"),
        n_other_rows=("other_row", "sum"),
    )
    result.insert(0, "target_serotype", target_serotype)
    result.insert(0, "target_allele", target_allele)
    result.insert(0, "query_mode", query_mode)
    result["best_support"] = result.apply(_best_support_label, axis=1)
    result = result.sort_values(
        [
            "n_mono_exact_rows",
            "n_multi_exact_rows",
            "n_mono_serotype_rows",
            "n_multi_serotype_rows",
            "n_class_only_sample_allele_rows",
            "n_support_rows",
            "peptide",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)
    if columns:
        available = [c for c in columns if c in result.columns]
        return result[available]
    return result


def generate_species_summary(mhc_class: str | None = None) -> pd.DataFrame:
    """Summarize MS-elution data coverage by species and MHC class.

    Reads directly from ``observations.parquet`` — NOT from
    ``pmid_overrides.yaml``. Every species present in the built index
    appears here, including those with zero curated metadata entries.

    Breaking change in v1.15.0 (#117): earlier versions sourced counts
    from ``pmid_overrides.yaml``'s curated ``ms_samples`` entries, which
    undercounted by orders of magnitude on non-human species (mouse had
    2 curated PMIDs vs 388 real; most non-human species were missing
    entirely). The old columns ``n_studies`` / ``n_sample_types`` /
    ``n_samples`` are replaced by ``n_pmids`` / ``n_peptides`` /
    ``n_observations`` sourced from the parquet.

    Parameters
    ----------
    mhc_class
        Optional filter (``"I"``, ``"II"``, or ``"non classical"``).
        Omit for all classes.

    Returns
    -------
    pd.DataFrame
        One row per (species, mhc_class). Columns:

        - ``species``: normalized MHC species from the allele
          (``mhc_species`` column).
        - ``mhc_class``: ``"I"`` / ``"II"`` / ``"non classical"``.
        - ``n_pmids``: unique PMIDs with rows in this (species, class)
          cell.
        - ``n_peptides``: unique peptide sequences.
        - ``n_observations``: total row count (one per assay IRI).
    """
    from .observations import is_built, load_observations

    if not is_built():
        return pd.DataFrame(
            columns=["species", "mhc_class", "n_pmids", "n_peptides", "n_observations"]
        )

    obs = load_observations(
        mhc_class=mhc_class,
        columns=["peptide", "mhc_species", "mhc_class", "pmid"],
    )
    if obs.empty:
        return pd.DataFrame(
            columns=["species", "mhc_class", "n_pmids", "n_peptides", "n_observations"]
        )

    # Drop rows with an empty mhc_species — these are rows where mhcgnomes
    # couldn't resolve the MHC restriction to a species. They'd clutter
    # the summary with an empty-string row. Count is usually small.
    obs = obs[obs["mhc_species"].notna() & (obs["mhc_species"] != "")]

    summary = (
        obs.groupby(["mhc_species", "mhc_class"], dropna=False)
        .agg(
            n_pmids=("pmid", "nunique"),
            n_peptides=("peptide", "nunique"),
            n_observations=("peptide", "size"),
        )
        .reset_index()
        .rename(columns={"mhc_species": "species"})
        .sort_values(["species", "mhc_class"])
        .reset_index(drop=True)
    )
    return summary


def validate_mhc_alleles() -> pd.DataFrame:
    """Parse all MHC alleles in pmid_overrides with mhcgnomes.

    Returns
    -------
    pd.DataFrame
        Columns: pmid, study_label, allele, parsed_name, parsed_type,
        species, valid.
    """
    try:
        from mhcgnomes import parse
    except ImportError:
        return pd.DataFrame(
            columns=[
                "pmid",
                "study_label",
                "allele",
                "parsed_name",
                "parsed_type",
                "species",
                "valid",
            ]
        )

    overrides = load_pmid_overrides()
    rows: list[dict] = []

    for pmid_int, entry in sorted(overrides.items()):
        study_label = entry.get("study_label", "")
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
                    "study_label": study_label,
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
