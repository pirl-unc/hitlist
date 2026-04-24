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

The main MS artifact is :func:`generate_ms_observations_table`, with
:func:`generate_observations_table` retained as a backward-compatible alias.
That join
per-peptide observations (from IEDB/CEDAR) with per-sample metadata
(from ``ms_samples``), and :func:`generate_training_table`, which
composes the built MS, binding, and peptide-mapping indexes into a
unified export surface for downstream training workflows.
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


def generate_ms_observations_table(
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
    """Alias for :func:`generate_observations_table` with explicit MS naming."""
    return generate_observations_table(
        mhc_class=mhc_class,
        species=species,
        instrument_type=instrument_type,
        acquisition_mode=acquisition_mode,
        is_mono_allelic=is_mono_allelic,
        min_allele_resolution=min_allele_resolution,
        mhc_allele=mhc_allele,
        gene=gene,
        gene_name=gene_name,
        gene_id=gene_id,
        serotype=serotype,
        columns=columns,
    )


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


def generate_sample_expression_table(
    mhc_class: str | None = None,
    cancer_type_backend=None,
) -> pd.DataFrame:
    """Per-sample expression-anchor resolution table.

    For every ``ms_samples`` entry (as emitted by
    :func:`generate_ms_samples_table`), resolves an expression anchor via
    :func:`hitlist.line_expression.resolve_sample_expression_anchor` and
    returns the flat provenance record downstream callers need in order
    to distinguish "exact JY RNA" from "generic EBV-LCL stand-in" from
    "melanoma cohort surrogate" (issue pirl-unc/hitlist#140).

    Parameters
    ----------
    mhc_class
        Forwarded to :func:`generate_ms_samples_table`.
    cancer_type_backend
        Optional callable for tier-4 cancer-type fallback (pirlygenes).

    Returns
    -------
    pd.DataFrame
        One row per sample with columns: ``sample_label``, ``pmid``,
        ``study_label``, ``mhc_class``, ``expression_backend``,
        ``expression_key``, ``expression_match_tier``,
        ``expression_parent_key``, ``expression_source_ids`` (semicolon-
        joined), ``expression_reason``, ``expression_matched_alias``.
    """
    from .line_expression import resolve_sample_expression_anchor

    samples = generate_ms_samples_table(mhc_class=mhc_class)
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "sample_label",
                "pmid",
                "study_label",
                "mhc_class",
                "expression_backend",
                "expression_key",
                "expression_match_tier",
                "expression_parent_key",
                "expression_source_ids",
                "expression_reason",
                "expression_matched_alias",
            ]
        )

    rows: list[dict] = []
    for _, s in samples.iterrows():
        anchor = resolve_sample_expression_anchor(
            str(s.get("sample_label") or ""),
            pmid=int(s["pmid"]) if pd.notna(s.get("pmid")) else None,
            study_label=str(s.get("study_label") or "") or None,
            cancer_type_backend=cancer_type_backend,
        )
        rows.append(
            {
                "sample_label": s.get("sample_label", ""),
                "pmid": int(s["pmid"]) if pd.notna(s.get("pmid")) else pd.NA,
                "study_label": s.get("study_label", ""),
                "mhc_class": s.get("mhc_class", ""),
                "expression_backend": anchor.expression_backend,
                "expression_key": anchor.expression_key,
                "expression_match_tier": anchor.expression_match_tier,
                "expression_parent_key": anchor.expression_parent_key or "",
                "expression_source_ids": ";".join(anchor.source_ids),
                "expression_reason": anchor.reason,
                "expression_matched_alias": anchor.matched_alias or "",
            }
        )
    result = pd.DataFrame(rows)
    result["pmid"] = result["pmid"].astype("Int64")
    return result


_PEPTIDE_ORIGIN_COLUMNS = (
    "peptide_origin_gene",
    "peptide_origin_gene_id",
    "peptide_origin_tpm",
    "peptide_origin_log2_tpm",
    "peptide_origin_dominant_transcript",
    "peptide_origin_n_supporting_transcripts",
    "peptide_origin_resolution",
)

_EXPRESSION_ANCHOR_COLUMNS = (
    "expression_backend",
    "expression_key",
    "expression_match_tier",
    "expression_parent_key",
)


def _build_transcript_lookup(gene_names: set[str], release: int):
    """Return a ``gene_name -> [(transcript_id, protein_seq)]`` closure.

    Built by iterating pyensembl's protein-coding transcripts for each
    requested gene.  Returns ``None`` if pyensembl is unavailable (the
    caller then falls back to the gene-only origin path).
    """
    try:
        from pyensembl import EnsemblRelease
    except Exception:
        return None

    try:
        ensembl = EnsemblRelease(release)
    except Exception:
        return None

    cache: dict[str, list[tuple[str, str]]] = {}

    def _lookup(gene_name: str) -> list[tuple[str, str]]:
        if gene_name in cache:
            return cache[gene_name]
        records: list[tuple[str, str]] = []
        try:
            genes = ensembl.genes_by_name(gene_name)
        except Exception:
            cache[gene_name] = records
            return records
        for gene in genes:
            for t in gene.transcripts:
                if getattr(t, "biotype", "") != "protein_coding":
                    continue
                try:
                    seq = t.protein_sequence
                except Exception:
                    seq = None
                if not seq:
                    continue
                records.append((str(t.id), str(seq)))
        cache[gene_name] = records
        return records

    # Warm the cache lazily on first access; no eager population.
    _ = gene_names  # accepted for signature symmetry; used for future prefetch
    return _lookup


def _attach_peptide_origin(
    df: pd.DataFrame,
    cancer_type_backend=None,
    proteome_release: int = 112,
) -> pd.DataFrame:
    """Add per-(peptide, sample) expression anchor + peptide_origin columns."""
    from .line_expression import (
        compute_peptide_origin,
        load_line_expression,
        resolve_sample_expression_anchor,
    )
    from .mappings import load_peptide_mappings

    if df.empty or "peptide" not in df.columns:
        for col in (*_EXPRESSION_ANCHOR_COLUMNS, *_PEPTIDE_ORIGIN_COLUMNS):
            if col not in df.columns:
                df[col] = pd.NA
        return df

    # ------------------------------------------------------------------
    # 1. Resolve the anchor once per unique (sample_label, pmid).
    # ------------------------------------------------------------------
    sample_cols = [c for c in ("sample_label", "pmid", "study_label") if c in df.columns]
    if not sample_cols:
        for col in (*_EXPRESSION_ANCHOR_COLUMNS, *_PEPTIDE_ORIGIN_COLUMNS):
            df[col] = pd.NA
        return df

    unique_samples = df[sample_cols].drop_duplicates().reset_index(drop=True)
    anchor_records: list[dict] = []
    for _, s in unique_samples.iterrows():
        anchor = resolve_sample_expression_anchor(
            str(s.get("sample_label") or ""),
            pmid=int(s["pmid"]) if "pmid" in s and pd.notna(s.get("pmid")) else None,
            study_label=str(s.get("study_label") or "") or None,
            cancer_type_backend=cancer_type_backend,
        )
        rec = {c: s.get(c, "") for c in sample_cols}
        rec.update(
            expression_backend=anchor.expression_backend,
            expression_key=anchor.expression_key,
            expression_match_tier=anchor.expression_match_tier,
            expression_parent_key=anchor.expression_parent_key or "",
        )
        anchor_records.append(rec)
    anchor_df = pd.DataFrame(anchor_records)
    if "pmid" in anchor_df.columns:
        anchor_df["pmid"] = anchor_df["pmid"].astype("Int64")

    df = df.copy()
    if "pmid" in df.columns:
        df["pmid"] = df["pmid"].astype("Int64")
    df = df.merge(anchor_df, on=sample_cols, how="left")

    # ------------------------------------------------------------------
    # 2. Preload TPM tables for every distinct line_key.
    # ------------------------------------------------------------------
    line_keys = sorted({str(k) for k in df["expression_key"].dropna().unique() if str(k)})
    tpm_by_line: dict[str, pd.DataFrame] = {}
    for lk in line_keys:
        sub = load_line_expression(line_key=lk)
        tpm_by_line[lk] = sub

    # ------------------------------------------------------------------
    # 3. Load mappings for every peptide we need to score (once).
    # ------------------------------------------------------------------
    wanted_peptides = sorted({str(p) for p in df["peptide"].dropna().unique() if str(p)})
    if wanted_peptides:
        mappings = load_peptide_mappings(
            peptide=wanted_peptides,
            columns=["peptide", "gene_name", "gene_id", "protein_id"],
        )
    else:
        mappings = pd.DataFrame(columns=["peptide", "gene_name", "gene_id", "protein_id"])
    candidates_by_peptide: dict[str, list[dict]] = {}
    if not mappings.empty:
        for pep, group in mappings.groupby("peptide"):
            unique_rows = group[["gene_name", "gene_id"]].drop_duplicates()
            candidates_by_peptide[str(pep)] = [
                {
                    "gene_name": str(r.get("gene_name") or ""),
                    "gene_id": str(r.get("gene_id") or ""),
                }
                for _, r in unique_rows.iterrows()
                if r.get("gene_name")
            ]

    # ------------------------------------------------------------------
    # 4. Set up an optional transcript lookup once.
    # ------------------------------------------------------------------
    has_any_transcript = any(
        (not t.empty) and "granularity" in t.columns and (t["granularity"] == "transcript").any()
        for t in tpm_by_line.values()
    )
    transcript_lookup = None
    if has_any_transcript:
        gene_universe: set[str] = set()
        for cands in candidates_by_peptide.values():
            for c in cands:
                if c["gene_name"]:
                    gene_universe.add(c["gene_name"])
        transcript_lookup = _build_transcript_lookup(gene_universe, proteome_release)

    # ------------------------------------------------------------------
    # 5. Compute peptide-origin per UNIQUE (peptide, expression_key) pair,
    #    then merge the small result table back onto the full frame.
    #    This keeps the per-row cost in pandas merge rather than a Python
    #    iterrows loop — ~N-unique-pairs work instead of ~N-rows.
    # ------------------------------------------------------------------
    pair_cols = df[["peptide", "expression_key"]].astype(str).fillna("")
    unique_pairs = pair_cols.drop_duplicates().reset_index(drop=True)

    empty_origin = {
        "peptide_origin_gene": "",
        "peptide_origin_gene_id": "",
        "peptide_origin_tpm": float("nan"),
        "peptide_origin_log2_tpm": float("nan"),
        "peptide_origin_dominant_transcript": "",
        "peptide_origin_n_supporting_transcripts": 0,
        "peptide_origin_resolution": "no_anchor",
    }

    origin_rows: list[dict] = []
    for _, pair in unique_pairs.iterrows():
        peptide = pair["peptide"]
        line_key = pair["expression_key"]
        if not line_key or line_key not in tpm_by_line or tpm_by_line[line_key].empty:
            scored = dict(empty_origin)
        else:
            scored = compute_peptide_origin(
                peptide=peptide,
                candidate_genes=candidates_by_peptide.get(peptide, []),
                line_expression_df=tpm_by_line[line_key],
                transcript_lookup=transcript_lookup,
            )
        origin_rows.append({"peptide": peptide, "expression_key": line_key, **scored})

    origin_df = pd.DataFrame(
        origin_rows, columns=["peptide", "expression_key", *_PEPTIDE_ORIGIN_COLUMNS]
    )

    # Ensure merge-key dtypes line up (both sides are plain strings now).
    df["peptide"] = df["peptide"].astype(str).fillna("")
    df["expression_key"] = df["expression_key"].astype(str).fillna("")
    df = df.merge(origin_df, on=["peptide", "expression_key"], how="left")

    # Fill the fallback for any (peptide, key) the merge didn't cover
    # — shouldn't happen in practice, but keeps the contract stable.
    for col, default in empty_origin.items():
        if col in df.columns:
            if isinstance(default, str):
                df[col] = df[col].fillna(default)
            elif isinstance(default, float):
                df[col] = df[col].astype(float)
            elif isinstance(default, int):
                df[col] = df[col].fillna(default).astype(int)

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
    with_peptide_origin: bool = False,
    cancer_type_backend=None,
    proteome_release: int = 112,
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

    When ``with_peptide_origin=True`` every MS row is additionally
    enriched with a per-sample expression anchor and a peptide-origin
    call (the most-likely source gene, its TPM, and the transcript that
    dominates when the backend is transcript-level).  Provenance columns
    (``expression_backend``, ``expression_key``, ``expression_match_tier``,
    ``expression_parent_key``) are preserved so consumers can distinguish
    exact-line evidence from class / tissue / cancer-type surrogates —
    see issue pirl-unc/hitlist#140.

    .. note::
       Transcript-isoform-aware scoring needs pyensembl translations.
       When transcript-level TPM is present for any resolved sample,
       :func:`_build_transcript_lookup` instantiates ``EnsemblRelease``,
       which will trigger a pyensembl cache download (~GB) on first use
       if the requested ``proteome_release`` has not already been
       materialized. Set ``proteome_release`` to a release you have
       already built observations against to avoid a surprise download.
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

    if with_peptide_origin:
        result = _attach_peptide_origin(
            result,
            cancer_type_backend=cancer_type_backend,
            proteome_release=proteome_release,
        )

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
