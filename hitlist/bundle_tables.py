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

"""Compact bundle dimension tables, modality narrowing, and canonical allele-set helpers (#990, #992).

Provides:

- ``samples.parquet`` — lightweight (<1 MB) cohort/sample index keyed by
  deterministic ``sample_id``, carrying repeated study/sample metadata and
  precomputed peptide/flank counts.
- ``alleles.parquet`` — lightweight (<100 KB) per-allele dimension table keyed
  by canonical ``mhcgnomes`` allele string with class, species, gene,
  resolution, ``mhcseqs`` accessioned status, and modality peptide/study counts.
- Canonical ``mhc_allele_set`` serialization contract: sorted, deduplicated
  canonical ``mhcgnomes`` strings joined by ``";"`` (chosen over Parquet
  ``list<string>`` to preserve pandas hashability in ``groupby`` /
  ``drop_duplicates`` and Parquet row-group min/max predicate pushdown).
- Modality-specific column narrowing and row-group sorting constants used by
  :mod:`hitlist.builder`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import bundle_path, dataset_path, dataset_snapshot
from .downloads import data_dir
from .parquet_io import atomic_write_parquet

#: Single documented canonical delimiter for ``mhc_allele_set`` across all bundle tables (#992).
ALLELE_SET_DELIMITER: str = ";"

#: Row-group sort key for fast Parquet predicate pushdown (#990).
MODALITY_SORT_COLUMNS: tuple[str, ...] = (
    "mhc_class",
    "mhc_species",
    "mhc_allele_set",
    "peptide",
)

#: Default Parquet row-group size for sorted modality tables (#990).
MODALITY_ROW_GROUP_SIZE: int = 250_000

#: Binding-only columns elided from ``observations.parquet`` when empty/NA/False (#990).
BINDING_ONLY_COLUMNS: tuple[str, ...] = (
    "quantitative_measurement",
    "quantitative_value",
    "measurement_units",
    "measurement_inequality",
    "is_binding_assay",
)

#: MS-only columns elided from ``binding.parquet`` when empty/NA/False (#990).
#: Note: ``is_binding_assay`` is deliberately retained on ``binding.parquet``
#: because ``hitlist.assays.assay_expression`` reads it for legacy heuristics.
MS_ONLY_COLUMNS: tuple[str, ...] = (
    "ms_outcome_status",
    "is_potential_contaminant",
    "attributed_sample_label",
    "peptide_modifications_raw",
    "has_ptm_raw",
    "peptidoform_review_status",
    "peptidoform_evidence",
    "supplementary_file",
    "antigen_processing_comments",
    "is_ms_observation",
    "src_adjacent_to_tumor",
    "src_activated_apc",
    "src_healthy_tissue",
    "src_healthy_thymus",
    "src_healthy_reproductive",
    "src_healthy_reproductive_female",
    "src_healthy_reproductive_male",
)

#: Sample/cohort grouping columns used to construct deterministic ``sample_id`` values.
SAMPLE_KEY_COLUMNS: tuple[str, ...] = (
    "pmid",
    "source",
    "mhc_class",
    "mhc_species",
    "mhc_allele_set",
    "cell_line_name",
    "cell_type",
    "source_tissue",
    "disease",
    "attributed_sample_label",
)

SRC_FLAG_COLUMNS: tuple[str, ...] = (
    "src_cancer",
    "src_adjacent_to_tumor",
    "src_activated_apc",
    "src_healthy_tissue",
    "src_healthy_thymus",
    "src_healthy_reproductive",
    "src_healthy_reproductive_female",
    "src_healthy_reproductive_male",
    "src_cell_line",
    "src_ebv_lcl",
    "src_ex_vivo",
)

_CANONICAL_PEPTIDE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_SPLIT_ALLELE_RE = re.compile(r"[;,|]")


def parse_allele_set(raw: object) -> tuple[str, ...]:
    """Parse a canonical ``;``-delimited ``mhc_allele_set`` (or list/tuple) into sorted unique tokens."""
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple, set, frozenset, np.ndarray)):
        tokens = [str(item).strip() for item in raw if item is not None and str(item).strip()]
        return tuple(sorted(set(tokens)))
    try:
        if pd.isna(raw):
            return ()
    except (TypeError, ValueError):
        pass
    text = str(raw).strip()
    if not text:
        return ()
    tokens = [tok.strip() for tok in _SPLIT_ALLELE_RE.split(text) if tok.strip()]
    return tuple(sorted(set(tokens)))


def format_allele_set(alleles: Iterable[str]) -> str:
    """Serialize an iterable of allele names into the canonical sorted ``;``-delimited string."""
    cleaned = sorted({str(a).strip() for a in alleles if a is not None and str(a).strip()})
    return ALLELE_SET_DELIMITER.join(cleaned)


def _column_is_vacuous(series: pd.Series) -> bool:
    """True if every value in ``series`` is NA, False (for bool), or empty whitespace string."""
    if series.empty:
        return True
    if series.isna().all():
        return True
    if pd.api.types.is_bool_dtype(series):
        return bool((~series.fillna(False)).all())
    if (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
        or series.dtype == object
    ):
        return bool(series.astype("string").fillna("").str.strip().eq("").all())
    return False


def narrow_modality_frame(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Drop cross-modality columns that are vacuous for ``modality`` (``"ms"`` or ``"binding"``) (#990).

    Never drops a column that contains non-empty / non-False values, so
    unexpected real evidence is never silently lost.
    """
    if modality == "ms":
        candidates = BINDING_ONLY_COLUMNS
    elif modality == "binding":
        candidates = MS_ONLY_COLUMNS
    else:
        raise ValueError(f"Unsupported modality for narrowing: {modality!r}")
    to_drop = [col for col in candidates if col in df.columns and _column_is_vacuous(df[col])]
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def derive_sample_id(
    pmid: object,
    source: str = "",
    mhc_class: str = "",
    mhc_species: str = "",
    mhc_allele_set: str = "",
    cell_line_name: str = "",
    cell_type: str = "",
    source_tissue: str = "",
    disease: str = "",
    attributed_sample_label: str = "",
) -> str:
    """Compute a deterministic cohort/sample ID from the canonical sample key tuple."""
    pmid_str = ""
    if pmid is not None:
        try:
            if not pd.isna(pmid):
                pmid_str = str(int(pmid))
        except (TypeError, ValueError):
            pmid_str = str(pmid).strip()
    parts = (
        pmid_str,
        str(source or "").strip(),
        str(mhc_class or "").strip(),
        str(mhc_species or "").strip(),
        str(mhc_allele_set or "").strip(),
        str(cell_line_name or "").strip(),
        str(cell_type or "").strip(),
        str(source_tissue or "").strip(),
        str(disease or "").strip(),
        str(attributed_sample_label or "").strip(),
    )
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    prefix = f"pmid{pmid_str}" if pmid_str else "unpub"
    return f"{prefix}:{digest}"


def _str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[col].astype("string").fillna("").str.strip()


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].fillna(False).astype(bool)


def _first_non_empty(series: pd.Series) -> str:
    for val in series:
        if val:
            return str(val)
    return ""


def _resolve_row_proteome_species(df: pd.DataFrame) -> pd.Series:
    """Resolve host/source proteome species per row without mutating ``df``."""
    from .curation import normalize_species

    raw = pd.Series("", index=df.index, dtype="string")
    for col in ("source_species", "source_organism", "species", "mhc_species"):
        if col not in df.columns:
            continue
        cand = df[col].astype("string").fillna("").str.strip()
        cand = cand.mask(cand.str.lower().isin({"unknown", "unidentified"}), "")
        raw = raw.where(raw.ne(""), cand)
    uniques = raw.unique()
    norm_map = {
        val: (normalize_species(val) or val) if val else "" for val in uniques if pd.notna(val)
    }
    return raw.map(norm_map).fillna("").astype("string")


def build_samples_table(
    obs: pd.DataFrame,
    *,
    primary_flanks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the compact cohort/sample index table (``samples.parquet``) from MS observations (#990).

    One row per unique :data:`SAMPLE_KEY_COLUMNS` tuple, carrying study/sample
    metadata, ``src_*`` flags, resolved ``proteome_id``, and explicit-unit
    peptide and flank counts.
    """
    base_cols = [
        "sample_id",
        "pmid",
        "reference_title",
        "source",
        "mhc_species",
        "mhc_class",
        "mhc_allele_set",
        "is_monoallelic",
        "cell_line_name",
        "cell_type",
        "source_tissue",
        "disease",
        "attributed_sample_label",
        *SRC_FLAG_COLUMNS,
        "proteome_id",
        "n_rows",
        "n_unique_peptides",
        "n_clean_peptides",
        "n_mapped_peptides",
        "n_flanked_peptides",
        "flank_coverage",
    ]
    if obs.empty:
        return pd.DataFrame(columns=base_cols)

    work = pd.DataFrame(index=obs.index)
    if "pmid" in obs.columns:
        work["pmid"] = pd.to_numeric(obs["pmid"], errors="coerce").astype("Int64")
    else:
        work["pmid"] = pd.Series(pd.NA, index=obs.index, dtype="Int64")
    work["_pmid_key"] = work["pmid"].astype("string").fillna("")

    for col in SAMPLE_KEY_COLUMNS[1:]:
        work[col] = _str_series(obs, col)
    work["reference_title"] = _str_series(obs, "reference_title")
    work["is_monoallelic"] = _bool_series(obs, "is_monoallelic")
    work["peptide"] = _str_series(obs, "peptide")
    work["proteome_id"] = _resolve_row_proteome_species(obs)

    clean_mask = (
        work["peptide"].str.match(_CANONICAL_PEPTIDE_RE, na=False)
        & ~_bool_series(obs, "has_ptm")
        & ~_bool_series(obs, "is_non_peptide_ligand")
        & ~_bool_series(obs, "is_potential_contaminant")
    )
    work["_clean_peptide"] = work["peptide"].where(clean_mask, pd.NA)

    for flag in SRC_FLAG_COLUMNS:
        work[flag] = _bool_series(obs, flag)

    has_flanks = primary_flanks is not None and not primary_flanks.empty
    if has_flanks:
        pf = primary_flanks.copy()
        prot_col = "proteome_id" if "proteome_id" in pf.columns else "proteome"
        pf_keys = pd.DataFrame(
            {
                "peptide": pf["peptide"].astype("string").fillna("").str.strip(),
                "proteome_id": pf[prot_col].astype("string").fillna("").str.strip(),
                "_has_flank": (
                    pf.get("n_flank", pd.Series("", index=pf.index))
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .ne("")
                    | pf.get("c_flank", pd.Series("", index=pf.index))
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .ne("")
                ),
            }
        ).drop_duplicates(subset=["peptide", "proteome_id"], keep="first")
        pf_keys["_is_mapped"] = True
        merged = work[["peptide", "proteome_id"]].merge(
            pf_keys, on=["peptide", "proteome_id"], how="left"
        )
        work["_mapped_peptide"] = work["peptide"].where(
            merged["_is_mapped"].fillna(False).to_numpy(dtype=bool), pd.NA
        )
        work["_flanked_peptide"] = work["peptide"].where(
            merged["_has_flank"].fillna(False).to_numpy(dtype=bool), pd.NA
        )

    group_cols = ["_pmid_key", *SAMPLE_KEY_COLUMNS[1:]]
    agg_spec: dict[str, tuple[str, object]] = {
        "pmid": ("pmid", "first"),
        "reference_title": ("reference_title", _first_non_empty),
        "is_monoallelic": ("is_monoallelic", "any"),
        **{flag: (flag, "any") for flag in SRC_FLAG_COLUMNS},
        "proteome_id": ("proteome_id", _first_non_empty),
        "n_rows": ("peptide", "size"),
        "n_unique_peptides": ("peptide", "nunique"),
        "n_clean_peptides": ("_clean_peptide", "nunique"),
    }
    if has_flanks:
        agg_spec["n_mapped_peptides"] = ("_mapped_peptide", "nunique")
        agg_spec["n_flanked_peptides"] = ("_flanked_peptide", "nunique")

    grouped = work.groupby(group_cols, sort=False, observed=True).agg(**agg_spec).reset_index()
    if not has_flanks:
        grouped["n_mapped_peptides"] = np.int64(0)
        grouped["n_flanked_peptides"] = np.int64(0)
        grouped["flank_coverage"] = np.nan
    else:
        denom = grouped["n_unique_peptides"].to_numpy(dtype=np.float64)
        numer = grouped["n_flanked_peptides"].to_numpy(dtype=np.float64)
        grouped["flank_coverage"] = np.where(denom > 0, numer / denom, np.nan)

    grouped["sample_id"] = [
        derive_sample_id(*vals)
        for vals in grouped[list(SAMPLE_KEY_COLUMNS)].itertuples(index=False, name=None)
    ]
    grouped["pmid"] = pd.to_numeric(grouped["pmid"], errors="coerce").astype("Int64")
    for int_col in (
        "n_rows",
        "n_unique_peptides",
        "n_clean_peptides",
        "n_mapped_peptides",
        "n_flanked_peptides",
    ):
        grouped[int_col] = grouped[int_col].astype("int64")

    result = grouped[base_cols].sort_values(
        ["mhc_class", "mhc_species", "pmid", "sample_id"],
        kind="mergesort",
        na_position="last",
    )
    return result.reset_index(drop=True)


@lru_cache(maxsize=1)
def _mhcseqs_known_alleles() -> frozenset[str]:
    """Return the set of allele names with sequence records in bundled ``mhcseqs``."""
    try:
        import mhcseqs

        rows = mhcseqs.load_sequences_dict()
    except Exception:
        return frozenset()
    known: set[str] = set()
    for row in rows:
        if not row.get("sequence"):
            continue
        for key in ("two_field_allele", "representative_allele"):
            val = row.get(key)
            if val:
                known.add(str(val))
    return frozenset(known)


@lru_cache(maxsize=8192)
def _parse_allele_metadata(token: str) -> dict[str, object]:
    """Canonicalize an allele token and compute metadata for ``alleles.parquet`` (#992)."""
    import mhcgnomes
    import mhcseqs

    from .curation import classify_allele_resolution, mhc_class_of

    parsed = mhcgnomes.parse(token, raise_on_error=False)
    canonical = parsed.to_string() if parsed is not None else token
    species_name = ""
    if parsed is not None and getattr(parsed, "species", None) is not None:
        species_name = str(parsed.species.name or "")

    raw_class = mhc_class_of(canonical) or ""
    if not raw_class and parsed is not None:
        mc = getattr(parsed, "mhc_class", None) or ""
        if mc.startswith("II"):
            raw_class = "II"
        elif mc in {"I", "Ia"}:
            raw_class = "I"
        elif mc in {"Ib", "Ic", "Id"}:
            raw_class = "non classical"

    gene_name = ""
    if isinstance(parsed, mhcgnomes.Pair):
        alpha_gene = getattr(parsed.alpha, "gene_name", "") or ""
        beta_gene = getattr(parsed.beta, "gene_name", "") or ""
        gene_name = f"{alpha_gene}/{beta_gene}".strip("/")
    elif parsed is not None:
        gene_name = str(getattr(parsed, "gene_name", "") or "")

    resolution = classify_allele_resolution(canonical) or "unknown"

    known = _mhcseqs_known_alleles()

    def _chain_accessioned(obj) -> bool:
        if isinstance(obj, mhcgnomes.Pair):
            return _chain_accessioned(obj.alpha) and _chain_accessioned(obj.beta)
        if not isinstance(obj, mhcgnomes.Allele):
            return False
        s = obj.to_string()
        if s in known:
            return True
        try:
            norm = mhcseqs.normalize_allele_name(s)
        except Exception:
            norm = ""
        return bool(norm and norm in known)

    is_acc = _chain_accessioned(parsed) if parsed is not None else False
    return {
        "allele": canonical,
        "mhc_species": species_name,
        "mhc_class": raw_class,
        "gene": gene_name,
        "allele_resolution": resolution,
        "is_accessioned": is_acc,
    }


def _collect_allele_stats(
    frame: pd.DataFrame | None,
    *,
    mode: str,
    peptides_by_allele: dict[str, set[str]],
    mono_peptides_by_allele: dict[str, set[str]],
    multi_peptides_by_allele: dict[str, set[str]],
    studies_by_allele: dict[str, set[int]],
) -> None:
    """Accumulate per-allele peptide and study counts from one modality frame."""
    if frame is None or frame.empty or "peptide" not in frame.columns:
        return

    peptides = _str_series(frame, "peptide")
    allele_set = _str_series(frame, "mhc_allele_set")
    restriction = _str_series(frame, "mhc_restriction")
    raw_pool = allele_set.where(allele_set.ne(""), restriction)
    pmids = (
        pd.to_numeric(frame["pmid"], errors="coerce").astype("Int64")
        if "pmid" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    )
    is_mono = _bool_series(frame, "is_monoallelic") if mode == "ms" else None

    valid = peptides.ne("") & raw_pool.ne("")
    if not valid.any():
        return

    sub = pd.DataFrame({"pool": raw_pool[valid], "peptide": peptides[valid]})
    if mode == "ms":
        sub["is_mono"] = is_mono[valid].to_numpy(dtype=bool)
        for (pool_str, mono_flag), grp in sub.groupby(["pool", "is_mono"], sort=False):
            tokens = parse_allele_set(pool_str)
            if not tokens:
                continue
            unique_peps = set(grp["peptide"].unique())
            for tok in tokens:
                meta = _parse_allele_metadata(tok)
                canon = str(meta["allele"])
                is_specific_allele = meta["allele_resolution"] in {"four_digit", "two_digit"}
                is_mono_entry = (
                    len(tokens) == 1
                    and is_specific_allele
                    and (bool(mono_flag) or meta["allele_resolution"] == "four_digit")
                )
                if is_mono_entry:
                    mono_peptides_by_allele.setdefault(canon, set()).update(unique_peps)
                else:
                    multi_peptides_by_allele.setdefault(canon, set()).update(unique_peps)
    else:
        for pool_str, grp in sub.groupby("pool", sort=False):
            tokens = parse_allele_set(pool_str)
            if not tokens:
                continue
            unique_peps = set(grp["peptide"].unique())
            for tok in tokens:
                canon = str(_parse_allele_metadata(tok)["allele"])
                peptides_by_allele.setdefault(canon, set()).update(unique_peps)

    pmid_sub = pd.DataFrame({"pool": raw_pool[valid], "pmid": pmids[valid]}).dropna(subset=["pmid"])
    if not pmid_sub.empty:
        for pool_str, grp in pmid_sub.groupby("pool", sort=False):
            tokens = parse_allele_set(pool_str)
            if not tokens:
                continue
            unique_pmids = {int(v) for v in grp["pmid"].unique()}
            for tok in tokens:
                canon = str(_parse_allele_metadata(tok)["allele"])
                studies_by_allele.setdefault(canon, set()).update(unique_pmids)


def build_alleles_table(
    obs: pd.DataFrame | None = None,
    binding: pd.DataFrame | None = None,
    other: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the compact per-allele dimension table (``alleles.parquet``) (#992).

    One row per canonical ``mhcgnomes`` allele across MS, binding, and other
    assay tables.
    """
    columns = [
        "allele",
        "mhc_species",
        "mhc_class",
        "gene",
        "allele_resolution",
        "is_accessioned",
        "n_ms_monoallelic_peptides",
        "n_ms_multiallelic_peptides",
        "n_binding_peptides",
        "n_other_assay_peptides",
        "n_studies",
    ]
    ms_mono: dict[str, set[str]] = {}
    ms_multi: dict[str, set[str]] = {}
    bd_peps: dict[str, set[str]] = {}
    other_peps: dict[str, set[str]] = {}
    studies: dict[str, set[int]] = {}

    _collect_allele_stats(
        obs,
        mode="ms",
        peptides_by_allele={},
        mono_peptides_by_allele=ms_mono,
        multi_peptides_by_allele=ms_multi,
        studies_by_allele=studies,
    )
    _collect_allele_stats(
        binding,
        mode="binding",
        peptides_by_allele=bd_peps,
        mono_peptides_by_allele={},
        multi_peptides_by_allele={},
        studies_by_allele=studies,
    )
    _collect_allele_stats(
        other,
        mode="other",
        peptides_by_allele=other_peps,
        mono_peptides_by_allele={},
        multi_peptides_by_allele={},
        studies_by_allele=studies,
    )

    all_alleles = sorted(
        set(ms_mono) | set(ms_multi) | set(bd_peps) | set(other_peps) | set(studies)
    )
    if not all_alleles:
        return pd.DataFrame(columns=columns)

    rows = []
    for canon in all_alleles:
        meta = _parse_allele_metadata(canon)
        rows.append(
            {
                **meta,
                "n_ms_monoallelic_peptides": len(ms_mono.get(canon, ())),
                "n_ms_multiallelic_peptides": len(ms_multi.get(canon, ())),
                "n_binding_peptides": len(bd_peps.get(canon, ())),
                "n_other_assay_peptides": len(other_peps.get(canon, ())),
                "n_studies": len(studies.get(canon, ())),
            }
        )
    df = pd.DataFrame(rows, columns=columns)
    for col in (
        "n_ms_monoallelic_peptides",
        "n_ms_multiallelic_peptides",
        "n_binding_peptides",
        "n_other_assay_peptides",
        "n_studies",
    ):
        df[col] = df[col].astype("int64")
    df["is_accessioned"] = df["is_accessioned"].astype(bool)
    return df.sort_values(
        ["mhc_class", "mhc_species", "allele"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def samples_index_path(*, root=None, bundle: str | Path | None = None) -> Path:
    """Path to ``samples.parquet`` in the active or specified bundle."""
    if bundle is not None:
        return bundle_path("samples", bundle=bundle)
    return dataset_path("samples.parquet", root=data_dir() if root is None else root)


def alleles_path(*, root=None, bundle: str | Path | None = None) -> Path:
    """Path to ``alleles.parquet`` in the active or specified bundle."""
    if bundle is not None:
        return bundle_path("alleles", bundle=bundle)
    return dataset_path("alleles.parquet", root=data_dir() if root is None else root)


@dataset_snapshot()
def load_samples_index(
    *,
    bundle: str | Path | None = None,
    root=None,
    mhc_class: str | None = None,
    species: str | None = None,
    compute_if_missing: bool = False,
) -> pd.DataFrame:
    """Load the compact ``samples.parquet`` cohort summary table (#990)."""
    path = samples_index_path(root=root, bundle=bundle)
    if path.exists():
        df = pd.read_parquet(path)
    elif compute_if_missing:
        obs_p = (
            bundle_path("observations", bundle=bundle)
            if bundle is not None
            else dataset_path("observations.parquet", root=data_dir() if root is None else root)
        )
        df = build_samples_table(pd.read_parquet(obs_p))
    else:
        raise FileNotFoundError(
            f"{path} does not exist. Rebuild with `hitlist build` or run "
            "`hitlist.bundle_tables.build_bundle_tables(publish=True)` (or pass compute_if_missing=True)."
        )
    if mhc_class is not None and not df.empty:
        df = df[df["mhc_class"] == mhc_class]
    if species is not None and not df.empty:
        df = df[df["mhc_species"] == species]
    return df.reset_index(drop=True)


@dataset_snapshot()
def load_alleles(
    *,
    bundle: str | Path | None = None,
    root=None,
    mhc_class: str | None = None,
    species: str | None = None,
    accessioned_only: bool = False,
    compute_if_missing: bool = False,
) -> pd.DataFrame:
    """Load the compact ``alleles.parquet`` per-allele dimension table (#992)."""
    path = alleles_path(root=root, bundle=bundle)
    if path.exists():
        df = pd.read_parquet(path)
    elif compute_if_missing:
        base = (
            bundle_path(None, bundle=bundle)
            if bundle is not None
            else dataset_path(
                "observations.parquet", root=data_dir() if root is None else root
            ).parent
        )
        obs = (
            pd.read_parquet(base / "observations.parquet")
            if (base / "observations.parquet").exists()
            else None
        )
        binding = (
            pd.read_parquet(base / "binding.parquet")
            if (base / "binding.parquet").exists()
            else None
        )
        other = (
            pd.read_parquet(base / "other_assays.parquet")
            if (base / "other_assays.parquet").exists()
            else None
        )
        df = build_alleles_table(obs, binding=binding, other=other)
    else:
        raise FileNotFoundError(
            f"{path} does not exist. Rebuild with `hitlist build` or run "
            "`hitlist.bundle_tables.build_bundle_tables(publish=True)` (or pass compute_if_missing=True)."
        )
    if mhc_class is not None and not df.empty:
        df = df[df["mhc_class"] == mhc_class]
    if species is not None and not df.empty:
        df = df[df["mhc_species"] == species]
    if accessioned_only and not df.empty:
        df = df[df["is_accessioned"].astype(bool)]
    return df.reset_index(drop=True)


def write_bundle_summary_tables(
    obs: pd.DataFrame,
    binding: pd.DataFrame | None = None,
    other: pd.DataFrame | None = None,
    *,
    samples_out: Path,
    alleles_out: Path,
    primary_flanks: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Build and atomically write ``samples.parquet`` and ``alleles.parquet``."""
    samples_df = build_samples_table(obs, primary_flanks=primary_flanks)
    alleles_df = build_alleles_table(obs, binding=binding, other=other)
    atomic_write_parquet(samples_df, samples_out)
    atomic_write_parquet(alleles_df, alleles_out)
    return {"samples": samples_out, "alleles": alleles_out}


def build_bundle_tables(
    *,
    bundle: str | Path | None = None,
    root=None,
    output_dir: str | Path | None = None,
    publish: bool = False,
) -> dict[str, Path]:
    """Materialize bundle summary tables from an existing bundle (#990, #991, #992).

    Writes ``samples.parquet`` and ``alleles.parquet``, plus ``proteomes.parquet``
    and ``primary_flanks.parquet`` whenever ``peptide_mappings.parquet`` exists in
    the bundle. Published datacache bundles are immutable: pass ``output_dir`` to
    write sidecars outside the bundle, or ``publish=True`` to publish a new
    generation via :func:`hitlist.dataset.update_dataset`.
    """
    import json

    from .dataset import update_dataset
    from .host_flanks import build_primary_flanks_table, build_proteomes_registry

    base = bundle_path(None, root=root, bundle=bundle)
    obs = pd.read_parquet(base / "observations.parquet")
    binding = (
        pd.read_parquet(base / "binding.parquet") if (base / "binding.parquet").exists() else None
    )
    other = (
        pd.read_parquet(base / "other_assays.parquet")
        if (base / "other_assays.parquet").exists()
        else None
    )
    mappings_file = base / "peptide_mappings.parquet"
    has_mappings = mappings_file.exists()
    proteomes_df = None
    primary_flanks = None
    if has_mappings:
        mappings_df = pd.read_parquet(mappings_file)
        meta_file = base / "peptide_mappings_meta.json"
        m_meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        resolved_root = data_dir() if root is None else Path(root)
        proteomes_df = build_proteomes_registry(
            mappings_df,
            mappings_meta=m_meta,
            root=resolved_root,
        )
        primary_flanks = build_primary_flanks_table(mappings_df)
    elif (base / "primary_flanks.parquet").exists():
        primary_flanks = pd.read_parquet(base / "primary_flanks.parquet")

    if publish:
        if bundle is not None:
            raise ValueError("Cannot combine explicit bundle= with publish=True")
        resolved_root = data_dir() if root is None else Path(root)
        names = ["samples.parquet", "alleles.parquet"]
        if has_mappings:
            names.extend(["proteomes.parquet", "primary_flanks.parquet"])

        def _write():
            written = write_bundle_summary_tables(
                obs,
                binding=binding,
                other=other,
                samples_out=dataset_path("samples.parquet", root=resolved_root),
                alleles_out=dataset_path("alleles.parquet", root=resolved_root),
                primary_flanks=primary_flanks,
            )
            if has_mappings and proteomes_df is not None and primary_flanks is not None:
                p_out = dataset_path("proteomes.parquet", root=resolved_root)
                f_out = dataset_path("primary_flanks.parquet", root=resolved_root)
                atomic_write_parquet(proteomes_df, p_out)
                atomic_write_parquet(
                    primary_flanks,
                    f_out,
                    sort_by=("proteome_id", "peptide"),
                    row_group_size=MODALITY_ROW_GROUP_SIZE,
                )
                written["proteomes"] = p_out
                written["primary_flanks"] = f_out
            return written

        return update_dataset(tuple(names), _write, root=resolved_root)

    if output_dir is None:
        raise ValueError(
            "Published bundles are immutable; pass output_dir=... to write standalone tables "
            "or publish=True to publish a new dataset generation."
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = write_bundle_summary_tables(
        obs,
        binding=binding,
        other=other,
        samples_out=out / "samples.parquet",
        alleles_out=out / "alleles.parquet",
        primary_flanks=primary_flanks,
    )
    if has_mappings and proteomes_df is not None and primary_flanks is not None:
        p_out = out / "proteomes.parquet"
        f_out = out / "primary_flanks.parquet"
        atomic_write_parquet(proteomes_df, p_out)
        atomic_write_parquet(
            primary_flanks,
            f_out,
            sort_by=("proteome_id", "peptide"),
            row_group_size=MODALITY_ROW_GROUP_SIZE,
        )
        written["proteomes"] = p_out
        written["primary_flanks"] = f_out
    return written
