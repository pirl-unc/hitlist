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

"""Thin, zero-DSL ML dataset loaders reading directly from verified Parquet bundles (#993).

Eliminates repeated 150-300 line consumer-side filter/cleanup pipelines across
``mhcflurry`` (#494), ``presto``, and ``tsarina``:

- :func:`load_binding_dataset` — narrow typed binding table with canonical
  ``nM`` endpoint standardization (reusing
  :func:`hitlist.model_training.qualify_binding_measurements`).
- :func:`load_ms_dataset` — deduplicated clean MS ligands with optional
  host-proteome-matched primary flanks and retained provenance/sample keys.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle_tables import (
    SAMPLE_KEY_COLUMNS,
    _bool_series,
    _str_series,
    derive_sample_id,
    parse_allele_set,
)
from .dataset import bundle_path, dataset_path, dataset_snapshot
from .downloads import data_dir
from .host_flanks import attach_host_flanks
from .model_training import MOLAR_AFFINITY_ENDPOINTS, qualify_binding_measurements

_CANONICAL_PEPTIDE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

_BINDING_DATASET_COLUMNS: tuple[str, ...] = (
    "peptide",
    "allele",
    "mhc_species",
    "mhc_class",
    "measurement_kind",
    "value_nm",
    "inequality",
    "qualitative",
    "pmid",
    "source",
)

_MS_DATASET_BASE_COLUMNS: tuple[str, ...] = (
    "peptide",
    "mhc_allele_set",
    "mhc_species",
    "mhc_class",
    "is_monoallelic",
    "sample_id",
    "pmid",
    "provenance_id",
    "source",
)

_MS_DATASET_FLANK_COLUMNS: tuple[str, ...] = (
    "primary_proteome_id",
    "primary_protein_id",
    "primary_gene_name",
    "n_flank",
    "c_flank",
    "flank_mapped",
    "has_flank",
    "cohort_flank_coverage",
)


def _resolve_modality_path(name: str, *, bundle: str | Path | None, root=None) -> Path:
    if bundle is not None:
        return bundle_path(name, bundle=bundle)
    return dataset_path(f"{name}.parquet", root=data_dir() if root is None else root)


@dataset_snapshot()
def load_binding_dataset(
    bundle: str | Path | None = None,
    *,
    mhc_class: str | None = "I",
    species: str | None = None,
    clean: bool = True,
    include_qualitative: bool = True,
    root=None,
) -> pd.DataFrame:
    """Load a narrow, task-ready binding dataset directly from a verified Parquet bundle (#993).

    Returns columns:
    ``(peptide, allele, mhc_species, mhc_class, measurement_kind, value_nm,
    inequality, qualitative, pmid, source)``.
    """
    from .curation import normalize_species

    path = _resolve_modality_path("binding", bundle=bundle, root=root)
    filters = []
    if mhc_class is not None:
        filters.append(("mhc_class", "==", mhc_class))
    if species is not None:
        filters.append(("mhc_species", "==", normalize_species(species)))

    df = pd.read_parquet(path, filters=filters or None)
    if df.empty:
        return pd.DataFrame(columns=list(_BINDING_DATASET_COLUMNS))

    if clean:
        pep = _str_series(df, "peptide")
        mask = pep.str.match(_CANONICAL_PEPTIDE_RE, na=False)
        for flag in ("has_ptm", "is_non_peptide_ligand", "is_potential_contaminant"):
            if flag in df.columns:
                mask &= ~_bool_series(df, flag)
        if "allele_resolution" in df.columns:
            mask &= _str_series(df, "allele_resolution").eq("four_digit")
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame(columns=list(_BINDING_DATASET_COLUMNS))

    qualified = qualify_binding_measurements(df)
    family = qualified["binding_endpoint_family"].astype("string").fillna("")
    std_unit = qualified["standardized_unit"].astype("string").fillna("")
    std_val = pd.to_numeric(qualified["standardized_value"], errors="coerce")
    unit_ok = (
        qualified["unit_conversion_status"]
        .astype("string")
        .isin({"canonical_unit", "linear_converted"})
    )
    is_molar_nm = (
        family.isin(MOLAR_AFFINITY_ENDPOINTS) & std_unit.eq("nM") & unit_ok & std_val.notna()
    )
    qual_text = _str_series(qualified, "qualitative_measurement")

    if clean:
        keep = is_molar_nm | (qual_text.ne("") if include_qualitative else False)
        qualified = qualified[keep].copy()
        if qualified.empty:
            return pd.DataFrame(columns=list(_BINDING_DATASET_COLUMNS))
        family = family[keep]
        std_val = std_val[keep]
        is_molar_nm = is_molar_nm[keep]
        qual_text = qual_text[keep]

    allele_col = _str_series(qualified, "mhc_restriction")
    if "mhc_allele_set" in qualified.columns:
        aset = _str_series(qualified, "mhc_allele_set")
        allele_col = allele_col.where(allele_col.ne(""), aset)

    out = pd.DataFrame(
        {
            "peptide": _str_series(qualified, "peptide"),
            "allele": allele_col,
            "mhc_species": _str_series(qualified, "mhc_species"),
            "mhc_class": _str_series(qualified, "mhc_class"),
            "measurement_kind": family,
            "value_nm": pd.Series(
                np.where(
                    is_molar_nm.to_numpy(dtype=bool), std_val.to_numpy(dtype=np.float64), np.nan
                ),
                index=qualified.index,
                dtype="float64",
            ),
            "inequality": qualified["normalized_inequality"].astype("string").fillna(""),
            "qualitative": qual_text,
            "pmid": (
                pd.to_numeric(qualified["pmid"], errors="coerce").astype("Int64")
                if "pmid" in qualified.columns
                else pd.Series(pd.NA, index=qualified.index, dtype="Int64")
            ),
            "source": _str_series(qualified, "source"),
        },
        columns=list(_BINDING_DATASET_COLUMNS),
    )

    if clean:
        out = out.drop_duplicates(
            subset=[
                "peptide",
                "allele",
                "measurement_kind",
                "value_nm",
                "inequality",
                "qualitative",
                "pmid",
            ],
            keep="first",
        )
    return out.reset_index(drop=True)


def _compute_sample_ids(df: pd.DataFrame) -> pd.Series:
    key_df = pd.DataFrame(index=df.index)
    key_df["pmid"] = (
        pd.to_numeric(df["pmid"], errors="coerce").astype("Int64")
        if "pmid" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    for col in SAMPLE_KEY_COLUMNS[1:]:
        key_df[col] = _str_series(df, col)
    unique_keys = key_df.drop_duplicates()
    mapping = {
        row: derive_sample_id(*row) for row in unique_keys.itertuples(index=False, name=None)
    }
    return pd.Series(
        [mapping[row] for row in key_df.itertuples(index=False, name=None)],
        index=df.index,
        dtype="string",
    )


@dataset_snapshot()
def load_ms_dataset(
    bundle: str | Path | None = None,
    *,
    mhc_class: str | None = "I",
    species: str | None = None,
    monoallelic_only: bool = False,
    with_flanks: bool = False,
    clean: bool = True,
    min_flank_coverage: float | None = None,
    host_proteome: str | None = None,
    root=None,
) -> pd.DataFrame:
    """Load deduplicated, task-ready MS ligands directly from a verified Parquet bundle (#993).

    When ``with_flanks=True``, attaches deterministic host-proteome-matched
    ``(n_flank, c_flank)`` via :func:`hitlist.host_flanks.attach_host_flanks`
    and computes per-cohort ``cohort_flank_coverage``.
    """
    from .curation import normalize_species

    cols_out = list(_MS_DATASET_BASE_COLUMNS) + (
        list(_MS_DATASET_FLANK_COLUMNS) if with_flanks else []
    )
    path = _resolve_modality_path("observations", bundle=bundle, root=root)
    filters = []
    if mhc_class is not None:
        filters.append(("mhc_class", "==", mhc_class))
    if species is not None:
        filters.append(("mhc_species", "==", normalize_species(species)))

    df = pd.read_parquet(path, filters=filters or None)
    if df.empty:
        return pd.DataFrame(columns=cols_out)

    if clean:
        pep = _str_series(df, "peptide")
        mask = pep.str.match(_CANONICAL_PEPTIDE_RE, na=False)
        for flag in ("has_ptm", "is_non_peptide_ligand", "is_potential_contaminant"):
            if flag in df.columns:
                mask &= ~_bool_series(df, flag)
        aset = _str_series(df, "mhc_allele_set")
        mask &= aset.ne("")
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame(columns=cols_out)

    if monoallelic_only:
        aset = _str_series(df, "mhc_allele_set")
        unique_pools = aset.unique()
        from .curation import classify_allele_resolution

        mono_pool_map = {}
        for pool in unique_pools:
            toks = parse_allele_set(pool)
            mono_pool_map[pool] = (
                len(toks) == 1 and classify_allele_resolution(toks[0]) == "four_digit"
            )
        mono_mask = aset.map(mono_pool_map).fillna(False).astype(bool)
        df = df[mono_mask].copy()
        if df.empty:
            return pd.DataFrame(columns=cols_out)

    df["sample_id"] = _compute_sample_ids(df)

    if with_flanks:
        df = attach_host_flanks(
            df,
            bundle=bundle,
            root=root,
            host_proteome=host_proteome,
            compute_if_missing=True,
        )

        def _group_flank_cov(group: pd.DataFrame) -> float:
            n_total = group["peptide"].nunique()
            if n_total <= 0:
                return 0.0
            return float(group.loc[group["has_flank"], "peptide"].nunique() / n_total)

        cov = (
            df.groupby("sample_id", sort=False)
            .apply(_group_flank_cov, include_groups=False)
            .rename("cohort_flank_coverage")
        )
        df["cohort_flank_coverage"] = df["sample_id"].map(cov).astype("float64")
        if min_flank_coverage is not None:
            df = df[df["cohort_flank_coverage"] >= float(min_flank_coverage)].copy()
            if df.empty:
                return pd.DataFrame(columns=cols_out)

    dedupe_cols = (
        ["peptide", "mhc_allele_set", "n_flank", "c_flank", "sample_id"]
        if with_flanks
        else ["peptide", "mhc_allele_set", "sample_id"]
    )
    df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    out = pd.DataFrame(
        {
            "peptide": _str_series(df, "peptide"),
            "mhc_allele_set": _str_series(df, "mhc_allele_set"),
            "mhc_species": _str_series(df, "mhc_species"),
            "mhc_class": _str_series(df, "mhc_class"),
            "is_monoallelic": _bool_series(df, "is_monoallelic"),
            "sample_id": df["sample_id"].astype("string"),
            "pmid": (
                pd.to_numeric(df["pmid"], errors="coerce").astype("Int64")
                if "pmid" in df.columns
                else pd.Series(pd.NA, index=df.index, dtype="Int64")
            ),
            "provenance_id": _str_series(df, "provenance_id"),
            "source": _str_series(df, "source"),
        }
    )
    if with_flanks:
        for col in _MS_DATASET_FLANK_COLUMNS:
            out[col] = df[col]
    return out[cols_out].reset_index(drop=True)
