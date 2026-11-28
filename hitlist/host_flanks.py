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

"""Host-proteome registry (``proteomes.parquet``) and deterministic 1:1 host flanks (#991).

Eliminates unsafe peptide-only joins against multi-species ``peptide_mappings.parquet``
and surfaces silent empty-vs-real flank artifacts per cohort.

Tables written into each bundle alongside ``peptide_mappings.parquet``:

- ``proteomes.parquet`` — host reference proteome registry mapping
  ``proteome_id`` -> ``(species, taxon_id, source, fasta_path, sha256,
  n_proteins, n_peptides_mapped)``.
- ``primary_flanks.parquet`` — deterministic 1:1
  ``(peptide, proteome_id) -> (n_flank, c_flank, protein_id, gene_name,
  gene_id, transcript_id, is_canonical_transcript, position, n_mappings)``
  selected via canonical-transcript tie-breaking and sorted by
  ``(proteome_id, peptide)`` for millisecond predicate pushdown.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle_tables import (
    SAMPLE_KEY_COLUMNS,
    _resolve_row_proteome_species,
    _str_series,
    derive_sample_id,
)
from .dataset import bundle_path, dataset_path, dataset_snapshot
from .downloads import data_dir, list_proteomes, lookup_proteome
from .parquet_io import atomic_write_parquet

_PROTEOME_REGISTRY_COLUMNS: tuple[str, ...] = (
    "proteome_id",
    "species",
    "taxon_id",
    "source",
    "fasta_path",
    "sha256",
    "n_proteins",
    "n_peptides_mapped",
)

_PRIMARY_FLANK_COLUMNS: tuple[str, ...] = (
    "peptide",
    "proteome_id",
    "n_flank",
    "c_flank",
    "protein_id",
    "gene_name",
    "gene_id",
    "transcript_id",
    "is_canonical_transcript",
    "position",
    "n_mappings",
)

_CANONICAL_PEPTIDE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

_ENSEMBL_TAXON_IDS: dict[str, int] = {
    "Homo sapiens": 9606,
    "Mus musculus": 10090,
    "Rattus norvegicus": 10116,
    "Gallus gallus": 9031,
    "Macaca mulatta": 9544,
    "Sus scrofa": 9823,
    "Bos taurus": 9913,
    "Canis lupus": 9615,
    "Canis lupus familiaris": 9615,
}


def _file_sha256_and_proteins(path: Path) -> tuple[str, int]:
    """Compute SHA-256 and FASTA header count (``>`` lines) in one streaming pass."""
    digest = hashlib.sha256()
    n_proteins = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            n_proteins += chunk.count(b"\n>")
        # Handle first line starting with ">"
    with path.open("rb") as fh:
        if fh.read(1) == b">":
            n_proteins += 1
    return digest.hexdigest(), n_proteins


def _relative_or_str(path_str: str, root: Path | None) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if root is not None:
        try:
            return str(p.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return str(p)


def build_proteomes_registry(
    mappings_df: pd.DataFrame | None = None,
    *,
    mappings_meta: dict | None = None,
    manifest_proteomes: dict | None = None,
    root: Path | None = None,
    hash_fastas: bool = True,
) -> pd.DataFrame:
    """Build ``proteomes.parquet`` indexing every proteome present in mappings/metadata (#991)."""
    if manifest_proteomes is None:
        try:
            manifest_proteomes = list_proteomes()
        except Exception:
            manifest_proteomes = {}
    resolved_root = Path(root) if root is not None else data_dir()

    # Map normalized species/label -> manifest entry
    manifest_by_key: dict[str, dict] = {}
    for key, entry in (manifest_proteomes or {}).items():
        if isinstance(entry, dict):
            manifest_by_key[str(key)] = entry
            manifest_by_key[str(key).lower()] = entry
            if entry.get("upid"):
                manifest_by_key[str(entry["upid"])] = entry

    mapped_stats: dict[str, dict[str, object]] = {}
    if mappings_df is not None and not mappings_df.empty:
        prot_col = "proteome" if "proteome" in mappings_df.columns else "proteome_id"
        if prot_col in mappings_df.columns and "peptide" in mappings_df.columns:
            prot_s = mappings_df[prot_col].astype("string").fillna("").str.strip()
            pep_s = mappings_df["peptide"].astype("string").fillna("").str.strip()
            src_s = (
                mappings_df["proteome_source"].astype("string").fillna("").str.strip()
                if "proteome_source" in mappings_df.columns
                else pd.Series("", index=mappings_df.index, dtype="string")
            )
            protein_s = (
                mappings_df["protein_id"].astype("string").fillna("").str.strip()
                if "protein_id" in mappings_df.columns
                else pd.Series("", index=mappings_df.index, dtype="string")
            )
            work = pd.DataFrame(
                {
                    "proteome_id": prot_s,
                    "peptide": pep_s,
                    "source": src_s,
                    "protein_id": protein_s,
                }
            )
            work = work[work["proteome_id"].ne("")]
            if not work.empty:
                for pid, grp in work.groupby("proteome_id", sort=False):
                    first_src = ""
                    non_empty_src = grp["source"][grp["source"].ne("")]
                    if not non_empty_src.empty:
                        first_src = str(non_empty_src.iloc[0])
                    mapped_stats[str(pid)] = {
                        "source": first_src,
                        "n_proteins_observed": int(
                            grp["protein_id"][grp["protein_id"].ne("")].nunique()
                        ),
                        "n_peptides_mapped": int(grp["peptide"][grp["peptide"].ne("")].nunique()),
                    }

    # Include proteomes searched with 0 matches from peptide_mappings_meta.json
    per_proteome = (mappings_meta or {}).get("per_proteome") or {}
    all_ids = sorted(set(mapped_stats) | {str(k) for k in per_proteome})
    if not all_ids:
        return pd.DataFrame(columns=list(_PROTEOME_REGISTRY_COLUMNS))

    rows = []
    for pid in all_ids:
        stat = mapped_stats.get(pid, {})
        meta_entry = per_proteome.get(pid, {}) if isinstance(per_proteome, dict) else {}
        man_entry = manifest_by_key.get(pid) or manifest_by_key.get(pid.lower()) or {}

        reg_entry = None
        try:
            reg_entry = lookup_proteome(pid)
        except Exception:
            reg_entry = None

        species_name = pid
        if isinstance(reg_entry, dict) and reg_entry.get("canonical_species"):
            species_name = str(reg_entry["canonical_species"])
        elif man_entry.get("species"):
            species_name = str(man_entry["species"])

        taxon_id = pd.NA
        raw_tax = man_entry.get("taxon_id") or man_entry.get("expected_taxon_id")
        if raw_tax is None and isinstance(reg_entry, dict):
            raw_tax = reg_entry.get("expected_taxon_id") or reg_entry.get("taxon_id")
        if raw_tax is None:
            raw_tax = _ENSEMBL_TAXON_IDS.get(species_name) or _ENSEMBL_TAXON_IDS.get(pid)
        if raw_tax is not None:
            try:
                taxon_id = int(raw_tax)
            except (TypeError, ValueError):
                taxon_id = pd.NA

        source = str(stat.get("source") or "")
        if not source:
            if man_entry.get("source"):
                source = str(man_entry["source"])
            elif isinstance(reg_entry, dict):
                if reg_entry.get("kind") == "ensembl":
                    rel = reg_entry.get("release", "")
                    source = f"ensembl:{rel}" if rel else "ensembl"
                elif reg_entry.get("proteome_id"):
                    source = f"uniprot:{reg_entry['proteome_id']}"

        fasta_raw = str(man_entry.get("path") or man_entry.get("url") or "")
        sha256 = str(man_entry.get("sha256") or "")
        n_proteins = int(man_entry.get("n_proteins") or 0)

        if fasta_raw:
            fasta_p = Path(fasta_raw)
            if fasta_p.is_file() and hash_fastas and (not sha256 or n_proteins == 0):
                try:
                    computed_sha, computed_n = _file_sha256_and_proteins(fasta_p)
                    if not sha256:
                        sha256 = computed_sha
                    if n_proteins == 0:
                        n_proteins = computed_n
                except OSError:
                    pass
            fasta_rel = _relative_or_str(fasta_raw, resolved_root)
        else:
            fasta_rel = ""

        if n_proteins == 0:
            n_proteins = int(stat.get("n_proteins_observed") or 0)

        n_peptides_mapped = int(
            stat.get("n_peptides_mapped")
            if "n_peptides_mapped" in stat
            else (meta_entry.get("peptides_matched") or 0)
        )
        rows.append(
            {
                "proteome_id": pid,
                "species": species_name,
                "taxon_id": taxon_id,
                "source": source,
                "fasta_path": fasta_rel,
                "sha256": sha256,
                "n_proteins": n_proteins,
                "n_peptides_mapped": n_peptides_mapped,
            }
        )

    df = pd.DataFrame(rows, columns=list(_PROTEOME_REGISTRY_COLUMNS))
    df["taxon_id"] = pd.to_numeric(df["taxon_id"], errors="coerce").astype("Int64")
    df["n_proteins"] = df["n_proteins"].astype("int64")
    df["n_peptides_mapped"] = df["n_peptides_mapped"].astype("int64")
    return df.sort_values("proteome_id", kind="mergesort").reset_index(drop=True)


def build_primary_flanks_table(mappings_df: pd.DataFrame) -> pd.DataFrame:
    """Select one deterministic canonical-isoform flank row per ``(peptide, proteome_id)`` (#991).

    Tie-breaking order (highest priority first):
    1. ``is_canonical_transcript == True``
    2. ``gene_biotype == "protein_coding"``
    3. Non-empty ``gene_name``
    4. Longer total flank context ``len(n_flank) + len(c_flank)``
    5. Lexicographic ``(protein_id, position, n_flank, c_flank)``
    """
    if mappings_df is None or mappings_df.empty:
        return pd.DataFrame(columns=list(_PRIMARY_FLANK_COLUMNS))

    prot_col = "proteome" if "proteome" in mappings_df.columns else "proteome_id"
    if "peptide" not in mappings_df.columns or prot_col not in mappings_df.columns:
        return pd.DataFrame(columns=list(_PRIMARY_FLANK_COLUMNS))

    work = pd.DataFrame(index=mappings_df.index)
    work["peptide"] = _str_series(mappings_df, "peptide")
    work["proteome_id"] = _str_series(mappings_df, prot_col)
    work["n_flank"] = _str_series(mappings_df, "n_flank")
    work["c_flank"] = _str_series(mappings_df, "c_flank")
    work["protein_id"] = _str_series(mappings_df, "protein_id")
    work["gene_name"] = _str_series(mappings_df, "gene_name")
    work["gene_id"] = _str_series(mappings_df, "gene_id")
    work["transcript_id"] = _str_series(mappings_df, "transcript_id")
    work["is_canonical_transcript"] = (
        mappings_df["is_canonical_transcript"].fillna(False).astype(bool)
        if "is_canonical_transcript" in mappings_df.columns
        else False
    )
    work["position"] = (
        pd.to_numeric(mappings_df["position"], errors="coerce").fillna(-1).astype("int64")
        if "position" in mappings_df.columns
        else np.int64(-1)
    )
    biotype = _str_series(mappings_df, "gene_biotype")

    valid = work["peptide"].ne("") & work["proteome_id"].ne("")
    work = work[valid].copy()
    if work.empty:
        return pd.DataFrame(columns=list(_PRIMARY_FLANK_COLUMNS))
    biotype = biotype[valid]

    # Count total mapping occurrences per (peptide, proteome_id) before collapsing 1:1.
    counts = (
        work.groupby(["proteome_id", "peptide"], sort=False, observed=True)
        .size()
        .rename("n_mappings")
    )

    # Sort descending on preference booleans/lengths, ascending on deterministic tie-breakers.
    work["_rank_canon"] = (~work["is_canonical_transcript"]).astype(np.int8)
    work["_rank_coding"] = (biotype != "protein_coding").astype(np.int8)
    work["_rank_gene"] = work["gene_name"].eq("").astype(np.int8)
    work["_rank_flank_len"] = -(
        work["n_flank"].str.len().to_numpy(dtype=np.int16)
        + work["c_flank"].str.len().to_numpy(dtype=np.int16)
    )

    work = work.sort_values(
        [
            "proteome_id",
            "peptide",
            "_rank_canon",
            "_rank_coding",
            "_rank_gene",
            "_rank_flank_len",
            "protein_id",
            "position",
            "n_flank",
            "c_flank",
        ],
        kind="mergesort",
        na_position="last",
    )
    primary = work.drop_duplicates(subset=["proteome_id", "peptide"], keep="first").drop(
        columns=["_rank_canon", "_rank_coding", "_rank_gene", "_rank_flank_len"]
    )
    primary = primary.merge(counts, on=["proteome_id", "peptide"], how="left")
    primary["n_mappings"] = primary["n_mappings"].astype("int64")
    return primary[list(_PRIMARY_FLANK_COLUMNS)].reset_index(drop=True)


def proteomes_path(*, root=None, bundle: str | Path | None = None) -> Path:
    """Path to ``proteomes.parquet`` in the active or specified bundle."""
    if bundle is not None:
        return bundle_path("proteomes", bundle=bundle)
    return dataset_path("proteomes.parquet", root=data_dir() if root is None else root)


def primary_flanks_path(*, root=None, bundle: str | Path | None = None) -> Path:
    """Path to ``primary_flanks.parquet`` in the active or specified bundle."""
    if bundle is not None:
        return bundle_path("primary_flanks", bundle=bundle)
    return dataset_path("primary_flanks.parquet", root=data_dir() if root is None else root)


@dataset_snapshot()
def load_proteomes(
    *,
    bundle: str | Path | None = None,
    root=None,
    species: str | None = None,
    compute_if_missing: bool = False,
) -> pd.DataFrame:
    """Load the host-proteome registry table (``proteomes.parquet``) (#991)."""
    path = proteomes_path(root=root, bundle=bundle)
    if path.exists():
        df = pd.read_parquet(path)
    elif compute_if_missing:
        base = path.parent
        mp = base / "peptide_mappings.parquet"
        mm = base / "peptide_mappings_meta.json"
        mappings_df = pd.read_parquet(mp) if mp.exists() else pd.DataFrame()
        meta = json.loads(mm.read_text(encoding="utf-8")) if mm.exists() else {}
        df = build_proteomes_registry(
            mappings_df,
            mappings_meta=meta,
            root=data_dir() if root is None else Path(root),
        )
    else:
        raise FileNotFoundError(
            f"{path} does not exist. Rebuild with `hitlist build` or pass compute_if_missing=True."
        )
    if species is not None and not df.empty:
        df = df[(df["species"] == species) | (df["proteome_id"] == species)]
    return df.reset_index(drop=True)


@dataset_snapshot()
def load_primary_flanks(
    *,
    bundle: str | Path | None = None,
    root=None,
    proteome_id: str | Iterable[str] | None = None,
    peptides: Iterable[str] | None = None,
    compute_if_missing: bool = False,
) -> pd.DataFrame:
    """Load deterministic 1:1 ``primary_flanks.parquet`` with optional PyArrow pushdown (#991)."""
    path = primary_flanks_path(root=root, bundle=bundle)
    prot_list: list[str] | None = None
    if proteome_id is not None:
        if isinstance(proteome_id, str):
            prot_list = [proteome_id]
        else:
            prot_list = sorted({str(p).strip() for p in proteome_id if p and str(p).strip()})

    if path.exists():
        filters = []
        if prot_list is not None:
            if not prot_list:
                return pd.DataFrame(columns=list(_PRIMARY_FLANK_COLUMNS))
            filters.append(("proteome_id", "in", prot_list))
        df = pd.read_parquet(path, filters=filters or None)
    elif compute_if_missing:
        mp = path.parent / "peptide_mappings.parquet"
        if not mp.exists():
            return pd.DataFrame(columns=list(_PRIMARY_FLANK_COLUMNS))
        filters = [("proteome", "in", prot_list)] if prot_list else None
        df = build_primary_flanks_table(pd.read_parquet(mp, filters=filters))
    else:
        raise FileNotFoundError(
            f"{path} does not exist. Rebuild with `hitlist build` or pass compute_if_missing=True."
        )

    if peptides is not None and not df.empty:
        pep_set = {str(p).strip() for p in peptides if p}
        df = df[df["peptide"].isin(pep_set)]
    return df.reset_index(drop=True)


def attach_host_flanks(
    df: pd.DataFrame,
    *,
    primary_flanks: pd.DataFrame | None = None,
    bundle: str | Path | None = None,
    root=None,
    host_proteome: str | None = None,
    compute_if_missing: bool = True,
) -> pd.DataFrame:
    """Attach host-proteome-matched primary ``(n_flank, c_flank)`` columns to ``df`` (#991).

    Never joins on ``peptide`` alone: every row's host proteome is resolved from
    ``host_proteome`` or ``(source_species, source_organism, species, mhc_species)``
    and matched strictly on ``(peptide, proteome_id)``.
    """
    out = df.copy()
    if out.empty:
        for col in (
            "primary_proteome_id",
            "primary_protein_id",
            "primary_gene_name",
            "n_flank",
            "c_flank",
        ):
            out[col] = pd.Series(dtype="string")
        out["flank_mapped"] = pd.Series(dtype=bool)
        out["has_flank"] = pd.Series(dtype=bool)
        out["n_host_mappings"] = pd.Series(dtype="int64")
        return out

    if host_proteome is not None:
        out["primary_proteome_id"] = pd.Series(str(host_proteome), index=out.index, dtype="string")
    else:
        out["primary_proteome_id"] = _resolve_row_proteome_species(out)

    needed_proteomes = sorted({p for p in out["primary_proteome_id"].unique() if p})
    if primary_flanks is None:
        pf = load_primary_flanks(
            bundle=bundle,
            root=root,
            proteome_id=needed_proteomes,
            compute_if_missing=compute_if_missing,
        )
    else:
        pf = primary_flanks.copy()
        if "proteome" in pf.columns and "proteome_id" not in pf.columns:
            pf = pf.rename(columns={"proteome": "proteome_id"})

    if pf.empty:
        out["primary_protein_id"] = ""
        out["primary_gene_name"] = ""
        out["n_flank"] = ""
        out["c_flank"] = ""
        out["flank_mapped"] = False
        out["has_flank"] = False
        out["n_host_mappings"] = np.int64(0)
        return out

    pf_small = (
        pf[
            [
                "peptide",
                "proteome_id",
                "protein_id",
                "gene_name",
                "n_flank",
                "c_flank",
                "n_mappings",
            ]
        ]
        .drop_duplicates(subset=["peptide", "proteome_id"], keep="first")
        .rename(
            columns={
                "proteome_id": "primary_proteome_id",
                "protein_id": "primary_protein_id",
                "gene_name": "primary_gene_name",
                "n_mappings": "n_host_mappings",
            }
        )
    )
    pf_small["_flank_mapped"] = True

    # Drop any pre-existing target columns cleanly before merging.
    drop_existing = [
        c
        for c in (
            "primary_protein_id",
            "primary_gene_name",
            "n_flank",
            "c_flank",
            "flank_mapped",
            "has_flank",
            "n_host_mappings",
        )
        if c in out.columns
    ]
    if drop_existing:
        out = out.drop(columns=drop_existing)

    # Preserve original index cleanly through merge.
    orig_index = out.index
    merged = out.merge(pf_small, on=["peptide", "primary_proteome_id"], how="left")
    merged.index = orig_index
    for col in ("primary_protein_id", "primary_gene_name", "n_flank", "c_flank"):
        merged[col] = merged[col].astype("string").fillna("")
    merged["flank_mapped"] = merged.pop("_flank_mapped").fillna(False).astype(bool)
    merged["has_flank"] = merged["n_flank"].ne("") | merged["c_flank"].ne("")
    merged["n_host_mappings"] = merged["n_host_mappings"].fillna(0).astype("int64")
    return merged


@dataset_snapshot()
def load_flanked_ligands(
    *,
    species: str | None = None,
    mhc_class: str | None = "I",
    bundle: str | Path | None = None,
    root=None,
    host_proteome: str | None = None,
    clean: bool = True,
    require_flanks: bool = False,
    min_flank_coverage: float | None = None,
) -> pd.DataFrame:
    """Load MS ligands with host-proteome-matched flanks and cohort flank coverage (#991)."""
    from .curation import normalize_species

    obs_path = (
        bundle_path("observations", bundle=bundle)
        if bundle is not None
        else dataset_path("observations.parquet", root=data_dir() if root is None else root)
    )
    filters = []
    if mhc_class is not None:
        filters.append(("mhc_class", "==", mhc_class))
    if species is not None:
        filters.append(("mhc_species", "==", normalize_species(species)))

    df = pd.read_parquet(obs_path, filters=filters or None)
    if clean and not df.empty:
        pep = _str_series(df, "peptide")
        mask = pep.str.match(_CANONICAL_PEPTIDE_RE, na=False)
        for flag in ("has_ptm", "is_non_peptide_ligand", "is_potential_contaminant"):
            if flag in df.columns:
                mask &= ~df[flag].fillna(False).astype(bool)
        df = df[mask].copy()

    df = attach_host_flanks(
        df,
        bundle=bundle,
        root=root,
        host_proteome=host_proteome,
        compute_if_missing=True,
    )
    if df.empty:
        df["sample_id"] = pd.Series(dtype="string")
        df["cohort_flank_coverage"] = pd.Series(dtype="float64")
        return df

    # Attach deterministic cohort sample_id and per-cohort flank_coverage.
    key_cols_frame = pd.DataFrame(index=df.index)
    key_cols_frame["pmid"] = (
        pd.to_numeric(df["pmid"], errors="coerce").astype("Int64")
        if "pmid" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    for col in SAMPLE_KEY_COLUMNS[1:]:
        key_cols_frame[col] = _str_series(df, col)
    unique_keys = key_cols_frame.drop_duplicates()
    key_to_sid = {
        row: derive_sample_id(*row) for row in unique_keys.itertuples(index=False, name=None)
    }
    df["sample_id"] = [key_to_sid[row] for row in key_cols_frame.itertuples(index=False, name=None)]

    cov = (
        df.groupby("sample_id", sort=False)
        .apply(
            lambda g: float(g.loc[g["has_flank"], "peptide"].nunique() / g["peptide"].nunique())
            if g["peptide"].nunique() > 0
            else 0.0,
            include_groups=False,
        )
        .rename("cohort_flank_coverage")
    )
    df["cohort_flank_coverage"] = df["sample_id"].map(cov).astype("float64")

    if min_flank_coverage is not None:
        df = df[df["cohort_flank_coverage"] >= float(min_flank_coverage)]
    if require_flanks:
        df = df[df["has_flank"]]
    return df.reset_index(drop=True)


def write_host_flank_tables(
    mappings_df: pd.DataFrame,
    *,
    proteomes_out: Path,
    primary_flanks_out: Path,
    mappings_meta: dict | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and atomically write ``proteomes.parquet`` and ``primary_flanks.parquet`` (#991)."""
    proteomes_df = build_proteomes_registry(
        mappings_df,
        mappings_meta=mappings_meta,
        root=root,
    )
    primary_flanks_df = build_primary_flanks_table(mappings_df)
    atomic_write_parquet(proteomes_df, proteomes_out)
    atomic_write_parquet(
        primary_flanks_df,
        primary_flanks_out,
        sort_by=("proteome_id", "peptide"),
        row_group_size=250_000,
    )
    return proteomes_df, primary_flanks_df
