"""Bulk (non-MHC) proteomics detectability indices.

All data in this module is **shotgun / whole-cell MS**, not MHC-ligand
immunopeptidomics. It lives alongside ``observations.parquet`` (MHC
MS-elution) and ``binding.parquet`` (in-vitro binding) as a third,
strictly non-MHC table so downstream consumers can use it as a
detectability prior without ever conflating it with immunopeptidomics
data.

Two levels of granularity, each with its own loader:

- ``load_bulk_proteomics`` — protein-level abundance per cell line
  (CCLE; Nusinow et al. 2020, PMID 31978347). Good for "is this gene
  expressed in this sample?"

- ``load_bulk_peptides`` — peptide-level detection per cell line
  (Bekker-Jensen et al. 2017, PMID 28591648). Good for "within this
  protein, which tryptic peptides are ever observable by MS?" (the
  intra-protein detectability bias model).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files

import pandas as pd

_DATA_MODULE = "hitlist.data.bulk_proteomics"


@lru_cache(maxsize=1)
def _load_ccle() -> pd.DataFrame:
    path = files(_DATA_MODULE) / "ccle_nusinow_2020.csv.gz"
    return pd.read_csv(str(path), compression="gzip")


@lru_cache(maxsize=1)
def _load_bj() -> pd.DataFrame:
    path = files(_DATA_MODULE) / "bekker_jensen_2017_peptides.csv.gz"
    return pd.read_csv(str(path), compression="gzip")


def load_bulk_proteomics(
    cell_line: str | Iterable[str] | None = None,
    gene_name: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Protein-level bulk proteomics abundance (shotgun MS, NOT MHC ligands).

    Parameters
    ----------
    cell_line
        Filter to a single cell line (e.g. ``"MDA-MB-231"``) or iterable
        of cell lines. Matched case-insensitively against the canonical
        name column.
    gene_name
        Filter to one or more HGNC gene symbols (exact match, case-sensitive).

    Returns
    -------
    DataFrame with columns:
        cell_line, gene_symbol, uniprot_acc, protein_id,
        abundance_log2_normalized, source, reference.
    """
    df = _load_ccle()
    if cell_line is not None:
        if isinstance(cell_line, str):
            cell_line = [cell_line]
        wanted = {c.casefold() for c in cell_line}
        df = df[df["cell_line"].str.casefold().isin(wanted)]
    if gene_name is not None:
        if isinstance(gene_name, str):
            gene_name = [gene_name]
        df = df[df["gene_symbol"].isin(list(gene_name))]
    return df.reset_index(drop=True)


def load_bulk_peptides(
    cell_line: str | Iterable[str] | None = None,
    gene_name: str | Iterable[str] | None = None,
    uniprot_acc: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Peptide-level bulk proteomics detections (shotgun MS, NOT MHC ligands).

    Identifies which tryptic peptides *within* a protein were ever
    observed by deep shotgun MS on a given cell line — the intra-protein
    detectability prior for MHC-ligandome analyses. Source: Bekker-Jensen
    et al. 2017 (PMID 28591648), covering HeLa, A549, HCT116, HEK293,
    MCF7. Tryptic digest, 46-fraction pre-fractionation, ≥2 biological
    replicates per cell line; a peptide is included here if detected at
    non-zero intensity in any replicate of that cell line.

    Parameters
    ----------
    cell_line
        Filter to one or more cell lines (case-insensitive match).
    gene_name
        Filter to one or more HGNC gene symbols (exact match).
    uniprot_acc
        Filter to one or more UniProt accessions (exact match).

    Returns
    -------
    DataFrame with columns:
        peptide, cell_line, uniprot_acc, gene_symbol, length,
        start_position, end_position, source, reference.
    """
    df = _load_bj()
    if cell_line is not None:
        if isinstance(cell_line, str):
            cell_line = [cell_line]
        wanted = {c.casefold() for c in cell_line}
        df = df[df["cell_line"].str.casefold().isin(wanted)]
    if gene_name is not None:
        if isinstance(gene_name, str):
            gene_name = [gene_name]
        df = df[df["gene_symbol"].isin(list(gene_name))]
    if uniprot_acc is not None:
        if isinstance(uniprot_acc, str):
            uniprot_acc = [uniprot_acc]
        df = df[df["uniprot_acc"].isin(list(uniprot_acc))]
    return df.reset_index(drop=True)


def available_cell_lines() -> list[str]:
    """Return the union of cell lines across all bulk proteomics indices."""
    protein = set(_load_ccle()["cell_line"].unique())
    peptide = set(_load_bj()["cell_line"].unique())
    return sorted(protein | peptide)


def available_protein_cell_lines() -> list[str]:
    """Cell lines covered by the protein-level index (load_bulk_proteomics)."""
    return sorted(_load_ccle()["cell_line"].unique().tolist())


def available_peptide_cell_lines() -> list[str]:
    """Cell lines covered by the peptide-level index (load_bulk_peptides)."""
    return sorted(_load_bj()["cell_line"].unique().tolist())
