"""Bulk (non-MHC) proteomics detectability index.

Protein-level abundance from whole-cell / deep-proteome shotgun MS, keyed
on cell line. Distinct from ``observations.parquet`` (MHC-restricted
MS-elution) and ``binding.parquet`` (in-vitro binding assays). Purpose:
provide a detectability prior per (sample, source protein) — so a peptide
missing from the MHC immunopeptidome can be contextualized against
whether the parent protein is even expressed in that cell line.

Current source: CCLE proteome (Nusinow et al. 2020, Cell, PMID 31978347),
filtered to cell lines with substantial hitlist MHC MS coverage. Values
are log2-normalized TMT abundances relative to the CCLE panel median —
positive means above median, negative means below.

This is the pilot wiring for issue #67. Schema and API are intentionally
small so additional sources (ProteomicsDB, CPTAC, per-study PRIDE
deposits) can be bolted on later.
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


def load_bulk_proteomics(
    cell_line: str | Iterable[str] | None = None,
    gene_name: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load protein-level abundance from the bulk proteomics index.

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


def available_cell_lines() -> list[str]:
    """Return the canonical names of cell lines currently in the index."""
    return sorted(_load_ccle()["cell_line"].unique().tolist())
