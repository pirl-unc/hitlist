"""Bulk (non-MHC) proteomics detectability indices.

All data in this module is **shotgun / whole-cell MS**, not MHC-ligand
immunopeptidomics. It lives alongside ``observations.parquet`` (MS
elution) and ``binding.parquet`` (in-vitro binding) as a third,
strictly non-MHC table so downstream consumers can use it as a
detectability prior without ever conflating it with immunopeptidomics
data.

Data flow:

- **Source CSVs + metadata** ship inside the package under
  ``hitlist/data/bulk_proteomics/`` (always readable, no build needed).
- **``bulk_proteomics.parquet``** is written by
  :func:`hitlist.builder.build_bulk_proteomics` into ``~/.hitlist/`` as
  a long-form table with both protein- and peptide-level rows plus
  per-source acquisition metadata (instrument, digest, fragmentation,
  quantification, …) denormalized onto every row. The acquisition
  column names are harmonized with the per-sample schema used by
  ``observations.parquet`` so the same columns can be extracted from
  either index for joint MS-bias modeling.

Three loaders, each with its own granularity:

- ``load_bulk_proteomics`` — protein-level abundance per cell line.
  Union of CCLE (Nusinow et al. 2020, PMID 31978347; TMT-normalized)
  and Bekker-Jensen (PMID 28591648; label-free sum-of-intensities).
  Use the ``source=`` filter to pick one.

- ``load_bulk_peptides`` — peptide-level detection per cell line
  (Bekker-Jensen et al. 2017). Good for "within this protein, which
  tryptic peptides are ever observable by MS?" (the intra-protein
  detectability bias model).

- ``load_bulk_sources`` — per-source metadata (instrument, digest,
  fractionation, quantification method, cell lines covered). Consult
  before using intensity/abundance values across sources — CCLE is
  TMT-normalized, Bekker-Jensen is label-free; their numbers are not
  directly comparable.

Loaders prefer the built parquet (fast, with full acquisition metadata
columns) and fall back to the packaged CSVs when it has not been built.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import pandas as pd
import yaml

_DATA_MODULE = "hitlist.data.bulk_proteomics"


def bulk_proteomics_path() -> Path:
    """Path to the built ``bulk_proteomics.parquet`` in ``data_dir()``."""
    from .downloads import data_dir

    return data_dir() / "bulk_proteomics.parquet"


def is_bulk_proteomics_built() -> bool:
    """Check whether the bulk proteomics parquet has been built."""
    return bulk_proteomics_path().exists()


@lru_cache(maxsize=1)
def _load_ccle() -> pd.DataFrame:
    path = files(_DATA_MODULE) / "ccle_nusinow_2020.csv.gz"
    df = pd.read_csv(str(path), compression="gzip")
    df["source"] = "CCLE_Nusinow_2020"
    return df


@lru_cache(maxsize=1)
def _load_bj_protein() -> pd.DataFrame:
    path = files(_DATA_MODULE) / "bekker_jensen_2017_protein_abundance.csv.gz"
    return pd.read_csv(str(path), compression="gzip")


@lru_cache(maxsize=1)
def _load_bj() -> pd.DataFrame:
    path = files(_DATA_MODULE) / "bekker_jensen_2017_peptides.csv.gz"
    return pd.read_csv(str(path), compression="gzip")


@lru_cache(maxsize=1)
def _load_sources_yaml() -> list[dict]:
    path = files(_DATA_MODULE) / "sources.yaml"
    data = yaml.safe_load(path.read_text())
    return data.get("sources", [])


def _source_defaults(source_id: str) -> dict:
    """Source-level metadata defaults for a given source_id.

    Used by the CSV-fallback path in :func:`load_bulk_proteomics` when
    ``hitlist data build`` hasn't been run and the CSVs don't carry all
    the axes the parquet would have. Mirrors the stamping the builder
    does in :func:`hitlist.builder.build_bulk_proteomics`.
    """
    for s in _load_sources_yaml():
        if s.get("source_id") == source_id:
            return dict(s)
    return {}


def _load_parquet_or_none() -> pd.DataFrame | None:
    """Return the built parquet if present, else None."""
    p = bulk_proteomics_path()
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _apply_cell_line_filter(df: pd.DataFrame, cell_line) -> pd.DataFrame:
    if cell_line is None:
        return df
    if isinstance(cell_line, str):
        cell_line = [cell_line]
    wanted = {c.casefold() for c in cell_line}
    return df[df["cell_line_name"].str.casefold().isin(wanted)]


def _apply_gene_filter(df: pd.DataFrame, gene_name) -> pd.DataFrame:
    if gene_name is None:
        return df
    if isinstance(gene_name, str):
        gene_name = [gene_name]
    return df[df["gene_symbol"].isin(list(gene_name))]


def _apply_uniprot_filter(df: pd.DataFrame, uniprot_acc) -> pd.DataFrame:
    if uniprot_acc is None:
        return df
    if isinstance(uniprot_acc, str):
        uniprot_acc = [uniprot_acc]
    return df[df["uniprot_acc"].isin(list(uniprot_acc))]


def _apply_enzyme_filter(df: pd.DataFrame, digestion_enzyme) -> pd.DataFrame:
    if digestion_enzyme is None or "digestion_enzyme" not in df.columns:
        return df
    if isinstance(digestion_enzyme, str):
        digestion_enzyme = [digestion_enzyme]
    return df[df["digestion_enzyme"].isin(list(digestion_enzyme))]


def _apply_fractions_filter(df: pd.DataFrame, n_fractions_in_run) -> pd.DataFrame:
    if n_fractions_in_run is None or "n_fractions_in_run" not in df.columns:
        return df
    if isinstance(n_fractions_in_run, int):
        n_fractions_in_run = [n_fractions_in_run]
    return df[df["n_fractions_in_run"].isin(list(n_fractions_in_run))]


def _apply_ph_filter(df: pd.DataFrame, fractionation_ph) -> pd.DataFrame:
    if fractionation_ph is None or "fractionation_ph" not in df.columns:
        return df
    if isinstance(fractionation_ph, (int, float)):
        wanted = [float(fractionation_ph)]
    else:
        wanted = [float(v) for v in fractionation_ph]
    return df[df["fractionation_ph"].isin(wanted)]


# ``enrichment`` filter is slightly special: we accept a sentinel
# "__default__" string so the loader defaults can pass through "match
# only the non-enriched rows" semantics even when the caller explicitly
# passes ``enrichment=None`` to mean "don't filter" (both populations).
_ENRICHMENT_DEFAULT = "__default__"


def _apply_enrichment_filter(df: pd.DataFrame, enrichment) -> pd.DataFrame:
    if enrichment is None or "enrichment" not in df.columns:
        return df
    return df[df["enrichment"] == enrichment]


def load_bulk_proteomics(
    cell_line: str | Iterable[str] | None = None,
    gene_name: str | Iterable[str] | None = None,
    source: str | None = None,
    digestion_enzyme: str | Iterable[str] | None = None,
    n_fractions_in_run: int | Iterable[int] | None = None,
    enrichment: str | None = _ENRICHMENT_DEFAULT,
    fractionation_ph: float | Iterable[float] | None = None,
) -> pd.DataFrame:
    """Protein-level bulk proteomics abundance (shotgun MS, NOT MHC ligands).

    Union of CCLE and Bekker-Jensen protein-level abundance. The two
    sources use different quantification schemes — CCLE is TMT log2-
    normalized relative to the CCLE panel median; Bekker-Jensen is
    label-free log2 sum-of-peptide-intensity. **Intensity values are
    not directly comparable across sources.** Use ``abundance_percentile``
    (rank within cell line) for cross-source comparisons, or filter to
    a single source with ``source=``. See ``load_bulk_sources()`` for
    full per-source metadata.

    Parameters
    ----------
    cell_line
        Filter to a single cell line (e.g. ``"MDA-MB-231"``) or iterable
        of cell lines. Matched case-insensitively.
    gene_name
        Filter to one or more HGNC gene symbols (exact match).
    source
        Restrict to one source (``"CCLE_Nusinow_2020"`` or
        ``"Bekker-Jensen_2017"``). ``None`` returns the union.
    digestion_enzyme
        Filter Bekker-Jensen rows to one or more digestion enzymes
        (exact match against ``digestion_enzyme`` column — canonical
        values: ``"Trypsin/P (cleaves K/R except before P)"``,
        ``"Chymotrypsin"``, ``"GluC"``, ``"LysC"``). CCLE rows are
        unaffected (all CCLE is tryptic).
    n_fractions_in_run
        Filter Bekker-Jensen rows to one or more fractionation depths
        (integer in ``{12, 14, 39, 46, 50, 70}`` — authoritative values
        discovered in PXD004452). CCLE rows are unaffected.
    enrichment
        Filter Bekker-Jensen rows by enrichment: ``"none"`` (baseline,
        default), ``"TiO2"`` (phosphopeptide enrichment), or ``None``
        for both. **Defaults to ``"none"``** so baseline detectability
        queries do not mix in phospho-biased TiO2 rows; callers opt
        into TiO2 explicitly. CCLE is not phospho-enriched so the
        filter is a no-op there.
    fractionation_ph
        Filter by the high-pH reverse-phase SPE fractionation buffer
        pH: ``10.0`` is the default everywhere in the deposit except
        the explicit Bekker-Jensen ``Tryp-Phos-pH8`` TiO2 arm, which is
        stamped ``8.0``. Most queries should leave this unset (default
        ``None`` = all pH values included). Pass a single float or an
        iterable of floats to restrict.

    Returns
    -------
    DataFrame with columns: evidence_kind, granularity, cell_line_name,
    gene_symbol, uniprot_acc, abundance_percentile (0-1 within arm for
    Bekker-Jensen; within cell_line for CCLE), acquisition metadata
    (instrument, fragmentation, labeling, ...), and bulk-specific prep
    fields (digestion, digestion_enzyme, fractionation, n_fractions,
    n_fractions_in_run, enrichment, ...). When the parquet is built,
    columns match :func:`hitlist.export.generate_ms_samples_table` for
    the acquisition fields so the same schema works across indexes.
    """
    # Resolve the enrichment default. "__default__" means "filter to
    # non-enriched rows" (the baseline detectability prior); explicit
    # None means "don't filter" (both populations).
    if enrichment == _ENRICHMENT_DEFAULT:
        enrichment = "none"

    parquet = _load_parquet_or_none()
    if parquet is not None:
        df = parquet[parquet["granularity"] == "protein"].copy()
    else:
        ccle = _load_ccle().copy()
        ccle["abundance_percentile"] = ccle.groupby("cell_line")["abundance_log2_normalized"].rank(
            pct=True
        )
        # CCLE CSV doesn't carry the Fig 1b row-level axes — stamp
        # fractionation_ph from the source-level default in sources.yaml
        # so the column is populated even on the CSV-fallback path
        # (no ``hitlist data build`` required). Mirror the builder's
        # source-level stamping.
        ccle_src = _source_defaults("CCLE_Nusinow_2020")
        if "fractionation_ph" not in ccle.columns:
            ccle["fractionation_ph"] = ccle_src.get("fractionation_ph")
        bj = _load_bj_protein().copy()
        df = pd.concat([ccle, bj], ignore_index=True)
        df = df.rename(columns={"cell_line": "cell_line_name"})

    if source is not None:
        df = df[df["source"] == source]
    df = _apply_cell_line_filter(df, cell_line)
    df = _apply_gene_filter(df, gene_name)
    df = _apply_enzyme_filter(df, digestion_enzyme)
    df = _apply_fractions_filter(df, n_fractions_in_run)
    df = _apply_ph_filter(df, fractionation_ph)
    # For CCLE rows, the `enrichment` column will not be set (or will
    # be empty/NA) — we only apply the enrichment filter to rows that
    # have a populated value, so CCLE passes through untouched.
    if enrichment is not None and "enrichment" in df.columns:
        has_enrich = df["enrichment"].notna() & (df["enrichment"] != "")
        keep_enrich = has_enrich & (df["enrichment"] == enrichment)
        keep_no_enrich = ~has_enrich
        df = df[keep_enrich | keep_no_enrich]
    return df.reset_index(drop=True)


def load_bulk_sources() -> list[dict]:
    """Per-source metadata for the bulk proteomics indices.

    Returns a list of dicts with keys: ``source_id``, ``reference``,
    ``pmid``, ``study_label``, ``species``, ``title``, ``granularity``,
    ``digestion``, ``digestion_enzyme``, ``instrument``, ``fragmentation``,
    ``acquisition_mode``, ``labeling``, ``fractionation``, ``n_fractions``,
    ``quantification``, ``search_engine``, ``fdr``, ``cell_lines_covered``,
    ``note``.

    The acquisition fields (``instrument``, ``fragmentation``,
    ``acquisition_mode``, ``labeling``, ``search_engine``, ``fdr``) use
    the same names as the per-sample schema emitted by
    :func:`hitlist.export.generate_ms_samples_table` for joint MS-bias
    modeling across observations and bulk proteomics.

    Use when building detectability models — the digest enzyme determines
    which peptide bonds are cleavable (affects which theoretical peptides
    could have been observed), and the instrument/fractionation determine
    dynamic range and sensitivity limits.
    """
    return [dict(s) for s in _load_sources_yaml()]


def load_bulk_peptides(
    cell_line: str | Iterable[str] | None = None,
    gene_name: str | Iterable[str] | None = None,
    uniprot_acc: str | Iterable[str] | None = None,
    digestion_enzyme: str | Iterable[str] | None = None,
    n_fractions_in_run: int | Iterable[int] | None = None,
    enrichment: str | None = _ENRICHMENT_DEFAULT,
    fractionation_ph: float | Iterable[float] | None = None,
) -> pd.DataFrame:
    """Peptide-level bulk proteomics detections (shotgun MS, NOT MHC ligands).

    Identifies which peptides *within* a protein were ever observed by
    deep shotgun MS on a given cell line — the intra-protein
    detectability prior for MHC-ligandome analyses. Source: Bekker-Jensen
    et al. 2017 (PMID 28591648) across the full Figure 1b design
    matrix: HeLa across four enzymes (Trypsin/P + Chymotrypsin / GluC /
    LysC), four fractionation depths (14, 39, 46, 70), and ± TiO2
    phospho enrichment; plus the 46-fraction tryptic panel for A549,
    HCT116, HEK293, and MCF7. A peptide is included here if detected
    at non-zero intensity in any replicate of that arm.

    Every row carries four per-row axes: ``digestion_enzyme``,
    ``n_fractions_in_run``, ``enrichment``, and ``modifications``. The
    default call applies ``enrichment="none"`` so baseline detectability
    queries are not contaminated by phospho-biased TiO2 rows — pass
    ``enrichment="TiO2"`` (or ``enrichment=None`` to see both) to opt
    into the phospho arms.

    Parameters
    ----------
    cell_line
        Filter to one or more cell lines (case-insensitive match).
    gene_name
        Filter to one or more HGNC gene symbols (exact match).
    uniprot_acc
        Filter to one or more UniProt accessions (exact match).
    digestion_enzyme
        Filter to one or more canonical enzyme strings — e.g.
        ``"Trypsin/P (cleaves K/R except before P)"``, ``"Chymotrypsin"``,
        ``"GluC"``, ``"LysC"``. Non-tryptic arms are HeLa-only.
    n_fractions_in_run
        Filter to one or more fractionation depths (authoritative set
        discovered in PXD004452: ``{12, 14, 39, 46, 50, 70}`` — 14/39/46/
        70 are the Fig 1b tryptic sweep; 12 and 50 are the two TiO2
        phospho arms).
    enrichment
        ``"none"`` (baseline, the **default**), ``"TiO2"`` (phospho
        enrichment), or ``None`` to include both populations.
    fractionation_ph
        Filter by high-pH SPE buffer pH. Every arm in the deposit is
        at pH 10 except the explicit ``Tryp-Phos-pH8`` TiO2 arm, which
        is stamped ``8.0``. Pass ``8.0`` or ``10.0`` (float) or an
        iterable to restrict; ``None`` (default) returns all pH values.

    Returns
    -------
    DataFrame with columns:
        peptide, cell_line_name, uniprot_acc, gene_symbol, length,
        start_position, end_position, digestion_enzyme,
        n_fractions_in_run, enrichment, fractionation_ph, modifications,
        n_replicates_detected, source, reference (plus acquisition
        metadata when the parquet is built).
    """
    if enrichment == _ENRICHMENT_DEFAULT:
        enrichment = "none"

    parquet = _load_parquet_or_none()
    if parquet is not None:
        df = parquet[parquet["granularity"] == "peptide"].copy()
    else:
        df = _load_bj().copy().rename(columns={"cell_line": "cell_line_name"})

    df = _apply_cell_line_filter(df, cell_line)
    df = _apply_gene_filter(df, gene_name)
    df = _apply_uniprot_filter(df, uniprot_acc)
    df = _apply_enzyme_filter(df, digestion_enzyme)
    df = _apply_fractions_filter(df, n_fractions_in_run)
    df = _apply_enrichment_filter(df, enrichment)
    df = _apply_ph_filter(df, fractionation_ph)
    return df.reset_index(drop=True)


def available_cell_lines() -> list[str]:
    """Return the union of cell lines across all bulk proteomics indices."""
    protein = set(_load_ccle()["cell_line"].unique())
    peptide = set(_load_bj()["cell_line"].unique())
    return sorted(protein | peptide)


def available_protein_cell_lines() -> list[str]:
    """Cell lines covered by the protein-level index (load_bulk_proteomics)."""
    ccle = set(_load_ccle()["cell_line"].unique())
    bj = set(_load_bj_protein()["cell_line"].unique())
    return sorted(ccle | bj)


def available_peptide_cell_lines() -> list[str]:
    """Cell lines covered by the peptide-level index (load_bulk_peptides)."""
    return sorted(_load_bj()["cell_line"].unique().tolist())
