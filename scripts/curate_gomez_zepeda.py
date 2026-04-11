#!/usr/bin/env python3
"""Curate Gomez-Zepeda 2024 (PMID 38480730) supplementary data.

Downloads supplementary tables S4-S7 from Nature Communications,
extracts per-cell-line peptide lists with NetMHCpan allele assignments,
and writes supplementary CSVs for the hitlist pipeline.

Usage:
    python scripts/curate_gomez_zepeda.py
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "hitlist" / "data" / "supplementary"

# Allele format conversion: HLA-A0201 → HLA-A*02:01
_ALLELE_RE = re.compile(r"^HLA-([A-C])(\d{2})(\d{2})$")


def _convert_allele(compact: str) -> str:
    m = _ALLELE_RE.match(compact)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    return compact


def _load_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load S4-S7 from /tmp/gomez_zepeda/."""
    base = Path("/tmp/gomez_zepeda")
    s4 = pd.read_csv(base / "MOESM7" / "S4_pep_all_06154.csv", low_memory=False)
    s5 = pd.read_csv(base / "MOESM8" / "S5_mhcpred_all_06154.csv", low_memory=False)
    s6 = pd.read_csv(base / "MOESM9" / "S6_pep_all_06133b.csv", low_memory=False)
    s7 = pd.read_csv(base / "MOESM10" / "S7_mhcpred_all_06133b.csv", low_memory=False)
    # Normalize cell line names
    s5["Cell.Line"] = s5["Cell.Line"].replace({"SKMEL37": "SK-MEL-37"})
    return s4, s5, s6, s7


def _best_allele_per_peptide(predictions: pd.DataFrame, cell_line: str) -> dict[str, str]:
    """For each peptide, find the best allele assignment (lowest EL_Rank among SB/WB)."""
    sub = predictions[predictions["Cell.Line"] == cell_line]
    binders = sub[sub["Binder"].isin(["SB", "WB"])].copy()
    if binders.empty:
        return {}
    # Keep lowest EL_Rank per sequence
    best = binders.sort_values("EL_Rank").drop_duplicates("Sequence", keep="first")
    return dict(zip(best["Sequence"], best["Allele"].map(_convert_allele)))


def _binder_sequences(predictions: pd.DataFrame, cell_line: str) -> set[str]:
    """Set of sequences that are SB or WB for this cell line."""
    sub = predictions[predictions["Cell.Line"] == cell_line]
    return set(sub[sub["Binder"].isin(["SB", "WB"])]["Sequence"])


def _unique_peptides(pep_df: pd.DataFrame, cell_line: str) -> set[str]:
    """Get unique peptide sequences for a cell line, filtered to 8-15aa."""
    sub = pep_df[pep_df["Cell.Line"] == cell_line]
    seqs = sub["Sequence"].unique()
    return {s for s in seqs if 8 <= len(s) <= 15}


def _make_csv(
    peptides: set[str],
    allele_map: dict[str, str],
    binder_set: set[str],
    output_path: Path,
) -> pd.DataFrame:
    """Create a supplementary CSV with contaminant flag."""
    rows = []
    for pep in sorted(peptides):
        allele = allele_map.get(pep, "")
        is_contaminant = pep not in binder_set
        rows.append({
            "peptide": pep,
            "mhc_class": "I",
            "mhc_restriction": allele,
            "is_potential_contaminant": is_contaminant,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    n_binder = (~df["is_potential_contaminant"]).sum()
    n_contam = df["is_potential_contaminant"].sum()
    n_allele = (df["mhc_restriction"] != "").sum()
    print(f"  {output_path.name}: {len(df):,} peptides "
          f"({n_binder:,} binders, {n_contam:,} potential contaminants, "
          f"{n_allele:,} with allele)")
    return df


def main():
    print("Loading supplementary data...")
    s4, s5, s6, s7 = _load_csvs()

    # --- JY: merge S4 (method) + S6 (deep profiling) ---
    print("\n=== JY ===")
    jy_peps = _unique_peptides(s4, "JY") | _unique_peptides(s6, "JY")
    # Merge predictions from both tables (S7 deep profiling has priority)
    jy_alleles_s7 = _best_allele_per_peptide(s7, "JY")
    jy_alleles_s5 = _best_allele_per_peptide(s5, "JY")
    jy_alleles = {**jy_alleles_s5, **jy_alleles_s7}  # S7 overwrites S5
    jy_binders = _binder_sequences(s7, "JY") | _binder_sequences(s5, "JY")
    _make_csv(jy_peps, jy_alleles, jy_binders, DATA_DIR / "gomez_zepeda_2024_jy.csv")

    # --- HeLa ---
    print("\n=== HeLa ===")
    hela_peps = _unique_peptides(s4, "HeLa")
    hela_alleles = _best_allele_per_peptide(s5, "HeLa")
    hela_binders = _binder_sequences(s5, "HeLa")
    _make_csv(hela_peps, hela_alleles, hela_binders, DATA_DIR / "gomez_zepeda_2024_hela.csv")

    # --- SK-MEL-37 ---
    print("\n=== SK-MEL-37 ===")
    skmel_peps = _unique_peptides(s4, "SK-MEL-37")
    skmel_alleles = _best_allele_per_peptide(s5, "SK-MEL-37")
    skmel_binders = _binder_sequences(s5, "SK-MEL-37")
    _make_csv(skmel_peps, skmel_alleles, skmel_binders, DATA_DIR / "gomez_zepeda_2024_skmel37.csv")

    # --- Raji ---
    print("\n=== Raji ===")
    raji_peps = _unique_peptides(s6, "Raji")
    raji_alleles = _best_allele_per_peptide(s7, "Raji")
    raji_binders = _binder_sequences(s7, "Raji")
    _make_csv(raji_peps, raji_alleles, raji_binders, DATA_DIR / "gomez_zepeda_2024_raji.csv")

    # --- Plasma ---
    print("\n=== Plasma ===")
    plasma_peps = _unique_peptides(s4, "Plasma")
    plasma_alleles = _best_allele_per_peptide(s5, "Plasma")
    plasma_binders = _binder_sequences(s5, "Plasma")
    _make_csv(plasma_peps, plasma_alleles, plasma_binders, DATA_DIR / "gomez_zepeda_2024_plasma.csv")

    # Summary
    all_peps = jy_peps | hela_peps | skmel_peps | raji_peps | plasma_peps
    print(f"\n=== Total ===")
    print(f"Unique peptides across all cell lines: {len(all_peps):,}")


if __name__ == "__main__":
    main()
