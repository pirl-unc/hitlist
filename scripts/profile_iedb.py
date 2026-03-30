#!/usr/bin/env python3
"""Profile IEDB + CEDAR data quality using hitlist."""

from hitlist.downloads import get_path
from hitlist.scanner import scan

iedb_path = get_path("iedb")
cedar_path = get_path("cedar")

print("=" * 70)
print("HITLIST: IEDB + CEDAR Full Profile")
print("=" * 70)
print(f"IEDB:  {iedb_path}")
print(f"CEDAR: {cedar_path}")
print()

# Full scan -- all rows, both sources, with classification
print("Scanning all rows (this takes a few minutes on ~14 GB)...")
df = scan(
    peptides=None,  # ALL peptides, not just targets
    iedb_path=str(iedb_path),
    cedar_path=str(cedar_path),
    human_only=True,
    hla_only=True,
    classify_source=True,
)

total = len(df)
print(f"\nTotal human HLA-restricted rows (deduplicated by assay IRI): {total:,}")
print()

# ── MHC class distribution ──────────────────────────────────────────────
print("── MHC Class Distribution ──")
class_counts = df["mhc_class"].value_counts()
for cls, count in class_counts.items():
    print(f"  Class {cls}: {count:,} rows ({count/total*100:.1f}%)")
print()

# ── Source classification ───────────────────────────────────────────────
print("── Source Classification ──")
for flag in [
    "src_cancer",
    "src_adjacent_to_tumor",
    "src_activated_apc",
    "src_healthy_tissue",
    "src_healthy_thymus",
    "src_healthy_reproductive",
    "src_cell_line",
    "src_ebv_lcl",
    "src_ex_vivo",
]:
    if flag in df.columns:
        count = df[flag].sum()
        print(f"  {flag:<30s} {count:>10,} ({count/total*100:.1f}%)")
print()

# ── Class I only stats ──────────────────────────────────────────────────
c1 = df[df["mhc_class"] == "I"]
print(f"── Class I Only: {len(c1):,} rows ──")
print()

# Tissue coverage (class I healthy tissue)
healthy_c1 = c1[c1["src_healthy_tissue"]]
if len(healthy_c1) > 0:
    tissue_counts = healthy_c1["source_tissue"].value_counts().head(20)
    print(f"  Healthy tissue coverage (class I): {healthy_c1['source_tissue'].nunique()} tissues")
    for tissue, count in tissue_counts.items():
        print(f"    {tissue:<30s} {count:>8,}")
    print()

# Cancer disease breakdown (class I)
cancer_c1 = c1[c1["src_cancer"]]
print(f"  Cancer rows (class I): {len(cancer_c1):,}")
if len(cancer_c1) > 0:
    disease_counts = cancer_c1["disease"].value_counts().head(15)
    print(f"  Cancer disease coverage: {cancer_c1['disease'].nunique()} diseases")
    pct_with_disease = (cancer_c1["disease"] != "").mean() * 100
    print(f"  Rows with disease annotation: {pct_with_disease:.1f}%")
    print()
    print(f"  Top cancer types:")
    for disease, count in disease_counts.items():
        label = disease if disease else "(empty)"
        print(f"    {label:<35s} {count:>8,} ({count/len(cancer_c1)*100:.1f}%)")
    print()

# Cell line names
if "cell_line_name" in c1.columns:
    cell_lines = c1[c1["cell_line_name"] != ""]["cell_line_name"]
    if len(cell_lines) > 0:
        cl_counts = cell_lines.value_counts().head(15)
        print(f"  Named cell lines: {cell_lines.nunique()}")
        for cl, count in cl_counts.items():
            print(f"    {cl:<30s} {count:>8,}")
        print()

# ── Unique peptides ─────────────────────────────────────────────────────
print(f"── Unique Peptides ──")
print(f"  Total unique peptides: {df['peptide'].nunique():,}")
print(f"  Class I unique peptides: {c1['peptide'].nunique():,}")
c1_cancer = c1[c1["src_cancer"]]
c1_healthy = c1[c1["src_healthy_tissue"]]
print(f"  Class I cancer peptides: {c1_cancer['peptide'].nunique():,}")
print(f"  Class I healthy somatic peptides: {c1_healthy['peptide'].nunique():,}")
overlap = set(c1_cancer["peptide"]) & set(c1_healthy["peptide"])
print(f"  Overlap (cancer AND healthy): {len(overlap):,}")
print()

# ── HLA allele coverage ────────────────────────────────────────────────
print(f"── HLA Allele Coverage (class I) ──")
allele_counts = c1["mhc_restriction"].value_counts().head(20)
print(f"  Unique alleles: {c1['mhc_restriction'].nunique()}")
for allele, count in allele_counts.items():
    print(f"    {allele:<25s} {count:>8,}")
print()

print("Profile complete.")
