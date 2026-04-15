# Per-sample MHC-allele curation audit

## Problem

Many `ms_samples` entries in `pmid_overrides.yaml` store a single sample
row whose `mhc:` field is the **union of alleles across multiple
donors/transfectants**, not a per-sample genotype.  The sample-metadata
join in `generate_observations_table` treats that `mhc:` as one sample's
6-locus HLA type and ends up attributing peptides against dozens of
pooled alleles — a meaningless operation.

Limits for a real per-sample class-I genotype: **≤ 6 alleles**
(heterozygous A/B/C).  Class II: **≤ 10 alleles** (DRB1/3/4/5 +
DPA/DPB + DQA/DQB heterozygous pairs).  Anything above is a pooled
union.

## Audit results (pre-1.7.3 scan)

**Category 1 — Mono-allelic transfectant papers** (one transfectant
per listed allele; easy mechanical split):

| PMID | Study | Sample label | n | alleles | Status |
|---|---|---|---:|---:|---|
| 31844290 | Sarkizova 2020 | 721.221 transfectants | 95 | 95 | **FIXED in 1.7.3** |
| 28228285 | Abelin 2017 | 721.221 transfectants | 16 | 16 | **FIXED in 1.7.3** |
| 28904123 | Di Marco 2017 | C1R HLA-C transfectants | 15 | 15 | **FIXED in 1.7.3** |
| 25418920 | Schittenhelm 2015 | C1R B*27 transfectants | 8 | 8 | **FIXED in 1.7.3** |
| 31495665 | Abelin 2019 MAPTAC | class I mono-allelic | 8 | 8 | **FIXED in 1.7.3** |

Total: **5 pooled entries → 142 per-transfectant entries** in 1.7.3.

Previously, every peptide in these studies was joined to a 95-allele
(or 16/15/8) "sample" that encompassed all transfectants — the sample
metadata was useless for per-allele attribution.  Now each peptide's
IEDB `mhc_restriction` correctly joins to the single transfectant
sample with that allele.

**Category 2 — Multi-donor pooled tissue samples** (need per-donor
HLA typings from paper supplements; not fixed in this PR):

| PMID | Study | Sample | donors | pooled alleles |
|---|---|---|---:|---:|
| 33858848 | Marcu 2021 HLA Ligand Atlas | autopsy tissue (I) | 14 | 40 |
| 33858848 | Marcu 2021 HLA Ligand Atlas | autopsy tissue (II) | 14 | 22 |
| 33858848 | Marcu 2021 HLA Ligand Atlas | living donor thymus (I) | 5 | 24 |
| 33858848 | Marcu 2021 HLA Ligand Atlas | living donor ovary (I) | 2 | 9 |
| 35580925 | Khazan-Kost pleural effusion | cancer | 9 | 28 |
| 35580925 | Khazan-Kost pleural effusion | non-malignant | 5 | 22 |
| 32157095 | Chong 2020 | melanoma cell lines | 7 | 25 |
| 32157095 | Chong 2020 | lung tumor | 2 | 11 |
| 32157095 | Chong 2020 | matched normal lung | 2 | 11 |
| 33592498 | Bassani-Sternberg GBM | parental (class I) | 3 | 17 |
| 33592498 | Bassani-Sternberg GBM | CIITA-transduced | 3 | 17 |

Fix path: read each paper's Supplementary Table for per-donor HLA
genotypes and split each pooled sample entry into N donor-level
entries.  Tackle in follow-up PRs, smallest-first (Bassani-Sternberg
n=3, Chong n=2 — easier than Marcu n=14).

## Verification against built 1.7.2 index

Rebuilt `hitlist data build --force` on 1.7.2:

- MS `observations.parquet`: **4,053,693 rows** (2,334,515 gene-annotated, 57.6%)
- `binding.parquet`: **895,785 rows** (106,366 gene-annotated, 11.9%)
- `peptide_mappings.parquet`: 687,115 rows / 554,471 unique peptides / 305 proteomes
- `serotypes` column present ✓

Mono-allelic fix from PR #48 verified:

| PMID | rows | is_monoallelic=True | de-flagged |
|---|---:|---:|---:|
| All 11 non-Sarkizova mono PMIDs | (unchanged) | (all True) | 0 |
| 31844290 (Sarkizova) | 256,262 | 219,909 | **36,353** |

The de-flagged count matches the "Glial cell + Other + PBMC = 36,353 class-only rows" prediction exactly.
