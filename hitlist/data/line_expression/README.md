# APM host RNA references

The packaged profiles are independently measured references. A parental
profile is a surrogate for an engineered sample, not evidence that a
knockout or transgene leaves its transcriptome unchanged.

| Host | Measured material | Source | Processing |
| --- | --- | --- | --- |
| HAP1 | DepMap ACH-002475, default RNA PR-QtHaIL | [DepMap 24Q4](https://doi.org/10.25452/figshare.plus.27993248.v1) | Original RSEM protein-coding gene values; invert log2(TPM+1). |
| C1R | Parental Hmy2.CIR (`SRR7065975`, `SAMN08984896`) | [SRP142649 / PMID 30479382](https://pubmed.ncbi.nlm.nih.gov/30479382/) | `kallisto 0.51.1 --rf-stranded` on Ensembl 110 GRCh38 cDNA+ncRNA; summed to 23,217 protein-coding genes (83.4% pseudoaligned). |
| 721.221-Cas9-sgNT | 721.221 Cas9/sgNT controls (`SRR23958328-330`, `n=3`) | [GSE228188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE228188) | `kallisto 0.51.1 --rf-stranded` on Ensembl 110 GRCh38 cDNA+ncRNA; mean across 3 biological replicates (92.0–93.6% pseudoaligned; r=0.982–0.989). |

Parental HAP1 and parental C1R resolve at tier 1; their explicitly named
engineered children (the Shapiro 2025 HAP1 KO panel, PMID 40113210, and
C1R-HLA single-allele transfectants) use the same profiles as tier-2 parent
references. Because GSE228188 measures Cas9/sgNT engineered controls rather
than unmodified parental 721.221, both `721.221` and `721.221-HLA` arms
resolve to `721.221-Cas9-sgNT` at tier 2 (engineered-control surrogate), never
tier 1. All three profiles preserve their protein-coding subset and original
transcriptome denominator (gene TPM sums to ~920k–950k rather than 1e6), and
zero values are retained.

To reproduce with pandas, NumPy and PyYAML installed:

```sh
python scripts/package_apm_host_rna.py \
  --depmap-gene-csv /path/to/OmicsExpressionProteinCodingGenesTPMLogp1.csv \
  --kallisto-dir /path/to/apm-rna-358
```

The recipe refuses any input other than the original 24Q4 gene matrix
(size and MD5), requires exactly one ACH-002475 row, and writes a
deterministic gzip file plus `apm_rna_build_report.yaml`. The report's
`content_sha256` hashes the decompressed CSV, so it is comparable across
zlib builds; `sha256` is the committed file. The CSV carries only value
columns; `build_line_expression` stamps source metadata from `sources.yaml`.
Installing the optional DepMap bundle adds HAP1 transcript rows, never a
second copy of the gene row.

These reference measurements do not establish expression in a particular
immunopeptidomics sample, culture condition or knockout arm.
