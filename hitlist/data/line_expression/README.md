# APM host RNA references

The packaged profiles are independently measured references. A parental
profile is a surrogate for an engineered sample, not evidence that a
knockout or transgene leaves its transcriptome unchanged.

| Host | Measured material | Source | Processing |
| --- | --- | --- | --- |
| HAP1 | DepMap ACH-002475, default RNA PR-QtHaIL | [DepMap 24Q4](https://doi.org/10.25452/figshare.plus.27993248.v1) | Original RSEM protein-coding gene values; invert log2(TPM+1). |

Parental HAP1 resolves at tier 1; its explicitly named engineered children
(the Shapiro 2025 KO panel, PMID 40113210) use the same profile as a tier-2
parent reference. HAP1 preserves the release's protein-coding subset and
original denominator, so gene TPM sums to about 950,000 rather than one
million, and zero values are retained.

C1R and 721.221 do not yet have measured references (#358). They keep their
family surrogate until reproducible quantifications land.

To reproduce with pandas, NumPy and PyYAML installed:

```sh
python scripts/package_apm_host_rna.py \
  --depmap-gene-csv /path/to/OmicsExpressionProteinCodingGenesTPMLogp1.csv
```

The recipe refuses any input other than the original 24Q4 gene matrix
(size and MD5), requires exactly one complete 19,193-gene ACH-002475 row,
and writes a deterministic gzip file plus `apm_rna_build_report.yaml`.

These reference measurements do not establish expression in a particular
immunopeptidomics sample, culture condition or knockout arm.
