# Training Table Design

## Goal

One flat table where each row is a peptide observation, ready for ML feature extraction:

```
peptide | mhc | copies_per_cell | cell_type | condition | tissue | species |
instrument | instrument_type | acquisition_mode | quantification_method |
fragmentation | mhc_class | is_cancer | is_cell_line | is_mono_allelic |
gene_name | n_flank | c_flank | pmid | study
```

## What exists today

### observations.parquet (from IEDB/CEDAR via builder.py)
Per-peptide rows with: peptide, mhc_restriction, mhc_class, mhc_species, pmid,
source (iedb/cedar), src_cancer, src_healthy, src_cell_line, src_mono_allelic,
+ optional flanking context (gene_name, n_flank, c_flank)

### ms_samples table (from export.py)
Per-sample metadata: sample, mhc, condition, perturbation, instrument,
instrument_type, acquisition_mode, fragmentation, labeling, mhc_class, pmid

### Missing: copies_per_cell
Only available from supplementary data of targeted studies (Stopfer 2020/2021,
Wu 2019, etc.). Not in IEDB. Needs separate ingestion from PRIDE/paper supps.

## Architecture for the join

1. **observations** has pmid + mhc_restriction per peptide
2. **ms_samples** has pmid + mhc (allele list) + metadata per sample
3. Join on pmid; for multi-sample studies, match by mhc_restriction ∈ sample.mhc

For mono-allelic studies: peptide.mhc_restriction == sample.mhc (exact)
For multi-allelic studies: peptide.mhc_restriction ∈ sample.mhc.split() (any match)

## Copies-per-cell ingestion

Separate pipeline: download PRIDE supplementary tables, parse peptide-level
quantification, output standardized CSV with:
  peptide, pmid, sample_id, copies_per_cell, quantification_method

Then LEFT JOIN into the training table.

## Implementation plan

1. Add `generate_training_table()` to export.py that joins observations + ms_samples
2. Add CLI command: `hitlist export training`
3. Add copies_per_cell ingestion for the 5 targeted studies
4. Output as parquet for efficient ML consumption
