# PR 7 draft — typed numeric supporting measurements (#751)

Prerequisite: PR 6 merged and PyPI 1.82.23 published. Branch from its exact
main commit and bump to 1.82.24. Append a new immutable evidence generation
without modifying or dropping the full PR 6 package. This is the final source
migration before the public 2.0 query/recipe PR #739.

## Source audit and semantic grain

The pinned v4 native `bulk_proteomics.parquet` has 2,269,288 source rows:
2,047,003 peptide detections with no populated numeric quantity, and 222,285
protein abundance rows. The 156,211 Bekker-Jensen protein rows have `log2_intensity` and
`abundance_percentile`; the 66,074 CCLE protein rows have
`abundance_log2_normalized` and `abundance_percentile`. All 222,285
protein rows therefore contribute a percentile cell.
They also have 156,211 `n_peptides` support counts, which are not abundance
values. Source percentiles range from roughly 0 to 1, so preserve that
fractional scale rather than displaying them as 0–100 percentages. Native `line_expression.parquet` has 42,909 gene-profile rows, each
with populated `tpm` and `log2_tpm`; 3,771 rows are exactly zero
on each scale. Preserve those zeros as measured values. The existing typed result projections
retain source row identity and modality, but omit these numeric cells despite
catalog-declared quantitative fields.

Write one typed numeric measurement per populated native value: 444,570 bulk
protein measurements and 85,818 gene-expression measurements. A measurement
has a stable ID, linked typed observation ID and source record ID, native row
index, native source field, numeric value, reported/normalized scale, unit,
value status and provenance. Retain separate records for log2 intensity,
normalized log2 abundance and percentile; likewise TPM and log2 TPM. Preserve
zero, null and inequality status distinctly. Do not create numeric peptide
measurements or promote peptide detection to protein abundance.

## Generation, catalog and verification

- Add typed files and catalog entries `bulk_numeric_measurement` and
  `line_expression_measurement` to a complete verified successor package.
  All original source files, typed relations, receipts and manifests remain
  available and hash-checked. The successor snapshots bytes independently
  from ingestion and predecessor paths, addressing the final-generation
  immutability concern in #748; use copy-on-write clones where available or
  an independent copy fallback. The new manifest names its parent hash, source
  generation and all appended file hashes.
- Field-level source disposition maps each numeric representation to its
  native column, value scale and typed field. Ingest requirements are distinct
  from later task admission requirements. Mark existing typed result
  `quantitative_value` fields as relation-backed rather than direct columns.
- Independently re-read every native numeric cell and compare IDs, values,
  nulls, counts and native-row alignment with the new typed files. Reconcile
  exact per-source and per-representation counts. Verify no predecessor file
  or materialized relation disappeared; no assay, bulk result, expression row,
  source or contributor count changes. Mutating a source file after publication
  must leave the new package and its verifier unchanged.
- Add offline fixtures for multiple representations on one protein row,
  source nulls, zero values, distinct peptide detection and expression scales.

## Completion

Run format, lint, full local tests and final-head CI. Review the exact diff,
merge and deploy from clean main. Verify PyPI 1.82.24 wheel and sdist hashes
before branching PR 8. Link #751 in the PR.
