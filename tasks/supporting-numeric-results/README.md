# Supporting numeric evidence generation

Hitlist 1.82.24 appends two source-faithful numeric measurement relations to
the verified T-cell evidence generation. The new package schema is 6 and the
catalog schema is 7. The 103-file generation at
`/private/tmp/hitlist-v2-numeric-18224-audited` has manifest SHA-256
`560770376121758b3c18679c3bf4db78bc89f9e95cc72785a7314fe35167cb1e`.
Its predecessor manifest SHA-256 is
`4c3275ca6da9a9e1de084fcc1a8d1243d4e67b1e9c3ed4d6307cbcce5c3738e3`.
The [machine-readable manifest](manifest.json) pins all source and typed bytes,
schemas, source-column disposition, row counts and predecessor files.

| Source and grain | Native rows | New numeric measurements | Representation counts |
| --- | ---: | ---: | --- |
| Bekker-Jensen protein | 156,211 | 312,422 | 156,211 log2 intensity; 156,211 abundance percentile |
| CCLE protein | 66,074 | 132,148 | 66,074 normalized log2 abundance; 66,074 abundance percentile |
| Bekker-Jensen peptide detection | 2,047,003 | 0 | Native numeric cells are blank; detection is retained without invented quantity |
| ENCODE GM12878 gene profile | 23,716 | 47,432 | 23,716 TPM; 23,716 source log2 TPM |
| DepMap HAP1 gene profile | 19,193 | 38,386 | 19,193 TPM; 19,193 source log2 TPM |
| **Total** | **2,312,197** | **530,388** | **444,570 bulk protein; 85,818 expression** |

A source row can carry two numeric representations. Those measurements retain
the same source record and typed result identity; they are not extra assays,
specimens, donors or independent samples. All 3,771 HAP1 zero-TPM rows yield
measured zero TPM and source log2 TPM values. The source abundance percentile
is a 0–1 fraction; it is not multiplied by 100. Unit status remains
`unreported` except the source-named TPM column. No expression transform is
recomputed or inferred.

The writer streamed all 2,269,288 native bulk rows and 42,909 native
expression rows alongside their existing typed result rows, checking every
native index, source record, observation ID, dataset, grain, and expression
gene/profile identity. It validated exact source-specific measurement counts,
zero values and the absence of peptide numeric values. A second, independent
streaming replay re-read every populated numeric cell from the original native
Parquet files and compared the typed value, source field, scale, endpoint,
result ID, source ID and row locator. Mismatches were zero. The package verifier
passed for all 103 files and every nested predecessor; no successor file shares
an inode with its corresponding PR6 parent file. An offline fixture also
mutates an ingestion input after snapshotting and confirms the package copy
retains the original bytes (#748).

The numeric relations are registered as `bulk_numeric_measurement` and
`line_expression_measurement`. The existing source result counts, assay counts,
MHC/T-cell labels and mapping relations are unchanged. PR8 will expose these
relations through the unified query and training recipe API.
