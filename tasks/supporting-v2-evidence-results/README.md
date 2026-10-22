# Supporting evidence v2 conversion (1.82.21)

This audit uses the complete, hash-pinned 1.82.20 biological-context package
(`0d16dd1b9fd0708087d9f04df5c4c57c0f3166f48db513d41816c9bda2685e6b`)
and the pinned 11-file supporting-source receipt
[`supporting_v4_assets.json`](../../hitlist/data/evidence_revisions/supporting_v4_assets.json).
The output is one atomic generation containing the entire parent package,
its original manifest, all 11 source assets, nine typed relations and an
exact-hash inventory of 52 files. The audited output manifest is
[`manifest.json`](manifest.json), SHA-256
`70f1cdb2124aea18f1ec4051dfecb58c73c34324dd06cb8d0b225b2b6a9fdb9c`.
It is a receipt, not a substitute for the large package files.

| Source population | Native rows | Typed result or relation rows | Interpretation |
| --- | ---: | ---: | --- |
| Bekker-Jensen peptide | 2,047,003 | 2,047,003 | Observed within search; no peptide intensity |
| Bekker-Jensen protein | 156,211 | 156,211 | Label-free log2 intensity and optional derived percentile |
| CCLE protein | 66,074 | 66,074 | TMT-normalized relative log2 abundance and optional percentile |
| ENCODE GM12878 gene | 23,716 | 23,716 | Source TPM plus derived log2(TPM+1) representation |
| DepMap HAP1 gene | 19,193 | 19,193 | Source log2(TPM+1) plus derived TPM representation |
| Peptide/protein occurrences | 6,205,013 | 6,205,013 | Alternative mappings; never additional assays |

The 2,269,288 bulk source results have 2,491,573 typed measurement
representations. The 42,909 expression results have 85,818 representations;
there are zero transcript results in this v4 generation. The bulk data span
19 observed source/grain/enzyme/fraction/enrichment combinations. Five
source-scoped protocol claims retain the reviewed search specification
separately from the contradictory legacy Trypsin/P display label (#745).
The CCLE search specification remains unreported. The 21 line-anchor claims
include only two lines with measured v4 expression; a line name does not
establish a paired specimen or experimental material.

The 6,205,013 occurrence rows map 1,145,721 eligible unique input peptides.
Among 1,506,664 eligible unique peptides in the three MHC assay partitions,
360,943 have no retained occurrence and receive an explicit `unmapped`
coverage row. The native assay partitions contain 1,506,973 nonempty unique
strings when flagged non-peptide-ligand rows are also counted; those flagged
rows remain native assay evidence. All alternatives, including duplicate-looking
position claims, remain distinct source rows. Mapping coverage has 1,506,664
rows and never changes the MHC assay count.

## Verification

`verify_supporting_evidence_package` checked the parent manifest, every
source and output hash, native and typed schemas, exact file inventory and
count conservation. An independent streaming replay compared every typed
projection against the native tables: 2,269,288 bulk rows across 19 copied
fields, 42,909 expression rows across 11 copied fields, and 6,205,013 mapping
rows across all 12 source fields. It found zero value or row-index mismatches.
Another streaming replay checked the endpoint/quantity representation of
every bulk result and both expression values/statuses for all 42,909 RNA rows.
The source receipt checks each original CSV/YAML file's SHA-256 and byte size.

These are source rows, not independent specimens or donors. Replicate and
fraction counts do not multiply biological observations. The source labels do
not establish exact MHC-to-bulk material pairing. The indexed expression
profiles and source-specific protein scales do not supply normalized training
labels across sources without an explicit later recipe.
