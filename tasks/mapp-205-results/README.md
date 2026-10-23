# MAPP successor reconciliation against released Hitlist 2.0.4

The #760 merge candidate appends pinned PXD009092 MAPP assets to either the
verified 2.0.4 adjudication package or its verified schema-7 reference parent.
Both outputs use package schema 9 and catalog 11. This report compares source
records and recipe data; it does not assert a new biological target or model fit.

## Frozen inputs and outputs

| Package | Top-level manifest SHA-256 | Parent |
| --- | --- | --- |
| Released adjudication input | `8f35dc5529b376f1b2e3b25f91dd2f2dc57d0a53c4d8a76c094294193708c9b6` | schema 7 reference |
| MAPP plus adjudications | `0206334cf7435051fc2ee019c7679d50445df4bd7d87264d36c8baaa23d30fbf` | exact adjudication manifest |
| MAPP plus reference | `7d73a5e9ef4fad02553fbab1475bebe9cf958b96f190f5d8de658a9da24777f6` | exact reference manifest |
| Prior MAPP comparison package | `83a21da963d7d938c68a9570d7b868e578107302ba466ac55df09c839309b693` | historical external successor |

The input MAPP source assets came from the verified prior package's `assets/`
directory and were checked against the pinned `mapp_assets.json` receipt by the
new writer. Independent SHA-256 checks found all 126 adjudication-parent files
and all 123 reference-parent files unchanged in their respective successors;
every copied file has a different inode. Each embedded predecessor manifest is
byte-identical to its original top-level manifest. The outputs were written
atomically into new directories; predecessors were not modified.

The adjudication-based output opens through `EvidenceStore` with 72 materialized
relations. All 33 pinned baselines match; its registry includes PXD009092 and the
nine source-reviewed multimer claims. It has 10,673 MAPP peptide rows. The
reference-based output opens with the MAPP relation and reports the review
relation as `not_in_generation`; it does not advertise the nine absent reviews.
Both historical schema-8 package families remain readable under explicit
family dispatch.

## Independent source and recipe replay

All 18 PXD009092 typed Parquet relations have the same schema, row order and
canonical row values as the prior MAPP package. The comparison streamed each
Parquet file in 4,096-row batches, hashed sorted-key JSON for every row, and
matched **1,829,008 rows across distinct grains**. This total must not be
interpreted as assay, peptide, sample or donor count.

Three registered distribution recipes materialized and verified against the new
source package. The source manifest in each output changed as intended. Every
shared semantic file (`examples`, `contributors`, peptide dispositions, parents,
and complete-run observations where applicable) is byte-identical to the prior
Presto output:

| Recipe | Admitted peptides | Examples | Changed semantic files |
| --- | ---: | ---: | ---: |
| `mapp_captured_composition_v1` | 544 | 130 | 0 |
| `mapp_captured_composition_min2_v1` | 616 | 202 | 0 |
| `mapp_complete_run_composition_v1` | 369 | 142 | 0 |

The complete-run recipe retains 3,696 per-run observations. Parent and
contributor identities, admission/exclusion reasons, and target values are
unchanged. These are relative captured-fragment distributions conditional on
observed candidate sets; they do not identify intracellular concentration,
precursor abundance, independent specimens or unobserved negatives.

## Reproduction

Use `hitlist evidence append-sources` with the schema-7 reference or schema-8
adjudication package and `--source-root pxd009092=ASSETS`. Then open the new
package with `EvidenceStore.open`, run `store.audit()`, and materialize the three
registered recipe IDs above. The package verifier checks all source receipts,
typed relation counts, predecessor manifests and file hashes. The independent
comparisons in this report also checked physical inode isolation, logical
Parquet rows and recipe output bytes against the prior generations. Full project
and hosted CI results are tracked on PR #760.
