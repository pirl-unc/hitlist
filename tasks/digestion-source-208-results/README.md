# Soh 2024 purified-proteasome source admission (Hitlist 2.0.8 candidate)

This is the source-only review for #774, under the broader quantitative-digestion request #770. The prospective grouped kinetic recipe and all-admitted-substrate arithmetic are #775. Original source: [Soh et al. 2024](https://doi.org/10.1038/s41467-024-45339-3), MassIVE MSV000092631 / PXD044463 (CC0). The [pinned independent source audit](https://github.com/pirl-unc/presto/blob/1503d67/experiments/2026-10-23_1018_codex_shared-digestion-source/README.md) documents original recovery, RData decoding and earlier source verification.

## Package and conservation

Real candidate: `/private/tmp/hitlist-v2-digestion-208-unit-candidate`, version 2.0.8, package schema 9/catalog 14, manifest SHA-256 `ad62414d511b74a6aa1e4da457fdff68d9f4527874d1a2e385c65e5b98df41e8`. The package contains 251 hashed files, including the 18 original source assets (processed archive, identifier workbook, methods document and 15 literal FASTAs). All 218 predecessor files from `/private/tmp/hitlist-v2-controlled-207-reviewed-candidate` have identical relative paths and SHA-256/size receipts, with zero shared inodes. Four sources coexist in one verified generation.

| Relation | Original grain | Rows |
| --- | --- | ---: |
| `digestion_members` | nested archive member | 391 |
| `digestion_metadata` | workbook row | 442 |
| `digestion_constructs` | deposited construct | 15 |
| `digestion_runs` | main-search acquisition | 388 |
| `digestion_contexts` | purified digestion condition | 17 |
| `digestion_identifications` | assignment row | 649,789 |
| `digestion_ms1` | Skyline row | 5,792,969 |
| `digestion_raw_quantities` | peptide/run quantity | 545,664 |
| `digestion_filtered_vectors` | technical vector | 121,556 |
| `digestion_final_vectors` | biological vector or annotation | 53,413 |
| `digestion_final_values` | reported vector slot | 280,664 |
| `digestion_conflicts` | source claim disagreement | 2,284 |

Rows above have different grains and must not be added as independent biological observations. The 391-member inventory has 85 typed source tables, 272 retained original-only files and 34 directory entries. The original-only inventory includes logs, TICs, plots, assignment RData and other assets; their bytes stay inside the pinned archive.

## Independent replay

`python scripts/verify_digestion_roundtrip.py PACKAGE ORIGINAL_ASSETS DECODED_RDATA` imports no Hitlist modules. Against the separately decoded RData exports from the source audit, it checks original nested member metadata/CRC, every row and source locator in the five typed source layers, all 442 workbook rows and column names, all 15 FASTA sequences/hashes, acquisition/instrument/time joins, MS1 and raw numeric values/statuses, and every reported final vector slot. The canonical `EvidenceStore` opens the full package and queries the new member, instrument, product, raw-quantity and final-slot aliases. The exact [round-trip result](roundtrip.json) passed; SHA-256 `94c93545d2161be23a75ff59ebe85a70830b0458c93257ae3d0ba5cb8d5168c0`.

A real `EvidenceStore.query("digestion_raw_quantities", joins=["acquisition", "condition"])`
returned source-backed instrument, acquisition time, purified-enzyme species
and input substrate concentration at the raw-quantity grain. This verifies
the named joins without projecting a run into a biological specimen.

The source has 388 main-search acquisitions (316 Q Exactive HF-X; 72 Orbitrap Fusion), 15 constructs in 17 conditions, and three Ffh ratios sharing one construct. Substrate species is directly specified in four conditions and unresolved in 13; purified human 20S from HeLa does not set the substrate species or create an APC specimen. All 2,278 RF1 quantity-time versus workbook/filename-time disagreements remain source conflicts, along with six contaminant-enzyme metadata disagreements. The 152 Skyline rows with blank source filenames remain unlinked and are explicitly labeled. Final vectors retain 53,144 reported rows plus 269 annotation-only rows; their 280,664 slots include 213,022 positive signals, 62,860 forced time-zero values and 4,782 missing values. None is converted to a negative assay label.

Skyline peak area, aSPIRE raw peptide/run quantity and background-adjusted biological mean are distinct `quantity_scale` values in source MS1 peak-area arbitrary units. The article states these label-free MS1 intensities had no further normalization; they are neither calibrated peptide amounts nor intracellular concentration.

## Verification and limits

Format, lint, 37 focused source/catalog/successor tests and strict documentation build passed after the review changes. The initial local `./test.sh` attempt was safely refused by its unchanged memory guard at 0.68 GiB free versus the 2.5 GiB one-worker requirement; rerun or hosted full-suite/final-head CI is required before merge. The first draft PR head passed hosted Python 3.10–3.14 unit jobs; final-head CI is still required for these review changes.

This source package has no kinetic training recipe. RF1 time identity, all-substrate preprocessing arithmetic, source product-selection/mapping, replicate temporal signal, grouped splits and an observation likelihood remain #775. Purified standard-proteasome products do not label TAP, APC clearance, MHC binding or molar intracellular abundance.
