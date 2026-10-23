# Calibration successor against released MAPP 2.0.5

PR #767's 2.0.6 candidate appends pinned PXD004886 and PXD009092 assets to the
verified 2.0.4 adjudication predecessor. Its schema-9/catalog-12 package
exposes both sources, the nine source-reviewed multimer claims and all earlier
MHC, T-cell, numeric and reference relations. It does not claim new source
measurements, a changed training target or a calibrated biological concentration.

## Frozen comparison roots

| Package | Top-level manifest SHA-256 | Role |
| --- | --- | --- |
| MAPP plus adjudications, built with released 2.0.5 code | `0206334cf7435051fc2ee019c7679d50445df4bd7d87264d36c8baaa23d30fbf` | Historical catalog 11 reader check and MAPP logical rows |
| Prior verified MAPP plus calibration | `42317a5efb7d642be5b8265131e6a6f7963e469adda946b780dac4d6aebb05b0` | Source-file and recipe payload baseline |
| New 2.0.6 combined candidate | `d53bdb1706e26dfad8c09144be29f3ea2e7a4ac18bcea991556d5c16ae302d1b` | Reviewed successor |

The 2.0.6 candidate was produced from the verified adjudication package and
the pinned PXD009092/PXD004886 asset roots with `write_external_successor`.
Both source adapters independently verify their source receipts and typed
relations during production and on `EvidenceStore.open`.

## Conservation and query audit

The candidate contains 183 hashed files: 126 adjudication predecessor files,
its embedded manifest, 33 MAPP files and 23 calibration files. All 126 parent
files match the original SHA-256 receipts and occupy different inodes. The
embedded adjudication manifest is byte-identical. Every one of the 183 listed
file digests matches the prior verified combined package, including every
PXD004886 file. The two top-level manifests differ only in `hitlist_version`
(`2.0.5` versus `2.0.6`); no source file was rewritten.

The released MAPP-only candidate uses a different Parquet encoding for its 18
typed MAPP files than the prior combined candidate. A separate streaming check
compared Arrow schema, row order and canonical values in 4,096-row batches and
matched all **1,829,008 mixed-grain logical rows**. Non-Parquet MAPP source
files retain exact SHA-256 hashes. This total combines different relation
grains and must not be read as an assay, specimen, peptide or donor count.

`EvidenceStore.open` reads the released schema-9/catalog-11 MAPP package under
the 2.0.6 code: MAPP and review relations are materialized, while calibration
is `not_in_generation`. A new regression also accepts this historical package
version and rejects unknown future catalog versions. The new combined package
opens with 85 materialized types, three contract-only types and zero absent
types. All 33 pinned baselines match. Its source registry includes both
PXD009092 and PXD004886 alongside the reviewed multimer source. Direct rows
include 10,673 MAPP peptides, 6,138 S4 calibration measurements across raw
and normalized representations, and nine
review claims.

## Registered recipe replay

All four recipes were materialized anew from the combined package and verified
against its source manifest. Every frozen semantic payload listed below is
byte-identical to the prior combined generation. Only the source-generation
envelope and manifest binding changed.

| Recipe | Compared semantic payloads | Changed |
| --- | ---: | ---: |
| `mapp_captured_composition_v1` | examples, contributors, dispositions, parents | 0 |
| `mapp_captured_composition_min2_v1` | examples, contributors, dispositions, parents | 0 |
| `mapp_complete_run_composition_v1` | preceding four plus six-run observations | 0 |
| `ms_calibration_raw_s4_nominal_response_v1` | source tables, examples, contributors, dispositions, summary, contract | 0 |

The MAPP targets are relative captured signal conditional on observed sets;
the calibration targets are reported raw MS peak-area responses under nominal
loading. Neither endpoint is an intracellular concentration, a true zero for
an unreported peptide, nor a verified detector transfer to MAPP. Source
limitations and exact scientific admission remain in the source and recipe
specifications; this release integration preserves their reported meaning.

## Software gates

Merge-resolution focused tests: 73 passed. Format, lint and strict docs pass.
Normal guarded `./test.sh` passes 3,319 tests with 84% total coverage (10
expected warnings). Its first attempt refused at 0.26 GB free against a 2.5 GB
minimum; after the unrelated large process exited, the unchanged guard
admitted four workers with 10.99 GB free. The 2.0.6 wheel and sdist build,
pass Twine and license checks, and each contain the calibration implementation
and pinned evidence assets. Final-head hosted CI, merge and PyPI publication
are tracked on PR #767 and in `tasks/todo.md`.
