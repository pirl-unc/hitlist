# Release controlled MAPP evidence after calibration

Date: 2026-10-23. Owner: #759 and PR #772. This is the release plan for the
existing controlled-panel candidate, not a new source-selection exercise.

## Baseline and version

Reconcile this branch with the exact merged PR #767 main commit before changing
code. Keep PR #772 draft until Hitlist 2.0.6 appears on PyPI. Reserve 2.0.7
for the next PR. Continue to read the released schema-9/catalog-11 MAPP package and
schema-9/catalog-12 calibration package; write catalog 13 only for new
controlled-panel successors. Do not use a catalog bump to reinterpret any
previous relation or recipe.

## Source and recipe scope

Retain both verified original processed panels: PXD009508 and PXD007263. The
canonical package must expose all 9,206 peptide rows, 46,777 evidence rows,
30 acquisitions and 146,832 peptide/run quantities, including source zeros,
missing intensities, reverse/contaminant records and reference alternatives.
All 37 and 25 original summary rows stay at their source grain; they do not
become 62 acquisitions. Preserve the full native names and the unique derived
controls aliases, without creating specimen or biological pairing IDs.

Release `mapp_controlled_run_intensity_v1` and the separately declared
`mapp_controlled_complete_alpha6_response_v1` through the same materialize,
verify and CLI entry points. The contrast retains all source and base-run rows,
all contrast dispositions, six original intensity values and measurement IDs,
and the two arithmetic arm means. It admits only complete positive alpha6
measurements under the frozen direct-MS/MS rule. It does not convert unknown or
zero values to biological negatives, assign a gene knockout, or claim molar
concentration. The time-course literal TNF label and paper-level combined
cytokine assertion remain distinct.

## Verification

1. Build from the verified 2.0.6 combined predecessor with the two original
   source roots. Check every predecessor file SHA and separate inode, both
   source receipt digests, all native relation row identities, source metadata,
   and all 146,832 quantity/contributor links against independent source data.
2. Compare every old catalog relation and the four existing recipe semantic
   payloads against their 2.0.6 values; accept byte-level Parquet encoding
   changes only after a logical row/schema comparison. Audit source registry,
   binding presence and all 33 pinned baselines.
3. Materialize and independently replay both controlled recipes. Reproduce
   source-table and base-run outputs, every disposition, the six-run admission
   count and train/validation/test memberships without importing the producer.
   Check tamper rejection and old-package readers.
4. Run `./format.sh`, `./lint.sh`, `./test.sh`, strict MkDocs, wheel/sdist build,
   Twine, exact license and packaged-file checks. If the normal memory guard
   refuses, record the refusal and rely on hosted full CI without weakening it.
5. Update PR #772 against main with its final head, review all hosted checks,
   merge, and publish exactly the CI-built 2.0.7 artifacts from clean main.
   Verify PyPI hashes and record unresolved source metadata under #759.

## Scientific limits

The response is selected on positive observed signal in both arms and is a
captured-product measurement conditional on recovery and MS response. The
reported repeatability check is acquisition-level evidence on training
components. It does not establish independent biological replication,
specimen pairing, intracellular free peptide concentration, or a validated
sequence-model improvement. Keep source admission and any model evaluation
separate in the report.
