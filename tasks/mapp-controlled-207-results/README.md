# Controlled MAPP 2.0.7 release candidate

Date: 2026-10-23. Issue #759; draft PR #772. Code: `3b05cb3` plus the
independent verifier and report in this branch. This is source and recipe
qualification, not a sequence-model performance claim.

## Inputs and immutable generation

The successor was built from the verified adjudication predecessor at
`/private/tmp/hitlist-v2-adjudication-204-reviewed` and these three pinned
caller-supplied roots:

- PXD009092: `/private/tmp/hitlist-v2-combined-205-canonical/external/pxd009092/assets`
- PXD004886: `/private/tmp/hitlist-v2-combined-205-canonical/external/pxd004886/assets`
- PXD009508 and PXD007263: `/private/tmp/hitlist-mapp-controlled-assets`

New package:
`/private/tmp/hitlist-v2-controlled-207-reviewed-candidate`, schema 9,
catalog 13, Hitlist 2.0.7. Its manifest SHA-256 is
`48b211da703275fb97f5a7ad472ab7656a13d719460b7d9b901ed80c919a74f2`.
All 183 hashed files from the verified 2.0.6 combined generation have identical
manifest digests in the 218-file successor and occupy separate inodes. This
includes all 126 adjudication predecessor files and nine review claims. The
35 new files belong to the controlled source. Source workbooks and processed
archives are not included in the wheel or sdist.

## Source coverage and query semantics

| Deposited panel | Peptides | Evidence rows | Acquisitions | Peptide/run results | Original summary rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| PXD009508 controls | 6,060 | 19,581 | 18 | 109,080 | 37 |
| PXD007263 time course | 3,146 | 27,196 | 12 | 37,752 | 25 |

The full original-record relation has 57,954 rows, including the 62 summary
rows above at their own grain; only 30 rows name actual acquisitions. All
146,832 source results and 193,609 contributor edges are retained. All 900
unreported evidence intensities remain missing; source zeros are not
biological negatives. The 30 complete-label run aliases preserve their
literal names and derived matching rule. Reference alternatives remain linked
claims, never duplicated measurements.

`EvidenceStore.open` reads the released catalog-11 MAPP and catalog-12
calibration packages under the new code. Their unavailable sources report
`not_in_generation`. In the new package MAPP, calibration and controlled
processing are all materialized. The audit reports 102 materialized evidence
types, three contract-only types, zero absent types and all 33 baselines
matched.

## Independent source and recipe replay

The copied, path-adjusted [Presto verifier](https://github.com/pirl-unc/presto/blob/80ffa7e/experiments/2026-10-23_0645_codex_controlled-mapp-signal/code/verify.py)
imports no Hitlist code. It passed on the new package and run export: all 218
file inventories, the 183 predecessor files, 17 canonical queries with
1,098,915 mixed-grain rows, all original source fields and quantities, complete
reference occurrence scans, every contributor, all admission/exclusion
decisions and split assignments. The run recipe admits 21,607 positive
observations and retains 125,225 excluded results. No held-out aggregate
response metric was evaluated by this verification.

The independently runnable
[`scripts/verify_mapp_controlled_contrast.py`](../../scripts/verify_mapp_controlled_contrast.py)
compares the contrast to the source-verified run export and original peptide
IDs without importing the Hitlist producer. It passed on all 9,206 peptide
decisions, 87 complete-case responses (59 train, 14 validation, 14 test),
9,119 exclusions and 522 source measurement IDs. Each admitted response
retains six reported values and their arithmetic arm means. Four embedded
base-run payloads are byte-identical to the standalone run export.

All four previously registered recipe semantic payload sets were rebuilt from
the 2.0.7 package and match their verified 2.0.6 outputs byte for byte:
`mapp_captured_composition_v1`, `mapp_captured_composition_min2_v1`,
`mapp_complete_run_composition_v1` and
`ms_calibration_raw_s4_nominal_response_v1`. Their source-generation manifests
correctly differ because the new package adds a source.

## Software gates and limits

`./format.sh`, `./lint.sh`, strict MkDocs and 63 focused tests pass. The normal
local `./test.sh` first refused before collection at 0.16 GiB available versus
its unchanged 2.5 GiB minimum. The 2.0.7 wheel and sdist build with the
controlled implementation and pinned assets, and pass Twine and exact license
checks. An initial `uv build` attempt could not open its external user cache
inside the sandbox; `python -m build --no-isolation` built the same source tree
without changing any package dependency. The full-suite retry, final-head
hosted CI, merge and exact PyPI publication remain gates.

The six-run response selects peptides with positive captured signal in both
arms. It is an acquisition-level comparison, not a source-backed biological
specimen pair or an intracellular concentration. The original 2016 search
FASTA, exact alpha6 gene assignment, time-course cytokine scope and
reporter/Velcade run data remain unresolved under #759.
