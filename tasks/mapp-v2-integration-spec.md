# MAPP source and distribution recipes in Hitlist 2.0

Prospective integration specification, 2026-10-23. Main 0dfa170 delivered EvidenceStore and canonical materialize/verify-recipe. Draft758 contains the MAPP adapter/reference evidence; draft760 adds three validated distribution contracts. The existing experiments remain pinned to their original package and code. No targets, training or old receipts are rewritten.

## Design

1. Merge current main into the existing recipe branch, preserving both histories. Consolidate source and recipe changes in PR760 against main once verified; mark draft758 superseded only after its full changes are present. Use next available 2.0 patch version, not the draft 1.82 versions. Preserve unrelated checkouts and shared environment.
2. Add an immutable external-source successor generation to the current schema7 reference package. Schema8 embeds the exact predecessor manifest and copies every predecessor asset with independent inodes; append new registered sources only. Do not rebuild the older external generation and lose newer T-cell, numeric or mouse-reference modalities. Reuse the existing MAPP adapter unchanged unless a verified defect requires a separately recorded correction. Verify predecessor recursively, new sources/adapter versions/rights/typed counts, complete inventory and all hashes. Old schema4 packages remain independently verifiable as historical receipts.
3. EvidenceStore.open supports schema8 and exposes every MAPP typed relation through declared evidence bindings, logical/native fields and named relationships. Scope binding availability to the sources actually present, so older schema6/7 packages remain usable. Preserve quantity-result versus measurement versus publisher median versus run, parent alternatives, direct/transferred evidence, unknown specimen links, and all source assertions. One-to-many relationships must not multiply measurements through unsafe joins. Source registry includes actual source receipts and unresolved statuses.
4. Canonical materialize accepts a registered distribution-task recipe alongside row-wise recipes. Named contracts are immutable and discoverable in the same catalog. The MAPP compiler consumes complete typed relations through EvidenceStore queries, not a second public file-reader API. A distribution example has an observed candidate set and nested target probabilities; do not flatten it into fictitious independent measurements. Preserve parent/disposition/contributor/run relations plus the standard recipe/source/hash/coverage envelope. The common verifier dispatches by explicit output kind/version and validates all required relations; unknown recipes/keys/sources fail before writing. No second public write_training_recipe or verify_training_recipe entry point is retained.
5. Preserve exact existing recipe definitions, all-source grouping/split salt, eligibility, normalization, values and IDs. Common provenance/envelope changes are expected and recorded separately. Full replay must compare every prior parents/examples/dispositions/contributors/run record, not only aggregate counts. Zero and missing statuses remain distinct. Complete-run selection is not a general training filter.

## Verification and artifacts

- Fixtures exercise schema7 parent embedding, source collision/path traversal, changed source bytes/metadata, binding absence/presence, explicit safe joins, every registered recipe and canonical CLI/API verification. Existing row-wise materialization must retain behavior.
- Register actual source materialization/replay under Presto experiments before execution; pin code, dirty diff, manifest paths/hashes and exact environment. Use the verified latest reference package plus the original pinned MAPP assets. Independently verify all predecessor bytes and inode isolation, every MAPP table against the historical package, and every recipe record/target/split against frozen exports. Explain any producer metadata delta. Preserve failures rather than overwriting them.
- Run format.sh, lint.sh, focused tests, normal test.sh with intact memory guard, strict docs and final-head CI. No cloud training. Do not call a guard refusal a passed suite.
- Publish source/query/recipe conservation results and update PR760 and source issue752. Preserve the separate calibration762, precursor757 and processing759 tasks. Follow reviewed merge/clean-main release checks; do not claim deployment before verified package publication.

## Scientific boundary and next experiment

This integration makes existing evidence reachable from the actual current shared interface. It does not add a concentration label, calibration, precursor estimate or machinery effect. Do not refit the inspected MAPP test. After integration, qualify independent known-load response data and controlled processing sources to separate observation response from processing yield/clearance. Future exact and sampled comparisons must share that coherent observation objective and reserve untouched biological evaluation.

## Implementation checkpoints

- [ ] Merge and reconcile version/catalog without losing either source history.
- [ ] Immutable successor package plus canonical query bindings and tests.
- [ ] Registered distribution recipe through canonical materialize/verify and tests.
- [ ] Actual package/recipe admission and independent full conservation replay.
- [ ] Full software checks, report, publication and reviewed release handoff.
