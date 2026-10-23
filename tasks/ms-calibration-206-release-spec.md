# Integrate calibration source after the released MAPP generation

Date: 2026-10-23. PR #767 is stacked on #760; this plan applies after #760
merges and Hitlist 2.0.5 is verified on PyPI. It amends the original
`evidence-successor-compat-spec.md` version reservation without changing the
reviewed scientific admission contract.

## Target

Publish the PXD004886 known-load non-MHC MS source and bounded raw-S4 response
recipe as Hitlist 2.0.6. The new package must expose MAPP, calibration and the
nine source-reviewed multimer claims together. Keep all previous MHC, T-cell,
bulk, expression and mouse-reference relations. Distinguish nominal load,
reported raw peak area, normalized response, source curve cells, protocol
context, molecular-form uncertainty and recipe exclusions. No new biological
concentration or detector-transfer claim.

## Sequence

1. Confirm the exact released 2.0.5 main commit and the live #767 head before
   modifying the draft. Merge main into the calibration branch, retaining the
   reviewed MAPP 2.0.5 interface and source contracts. Resolve any crisscross
   merge-base conflict by comparing source files and typed relation contracts,
   never by discarding a generation.
2. Bump Python package version to 2.0.6 and keep external package schema 9,
   catalog 12. Use `predecessor_package` consistently for the writer argument
   and documentation because both reference and adjudication predecessors are
   valid. Correct the calibration guide's stale reference-only instructions.
3. Open historical schema-7, both historical schema-8 families and the new
   schema-9 package. Build a real 2.0.6 combined successor from the verified
   adjudication input and pinned MAPP/PXD004886 assets. Verify source snapshots,
   all nine review claims, both source registries and every physical binding.
4. Independently compare all earlier MAPP typed relation rows and all three
   distribution-recipe semantic files to their frozen 2.0.5 counterparts.
   Check PXD004886's 13 typed relations and raw-S4 recipe against the prior
   verified calibration generation; explain any provenance-envelope changes.
5. Run focused regressions, `./format.sh`, `./lint.sh`, normal `./test.sh`,
   strict docs, wheel/sdist build, Twine and file-inventory checks. Update PR
   title/body and release evidence, pass final-head hosted CI, review, merge,
   then run the clean-main release gate and verify exact PyPI hashes.

## Review gates

- [ ] Released 2.0.5 base and current draft head confirmed.
- [ ] Merge, version and terminology changes reviewed.
- [ ] Historical and real combined packages and all four recipes replayed.
- [ ] Local required gates and final-head hosted CI pass.
- [ ] PR merged and exact 2.0.6 PyPI artifacts verified.
