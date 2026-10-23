# Rebase MAPP evidence onto released Hitlist 2.0.4

Date: 2026-10-23. Base: clean main `0a21661` / PyPI 2.0.4. Target: #760 as
2.0.5, followed by the stacked calibration PR #767 as a later version.

## Conflict and required outcome

The existing MAPP draft predates source-reviewed multimer adjudications. Its
external successor reuses package schema 8, which 2.0.4 assigns to the review
package. Merge current main into the MAPP branch and resolve the collision before
publishing MAPP. The new MAPP successor must accept a verified reference or
adjudication predecessor, preserve every predecessor file and source claim, and
expose only sources present in the package. Existing review-only packages and
earlier draft external packages must remain readable without false relations.

## Implementation

1. Preserve the released source review, verifier, catalog fields, release gate
   and test runner from main. Keep the source-faithful MAPP adapter, 18 typed
   relations and three distribution recipes from #760. Set package schema 9,
   explicitly dispatch historical schema-8 families, and give the MAPP catalog
   a new version. Retain the schema-8 review as an embedded predecessor when
   requested; never replace its `manifests/reference.json` or source assets.
2. Keep calibration implementation and its recipe in #767. The MAPP package
   must not advertise calibration relations that are absent. MAPP-only packages
   may be built from either supported predecessor and show the nine reviews only
   when the review package is the predecessor.
3. Reconcile source rows, typed relations, contributors, source receipt and all
   three recipe data files against the prior MAPP generation. Report counts at
   their actual grains; no new training interpretation is claimed.

## Gates

- Merge main and inspect every conflict against both semantic contracts.
- Add family/parent/registry, tamper and absence tests.
- Run format, lint, normal test script and strict docs; preserve a memory-guard
  refusal if local capacity is insufficient. Hosted full CI must pass.
- Build and independently inspect a real MAPP-only schema-9 package from the
  pinned source assets, including both reference and adjudication predecessor
  paths. Compare the three recipe data inventories and source files to the old
  generation.
- Review the diff, merge #760, release 2.0.5 from clean main and verify PyPI
  hashes. Only then update #767 to the merged base and reserve 2.0.6.
