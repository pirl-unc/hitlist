# PXD009360 precursor aggregation and pairing qualification (issue #803)

## Contract

Register a declared, replayable precursor aggregation recipe over the lossless
schema-21 quantity relations. The recipe may summarize rows by the exact
source-grain keys that are present, but it must never turn MaxQuant LFQ or
intensity into absolute copies-per-cell, intracellular abundance, condition,
specimen or MAPP labels. Every aggregate must retain parent row IDs,
contributor IDs, missing/filter/imputation states and source hashes.

## Design

- Define explicit parent scopes (`source_file`, acquisition label, source row
  and deposited table) and frozen split policy. No run-to-condition,
  run-to-specimen, aliquot or MAPP crosswalk is admitted unless a source-backed
  assertion resolves it.
- Register a recipe that emits a qualification/availability relation rather
  than training examples while all biological pairings remain unresolved.
- Preserve ambiguous or missing contributors as explicit statuses; do not
  silently drop ambiguous protein/peptide/evidence memberships.
- Add a pairing contract with assertion status, evidence basis, scope and
  material-link precision. Source labels and abundance correlation remain
  insufficient evidence.
- Replay the recipe against synthetic and real materialized packages,
  validating deterministic arithmetic, parent coverage, source digest links,
  no duplication, and zero admitted training rows.

## Implementation checklist

- [x] Bump package/catalog schema versions and versioned revision.
- [x] Add typed recipe, qualification and pairing relations and catalog entries.
- [x] Register the recipe in `EvidenceStore` with explicit no-training policy.
- [x] Add arithmetic, ambiguity, missingness, split and replay tests.
- [x] Update docs, child/parent task tracking and issue evidence.
- [x] Run `./format.sh`, `./lint.sh`, and the full bounded `./test.sh`.
- [ ] Open PR, pass hosted checks, merge, run the clean-main release build,
      verify and upload artifacts to PyPI, then close issue #803.

## Review

The real replay of the pinned PXD009360 archive produces one declared recipe,
four source-relation qualification rows and three unresolved pairing
qualification rows over 2,423,504 quantity rows. Qualification counts are
352,666 evidence rows (13,785 missing intensities; 226,685 ambiguous
contributors), 37,416 peptide rows (36,676 ambiguous), 2,028,032 all-peptide
rows (109,513 ambiguous), and 5,390 protein-group rows (5,355 ambiguous).
Every pairing claim remains `source_backed=false`, `training_status=unavailable`
and frozen to source-file scope until a source-backed condition/specimen/aliquot
crosswalk exists. The replay verifier recomputes each relation digest and
qualification count, and reports zero training examples.
