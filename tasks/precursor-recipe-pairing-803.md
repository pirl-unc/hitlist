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
- [x] Open PR, pass hosted checks, merge, run the clean-main release build,
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

PR #810 merged into clean main as `ba4774217e76ebbfd3a11e99b7deeb08e03583b1`.
The merged-main checks passed (tests run `40102292686`, docs run
`40102292729`) and the clean-main release build `40103757775` passed in 20m9s.
The release manifest verifier matched both distributions to that commit and
version 2.0.24. PyPI serves wheel SHA-256
`eaf5f6f9f407dcec32ab8d5c37119d710d4a679950fcc3b74abc55b7123e5110` and sdist
SHA-256 `4cff9737acb1e0921867a0dfb95a3488658ef686fee8914ca0657b890e878400`.
The local `deploy.sh` attempt completed lint and the unit suite but correctly
refused the integration phase twice because the host had about 4 GiB free
against the guarded ~24 GiB requirement; hosted integration and the release
workflow supplied the full gate instead.
