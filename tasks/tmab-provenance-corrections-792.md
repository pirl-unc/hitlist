# TMAB reuse, identity and protocol provenance corrections (#792)

## Specification

The independent audit of the schema-15 package found semantic labeling defects,
not source-value corruption. This successor keeps all four hash-pinned
workbooks, source rows and ratio cells byte-for-byte while separating candidate
reuse, tentative identification, modification annotations, publishing-study
protocol and original acquisition provenance.

## Plan

- [x] Derive source-independent candidate reuse keys from cell, experiment,
      sequence text, charge, observed mass and the first five ratio values;
      retain source-specific IDs and unknown specimen independence.
- [x] Parse tentative parentheses and known/unresolved modifications without
      converting annotation text into biological sequence; defer tentative or
      unresolved identities from confident sequence recipes.
- [x] Correct source-specific instrument metadata and expose publishing-study
      versus inherited-acquisition scope.
- [x] Preserve exposure and wash/harvest intervals as separate raw assertions,
      including the source-supported inhibitor-presence scope.
- [x] Correct Gelman heatmap sequence projection to column F and add regression
      tests for all cited real examples plus a two-source reuse fixture.
- [x] Bump schema/catalog and version and run format/lint/full tests.
- [ ] Pass hosted gates, merge, and publish 2.0.16.

## Review

The local replay preserves 10,564 source rows and 61,908 measurement slots;
candidate reuse is conservative and never asserts specimen independence.
Hosted review and release evidence are pending.
