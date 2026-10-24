# PXD009360 precursor quantity materialization (issue #802)

## Contract

Extend the standalone `precursor_context` package from schema 20/catalog 25 to
the next version without deleting or rewriting the source-contract relations.
Materialize the deposited MaxQuant tables that carry precursor, peptide and
protein-group evidence from the pinned PXD009360 search archive. Every source
row must remain queryable with its original tab-delimited values, including
blank cells, `NaN`, semicolon-separated contributors, modifications, charge,
match-between-runs, LFQ/intensity columns, identification representations and
protein/peptide/evidence identifiers. No condition, specimen, aliquot or
absolute-abundance inference is allowed.

## Design

- Keep the archive and source receipt as immutable provenance assets.
- Add one typed relation per materialized table, with stable source row IDs,
  source member, source line number, raw row JSON, and typed values for the
  fields needed for queries. Preserve every original column in a deterministic
  `columns_json`/`values_json` pair so columns are never silently dropped.
- Materialize the four requested tables (`evidence`, `peptides`,
  `allPeptides`, `proteinGroups`) and retain the remaining archive members in
  the inventory as source-only assets.
- Use streaming extraction from the archive; do not load the complete archive
  into memory. Verify source member size, row count and a deterministic digest
  of each emitted relation on replay.
- Keep quantity rows as observational assay evidence, not training examples;
  all pairing and biological attribution fields remain explicitly unresolved.
- Expose aliases and catalog entries through `EvidenceStore`, with package
  verification rejecting changed schemas, row counts, source hashes or row
  payloads.

## Implementation checklist

- [x] Bump package/catalog schema versions and versioned revision.
- [x] Add streaming table schemas, extraction and replay verification.
- [x] Extend source registry, family/availability ledgers and catalog entries.
- [x] Add losslessness, source-hash, row-count and `EvidenceStore` tests.
- [x] Update package docs and parent task tracking.
- [x] Run `./format.sh`, `./lint.sh`, and the full bounded `./test.sh`.
- [x] Open PR, pass hosted checks, merge, run clean-main release build, verify
      and upload artifacts to PyPI, then record hashes and close issue #802.

## Review

The focused package/catalog tests pass. The host-level serial full
non-integration gate passes all 3,427 tests with 392 warnings. The sandboxed
attempt had only environment failures (localhost fixture binding and mocked
memory probes), so the required gate was rerun with host-level permissions.

The real PXD009360 replay verifies schema 21/catalog 26 and materializes
352,666 evidence rows, 37,416 peptide rows, 2,028,032 all-peptide rows and
5,390 protein-group rows (2,423,504 quantity rows total). All four relations
start at source row index 1, retain `source_dataset=PXD009360`, and verify the
pinned uncompressed member hashes and row-count ledger.

PR #808 merged as `897c98cbd514b03ea3156733f48e1e2d5354a534`. Clean-main
release workflow `40094966489` passed in 27m1s; its artifacts match that exact
commit and version 2.0.21. PyPI now serves the wheel SHA-256
`e21b3019c6c9798174ded0f4be36e66eed1c621f3b78ceb14b02f26515a75f58` and sdist
SHA-256 `a981791ef83ff85c93aaa45a368307f1e8a9784281f32e4411d24d37ec45650a`.
