# PXD009360 precursor source contract (#801 / #800)

## Specification

The PXD009360 parallel tryptic proteome is useful context for MAPP fragment
measurements, but its run-to-condition and run-to-specimen maps are not
source-supported. This first child PR admits only hash-pinned source metadata,
archive member inventory, acquisition labels and explicit unresolved-link
claims. It does not create precursor training examples or infer abundance.

The source-contract package uses schema 20/catalog 25 and family
`precursor_context`; child PR #802 promotes it to schema 21/catalog 26. Every
source asset is copied into the standalone package and verified against its
receipt. The v1 package marked quantity tables available-in-source/deferred.
Child PR #802 materializes their rows losslessly while preserving the same
source contract. Family and availability ledgers make both states queryable
instead of hiding them in an external sidecar.

## Plan

- [x] Freeze the project/files, search-archive, parameters and summary receipts.
- [x] Add typed source-asset, acquisition, archive-table and pairing relations.
- [x] Add schema-20/catalog-25 package recognition, EvidenceStore filtering,
      aliases and source registry.
- [x] Preserve unresolved condition/specimen/MAPP links as explicit claims.
- [x] Replay the real local source root.
- [x] Run required local format, lint and full test gates.
- [x] Run hosted gates and merge PR #804 at `6416616a9b496a4706d9bd161242a247ace103d2`.
- [x] Verify release run 40089253494 and deploy 2.0.20 to PyPI.

## Review

The pinned source contains five source assets, a 17-member MaxQuant archive
inventory and nine acquisitions labeled 448–456. The real package replay
verifies schema 20/catalog 25, four family rows, four availability rows, five
assets, nine acquisitions, 17 archive members and three unresolved pairing
claims. It admits zero precursor training rows in this child PR; all quantity
tables are deferred with source-backed availability states. No condition,
specimen, aliquot or MAPP pairing is inferred.

PR #804 merged at `6416616a9b496a4706d9bd161242a247ace103d2`. Release run
`40089253494` passed on clean main. The verified wheel has SHA-256
`bdebbe850890f92cfed5b9925ce5f5f03c38cb6958e574519d02a1f7889ddcaa`; the
sdist has SHA-256
`aabab8b9a0dc30853ca13d52247940d110d05ca5dec5a3a1185d138ab257ede7`.
PyPI's release index reports both files for version 2.0.20.

The host-level serial full non-integration suite passes all 3,427 tests (392
warnings). `./format.sh` and `./lint.sh` pass.
