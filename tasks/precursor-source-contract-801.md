# PXD009360 precursor source contract (#801 / #800)

## Specification

The PXD009360 parallel tryptic proteome is useful context for MAPP fragment
measurements, but its run-to-condition and run-to-specimen maps are not
source-supported. This first child PR admits only hash-pinned source metadata,
archive member inventory, acquisition labels and explicit unresolved-link
claims. It does not create precursor training examples or infer abundance.

The package uses schema 20/catalog 25 and family `precursor_context`. Every
source asset is copied into the standalone package and verified against its
receipt. Quantity tables remain available-in-source/deferred until child PR 2
materializes their rows. Family and availability ledgers make that state
queryable instead of hiding it in an external sidecar.

## Plan

- [x] Freeze the project/files, search-archive, parameters and summary receipts.
- [x] Add typed source-asset, acquisition, archive-table and pairing relations.
- [x] Add schema-20/catalog-25 package recognition, EvidenceStore filtering,
      aliases and source registry.
- [x] Preserve unresolved condition/specimen/MAPP links as explicit claims.
- [x] Replay the real local source root.
- [x] Run required local format, lint and full test gates.
- [ ] Run hosted gates, merge and deploy 2.0.20.

## Review

The pinned source contains five source assets, a 17-member MaxQuant archive
inventory and nine acquisitions labeled 448–456. The real package replay
verifies schema 20/catalog 25, four family rows, four availability rows, five
assets, nine acquisitions, 17 archive members and three unresolved pairing
claims. It admits zero precursor training rows in this child PR; all quantity
tables are deferred with source-backed availability states. No condition,
specimen, aliquot or MAPP pairing is inferred.

The host-level serial full non-integration suite passes all 3,427 tests (392
warnings). `./format.sh` and `./lint.sh` pass.
