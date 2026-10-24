# Standalone proteasome digestion processing package (#797 / #770)

## Specification

The Soh et al. purified-proteasome source is already typed, but it is only
visible when callers know the external appendage path and schema-9 source
selector. This successor gives the source its own package/query envelope while
retaining every original asset and typed relation. It is processing evidence,
not APC, TAP, MHC or intracellular concentration supervision.

## Plan

- [x] Freeze the schema-9 predecessor manifest, source receipt, relation schemas
      and the 15-substrate/17-condition source-count anchors.
- [x] Define the schema-19/catalog-24 `digestion_processing` envelope, family
      ledger and availability ledger without reinterpreting source quantities.
- [x] Copy source assets and all typed relations losslessly, retain predecessor
      manifest hashes, and expose run/instrument, construct, context, raw,
      filtered, final, conflict and recipe-support relations.
- [x] Register aliases and EvidenceStore filtering so source aggregates do not
      multiply observations; preserve explicit RF1, forced-zero, missing,
      annotation-only and non-contiguous product statuses.
- [x] Run required local gates and real-source replay.
- [ ] Run hosted gates, merge, and deploy 2.0.19.

## Review

The real package replay from the pinned schema-10 predecessor verifies schema
19/catalog 24 and preserves all 388 runs, 5,792,969 MS1 rows, 545,664 raw
quantities, 53,413 final vectors, 280,664 final values and 2,284 conflict rows.
EvidenceStore exposes 14 bindings, including five family and five availability
rows; aliases return the retained source layers without adding observations.
The cellular APC family is explicitly unavailable, and RF1 time/metadata
conflicts remain source rows rather than being silently adjudicated. Focused
tests, formatting and lint pass; the one-worker full non-integration gate now
passes all 3,422 tests (392 warnings) on the constrained host.
