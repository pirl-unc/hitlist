# Preserve adjudications when appending external evidence

Date: 2026-10-23. Agent/model: Codex / GPT-6.

## Problem and bounded outcome

Released-main commit 0a21661 adds the assay-adjudication package at schema 8 /
catalog 10. Draft source commit e186dd7 independently uses package schema 8 /
catalog 10 or 11 for external sources. Each EvidenceStore.open dispatches every
schema-8 package to its own verifier. Appending MAPP/calibration to the released
reference base also omits the nine newer adjudications. Resolve this collision
before admitting the recovered processing controls. This task does not change
scientific targets or claim new model performance.

## Implementation contract

1. Merge current main into an isolated branch based on the source draft. Retain
   all upstream adjudication, release-verification and test-runner changes. Use
   version 2.0.5 (recheck main before publication) and catalog 12 for the combined
   interface. The underlying old package versions remain immutable.
2. Write new external successors as package schema 9. Accept either the verified
   schema-7 reference package or the verified released schema-8 adjudication
   package as predecessor. Record predecessor kind and exact manifest path/hash;
   copy every predecessor byte to a separate snapshot without overwriting an
   embedded reference manifest. No arbitrary recursive source-append chains.
3. Continue reading existing draft schema-8 external packages by an explicit,
   exclusive manifest discriminator (appended_sources versus review_revision).
   Reject ambiguous/missing discriminators, wrong catalog versions and unknown
   predecessor kinds. Do not infer package kind from catalog 10 alone.
4. Extend the adjudication verifier only to allow a named embedded manifest,
   retaining strict top-level inventory checks and all source-anchor checks.
   Verify predecessor file digests and byte conservation in the successor.
5. Make evidence bindings and registry reflect actual source presence. Reference
   packages must not advertise adjudication rows; external packages with an
   adjudication predecessor must expose exactly the nine existing reviews as
   well as all MAPP/calibration relations. All four old registered recipes retain
   targets, contributors and membership; changed source manifest provenance is
   intentional and must be reported separately from target conservation.

## Verification and publication

- [x] File a concrete issue after checking existing owners.
- [x] Merge upstream and implement the minimal dispatch/snapshot changes.
- [x] Regression fixtures cover both schema-8 families, schema-9 predecessors,
      missing/ambiguous tags, tampering, source replacement, registry and bindings.
- [x] Run format.sh, lint.sh, normal test.sh and strict docs; preserve any memory
      guard refusal without bypassing it. Use hosted full tests when necessary.
- [x] Prospectively register a real package-conservation family in Presto before
      generating it. Build actual adjudication plus appended sources from pinned
      originals. Independently check every predecessor file, nine review rows,
      source tables, canonical queries and the four registered recipe outputs.
- [x] Publish draft changes and exact readbacks, link issue and record limitations.

No training, new biological split, held-out prediction, cloud compute or shared
environment modification belongs to this compatibility task. Controls source
admission and a frozen local biological comparison remain the next goal step.

## Review

Executed code0219827 passes the local normal3317-test suite, format/lint/docs and independent real183-file conservation. Published report and exact provenance: https://github.com/pirl-unc/presto/blob/82feaf2/experiments/2026-10-23_0551_codex_adjudication-source-conservation/README.md. Draft767 includes this integration; release is not claimed. The source-generation and recipe results are data conservation, not a new biological fit.
