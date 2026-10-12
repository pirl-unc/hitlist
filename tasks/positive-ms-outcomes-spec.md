# Established positive MS outcomes — #665

## Contract

Keep assay modality independent from result. Current admission accepts only MS
with Positive, Positive-High, Positive-Intermediate or Positive-Low (case and
surrounding whitespace ignored), or an absent result under the reviewed
observed-MS supplement contract. Explicit unknown, inconclusive, arbitrary and
negative outcomes never establish a positive observation. Structured method and
response conflicts retain precedence over the supplement fallback. Preserve raw
records and expose the derived outcome status to explain exclusions.

Apply one admission predicate to scalar scanner annotations, pandas masks and
projected Arrow reads. Cache/build fingerprints must invalidate old admission
partitions. Reports, live observation exports, human bundles and canine bundles
must agree for new evidence. Retain all excluded rows and source contributors.

## Portable compatibility

New human CTA bundles save a new admission-policy version. Verification dispatches
explicitly between the original heuristic policy, modality-only policy 2, and
the new known-positive policy. Old files must replay their saved semantics;
unknown policy versions must fail. Species schema 1 already has the stricter
outcome policy; preserve its output and replay contract while sharing logic where
that does not alter the saved artifacts. Audit normal-tissue filtering separately
so unknown outcomes cannot inflate donor exclusions.

## Verification and delivery

First add failing regression coverage for unknown/blank/negative/positive results,
curated blank supplements, conflicting structured fields, scalar/pandas/Arrow
parity, scanner partition/contributor retention and old/new portable replay.
Inspect the stored corpus vocabulary and quantify changed admission using grouped
source fields rather than materializing the full dataset. Validate pandas 2 and
3 where available. Run format, lint and the full test script; use existing CI
release infrastructure for memory-heavy integration gates without lowering guards.
Bump patch version, merge only after final-head checks pass, dispatch clean-main
deploy build, verify and publish exactly its tested artifacts, then verify PyPI
bytes. Review #666's sources and dependencies after publication.

## Follow-on boundary

The next dog-specific Hitlist work is #666: checked normal-tissue bulk-MS sources,
source reference and donor/pool provenance. OncoRef expression facts, Canvax
selection and downstream vaccine assembly retain their existing repository owners.
