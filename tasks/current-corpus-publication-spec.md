# Current corpus publication and qualification — Hitlist #685 / #680

## Publication contract

Pin exactly one immutable dataset generation before reading any index. An explicitly
selected legacy directory remains supported, with its original artifact version;
never fill a missing current-generation file from retained flat files. Keep public
manifest v1 readable for the existing v2 CI corpus. Current artifact-v10 exports
use public manifest v2 and include observations, binding, other assays, complete
contributors, bulk proteomics, line expression, peptide mappings and sanitized
mapping metadata. Inventory and original contract must agree exactly.

Validate original index size/mtime fingerprints, content hashes in contributor
provenance, and mapping-to-observation fingerprints before copying. Recheck the
source snapshot after copying and verify every copied hash before publication.
Sanitize private filesystem locations through an explicit portable projection;
preserve source dataset identifiers, byte hashes/sizes, original locator policy,
producer/schema versions, mapping settings and unresolved-reference limitations.
Do not rewrite copied artifact provenance to claim a newer source contract.

Use datacache to stage and publish the complete portable snapshot. Public metadata
and README explain every file and the limits of the corpus. The existing shell
publisher uploads only the verified materialization; no mutable root-path copies,
manual temporary-file lifecycle or replacement of existing release tags. Preserve
v1 support until the new real corpus is downloaded and verified. Workflow corpus
pins change only after that verification. Release receipts derive their complete
inventory from the verified corpus manifest, retaining v1 compatibility.

## Regression evidence

Use the actual current builder with a bounded source fixture, current contributor
sidecars and a real mapping metadata contract. Cover missing contributors/other
assays, stale mappings, changed source hashes, corruption, concurrent generation
switch, legacy v1 preservation, private-path redaction and interrupted snapshots.
Exercise the real publisher with a fake GitHub boundary, retaining the no-clobber
contract. Run format/lint/unit/integration/release gates; bump version, merge and
deploy before the refreshed public corpus is adopted by CI.

## Actual corpus work

The local legacy corpus is artifact v5 and cannot establish the current lineage
contract. Rebuild into a separate datacache root from hash-pinned registered raw
sources and the exact merged code/curation versions; preserve the existing cache.
Record the raw source snapshot age rather than implying a fresh database download.
Use adequate bounded CPU/memory/disk resources, with resumable shared input copies
and existing atomic publication. Never suppress build freshness or lineage checks.

Freeze a training bundle with identities/contributors/reviewed registry and source
metadata. Publish observation-weighted coverage of cell/assay/MHC/context axes,
actual partition assignments and source/sequence/experiment/specimen/donor audits.
Known original/reanalysis reuse stays together. Unresolved axes remain explicit;
independence claims are restricted to the supported cohort/policy. Downloadable
artifacts must include hashes and readable inspection summaries. Full scientific
readiness still requires Presto's support census, bounded fitting and held-out
baselines; this publication PR alone must not close #678 or #680.
