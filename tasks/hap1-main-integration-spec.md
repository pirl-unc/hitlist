# Reconcile HAP1 source support with the concurrent precursor-context release

Main6416616 added PXD009360 precursor_context after HAP1 source PR806 was tested.
Preserve both independent package families, all evidence definitions, public APIs,
store dispatch paths and source provenance. Do not drop either side of merge conflicts.

Both families explicitly carry schema20 plus distinct family/revision discriminators.
The package dispatcher already identifies families using the full envelope; schema
numbers alone are not globally unique identifiers. Keep these immutable family
contracts and the original HAP1 schema20/catalog25 package reproducible. Advance the
combined live catalog to26 and the source PR package version to2.0.21. No typed data,
source digest, quantity policy or frozen experiment manifest is rewritten. The
stacked response PR must subsequently bump to2.0.22 when integrating this base.

Merge current main into the feature branch, retaining explicit dispatch for both
families and the HAP1 nullable-join correction. Add a focused same-schema/different-
family regression. Run format/lint, focused package/catalog/store tests, the required
guarded test script and original-package verification; do not alter source hashes to
make a stale package pass. Run expensive local checks after the concurrent biological
fits finish so their observed runtime is not intentionally distorted by another suite.
Publish exact-head validation status; merging/releasing remains subject to the normal
upstream workflow. Update PR metadata to explain the family namespace and version.
