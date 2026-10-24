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

## Successor main 897c98c (PR808)

While this integration was validated, main added lossless precursor quantities,
advanced precursor_context to its next immutable source generation and released
2.0.21 with live catalog26. Reconcile that implementation too: preserve every new
precursor relation and source envelope, retain original HAP1 schema20/catalog25,
and advance the combined live catalog to27 and HAP1 source consumer to2.0.22.
The stacked response consumer then becomes2.0.23. Update all current-catalog
assertions found by repository-wide search; never change pinned-source numbers.
Repeat integration checks after this actual code change, retain the already-passed
prior-head suites and original package replay, and revalidate latest main before
claiming release. No biological model rerun or new test exposure is needed.

Upstream v2 accepts a different precursor envelope and no longer accepts v1.
Keep that observed compatibility limit explicit: dispatch tests now cover HAP1v1/schema20 and
precursorv2/schema21 and reject the older precursor envelope. The original HAP1
package remains directly readable; precursorv1 reproduction requires its pinned
historical consumer. Do not relabel files or claim current support for v1.
