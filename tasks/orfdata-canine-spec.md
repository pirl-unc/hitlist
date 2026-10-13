# Released orfdata canine protein consumer — #668 stage A

Add `provider="orfdata"` and `orfdata_cache_root=None` to
`curate_canine_bulk`. The default remains `provider="source"`; existing local
assets, downloads, public tables, source IDs and frozen schema-1 readers remain
unchanged. Only the reviewed Pfizer source supports the new provider. Reject
mixed local-assets/provider arguments and unsupported sources before acquisition.

The reviewed YAML pins the upstream dataset/version, canonical manifest hash
and compressed/installed artifact hashes/sizes. Use the released public
DatasetRegistry methods for discovery, explicit acquisition, offline opening
and matching native reference access. Require orfdata[data]>=0.10.0 as a normal,
lazily imported dependency. Opening a provider with allow_download=False must
never acquire or repair data. Validate pins, taxon/reference/source identities
and resource bounds before downloading or processing. Add a separate installed
upstream size bound; retain existing original-input and published-output limits.

Stream the two native unit series in corresponding order, requiring one iBAQ
and one rIBAQ observation per original line and matching target/sample/assay
and provenance. Recover native strings and source sample attributes through
the public API, using bounded disk staging to restore numeric source-line order.
Reconstitute the exact pinned summary bytes (including original text precision,
NA and transformed -Inf) and verify their original source checksum. This gives
the existing importer an identical input and proves complete source-cell replay
without a second download of the original table. The matching reference comes
from orfdata's verified SearchReference; retain its exact FASTA bytes.

Reuse the existing bundle writer, YAML sample contexts and schema-1 SQL views.
Record the upstream package version, dataset manifest, distribution pins,
actual dataset digest and search specification in the new bundle manifest
before atomic publication. Old bundles remain self-contained/offline; readers
never consult a live orfdata catalog. No source-study duplication, peptide
inference, unique protein allocation, donor expansion or DLA evidence is added.

Validation: real offline fixture equivalence of every table and reference;
explicit unit filtering and provenance; missing units, changed scope/manifest,
corrupt assets, resource limits and no-network/default behavior; historical
bundle verification after catalog mutation. Reconcile the full 313,263 source
rows, 38 pools and 20,649 reference entries against the existing source route.
Run format/lint/test, pandas 2/3 consumer checks and final-head CI; bump to
1.72.0, merge via PR, deploy from clean main and verify PyPI artifacts. Keep
#668 open for peptide/group enrichment in orfdata#23 and DLA gap #674 separate.
