# Training input preflight — #679

New training bundles must establish index freshness before loading the corpus.
Existing frozen bundles remain verifiable offline, and ordinary historical
table reads retain their behavior. `require_current_indexes=True` is the new
bundle default. The explicit archival opt-out is recorded in the manifest and
never represents stale/unknown inputs as current. CLI spelling:
`--allow-stale-indexes`, valid only with `--bundle`.

Use the existing quiet observation-cache predicate: `True` is current, `False`
is stale, `None` is unverifiable because sources are unavailable. Validate the
mapping sidecar when mapping expansion, gene filtering or peptide-origin enrichment consumes it,
including the deprecated expansion alias. Check against its recorded build
settings, not the unrelated transcript-enrichment release argument. Missing or
invalid mapping contract metadata fails closed. No preflight downloads,
rebuilds, or full-table reads are allowed.

Record status and policy as `input_validation` alongside existing fingerprints.
Check again before manifest publication; input mutation aborts and cleans up.
The current-input requirement covers freshness only: labels, mapping coverage,
lineage resolution, partition independence and model quality need their own
acceptance reports. Verify old manifests without consulting mutable local
sources.

Tests must prove early failure without a generator call or destination write,
the observation tri-state, actual stale artifact/curation metadata, consumed
mapping checks (including gene filtering), recorded nondefault mapping settings,
archival status retention, API/CLI parity and mutation cleanup. Keep the real
tiny-source bundle round trip and legacy/archive tests. Bump the minor version
because default bundle admission changes.

Series: #678 acceptance tracker; #680 reviewed holdout lineage; #681 qualified
detectability coverage; pirl-unc/presto#70/#71 consumer fixes;
openvax/mhcflurry#494 migration. The first PR does not rebuild the shared cache
or claim to complete the frozen training release.
