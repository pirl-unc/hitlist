# Training provenance, lineage and split audits (#622, #616)

## Scope

One PR, based on released 1.63.18 / #623, delivers contributor retention first,
then a shared export bundle/manifest and reviewed experimental lineage with a
split-audit API/CLI. Keep existing retained observation counts, biological
classification and evidence weighting, except the reproduced blank-identifier
data-loss bug (#624): unrelated rows lacking both identifiers must survive. Do not infer specimen equivalence from
peptide overlap, HLA typing, a cell-line name or the counting fallbacks in
`sample_identity.py`. Existing donor-aware evidence IDs remain unchanged.

## Source contributors (#622)

Capture original records before all four loss points: scanner assay deduplication,
cross-database assay deduplication, within-supplement deduplication and the
supplement/database anti-join. Record source dataset, SHA-256 snapshot, the ingested
CSV's logical data-row locator, original assay/reference/PMID/sample fields and
reported attribution. A canonical/extracted supplement's row is not claimed to
be the original publication spreadsheet's row; preserve its upstream source
description and leave unavailable upstream locators explicit.

Use a temporary SQLite collector during builds, with batched writes and bounded
memory, rather than retaining a second corpus-sized pandas frame. Keep original
source records and directed retention links. Each emitted observation has a
readable `provenance_id` identifying its contributor set in this build. Donor
expansion creates separate observation nodes pointing to the same original
record. Deduplication redirects discarded nodes to retained nodes without adding
evidence rows. Preserve all paths through successive deduplication stages.

Exact assay copies remain distinct source contributors. Within-file key matches
and supplementary `(peptide, MHC, PMID)` overlaps carry explicit overlap status;
these keys alone do not prove specimen identity. Where a discarded supplement
matches multiple retained observations, preserve candidate relationships to all
matches as unresolved, without assigning extra observation weight.

Materialize `observation_contributors.parquet` in bounded batches, one row per
retained contributor relationship, including original attribution fields and
relationship/evidence status. Retain `provenance_id` through MS/binding indexes,
ordinary training exports, projection and mapping expansion. Bind this sidecar
and source snapshot hashes into the existing `observations_meta.json`; include
it in cache validity and bump the observation artifact version. An old index is
explicitly `legacy_missing`, never falsely complete. Missing/mismatched sidecars
for an index claiming provenance raise a rebuild error.

## Reviewed lineage (#616)

Add one YAML registry for typed entities (experimental origin, specimen, donor,
acquisition), namespaced reported IDs, reviewed aliases and per-study/arm
contexts. Every resolved identity/alias needs evidence/reference text. Validate
references, entity kinds, aliases and context uniqueness. Contexts distinguish
resolved, unknown, pooled and ambiguous attribution for each biological axis.
Different specimens can share a donor; a cell-line designation is not an
experimental-origin or specimen identity. Unknown axes stay blank/unknown.

Apply registry contexts through explicit study + curated arm identity, exposing
the observation's sample-attribution evidence alongside the reviewed lineage.
Unassigned/ambiguous observation arms remain unresolved. Export the registry and
per-observation identities so projections need not duplicate the whole registry.
Seed the known Abelin/Sarkizova monoallelic reuse and Bassani-Sternberg/Liepe
fibroblast reuse only at the precision supported by primary sources. Preserve
comparison-only Abelin arms as unprofiled; do not invent physical specimen or
donor IDs when the sources establish only experimental-dataset reuse.

## Shared training bundle and manifest

Provide `write_training_bundle(directory, **training_options)` and
`hitlist export training --bundle DIR ...`. Use the ordinary training generator,
then save the requested training projection, selected contributor links,
per-observation identity table and reviewed registry in a new destination.
Do not silently overwrite an existing bundle. Stream contributor payloads in bounded Arrow batches, retaining only compact
audit fields in memory. Stage output and publish the manifest only after all
files are successfully written. Audit every original contributor PMID and
treat unresolved contributor links as incomplete biological-lineage coverage.

The versioned manifest binds every output with filename, SHA-256 and byte size;
records Hitlist/dependency versions, all effective filter/mapping/projection
options, the selected split policy, explicit no-randomness/seed-null, source
snapshots, build metadata and input-index/current-curation hashes. Include row,
observation, resolved-specimen and unresolved-coverage counts. Verify inputs
remain unchanged during export; reject nonserializable backend options rather
than claiming an unreproducible manifest. A verification API checks output
hashes and the relationships between export, identity and contributor tables.
Legacy input bundles remain usable with explicit incomplete coverage.

## Split audits

Provide an API over named partition data frames, and a CLI over verified bundles.
Report exact peptide/pMHC, original observation/mapping alternatives, reviewed
origin/specimen/donor/acquisition and optional conservative whole-study overlap
separately. Count observations once despite protein expansion. Include unresolved
coverage separately from known overlaps; absent columns or unresolved identities
must never become zero overlap or proof of independence. Preserve reviewed alias
relationships across publications without comparing labels heuristically.

Policies are explicit: report-only, peptide-disjoint, pMHC-disjoint, independent
experiments, specimen-disjoint, donor-disjoint or conservative whole-study.
Verdicts are pass/fail/inconclusive for the selected claim: known forbidden
overlap fails; missing required resolution makes a no-overlap result inconclusive.
The audit does not create random partitions, retrain models or certify historical
weights/external predictors.

## Verification

- Reproduce contributor loss and add fixture builds through the real scanner,
  builder, parquet loader and training API. Cover database copies, repeated raw
  assay rows, supplement overlap and chained deduplication, donor expansion,
  independent samples sharing pMHC, missing identifiers and upstream locators.
- Compare every pre-existing output value and retained observation count with
  the baseline on fixture builds; exclude only new provenance columns.
- Test projected/mapping-expanded exports, legacy indexes, empty exports,
  corrupted/missing sidecars, interrupted writes and deterministic manifests.
- Test reviewed cross-study aliases, one donor/multiple specimens, pooled and
  unknown samples, known cross-PMID reuse, mapping duplication across partitions,
  policy-specific verdicts and tamper detection.
- Keep local work to bounded fixtures; use full Python-matrix/integration CI for
  the corpus. Run format, lint, test.sh and the editable-install verifier with
  a version bump. Review the final diff and open the requested PR with evidence.
- Follow the repository's merge/release workflow, unless the user steers the PR
  to remain open for review. Record verified release artifacts and remaining work.
