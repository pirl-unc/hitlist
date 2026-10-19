# Actual human-cohort qualification — #680 / #678

Use the completed, separately rebuilt generation from run
`2d794b88453e75e4ae548326c4ffaaea05748d7df81140fcd24dbf0c8053ff71`.
Require its exact successful mapping-stage receipt SHA-256
`bb3177fea10370a13ff101f8babec125252896e450c84e5300968994ef7f02c0`
and every generation hash before exporting. Continue using the published
Hitlist 1.76.0 producer and the frozen Linux requirements; this PR adds
qualification recipes, not a retrospective claim that 1.78 produced the inputs.
Do not rebuild sources or modify existing generations.

## Prospective cohort

Export human MHC observations with reported human peptide source, lengths 8–25,
both MS and binding evidence, both classes, no allele-resolution shortcut.
Preserve all original assay endpoints, units, censoring, cell types/conditions,
cellular MHC sets, assay restrictions, source contributors and reviewed lineage.
Use the existing compact training-bundle API with current-index validation.
Do not expand protein mappings into additional independent observations.
The exact mapping sidecar stays pinned in the source manifest; this first compact
cohort does not supply flanks or claim unique source proteins. No optional
expression backfill, new reference downloads or guessed donor identities.

The human-source restriction makes this a bounded human proteome cohort; it is
not a claim about every source organism or every possible Presto output. Publish
the full filter contract and before/after counts. Keep other assay families in
the public parent corpus even when they are not this training export's targets.

## Execution and inspection

Run one CPU-only Modal job, at most eight CPUs, 64 GiB RAM and 90 minutes, with
an 85-minute work alarm and the provider's 512 GiB ephemeral-disk minimum.
Keep source data and reusable outputs in shared datacache. Export on bounded
scratch, verify the native training bundle, then publish its complete five-file
inventory atomically through datacache. Preserve failures, runtime, versions,
peak runner RSS and exact script/recipe hashes. Do not weaken freshness gates.

Stream the resulting training parquet for human-readable source/context counts
and coverage: citation identifiers, evidence kind, MHC class, source/host/MHC
species, original/resolved MHC roles, cell type/line, condition, assay endpoint,
units and qualifier. Include explicit missing values and a deterministic preview.
Never count mapped alternatives or contributor links as independent observations.
Report observation-weighted reviewed experimental-origin/specimen/donor/acquisition
statuses, including unknown, pooled and ambiguous values, and pin the registry.

## Frozen partitions and consumer checks

Hand the exact verified training bundle to Presto's existing frozen-input route.
Select a deterministic bounded consumer cohort before fitting, publish its exact
membership and all parent relationships, and keep shared peptides, mirrored source
records and reviewed reused experiments in one partition. Audit original and
reanalysis contributors, including the curated Abelin/Sarkizova and
Bassani-Sternberg/Liepe links. Report giant components or inadequate held-out
support as limitations; do not break identities to obtain preferred split sizes.
Unknown biological identities remain inconclusive. Only peptide-disjoint claims
are permitted unless the relevant independence audit actually passes.

Actual support/fitting runs need a registered prospective endpoint/threshold and
bounded resource contract in Presto before execution. This PR's export and
coverage are prerequisites; neither software tests nor a successful export
certifies full training readiness. Publish downloadable artifact hashes and
inspection reports; keep #678 open while unresolved qualification remains.

Version 1.78.0, separate PR after #692; format/lint/full tests, final-head CI,
merge and PyPI verification remain required.

## Receipt failure and recovery (#695)

The export completed and verified all ten artifacts at 2026-10-16T03:49:57Z,
but the receipt materialization rejected an empty input map. Preserve the failed
Modal launcher log. Publish receipts from hash-pinned serialized result/log inputs,
including failures, and exercise that path with real datacache in tests. Recover
only the original generation pinned by its datacache manifest hash and verify all
native/source hashes. Record recovery as recovery, not a successful original run;
original runtime/RSS were lost and must stay unknown. No export rerun is needed.
Keep producer 1.76.0 and the newly discovered #694 curation limits explicit.
