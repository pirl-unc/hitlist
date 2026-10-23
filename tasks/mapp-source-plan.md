# MAPP shared source generation (Hitlist #752)

October 22, 2026; Codex / GPT-6. Prior goal turn made progress: Presto80d5f9c published a verified lazy/exact span kernel; all four CI jobs now pass. The next biological fit needs actual shared MAPP ingestion. Hitlist739 remains pending; source752 has no PR at inspection. Work in isolated /private/tmp/hitlist-mapp-source from ca0c729, preserving other checkouts.

## Contract and implementation

Implement one adapter through existing external_evidence registration and immutable package mechanism. Read pinned original MaxQuant tables and publisher workbooks; retain source bytes and lossless original row/cell locators alongside typed observations. No Presto-only curation schema or fitting from audit ledgers. MAPP is proteasome-associated HEK293 fragments, not free intracellular concentration or MHC-selected ligands.

Preserve all 10,673 peptide rows, 149,959 evidence rows, 18 runs and raw group rows (36 missing-ID groups quarantined). Separate source evidence sums, BEADS median and published bead-filtered intensities, with an explicit dependency graph and reconstructed twofold threshold. Keep absent source observations, filtered zeros and imputed publisher cells distinct. Preserve UT/TNF+IFN/CHX/NO-DSP/NO-INH/BEADS labels using source experiment columns, not common filename prefix. Context describes human measured HEK293, treatments and procedural preservation controls; no inferred gene knockout. Source confidence claims conflict and remain separate; length6–40 search scope is observability, not negative-label support.

Keep source parent accession lists, leading parent, one-based coordinates/flanks and original reference filename. Add separately versioned human reference occurrence evidence, retaining all full-sequence alternatives and unresolved source mappings; never duplicate measurement identities. Original2016FASTA unavailable. Later2023DAT/2026_03FASTA remain explicit model references. Preserve publisher Data2 values/header conflict and Data3 selected-gene panel as supporting representations, not corrected targets or denominators. Parallel PXD009360 tryptic runs remain unresolved supporting context.

Provide typed manifest/catalog declarations with exact grains, units, source statuses and relation keys. No training eligibility silently inferred by ingestion. Source adapters verify conservation, reference associations and derivations; unknowns survive export.

## Verification and closure

Use small offline fixtures for filtered versus missing zero, reverse/contaminant and missing groups, alternative/repeat parent mappings, conflicting/imputed headers and unresolved pairing. Run existing package/tamper tests plus normal format/lint/test gates. Materialize real source generation under canonical external package and independently compare original source cells and earlier verified counts. Record source licenses, hashes, invocation, elapsed time, observations/quantities/contexts, output hashes and remaining gaps in a registered Presto family. No predictive split/metrics apply until real training occurs; source conservation is not model accuracy. Publish Hitlist PR linked to752; do not close full availability goal or claim source completion before reviewed deployment.

## Task checklist

- [ ] Freeze source/rights and source adapter specification.
- [x] Implement typed MAPP source quantities and shared registration; reference occurrences pending.
- [x] Verify fixtures and full original source conservation (3,224 full tests, independent TSV/XML replay).
- [ ] Publish shared PR and biological-consumer handoff.
