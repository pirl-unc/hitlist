# Goldberg2025 shared source and baseline candidate admission

Codex / GPT-6, 2026-10-24, before implementation. Parent Hitlist825.
Preceding source audit: experiments/2026-10-24_2130_codex_mapp2025-source.

## Scientific contract

Recover source evidence for baseline peptide amounts that do not cancel in a
treatment/control ratio. Preserve the full availability goal; this source alone
does not establish professional-APC concentrations, short unobserved products,
causal rates or independent MS response. No pathogen-response modeling or
activity-ranked sequence selection. Other arms remain provenance/source evidence.
No model selection, fit or previous held-out outcome access during admission.

## Implementation

Isolated Hitlist branch from9583aeb/main2.0.29. Implement one standalone pinned
source package, with typed EvidenceStore/catalog access and the shared
MATERIAL_CONTEXT_SCHEMA/MATERIAL_LINK_SCHEMA. Reuse existing source snapshot,
Parquet sink, stable identifiers and replay conventions. Bump version for the PR.
Do not install it into or mutate the user's shared editable environment.

Pinned assets are the already audited original MAPP archive, publisher ZIP,
original deposition workbook, PRIDE project/file metadata and community SDRF.
Original hashes/bytes must match before compilation. Retain complete assets;
stream relevant tables, never copy only the publisher-selected subset.

Required relations:
- Original row records with all native fields and row/file locator; peptide,
  protein-group, evidence and modification-specific rows retain native IDs,
  all parent alternatives, flags, PEP, sequence/form and matching provenance.
- One quantity per source peptide/run intensity cell, including reported zero,
  missing/invalid states. Keep summed peptide values and evidence contributions
  distinct, with native contributor-ID lists. Do not infer total intracellular
  amount or intensity aggregation from numeric coincidence.
- Source acquisitions and exact original run/experiment labels; original
  deposition and community assertions separately; exact source-acquisition links
  only when explicit identity supports them. No sample-label-based physical pairs.
- Publisher source cells/rows/quantities with exact workbook coordinates and
  datatype. Original and published values are separate assertions; exact-ID/name
  reconciliation links verify agreement or expose conflicts. Whole-cell LFQ,
  captured-protein LFQ and secretome observations remain different assay types.
- Original protein-header organism assertions with partial/mixed/missing status;
  explicit source-human-only classification, not experimentally proven identity.
  Preserve the absent full search reference and truncated gene/taxonomy values.
- Table3 ordered label/value slots, including duplicate labels and zeros;
  unresolved sample, unit, experiment and material scope. No dictionary collapse,
  activity-score selection, inferred KO target, or supervised use of pooled slots.
- Common material contexts/links plus scoped assertions for cell species/type,
  capture, measurement pool, assay, source search/reference, repeat labels and
  conflicts. Unknown machinery is not WT, and supplied cell species does not
  establish peptide-source species.

## Candidate export contract

After source conservation passes, register a separately named baseline candidate
recipe from original control acquisitions. Retain every candidate/disposition,
the exact source target, context, peptide, all parents and contributor provenance.
Positive finite canonical-sequence, unflagged, explicitly human-only source
annotation is a candidate condition, not sufficient proof of processing-model
eligibility. Preserve confidence/matching status for declared downstream policies.
No filtering on predictive outcomes or interpreting zero as a biological negative.
Full-reference support, exact physical precursor pairing and confidence/selection
requirements must be reported as separate axes. A sequence-only candidate must
not be mislabeled fully qualified for the processing model. No partition or fit
until a subsequent experiment specification freezes those choices.

## Verification and bounded execution

Focused fixtures exercise repeated labels, identical peptides from distinct
sources, species ambiguity/truncation, zeros/missingness, conflicting acquisition
labels, and refusal of unsafe physical joins. Test source tampering and replay
verification. Full real-source compilation and independent original-file replay
must conserve original rows/cells and reproduce baseline values/dispositions.
Report all failures with original artifacts. No synthetic biological outcomes;
fixtures test software only. No validation/test predictions because no fit occurs.

Local CPU, one numerical thread; peak process memory gate6GiB, canonical package
plus audit outputs<=10GiB, each compile/replay<=20minutes. Original data are reused
without download. Re-plan before changing caps. Run required Hitlist format,
lint and full tests; PR/merge/release follow repository rules and actual live-job
state. Do not call the dependency shipped until exact tested artifacts are on
PyPI. Preserve unrelated Presto tasks and other worktrees.

## Closure

Timestamped Presto admission directory: frozen launch/env/git/source bundle,
independent summaries, package/recipe manifests, resource logs, README and
canonical registry. Record data availability separately from model progress.
Hitlist825 remains open for unresolved source recovery; do not close it merely
because an importer lands. Next action is a prospective biological comparison
once required input/target qualification passes.

- [x] Implement pinned source, common context and query interfaces.
- [x] Verify fixtures and real-source conservation.
- [x] Define and audit baseline candidate export.
- [ ] Pass required checks, publish PR and release through verified artifacts.
- [ ] Close admission report and registry; specify the next biological test.
