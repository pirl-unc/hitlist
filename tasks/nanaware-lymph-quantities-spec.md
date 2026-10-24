# Nanaware2024 quantitative lymph/DC evidence

## Objective and boundary

Make original human native-lymph and mouse DC I-Ab peptide quantities queryable through the shared typed Hitlist EvidenceStore, preserving study/sample/assay distinctions and original cells. This is source evidence, not a concentration model or automatic training admission. Presto must not own a parallel source curator. A subsequent registered recipe will choose an explicit endpoint and disposition; no biological model is fit in this source change.

## Inputs and source contract

Pin the recovered PMC11233987.319 primary metadata/XML and seven original workbooks by source URL, exact byte size and SHA256. Preserve original assets, CC BY-NC-ND metadata and public access provenance. Compile exactly three reviewed peptide blocks: S6(a) columns A:Q for human native lymph; S4(e) A:AI and AK:BS for separate mouse DC DIA database/library outputs. Their side-by-side positions do not imply row alignment. The human S6(a) S:AB protein summary is derived from its peptides and is not independent precursor input. All other workbook blocks remain source-only/unreviewed for quantity projection, explicitly advertised.

All peptide-bearing source rows remain, including modified/ambiguous/low-score rows and missing/zero/invalid quantities. Keep raw cell values and exact positions, separate charge/forms, source parent alternatives, scores, PTMs, scans, coordinates, areas and intensities. Formula values are preserved as formulas, never silently evaluated. Parse only recognized sequence/flank/mass-delta notation and record parse/length status; unsupported strings remain raw evidence. No threshold filtering, averaging, missing-value imputation, normalization, source-quantity relabeling or prediction-derived MHC core labels.

## Typed relations and uniform semantics

Use a standalone versioned family integrated with evidence_package, EvidenceStore, catalog, bindings and same-grain links, following existing HAP1/ProteomeTools families. Relations:

- assets: original file, URL, role, bytes/hash, rights evidence.
- source_records: one original independent algorithm/block row; cell addresses, typed JSON, asset/sheet/block/row, original headers.
- contexts: species/taxon, tissue, cellular compartment, measured pool, APC identity, MHC class/genotype/capture, acquisition, analytical digestion, instrument, normalization/confidence scope, intervention and machinery-state status, evidence scope.
- samples: actual exported channel label, anatomical group/context, source description versus header replicate labels, donor/specimen/aliquot/run links explicitly unknown where unreported. An algorithm reanalysis is not another sample.
- observations: source row identity, parsed/raw peptide and molecular form/charge, source parent string and raw start/end, confidence/PTM/scan, parser/length status. Parent candidates may be ambiguous; no guessed full sequences.
- quantities: one original measurement cell, observation/source/sample/context, area or intensity kind, raw value, finite numeric value and explicit missing/zero/positive/negative/nonfinite/invalid/formula status, source-dependent arbitrary units and unqualified normalization.
- assertions: source-backed scope/predicate/value/status/evidence location; preserve conflicts and unknowns rather than resolving them by inference.

Source metadata must capture native extracellular human IBD lymph versus mouse MHC-selected DC material. DCs were expanded using B16-FLT3-L treatment; do not label baseline untreated APCs. Mouse strain C57BL/6J and study-level sex/age context cannot be assigned to individual unidentified samples. DC header declares 3 cervical/6 mesenteric and controls, but exports three area/intensity channels per anatomical group: preserve this unresolved mapping. No known gene KO is reported for these DC samples; unknown is not wildtype. I-Ab is capture context, not an individually measured restriction label. Mouse S4(n) no-enzyme searches reuse files described under proteolytic digestion and are ineligible as native biological cleavage labels until preparation is resolved. Human Sample1 has unresolved donor/pooling identity. Human study-level protein averages are not matched precursor concentrations.

## Integrity, API and tests

Writer stages snapshots and typed tables atomically and refuses existing destinations. Fixed source receipt requires exact hashes/sizes. Verifier checks exact family/revision/schema, full inventory, source hashes and deterministic replay of typed tables from original workbook blocks. Queries expose logical same-grain fields and explicit record/observation/sample/context links without unreviewed joins. Source-only packages advertise no training recipe. Do not change any historical package identity or content.

Use small handmade workbook fixtures solely for parser/API tests, never biological labels. Test divergent side-by-side row order, duplicate peptide strings/charges, algorithm reuse of sample IDs, zero versus missing, unsupported sequence syntax, formula quantities, header drift, source tampering and typed-row tampering. Run required format/lint/full tests plus an actual pinned-source materialization and independent original-cell comparison. Real materialization belongs in a registered Presto experiment with source git/dirty/env, source/manifest hashes, counts, resources and exact launch. No predictive metric/test split applies until a separate recipe/model plan is frozen.

## Review and next step

Deliver a source PR with version bump and public source issue; preserve the user's editable checkout/environment. Complete merge/release only after required checks. The next modeling plan must explicitly test processing plus a fixed non-MHC MS-response term against simple sequence/condition controls, account for chemistry/instrument transfer, freeze fresh component/parent splits and quantify support before fitting exact and sampled estimators with the same endpoint. This source alone does not establish APC concentrations or a biological RL advantage.

## Deposited sequence references, before implementation

The source-listed PXD051024 mzIdentML contains255full DBSequence entries, unlike a current external reference substitution. Include this exact processed file and the eight original source-listed MSV000094761 result mzIdentML files within the existing byte caps. Preserve each original DBSequence/accession/sequence/search-database/source-file identity and source peptide-evidence alternatives in typed reference/identification relations. Exact occurrence resolution must retain alternatives and not claim measured parent origin. The community SDRF reports a Fusion Lumos for human PXD051024 whereas the primary native-peptide method reports Q-Exactive HF; preserve conflict at source scope, do not overwrite the primary protocol. No acquisition/canonical-sample link may be inferred from row order alone.
