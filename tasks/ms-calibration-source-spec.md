# PXD004886 shared calibration source and bounded response admission

Prospective implementation specification, October 23, 2026. Owner: #762. Base: canonical MAPP integration6e275fe (#760), itself based on current main2.0.1. Reserve2.0.3, preserving2.0.2 for760; no shared-environment installation or source-generation overwrite. Previous goal turn made progress: all original supplements and12,941 cells/formulas independently audited and published in Presto3f1ae1b.

## Source contract

Pin the original Supplementary Data1/2/4/8 and article XML, with exact asset hashes, source links, public archive checksum/equivalence receipt, CC-BY4.0 and third-party-credit caveat. Article context asserts centrally prepared synthetic standards spiked after background digestion, human HEK293 background, porcine trypsin background enzyme, instrument/acquisition settings and response-informed panel/group selection. APC machinery/MHC context is not applicable. Unknown lot, purity, amount uncertainty, stock volume, file/day/run and specimen identity remain explicitly unknown. Preserve workbook self-label versus publication label conflict.

Use the existing external adapter/source registry and common context/assertion, quantity, calibration and source-cell conventions. The original files are caller-supplied and remain outside the wheel. No Presto-only curated table.

## Typed relations and identities

- Source sheets/cells: every original populated XML cell, exact coordinate, original type, raw XML, expanded formula and cached value separately; explicit empty strings survive. Sheet state and merged ranges are explicit. Response positions with no populated source cell remain absent observations, not fabricated cell records or zero.
- Nominal standards:30 exact canonical sequences with original heavy-form/protein/group labels. Nominal loads:25 group×sample design cells, actual fmol-on-column units and original misleading concentration label. Group assignment is not an independent biological feature.
- Reporter forms:31 distinct original labels; map to the nominal panel via an explicit sequence-level relation, retaining CAM/PCm/isotope/sum3/sum4 annotation. Do not transfer full nominal amount to multiple forms as certified form-specific molarity. Cross-source label equality is verified; no collapsed measurements.
- S4 source column slots:99 original positions under11 supplied site headers. These are structural slots, not reconstructed acquisition files/day/replicate IDs. Quantity observations:31×99×2 raw/normalized source slots with value/status/source-cell/representation IDs; raw/normalized are not independent biological measurements. Link nominal S4 load through the explicit sample heading and panel group, retaining the rounded per-row amount as separate source evidence.
- Source summary quantities:186 reported Avg/SD/CV cells, including empty states. Preserve independently reproducible values without replacing source figures.
- Curve rows/points:31×11×2 rows and their reported RSQ argument ranges, original header amounts, response cells, formula/cache and representation. Use actual formula references and source headers, not guessed row offsets. Sample/nominal-load joins remain unresolved. Missing grid responses are not true nondetections; exact censor/saturation disposition is unknown unless explicitly supplied.
- Source assertions/quality diagnostics: preserve curated protocol assertions, title conflict, normalized-only positives and nonuniform raw/normalized ratios. Any inferred diagnostic coefficient is labeled diagnostic, never a supplied normalization coefficient. No changed-source correction.

## Integration and scope

Register source pxd004886, adapter schema1 and catalog schema11. EvidenceStore exposes every typed relation, native columns and compatible logical aliases; safe joins connect observation→reporter/slot/nominal load and reporter→standard. A standard→multiple forms relationship must remain explicitly many-valued. Older schema6/7/8 generations must still open without falsely advertising calibration availability; accept historical catalog10 receipts unchanged.

Use the existing schema7→schema8 append-sources interface to build a fresh complete generation with both MAPP and calibration adapters. This avoids changing lineage semantics in this source PR. Independently compare every existing source file and all71 prior typed bindings against the verified MAPP generation; retain its exact parent/reference and biological recipe targets. This is a new immutable generation from the same reference base, not a relabeled old manifest. The currently reference-only append interface is explicit; recursive appends to schema8 are not required for this admission and will not be silently simulated by overwriting parent manifests.

Source ingestion and training eligibility are separate. First implement/publish the complete source substrate. Then define a canonical raw-S4 nominal-response recipe with explicit molecular-form exclusions and all source dispositions. Full curve calibration requires additional source-backed sample/acquisition/selection reconciliation; no model fit before the shared recipe is independently admitted. Processing/yield/clearance and MAPP model failure remain unresolved.

## Verification

- Unit tests exercise exact empty/formula/missing distinctions, nominal unit/identity joins, modification multiplicity, unsafe joins, source labels, actual curve formula ranges, corruption rejection and older-generation binding absence.
- Full original-source experiment registered in Presto before materialization. Freeze source commit/dirty diff, commands/env, input/output hashes. Independently replay all12,941 original values/formulas,30 standards,25 amounts,6,138 S4 slots,682 curve rows,3,410 planned curve positions and162 source-summary arithmetic checks. Compare prior MAPP/reference data and all prior recipes byte-for-byte; query every new relation without truncation.
- Preserve source discrepancies, including26 normalized-only positives and39/97 nonuniform column ratios, without using them as corrections. Preserve the exact underlying positive and missing counts, but do not confuse diagnostic source slots with independent measurements.
- Run format.sh, lint.sh, focused tests, normal test.sh with its intact memory guard, strict docs and current-head CI. Never call a preflight refusal a passed full suite. No cloud training, no unrelated checkout/environment edits.
- Close out the canonical experiment report/registry and #762 with exact scope, artifacts and remaining admission work; source correctness is not predictive quality. Finish via review and deployment gates, without claiming release beforehand.

## Checkpoints

- [ ] Pinned source contract and typed adapter with semantic verification.
- [ ] Canonical registry/catalog/query bindings and safe relationships.
- [ ] Independent complete source/old-generation conservation and software checks.
- [ ] Published source evidence and precise raw-S4 training-admission follow-up.

Metadata amendment before any materialization: a whole-Methods check found a reported2µL reference-protocol injection volume, so it must not be asserted unreported. The already pinned public archive also supplies Data13 per-site LC/MS configuration, Data14 stage/site-specific gradients, and Data15 the64-window acquisition scheme. Preserve these original assets and source cells and add one homogeneous protocol-parameter relation with source site, stage, position, parameter/value/units, explicit empty/absent states and exact locators. Do not infer that a local S4 column has a particular day/file, and do not apply pilot-stage gradients to the main study. Scope reference-protocol assertions separately from individual-run metadata. Original four-workbook source audit remains unchanged; full new source counts are verified in the upcoming integration experiment.
