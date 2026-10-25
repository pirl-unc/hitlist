# Shared intact-degradomics admission before kinetic fitting

Codex / GPT-6, October 24, 2026. Follows source family2338 and
[Hitlist831](https://github.com/pirl-unc/hitlist/issues/831). No model, split or
optimization is selected by this source implementation plan.

## Purpose

Make the original substrate/product time courses queryable through the shared
Hitlist EvidenceStore with their actual uncertainty. This provides a biological
constraint for generation and destruction in Presto's availability model. It
does not calibrate concentrations or establish intact-cell APC generalization.

## Implementation boundary

Use a standalone evidence package, following the existing Goldberg/HAP1/MS1
package interfaces. Avoid copying the multi-GB general corpus. Add one source
module, a pinned asset/interpretation contract, catalog/bindings/package/store
registration, focused tests and source documentation. Reuse material_context,
snapshot/provenance and typed-table infrastructure. A supervised recipe remains
a separate, subsequently registered contract.

Assets are the eight original XLSX, publication XML, SI PDF, PRIDE project and two
inventory pages, plus the public PMC listing. Pin exact SHA256 and sizes from2338.
Keep PRIDE's CC0 and publication-asset rights separately. No RAW acquisition,
reprocessing, author contact, access bypass or original-file editing.

## Data relations and invariants

1. Assets and original source cells: preserve coordinates, literal types/values,
   cached formula values, original formulas, style references and error/missing
   tokens. Do not recalculate source workbooks. Original assets stay available
   for audit; structured source cells retain the complete populated-cell census.
2. Acquisition/source-view identity: use original RAW filenames and explicit sheet
   mappings. Retain nine repeated F1/F2 views with equality evidence, not additional
   independent observations. Record missing spleen30minuteR3 without generating a
   measurement. File identity, reaction, specimen and aliquot are distinct axes.
3. Input masses, charge envelopes and assignments: separate the observation and
   each candidate sequence/span/type interpretation. Preserve unassigned masses,
   source-local IDs, assignment counts, errors, modifications and fit-factor
   settings. A sequence may have several measured masses; no implicit sum or
   best-assignment selection. All source rows survive independently of eligibility.
4. Quantities: positive, zero, missing, error and conflicting states remain typed.
   Raw sum intensity, charge-normalized envelope intensity, assignment-associated
   intensity and publisher-view intensity have separate measurement grains.
   Expose numeric value with original cell, raw unit label and scale status.
   Do not name deconvoluted sums TIC or convert them to concentrations.
5. Publisher views and reconciliation: source identity first, exact observed mass
   only for resolving multiple structural identities. Intensity never chooses a
   match. Retain all alternatives and the one S4/deposit disagreement. Equality
   links are evidence; they do not erase original values or duplicate lineage.
6. Shared material graph and assertions: human substrate, mouse-organ enzyme,
   expression host and acquisition platform are separate roles. One source animal,
   reported R1–R3 labels, unresolved replicate independence and unknown physical
   links must survive export. Purified20S mixtures have no APC cell type or MHC
   capture/class/allele. Numeric PSMB state remains unavailable without its source.
7. Interpretation exceptions are pinned to original asset/sheet coordinates:
   F2,40minuteR1 extra H identifier; F3,brain-zeroR2 missing input IDs; abbreviated
   count header and string/integer ID equivalence. Preserve source mappings and
   validate all interpreted spans. Never infer absent input IDs from row numbers.
8. Discrepancies include missing input links, assignment/input intensity conflicts,
   missing replicate, unknown normalization and pre-zero product pool. Querying a
   source does not imply processing-training eligibility or a new FDR guarantee.

## Verification before modeling

Register a timestamped canonical-admission family before materialization. Freeze
runtime source/asset contract and exact invocation. Independently reconstruct from
the original OOXML/cells: every source value, row/quantity count, source-view link,
sample/time label, structural span, ambiguity and exception. Compare against2338
without importing the new producer's parsing helpers. Verify through public store
queries, not just internal Parquet readers. Round-trip manifests and source hashes.

Focused fixtures cover both body-layout exceptions, absent IDs, abbreviated/shared
formulas, missing-versus-zero/error cells, mass ambiguity, repeated views, publisher
conflict and absent replicate. Synthetic fixtures are software tests only. Run
format/lint and relevant package/store/catalog tests. Full local test suite remains
deferred below the existing2GiBfree-space gate; hosted full gates are required.
Feature branch, version bump, PR, merge and verified release follow Hitlist policy.

## Resource contract

Local CPU with one numerical thread, no installation or shared-runtime change.
Source package plus audit should add at most128MiBphysical bytes. Preserve a hard
1GiBfree-space floor plus32MiBmargin;3GiBRSS and20minutes per materialization/audit.
Reuse original assets with verified snapshots; preserve failures and partial
outputs. Stop on a resource breach rather than lowering its guard.

## Next modeling question

After shared admission, register a separate comparison of direct temporal controls
and a minimal substrate-to-fragments generation/destruction model. Exact expected
span amounts and discrete sampled cleavage histories must represent the same
process, observer, initial-state assumptions and objective. Include short products
and destruction rather than rewarding only terminal detected survivors.

The observation model must account for unknown peptide response and run scaling;
otherwise cross-peptide MS signal can be misread as abundance. Nominal zero cannot
be forced to intact-substrate-only. Any estimated rate is an effective parameter
unless data identify its physical meaning. Actual material/time/sequence support
determines a defensible forecast/holdout contract. One substrate/animal precludes
claiming independent new-protein or APC concentration validation. Previously
inspected biological tests remain closed; the full goal stays unchanged.
