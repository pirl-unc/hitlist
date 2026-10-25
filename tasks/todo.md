# Release the merged Wolf-Levy material graph after the 2.0.38 collision (planned 2.0.39)

## Specification

PyPI already contains a nonmatching 2.0.38 distribution. Do not replace it,
and do not represent merged Wolf-Levy material-graph commit `82aa4aa` as having
been deployed under that version. Make a version-only follow-up from exactly
that merge commit. It must preserve every source and assay unchanged, use a new
patch version, and be released only after the hosted workflow's source commit
and wheel/sdist hashes match the release receipt.

### Plan

- [x] Record the PyPI collision: both existing 2.0.38 files differ from the
      validated hosted build for `82aa4aa`.
- [x] Bump only the package version to 2.0.39; inspect the diff to prove no
      evidence records, APIs, or source receipts changed.
- [ ] Run format, lint, tests and the editable-install metadata check; open a
      version-only PR and wait for its complete hosted CI result.
- [ ] Merge, build from clean main, validate the exact artifacts, publish and
      compare PyPI hashes to the release receipt. Report the result on the
      material-graph issues.

## Review

In progress on `codex/release-2-0-39-20261025`, based exactly on merged
`82aa4aa`. This PR exists solely to resolve the public-version collision; it
does not alter the PXD009360 source audit or material graph.

# Qualify Wolf-Levy companion tryptic context and material mapping (#823 / #757, planned 2.0.38)

## Specification

Extend the independently replayable PXD009360 precursor package so its
acquisition, provenance-assertion and material-graph facts use the same
source-scoped field layout and relationship pattern as canonical MAPP. This is
an additive evidence improvement: existing lossless MaxQuant rows and their
raw source payloads remain untouched, while the new relations make the known
run/project facts and explicit absence of a physical crosswalk queryable.

Audit the original PRIDE project and file metadata for all three Wolf-Levy
companion tryptic deposits: PXD009360 (nine raw runs), PXD009181 (24), and
PXD009112 (14). Freeze the byte hashes, raw-file counts and empty
`sampleProperties` result in a negative-evidence ledger. The audit establishes
only that the reviewed inventories do not provide a condition, specimen,
aliquot, or MAPP crosswalk; it does not assert that such a mapping cannot
exist in a future original submission or author correction.

For the already packaged PXD009360 source, materialize source-scoped
`precursor_source_assertion`, `precursor_material_context` and
`precursor_material_link` relations. Their field names and semantics must
match `mapp_source_assertion`, `mapp_material_context` and
`mapp_material_link` at matching grains, including `source_dataset`, locator,
assertion status, and qualified link precision. Project and source-acquisition
contexts may be linked at `project_context`; no condition/study arm,
specimen, aliquot, duplicate, donor, or precursor-to-MAPP physical link may be
invented. A consumer may query the same assertion/material API pattern for
MAPP and precursor evidence, but still must open the package that owns the
source rows.

### Plan

- [x] Retrieve and hash-check the current original PRIDE project/file metadata
      for PXD009181 and PXD009112; compare it with the already pinned
      PXD009360 file inventory and record the raw-file/sample-property audit.
- [x] Add the immutable negative-evidence receipt and source-audit report with
      exact endpoints, byte hashes, counts and scope limitations.
- [x] Add source-scoped PXD009360 assertions, project/acquisition material
      contexts and qualified project-membership links, preserving all existing
      `precursor_*` source rows and zero-training policy.
- [x] Register the new relations, logical aliases and named EvidenceStore joins
      so queries use the same source-assertion/material-graph pattern as MAPP.
- [x] Add replay, schema, source-absence and negative-inference tests; prove a
      project context does not satisfy an exact specimen/aliquot pairing.
- [x] Document the MAPP/companion-tryptic coverage boundary and uniform query
      contract; bump the version.
- [ ] Run format, lint and the full test suite.
- [ ] Open, merge and deploy the PR from clean main; report release evidence on
      #823 and #757.

## Review

In progress on `codex/wolf-levy-material-crosswalk-20261025`, based on released
2.0.37 / main `d2e0cb6`. The audit has so far confirmed that the current PRIDE
file inventories list 9 PXD009360, 24 PXD009181 and 14 PXD009112 raw files and
that none of their raw-file records carries nonempty `sampleProperties`.

# Qualify PXD052314 condition and source-sample material contexts (#825, planned 2.0.37)

## Specification

Build the next source-conserving PXD052314 improvement on released canonical
MAPP v2. The Goldberg deposition already reports a raw file, experiment and
sample label for each of 36 acquisitions, while the original MaxQuant summary
provides an exact run-to-deposition raw-file link. The current graph preserves
those facts but scopes a study arm to `(Experiment, Sample Name)`, so it cannot
express the source-backed fact that identically reported labels recur across
distinct assay views without pretending they are a shared physical sample.

Audit the pinned deposition workbook, original archive, publisher Table5 ReadMe
and community SDRF byte-for-byte. Add only the following evidence when directly
reported: a source-sample context for an exact deposited sample-label literal;
source acquisition-to-label-context links; and condition/time/replicate claims
with original locators and a status that distinguishes original deposition,
original search, publisher and secondary SDRF assertions. The canonical MAPP
projection must expose the same relations and named links as the other sources.

No label, timing, replicate, assay-group or parallel-analysis statement may
become a donor, specimen, aliquot, exact cross-assay pairing, concentration,
treatment-target, HLA/presentation, negative label or training example. A
conflicting or absent source fact remains an explicit competing assertion or
null—not a repair. Table3 pooled labels stay ordered unresolved slots unless a
source document supplies a precise identifier/crosswalk for the exact slot.

### Evidence decision matrix

| Candidate fact | Admissible source | Permitted graph effect | Forbidden inference |
| --- | --- | --- | --- |
| Raw file belongs to deposited `Experiment` / `Sample Name` | `Data_Deposition_Table.xlsx` | source acquisition → source sample-label context | specimen/aliquot or independent biological replicate identity |
| Original search run belongs to deposited raw file | original `summary.txt` plus exact deposition name | source acquisition assertion | mapping a label to a different raw file |
| Control duration / reported replicate wording | exact publisher Table5 ReadMe cell/row | qualified source assertion or source-sample context attribute | treatment target or physical-material pairing |
| Conflicting run/replicate/protocol annotation | community SDRF with locator | a competing secondary assertion | correction, alias, or overwrite of deposition truth |
| Table3 pooled label/value slot | Table3 exact cell/slot | retain ordered unresolved slot | run/specimen/condition/quantity-unit assignment |

## Plan

- [x] Create an immutable audit receipt of the exact pinned deposition, original
      summary and publisher ReadMe cells/rows used for any new fact; enumerate
      every 36-run experiment/sample-label group and all source conflicts.
- [x] Extend the Goldberg source adapter with a source-sample context and
      qualified links only where its exact label occurs in the deposition;
      retain source assertions and their original locators rather than replacing
      raw labels.
- [x] Represent publisher time/control/replicate facts at their actual source
      scope, and preserve any disagreement with original/SDRF evidence.
- [x] Project every new source table/field to canonical `mapp_*` relations,
      update the ledger/catalog/bindings and add query links that cannot imply
      specimen or aliquot identity.
- [x] Add exact replay, conflict, cross-assay-label and negative-inference
      tests; prove Table3 remains unjoined and historic canonical packages keep
      replaying.
- [x] Document the observed mapping coverage and unresolved remainder; bump the
      version and run `./format.sh`, `./lint.sh`, `./test.sh` before review.
- [ ] Open, merge and deploy the PR from clean main; report results on #825.

## Review

Implemented on `codex/goldberg-material-mapping-20261025`, starting from released
2.0.36 / main `8008fd7`. The v2 Goldberg receipt now writes 12 nonphysical
project-scoped source-label contexts and 36 acquisition-to-label links. The
Table5 bridge covers all 36 acquisition headers and ReadMe descriptions, with
typed source assertions for reported role, duration and replicate. It accepts
only an exact sheet/experiment mapping, raw-file token and deposited label
suffix; the original MAPP summary's 5450 and 5454 labels remain explicit
conflicts. Table3 remains unjoined; all specimen/aliquot fields and
exact-physical links remain null.

The canonical API exposes the new rows through the existing material relations
and now adds optional `mapp_source_assertion.subject_acquisition` and
`subject_material_context` links, making source-scoped condition facts queryable
without coercing another assertion subject into an acquisition. The standalone
Goldberg receipt revision is v2/package schema26 and the package version is
2.0.37. The audit receipt is
`tasks/goldberg-material-mapping-results/README.md`.

Verification: real pinned-source audit reproduced 36 header bridges, 36 ReadMe
descriptions, 36 role/duration/replicate claims and two original-search conflicts;
focused tests passed (41 passed); `./format.sh` and `./lint.sh` passed; the
standard `./test.sh` suite passed outside the sandbox after its memory guard
could run (3,526 passed, 1 skipped, 10 expected warnings, 338.31 seconds);
`./develop.sh` confirmed a single matching editable 2.0.37 install. Pending:
open, merge and deploy the PR, then record released artifact hashes on #825.

# Project Goldberg PXD052314 onto the canonical MAPP API (#835, planned 2.0.36)

## Specification

Extend the deliberately breaking canonical captured-fragment MAPP package with
the existing, independently verified Goldberg2025 / PXD052314 source package.
The resulting public `mapp_*` API must cover the four currently admitted
datasets plus PXD052314 without flattening one source's search, publisher or
pooled-table semantics into another's. The existing standalone `goldberg_*`
package remains an immutable, independently verifiable source receipt during
and after this migration.

Every canonical row must retain `source_dataset=PXD052314` and an explicit
source-scope precision. Semantic equivalents must share a canonical base grain:
source record, source asset, peptide, form, group, acquisition,
identification, result, measurement, source assertion, material context and
material link. Source-only properties must be retained in same-grain extensions
(including original search quantity details, publisher XML cells/quantities and
links, source FASTA-header assertions, and ordered pooled-table slots), never
dropped or padded into unrelated base rows. The canonical ledger must enumerate
every native typed table, its field inventory and its base/extension projection.

The projection does not create donor, specimen, aliquot, HLA, MHC presentation,
concentration, peptide-negative, exact precursor pairing or coherent pooled
experiment assertions. Raw zero, missing, invalid and positive values retain
their distinct source status. PXD009360/PXD009181/PXD009112 remain separately
typed tryptic/conventional MS sources, not captured-fragment MAPP.

### Projection ledger design

| Goldberg native relation | Canonical home | Conservation rule |
| --- | --- | --- |
| `assets` | `mapp_source_asset` | One pinned source asset; retain URL, role, SHA-256 and `size_bytes`. |
| `source_records` | `mapp_source_record` | Normalize source file/path, locator, row index and ordered raw-row JSON without parsing away duplicate/order information. |
| `acquisitions` | `mapp_acquisition` plus `mapp_acquisition_source_detail` | One deposited raw file; retain literal assay/sample/cell/pool labels and null specimen/aliquot fields. |
| `groups` | `mapp_group` plus `mapp_group_source_detail` | One original MaxQuant group; retain all parent, FASTA-header and linked-ID literals. |
| `references` | `mapp_source_reference` | A FASTA-header assertion is not a complete reference sequence or validated organism call. |
| `peptides` | `mapp_peptide` plus `mapp_peptide_source_detail` | One original MaxQuant peptide; preserve native IDs, source organism/confidence status and raw flanks/coordinates. |
| `forms` | `mapp_peptide_form` | One modification-specific original peptide row; do not collapse it into an unmodified peptide. |
| `identifications` | `mapp_identification` plus `mapp_identification_source_detail` | One evidence row and its source type/modification/confidence fields. |
| `quantities` | `mapp_quantity_result`, `mapp_quantity_measurement` and `mapp_search_quantity_detail` | Keep one source quantity identity and raw/value-status representation; a one-to-one result is structural, not an inferred aggregation. |
| `assertions` | `mapp_source_assertion` | Retain source scope, subject, locator and status. |
| `publisher_cells` / `publisher_quantities` / `publisher_links` | `mapp_publisher_cell`, `mapp_publisher_quantity`, `mapp_publisher_reconciliation` | Publisher cells are a distinct assay view; raw XML/formula/style and original reconciliation claims remain queryable. |
| `pooled_slots` | `mapp_pooled_quantity_slot` | Preserve order and repeated labels; these rows never become acquisition/specimen/aliquot facts. |
| `material_contexts` / `material_links` | `mapp_material_context` / `mapp_material_link` | Copy source-backed graph assertions at their existing precision. |

Existing PXD009092/PXD009508/PXD007263/PXD009438 source tables will be remapped
to the same semantic base relations. Their capture, treatment, reference,
derived-measurement, label-alias and source-specific result fields remain
extensions. The ledger test must reject an unmapped native field in either the
existing or Goldberg sources.

## Plan

- [x] Audit the exact Goldberg source schemas against the canonical base
      relations and write an explicit field-to-base-or-extension mapping before
      changing the projection.
- [x] Refactor the Goldberg compiler into a registered external-source adapter
      while preserving the standalone source package, all frozen asset hashes,
      original typed relations and independent replay verifier.
- [x] Upgrade the canonical MAPP schema to a semantic, source-neutral base
      layout; map the existing four-source relations and PXD052314 at matching
      grains, retaining source-only fields in explicit extensions.
- [x] Add PXD052314 source assets, source rows, peptide/forms/groups,
      acquisitions, identifications, quantities, publisher views, pooled slots,
      source assertions and material graph to the canonical ledger with exact
      field/row/value replay checks.
- [x] Replace the public catalog, bindings and EvidenceStore surface with the
      expanded uniform `mapp_*` relations; retain no Goldberg-specific public
      API aliases in the canonical package.
- [x] Add cross-source query tests and negative tests proving sample labels,
      parallel assays, publisher selection and ordered pooled slots do not
      become physical specimen/aliquot or training facts.
- [x] Update MAPP coverage, query and recipe documentation; bump the version;
      run `./format.sh`, `./lint.sh` and `./test.sh` before review.
- [ ] Open, merge and deploy the PR from clean main; verify PyPI artifact
      hashes, then record the exact coverage and remaining material-mapping
      gaps on #835 and #825.

## Review

The prerequisite canonical API landed as #836 and released on PyPI as 2.0.35;
this 2.0.36 branch begins from that shipped main commit. It admits the existing
independently replayable Goldberg/PXD052314 source adapter into canonical MAPP
without replacing, modifying or deleting its standalone package. A six-asset
generic receipt and the canonical projection both reproduce source rows.

Coverage now includes the three deposited 12-run PXD052314 assay views (36
source acquisitions), original search quantities/identifications, native
peptide/forms/groups/FASTA-header assertions, publisher XML cells and quantities,
publisher reconciliation assertions, and every ordered pooled Table3 slot. The
new source-detail relations retain the complete native schema beside matching
canonical MAPP relations. The ledger test proves that every native field from
all four adapters has either a canonical base or source-detail home; historical
three-source canonical v1 packages still replay exactly.

Uniform query links expose only exact same-record relationships: source detail
to canonical source records, acquisitions, peptides/forms, identifications,
search measurements, assertions, publisher cells/measurements, and optional
publisher reconciliation claims. Pooled slots link only to an original source
record. No source label, replicate number, raw-file name, publisher selection,
or Table3 repeated label becomes a donor/specimen/aliquot, condition,
perturbation, HLA/presentation, concentration, negative label, or training
example. Pooled-run/material identity resolution remains the explicit follow-up
in #825.

Verification: `./format.sh` and `./lint.sh` passed; 99 focused adapter,
canonical-package, catalog/store and successor tests passed; the full standard
suite passed outside the sandbox (temporary loopback HTTP fixtures require it):
3,523 passed, 1 skipped, 10 expected warnings in 381.94 seconds. `./develop.sh`
also verified one matching editable 2.0.36 install and metadata. The first
sandboxed full attempt was not used as evidence: it could not bind loopback
HTTP and inherited its explicit memory-probe fallback into runner self-tests.

# Canonical captured-fragment MAPP package and query API (#834, planned 2.0.34)

## Specification

Replace the public source-specific captured-fragment MAPP surface with one
breaking `mapp_*` package. The package must expose PXD009092,
PXD009508, PXD007263 and PXD009438 at the same grains, with the same physical
schema for each canonical relation and a mandatory `source_dataset` on every
row. It must preserve all source assets and existing typed source relations
inside their owned source directories, then materialize canonical union
relations in one owned `mapp/typed/` directory. No source-native field, row,
quantity representation, assertion, material context/link or unresolved state
may be deleted.

The uniform base relations are source records, peptide rows, protein groups,
acquisitions, identifications, quantity results, quantity measurements,
derivation inputs, source assertions, material contexts, material links and
label aliases. Their logical API is respectively `mapp_source_record`,
`mapp_peptide`, `mapp_group`, `mapp_acquisition`, `mapp_identification`,
`mapp_quantity_result`, `mapp_quantity_measurement`,
`mapp_derivation_input`, `mapp_source_assertion`, `mapp_material_context`,
`mapp_material_link` and `mapp_label_alias`. A stable native/source identifier
and provenance locator remain queryable. Typed extension relations hold
capture/treatment/time-course/run-context metrics and publisher-workbook
material rather than deforming the base schemas.

The package remains evidence, not a presentation label, concentration,
specimen/aliquot/donor identity or cross-modal pairing. Source-only run/sample
labels and all unknown/conflicting material facts must remain explicit. The
legacy `processing_*`, `pbmc_mapp_*` and one-source `mapp_*` public relations
are removed: existing consumers must migrate to the canonical API.

## Plan

- [x] Verify the current four-source coverage and identify the incompatible
      public namespaces, schemas and storage locations; record #834.
- [x] Specify canonical schemas, extension schemas, source mapping ledger and
      required conservation invariants before changing public bindings.
- [x] Materialize a canonical union from the four source-owned typed packages,
      preserving source tables/assets byte-for-byte and requiring `source_dataset`
      in every canonical relation.
- [x] Replace the catalog and EvidenceStore bindings/links with the one
      `mapp_*` API; delete the legacy public source-specific relation names.
- [x] Update recipes and documentation to consume the canonical surface without
      weakening endpoint, context, provenance or material-precision contracts.
- [x] Add source-by-source row/field/ID conservation tests and a uniform
      cross-source query test covering peptide, acquisition, condition,
      measurement, protocol and material context.
- [x] Update the MAPP coverage inventory: distinguish admitted captured-fragment
      deposits, active candidate Goldberg PXD052314 (#835, after #825), and
      adjacent tryptic/conventional proteomics deposits excluded from this assay family.
- [x] Bump the breaking-release version and run `./format.sh`, `./lint.sh` and
      `./test.sh`.
- [ ] Review, open and merge the PR; build from clean main and publish to PyPI.

## Review

Canonical captured-fragment MAPP now has a single public `mapp_*` namespace,
four admitted project datasets (62 fixture acquisition rows), and a mandatory
source dataset plus precision on every row. All native source tables/assets
remain adapter-owned; canonical rows replay from them and the mapping ledger
records every base/extension projection. The public external-successor test
proves all three source adapters materialize the projection and rejects a
manifest-rehashed canonical-table change. A native-field invariant proves every
retained source-table field has a canonical base or explicit extension home.

Validation on 2026-10-24: `./format.sh`, `./lint.sh`, and serial
`./test.sh` passed after final rebase (3,518 passed, 1 skipped). The full run used an isolated
temporary test directory and a 0.1 GiB provenance free-space reserve because
the shared volume had approximately 1.1 GiB available; the default 1 GiB
reserve otherwise refused the fixture builder before tests ran. No source or
package artifact was removed. Goldberg PXD052314 canonical projection is
tracked as #835 after source recovery #825.

# PXD009438 primary-PBMC MAPP admission (#822, planned 2.0.30)

## Specification

Admit the deposited PXD009438 nonspecific-digest/MAPP search results through a
new source-specific adapter without changing, replacing, or deduplicating any
existing PXD009092/PXD007263/PXD009508 evidence. Freeze the PRIDE project
metadata, complete file inventory and `SLE-search.tar.gz` source archive with
content hashes and byte/row receipts. Retain every input row and raw value,
including excluded, reverse, contaminant and unquantified records, in typed
relations with lossless original record locators.

The admitted endpoint is proteasome-associated captured-fragment MS signal,
not peptide concentration, antigen presentation, an MHC ligand, a cleavage
rate, or a training label. Patient cohort facts, clinical group labels, source
sample labels, physical specimen IDs, aliquot IDs and acquisition identifiers
must remain distinct. Populate a clinical-group context only where the source
explicitly supports it. Populate physical specimen/aliquot identity only when
the source table gives an explicit identifier. Do not match archive run number
or raw-file order to patients, clinical groups, supplementary tables, or any
other proteomic modality. Record every unavailable crosswalk as an explicit
negative-evidence assertion instead of silently dropping it.

Materialize the archive in the same typed interface and material-context graph
as the first MAPP PR, with a dedicated catalog surface so no existing relation
or source identity is overloaded. Lock source cell/row conservation, typed
counts, graph invariants, and source receipts in tests. No recipe or training
consumer gains examples from this PR. Bump the breaking-release version and
update source/catalog documentation.

## Plan

- [x] Read #822 and the parent material-linkage contract; create an isolated
      branch from released 2.0.29.
- [x] Inspect the primary article and PRIDE project/file inventory; begin
      freezing the complete MaxQuant search archive and raw-acquisition
      inventory.
- [x] Inventory the archive's unmodified tables and all reported sample/run
      labels; retain the publication's cohort statement only as source-scoped
      supporting evidence, never as a physical-material crosswalk.
- [x] Write the source receipt/contract and lossless PXD009438 adapter with
      separate project, clinical-arm, acquisition, specimen and aliquot
      contexts plus explicit unresolved links.
- [x] Register the typed relations/catalog bindings, source package and
      documentation while preserving all existing source packages bytewise.
- [x] Add fixture and real-source conservation tests, then independently
      replay rows, quantities, conditions and graph precision.
- [ ] Bump the version; run `./format.sh`, `./lint.sh`, `./test.sh`, hosted
      checks and the clean-main release; merge and publish to PyPI.
- [ ] Record receipts, coverage and remaining unknowns on #822/#820, then
      continue with #823 only after this source admission ships.

## Review

Initial PRIDE metadata describes PXD009438 as the original
"Defining the cellular proteolytic landscape in systemic lupus erythematosus
– nonspecific digest" submission. Its public inventory names one 199,785,170
byte search archive and fourteen raw acquisitions. The article reports six
healthy and eight SLE participants. The directly deposited `summary.txt` then
supplies fourteen acquisition labels: `CTRL_1` through `CTRL_6` and `LUPUS_1`
through `LUPUS_8`, each joined to one raw file. The adapter records those as
literal source group/sample-label/acquisition facts. It intentionally leaves
all specimen and aliquot IDs null: a label such as `CTRL_1` is not itself a
physical-material identifier.

The frozen search archive is 199,785,170 bytes, matches PRIDE's published SHA-1
`cb60adadcdfe8afb149543d0e564bcf84347a0c0`, and has SHA-256
`36f8fcab917a25054a351adfa21b3cbb5439079e3afdab708a8ac3deb5bd83e9`.
The real-source replay conserves 10,207 rows across the five selected source
tables: 985 peptide rows (including repeated sequences), 283 groups, 8,851
evidence rows, 14 acquisitions, 13,790 peptide/run results and 27,580 quantity
representations. All reported peptide/run intensities replay from evidence at a
maximum relative error of 0.00006500425353919898. The graph has 31 contexts
(project, two groups, fourteen source sample labels and fourteen acquisitions)
and 30 qualified links, none at physical-specimen precision.

The project metadata's `blood serum` organism-part claim conflicts with the
paper's PBMC description; both are source assertions. No source supports an
aliquot, donor, clinical annotation beyond the literal source label, or a
cross-modal/other-project pair. The test fixture and real replay pass. A
pre-existing controlled-MAPP label-alias catalog-key mismatch found during the
catalog work is tracked in #826 and corrected here: both controlled and PBMC
alias relations now expose their physical `alias_id`, rather than claiming a
nonexistent source-record key.

# Release integration memory headroom and diagnostics (#754, 2.0.3)

## Specification

The first clean-main 1.82.22 release attempt (run 39836089683, attempt 1)
ended with exit 143 after 32 of 96 integration tests. It had only 14.52 GiB
`MemAvailable` before that phase. No pytest failure, `/usr/bin/time` integration
profile, or kernel OOM evidence was captured. The retry passed. The 2.0.2
release also passed all 96 tests but reached 14,667,796 KiB peak process RSS.
Memory pressure is plausible, not a proved cause of the historical SIGTERM.

The separate CI integration job already provisions 8 GiB of disk-backed swap.
Give the clean-main release job the same bounded capacity before its complete
test gate. Record memory and swap state before and after the gate, including on
a nonzero exit when the runner remains alive. Emit each integration test name
and outcome as it completes, so the last observed case is available if a
future process is terminated. Preserve the existing source, corpus, complete
unit and integration, packaging, license, and exact-artifact release gates.
The release artifact must never upload after an incomplete test phase.

This PR changes release reliability and diagnostics only. It does not claim a
confirmed root cause or change scientific data. Bump the package to 2.0.3.

## Plan

- [x] Read the incident, failed attempt, successful release, release workflow,
      integration CI workflow, test launcher, and relevant project lessons.
- [x] Add the same tested swap setup to the release workflow and bounded
      before/after diagnostics that cannot turn a failed gate green.
- [x] Add integration-only per-test progress to `test.sh`; verify a terminated
      integration phase remains nonzero and cannot reach packaging/upload.
- [ ] Run `./format.sh`, `./lint.sh`, and `./test.sh`; use full hosted
      release-build validation if local memory preflight cannot run.
- [ ] Review the final diff and PR checks, merge, build from clean main,
      verify exact artifact provenance, publish 2.0.3, and compare PyPI hashes.
- [ ] Record release evidence and remaining uncertainty on #754, then select
      the next foundational backlog issue.

## Review

Historical exit 143 identifies SIGTERM, but the recorded logs do not establish
whether the runner, memory pressure, or a test-side process sent it. The
release workflow now matches the regular integration job's 8 GiB swap setup.
It reports memory state after success or failure and attempts bounded kernel
messages only after failure. `test.sh` prints per-case integration progress.
The existing release gating still skips artifact upload on a failed build;
stubbed exit 143 now exercises the non-retry behavior explicitly.

Focused shell/deployment tests: 38 passed. `./format.sh` and `./lint.sh`
passed. The local `./test.sh` first lacked a permitted macOS page-size probe
inside the sandbox; with the probe permitted, it correctly refused because
only 0.22 GiB was available against its 2.5 GiB minimum. Full suite and
actual swap provisioning remain for hosted release-build CI.

# Atlas acquisition lineage and exact-cohort audit (#705, #680)

## Specification

The published Atlas donor cohort is a source-only, donor-disjoint positive-ligand
cohort. Its donor/tissue/class group is not a physical specimen or an acquisition.
Review the community SDRF, official PRIDE file inventory, primary Atlas sample-hit
tables and the publication's QC rules before linking any run to a reported group.
Keep primary and community annotations distinct. Preserve the original donor
cohort and the historical Presto partitions as immutable inputs.

Produce a versioned, content-bound Hitlist audit that retains each original SDRF
row and inventory record, explicit file-alias evidence, accepted/rejected/
unresolved joins, and the mapping from QC-included source rows to the published
Atlas observations. Require uniqueness at every claimed join. Report coverage by
HLA class, donor and frozen partition, including unknown specimen/acquisition
counts and known cross-partition reuse. A source-name, tissue label, genotype or
filename alone never becomes a specimen or acquisition identity. If the public
sources cannot establish a claim, publish the gap and an inconclusive verdict.

Use pinned sources through datacache, a clean deterministic recipe, independent
verification and source-cited readable reports. Keep source-only positive labels,
peptide restriction uncertainty, and the old donor split unchanged. This PR is a
Hitlist data/provenance improvement; it makes no model-quality claim.

## Plan

- [x] Inspect the exact public source schemas, hashes, aliases and QC inclusion
      evidence; record a source-by-source join contract and observed limits.
- [x] Reproduce the current unresolved filename and group mappings with a
      source-backed report before changing any lineage assignment.
- [x] Implement pinned acquisition and a deterministic crosswalk/audit with
      explicit collision, ambiguity and exclusion states; add focused regressions.
- [x] Independently replay the actual source joins and publish observation-weighted
      coverage and overlap for the frozen Atlas donor cohort.
- [x] Review the science and diff, bump the version, run ./format.sh, ./lint.sh,
      ./test.sh and relevant source-data verification.
- [ ] Open the Hitlist PR, require final-head CI, merge, deploy from clean main,
      verify PyPI and any public audit artifacts, then update #705/#680.

## Review

The community SDRF changed in place on 2026-10-20. Its new exact bytes cover
both HLA classes, and all 1,468 SDRF peak files now exactly match unique PRIDE
inventory records; the old 731 filename mismatches are historical, not current.
Of 418 published groups, all have a community context but five have multiple
source names. There are 1,466 exact `.raw` stem matches, two unresolved raw
aliases, three peak files outside the SDRF, and 483 result records without a
source-backed QC join. No public input links each published peptide to a
specific included run or physical specimen, so all 967,449 observation-level
acquisition/specimen statuses remain unknown. The frozen split's candidate
peak-file sets have no cross-partition reuse, but actual run independence is
inconclusive. The complete crosswalk and frozen source snapshots are in
`tasks/atlas-crosswalk-results/`. The independent replay passes all source
rows, counts, partitions and output hashes. Six focused crosswalk tests and
ten existing Atlas lineage tests pass. `./format.sh` and `./lint.sh` pass.
The first `./test.sh` attempt hit the #483 memory preflight at 0.23 GB free;
after memory recovered, the full unit gate passed: 3,137 tests at 89%
coverage. Final-head integration CI remains necessary.

# PR 10: #593 Leddy TAP1-control claims (1.82.11)Specification: the SureQuant source (PMID 39438697, PMC12012165 Figure 5
and its author-supplied `conditions_table_TAP.csv`) distinguishes parental
wild-type from TAP1-knockout THP-1 macrophages. Its other comparison uses
primary human monocyte-derived macrophages, with no gene knockout reported.
Record `condition_knockout_genes: "none"` only on those four non-knockout arms,
and attach primary-source evidence to each. Preserve the existing sample
identities, infection states and all observed data. Remove exactly their four
known #593 findings, assert the loaded control claims, and verify that no
engineered-material verdict or expression-anchor tier changes. Bump 1.82.11;
stack on #717, merge and publish only after 1.82.10 reaches PyPI.

- [x] Read the source and review the four curated arms and audit entries.
- [x] Add the four claims, evidence and narrow regression coverage.
- [x] Compare audit, verdict and tier before/after; run format/lint and focused tests.
- [ ] Run the full `./test.sh` gate when memory permits, and require CI.
- [ ] Review diff, open PR, require final CI, merge and publish in sequence.

## Review

The source names the parental WT and TAP1-KO THP-1 lines separately, and
Figure 5a describes primary human monocyte-derived macrophages. The four
source-backed `none` claims remove exactly the four PMID 39438697 drift
entries: the global audit is 55 findings in nine studies (39 error, 16 warn),
down from 59 in ten (43/16). The loaded arms retain their pre-change
`is_engineered_material()` verdict (`False` on all four) and expression
anchor tiers (5/5 for THP-1, 6/6 for primary hMDMs). `./format.sh` and
`./lint.sh` pass; all 22 engineering-drift tests pass. `./test.sh` was
attempted but its #483 memory guard refused at 1.00 GB free versus 2.5 GB
required, so full local verification and CI remain pending.

# PR 9: #684 Python 3.10-3.14 support policy (1.82.10)

Specification: align the supported Python minimum with the approved OpenVax
policy while keeping no upper bound. Set `requires-python` and Ruff target to
3.10, update the local contributor rule and stale Python 3.9 comments, and run
the unit CI matrix on 3.10 through 3.14. Keep the full-corpus integration
runner on its current tested Python, with editable-install and full unit tests
on every matrix leg. Refresh the tracked uv lock using the existing resolver
without forcing incompatible versions; inspect dependency markers and any
material lock churn. Verify actual 3.10 and 3.14 dependency installs and tests
where feasible, then format, lint, full local tests and CI. Bump to 1.82.10,
stack on #716, merge only after 1.82.9 publishes, and release from clean main.

CI gate adjustment (#718): on final head `f17101e`, all five unit legs
passed; the integration log records **96 passed** and pytest exit status 0
after 18m34s, then GitHub canceled the job at its 20-minute cap before
coverage upload. The retry repeated this exact pattern. Carry the narrow
30-minute integration cap from draft #719 into this PR so its own final head
can be green; preserve every test and workflow step. Close the redundant
stacked draft #719 after this change is pushed. Repeat format/lint/local tests,
then require final-head CI and the clean-main release.

- [x] Audit all supported-version declarations, dependency markers and lockfile.
- [x] Change metadata, CI, contributor guidance, comments, version and lockfile.
- [x] Run format, lint, local tests, 3.10/3.14 installs and tests; review diff.
- [x] Raise the integration CI cap to 30 minutes without changing its suite.
- [ ] Open PR, validate full CI and release, merge and publish in order.

## Review

The version declaration, Ruff target, CI matrix, contributor rule and tracked
uv lock all used the old floor. `uv lock` resolves 64 packages for >=3.10;
its 1,016 removed lines primarily discard Python 3.9-only wheel and branch
entries, while synchronizing stale datacache/mhcgnomes records with declared
floors. `uv lock --check` passes. Ruff's new 3.10 target activates six rules;
the compatible import/type changes and `itertools.pairwise` preserve behavior.
`./format.sh` and `./lint.sh` pass. Local `./test.sh` passes 3,129 unit tests
at 89% coverage on Python 3.12. Isolated lockfile installs on Python 3.10
and 3.14 both pass the editable-install check and 92 focused proteome,
CLI/version and installer tests. A broader 3.14 run exposed native argparse
color introduced in that interpreter; disabling it within Hitlist's existing
custom-color parser restores the established help output. The twelve localhost
HTTP fixture errors in that run came from the sandbox denying a loopback
socket bind. With loopback access, the full 3.14 unit suite passes: 3,128
passed, one skipped; the five CLI-help tests also pass on both endpoints.
The post-fix 3.12 `./test.sh` rerun passes 3,129 tests at 89% coverage. Full
CI, including the five-version unit matrix and corpus integration gate,
remains pending.

The #718 follow-up changes only the integration timeout/comment in the test
workflow. Two final-head attempts were canceled at the 20-minute job cap;
the retry reached 96 passing integration tests and pytest exit status 0
after 18m34s in pytest. The cap is now 30 minutes, preserving all
steps. Format and lint pass. A new local `./test.sh` attempt was refused at
0.53 GB free because separate active `mhctools` tests consume several GB;
the previous full 3.12 and 3.14 suites passed on the same application code,
and the isolated #719 branch's identical timeout change passed 57 focused
tests plus the full 3,129-test local suite. Final-head CI remains required.

# PR 8: #593 source-backed control absences (1.82.9)

Specification: adjudicate five small same-material control studies
identified by `engineering_drift_audit`: WT ERAAP BMDCs (two class scopes,
PMID 27371725), WT H2-O splenic cells (30573663), WT 721.221-B*51:01
(31092671), control shRNA in
HeLa.ABC-KO.B51 (32161166), and siCTRL/Mock HeLa-CIITA (36215666).
Primary methods describe each as a control without the sibling's named
knockout or knockdown. Record `none` only for that specific
axis, preserving other engineered facts and all arm/row identity. Remove
exactly seven corresponding entries from the audit baseline (66 -> 59),
verify no new findings or expression-tier changes, and leave the remaining
study-grouping and vocabulary debt in #593 open. Bump to 1.82.9, run format,
lint and tests, open a PR stacked on #714, then merge and publish after
1.82.8. Include source citations and the measured audit delta in the PR.

- [x] Verify all five primary sources and seven exact audit findings.
- [x] Add the seven source-backed `none` values and remove matching baseline entries.
- [x] Check audit equality, engineered/material and expression parity; run gates.
- [ ] Review diff, open PR, pass CI, merge and release 1.82.9 in order.

## Review

Source review found that PMID 29769354's ERAP2-negative comparison includes a
PF-GFP transductant. A blank overexpression field there cannot safely become
`none` while the curated arm fails to distinguish nontransduced PF from
PF-GFP. Keep that finding in #593 and track the arm identity in #715; do
not assert a false absence to clear the audit. The other five studies have
source-backed WT, mock or non-targeting controls.
The initial six-study plan was trimmed to five after reviewing PF-GFP.
Seven new parameter cases fail before the
change and pass after it. The audit baseline equality test passes at 59
findings (43 error, 16 warn), down from 66 (47 error, 19 warn). A structured
before/after check finds only seven new `none` fields; every arm retains its
prior engineered-material verdict, so expression-anchor tier selection is
unchanged. The PF/PF-GFP ambiguity is filed as #715 and recorded on #593.
`./format.sh` and `./lint.sh` pass; 553 broader curation, conditions,
expression and QC tests pass. The first full-unit run exposed three stale
assertions that treated control knockdown fields as blank. Existing Chen and
Sarango source/export tests now expect explicit `none` on control arms and
on shared Mock+siCTRL rows, while shared statements including siT6BP remain
blank. Eleven targeted checks pass after that correction. The full local
unit suite passes: 3,129 tests, 89% coverage. `./format.sh` and `./lint.sh`
pass after the final changes. The direct development-install check reports
that the shared editable environment targets the original checkout at
1.82.2; this isolated PR checkout does not repoint it. CI integration
remains required before merge.

# PR 7: #656 cancel superseded PR validation (1.82.8)

Specification: add workflow-level concurrency to `tests.yml` keyed by its
workflow and `github.ref`, cancelling in-progress runs only for
`pull_request` events. In `release-build.yml`, keep its existing per-ref
group but make cancellation conditional on the same event. GitHub documents
PR refs as `refs/pull/<number>/merge`, stable across revisions and distinct
from `refs/heads/main` used by clean-main dispatches. Distinct workflow
prefixes keep test and release-build groups separate. Preserve complete
validation on the newest PR head and do not interrupt main publication
builds. Bump to 1.82.8 and review the four-event group/cancellation matrix.
Run format, lint, unit tests and CI; the changed release workflow must pass
its own PR build. Stack on #713, then merge and publish after 1.82.7.

- [x] Confirm the current workflow triggers and concurrency semantics.
- [x] Change only workflow concurrency and version; inspect the four-event matrix.
- [x] Run format, lint, test gates and review the scoped diff.
- [ ] Open PR, pass tests and release-build CI, merge in order, publish 1.82.8.

## Review

GitHub's workflow/concurrency documentation confirms that an open PR uses a
stable `refs/pull/<number>/merge` ref, while a clean-main dispatch uses
`refs/heads/main`. A parsed four-event matrix yields separate `tests-` and
`release-build-` groups; each PR cancels only its own older runs, and main
push/dispatch runs preserve in-progress work. `./format.sh` and `./lint.sh`
pass. The 49 release-artifact and corpus tests pass. Local `./test.sh` was
attempted but its memory preflight stopped at 2.32 GB free, below the 2.5 GB
one-worker floor; CI remains the full validation gate. The direct
development-install check reports that the shared editable install still
targets the original checkout at 1.82.2; this isolated PR checkout does not
repoint it. The diff touches only both concurrency blocks, the patch version
and this plan.

# PR 6: #588 soluble HLA capture context (1.82.7)

Specification: add the existing `soluble_mhc` capture token to the ten
Spencer 2015 and one Gilchuk 2013 arms whose primary-source methods and
current notes already establish supernatant W6/32 purification, plus five
Trolle 2016 arms whose source describes VLDLr-tagged soluble HLA purified
from supernatant. Keep the
`monoallelic` material token, arm identities, and all expression inputs
unchanged. Check the rest of the override inventory for an equally explicit
omission, but do not infer capture from a study name alone. Pin the eleven
values in a regression and verify that engineering and expression-anchor
semantics are unchanged. Bump to 1.82.7, review the scoped diff, run format,
lint, focused and full tests, then open a separate PR stacked on #712.
Merge only after 1.82.6 publishes; release from clean main and verify PyPI
artifact hashes.

- [x] Confirm the three primary-source methods and count exactly sixteen affected arms.
- [x] Add focused regression and update only the source-backed context values.
- [x] Check curation/audit and expression parity; run format, lint and tests.
- [ ] Review, open PR, pass CI, merge in order, publish 1.82.7 and close #588.

## Review

The inventory sweep found only these sixteen arms with an explicit
secreted/supernatant source and missing `soluble_mhc`: 10 in PMID 26768311,
one in 23543059 and five in 26783342. The Trolle 2016 Methods establish
supernatant purification by the VLDLr tag (PMC4744552); the Spencer and
Gilchuk source references are already recorded on the eleven arms.
The three new parameter cases fail on the parent and pass with the fix. A
structured YAML comparison confirms the sixteen context strings are the only
curation changes and `is_engineered_material` is unchanged for every arm;
the expression resolver uses that predicate. Existing expression-anchor and
engineering-drift tests pass. `./format.sh` and `./lint.sh` pass, as do 262
focused conditions/expression/QC tests and the full local `./test.sh` unit
suite (3,122 passed; 89% coverage). The direct development-install check
reports that the shared editable install still targets the original checkout
at 1.82.2; this temporary PR checkout intentionally does not repoint it.
Clean CI must supply the integration gate.

# Open defect PR series — 2026-10-19

Specification: ship independent, versioned PRs against current `main`, in this
order where evidence permits: #599 serotype candidate intersection, #603 index
row-selection memory waste, #604 populated class-scoped haplotypes, and #606
non-roster exact molecules. Investigate #642 only after measuring its current
mechanism. Preserve reported MHC restrictions and explicit provenance; never
promote an unverified designation or invent an allele from a catalog alone.
For each PR, compare baseline and changed behavior, run format/lint/tests and
CI, review the diff, merge, and publish the corresponding clean-main PyPI
release. Close an issue only for the portion verified as fixed; record remaining
cases in the issue. If full local integration is blocked by the memory guard,
use the existing release workflow and verify its artifact and PyPI hashes.

## PR 1: #599 serotype candidate intersection (1.82.2)

Input is a parsed serotype and the existing strongest exact typing tier:
peptide attribution, donor typing, then PMID pool. Match candidates through
the current mhcgnomes-backed allele-to-serotype map, retaining species and
class checks. Do not expand a serotype into all catalog members; do not fall
through from an incompatible stronger typing tier to a broader pool. Emit
`peptide_serotype_match`, `sample_serotype_match`, or
`pmid_serotype_pool` as appropriate. Preserve `mhc_restriction` so consumers
cannot mistake inferred candidates for reported exact typing. Leave blank and
unparseable restrictions unmatched, and keep class/locus behavior unchanged.

- [x] Add focused positive and negative tests across typing tiers, broad and
  public serotypes, species/class guards, absent catalog membership, and
  source-restriction preservation in scanner/supplement paths.
- [x] Implement the smallest shared intersection change and document the new
  provenance values.
- [ ] Run focused tests, `./format.sh`, `./lint.sh`, and `./test.sh`; inspect
  behavior and complete a code review.
- [ ] Open PR, verify CI, merge, deploy 1.82.2 from clean main, and confirm
  PyPI artifact hashes and issue status.

## PR 2: #603 index row selection

Specification: `_repair_scoped_peptide_attributions` currently returns a
filtered whole frame, then the default non-peptide filter selects the entire
remaining frame again. Have repair mutate labels and return a positional
keep vector for duplicate donor copies. Carry that one boolean vector as a
temporary frame column across the intervening row-wise serotype, allele-set,
length and class-label filters; at the existing non-peptide point, pop it and
intersect it with the normalized non-peptide mask. This retains the current
normalization order and performs one default whole-frame selection. When a
caller opts into non-peptide rows, still apply the donor-copy repair. The
temporary column must never leak into projected or full output; preserve row
order, original index, dtypes and values. Do not change parquet pushdown or
source identity rules.

- [x] Add a mixed fixture exercising invalid donor duplicates and non-peptide
  rows together, with default/opt-in, projection and intervening filters;
  assert order, index, values and temporary-column absence against the old
  two-filter semantics.
- [x] Implement the deferred mask and run focused parity tests.
- [x] Benchmark matched baseline/candidate loads, comparing output hashes,
  time and observed memory; use full-corpus CI for the release gate and avoid
  unsupported memory claims.
- [ ] Bump version to 1.82.3, run format/lint/test gates and complete diff
  review, then open a separate PR based on #708 (retarget to main after merge).
- [ ] Verify CI, merge, deploy from clean main and verify PyPI artifacts.

Review to date: the mixed default/opt-in, projection and intervening-filter
fixture passed on both the unchanged parent and the candidate. All 128 focused
observation, attribution-scope and export-filter tests pass on the candidate.
A separate 80,000-row, 16 Arrow-string-column sample returned exactly 79,362
rows and the same full-frame and dtype SHA-256 on both versions in alternating
fresh-process runs. Sampled process peak RSS was 608/616 MB on the parent and
593/582 MB on the candidate; elapsed times were 0.742/1.068 s and
0.913/0.797 s respectively. These paired small-sample figures show parity
and no observed memory increase; they do not estimate the full-corpus peak,
and the timing difference is within run variation. `./format.sh` and
`./lint.sh` pass; 86 observation/attribution tests also pass with pandas'
future Arrow string inference enabled. `./test.sh` was attempted but its
macOS memory preflight stopped at 0.38 GB free versus the 2.5 GB unit floor,
before pytest. Full-corpus CI remains the release gate.

## Later candidates

## PR 3: #604 populated class-scoped haplotype rosters (1.82.4)

Specification: treat a parsed `Haplotype` as a class-level restriction only
when it has a populated allele roster and exactly one of `is_class1` or
`is_class2`. Derive its class from that explicit scope. The roster's exact
members are the candidate set, filtered by the haplotype and row class, with
new `haplotype_roster` provenance. Preserve the reported haplotype text and
never label roster members `exact` or synthesize class-II pairs from chains.
Do not fall back to a PMID pool or guess identities for empty-roster RT1/ELA
forms, the upstream `H2-IEg7` parse gap, or a haplotype spanning both classes.
The roster is the source-backed bound; any later donor-typing intersection
needs a separate model of paired class-II molecules.

- [x] Add regression cases for populated class I/II rosters, class mismatch,
  empty and ambiguous haplotypes, reported identity and CLI provenance.
- [x] Implement resolution, class derivation and roster candidates with the
  smallest shared changes; validate scanner/supplement behavior.
- [x] Compare behavior on the available historical corpus vocabulary and
  record exact affected counts without projecting them onto newer corpora.
- [ ] Bump to 1.82.4, run format/lint/test and focused gates, review diff,
  open a PR stacked on #709, then retarget after preceding merges.
- [ ] Verify CI, merge and deploy from clean main with PyPI hash checks.

Review to date: before the change, eight focused expectations failed because
populated class-scoped haplotypes were `unresolved` and `unmatched`. They now
pass; empty/class-ambiguous rosters remain unmatched, and source text survives
scanner and supplement paths. The local September `artifact_version 5` corpus
contains 4,440,428 observation and 892,827 binding rows. A streaming
vocabulary audit found 15,537 observation plus 59 binding rows with populated
class-scoped rosters; every one returns `haplotype_roster` when replayed with
its actual row class, typing and PMID. The remaining 51 observation plus 228
binding haplotype rows remain unresolved. This exactly reconciles #604's
15,596/279 historical split; it is not a count for the current v4 corpus.
`./format.sh`, `./lint.sh` and 337 focused tests pass. `./test.sh` was
attempted but the macOS memory preflight stopped before pytest at 1.43 GB
free versus its 2.5 GB unit floor; full CI is required.

## PR 4: #606 source-backed non-roster alleles (1.82.5)

Specification: require mhcgnomes 3.64.6, whose exact, gene-specific
`known_alleles.yaml` entries cite the primary evidence for H2-Kbm1, Kbm8,
Kwm7, Dbm13 and AnasPlat-UAA*sd (upstream #207/#209). Accept a parsed
single-field allele as one molecule when its field either names a species
haplotype or matches a nonnumeric `Species.get_known_allele(gene, field)`.
Numeric-only catalog entries such as SahaI*49 are parser components, not
endorsed standalone molecules. Keep numeric
IPD allele groups and invented or wrong-gene spellings coarse. Preserve
canonical allele text and reported restriction. Compare the historical
vocabulary before and after to identify every newly promoted designation.
Do not add a Hitlist-specific allow-list or infer names from a pattern.

- [x] Add focused resolution, exact-candidate, and negative-control tests;
  verify they fail against the unchanged downstream implementation.
- [x] Implement the one-field catalog check, dependency floor, version bump,
  and documentation correction; review the full diff.
- [x] Run `./format.sh`, `./lint.sh`, focused tests and `./test.sh`; record any
  memory-gate refusal and rely on CI for unavailable full-corpus testing.
- [ ] Merge and deploy mhcgnomes 3.64.6 first. Then open the stacked Hitlist
  PR, pass CI, merge in order, and publish 1.82.5 from clean main.

Review to date: seven positive spellings failed at `two_digit` before the
change and now yield one canonical exact candidate each. Invented names,
wrong-gene names, numeric IPD groups and SahaI*49 (an upstream parsing
component, not a standalone allele) remain unmatched. All 350 focused
curation, floor-check and candidate tests pass with upstream 3.64.6 source;
`./format.sh` and `./lint.sh` pass. The available October corpus has
4,486,991 observation and 814,268 binding rows. A streaming check of its
previously `two_digit` rows finds exactly four newly promoted binding rows,
all `H2-K*wm7`; no observation rows or other designations change in that
snapshot. `./test.sh` was attempted, but the sandbox denied its macOS
`sysctl hw.pagesize` memory probe, so it stopped before pytest. CI and the
release workflow must run the full suite and corpus validation.

## PR 5: #642 empty-fallback export fragmentation (1.82.6)

Specification: when no single-sample PMID fallback exists, the export currently
inserts every `<meta>_fb` column as all missing and pops each one immediately
during coalescing. This generates pandas fragmentation warnings and allocates
dozens of full-length object arrays with no information. In that branch,
coalesce the already-present allele-match columns straight to their scalar
defaults (`""` or `False`) using the existing categorical-safe helper.
Preserve the nonempty fallback path and column-wise bounded-memory coalescing,
the exported schema, values, dtypes, row order and pandas 2/3 compatibility.
Reproduce warnings and output on a small public scanner export, compare
matched baseline/candidate memory and time, then check full-corpus CI memory.
If the full-corpus validation cannot run or regresses, do not claim #642
closed: record the partial finding and keep the PR unmerged.

- [x] Add a realistic empty-fallback export regression checking semantic and
  dtype parity, zero `_fb` leaks, and warning behavior; run it on the parent.
- [x] Implement the branch simplification and compare matched small-export
  warning counts, output hashes, elapsed time and peak memory.
- [x] Run format, lint, focused and full tests; review the diff.
- [ ] Open a versioned PR stacked on #606, pass full-corpus CI, merge in
  sequence, deploy from clean main and verify PyPI hashes.

Review to date: the public-format scanner fixture failed with
`PerformanceWarning` at the empty `_fb` insertion on the parent and passes
with the branch simplification. All 520 selected non-integration candidate,
export and curation tests pass; `./format.sh` and `./lint.sh` pass. A matched
2,000-row/140-column export in alternating fresh processes returned the same
full-frame/dtype digest (`873088e5...`) both times: parent warnings 36/36,
candidate 0/0; elapsed 0.448/0.427 s versus 0.386/0.390 s; peak process
RSS 255/252 MB versus 255/254 MB. These small runs establish warning removal
and parity, not a full-corpus memory saving. The available October corpus
has 69 eligible single-sample curated PMIDs, so its unfiltered build takes
the unchanged nonempty-fallback branch. `./test.sh` was attempted but the
sandbox denied its `sysctl hw.pagesize` memory probe, so it stopped before
pytest; the full CI profile remains the release gate.

## Review

PR 1 review: the pre-change serotype cases returned `unmatched`; 325 focused
curation, scanner, supplement and CLI tests now pass. `./format.sh` and
`./lint.sh` pass. Two `./test.sh` attempts stopped at the existing macOS
memory preflight (2.34 GB and 0.51 GB free, below its 2.5 GB unit threshold)
before running tests; full suite and corpus validation must run in PR CI and
the clean-main release workflow. The change intersects only the strongest
exact-typing tier. A missing catalog membership remains unmatched; it is not
filled by enumerating a serotype's possible alleles. The reported restriction
survives both scanner and supplement paths. The review also found that the
public/CLI provenance vocabulary lacked the previously emitted locus values;
both the locus and new serotype values are now included.

# PR #608 review, refresh, and release — 2026-10-18

Specification: preserve the September 28 allele and memory measurements as
historical findings, reconcile every claim that reads as current advice with
the released code and issue states, and land the documentation on current
`main`. Keep the measured numbers attributed to the v2 corpus and the old CI
runner; do not imply that those figures describe the current corpus or runner.
Preserve subsequent task notes when resolving the old branch's conflicts. Bump
the package patch version once against current `main`, then use the repository's
format, lint, test, PR, merge, and clean-main deployment gates.

- [x] Review the PR diff and verify the measured claims against source records.
- [x] Merge current `main`; reconcile the historical notes and issue status.
- [ ] Review the resulting diff and run format, lint, and tests.
- [ ] Push the updated PR, verify CI, merge, and publish the release to PyPI.
- [ ] Record the final review and release result here.

## Review

The original PR's historical measurements agree with the linked #566/#597
investigations, but its unqualified 800 MB runner-margin recommendation and
"open" issue list became misleading after subsequent releases. This refresh
pins the measurements to `ci-corpus-v2`, records #610's measured 2.06 GB Linux
peak-RSS reduction, reconciles issue states, and preserves the newer mainline
notes. The PR diff against current `main` remains documentation plus the
required 1.82.1 patch bump. Format, lint, tests, CI, merge, and publication
remain to be verified.

# Unmatched alleles and the #566 memory axis — 2026-09-28

These measurements describe the September 2026 `ci-corpus-v2` artifact and
the then-current 16 GB integration runner. Two asks: explain the 1,057,558
`unmatched` rows (23.8% of that corpus), and find what dominated #566's
15.2 GB integration peak. Both were answered after five releases shipped
(1.63.7 -> 1.63.11). Later releases and corpus generations are tracked below.

## The unmatched rows: a code defect, not the curation backlog

87.6% (925,276 rows over 73 studies) are in *curated* PMIDs, so #33/#35/#36
were never the cause. `classify_allele_resolution` counted allele fields --
HLA's convention -- against species that name an allele by haplotype, so
`H2-K*b` (one sequence, no finer form) was called truncated and
`expand_allele_set` refused it. Zero of the 164,027 rows in that bucket were
human. Validating the lone field against the species' haplotype roster
recovered **163,973 observation rows and 15,376 binding rows** across 582
PMIDs and 71,013 peptides; 84,498 of them sit in uncurated PMIDs, so no
curation work could have reached them. Shipped in 1.63.10 (#597/#598).

In that snapshot, the top 10 PMIDs were 68.0% of the unmatched mass, and the
top 50 were 96.2%. The largest deposit was PMID 29557506 (206,497 rows).

## #566: the peak is the frame's shape

14.7 GB of live Arrow string buffers peaked at the very end of the v2 build,
immediately before the categorical tightening releases all but 0.29 GB. Of
that, **3.28 GiB was pure Arrow offsets** (8 bytes/row/column, ~100 string-ish
columns, independent of content) against 5.4 GB of end-state live content.
125 columns x 4.4M rows in one pandas frame was the cost. The column-level
levers measured in this pass were worth 1-2 GB, not the 6 GB initially sought.

Four approaches evaluated in this pass, all recorded on #566:

- narrowing metadata columns *later* -- cannot help, the pages are touched
- narrowing them *earlier* -- **+337 MB CI peak RSS** while cutting live bytes
  1.3 GB (the #569 `astype` trap)
- dictionary-encoded loading -- paired A/B measured **-10,856 kB, 0.11x the
  job's own noise spread**
- #603's two whole-frame row-selection copies -- real waste, but early in the
  build, so removing them alone could not move that end-of-build peak

At the time this pass ended, the recommendation was to accept roughly 800 MB
of runner margin and monitor the peak (98 MB spread over five runs). That
recommendation was superseded by #610: releasing retained join frames cut the
same v2 integration peak from a 15.19 GB baseline mean to 13.13 GB; both
candidate runs were below both baseline runs. #566 then closed. These numbers do
not establish the memory margin for current corpus generations or runners.

#602 still earned its merge: bit-identical output, 13% faster, and it fixes a
latent misalignment bug (`from_dict(orient="index")` reorders against
insertion-ordered presence rows when winner key sets are heterogeneous --
689/3000 divergence on main, 0/3000 after). Both call sites pass heterogeneous
key sets on the v2 corpus; the reordering just did not happen to trigger.

## Released

1.63.7 #586 · 1.63.8 batch notes · 1.63.9 #590 (a live indefinite hang:
`fetch_file` inherited `timeout=None`, reintroducing #255/#402 on the newer
datacache path) · 1.63.10 #597 · 1.63.11 #566 partial.

## Follow-up status as of 2026-10-18

- **#599** remains open for serotype and coarse-restriction cases. Its
  270,743-row measurement is from the earlier v2 artifact; #633 shipped the
  gene/locus intersection and shared species/class checks from **#600** in
  1.64.5. The old `HLA-DR` example no longer describes the released path.
- **#603** (row-selection copies), **#604** (Haplotype parse results), and
  **#606** (non-roster single-molecule designations) remain open. #604's
  current issue title reports 15,596 rows, replacing the earlier 15,875
  estimate. **#605** (categorical `fillna`) and **#589** (download
  consolidation) are closed.
- Upstream mhcgnomes#201 remains open (`H2-IEg7` dispatches to `Haplotype`
  while its siblings parse as a `Pair`).

# Source-backed bulk experiment counts — #699

Local full test.sh passes 3,074 tests; format/lint and 60 focused regressions pass.
Clean audit 832c95a preserves all 2,047,003 rows. Exactly 217,514 pooled HeLa rows
have unresolved numerators and corrected inventory totals; the primary summary
confirms all 14 experiment/acquisition inventories and hashes. All 18 public assets
independently download/hash-verify; every public row and count replays against the
original source. CI/integration and PyPI release remain.

- [x] Review existing source inventory and specify explicit experiment/count semantics.
- [x] Implement nullable source-backed counts and consistent filtering/build paths.
- [x] Verify supported, unresolved, invalid, mixed-source and CSV/parquet regressions.
- [x] Publish/verify the completed clean real-source audit (all rows independently checked).
- [ ] Format/lint/full CI/integration, review, merge, deploy 1.81.0 and verify PyPI.

# Bekker-Jensen fractionation evidence — #688

- [x] Audit every legacy scope against pinned primary methods and supplements.
- [x] Implement one YAML-backed correction across loaders, builder and ingest.
- [x] Publish all 15 datacache artifacts and verify downloads plus all 2,203,214 serialized peptide/protein-arm rows.
- [ ] Format/lint/full and integration checks, review, merge and deploy 1.80.0.

Review: all original rows, quantities and order are preserved. The source-backed
projection distinguishes reported buffer pH, deposited experiment labels and
unknown protocols; 98 focused cases pass on pandas 2 and 3. Local full tests
were refused by the capacity guard (0.27 GB available), so full/integration and
clean-main deployment validation run in CI. Separate replicate-denominator
inference is tracked in #699.

# Corrected frozen corpus generation

- [x] Specify immutable baseline, released producer and source/reference reuse boundaries.
- [x] Implement released-producer/reference pins and fail-closed verification.
- [x] Run focused gates, freeze source and execute the six bounded stages.
- [x] Publish and independently verify complete generation and human inspection artifacts.
- [ ] Run full gates, review, merge and deploy 1.79.0; register new consumer qualification.

## Corrected-generation prelaunch review

Format/lint and 15 focused qualification tests pass. Only the Hitlist package pin
changes (1.76 to 1.78); the historical baseline lock remains intact. Reused
reference bytes and original receipt are hash-checked, cache-format library
versions must agree, and observations/mappings rebuild independently. All six stages succeeded; the complete public v4 corpus and all 18 assets were
independently verified. The field audit covers all 4,486,991 MHC-MS rows and
71 columns; exactly 79,406 predicted-restriction evidence cells changed.
Twenty-one focused tests pass, including historical recovery isolation (#698)
and native observation-ID preservation. The corrected human export is complete:
4,071,463 observations; 89,567 rows change only in metadata. All 21 public assets
and the native consumer bundle independently verify. Presto has a frozen new
qualification recipe using released 0.1.13. Final-head CI and deployment remain.

# Human-cohort export recovery — #695

- [x] Preserve the original failed launcher and immutable generation inventory.
- [x] Fix receipt input pinning and test success/failure publication with datacache.
- [x] Recover and independently verify all ten original files and native lineage.
- [x] Record measured coverage and unresolved independence/cell/condition metadata.
- [x] Run format/lint and full local suite (3,039 passed).
- [x] Publish and independently verify all fourteen public baseline files.
- [ ] Final CI, review, merge and PyPI release.

# Actual human-cohort qualification — 2026-10-15

Specification: [human-cohort-qualification-spec.md](human-cohort-qualification-spec.md).

- [x] Pin the completed parent corpus and write the bounded cohort contract.
- [ ] Export and verify the actual human-source/HLA compact training bundle.
- [ ] Publish readable source/context and observation-weighted lineage coverage.
- [ ] Hand exact pins to Presto frozen membership and qualification.
- [ ] Format/lint/full tests, review, merge, deploy and verify public artifacts.

# Separate real corpus qualification — 2026-10-15

Specification: [corpus-qualification-spec.md](corpus-qualification-spec.md).

- [x] Ship complete-generation publication and public pinned installer (1.76.0).
- [x] Hash raw source snapshots, inspect existing remote inputs and define limits.
- [x] Freeze the producer dependency hashes and verify the bounded cloud environment.
- [x] Complete raw-source transfer and verify remote content pins.
- [x] Build separate current indexes, then verified human-reference mappings.
- [x] Publish complete corpus and verify every public download.
- [ ] Fix actual-corpus curation regressions and pass final integration CI.
- [ ] Freeze actual cohort membership and publish coverage/leakage/lineage audits.
- [ ] Hand exact pins to canonical Presto preflight/fitting and held-out baselines.
- [ ] Review/version/format/lint/test/merge/deploy any code or workflow changes.

## Review

Source snapshots total 14,719,407,889 bytes; export dates remain unverified. Existing
remote exports differ in size and are not accepted as the pinned sources. The
old caches remain untouched. Current-generation publication shipped in #690 as
1.76.0 with verified public PyPI bytes. Current source/environment/reference stage
receipts are downloaded and independently hash-verified; see
[the receipt inventory](corpus-qualification-results/README.md). The source stage
verified both raw CSVs and 26 paper assets. The complete human reference is indexed
on POSIX scratch and shared through datacache (#125 tracks the Modal hard-link
limitation). Actual observation indexes completed (4,486,991 MS / 814,268 binding / 26,537 other
assay rows); human-reference mappings and complete portable publication finished.
All ten ci-corpus-v3 public files were independently downloaded and hash-verified.
The test/release workflows now consume the pinned full inventory through datacache.
The full local suite
passed 3,035 tests; final code CI passed all four Python unit jobs and integration
at a889bf9. Final workflow CI exposed #694: missing/stale arm verdicts in 90,455 rows.
The complete source replay now resolves 62,512 sole-line observations, retains
16,894 multi-line observations and all 11,049 treatment-uncertain HCT 116 rows.
It also exposed and fixed reversed complete-pair identity (#696). All 196 source
statements are curated; the 202-context regression fixture and focused set pass
16 tests. Format/lint pass. Local full-test attempts could not establish enough
free memory (one sandboxed probe failure, then 1.61 GiB versus required 2.5 GiB);
release CI runs ./format.sh, ./lint.sh and ./test.sh --all without relaxing guards.
Actual cohort qualification and final release gates remain pending.

# Current CI corpus publication — 2026-10-15

Specification: [current-corpus-publication-spec.md](current-corpus-publication-spec.md).

- [x] Reproduce the current/legacy inventory mismatch and inspect consumers.
- [x] Record portable-generation contract and actual-corpus scope before edits.
- [x] Implement complete datacache snapshots and versioned public manifests.
- [x] Add current-builder, stale-mapping, corruption and legacy regressions.
- [ ] Verify, review, version, merge and deploy the publication tooling.
- [ ] Rebuild independently from pinned sources, publish and verify actual corpus.
- [ ] Freeze actual cohorts and publish coverage/lineage/leakage limitations.

## Review

The tooling now pins one generation, validates every recorded generation hash,
all six index fingerprints, captured contributor hashes and source-matched
mapping metadata. Public manifest v2 retains all seven parquets, portable mapping
metadata, original source hashes and a readable inventory; v1 remains supported.
Datacache owns publication and hash-pinned downloads. Release receipts include the
complete verified inventory. The real-builder/fault/download/release regression
set passes 70 tests, with format/lint passing. Full final-head gates and the actual
new corpus remain pending; workflows still use the verified v2 legacy release.

# Scoped PXD013455 publication — 2026-10-15

Specification: [pxd013455-publication-spec.md](pxd013455-publication-spec.md).

- [x] Locate the primary public deposit and original/reanalysis publications.
- [x] Write the recovery/publication contract before implementation.
- [x] Recover and hash source settings/reference/results using datacache.
- [x] Rebuild and independently validate the documented scope.
- [ ] Publish legible source, label-policy and scope/support artifacts.
- [ ] Test, review, version, merge, deploy and verify public downloads.

## Review

Datacache #122 shipped as 1.27.0 and recovered the complete selected ZIP members
without downloading the 35.4 GB archive. A decompression boundary regression
found during importer tests is fixed in datacache #124. Source reconstruction
reproduces 172,033 observations (150,855 MS/MS and 21,178 matching), 12,287 parents
and all 46 raw-file/fraction assignments. Paper review corrected the pH default
with explicit source attribution (#688). The new source and publication tests
pass 43 cases, including independent audit failures and prior-generation reuse.
Full candidate audit, complete software gates and public release remain required.

The full audit now passes for 1,656,863 candidate occurrences, including 203,635
observed occurrences and 163,994 distinct observed sequences. A separate forced
rebuild reproduced the census. The 23-file portable archive is 84,673,182 bytes,
SHA256 9684ae010280fa84ee952439613391fe8f8bce04cceb8818d296615ef636bc9b;
manifest SHA256 441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b.
The pinned published-download API avoids rebuilding sources and verifies archive
and inventory hashes through datacache. The expanded focused suite passes 178
tests. First CI unit jobs hit a transient index propagation delay for datacache
1.27.1; the dependency is now discoverable. The next full unit run passed 3,016
cases and caught one YAML duplicate-key guard violation. Both new registry
readers now use the mandatory guard; 102 focused source/registry cases pass.
The corrected source recipe was rebuilt and audited before repinning the archive.
Final-head CI and public release remain required.

# Dataset qualification series — 2026-10-15

Specification: [atomic-generations-spec.md](atomic-generations-spec.md).

- [x] Inspect publication, datacache storage, consumer contracts and live trackers.
- [x] Create an isolated feature worktree and record the approved scope/contract.
- [x] Add failure-injection and reader-consistency regressions for #645.
- [x] Implement datacache generations, pinned reads and safe standalone updates.
- [ ] Review, version, format, lint, full tests and exact-head CI.
- [ ] Merge and deploy Hitlist; verify public distribution bytes and release receipt.
- [ ] Presto #71: frozen inputs, explicit assay policy and retained parent lineage.
- [ ] Hitlist #681 / Presto #70: durable scoped detectability data and consumer.
- [ ] Hitlist #680: pinned corpus, fixed partitions, human inventory and leakage audit.
- [ ] Presto #48/#53: measured output support and bounded real-data qualification.

## Review

The regression-only run failed after contributor publication, demonstrating changed
old bytes. Seven output-write interruption tests pass after implementation;
additional coverage verifies final publication/validation failures, pinned readers,
legacy migration, damaged-store recovery, standalone updates and concurrent exports.
Format/lint pass. Full local test.sh refused 0.13 GiB available; no guard bypass.
The broader focused suite passed 254 tests (four integration cases excluded).
Review also checks mapping bytes against the published generation receipt, so
same-size sidecar tampering cannot be re-certified by a new training export.
The existing CI corpus publisher's outdated inventory/flat paths are tracked
in #685 for the actual corpus publication stage. Initial full Python 3.9 CI passed 2,954 tests and found one obsolete assay-test
mapping stub that captured frames without producing its declared outputs. The
stub now writes both mapping artifacts; publication validation stays strict.
Review found that an explicit independent expression output unnecessarily
resolved the default cache. A regression demonstrated the failure with a damaged
default generation; independent destinations now remain independent, while writes
to managed generation paths still publish safely or reject old generations.
The actual mapping writer also reproduced an external-frame override bypass
when its previous sidecar was already current; the staged-only override guard
now applies before considering a cache hit. Old-generation bytes remain intact.
Exact-head CI and clean-main release remain required.

Datacache 1.25.1 provides public materialize/inspect_materialization APIs;
use its complete-generation publication and retain prior files. MHCflurry is
explicitly deferred. The prior #682/#72 releases are shipped, but #678 remains
open until concrete data/model acceptance artifacts exist.

# Review and ship Hitlist training-readiness PR — 2026-10-14

- [x] Review the full diff, caller contracts, failure paths and regression coverage.
- [x] File discovered defects; fix them with meaningful before/after evidence.
- [ ] Finish formatting, lint, full tests and exact-head CI.
- [ ] Merge the reviewed PR and validate/build from clean main.
- [ ] Publish verified release artifacts and confirm their public bytes/version.
- [ ] Record the release receipt and next dependency in the readiness tracker.

## Review/release scope

Review index admission and archival/offline compatibility in Hitlist #682, and
missing detectability masks/provenance in Presto #72. Preserve existing user
worktrees and caches; use isolated checkouts and CI for memory-heavy checks.
Do not treat software release as certification of the training corpus.
Hitlist uses its existing clean-main release workflow and PyPI upload. Presto
has no release workflow and its distribution name collides with an unrelated
PyPI project; prepare verified GitHub wheel/source artifacts while resolving
the requested publication destination. Never upload to the unrelated project.
A release must identify the merged SHA, test results and artifact hashes.

Review found #683: origin enrichment consumes mappings without expansion or a
gene filter. The exact preflight function previously returned `mappings=not_used`
for this option with absent mappings; it now rejects it. The real cache-backed
regression joins the admission parameter cases. Prior-head CI passed 2,931 unit
tests (two skips) and 96 integration tests; rerun CI for this reviewed head.

# Training input preflight — #679 / training release series #678

Specification: [training-input-preflight-spec.md](training-input-preflight-spec.md).

- [x] Audit source/cache semantics and file remaining cross-repository issues.
- [x] Create isolated feature worktree and write the bounded contract.
- [x] Add meaningful regressions and demonstrate the previous failure.
- [x] Implement bundle freshness policy, manifest record and CLI parity.
- [ ] Version, format, lint, full tests and final-head CI.
- [x] Open the first PR and record validation and remaining series dependencies.

## Review

Regression-only CI on 10cda44 failed all 12 intended admission cases, with
2,916 passing tests and two skips in the Python 3.12 job. Stale/unknown inputs
reached the sentinel generator instead of failing before corpus generation.
Implementation, CLI/archive/offline-verification coverage and documentation
are now present; formatting and lint pass. Full local testing requires adequate
available memory; do not bypass the guard or count a refusal as a pass. Shared
sources/cache remain intact. Final implementation-head CI remains required.

# orfdata canine protein consumer — #668 stage A

Contract: [orfdata-canine-spec.md](orfdata-canine-spec.md).

- [x] Inspect released upstream APIs and write the bounded consumer contract.
- [x] Add failing offline consumer tests using real source fixtures.
- [x] Implement explicit provider and immutable upstream receipt.
- [x] Reconcile complete real source rows, sample contexts and reference bytes.
- [ ] Version, format, lint, full tests, pandas checks and final-head CI.
- [ ] Merge, deploy from clean main and verify public PyPI artifacts.
- [ ] Update integration milestones and identify the next remaining dependency.

## Review

The initial real-fixture test failed because `provider` was not supported.
The implemented route reconstructs the exact original table checksum from the
published orfdata artifact and retains the matching FASTA bytes. Full replay:
313,263 raw rows (14 fields including source row), 313,263 normalized rows
(41 fields), 38 sample rows (10 fields), 20,649 sequence rows and all context
fields match the original-source route, with zero differences. Peptide rows
remain empty. The upstream adapter phase took 35.4 seconds in this local run;
the OS denied the resource profiler's memory read, so no RSS result is claimed.

All 52 focused canine tests pass on pandas 3.0.6 and 2.3.3; formatting, lint and
isolated dependency checks pass. The guarded full unit suite is running.
Final-head CI, merge, clean-main release and public artifact verification remain
required. #668 remains open for later peptide/group enrichment; #674 tracks the
independent normal-tissue DLA gap. This PR changes no biological sample curation.

# Canine sample curation in YAML — #672

Contract: [canine-sample-curation-spec.md](canine-sample-curation-spec.md).
Keep all 56 real sample annotations unchanged while moving source facts to YAML.

- [x] Identify the repository-rule violation and file #672.
- [x] Write the explicit sample-context contract before implementation.
- [x] Prove regressions, move source facts and check frozen replay.
- [x] Recurate both real sources and compare every sample field.
- [ ] Version, format, lint, full tests, docs and final-head CI.
- [ ] Merge after 1.71.1, deploy and verify public PyPI bytes.

## Review

Both initial regressions failed before the fix. All 61 canine/reference tests
pass on pandas 2 and 3; formatting, lint and strict documentation build pass.
Real-source imports preserve every normalized sample field for all 38 Pfizer
pools and 18 Labrador samples. Previously frozen bundles pass verification.
The guarded full local unit suite passes 2,902 tests (one skip). Final-head CI,
merge and the clean-main 1.71.2 release remain required; 1.71.1 is published.

# Pfizer canine instrument provenance — #670

Contract: [pfizer-instrument-spec.md](pfizer-instrument-spec.md). Preserve the
publication/deposit conflict explicitly; do not guess the acquisition model.

- [x] Compare primary publication and deposit metadata; file #670.
- [x] Prove the missing conflict metadata with an offline regression.
- [x] Preserve both reports in the catalog, manifest and normalized rows.
- [x] Document, version, format, lint and run final-head tests/CI.
- [x] Merge after 1.71.0, deploy from clean main and verify PyPI bytes.

## Review

The new regression failed with missing `instrument_status` before the data
correction. The catalog now preserves both cited reports and their conflict;
the existing importer snapshots that metadata and exposes the unresolved model.
No view/schema changes or rewriting of previously frozen bundles is involved.
All 53 canine/reference tests pass on pandas 2 and 3. Format, lint, strict docs
build and the guarded local full unit suite pass: 2,894 tests, one skip.
Final-head CI and clean-main release 38202949295 passed 2,894 unit tests
(one skip) and 96 integration tests. PR #671 merged as 670d51e and 1.71.1
was published; both public PyPI distributions match the tested artifacts.

# Canine bulk-MS source curation — #666

Contract: [canine-bulk-spec.md](canine-bulk-spec.md). Isolated checkout; release
after #667. Bulk protein/peptide evidence never becomes DLA presentation.

- [x] Review original studies, deposit inventories and checked source layouts.
- [x] Specify source scope, bounded storage, native measurements and identities.
- [x] Add failing offline source/ambiguity/resource regression fixtures.
- [x] Implement shared managed-asset checks, catalog, import and bounded reads.
- [x] Replay real source data and audit missing presentation coverage.
- [x] Document APIs and limitations; format, lint, full tests and final-head CI.
- [x] Merge after #667, deploy from clean main and verify PyPI bytes.

## Review

Pfizer flat-file evidence is protein-level only; Labrador
archive tissue observations are experiment aggregates with unresolved per-dog
attribution. Neither source supplies naturally presented nonmalignant DLA ligands.
Four initial API tests failed before implementation. Four additional failures
proved nonfinite/overflow handling and semantic view verification before fixing
those edges. The 52 canine-source/shared-reference tests pass on pandas 2 and
the full pandas 3 unit suite passes 2,893 tests (one skip). Format and lint pass.
The source database retains controls, all reported protein alternatives, source
cells and native quantities. Scoped literature inspection did not establish a
two-dog nonmalignant heart/brain/lung DLA panel; coverage remains missing.

Filed #668 for the Pfizer vendor peptide extraction still required by #666.
The public flat table has no peptide evidence. Its 86 GB pdResult input exceeds
this machine's currently available scratch space, and the inspected endpoint
did not honor Range. No pool expansion or peptide inference substitutes for
that unresolved data dependency; this PR must not close #666.

Final Pfizer replay retains 313,263 protein records and 38 pool identities.
Import, verification and a heart query took 14.679 seconds with 190,889,984-byte
process peak RSS; output occupied 75,664,448 bytes. This is a scoped measurement,
not a full-corpus resource claim. All 8,579 heart rows belong to one pool and
none qualify as positive presentation evidence.

Final Labrador replay preserves 528,272 experiment peptide records, 1,009,420
protein mappings, 775,003 modified-peptide records, 18 samples and the complete
131,774-entry deposited reference. Import, verification and heart query took
94.782 seconds with 287,883,264-byte process peak RSS; output occupied 572,242,014
bytes. Heart retains 48,007 records: 47,693 target detections, 278 contaminant-only
and 36 decoy-only records. All remain one unresolved aggregate experiment and
none qualify as presentation evidence. PR #669 merged as ed40f6b; clean-main
release 38200219304 passed 2,893 unit tests (one skip) and 96 integration tests.
Published 1.71.0 and verified both public PyPI files against tested artifacts.
Issue #666 remains partial because #668 is still outstanding.

# Positive MS outcomes, then canine normal-tissue curation

Plan: [positive-ms-outcomes-spec.md](positive-ms-outcomes-spec.md). Core canine
curation and species bundles shipped through #663/#664 as 1.69.0/1.70.0;
the older unchecked release items below are historical records.

- [x] Confirm merged canine releases and inspect Hitlist/downstream open issues.
- [x] Specify #665 admission, old-bundle replay and follow-on scope before code.
- [x] Reproduce unknown-result admission with regression tests.
- [x] Implement shared outcome classification and versioned portable replay.
- [x] Audit live/projected/scanner/tissue paths and stored outcome vocabulary.
- [x] Pass format, lint, full tests and final-head CI; review the complete diff.
- [x] Merge, deploy from clean main, and verify published artifact bytes.
- [ ] Implement and ship the next canine data dependency (#666) in separate PRs.

## Review

Fifteen initial regressions failed before implementation; 97 focused assay,
species and human-bundle cases then passed. The new shared outcome field keeps
MS modality separate from detection and new human bundles capture policy 3;
versions 1 and 2 replay their historical admissions. Existing species schema-1
regressions pass unchanged. Small existing positive-MS fixtures now declare their
positive outcome explicitly. Format and lint pass. Local full test.sh correctly
refused at 0.66 GiB available against the unchanged 2.5 GiB guard, so complete
validation and release gates move to CI. No memory guard was bypassed.

First full CI found 45 attribution/build fixture failures: four additional
fixture helpers omitted the outcome. Their positive intent is now explicit;
all 153 tests in the affected modules pass. The final focused assay/bundle/species/
report set passes 106 tests. Superseded CI runs were canceled before restarting
validation on the corrected head; no failed or canceled gate counts as passing.

Final review found a scalar-only edge case: numeric outcomes were normalized to
blank and could qualify as curated supplement results. Its regression failed
before correction; scalar normalization now retains nonmissing numeric values,
matching pandas and Arrow. All 99 focused assay/human/species bundle tests pass.
A bounded read-only audit checked every row of the stored observation (4,440,428)
and binding (892,827) artifacts: policy-2 versus policy-3 positive-MS admission
is unchanged in that snapshot (4,418,452 and 110,621 respectively). This is an
outcome-only comparison, not an assertion that those old partitions are correct.
Normal source paths were reviewed: the human Atlas has an explicit observed-hit
contract; the species normal gate already excludes unestablished outcomes.

Shipped in #667 at c5766c84cc60ea45000494b97354743a245143c2 as 1.70.1.
Clean-main release run 38196796458 passed format/lint, 2,867 unit tests (one
skip) and 96 integration tests. Downloaded public PyPI wheel and sdist bytes
match the verified clean-main artifact manifest.

# Species-scoped evidence implementation — #661

Concrete contract: [species-evidence-contract.md](species-evidence-contract.md).
The input is an explicit local manifest; existing human defaults remain intact.

- [x] Specify pinned reference, expression, source-occurrence, observation and policy inputs.
- [x] Demonstrate an initial failing offline contract test.
- [x] Implement bounded full-reference mapping, expression audits and frozen replay.
- [x] Exercise ambiguity, provenance, missing normal coverage and resource failures.
- [x] Replay a real scoped canine source against the full supplied dog reference.
- [x] Document the contract and review the resulting consumer workflow.
- [ ] Bump version, format, lint, tests, then final-head CI.
- [ ] Merge and deploy after #663; verify published artifacts.

## Species-bundle review

Review of downstream Tsarina #209 exposed that the shared MS predicate admits
explicit unknown outcomes. Filed Hitlist #665 rather than silently changing
historical human replay. Five failing species regressions precede a stricter
new-mode gate: known-positive results or a blank under the curated ligand-table
contract; explicit unknowns stay excluded. Re-run final-head release checks.

The new public API preserves schema-1 human bundle replay and does not consult
live indexes, OncoRef or the human Atlas. Initial contract tests failed before
implementation. The focused suite includes missing normal coverage even when
testis expression is fully measured, distinct donor counting across alleles,
shared heart/testis proteins, incorrect contributor links, and semantic replay
after artifact checksums are recomputed. Format and lint pass. Final-head full
CI remains a release gate; a memory-preflighted local unit run is in progress.

Real scoped replay used all 61,398 supplied validated Canvax occurrences and the
complete NCBI GCF_011100685.1 AR106 protein FASTA (SHA256
`478ff85988e53be0f8c36fcf290a4d512792bbdb8148a0c5132e1022e15eeb49`).
All eight curated study arms contributed three observations: 24 complete source
records yielded 116 exact mappings; three reported strings had no exact match.
The nine human-host DLA observations stayed separate from 15 native dog ligands
without peptide-level allele assignment. Missing canine normal MS remained
explicit. This exercised reference/MS interoperability without inventing matched
expression candidates or claiming spectrum reanalysis. Write plus two offline
replays on final 1.70.0 code took 33.571 seconds from the frozen input manifest,
with measured process peak RSS 468,746,240 bytes; the portable output occupied
71,020,311 bytes. Input preparation is outside that timing. These are scoped
measurements, not a full-corpus memory claim. All 92 final focused/human tests
pass on pandas 2.3.3 and 3.0.6. The earlier complete local unit run passed
2,842 tests (one skip); final-head rerun/CI also cover the six subsequent
contributor/outcome regression cases.

# Canine ligand curation and evidence bundles

Implementation specifications: [offline curation](canine-curation-spec.md) and
[species-scoped evidence design](canine-evidence-design.md).

- [x] Independently review the original paper, six checked workbooks and existing patch.
- [x] Write the offline API, sample/species, donor/IP and source-provenance contract.
- [x] Prepare an isolated worktree and lockfile-based environment.
- [x] Verify historical 1.68.1 is published before beginning canine implementation.
- [x] Add failing offline/source/species/lineage regression fixtures.
- [x] Implement reviewed local scanner inputs and deterministic workbook curation.
- [x] Curate exact experimental arms, species mechanisms and donor/specimen links.
- [x] Validate all 16,359 real observations and complete contributor coverage.
- [ ] Version, format, lint, focused tests and full final-head CI.
- [ ] Merge and deploy #660; verify published artifact bytes.
- [ ] Finalize and implement #661 against the frozen #660 observation boundary.

## Review

The historical-reference PR #662 merged at
`1b0efc92aaa82a96a5f42bcfe83b6695173dbc20` after all PR checks passed.
Clean-main release run 37849519239 passed format/lint, 2,787 unit tests (one
skip) and 96 integration tests. Downloaded PyPI wheel/sdist bytes match the
tested artifacts. Historical 1.68.1 was published before canine implementation.

Eight offline API regressions failed before implementation. The broader canine,
curation, conditions, provenance and lineage suite passes 420 tests; the final
39 focused cases pass on pandas 2 and 3. Format/lint pass. Local `test.sh`
refused at 2.11 GiB available against the unchanged 2.5 GiB minimum; full
final-head CI remains required. The package version is 1.69.0.

The real offline replay retained 16,359 observations and contributors: 3,580
human-host monoallelic DLA rows and 12,779 unassigned canine rows, 12,181 distinct
canine strings and four resolved dog donors. The two Lola IPs retain one donor,
one tumor specimen and two experiment origins. Combined source scan and scoped
observation enrichment took 5.3 seconds with 567,951,360-byte peak RSS on this
machine. This is not a full corpus rebuild or raw-spectrum reanalysis.
The reviewed mixed-source study has no whole-paper source-proteome fill or
allele pool. The documented HCT116/DLA chimeras were added to the existing
species-mismatch regression's explicit, staleness-checked exceptions.

# Historical references and canine generalization

See [the specification](historical-and-canine-spec.md).

- [x] Inspect #658/#660/#661, current main and existing work; create isolated branch.
- [x] Add real historical taxonomy regression and fix annotated OX parsing.
- [x] Stream and validate full 2015_10 extraction; independently audit sequences and coverage.
- [x] Publish historical asset, add catalog entry and verify managed fetch/reuse.
- [x] Version, format, lint, full tests and final-head CI; merge and deploy to PyPI.
- [x] Locate/review canine patch and primary sources.
- [x] Specify #660 implementation after historical release.
- [ ] Ship offline supplementary inputs and canine ligand curation (#660).
- [ ] Specify and implement explicit species/reference/policy evidence bundles (#661).
- [ ] Verify canine releases, document results and review follow-up dependencies.

## Review

Historical extraction succeeded in Actions run 37844637820 with complete source
size/MD5 checks and the expected 170,921 sequences. The 82,758,858-byte FASTA's
SHA256 is `b657a37c551326bf91a118dfa52acfab96a4f65bf2052d761d5b5f7b6252020a`.
An independent bounded audit matches all 661,142 distinct original Bekker-Jensen
peptides across the 14 packaged scopes (2,047,003 rows), using exact substrings
with I/L distinct. Swiss-Prot canonical explains 649,785; adding isoforms explains
657,123; the complete human selection explains all. Peak RSS was 70,516,736 bytes,
scratch SQLite 16,891,904 bytes, runtime 110 seconds. The E9PBK2 header and
73-residue sequence also match the independent historical UniSave fixture.
This establishes compatibility, not original search-database identity (#654).

The taxonomy regressions failed before the parser fix; all 49 focused tests now
pass. Format/lint pass. Local `test.sh` refused at 1.38 GiB available against its
unchanged 2.5 GiB guard; require the complete unit/integration CI and clean-main
release workflow before deployment. Historical asset publication, real managed
download/reuse, final-head checks and PyPI verification are release gates.

Canine source review independently verified all six original workbooks against
published MD5s: 3,580 human-host monoallelic rows and 12,779 endogenous canine
tumor rows from four dogs. Two Lola IPs are one donor. Preserve all reported
8–30-residue observations, not only the paper's 8–14-residue summary subset.

# Versioned UniProt references

See [the detailed specification](uniprot-references-spec.md).

- [x] Confirm requested managed-download packaging and inspect existing registries.
- [x] Create isolated feature branch; record provenance and resource-bound design.
- [ ] Recover and validate historical 2015_10 (upstream outage; tracked separately in #658).
- [x] Independently acquire 2026_03 with release/count validation and a complete build receipt.
- [x] Implement release catalog, bounded verified cache, API/CLI and inventory integration.
- [x] Add client/extractor/provenance regressions and document supported releases.
- [x] Publish the managed 2026_03 asset and validate real managed fetch/reuse.
- [x] Verify wheel/sdist contain the catalog and exclude sequence payloads.
- [x] Correct the duplicate-key YAML-loader violation caught by full CI.
- [ ] Version, format, lint, full tests, PR and final-head checks.
- [ ] Merge, deploy to PyPI and verify published artifacts; review follow-up queue.

## Review

The verified 2026_03 human FASTA is 127,856,511 bytes, SHA256
`02eecbba941194d2004720ddb02555f4647ceae883ac2bbda82e9287ac45a2eb`.
It contains 20,431 reviewed canonical sequences, 22,131 reviewed isoforms and
190,275 unreviewed canonical sequences (232,837 total). Canonical counts agree
with UniProt's search endpoint; both endpoints identify 2026_03. The receipt
records the compressed source checksum, final checksum, exact builder checksum,
release headers and licensing. This is not a historical-reference substitute.
The isolated `.venv` resolves pandas 3.0.6, gtfparse 3.0.2, PyEnsembl 2.26.0 and
datacache 1.16.1; compatibility inspection passes. Shared editables are untouched.
Real managed download/reuse verifies 127,857,385 total cached bytes, including
the FASTA's license statement and receipt. Local `test.sh` refused at 1.90 GiB
available versus its unchanged 2.5 GiB minimum; full checks run in CI. The user
confirmed shipping 2026_03 cache support independently of historical #658.

# Detectability follow-up — #361 / #654

See [the detailed specification](detectability-spec.md). User confirmed protocol-
scoped labels and first-seen depth only within demonstrably comparable protocols.

- [x] Audit original PRIDE search parameters/summary with bounded range requests.
- [x] Review Presto's proposed contract; no implemented prototype was found.
- [x] File source correctness defects as #654; checkpoint the specification.
- [x] Resolve primary enzyme definitions and API defaults; document unrecoverable original FASTA.
- [x] Implement occurrence-preserving digest and validated scoped dataset joins.
- [x] Verify controls, ambiguity, graded labels, bounds and pandas compatibility.
- [x] Complete #653 release (1.66.0, verified PyPI); rebase onto merged main.
- [ ] Version, PR, full checks, merge, PyPI publish and artifact verification.

# Assay modality correctness — #644; then detectability training — #361 (2026-10-07)

## Specification

The user requested #644 first and #361 next, each through a versioned PR and
release. #644 remains open on 1.65.1. A bounded scan of the local legacy indexes
confirms the MS partition includes fluorescence binding, thermodynamic/kinetic
assays, crystallography, microscopy and non-MS ligand methods. Conversely the
binding partition includes some explicitly MS rows because qualitative tiers
and comments overrode assay method. Preserve raw fields and contributor lineage.

Implement one modality contract shared by scans, builds, readers, reports and
CTA exports. Separate `assay_modality` (ms, binding, structural, non_ms_ligand,
other, unknown) from result polarity and record `assay_modality_source`.
Structured methods/responses take precedence over legacy comment/tier heuristics.
Explicit MS method plus a compatible ligand response establishes MS modality;
negative outcomes are never positive presentation evidence. Blank methods on
registered curated MS-only supplements retain their explicit source basis.
Missing/conflicting evidence never defaults to MS; a false binding flag is not
proof of MS. Keep fluorescence/stability/kinetic binding assays in binding;
retain structural, non-MS ligand, unknown and negative-MS rows in a separate
`other_assays.parquet`, accessible through `load_other_assays` with normal
filters. Expose these rows in the explicitly broad `load_all_evidence` union,
without broadening the MS/binding training API's accepted modalities.

Build all three disjoint partitions, preserve contributor links and mapping
coverage, add the third artifact to freshness/provenance fingerprints, and bump
the observation semantic version. Old indexes must fail freshness checks. Reads
must prevent legacy non-MS leakage even under column projection; recovered
binding/other rows require a rebuild for complete new partitions. Avoid broad
whole-frame copies solely for filtering. New scans and report paths must agree
on MS admission. Keep experimental restriction and biological-source attribution
separate from this assay-category correction. Whole-generation publication is
already tracked separately in #645; do not claim it is fixed here.

Regression cases include the reported PAGE4/MAGE fluorescence, crystallography,
thermal stability and microscopy examples; valid cellular/secreted MS; Edman,
coelution and T-cell ligand detection; missing/conflicting metadata; negative MS
and positive tiers; curated supplements; scan/build/report/bundle parity;
projected old-index reads; contributor integrity and cache invalidation. Audit
real assay signatures and a bounded fresh raw scan, retaining explicit scope.

## Plan

- [x] Verify issues remain open; inspect code, lessons and actual assay signatures.
- [x] Create isolated feature branch and checkpoint the modality design.
- [x] Add failing regressions for the shared contract and end-to-end partitions.
- [x] Implement modality, retention, read/report consistency and freshness.
- [x] Audit real inputs and update fixtures to state their intended modality.
- [ ] Document the contract, bump version, run format/lint/test and pandas 2/3 checks.
- [ ] Open #644 PR, pass final-head gates, merge and publish verified artifacts.
- [ ] Inspect #361's primary source/search metadata and Presto prototype; write its detailed spec.
- [ ] Implement and verify bounded candidate generation and condition-scoped labels.
- [ ] Ship #361 separately with full validation and PyPI verification.

## #644 CI follow-up

The first integration run passed 94 tests and exposed two historical fixture/
assertion assumptions. The resolved-study replay now preserves raw assay method,
response and outcome; its bounded real-data replay passed for 12,813 admitted
MS rows across three studies. Species coverage now checks exact admitted MS
counts and the presence of uncurated studies instead of retaining a >100 PMID
threshold inflated by non-MS evidence; the real-corpus test passes (47 mouse
class-I MS PMIDs). No scientific admission rule was relaxed. Format/lint pass;
new final-head CI and the full local unit suite are required after this patch.

## #644 implementation review (before PR)

- Version 1.66.0; observation semantic artifact version 9. All three partitions
  retain contributors and participate in mapping/cache contracts. Arrow applies
  admission before pandas, including projected legacy reads. CTA policy v2 keeps
  non-MS evidence auditable; old bundle verification retains its recorded policy.
- Regressions initially failed on the old classifier. Focused affected exports:
  331 passed. Broader fixture checks: 458 passed with three remaining binding
  fixture failures corrected and verified in the subsequent 195-test checks.
- Pandas 2.3.3 and 3.0.6: 195 focused tests passed each; additional real-build
  structural CTA export passed both versions. Format/lint and editable-install/
  dependency checks passed. The first full suite identified missing-modality
  synthetic fixtures; they now state their intended assay method. Final local
  full-suite attempts were refused by the unchanged memory guard (0.18–0.19 GiB
  available vs 2.5 GiB required). Complete verification remains assigned to CI.
- Bounded column audit: legacy observations contain 4,418,452 positive MS,
  19,222 structured/fallback binding and 2,754 other rows; legacy binding contains
  110,621 explicit positive-MS rows. This audit used four raw fields, omitting
  comment-only heuristics, and is not a rebuilt index. Fresh raw IEDB records
  1584690, 2007100 and 17935045 reproduce structural/binding/binding classification
  with no positive MS admission (original data rows 131001, 241187, 4155792).
- No full local corpus rebuild, shared-environment change, or claim of fixing
  whole-generation publication (#645). CI and publication evidence will be
  recorded on the PR after final-head checks.

## Review

Prior defaults PR #652 is merged and published as 1.65.1. Clean-main release
37638770041 passed 2,634 units (one skip), 96 integrations, format/lint and
artifact checks; PyPI wheel/sdist bytes matched. This task starts from clean
main a96b1e5. The shared Python environment is distinct from the task's isolated
release environment; preserve editable checkouts and compatible current versions.

# Expression input defaults (2026-10-07)

## Specification and plan

The user corrected mandatory input flags after 1.65.0. Provide the same defaults
through CLI and Python API, keeping all explicit overrides. With no file path,
look only for conventional expression.tsv/csv/tab, quant.sf, abundance.tsv and
RSEM *.genes.results / *.isoforms.results (including gzip) in the working
directory; an explicit directory uses the same search. Require exactly one
candidate. Do not recurse, choose the newest file, invent patient measurements,
or treat unrelated numeric files as expression.

Detect identifier headers case-insensitively (gene IDs/symbols, transcript IDs,
and neutral quantifier Name/target_id headers). Prefer stable gene IDs over
symbols, recognize standard Salmon/kallisto/RSEM profiles, and infer Ensembl
transcript IDs in neutral columns. Generic tables containing both gene and
transcript columns require an explicit level or identifier column. Infer a
single TPM-labelled value column; multiple sample TPM columns or no TPM-labelled
column require --tpm-column. Never infer TPM from read counts or FPKM. Explicit
column names remain exact and authoritative. Record the resolved path, columns,
level and whether each was inferred in manifest expression metadata.
RSEM's documented posterior estimates and uncertainty columns describe the same
sample: choose ordinary TPM by default while retaining explicit estimator overrides.

- [x] Inspect CLI/API and OncoRef's existing column conventions; checkpoint plan.
- [x] Add regression tests that fail under 1.65.0 defaults.
- [x] Implement shared path/column/level resolution with provenance.
- [x] Exercise gene/quantifier, ambiguous, override, gzip and CLI/API cases.
- [x] Update documentation and bump version to 1.65.1.
- [ ] Run format/lint/test and validate pandas 2/3 plus final-head CI.
- [ ] Merge and deploy from clean main; verify published artifact bytes.

## Review

Issue #651 tracks this fix. The first ten regression cases failed against the
required-argument implementation and now pass. Format/lint pass; 99 focused
checks pass on pandas 3.0.6 and 52 evidence checks pass on pandas 2.3.3.
CLI/API exports verify the selected values and manifest provenance. Standard
quantifier fixtures follow their primary format documentation, including RSEM
posterior estimates and uncertainty fields. The isolated 1.65.1 editable install
and dependency consistency checks pass. Local ./test.sh correctly refused its
memory preflight (0.24 GiB available versus 2.5 GiB required); full-suite and
release checks must pass in CI before merge and publication.

Preceding PR #650 shipped as 1.65.0 from main 1a0c9f5. Release run 37537861454
passed 2,613 unit tests (one skip) and 96 integration tests. Both PyPI artifacts
were downloaded and matched the tested CI bytes. This follow-up corrects input
usability, retaining expression units, specificity and blacklist policies.

# Expression-table CTA evidence bundles (2026-10-06)

## Scope and ownership

Finish and publish bounded-provenance PR #646 first. This follow-up belongs to
Hitlist and stops at a reproducible evidence bundle. Tsarina and Vaxrank own
ranking for treatment, vaccine assembly, and construct optimization. Do not add
vaccine sequence generation or substitute population prevalence for the supplied
expression measurements.

## Specification

Accept gene or transcript TPM tables without VCF/BAM. Require explicit input
columns/identifier level and retain original rows, source hashes, normalized
identifiers, mapping reference/version, expression values, and every exclusion
reason. Resolve gene identifiers through OncoRef and transcripts through the
recorded reference; never infer transcript expression from gene expression or
silently sum duplicate/ambiguous identifiers. Intersect with the live canonical
OncoRef strict/extended CTA definitions and snapshot the relevant HPA evidence.
Support configurable gene-pattern exclusions with exact exceptions (MAGE*
except MAGEA4), without redefining CTA membership or the self background.

For expressed eligible CTAs, export the exact observed peptide evidence and all
reference-protein/transcript mappings. Keep CTA-only shared sequences distinct
from non-CTA matches and retain all occurrences/coordinates. Report reference
coverage explicitly; a missing/partial mapping cannot establish specificity.
Preserve sample/donor/HLA context and raw contributor records; MS support must
have positive assay-modality evidence (existing Hitlist #644). Prediction is not
an observation. Gene selection never narrows the reference background.

Export a reusable sequence blacklist independently of the selected genes and
patient HLA: exact MS-observed sequences in heart, brain or lung in at least two
distinct people across those tissues, regardless of allele. Keep per-tissue and
union donor counts, evidence rows, source locations, and identity resolution.
Collapse repeated samples, assays and aliases of one donor. Do not use PMIDs,
assay counts, aggregate sample labels or sample-count fallbacks as people.
Unresolved/pooled identities remain an explicit review category and never imply
absence of risk. The user confirmed nonmalignant tissue only; tumor observations do not
trigger this blacklist. Record that choice as an explicit policy.

Current IEDB HLA Ligand Atlas rows omit the donor axis (recorded in the Marcu
33858848 curation), so a correct two-person rule requires the primary release's
sample_hits and peptides tables, with donor metadata and source hashes. Reuse
canonical Hitlist source/provenance mechanisms where possible; do not invent
patient IDs from per-tissue IEDB records. Official primary-source format:
https://hla-ligand-atlas.org/data (release 2020.12; CC-BY 4.0). Inspect its actual
schema and independent counts before implementation. Preserve exact observed
sequences; no inferred nested epitopes. Document the downstream blacklist as
forbidden contiguous sequences, including within longer antigen segments.

The bundle must include expression-resolution/selection tables, CTA/HPA
reference evidence, peptide summaries, complete mappings, presentation rows,
raw contributors, tissue-risk/blacklist evidence, source metadata, parameters,
versions, and a hash manifest with a verifier. Missing provenance must be
explicit or fail a strict full-provenance request. Write to a new destination
through staging and reject input changes, mismatched artifacts or overwrites.
Bound scans and contributor copies; do not require a second full index build or
load irrelevant raw source payloads into memory merely to export a subset.

## Plan

- [x] Merge and publish PR #646 as 1.64.8; verify exact PyPI artifacts.
- [x] Record corrected repository ownership in tasks/lessons.md.
- [x] Inspect existing exports, contributor bundles, mappings and donor identities.
- [x] Confirm blacklist tissue policy and inspect the official donor-level data.
- [x] File focused Hitlist #647 and link the known modality/identity limitations.
- [x] Finalize CLI/API and schema; checkpoint the concrete plan before code.
- [x] Add failing contracts for identifier/CTA/exclusion and risk/specificity behavior.
- [x] Implement expression selection and complete, bounded evidence export.
- [x] Implement/audit donor-level essential-tissue sequence blacklist.
- [x] Verify full provenance, input stability, manifest hashes and atomic publication.
- [x] Validate real inputs and reconcile blacklist counts independently.
- [x] Document downstream bundle consumption; bump version.
- [x] Run format.sh, lint.sh and test.sh; verify final-head CI.
- [x] Merge, deploy from clean main and verify PyPI artifacts.

## Review

Version 1.65.0 adds expression and standalone blacklist CLI/API exports.
The supplied Atlas snapshot has 967,449 sample hits and 223,246 peptide records;
170,615 heart/brain/cerebellum/lung observations cover 84,754 sequences.
The two-donor rule yields 26,487 forbidden sequences. An independent stdlib CSV
implementation matched every sequence and donor-union count; portable raw-source,
artifact-hash and scientific-relationship verification passes. The snapshot is
not presented as an exhaustive survey of all MS studies. Actual OncoRef PRAME,
MAGEA3, MAGEA4 and ACTB resolution and an Ensembl 112 PRAME transcript were checked;
the complete real CTA/HPA snapshot serializes successfully.

Upgraded the task's isolated environment to pandas 3.0.6, gtfparse 3.0.2,
PyEnsembl 2.24.1; dependency validation passes. Hitlist already permits these
versions. The committed uv.lock also needed a refresh: it now selects pandas
3.0.6 on Python >=3.11 (retaining 2.3.3 on older supported Python), gtfparse
3.0.2, PyEnsembl 2.26.0, datacache 1.16.1 and serializable 1.3.0. Polars
and simplejson are no longer needed in the locked dependency graph (#648). PyEnsembl's separate-session fix is merged. Pandas 2.3.3 compatibility
was checked in a separate environment. Empty mapping selections exposed #649:
Arrow compared large_string against an untyped empty set. The reader now returns
an empty schema-only table without scanning data. The local full `./test.sh` passed (2,611 tests, 1 skipped); final feature checks
pass on both pandas majors. Format/lint and documentation builds pass. A
mutation check proves older-bundle verification requires its captured tissue
policy: removing that argument fails the changed-policy regression. PR #650
tracks final-head CI and deployment. No full local index rebuild is claimed.

PR #646 merged at 9090611; clean-main release run 37530144788 passed all
2,576 unit and 96 integration tests. PyPI 1.64.8 wheel/sdist were downloaded
and their bytes match the tested CI artifacts. No vaccine
assembly is in scope. Existing sample_identity.py deliberately falls back to a
PMID for summary counting; that is not adequate evidence of distinct people for
this blacklist. Initial real-data validation found four Atlas sequences containing
selenocysteine (U); preserve these exact sequences, rather than rejecting the
snapshot or substituting another residue. The current reviewed lineage registry mainly establishes
experimental origins, so its missing donor identities must remain visible.

# Bounded provenance builds — #643 (2026-10-06)

## Specification

Preserve the existing contributor schema, readable identities, complete original
JSON source fields/rows, ancestry relation flags, donor labels, deterministic
ordering and lossless deduplication semantics. Do not change scientific filters
or mark a scoped index as a complete rebuild. Existing published contributors
must survive any failed provenance capture/export.

The current collector duplicates uncompressed source payloads in SQLite, then
executes a corpus-wide recursive UNION/DISTINCT/ORDER BY over those payloads.
The issue reports 41.46 GB of database plus at least 13 GB of SQLite scratch.
Replace payload storage with fast lossless compression and flatten only small
ordered batches of retained roots. Keep visited/frontier sets in explicitly
indexed scratch tables, deduplicate narrow graph identities before loading any
payload, and reuse their pages between batches. Avoid implicit SQLite sort or
recursive work files. Bound SQLite's page cache and output buffers; separately
measure scanner/build memory so no whole-build memory guarantee is inferred.

Use a private, disposable database with journaling disabled and an enforced
max_page_count limit; it is never a recoverable/published artifact. Provide a
documented scratch directory (falling back to TMPDIR), scratch-byte budget and
free-space reserve. Check capacity before allocating and periodically during
capture/export, translate actual SQLite/filesystem full errors, and clean up
owned scratch/partial files. The database cap does not cap the final lossless
Parquet artifact or the observation DataFrames; state this explicitly.

Measure baseline/head in fresh processes on identical deterministic graphs and
a bounded real CSV sample. Compare every contributor cell, ancestry chains,
multiple paths/cycles, donor expansions, retained/excluded roots, empty output,
Unicode/large rows and changed-source checks. Prove resource-limit failures
leave old artifacts intact and the configured scratch cap is enforced. Inspect
query plans for hidden temporary materialization. Use small inputs locally;
do not repeat the known 58 GB failure on this nearly full workstation.

- [x] Read issue, collector/scanner/build flow, project instructions and lessons.
- [x] Create feature branch and record specification/checkpoint before code.
- [x] Benchmark the old collector and add meaningful failing regressions.
- [x] Implement compressed records, bounded traversal and storage controls.
- [x] Verify exact baseline/head parity and report measured time/disk/memory.
- [x] Document configuration, limits and diagnostics; bump the patch version.
- [x] Run ./format.sh, ./lint.sh and ./test.sh; verify full final-head CI.
- [x] Merge PR and deploy from clean main; verify the PyPI release.
- [x] Record review evidence and dependency-ordered follow-up issues (#647, #644, #645).

## Review

Planning checkpoint: change the provenance collector first, where #643 measured
the storage explosion. No performance or full-corpus fit claim until measured.

Two fresh-process trials of each revision used the first 100,000 logical IEDB
rows, identical source payloads and a deterministic 80,000-root graph with
30,000/100,000-row parity comparisons (all columns, including exact JSON text).
This measures contributor capture/export, not scientific classification or a
full-corpus rebuild. Baseline database: 407,969,792 bytes; new: 118,153,216 bytes
(-71%). Peak process RSS: 395,182,080–395,427,840 bytes versus
252,313,600–253,100,032 (-36%). Output: 23,775,314 versus 7,652,532 bytes (-68%).
Capture: 3.90–4.08 versus 5.76–5.77 seconds; export: 4.37–4.65 versus 2.15
seconds. Compression trades capture CPU for space; total collector time is
only modestly faster (8.27–8.73 versus 7.91–7.92 seconds).

Independent graph-oracle tests cover cycles, duplicate edges, multiple labels
and relation paths across batch sizes 1, 7 and 256. SQLite bytecode checks find
no implicit sorter, ephemeral table or automatic index in traversal/export.
Real max_page_count exhaustion and injected filesystem/progress failures retain
old output and remove owned scratch/partials. Large Unicode rows round-trip
within a 256 KiB scratch cap. The collector now writes Zstandard Parquet and
checks both scratch and output filesystems. Fresh-index reuse skips scratch.

Focused scanner/build suite: 85 passed; expanded collector suite: 19 passed.
Format/lint pass. Version 1.64.8 editable-install audit passes in the isolated
repository environment. Local ./test.sh refused before tests: 0.19 GiB free
versus the required 2.5 GiB for one worker. Do not override that guard; full
verification and release use the existing CI workflows with memory preflights.

Filed #645 for the pre-existing non-atomic publication of the contributor,
observation, binding and metadata artifact set. The collector's error translation
is explicitly limited to failures before its publication, avoiding a false claim
that later builder failures preserve an already replaced contributor file.

Release re-plan: the first PR release job failed during dependency resolution,
before tests. Current PyEnsembl main requires gtfparse<3, whereas the workflow
requires current gtfparse master (3.0.2). Existing openvax/pyensembl#451 records
this conflict; an uncommitted `deps/gtfparse-3` checkout already exists, so avoid
changing that work. Keep the development-dependency requirement intact and
resolve the upstream compatibility blocker before rerunning release validation.
Final review also limits capacity-error translation to collector SQL/export
operations, so failures in other build stages keep their original diagnostics.

# Efficient tests, realistic memory budgets and default CTAs (2026-10-05)

## Specification

Resolve #636 through measured causes of excess unit-worker memory and runtime.
The baseline release on 6fc6229 ran 2,556 unit cases in 2,954.85 seconds and
46 integration cases in 606.15 seconds; the original local attempt was stopped
after 467 passes when its worker reached 6,756,736 KiB RSS. Preserve these
observations as distinct measurements, not comparable platform benchmarks.

First audit unit-test corpus access with a bounded read-interception diagnostic
that refuses real observation/binding/mapping reads before materializing them.
Profile representative safe stages twice on the same interpreter, dependencies,
inputs, coverage settings and worker count. Attribute cache retention and slow
calls from measurements. Do not rerun the known oversized suite locally.

Make unit tests independent of installed corpora and persistent user state.
Use small test-owned inputs for unit contracts; retain genuine corpus assertions
in the integration phase. Account for every relocated case and every skip so
isolation cannot silently remove coverage. Add regressions that fail on the
old behavior with a corpus present and protect repeatable unit resource use.

Reduce the measured dominant costs with minimal changes. Set documented
per-worker budgets using comparable baseline/head observations and headroom;
keep unavailable-probe refusal, actual worker budgeting and phase retries.
Do not lower thresholds or discard full release checks to advertise speed.
Record unit and integration timings/peak resource usage separately on the
hosted runner, including the current development dependency revisions.

Inspect release duplication after profiling. Retain publication's clean-main
source, full-suite, corpus, license and exact-artifact provenance requirements.
Prefer eliminating redundant computation inside a run and parallelizing
independent safe work; any cross-run reuse must prove the complete tested
source, dependencies, data and artifact identity, not merely a matching version.

Make CTA support part of the default install through canonical OncoRef, retaining
lazy query-time imports and the existing cta extra as a compatibility alias.
Use OncoRef's packaged panel without copying scientific membership into hitlist.
Verify a normal install's dependency metadata and real CTA query.

- [x] Review project lessons, original issue, baseline logs and current state.
- [x] Create a feature branch and record the profiling/verification contract.
- [x] Audit real corpus reads and measure safe baseline stages twice.
- [x] Re-plan around the measured causes; add failing resource/isolation cases.
- [x] Implement minimal runtime/memory changes and worker budgets; full CI verification pending.
- [x] Make canonical CTAs available in the default install; retain the cta alias.
- [ ] Bump the patch, run format/lint/test, compare full CI stage measurements.
- [ ] Review final-head CI, merge, release from clean main and verify PyPI files.
- [ ] Record final evidence and dependency-ordered next work.

## Review

Planning checkpoint: unit release runs are much slower with the corpus installed,
but runner differences, serial concurrency, coverage and Git dependency heads
are confounders. Real-corpus access is a hypothesis until the bounded audit
identifies actual callers. No memory improvement or speedup is claimed yet.

Measured re-plan: #638 records eight intercepted full peptide CSV reads in
unmarked unit tests. Retain all canonical-data assertions as integration cases,
share the repeated bulk parquet build, and clear bulk caches at the module
boundary before the observations fixture warms. Add small CSV/parquet fixtures
for filter and CLI contracts. Explicitly isolate the empty-index schema and
not-built tests; keep auditing for other hidden reads. Native safe YAML parsing
is 7.4–7.6 times faster in two matched parse-only trials (0.406/0.417 seconds
versus 0.055/0.056); use it with the same duplicate-key guard and test the Python
fallback and unsafe-tag rejection. This is a parser measurement, not a release
speedup. Keep both full release runs; avoid cross-run validation reuse.

The continued audit stopped intentionally after 1,350 passes and six intercepted
reads: two CLI mapping reads, two expression-index reads, the species summary,
and the real supplementary CSV scan. Ordinary tests get a fresh test-owned data
directory, while integration and explicit directory-resolution fixtures retain
their behavior. Thirteen supplementary corpus assertions currently scan the
reference data independently (twelve classified, one unclassified). Share the
classified result without modifying its callers' assertions. Before modifying
the scanner, measure duplicate MHC annotation calls on small matched inputs;
memoize only within one entry and on both restriction and reported class, so
reported-class conflicts and row ordering remain unchanged.

A focused cProfile of the three-row Bourne query finds another dominant cost:
three exports spend 15.69 seconds, including 21,627 iterrows calls and repeated
group construction across every curated study. Restrict attribution candidates
to the PMIDs actually present in the loaded observations before those loops.
Retain every eligible arm and class within each present PMID, plus the existing
full-study discriminator context: filtering unrelated studies must not turn a
multi-arm study into a single-arm fallback. Add an unrelated-study regression,
compare the identical covered query stage twice, and run all existing query
context/attribution parity regressions. This is additional measured scope, not
an inferred biological simplification.

Local evidence before PR: the unrelated-candidate regression fails on the old
behavior (the unrelated allele is scanned twice); restoring the old conftest
makes the corpus-isolation regression fail (one child case fails, integration
and explicit override still pass). YAML-native and supplementary-input reuse
regressions also fail before their changes. The 96 focused query/YAML/CTA cases
pass in 2.34 seconds. The final attribution change reduces the same two covered
Bourne cases from 44.01/42.62 seconds to 6.38/6.40, with identical inputs,
interpreter, dependencies and one serial worker. Stage peaks are 278.1/278.2 MB
afterward versus 295.7/301.5 MB before; these stage measurements do not establish
a full-suite memory estimate. The 100K-row supplementary comparison preserves
the entire frame and reduces 100K record-annotation calls to two; times are
0.724/0.748 seconds before, 0.689/0.662 after. No memory reduction is claimed.

Version is 1.64.7, lockfile refreshed without dependency upgrades, editable
metadata check passes, and format/lint pass. The required local test.sh gate
refuses 0.39 GiB available against its unchanged 2.5 GiB unit budget; it runs
zero tests. Full coverage, resource measurements and release validation move
to hosted CI. Do not treat the preflight refusal or earlier partial diagnostics
as a passing gate. #638/#639/#640 record the discovered causes alongside #636.

Keep the audit's prevention in the suite: the unit fixture also rejects reads
of installed indexes, canonical bulk/supplement CSVs, and external files over
10 MiB before pandas/Arrow allocate them. Test-owned files remain allowed and
integration is unaffected. A child suite with a configured sentinel corpus
proves isolation, explicit overrides, integration access and pre-read refusal.

Platform-budget re-plan before final verification: #603 records a ~19.5 GB
macOS integration physical footprint and large RSS variation under compression.
The Linux 11.6 GiB RSS observation cannot justify the same budget on macOS.
Use 14 GiB for Linux and 24 GiB for macOS/unknown platforms (both overrideable),
and regress the macOS refusal even with 19.8 GiB available. Cancel the obsolete
head's validations and require fresh full gates on this corrected source.

# Correctness gates, categorical reads, CTA provider and QC export (2026-10-05)

## Specification

Ship the reproducible correctness defects in #634, #635, #605 and #619, plus the
existing-metadata export gap in #18, as one patch release. Preserve biological
restriction evidence, sample attribution and raw source QC descriptions.

Memory preflights must fail before pytest when a platform probe is unavailable,
failed or malformed. Distinguish that from measured low capacity; only an
explicit opt-in may run with unknown capacity, using a single worker. Normal
CPU/worker ceilings, serial fallback and bounded low-memory phase retries retain
their behavior. Exercise failed commands, malformed output, opt-in, and loss of
the probe between phases with subprocess stubs that prove pytest did not start.

Audit categorical blank-fills across QC, observations, prediction, query,
reports and exports. Use one shared dtype-safe scalar fill, preserving category
encoding rather than materializing whole string columns as objects. Tests must
cover categorical columns without the blank category, both with and without
nulls, and public consumer paths on pandas 2.x and 3.x.

Migrate the packaged CTA registry and provider to current OncoRef's canonical
cta_gene_names API. Keep it optional, provide an installable extra and actionable
missing-dependency message, and validate against the real packaged OncoRef data
as well as dependency-error/unit fixtures. Avoid hard-coded panel counts or
duplicate biological filtering logic.

Propagate curated search_engine and fdr through observation attribution and
training defaults using the same exact-arm/consensus semantics as other MS
acquisition metadata. Preserve heterogeneous raw FDR strings; unmatched or
disagreeing arms cannot acquire invented QC, and binding rows receive blanks.
Exercise exact matches, ambiguous disagreement/agreement, projection and the
mixed MS/binding training API. Source-reading/backfill remains under #18.

- [x] Inspect current issues, project instructions, release gates and provider API.
- [x] Create feature branch and record implementation/verification scope.
- [x] Add failing regressions for each defect and establish baseline failures.
- [x] Implement minimal shared fixes and review every affected consumer.
- [x] Bump 1.64.5 to 1.64.6; verify editable-install metadata.
- [ ] Run format.sh, lint.sh, test.sh and focused cross-pandas checks.
- [ ] Open PR, verify final-head CI, review and merge.
- [ ] Run deploy.sh from clean main; verify release gates and PyPI artifacts.
- [ ] Record results and inspect relevant sibling issues for the next work block.

## Review

Re-plan after final-head CI: all 46 integration cases passed, but the unit
matrix exposed pandas 3's categorical replace behavior on the final added
sample-discrepancy case. A blank category is not enough for replacing it with
the display placeholder. Add the actual placeholder as a category and use a
value mask, preserving encoding. Re-run the complete focused set on both
pandas 2.3.3 and 3.0.5, then require a fresh matrix and release run on the
corrected head. The original release run is cancelled, not a passing gate.

The corrected placeholder mask passes all 60 focused cases on both pandas
2.3.3 and 3.0.5, including null and existing blank cell names. Format/lint
pass again. The actual OncoRef panel is identical to the provider output and
has no warning-only candidates. Fresh full CI remains required.

The initial regressions found 26 failures. Corrected the species-axis test's
helper name and supplied the prediction/report fixture's required columns;
public consumer tests then reproduce the actual categorical failures. The
baseline QC regressions fail because search_engine/fdr are absent, and the real
OncoRef provider fails on the retired cancerdata import.

The first fixed focused run passes 177 tests (one optional predictor skip).
The shared pandas 2.3.3 environment passes all 11 categorical consumer cases.
The isolated lockfile environment uses pandas 3.0.5 and OncoRef 1.8.207; its
editable metadata correctly reports 1.64.6. Format/lint pass. A sandboxed
test.sh run refuses the unavailable memory probe as intended; the unsandboxed
run establishes 3.70 GB available and starts one worker with unchanged budgets.

Also reproduced and filed #635: TEST_SH_MIN=0 launches serial -n 0 pytest with
zero measured memory. Enforce at least one process's budget, including when
unknown-memory opt-in is set. Full final-head validation and shipping pending.

All 58 focused regressions pass in the isolated pandas 3.0.5/OncoRef environment.
A mutation replacing safe fills with plain Series.fillna makes all seven public
categorical consumer regressions fail on pandas 2.3.3, confirming their coverage.
The local full-suite attempt was deliberately interrupted after 467 passes /
262 seconds: the sole worker reached 6,756,736 KiB RSS against a 2.5 GB preflight
budget, with measured free+speculative memory about 0.08 GB. This is an incomplete
run, not a passing gate or an observed OOM. Full validation moves to CI without
lowering or overriding memory thresholds.

# Restriction-compatible allele candidates (#599, shared #600 checks)

## Specification

Retain reported gene/locus evidence (including HLA-DR, DQ, DP and DRB1) and
derive a separate candidate set from available exact typing. Match the parsed
restriction's species, broad MHC class and gene/locus against parsed candidates.
Explicit restriction species wins over presenting-cell/host/source species;
generic class labels use curated MHC species context when available and retain
unknown species when none is evidenced. Never use mismatch-derived xeno flags.
Species compatibility follows the ontology's ancestor/subspecies relation.

Select the strongest available typing tier before intersecting: peptide-specific
typing, donor typing, then curated PMID pool. An incompatible nonempty tier must
remain unmatched rather than falling through to a broader pool. Accept exact
non-human designations with the same molecule-resolution predicate used for
reported restrictions. Use ontology class membership rather than HLA prefixes.
Reject low-resolution typing, partial pairs and arbitrary free text. Preserve
reported single chains and supplied complete pairs as candidates; never synthesize
partners or cross-product independent alpha/beta typings. A gene constraint may
match its chain within a supplied pair; a locus constraint requires every chain
of a supplied pair to belong to that locus.

New gene/locus inference uses explicit peptide_locus_match, sample_locus_match
or pmid_locus_pool provenance. It never replaces mhc_restriction, derived reported
identity, allele_resolution or restriction evidence, even for a singleton set.
Retain legacy class-only promotion behavior while making its candidate checks
species-aware. Blank restrictions remain unmatched and receive a separate audit
population. Serotype, low-resolution catalog and haplotype expansion remain
follow-ups. Invalidate persisted observation candidates through the artifact
schema version and document that existing artifacts require rebuilding.

Add regressions for all supported loci, DR secondary beta genes, species/class
conflicts, no-fallback precedence, non-human donor sets, partial/full class-II
pairs, scanner and supplement evidence preservation, and blank restrictions.
Build a bounded-memory reproducible before/after audit, with input fingerprints
and code version, reporting rows and distinct peptide sequences per paper and
globally. Distinguish candidate coverage from measured allele assignment and
count blanks separately; do not combine per-paper unique counts into a global
unique count. Verify the Ramarathinam paper scope against primary evidence.

- [x] Read instructions, lessons, candidate pipeline and primary-source evidence.
- [x] Create branch and write the implementation and verification specification.
- [x] Add failing regressions, implement candidate intersection and preserve evidence.
- [x] Add reproducible coverage audit and document measured paper/peptide gains.
- [ ] Run format, lint and test.sh; pass final-head CI and bump patch version.
- [ ] Merge via PR, deploy from clean main, verify PyPI and review next dependencies.

## Review

Implemented parsed species/class/locus intersection with tier precedence and
no fallback on incompatible typing. New locus provenance never promotes the
reported restriction, including singleton candidates and per-donor scanner
splits. Non-human exact typing shares the reported-resolution predicate;
class-II pairs remain intact without synthesizing partners. Genus-level BoLA
context is narrowed to evidenced cattle while explicit engineered MHC survives
an incompatible host context. Observation build artifact version is now 8.

The verified raw ci-corpus-v2 artifact has 4,440,428 rows and the same SHA256 as
the older local snapshot. A bounded replay gains 134,052 gene/locus rows across
29 PMIDs, covering 98,810 distinct sequences; 60,847 previously lacked candidates
anywhere in the corpus. Shared non-human checks gain another 6,164 class-only
rows. Combined first-candidate coverage gains 66,134 global sequences. Every one
of 135,446 blank rows stays unmatched. Counts and all paper gains are documented
in docs/allele-candidates.md; CI uploads the fingerprinted audit report. This
replay does not reconstruct old class-only statements already promoted in the
persisted artifact. Filed #632 to reconcile the paper/deposit coverage difference
for Ramarathinam without inventing evidence.

Focused verification: 419 tests pass. Six scanner/supplement preservation tests
fail when historical promotion is deliberately extended to the new provenance
values, proving they guard the inference/evidence boundary. ./format.sh and
./lint.sh pass. ./test.sh stopped before tests with 0.12 GB available; full CI
and clean-main release remain required. ./develop.sh refreshed the editable
install to 1.64.5 and its metadata check passes after moving the stale local
1.64.4 egg-info aside (known #553 behavior). Merge/release remain pending.

Source-head CI 37023468657 passed Python 3.9-3.12, integration and coverage;
its uploaded audit reproduces every local count. Final review added a regression
for incomplete donor typing falling through to the PMID pool. All three cases
(partial pair, allele group, serotype) failed before the guard. Incomplete
donor designations now block broader-pool locus inference as well as rejecting
invented partners. Final-head CI is required after this correction.

# Literature-backed cross-species experimental context

## Specification

Curate independent primary-literature evidence for the experimental systems
currently suggested by species-axis flags and the packaged study/arm narratives.
Start with a reproducible inventory: existing curated cross-species arms and
PMIDs reached by the current heuristic flags in the available observation corpus.
Use bounded column/batch reads; do not materialize the full export locally.
Record the corpus/version and inventory coverage, including unresolved studies.

Read Methods, experimental figures and supplements for each candidate study.
Distinguish the material actually used for deposited MS observations from other
experiments in the same paper. Keep presenting cells, in-vivo host, MHC origin
and peptide-source proteome separate. A difference in species, a parsed allele
or a mismatch-derived flag is a search lead, never literature evidence. Curate
xenograft passage, transgenic/transfected MHC, cross-presentation/foreign antigen,
native controls and unresolved contexts with their precise sample/arm scope.
Record source URLs/identifiers and section/table/figure locations; paraphrase
rather than copying long source text. No absent mention becomes a negative claim.

After inventory, refine the minimal data/reader schema in this specification
before implementation. Prefer existing curation/export contracts where they can
represent independent evidence accurately. Any new structured field must have
a declared reader and validation, retain unknowns, and reach attributed samples
without broadcasting an arm-specific fact across a PMID. Preserve existing raw
species flags and observations; this pass supplies evidence for later matching,
not an automatic exception inferred from mismatches or a broad #600 rewrite.

The literature result is a separate, packaged `species_contexts.yaml` registry,
not a reinterpretation of the mismatch-derived flags. One record identifies a
PMID plus a stable context ID and optional curated `condition_id` targets. It
records review status (`resolved`, `partial`, or `unresolved`), exact experimental
scope, controlled context kinds, and independent primary-source reference and
location. Its biological axes are distinct: presenting material species, host
species at harvest, earlier lineage host, introduced MHC species, and supported
foreign-source species. Candidate source species from the inventory are recorded
separately, so a reviewed but unsupported candidate does not become accepted.

Expose the complete registry as a table. Attach a record to `ms_samples` and
observation/training metadata only through an exact `(pmid, condition_id)` link;
study-level records without that link remain visible in the audit table but do
not annotate observations. On ambiguous observations, consensus may retain only
axis facts shared by every candidate arm; context identity, review status, scope
and citation are arm claims and must be blanked. Validate schema keys, controlled
values, source requirements, record identity, condition links and reviewed/queue
coverage. The frozen inventory manifest records ci-corpus-v2 / Hitlist 1.64.3,
21 reviewed PMIDs, 95,442 of 96,838 flagged rows (98.56%), 204 queued PMIDs and
423 flagged rows with no PMID. Counts describe review coverage, never truth.

Verify source-to-arm citations and regressions for true xenografts, engineered
MHC, foreign-antigen controls, mixed/unattributed rows and missing evidence.
Run format/lint/test.sh, full CI and a clean-main release; bump at least patch,
merge via PR, publish trusted artifacts and verify PyPI hashes. File discovered
problems on the appropriate repo and link them rather than silently expanding
scope. Record what is established and what remains unresolved.

- [x] Read instructions, lessons and existing species/curation contracts.
- [x] Create curation branch, preserve the correction and write this plan.
- [x] Inventory candidate studies/arms and define the reviewed population.
- [x] Read primary sources and record scoped evidence and unresolved cases.
- [x] Refine the data/reader contract, curate evidence and add regressions.
- [x] Verify coverage, format/lint/tests and pass final-head CI.
- [x] Merge, release, verify published artifacts and review next dependencies.

## Review

PR #631 closes #630 with 29 scoped records across 21 literature-reviewed PMIDs.
Those studies cover every previously curated candidate and 95,442 of 96,838
flagged corpus rows (98.56%). The frozen manifest retains the remaining 204
candidate PMIDs and 423 rows without a PMID as an explicit review queue rather
than silently treating them as resolved.

The registry keeps presenting material, in-vivo host, earlier lineage host,
introduced MHC and supported foreign-source species separate. Evidence reaches
exports only through exact PMID/condition links; ambiguous observations retain
only facts shared by all candidate arms. Literature review corrected PMID
26811146 to human C1R transfectants, separated human and transgenic-rat material
in PMID 29393594, and distinguished PDX-derived culture from patient biopsies in
PMID 39111711. The deposited PDX statement supports two treatment arms and is
therefore retained as explicit multi-arm evidence instead of being assigned to
one arm.

Local `./format.sh`, `./lint.sh`, focused regressions and `./develop.sh` pass.
`./test.sh` correctly stopped at its memory preflight with 0.10 GB available, so
the complete suite ran in GitHub Actions. Source-head run 36962565337 passed the
Python 3.9-3.12 matrix, 46 full-corpus integration tests and combined coverage.
An earlier full-corpus run exposed three unclassified PMID 39111711 statements;
the exact deposited wording and multi-arm case are now pinned by regressions.
Final-head run 36964228569 passed; PR #631 merged as 39ea8f6. Clean-main release
run 36965801751 passed 2,490 unit tests (one skipped), 46 integration tests,
format/lint and artifact validation. PyPI 1.64.4 wheel and sdist hashes/sizes
match the release manifest. Candidate restriction narrowing is the next
foundational block, with haplotype and broader coarse-restriction work following.

# #611 — release completed curated-label metadata

## Specification

Release the full-length `_label_matched` frame immediately after the curated
sample-label coalesce, including when nonempty labels have no curated hits.
Preserve attribution precedence, every returned value and dtype, row/index and
column order, category dictionaries/order, and the public API. Keep the change
minimal: remove the obsolete owning reference after its last use rather than
rewrite matching or allocation behavior. Bump the patch version to 1.64.3.

Extend the existing weak-reference lifetime regression to track the label join
separately from the allele/fallback joins. Exercise an actual curated label hit,
nonempty labels with no hits, blank labels and an absent label column. Check the
exact/fallback/unmatched output and assert completed joins are gone before final
annotation. Demonstrate failure on baseline for both hit/no-hit paths, then pass
all cases with the fix and nearby attribution regressions through test.sh.

Use verified ci-corpus-v2 on Linux for full-population output parity and repeated
RSS measurements. Run baseline 0d0f9ba and candidate in fresh processes sharing
one dependency environment; compare all column value fingerprints, dtypes,
categorical dictionaries/order, index and column order. Record export peak RSS
before fingerprinting so verification does not inflate the measured export.
Repeat each revision twice and report the spread; live Arrow ownership is not
an RSS claim. A temporary PR-only workflow may carry this experiment and its
artifacts; remove it from the final diff after capturing reproducible evidence.
Run format/lint, the complete Python CI matrix and integration suite, then merge
and run the clean-main release gate. Publish only verified release artifacts and
check their hashes/sizes on PyPI. Re-read related open issues after shipping.

- [x] Inspect source, issue requirements, existing regression and lessons.
- [x] Create feature branch, write specification and check in the plan.
- [x] Demonstrate baseline lifetime failures; implement the minimal release.
- [x] Pass focused tests, format/lint and editable version verification.
- [x] Verify full-output parity and repeated Linux RSS with identical dependencies.
- [x] Pass final-head CI, merge and publish verified 1.64.3 artifacts.
- [x] Record review evidence and identify the next dependency block.

## Review

Linux comparison [run 36877044705](https://github.com/pirl-unc/hitlist/actions/runs/36877044705)
compares baseline `0d0f9ba` with candidate `0044eba`. Both fresh-process pairs
passed full-output parity: 4,398,040 rows x 125 columns, all per-column value/null
fingerprints, dtypes, categories/order, column order and index. All four output
fingerprints agree; both runners installed identical dependencies (pandas 3.0.6).
The saved experiment script/workflow and dependency lists are in the run's two
`issue-611-comparison-*` artifacts and remain reproducible from commit `0044eba`.
The final production source is byte-identical to the measured candidate.

| Repeat | Baseline peak RSS (KiB) | Candidate peak RSS (KiB) |
| --- | ---: | ---: |
| 1 | 12,920,532 | 11,829,028 |
| 2, reversed order | 12,932,344 | 11,930,776 |
| Mean | 12,926,438 | 11,879,902 |

Mean peak RSS fell 1,046,536 KiB (8.10%). Baseline spread is 11,812 KiB;
candidate spread is 101,748 KiB. Even the worst candidate is 989,756 KiB below
the best baseline. Export time was 70.5–71.3 seconds across the four runs; no
runtime improvement is claimed. Peak RSS was sampled before output hashing.
This measures standalone full exports, separately from pytest integration RSS.

Both baseline lifetime runs failed exactly the hit/no-hit cases and passed the
absent/blank cases. The candidate passed all eight lifetime/attribution regressions through
`./test.sh`. Local format/lint and isolated editable-install
verification pass. The local test preflight refused at 0.15 GB; no heavy local
work ran. Full final-head CI and the clean-main release gate remain required.

Prior #615/#627 work is shipped in
1.64.2 via PRs #626/#628: release run 36814043601 passed format/lint, 2,475 tests
(1 skipped) and 46 integration tests. Both published distributions matched the
trusted release artifact hashes and sizes.

# #627 — legacy cache status compatibility

## Specification

Unblock the #615 release after current datacache added explicit null `url` and
`sha256` status fields for legacy receipts. Accept absence or None for exactly
those two fields, while checking all remaining status fields exactly. Preserve
the no-network assertion and byte/mtime comparisons for both the cached file
and receipt, plus the assertion that reads create no locks or metadata. Do not
change runtime behavior, suppress failures, or pin away the development build.
Verify the same test against released and current-development datacache, then
run format/lint/test.sh and full CI. Every PR bumps the version, so this narrow
follow-up publishes 1.64.2 including the already-merged #615 changes.

- [x] Inspect the failed release log and current datacache implementation.
- [x] File #627, create a feature branch and check in the revised plan.
- [x] Update the compatibility assertion and verify both dependency versions.
- [x] Pass required checks, merge the follow-up and publish verified artifacts.

## Review

Release run 36809092586 stopped before integration/build/upload: 1 failed,
2,474 passed, 1 skipped. The sole failure is the older exact status shape;
datacache development commit 153c19a returns the two additional null fields.
No 1.64.1 artifacts were published. Original PR #626 passed all PR checks and
merged as 8cffeb7; its filters and 19 new regressions passed the release suite.
The fix passes `./test.sh` with 35 tests against datacache development commit
153c19a and 16 registry tests against datacache 1.15.0. Both ran through the
normal memory preflight in the isolated environment. Format/lint, diff checks
and the 1.64.2 editable-install verifier pass; full CI/release remain required.

# #615 — binding filters in unified training exports

## Specification

Expose `assay_method`, `response_measured`, `measurement_units`,
`has_quantitative_value`, and inclusive `quantitative_value_min/max` as new
keyword-only parameters on `generate_training_table`. Forward these only to
`generate_binding_table`, before composition, mapping expansion and projection.
Keep existing defaults and binding matching semantics: case-insensitive method
substring, endpoint/unit exact membership, and numeric presence/bounds. In
`both` mode MS selection is unchanged; in `ms` mode these filters have no effect.

Share the existing binding CLI option registration with the training CLI and
forward the six options through ordinary exports and bundles. Bundle manifests
already capture effective API parameters; verify the selected filters and rows
are recorded. Preserve endpoint, units, original measurement, qualitative label
and inequality; no endpoint inference, unit conversion, censor reinterpretation
or new restrictions on the canonical indexes. Fix misleading numeric-presence
help text as part of the documented #615 selection semantics.

Document a human MHC-I example with explicit IC50/KD endpoint membership and nM
units. Separate exact numeric targets from censored bounds and qualitative-only
labels; require an explicit downstream policy rather than treating all numbers
as affinity. Cover mixed endpoints, nM non-affinity rows, log units, inequalities,
inclusive bounds, method matching, boolean tri-state, mixed/MS-only modes,
projection/mapping, CLI parsing and bundle option retention with small fixtures.
Use bounded local checks and full CI/release validation to avoid workstation
memory pressure. Bump to 1.64.1, merge the PR, and publish verified artifacts.

- [x] Inspect #615, existing filters, CLI, manifests and project lessons.
- [x] Create feature branch, specify behavior and check in the plan.
- [x] Add failing regressions, implement API/CLI forwarding and documentation.
- [x] Run format/lint/test.sh, review compatibility and verify editable version.
- [x] Pass full CI, merge, deploy from clean main and verify PyPI artifacts.

## Review

Implemented for 1.64.1 with 19 new fixture-based regressions. All three evidence
modes reproduced the unsupported-argument failure before implementation.
Format/lint pass. The surrounding local `test.sh` selection was interrupted
after 15 passing tests when workstation memory became scarce; this is not a
passing run. A subsequent unsandboxed memory preflight refused at 0.10 GB free.
The isolated editable-install verifier passes at 1.64.1. Full validation is
delegated to CI and the clean-main release build.

# #622 / #616 — contributor retention and training lineage (2026-09-30)

Specification: [training-provenance-spec.md](training-provenance-spec.md).

- [x] Inspect requirements, all dedup paths, export/index contracts and lessons.
- [x] Create feature branch and write the implementation/verification spec.
- [x] Capture and retain source contributors through every deduplication stage.
- [x] Add evidence-backed lineage entities, aliases and explicit uncertainty.
- [x] Add the shared export bundle/manifest and split audit API/CLI.
- [x] Verify fixture parity, legacy behavior, tamper detection and known reuse.
- [x] Bump version; run format, lint, test.sh and editable-install verification.
- [x] Open PR, pass full CI, review and follow the release workflow.

## Review

Shipped in PR #625 / 1.64.0. Full Python 3.9–3.12 CI passed. Clean-main release
run 36798035261 passed format/lint, 2,456 tests (1 skipped), and 46 integration
tests; published wheel/sdist hashes and sizes matched the trusted manifest.
Local format/lint and editable-install verification pass;
`./test.sh` focused regression selection: 237 passed. Full matrix and corpus CI
passed. Added #624 for the reproduced blank-identifier data loss and
included its narrow fix. Raw corpus rebuild/performance has not been measured.

#614 shipped in #623 / 1.63.18: PR CI and clean-main
release run 36754196130 passed; both published artifact hashes matched the tested
wheel/sdist. This also shipped the previously merged #613 cache fix.

# #614 — donor-aware training observation identity (2026-09-30)

Specification: [observation-identity-spec.md](observation-identity-spec.md).

- [x] Read #614/#616/#622, relevant lessons, current code and release workflow.
- [x] Create feature branch; write specification and check in the plan.
- [x] Add failing donor/mapping/projection regressions and implement stable IDs.
- [x] Verify the real six-observation reproducer; document migration and limits.
- [x] Bump version, run format/lint/focused tests and inspect the diff.
- [x] Pass full CI, merge and deploy from clean main; verify PyPI artifacts.
- [x] Record review/release evidence and identify the next dependency block.

## Review

The real PMID 31844290/SLLQHLIGL reproducer fails on main with six observations
and two IDs; it passes with six observation IDs and the same two source IDs.
Use readable, percent-escaped source/arm compound IDs following user feedback.
The eight identity tests cover donor separation, mapping/projection, filtering,
row order, display renames, ambiguous/missing curation, categorical nulls and
delimiter escaping. Non-attributed IDs and legacy locator fallbacks are retained.
Format/lint pass. The surrounding export suite passed 343 tests before the final
readable-encoding revision; complete current-head validation is delegated to CI.
The final encoding passes all eight identity tests and 11 focused tests through
`test.sh`. A narrow-column scan of every attributed row in the local index finds
144,408 observations, 81,357 source IDs and 144,408 distinct new IDs. Every row
resolves to a persistent curated arm; none uses the label fallback in this index.
The version is 1.63.18; the editable-install verifier passes in a fresh lockfile
environment after preserving stale build metadata in /tmp and reinstalling this
checkout. Scope is #614; contributor retention and experimental lineage remain
tracked by #622 and #616. Merge/publication evidence is still pending.

# #613 — retain the index just written (2026-09-29)

## Specification

Honor the documented oversized-entry exception after a successful atomic index
write. Pass the exact new cache path to eviction and exclude it from deletion;
mtime alone cannot identify that path because reads also update mtime. Evict
other entries oldest-first until the cap is met or only the protected entry
remains. An oversized index can therefore exceed the cap by itself and still
serve a later disk hit. Bind this eviction to the write's directory. An explicit
prune without a protected entry continues to enforce the cap in the current
directory. Preserve zero-cap disabling, atomic writes, filenames and cache roots.

Use tiny FASTAs and byte-sized synthetic entries to verify cold reload, a single
oversized entry, older entries with misleading mtimes, and ordinary LRU pruning.
Show the cold-reload regression fails before fixing production code. Full suites
and release validation run in CI to avoid workstation memory pressure.

- [x] Confirm #613 is open and #589 is merged and published in 1.63.16.
- [x] Write this specification and check in the implementation plan.
- [x] Add failing regressions and protect the successfully written cache path.
- [x] Bump version, run format/lint and focused tests; inspect the final diff.
- [ ] Open PR, pass full CI, merge and publish from clean main with verified artifacts.
- [ ] Record release proof and start the human class-I training-data PR (#614/#616).

## Review

Three real-write regressions fail on unmodified 1.63.16: a single oversized
index disappears, as does one written beside older entries; misleading future
mtimes retain an old entry instead of the new one. All 21 focused cache tests
pass after the fix, including disk reload with rebuilding forbidden, ordinary
LRU pruning, zero-cap disabling and changing roots mid-write. Format/lint and
the editable-install check pass at 1.63.17; the lockfile uses a dynamic project
version and needs no change. Full CI and publication remain pending.

Training scope is human class-I affinity, processing and presentation;
retain class-II and non-human evidence with explicit scope flags. Stable donor
observations and source-backed experimental lineage precede model-ready claims.

# #589 — datacache consolidation (2026-09-29)

Spec: [datacache_consolidation.md](datacache_consolidation.md).

- [x] Verify datacache 1.14.0 APIs and existing hitlist path/return contracts.
- [x] Write implementation and compatibility specification before code.
- [x] Replace custom transfer/retry/decompression with the datacache adapter.
- [x] Add read-only unified inventory and resumable mirrored assets.
- [x] Test public behavior and interrupted HTTP downloads; bump version/lock.
- [x] Run format/lint and full unit/integration CI; review and merge #617.
- [x] Publish first release (1.63.15) from clean main.
- [x] Migrate versioned registry with legacy path/manifest compatibility tests.
- [x] Run registry format/lint, compatibility comparisons and full unit/integration CI.
- [x] Merge and publish registry release (1.63.16) from clean main.
- [x] Finish curation/training-source assessment; user supplied #614/#615/#616 and
      #18 findings for discussion after #589 ships.
- [x] Review dependency/urgency after upstream publication; return to the user's
      training-data discussion after both hitlist releases ship.

Review: transfer and inventory regressions pass, including real HTTP resume across
separate calls with/without a trusted hash, 429/503 retries, failed-refresh
preservation, raw/expanded archive behavior, offline cross-root inventory,
receipt trust semantics and unchanged legacy files. Both stages pass format/lint,
the Python 3.9-3.12 unit matrix, combined coverage and all 45 corpus integration
tests. Python 3.12 reports 2,405 unit passes for #617 and 2,412 for #620 (two
optional-dependency/data skips each). Integration peak RSS is 13,165,756 kB and
13,098,972 kB respectively. These are validation runs, not new paired memory
benchmarks. #617 merged as 7df2947; its clean-main deployment passed formatting
and lint, then started full tests after the memory retry found 13.95 GiB free.
The local test run was subsequently stopped after the user's memory-pressure
report: 1,185 passes and an ENOSPC failure, not a passing gate. Remaining heavy
checks run in CI. The release-capacity re-plan also addresses #612's measured
timeout by increasing the workflow budget while retaining every check and one
worker. Publication receipts belong in the PR descriptions once PyPI hashes match.
No cache or biological labels were moved.

Publication verified: #617 shipped 1.63.15 after clean-main release run
36612458305 (2,406 unit passes, one skip, 45 integration passes). #620 merged
as 96035cf and shipped 1.63.16 after clean-main release run 36618631704 (2,413
unit passes, one skip, 45 integration passes). Both runs passed format/lint and
the full deploy.sh build gate; published PyPI wheel/sdist sizes and SHA-256 hashes
match the verified CI artifacts. #589, #618 and #612 are closed.

The registry prerequisite, datacache 1.15.0 / openvax/datacache#84, is merged
and published with matching wheel/sdist hashes. It supplies the fixed-path
registry and fixes the independently reproduced lost manifest update (#618).
Hitlist's direct comparison against 1.63.14 confirms matching legacy paths,
returns, status dictionaries and offline reuse. The requested next discussion
is observation identity (#614), dataset lineage/splits (#616), targeted biology
(#593/#588/#599), QC export (#18), and endpoint selection (#615); those defects
are not resolved by this infrastructure work.

# #566 — measure and release retained join buffers (2026-09-28)

The prior PR improved correctness/runtime but did not demonstrate lower hosted
Linux peak RSS. The export retains `allele_matched_df` and `single_matched_df`
until return even after output columns have been overwritten or popped. Measure
whether these references keep distinct Arrow buffers alive at the final peak.

## Specification

Start from pinned main d470c7f and the verified ci-corpus-v2 parquets, using the
same dependency environment for baseline and candidate. Inspect buffer addresses
at the pre-categorical stage to distinguish aliases from distinct allocations.
If ownership confirms the hypothesis, release each temporary reindexed frame
immediately after assigning its columns. Preserve values, nulls, row/index order,
column order, categorical vocabularies/order, filters, and public dtypes. Keep
the fix minimal; no new lazy/chunked public API or scientific-model changes.

Prove full-output equivalence with independent per-column value hashes and dtype
metadata, plus index and category dictionaries. Add a focused lifetime regression
only if it can fail on the old implementation without a fragile timing/RSS
threshold. Run format, lint, unit and integration tests. Compare Linux peak RSS
for pinned baseline/candidate with identical inputs/dependencies; examine swap
if the baseline is constrained by the 16 GB runner. Local live-byte deltas are
diagnostic only, never substituted for the Linux outcome. If this hypothesis
fails, record the measured reason and re-plan before broadening the change.

- [x] Review previous measurements and identify retained join-frame hypothesis.
- [x] Measure distinct buffers retained by local frames on the full corpus.
- [x] Implement the smallest ownership/lifetime fix and verify output equivalence.
- [x] Run format/lint, full unit/integration CI and paired Linux measurements beyond noise.
- [ ] Bump version, open PR, verify current-head CI, merge and deploy from clean main.
- [ ] Record review and triage next relevant work.

## Review

Local capacity re-plan: the shared machine fell below 0.3 GiB free. Clean-main
`./deploy.sh` refused both memory preflights (0.28 then 0.17 GiB); the existing
clean-main Release build workflow now runs the full format/lint/test/build gate
for cache release a6b5704. The duplicate local unit run was stopped after 1,245
passes and is explicitly not a full-suite pass. The corpus probe's unchanged
5 GiB preflight initially waited. A temporary CI comparison was prepared, but
the local budget recovered and the queued probes completed first. The unused
workflow has been removed, retaining the two-line production fix.

Full-corpus comparison: all 4,398,040 rows and 125 columns have equal value hashes,
dtypes, column/index order and categorical dictionaries/order. The two staging
frames hold 2,191,330,805 and 1,814,481,686 distinct Arrow bytes absent from `obs`
at the pre-categorical checkpoint: 4,005,812,491 bytes of obsolete buffers. The
candidate releases those references. This local ownership measurement diagnoses
the mechanism; the Linux process peaks below establish the practical result.

| Full integration run | Baseline peak RSS (kB) | Candidate peak RSS (kB) |
| --- | ---: | ---: |
| first | 15,212,300 | 12,976,868 |
| repeat | 15,162,020 | 13,280,028 |
| mean | 15,187,160 | 13,128,448 |

Mean peak RSS falls 2,058,712 kB (13.56%). All four runs pass the same 45 corpus
tests with identical dependency versions. Candidate spread is 303,160 kB;
the worst candidate is still 1,881,992 kB below the best baseline. Both candidate
revisions have identical export.py content; the repeat includes the cache fix.
All Python 3.9/3.10/3.11/3.12 unit CI gates and combined coverage pass on
0ba961e. Its third integration validation also passes all 45 tests at 13,127,728
kB peak RSS, consistent with the paired measurements. Both PyPI publications
and final clean-main release gates remain required.

Metadata correction: #609's incidental phrase "memory fix #610" automatically
closed #610 on merge. GitHub's ClosedEvent.closer identifies PR #609. The
reference has been corrected to "memory PR #610" and #610 is reopened; this was
not a user cancellation.

The small lifetime regression fails on d470c7f: both completed full-length joins
are still alive when final derived annotations run. Releasing each temporary
immediately after column assignment fixes that failure; all 11 targeted tests
pass on pandas 2.3.3 and 3.0.5. Repeated measurements and full-output equivalence
are recorded above; final release gates and publication remain pending.
The cache prerequisite is PR #609 / version 1.63.13; this memory fix reserves
1.63.14 and will ship after it. The cache branch is now included so final-head
CI also covers both fixes together.

# #591 and #566 — cache resolution and measured build memory (2026-09-28)

Scope: first ship the independently testable proteome-cache fix (#591), then
address #566 with comparable Linux peak-RSS measurements and full-output
equivalence checks. #589's downloader consolidation is a separate follow-up.

## #591 specification

Resolve the default proteome index cache lazily as
`downloads.data_dir() / "proteome_index_cache"`. An explicit
`set_disk_cache_dir(path)` remains highest priority; `set_disk_cache_dir(None)`
restores dynamic resolution, including subsequent environment/data-dir changes.
Use the resolver for reads, writes, eviction, clearing, spawned-worker settings,
and CLI reporting. Path queries must remain silent and create nothing.

Preserve existing unconfigured caches through the existing populated-legacy
rule in `data_dir()`, which already recognizes files in the proteome-cache
subdirectory. Explicit data-dir overrides take precedence even when a legacy
cache exists. Do not move or delete old files automatically; document that an
explicit relocation uses the chosen destination and the old cache remains.

Verify actual tiny-FASTA writes and cold reads, clearing/eviction isolation,
dynamic defaults and explicit overrides, legacy reuse, fresh-install paths,
CLI labels, and real spawned workers. Show new regressions failing on baseline.

- [x] Inspect #591, #566, #602, #589, local instructions and existing lessons.
- [x] Check in the implementation plan before editing code.
- [x] Add failing regression coverage and implement #591.
- [x] Run format/lint and unit CI; bump version and check editable metadata.
- [x] Open #609, verify final-head CI and merge as a6b5704.
- [ ] Run full clean-main release gate and publish 1.63.13 to PyPI.
- [x] Investigate #566 with pinned baseline/candidate and identical corpus/dependencies.
- [ ] Ship a verified memory improvement; record Linux peak RSS separately from live bytes.
- [x] Review next dependent/urgent issues in hitlist and relevant upstream repos.

## Review

The four new cache regressions fail on the original implementation; all 128
targeted cache/proteome/worker/CLI tests pass after the change. Format and lint
pass. Full unit CI passed on Python 3.9/3.10/3.11/3.12 and all 45 integration
tests passed. The private local lockfile run (pandas 3.0.5) was stopped after
1,245 passes under severe machine pressure; it is not a full-suite pass. The
clean-main Release build workflow 36450642510 passed format/lint and 2,391
unit tests (one optional-data skip), but its 37m16s unit phase left too little
time for integration before the job's 45-minute limit. Filed #612 with the
measured phase timings. Local memory recovered to about 16 GiB available, so
the existing `./deploy.sh` is being retried from clean main with two workers,
the verified corpus and every gate intact. Targeted tests also covered pandas
2.3.3.
Next queue: datacache #74 (empty-body rejection) and #75 (inspection/provenance)
unblock hitlist #589's downloader/unified-view consolidation; #611 tracks the
remaining curated-label join buffers without claiming another RSS saving.

Version 1.63.13 leaves 1.63.12 reserved by the already-open documentation PR
#608. The isolated editable-install check passes at 1.63.13.

#602's 84,840 kB difference is smaller than observed run-to-run spread; no
CI-memory improvement is established by that PR. Next hypothesis to measure:
the two full-length join-result frames remain referenced until return, retaining
old Arrow buffers after their columns are replaced on the output frame.

# Batch review — 2026-09-26/27

Asked to pick a path to "fixing all known problems" from 33 open issues and do
it. Shipped fourteen issues across 1.63.0-1.63.7; opened six more, all
pre-existing defects found by reading code nobody had reason to examine.

Released, in order: 1.63.0 HAP1 reference RNA (#358 first half) · 1.63.1 #553
develop-install metadata check · 1.63.2 #579 import no longer deletes
`<data_dir>/index` · 1.63.3 #574 + #563 predictor eligibility · 1.63.4 #576 +
#577 + #582 expression-anchor provenance and index staleness · 1.63.5 #558 +
#559 + #567 + #581 source-verified curation · 1.63.6 #291 datacache data dir ·
1.63.7 #586 sibling MHC context + the engineering-drift audit. #566's
wall-clock axis merged separately and ships no package content.

- [x] Wave 1, four parallel agents on disjoint files: predictor boundary,
  expression anchors, curation, dev-install check.
- [x] Every PR reviewed, most more than once. #584 took four rounds, #587 four,
  #592 two. Every round found something real; none was ceremony.
- [x] #566 promoted mid-batch when it stopped being a nuisance and started
  blocking merges — #584's `test (3.11)` was cancelled twice at the 25-minute
  cap on one unchanged commit. Splitting unit/integration/coverage took the
  test tier 22m56s → 14m40s, and #584 then passed unchanged.
- [x] Deployed each release from a clean main worktree, verifying PyPI hashes
  against the local build every time.

What the review rounds were for, since four rounds on one PR needs justifying:
they caught a 46,247-arm regression (a branch deleted on a reachability claim
that held only for mapped statements), a load-time validator that rejected
ordinary study designs, deposit facts silently dropped from refused rows, and a
remedy string that told curators to write a value the loader rejects. None of
those were visible in a passing test suite. The escalating gate is what found
them: six columns of three studies → every column of three studies → every
column of all 2,319. Each widening found something the previous one could not,
because the population had been chosen by the reasoning under test.

Left open deliberately, with reasons:

- #566 memory axis. The integration phase still peaks at 15.2 GB of 16 GB. A
  separate job is a separate runner, not a bigger one. Closing it means
  trimming the corpus fixture's residency, which is library work.
- #593, 66 findings across 15 studies, recorded as an equality-asserted
  baseline. Two shapes: genuine wild-type arms needing an explicit `none`, and
  different materials grouped together because `sample_group` is unset. The
  second is the majority and its remedy changes attribution (#359), so the
  grouping must be fixed before the tiers. Expression tiers are deliberately
  *not* demoted on silence for exactly that reason — demoting on an unreliable
  grouping would report the wrong line's RNA as a surrogate. The baseline fired
  correctly on first contact when #584 curated three of its entries away.
- #591, a third cache location: the proteome index cache is pinned to
  `~/.hitlist/proteome_index_cache` and ignores `HITLIST_DATA_DIR`. Largest of
  the three at 7,500+ files; moving it orphans an existing cache.
- #588 `soluble_mhc` completion; #572 the object-dtype join, which wants a
  quiet machine and its own diff; #358's C1R/721.221 half, still blocked on the
  ENA read download.
- Upstream: mhcgnomes#199 chicken BF2 aliases, tsarina#180.

Process note worth keeping: four agents sharing one scratch directory
overwrote each other's measurement scripts mid-run, silently corrupting two
corpus replays. Private per-task scratch directories, and a script that states
and asserts which build it loaded.

# #358 — measured HAP1 reference RNA (first half)

Recovered from the uncommitted `/private/tmp/hitlist-358` worktree, whose
committed history was only sibling fixes (#230, #289, #306, #357, #452, #508,
#509, #522, #526), all since landed. Ship only the half with verified data:
HAP1 is present in DepMap 24Q4 (ACH-002475, confirmed against Cellosaurus
CVCL_Y019; default RNA profile PR-QtHaIL), so package its complete 19,193-gene
row with original values, resolve HAP1 at tier 1 and the 12 Shapiro 2025 arms
(PMID 40113210) at tier 2 via the parent, and keep the downloaded DepMap gene
matrix from duplicating it. C1R (SRR7065975) and 721.221 (GSE228188 sgNT
controls) have no certified quantification yet — the ENA read download failed
partway — so their anchors, sources and quantification pipeline stay out and
#358 stays open. Minor bump to 1.63.0 for the new data source.

- [x] Port the HAP1 subset onto a fresh branch from main; drop C1R/721.221
  anchors, sources, `reference_match_tier` and the quantification path.
- [x] Regenerate the packaged CSV from the checksum-verified DepMap matrix
  and confirm it is byte-identical (sha256 d1295f71...).
- [x] Prove the builder source-selection regression fails without its fix.
- [x] Review (#575): fix findings, file what is out of scope.
- [x] Preserve the C1R/721.221 remainder on local branch
  `wip/358-c1r-721221-rna`.
- [ ] Pass format, lint, full tests, build smoke and final-commit CI.
- [ ] Merge, deploy from clean main and verify PyPI.

Review result: removed the stale-index compatibility layer rather than patch
it. It only added newly packaged (line, source) pairs, so an index built by
1.62.x with the DepMap bundle still returned HAP1 twice; it also raised
KeyError on a parquet without `source_id`, skipped builder enrichment and
doubled per-call cost. The root cause (the index is never compared with the
installed packaged inputs) affects every packaged source and is #577. HAP1 is
labelled `depmap_rna` (its values are DepMap's own RSEM row) and the CSV now
carries only value columns, with metadata stamped from `sources.yaml`; values
are unchanged. The report records a decompressed-content hash and a test ties
the committed file to it. A new test proves the bundle still adds HAP1
transcript rows. Engineered labels that match a parental alias resolving at
tier 1 predate this PR and affect 11 curated SaOS-2/HeLa/THP-1/721.221
samples; the curated HAP1 arms are correct. Filed as #576.

# PR #571 review and release

Review the complete diff against main and the failed Python 3.11 CI job.
Check validator acceptance at each supported typing precision, shared
prediction parsing, and the two-field ploidy audit. Reproduce any defect with
a focused regression before changing behavior; file confirmed problems as
issues and link them from the PR. Preserve reported scientific facts and
avoid expanding the curation scope. Keep the existing 1.62.63 patch bump.

Use an isolated release environment with the locked dependencies. Run
`./format.sh`, `./lint.sh`, and `./test.sh`, then verify CI on the final PR
commit before merging. Deploy with `./deploy.sh` from a clean main checkout,
verify the published version and artifacts, and inspect related open issues
to identify the next dependency-ordered block of work.

- [x] Identify PR, read project lessons, and establish the review plan.
- [x] Review implementation and failed CI; reproduce and file findings.
- [x] Fix findings and add focused regression coverage.
- [x] Pass formatting, lint, local tests, and final-commit CI.
- [x] Merge, deploy from clean main, and verify PyPI artifacts.
- [x] Record review results and prioritize follow-up issues.

Review findings: #573 records inconsistent serological typing validation;
preserve source precision and derive only unambiguous serotype loci. #574
records pre-existing predictor eligibility for gene-only / one-field names,
which needs a separate policy for unresolved candidate sets. The failed CI
log shows 2,047 unit tests passing before a runner shutdown during integration;
the existing runner-memory investigation is #566. No assertion failed there.
Nine focused regressions failed against the original PR (including two error
message assertions); after the fix the three focused suites pass: 94 passed,
one optional MHCflurry test skipped, one integration test deselected. Formatting
and lint pass. The full unit suite is running with one worker after the memory
preflight recovered to 11.23 GB available.

Re-plan after CI run 36239737967: all 2,061 unit tests passed, then the
integration process lost its runner with the same shutdown signal as the
original PR. Resolve the #566 gate before merging: run CI integration through
the existing serial fixture path (`-n 0`) so one consumer does not write and
reload a full Arrow cache; keep every integration test and accumulated coverage.
Enable unbuffered output for diagnostics and validate this change on the hosted
runner, recording its actual test results and peak RSS before drawing conclusions.

Second re-plan: serial integration passes all 43 tests locally, but peaks at
19,342,344,192 bytes RSS. Hosted run 36240670574 was SIGKILLed while building
the first corpus fixture at 15,625,788 KiB RSS; avoiding the Arrow cache alone
does not fit the build into the runner's RAM. Provision 8 GiB of swap only for
the corpus CI job, retain all tests and the 25-minute timeout, and log memory
availability. This addresses the runner's capacity while #566/#572 continue to
track production-memory reduction. Require a successful measured hosted run.

Review result: format and lint pass; the locked local environment passes 2,062
unit tests (one optional dependency skip) and all 43 integration tests. Across
all 794 curated samples, prediction candidates are unchanged, as are reported
loci for all 121 currently typed samples. CI run 36241357275 passes on Python
3.9–3.12 plus lint and documentation. The hosted integration run passes all 43
tests in 416.19 seconds with peak RSS 15,127,100 KiB and 10 GiB total swap after
the 8 GiB addition. This is a CI capacity fix, not a production-memory reduction.
PR #571 merged as 0da0efb06589645eb717269f06d9e6c7ca45167f. Clean-main deployment
of 1.62.63 from /private/tmp/hitlist-pr571-release passed 2,062 unit tests and
43 integration tests (one optional unit skip), built and license-checked both
distributions, and uploaded them to PyPI. Downloaded public wheel and sdist
SHA-256 hashes match the tested local artifacts. The release worktree remains
clean. Post-merge CI run 36242284304 also passes, including all 43 integration
tests. Issue #573 closed with the merge; #574 tracks the pre-existing predictor
precision gap separately.

Post-release queue reviewed across hitlist, mhcgnomes, mhcflurry and pyensembl:

- Next foundational block: measure and reduce full-export transient copies
  (#566/#572), benefiting every corpus export and release gate. CI capacity is
  now sufficient, but the production footprint has not been reduced.
- Independent correctness fixes: predictor eligibility #574 and parental
  class-II arm attribution #567. Neither depends on the memory refactor.
- Shared parser dependencies: mhcgnomes #196 broad/split serotype membership
  and #197 retired allele identity, followed by a parser release, dependency
  update, and downstream corpus impact audit.
- Upstream test reliability remains separate: pyensembl #398 shared-cache
  isolation and mhcflurry #436/#437 ineffective tests. These do not block this
  verified release.

# #520 specification — cellular typing and experimental restriction

Keep two biological facts separate. The existing `mhc` / exported `sample_mhc`
is the reported MHC candidate set used to interpret the experiment; it can be
a selected ligand restriction and must no longer be advertised as a complete
cellular genotype. Preserve its attribution semantics and the raw observation
restriction. Add an independently sourced, flat cellular-typing block:
`mhc_genotype`, `mhc_genotype_cell`, `mhc_genotype_complete_loci`, and
`mhc_genotype_source`. Derive reported loci with the existing MHC parser;
missing completeness means unknown, never an absent allele. `mhc_basis`
distinguishes source-reviewed selected restrictions from sample typing;
blank means that this distinction has not yet been curated. Do not infer
genotypes from unreviewed legacy values, pool cells/donors, or add background
alleles to observation candidates merely because they are present in a cell.

Use one field registry for validation and all export paths, including empty
tables, projections, training rows and categorical conversion. Keep a genotype
and its provenance together when candidates disagree. Reuse the existing
precision-aware MHC parser in prediction, removing the duplicate prefix-based
parsers. Predictor inputs must retain the experiment's candidate scope and an
identified sample; an unresolved study pool is not a biological genotype.

Read primary methods for the Lorente C1R example and a bounded engineered-host
audit (C1R, B721.221 and HLA-II constructs), plus the incomplete class-II typing
and moDC/A549 systems called out in the issue. Record verified background
typing, selected restrictions, source locations and remaining unknowns without
silently filling the whole catalogue from cell-line conventions. Source-only
questions outside this contract stay explicit follow-ups.

Prove that adding cellular typing does not change existing row attribution,
reported restrictions or allele-support summaries. Cover partial class-II
typing, distinct cell roles, ambiguous samples, selected-restriction prediction,
and full/filtered/projected exports. Measure corpus impact against main in
bounded batches. Review the implementation specifically for removable branches,
duplicated field lists and redundant parsing. Bump the patch version, run
format/lint/full tests and exact-head CI, then merge and publish verified
clean-main artifacts through the established release workflow.

- [x] Inspect issue, field consumers and baseline model; write this specification.
- [x] Verify primary sources and record the bounded source audit.
- [x] Add regression coverage and implement the shared typing contract.
- [x] Audit complete-corpus behavior and simplify the final implementation.
- [x] Pass format.sh, lint.sh, test.sh and final-head CI; review and open PR (#564).
- [ ] Merge, deploy, verify public artifacts and review follow-up dependencies.

Review: cellular typing is curated on six engineered-host / incomplete-typing
studies (31530632, 31844290, 28228285, 31495665, 33592498, 35051231) from
primary methods, and validated on YAML load. `mhc` values, attribution and
allele-support summaries are unchanged: the curation diff is purely additive.
`mhc_genotype*` travels through ms_samples, observations, training and
projected exports from one registry; an ambiguous join blanks the whole
typing block rather than pooling cells. Prediction now reuses the shared
precision-aware parser (dropping the two prefix-based ones), requires a named
sample, and keeps non-classical molecules out of a class-I predictor batch.

Corpus impact measured against main: zero differing cells across 794 sample
rows x 96 pre-#520 columns and 4,398,040 observation rows x the attribution
columns (sample_mhc, sample_match_type, matched_sample_count,
has_peptide_level_allele, restriction, class).

Review (high) found `mhc_basis` outliving the candidates it describes: both
`_consensus_meta`'s blanking and the class-pool union left the claim attached
to a column that no longer held that sample's `mhc`, and reassignment keyed
its "identified sample" guard on `sample_label`, which a class-pooled row
keeps. Both fixed and covered. PMID 33592498's three pan-class-II arms
claimed `sample_typing` over a DRB1-only candidate list against an HB245
elution that the paper reports yielding DP, DQ and DR ligands; the false
claim is blank and the candidate gap is filed as #565.

# September 24 follow-up — source rosters and attribution (#555, #556)

Resolve the six source/deposit leads in #555 and both attribution defects in
#556 in one focused PR. Establish each study's actual MS sample roster from
its deposited statements and primary methods/supplements: clinical material,
cultured primary cells, transformed lines, natural ERAP backgrounds and
experimentally perturbed arms are different facts. Review PMIDs 32502341,
36215666, 28063628, 26154972, 27846572 and 28228285. Keep missing typing unknown;
do not invent a genotype or reinterpret the selected-restriction contract in
#520. Document verified non-errors as well as corrections and inaccessible
evidence. Exact arm mappings must not resolve shared statements to one arm.

Reproduce Bourne's lost SU-DHL-6 identity and Ritz's first-picked shared
serum/plasma evidence with baseline-failing tests. Sample resolution must use
the study's complete discriminator context, including observations already
matched by allele, and remain invariant under row filtering and output
projection. A discriminator identifying one biological system may narrow its
candidate arms; statements naming multiple biological systems or materials
must preserve ambiguity. Keep the treatment admission guard's protection
against study-wide boilerplate, and retain explicit elution-condition mappings.
Implement the smallest shared resolver change that satisfies those contracts;
do not special-case these PMIDs in Python or change correct YAML to game scores.

Measure before/after attribution over the complete local corpus in bounded
batches, inspect every changed study, and check the affected studies under
full, filtered and projected exports. Preserve reported MHC restrictions and
raw evidence. Add meaningful regression coverage, bump the patch version,
run format/lint/full tests, inspect final-head CI, merge and publish through
the existing clean-main release procedure. Record results in the PR and this
task review; inspect the next foundational work after publication.

- [x] Inspect resolver paths and reproduce both attribution defects on baseline.
- [x] Read all six source/deposit rosters; record corrections and source evidence.
- [x] Implement resolver fixes and source-supported YAML updates with regressions.
- [x] Audit complete-corpus and filtered/projected before/after impact.
- [ ] Run format.sh, lint.sh, test.sh; review diff and final-head CI.
- [ ] Merge PR, publish verified PyPI artifacts and review follow-up dependencies.

Review: the six source studies and complete-corpus impact are documented in
`tasks/curation_followup_555_556.md`. Four baseline regression cases reproduce
#556. All eight source follow-up checks and the two updated source-contract
checks pass; format/lint pass. Corpus comparison covers 4,265,105 curated
observations as 14,120 distinct attribution patterns. Raw evidence/restrictions
are unchanged. The audit caught a shared-HeLa fallback interaction, so explicit
multi-arm mappings now block narrative guessing in both resolver paths. New
source/allele conflicts are filed as #558/#559; #520 remains foundational.
The first full run exposed two inventory-check defects with mixed class-only /
allele typing (#561); both are fixed and the ten source/inventory checks pass.
Final-head CI also exposed the same tokenizer defect in production QC; its
component splitter and build-token gate regression are fixed under #561. All
365 source patterns across the nine affected studies are invariant under
individual peptide filters and output projection. The first run also retained
an already-updated HeLa assertion in memory; it was stopped
and the suite restarted on the final code. Full tests, exhaustive filtered/projected
pattern checks, final-head CI and clean-main publication are in progress; the PR records their final outcomes. Local test.sh passed 1,995 tests (one
skip), all 365 query patterns passed, nine QC checks and two build smoke tests
passed. CI passed every compatibility/unit check, including 1,997 Python 3.11
unit tests (two skips), then canceled the corpus phase without a failed assertion.
During investigation, the full-study variance frame was compacted to distinct
patterns (#562); all 14,120 exported audit patterns remain identical. The cause
of cancellation is unconfirmed. After compaction, all 42 other corpus tests pass; the remaining test found
Ritz's newly ambiguous rows lacked an arm-resolution verdict. Added the source-
measured multi_arm_evidence verdict, extended projection checks to carry it, and
verified the ambiguity-verdict invariant over all 14,120 corpus patterns. Only
Ritz's arm_resolution field changes in that last comparison. Fresh final-head
CI and publication remain gates.

# September 24 curation sanity pass

Audit the complete packaged study/sample inventory for missing or contradictory
gene knockout/knockdown and cytokine condition columns, invalid or implausible
MHC typing, and mismatches between established cell lines, cultured primary
cells and clinical specimens. Inspect actual deposited observation comments
before trusting names or genotype inferred from prose. Treat heuristic flags
as leads, not evidence. Verify actionable findings against original papers or
deposits; retain unknown values and mixed-arm ambiguity when sources cannot
resolve them. Keep cellular genotype distinct from selected ligand restriction
(existing #520), and distinguish study-wide context from per-sample facts.

Correct supported errors in YAML with source provenance and narrow regression
coverage of exported metadata. File each confirmed problem in the appropriate
repository, and record unresolved leads without manufacturing biological facts.
Measure changed sample metadata and observation attribution against main.
Bump the patch version, run formatting, lint and full tests, then review, open
a PR, merge after CI and deploy from clean main. Inspect relevant open backlog
after shipping without silently expanding this audit into unrelated work.

- [x] Inventory all curated studies, sample fields and available evidence.
- [x] Run condition, MHC and source-type checks; source-verify actionable leads.
- [x] File issues; apply supported corrections and regression checks (#554–556).
- [x] Record counts, findings, limitations and before/after impact in audit report.
- [ ] Run format.sh, lint.sh and test.sh; review diff and final-head CI.
- [ ] Merge PR, deploy to PyPI and inspect the next dependent work.

Review: all seven regressions fail with main's curation; all 113 focused checks
pass with the correction. Format/lint pass. The 34,547-row before/after audit
and whole-corpus MHC token check are recorded in
`tasks/curation_sanity_2026_09_24.md`. Only four studies change; unresolved
source and matcher findings are explicitly tracked in #555/#556. Full tests,
final-head CI and publication remain pending. The local memory gate initially
refused tests; after stopping an obsolete audit process, the full suite started
in an isolated environment from uv.lock with one worker.

# September 22 backlog campaign

## #551 specification — isolated, checked NetMHCpan invocations

Two overlapping requests for one allele currently overwrite the same
16-bit-hash-named /tmp input file. A deterministic nested-call reproduction
changes the outer request's peptide from AAAAAAAAA to CCCCCCCCC. Separately,
a failed command's partial stdout is parsed as a valid prediction because
its exit status is ignored. These are invocation-lifecycle defects, not
changes to the biological genotype or ranking model.

Give each allele invocation a TemporaryDirectory and a closed peptide file
within it. Run the existing command with check=True, retaining its arguments,
capture settings and 600-second timeout. Leave existing parsing and allele
identity validation intact. Temporary inputs must disappear after success,
nonzero exit, missing executable and timeout. Preserve empty successful
output behavior. Bump to 1.62.56 after the already-reviewed runtime PR.

- [x] Reproduce input interference and ignored nonzero exit; file #551.
- [x] Add deterministic overlap, failed-command and cleanup regressions.
- [x] All five regressions fail before; implement the minimal wrapper fix.
- [x] Format/lint pass; 122 focused tests pass on Python 3.12 and 121 on
      Python 3.9 (one existing optional MHCflurry import skip); diff reviewed.
- [ ] Pass final-head CI and the full clean-main test/build gates.
- [ ] Merge, deploy both artifacts, and verify public PyPI hashes.

Review: a private temporary directory spans only writing the input and the
checked subprocess call; parsing occurs after cleanup. The failed-exit test
uses a real Python subprocess with partial stdout, and the overlap test
deterministically keeps two requests active without sleeps or timing races.
Existing allele-spelling, empty-output, genotype membership and scoring tests
still pass. No model execution, curation values, binder thresholds, genotype
selection or predictor output columns change. Full CI/publication pending.

## #549 specification — supported GitHub Actions runtimes

GitHub's September 2026 runner migration warns that our checkout, Python
setup, cache and artifact actions still target Node 20. Audit direct actions
and the Pages upload composite against their official release manifests and
migration notes. Update them to current stable Node 24 majors: checkout v7,
setup-python v7, cache v6, upload-artifact v7, upload-pages-artifact v5 and
deploy-pages v5. Coveralls v2 is a shell composite and needs no runtime bump.
The Node 24 actions require runner 2.327.1 or later; these workflows use
GitHub-hosted ubuntu-latest. Checkout's newer fork protections do not affect
our push, pull_request and workflow_dispatch events. Keep artifact archives,
inputs, corpus provenance, Python versions and release source guards intact.

Exercise the documentation build and Pages artifact upload on documentation
PRs as well as main. Restrict actual Pages deployment and write permissions
to the main push job; scope concurrency by ref so a PR cannot cancel a main
deployment. Preserve the prior Pages hidden-file behavior explicitly. Bump
the package to 1.62.55 and append this independent maintenance PR after the
already reviewed releases without changing their heads or active controller.

- [x] Read issue, direct/transitive action manifests and major migration notes.
- [x] Update workflows, version and review the complete diff.
- [x] Run format/lint and actionlint 1.7.12 workflow syntax validation.
- [ ] Verify normal CI, docs artifact and complete PR release workflow.
- [ ] Merge after predecessor publication; verify main docs and release build.
- [ ] Publish both PyPI artifacts and verify their public hashes.

Review: all 13 affected action references now use Node 24 actions, including
the transitive Pages artifact uploader. Official manifests retain archive
uploads by default. Format/lint and actionlint 1.7.12 pass; no local package
or dependency environment was changed. Documentation build/upload, normal
CI and the full release gates will run on hosted runners. No scientific or
curation data changes are part of this maintenance task.

## #541 specification — explicit safe mapping-worker startup

Linux/Python 3.12 release validation reports that the implicit mapping pool
forks a multithreaded parent. Python's ProcessPoolExecutor documentation
recommends a non-fork context in this case. Match the existing supervised
prefetch phase by explicitly using spawn, after #543 transfers parent cache
settings correctly. Keep worker counts, chunk size, ordering, mapping results
and the sequential fallback unchanged. Do not suppress the fork warning.

Exercise the real mapping build and the serialized result contract with a
live parent thread and deprecations treated as errors. The cached-FASTA tests
must use production's chosen context rather than forcing spawn themselves.
Document the usual main-module guard for scripts. Measure startup and peak
child RSS on a bounded cached-proteome workload; report the platform and scope
without claiming a full-human-proteome benchmark or a measured Linux speedup.
Run format/lint and supported-version checks, then full CI and clean-main
release gates. Bump 1.62.54 and ship separately after #544.

- [x] Read official Python 3.12 context/start-method guidance and release warnings.
- [x] Use explicit spawn and exercise actual workers from a threaded parent.
- [x] Compare mapping outputs, startup timing and child memory on cached inputs.
- [ ] Run full gates, review, open PR, merge/deploy and verify PyPI.

Review: format/lint and 55 focused checks pass on Python 3.9 and 3.12. Actual
mapping tests now use the production context with a live parent thread and
DeprecationWarnings treated as errors. A macOS/Python 3.12 audit maps 2,000
distinct 9/15-mer queries per canonical across four cached-proteome tasks from
a 2,011,479-byte Aeromonas bestiarum FASTA. Before/after, sequential and pooled
runs produce the same 8,020 mapping rows, multiplicities, coverage and canonical
task order. Rows within a canonical were already unordered; comparisons sort
all columns rather than equating that incidental order with biology. Two-worker
runs take about 1.9 seconds and peak at about 210 MiB for the largest child in
both versions. macOS already defaulted to spawn: this measures that bounded
path, not a Linux speedup or full-human-proteome memory. Official context guidance:
https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
Full CI and clean-main release gates remain required.

All five pre-restack CI jobs pass, including all 43 corpus checks without
skips. Restack unchanged after the updated cache prerequisite; the urgent
genotype fix #548 ships first. Prove patch identity and repeat final-head CI.

## #543 specification — preserve configured caches in mapping workers

A real cached FASTA maps one peptide sequentially but returns an unavailable
proteome from a spawned worker: set_data_dir() changes process-local state
that the mapping pool does not transfer. Prefetch already transfers the
resolved data directory. Pass the parent's resolved data directory to a
mapping worker initializer. Preserve the separately configurable on-disk
index cache directory too, so fresh workers use the caller's cache settings.
The conflicting-cache reproduction produces wrong protein IDs while marking
the build available, so bump the mapping artifact contract to invalidate these
previously valid but potentially incorrect cached mappings on the next build.
Keep the platform's existing start method, worker counts, task ordering,
chunk size, and mapping algorithms unchanged; explicit safe startup is #541.

Exercise build_peptide_mappings with real FASTA files, a custom directory
override, an unrelated environment fallback, and actual spawned workers.
Compare complete sequential/parallel frames and metadata coverage, including
empty and conflicting fallback caches. No network access or real cache writes
are needed. Bump 1.62.53, review the bounded diff and run format/lint and full
tests before shipping after #542. Keep active releases and dependencies intact.

- [x] Reproduce the real worker cache miss and file #543.
- [x] Add failing sequential/spawned build regressions.
- [x] Transfer resolved cache settings and verify behavior on Python 3.9/3.12.
- [ ] Run full required gates, review, open PR, merge/deploy and verify PyPI.

Review: before the fix, spawned builds return zero rows from an empty fallback
cache and all four wrong protein IDs from a conflicting fallback cache. After
the fix, the complete 12-column mapping frames and per-proteome coverage match
sequential builds in both cases; all four indexes are written into the caller's
chosen index cache. Contract-1/2 artifacts are rejected for rebuild. Format/lint
and 55 focused checks pass on Python 3.9 and 3.12. Full CI and clean-main release
gates remain required; no installed dependency or active release tree changed.

All five pre-restack CI jobs pass, including all 43 corpus checks without
skips. Restack unchanged after #548 and updated #542; require fresh CI and
verify the stable implementation patch before publication.

## #547 specification — preserve biological sample identity in reassignment

The independent reassign-alleles path chooses one global winner per peptide,
then attaches the first observation's sample. A two-sample synthetic example
returns cell_A with an allele present only in cell_B and discards cell_B.
Keep shared peptide/allele scoring, but select a finite-rank winner separately
for each peptide, PMID, sample label and original genotype. Deduplicate repeated
observations of the same context, retain different contexts (including repeated
sample labels across studies), and keep an unscored context with null prediction
fields rather than inventing an allele. Count distinct alleles; use deterministic
tie ordering. Filter peptide lengths before the empty-input fast path. Preserve
the input allele spelling through the per-allele NetMHCpan wrapper so its output
can join back to those exact candidates; allow an empty scored result.

Default API and CLI limits become six distinct class-I alleles, reflecting the
requested per-cell interpretation and MHCflurry's documented genotype interface.
The explicit maximum remains configurable for separately justified experimental
systems; never pool samples or silently truncate a genotype. No curation data,
scientific thresholds, model weights or predictor pair batching changes.

Regression coverage: two genotypes sharing a peptide; shared sample labels in
different studies; repeated observations; row-order invariance; equal-rank ties;
missing/nonfinite scores; no valid-length peptides; six versus seven distinct
alleles; repeated allele tokens; both backend paths and NetMHCpan input spelling;
CLI default and per-context output counts. Check existing pmhc regressions too.
Review the primary upstream predictor contract and biological literature before
the default change. Run format/lint, focused tests on Python 3.9/3.12 and complete
CI/release gates. Bump to 1.62.51 on a branch directly after #525; restack unrelated
pending PRs without changing their implementation. Merge and publish only after
their mandatory gates, and verify both public distribution hashes.

- [x] Reproduce the cross-sample wrong allele on current source; file #547.
- [x] Re-plan release order so unrelated fixes do not delay genotype correctness.
- [x] Add failing regressions and implement context-scoped selection.
- [x] Run format/lint and focused compatibility checks; review the source diff.
- [ ] Run complete final-head CI and the clean-main release gates.
- [ ] Merge, deploy and verify public PyPI artifacts.

Review: the synthetic two-cell corruption is fixed without changing curation.
Predictions are shared only by peptide/allele; winner selection is constrained
by the original sample context. Missing and nonfinite scores produce no allele
assignment. NetMHCpan results are checked with the existing allele-identity
resolver before restoring the input spelling; an unexpected allele fails loudly.
All 117 focused checks pass on Python 3.12; Python 3.9 passes 116 with the existing
optional MHCflurry import test skipped. Format/lint pass. The installed current
MHCflurry source (8b72541) and official Python tutorial explicitly describe one
genotype of up to six alleles and separate results for multiple named samples:
https://openvax.github.io/mhcflurry/python_tutorial.html . DTU's official output
documentation verifies the allele column and its naming conventions:
https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/ . The reviewed biological
background describes HLA-A/B/C and six distinct binding products in an individual
heterozygous at all three loci (Overview of the Immune Response, PMC2923430).
The default remains conservative; it does not redefine nonclassical or engineered
cell genotypes, which remain a separate explicit curation question in #520.

## #540 specification — verifiable public CI corpus provenance

The corpus publisher and both CI consumers transfer only five parquets.
The excluded-study integration check reads observations_meta.json, so its
absence permanently skips that regression. The current local build has
artifact contract 5, matching recorded parquet fingerprints, and zero
excluded-study rows among 4,440,428 observations. All five local hashes differ
from the existing public assets; its metadata must not be copied onto those
different files. Publish a new immutable corpus version after verification.

Add a small standard-library corpus helper. Export only the original artifact
contract version, original mhcgnomes version, a manifest format version and
the five exact file hashes/sizes. Require the existing builder metadata to
match the four tracked parquet size/mtime fingerprints; require contract 5
or later so the #444 check can execute. Never infer a new contract version
from the code running the helper, and never expose private source paths.
Preserve the original local metadata. Snapshot the five files, verify the
snapshot against the public manifest, and refuse to overwrite an existing
corpus tag. Publish the verified snapshot and public metadata together.

Move normal CI and release CI to the new public corpus version, require its
metadata and verify all hashes before tests/cache persistence. Use the
canonical public corpus for fork CI too. Record the metadata hash in release
provenance. The #444 test should require its feature's minimum contract (5),
not equality with the running builder's current contract (6 after #536).
Version 1.62.52 follows the independently shippable genotype fix #548.
The previous reviewed queue is published through 1.62.50.

- [x] Reproduce missing/stale/changed corpus provenance and the version guard.
- [x] Implement sanitized manifests, immutable snapshot publication and CI reads.
- [x] Verify the real local corpus and publish a new matching corpus bundle.
- [ ] Run format/lint, meaningful regressions, workflow checks and fresh CI.
- [ ] Review, merge and deploy after the existing queue; verify PyPI hashes.

Review: the original guard incorrectly skips a valid contract-5 corpus with
builder contract 6. The corrected guard accepts both versions. Manifest and
publisher regressions reject stale builder fingerprints, modified/missing
files, absent/old provenance, private builder metadata, and overwriting an
existing release. The publisher transfers a verified snapshot and preserves
the original metadata. Format/lint and actionlint pass; focused checks pass
on Python 3.9 and 3.12 (47 checks each). The complete guarded local
`./test.sh --all --retry-memory` passes: 1,950 regular tests and all 43
integration tests, with no skips. Its unchanged memory guard observed
5.24 GiB before two regular workers and 8.49 GiB before one corpus worker.
The full integration run therefore executes #444's exclusion regression
against the real corpus. Fresh CI validation remains required after the
public corpus is available.

The user explicitly approved public corpus publication. Published ci-corpus-v2
with exactly five parquets and sanitized metadata, then downloaded the entire
public bundle into a fresh directory. All five SHA-256 hashes and sizes match;
the downloaded metadata is byte-identical to the approved manifest. The
original artifact contract remains 5 and its original mhcgnomes version remains
3.64.1. No original private build metadata was published. Package releases
from the reviewed queue continue independently using v1 until this PR lands.

CI re-plan: two independent Python 3.11 jobs passed all regular assertions,
then the hosted runner sent a shutdown signal after 18/19 integration cases
with two workers. There is no explicit OOM diagnostic, so do not claim one.
The exact corpus already passed all 43 cases locally with one worker. Use one
corpus worker in ordinary CI and cap the full hosted release build at one;
retain every test and the existing memory guards. Add GNU time resource
measurements and repeat both complete workflows before merging. Keep the
independent v1 package-release queue unchanged. File the observed runner
problem and link it in this PR. The unchanged-head retry is stopped. The original PR release build also
received a hosted-runner shutdown signal during its two-worker corpus phase
after 1,949 regular passes; its failed result cannot authorize publication.
The repeated runner failures are tracked in #545.

Further #545 evidence: the one-worker cache PR run passes all 43 corpus tests,
but GNU time reports 15,525,600 KiB maximum RSS (about 14.8 GiB). The provenance
PR's single-worker run is terminated with exit 143. Inspection finds the Arrow
fixture builder still holds its original DataFrame and conversion table while
reading a second frame through mmap. Add a lifetime regression proving these
original buffers are released before the read, then discard them after the
atomic write. Preserve identical mmap-backed output, schemas, locking and
atomicity. Retain conservative concurrency and repeat full CI with resource
measurements; reducing concurrency alone is insufficient near the runner limit.

The new lifetime regression fails on the original helper: the original
DataFrame remains alive when the mmap reader opens. Releasing the DataFrame
and conversion table after the atomic write makes both frame and original
numeric-buffer weak references expire before the read. Existing dtype,
round-trip, source-GC and atomicity regressions still pass. Format/lint and all
67 targeted checks pass on Python 3.9 and 3.12; full hosted resource validation
is required again on this fix.

The next one-worker failures stop after exactly the first 12 integration
cases. Collection order identifies the restored exclusion regression as case
13. Earlier arm/condition tests have already built and retained the shared
full export, but this regression independently constructs that entire export
again before selecting two columns. Reuse full_observations_df and retain the
identical all-row PMID exclusion assertion. This removes the second complete
enrichment without dropping or narrowing any corpus check. Validate the full
43-case suite and compare its peak RSS again before publication.

Final-head pre-restack validation passes all five normal CI jobs and the
separate release workflow: 1,950 regular checks (one optional Ensembl-data
skip) and all 43 corpus checks without skips. Peak RSS remains about
14.1–14.5 GiB; the modest reduction does not establish ample headroom.
Further memory profiling remains tracked in #63. Restack after #548 and
bump to 1.62.52 without changing the corpus implementation or tests; prove
its stable patch identity and repeat complete CI on the new head.

## #538 specification — release artifacts from the existing CI runner

The requested correctness fixes are merged, but local memory repeatedly
prevents the mandatory corpus tests from starting. Use the existing hosted
runner for the full test/build phase and retain local PyPI authentication.
Add `deploy.sh --build-only`, preserving every existing lint/test/build/license
gate and ordinary deployment behavior. Reject unknown options before doing
work, and never label a build-only result as a completed deployment.

Add a release-build workflow with a manual main-branch entry point and a
PR validation path limited to its own files. Require all five corpus files;
missing corpus must fail, not silently skip integration coverage. Install
the six locally developed dependencies from their current development heads
and record their resolved revisions. Run format, require a clean tree, then
the full build-only release with at most two workers and unchanged memory budgets.

Record the source commit, package version, installed dependencies and exact
wheel/sdist hashes in the artifact bundle. Before local upload, require a
successful manual workflow run on the current clean main commit, verify the
bundle against that commit/version, run the existing license checker and
twine metadata check, then upload those exact files and verify public PyPI
hashes. No PyPI credential leaves this machine. Bump to 1.62.39; subsequent
unmerged queue versions must move up before they land.

- [x] Implement build-only behavior and meaningful release-script regressions.
- [x] Add workflow, artifact provenance and rejection checks.
- [ ] Run format/lint, script regressions, workflow validation and CI review.
- [ ] Merge, run the full release build from clean main on CI, publish locally,
      and verify both PyPI artifact hashes before marking the fixes shipped.

Review: format/lint and actionlint 1.7.12 pass. All 22 deployment/artifact
checks pass on Python 3.9 and 3.12; the earlier combined run also passed all
17 unchanged memory/test-runner regressions. Build-only failures propagate
from lint, tests and license verification. Artifact validation checks the
successful manual workflow, exact clean-main commit, package version and
file hashes, then compares the manifest with the actual retained GitHub
artifact. A forged local manifest cannot borrow a real successful run ID.
PR-run builds are deliberately ineligible for publication. Full CI release
validation and clean-main publication remain required.

Artifact review found that `pip install --upgrade` resolved all six Git heads
but retained five same-version PyPI installations. The first workflow therefore
does not prove development-source validation and cannot authorize publication.
Remove those six installations in the disposable runner before installing their
Git refs. Validate VCS provenance immediately after installation and again when
writing the manifest; add a regression for same-version packages without a Git
revision. Repeat the full release build on the corrected head before merging.
All 24 release/deployment regressions now pass on Python 3.9 and 3.12;
format, lint and actionlint pass. Final-head CI and actual main publication
remain pending. The release workflow now reports individual skip reasons.
The first runner had 14.5 GiB available in both phases. Allow the existing
memory/CPU guard to select up to two workers, matching successful corpus CI;
it still falls back to one when memory permits only one, and refuses when
the unchanged per-worker budget cannot fit.

## #511 specification — numeric cell-line identifiers

The group identifier tokenizer drops one-digit numeric suffixes, conflating
SK-MEL-2 with SK-MEL-5 and making both match SK-MEL-28. Preserve numeric
tokens of any length while retaining the existing minimum for alphabetic
words and exact token boundaries. Do not alter the fuzzy arm scorer. Verify
single-line and explicitly multiple-line evidence and existing LM-MEL groups.
Ship 1.62.25 before #457's corrected sample roster depends on these identities.

- [x] Reproduce three single-line failures and preserve ambiguity controls.
- [ ] Fix numeric tokens; run format, lint, tests and review CI.
- [ ] Merge, deploy from clean main, verify PyPI.

Review: three SK-MEL cases fail before the fix; two LM-MEL and two explicit
mixed-line controls already pass. Numeric tokens retain exact boundaries;
the change cannot turn a mixed-line match into a single winner. Format/lint
and the focused non-integration group suite pass. Full tests and CI remain
required before merge.

## #457 specification — source-verified study identities

Re-read primary papers, supplementary sample maps, and every deposited row
description for PMIDs 28834231, 26375851, and 32488085 before editing their
YAML blocks. Correct Ritz's JY arm to MAVER-1 with the deposited six-allele
genotype; remove its unsupported EBV reference. Correct Schellens's pathogen
to measles Edmonston B, retaining honest arm ambiguity where IEDB merges
conditions. Replace Stopfer's unsupported A375 arms with the four melanoma
lines actually profiled, separating vehicle, two palbociclib doses, and IFN-gamma;
retain MDA-MB-231 only for the technical/absolute-quantification experiments.
Do not infer missing HLA typing from another cell line or from prediction.
Measure attribution before/after on all affected observations and pin source
facts and ambiguity behavior in regressions. Change only the three study
blocks plus necessary attribution support if a reproducible defect is found
and filed. Bump to 1.62.26, review, run format/lint/test and CI, merge/deploy.

- [x] Verify papers, supplement sample maps, genotypes and deposited descriptions.
- [x] Correct the three YAML records and update resolution notes from evidence.
- [x] Compare all affected rows and add meaningful regression coverage.
- [ ] Run required checks, review and merge PR, deploy and verify PyPI.

Review: all three source-fact regressions fail against the previous curation;
326 focused checks pass with corrected data. Semantically only the three
study blocks changed. All 7,646 MAVER-1 rows now resolve, the 5,093 HEK293 rows
keep their correct identity, and all 8,024 Schellens rows retain infection-arm
ambiguity. With #511, all 15,821 Stopfer rows identify their cell line; the
3,544 MDA-MB-231 rows reach quantification validation, while all 12,277 melanoma
rows retain unknown treatment pending #512. Found and filed a separate
disjoint-genotype text-attribution defect as #514; it does not affect these
stored rows because their source tissue is populated.

Primary sources: PMC5846733 / doi:10.1002/pmic.201700177 (explicit SSO/SSP
genotypes); PMC4574158 / doi:10.1371/journal.pone.0136417 (Methods, Table 1,
S1 Table); PMC7265461 / doi:10.1038/s41467-020-16588-9 (Methods, Figures 2–6,
Supplementary Data 3 and 5 file maps). UniProt identifies UP000100252 as
measles Edmonston B. Stopfer's main paper also corrects the old instrument,
search engine and invented A375 copy-number claim. Its supplementary raw
headers sometimes say 1uM on 10uM sheets; dose identity was checked against
sheet titles, normalized columns, file maps and the main paper, not inferred
from those inconsistent headers. No quantitative values were ingested here.

## #409 specification — modern license metadata

Replace the deprecated license table and license classifier with SPDX
`Apache-2.0` and explicit `license-files = ["LICENSE"]`. Raise the setuptools
build floor to its PEP 639-supporting release (77.0.0). Retain the same license
text. Add a release-time artifact assertion after `python -m build` and before
upload: both wheel and sdist must declare the expression and contain the exact
repository LICENSE bytes. Validate a real build and demonstrate a corrupted
license is rejected. Release 1.62.24 after #490.

- [x] Update metadata and add the pre-upload artifact verification.
- [x] Build wheel and sdist, check metadata and license bytes, test rejection.
- [ ] Run format/lint/test and CI; review, merge, deploy, verify publication.

Review: a real isolated PEP 517 build produced a wheel and sdist with
`License-Expression: Apache-2.0`, `License-File: LICENSE`, and unchanged license
bytes. The release checker rejects a wheel whose license content is corrupted.
`twine check`, format, lint, and explicit script lint/format checks pass.

Reference: setuptools' pyproject configuration guide documents that SPDX
expressions and `project.license-files` were introduced in 77.0.0:
https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

## #490 specification — strict prediction candidate boundary

PR #492 fixed the common case but still scores a row's entire reported set
when its intersection with the requested alleles is empty. It also compares
canonical row alleles with raw query spellings, making that fallback reachable
for aliases. Use the same normalized, serotype-expanded query set used for
observation filtering as the scoring allow-list. Normalize each candidate
token before intersection. An empty intersection stays unscored and retains
the original evidence restriction; it must never widen to outside alleles.
Each named sample continues to be queried independently. Unfiltered scans
retain their existing candidate behavior; the MHCflurry genotype limit is
unchanged. Release 1.62.23 after #504.

- [x] Reproduce alias, serotype, and no-overlap escapes with regression tests.
- [x] Enforce normalized intersection including the empty-set case.
- [x] Verify mixed scored/unscored rows and independent named genotypes.
- [ ] Run format/lint/test, review diff and CI, merge, deploy, verify PyPI.

Review: six new regression cases failed against the previous implementation;
all 86 pMHC tests now pass. Coverage includes canonical and shorthand allele
inputs, serotype expansion, two independently queried sample genotypes, and
unscored rows on both sides of a scored row. Format and lint pass. The change
does not raise MHCflurry's genotype limit or pool alleles across sample names.

Backlog review: #489 closed after a real eight-pair/two-allele prediction with
development MHCflurry; #482 retired after Table 1 confirmed five benign
patients already correctly curated; #67 closed as implemented by #83/#84/
#90/#103, retaining #95/#96/#361. Observed MHCflurry's 21-key warning is already
tracked upstream in openvax/mhcflurry#425 and #372.

## #504 specification — required allele parser

Core curation imports `mhcgnomes.Species` at module scope, so declare the
existing `mhcgnomes>=3.54.0` requirement in base dependencies. Preserve the
`alleles` extra as an empty compatibility alias for existing install commands.
Keep the installed-version floor check, correcting its optional-extra wording.
Refresh the lockfile without upgrading unrelated packages. Validate wheel
metadata and a clean base install's curation import and CLI, plus required
format/lint/test and CI gates. Release 1.62.22 after #501 ships.

- [x] Update dependency declaration, diagnostic wording, and lockfile.
- [x] Verify built metadata and isolated base installation.
- [x] Run format, lint, test; review PR and CI.
- [ ] Merge, deploy from clean main, verify PyPI publication.

Review: built wheel declares unconditional `Requires-Dist: mhcgnomes>=3.54.0`
and still provides the `alleles` extra. A fresh environment installed only the
wheel (no extras), imported curation from site-packages, and ran `hitlist --help`.
The 10 floor-check tests, format, lint, and `uv lock --check` pass. Retained all
existing locked package versions and unrelated resolution entries. All 1,659
non-integration tests pass; lint and all four Python CI jobs pass. Reviewed
the dependency graph and confirmed that the base requirement matches the
existing unconditional import. Rebased onto merged #505 without code changes;
the new PR head must also finish CI before merge.

## Scope and release contract

Review every open issue against current code and source evidence. Each issue
requiring changes gets its own feature branch, patch-or-greater version bump,
reviewed PR, passing CI, merge, and deployment from clean main. Retire obsolete
issues only with evidence. File newly discovered dependency defects upstream
and link them from affected PRs. Preserve unrelated local files and branches.

## Ordered plan

- [x] Establish an isolated environment using current development revisions of
      all locally developed libraries in the dependency graph; record revisions.
- [x] #501: remove obsolete dependency guards before relying on coverage.
- [x] #504: declare the existing mandatory mhcgnomes runtime dependency.
- [ ] #490, #489: preserve queried biological genotypes in scoring, then correct
      the MHCflurry presentation API call without weakening its genotype limit.
- [ ] #456: audit retired allele aliases and preserve reported provenance.
- [ ] #409, #291, #289, #306: packaging, cache, errors, and schema foundations.
- [ ] #482, #457, #452, #314, #230: primary-source-backed curation corrections.
- [ ] #357, #358, #140, #56: expression coverage and sample associations.
- [ ] #67, #95, #96, #361: bulk proteomics and detectability training data.
- [ ] #63, #176: profile and improve the remaining build bottlenecks.
- [ ] #46, #40, #39, #37: experimental-system, source, and attribution models.
- [ ] #18, #24, #41, #42: acquisition, quantitative and custom-database evidence.
- [ ] #33, #35, #36, #7, #8, #13, #14: source-verified remaining study curation;
      reconcile umbrella issues with their children before claiming completion.

## #501 specification

Verification changed this plan: metadata calls `mhcgnomes` optional, but
`curation.py` imports `Species` unconditionally at module scope. Remove the
obsolete try/except, all 15 early returns, and four remaining conditional test
branches. Assertions must execute unconditionally; an unavailable required
dependency must fail collection loudly. Track the metadata mismatch in #504
instead of adding another unsupported optional path. CI installs the allele
extra on every test job. Bump 1.62.20 to
1.62.21, run format/lint/test, review the diff, create PR, check CI, merge, deploy,
and verify the published version before moving to the next issue.

## Review and release log

- Initial state: main at 62ad259, version 1.62.20, 38 open issues, no open PRs.
- The existing deploy script publishes the current version; despite the older
  AGENTS wording it does not accept a version or perform bump/commit/push. Keep
  version bumps on feature branches and use the actual release behavior.
- Dependency environment: isolated `.venv`, development snapshots verified
  against upstream HEAD: datacache ee20b5a, pyensembl 4392376, gtfparse cb3788e,
  serializable 19c38ce, mhcgnomes 8d8f30a, mhcflurry 8b72541. The latter is
  upstream master, not the unrelated feature branch in the sibling checkout.
- Confirmed openvax/sercol#4 still blocks a combined development install with
  serializable 1.1.0 and added the resolver reproduction upstream. sercol and
  mhctools are not in hitlist's dependency graph and are not needed for its
  subprocess-based NetMHCpan integration; no constraints were bypassed.

# Issue #478 — ArrowTypeError merging serotype dictionary columns

## Objective

`build_observations()` crashes rebuilding the MS observations index inside
the cross-source concat step, reported from tsarina with a full
reproduction: `pa.concat_tables(ms_tables, promote_options="default")`
raises `ArrowTypeError` because two per-source partitions' `serotype`
column ended up dictionary-encoded at different index widths (`int8` vs
`int16`). Blocks any fresh build; does not affect an already-built parquet.

## Root cause

`_compress_categoricals` runs once per source partition (IEDB, CEDAR)
before the Arrow conversion, so pyarrow picks each partition's dictionary
index width from *that partition's own* distinct-value count independently
(<=127 categories -> int8, more -> int16). `promote_options="default"`
promotes null/missing columns but does not reconcile two dictionaries of
the same value type at different index widths.

Not scoped to `serotype` alone: all 28 columns in
`_CATEGORICAL_BUILD_COLUMNS` go through the identical per-partition
compression before the same two `pa.concat_tables` calls (MS and binding),
so any of them could hit this the moment one source's cardinality happens
to straddle the boundary and another's doesn't.

## Design

- `promote_options="permissive"` (pyarrow's own documented next tier up)
  widens mismatched-but-compatible types -- including dictionary index
  width -- to a common denominator, and still rejects a genuine mismatch
  like `int64` vs `string`. Verified both properties directly against
  pyarrow before trusting it.
- New `_CONCAT_PROMOTE_OPTIONS` module constant (both call sites now read
  from it) documents why, and gives tests one source of truth instead of
  a literal string that could silently drift from what production uses.

## Plan

- [x] Reproduce the exact `ArrowTypeError` with minimal synthetic tables
      before touching any code.
- [x] Empirically verify `promote_options="permissive"` fixes the dictionary-
      width mismatch AND still rejects a genuine `int64` vs `string` type
      mismatch (i.e. it widens compatible types, not relaxes validation).
- [x] Confirm via `_CATEGORICAL_BUILD_COLUMNS` that the same per-partition
      compression risk applies to all 28 categorical columns, not only
      `serotype` -- the fix is at the concat call, so it covers all of them.
- [x] Fix both `pa.concat_tables` call sites (MS, binding).
- [x] Regression tests, verified against the unfixed value before trusting
      them: the exact failure pinned, the fix confirmed end to end through
      the real `_compress_categoricals`, and a boundary test proving the
      fix doesn't also start accepting truly incompatible types.
- [x] Full `test_builder.py` + build-smoke tests (exercise `build_observations`
      end to end) pass.
- [x] Full combined-suite run twice; PR, CI, merge, deploy.

## Review

Reproduced the exact `ArrowTypeError` with two lines of synthetic pyarrow
tables before touching any code, then again through the real
`_compress_categoricals` helper to confirm the mechanism, not just the
symptom. Checked `promote_options="permissive"`'s actual boundary before
relying on it: it correctly merges `int8`/`int16` dictionary indices of the
same value type, and correctly still rejects `int64` vs `large_string` --
confirmed both directly against pyarrow, and pinned the second as its own
test so a future pyarrow upgrade that changed this behavior would be caught
rather than silently trusted.

Verified the new test suite is not merely trivially passing: with the
constant's value reverted to `"default"`, the fix test fails with the
issue's own literal error message; restored, it passes.

No artifact-version bump: this fixes a build-time crash, not any stored
column's values or meaning -- an existing artifact built before this
change needs no invalidation.

Full combined-suite run (`pytest -n 5 tests`, the same invocation
`test.sh --all` and `deploy.sh` use) plus the build-smoke tests that
exercise `build_observations` end to end: 1649 passed, 1 skipped, twice in
a row -- 1646 from before plus the 3 new tests, zero regressions.

# Issue #470 — CEDAR column misparsing

## Objective

CEDAR's export carries one more column than IEDB's (`Epitope | Mutation`
inserted at index 27), and three columns' name candidates never had a working
match on either source, so all three silently fell back to IEDB's positional
index -- correct by luck on IEDB, one column off on CEDAR. Two measured,
silent consequences: a binding assay (T2 stabilization) entering the
eluted-ligand output because the misread `assay_comments` comes back empty,
and mono-allelic cell-line detection never firing on CEDAR rows because the
misread `cell_name` returns an IRI instead of a name.

## Root cause (deeper than the issue itself identified)

`_resolve_columns` combines the two header rows as `f"{cat} | {fld}"` before
matching. `_COLUMN_NAMES["antigen_processing_comments"]`,
`["assay_comments"]`, and `["cell_name"]`'s candidates never included the
`" | "` separator, so they never matched EITHER source's real header (real
category is non-empty: "Antigen Processing" / "Assay" / "Antigen Presenting
Cell") -- confirmed empirically against both registered real files before
writing any fix. They "worked" on IEDB only because `_FALLBACK_INDICES` was
calibrated against IEDB's own column count. `assay_iri`/`ref_iri` have the
identical bug for a different reason: CEDAR literally names its first two
columns "CEDAR IRI" instead of "IEDB IRI".

## Design

- Add the missing pipe-separated candidates for all four broken keys.
- After the fix, verified all 27 `_COLUMN_NAMES` keys resolve by name alone
  on every known real/simulated layout (real IEDB, real CEDAR, and CEDAR with
  the exact reported extra column) -- the positional fallback is now a pure,
  never-triggered safety net for a genuinely unrecognized header.
- `_resolve_columns` now raises `ValueError` (naming the unresolved keys,
  the column count, and nearby header text) instead of silently using
  `_FALLBACK_INDICES` -- issue's suggestion #2, made safe by the point above.

## Plan

- [x] Verify the bug against real registered IEDB + CEDAR files (the local
      CEDAR predates the extra column, so also simulated the exact reported
      113-column layout by inserting it programmatically).
- [x] Fix the four broken `_COLUMN_NAMES` entries.
- [x] Fail-closed guard in `_resolve_columns`.
- [x] Fix 21 pre-existing scanner tests broken by the guard (three shared
      CSV-writing helpers relied on silent fallback for columns they didn't
      bother naming) by adding the missing names, not weakening the fix.
- [x] New regression tests: name-resolution position-independence, the
      exact three broken keys on the exact reported layout, the fail-closed
      guard, and two full `scan()` end-to-end tests reproducing both
      consequences verbatim from the issue (T2-stabilization binding assay,
      HMy2.C1R mono-allelic detection) -- verified each new test genuinely
      fails against the unfixed code with the exact reported symptom before
      trusting it.
- [x] Format, lint, full combined-suite run twice; PR, CI, merge, deploy.

## Review

Every new test was checked against the unfixed code before being trusted, not
just written and assumed correct: `git stash` on `hitlist/scanner.py` alone,
re-run, confirm each fails with the exact reported symptom
(`is_binding_assay` False instead of True, `is_monoallelic` False instead of
True, index 88 instead of 89, no raise where one is now expected), then
restore.

The first version of the new tests used blank category headers (matching the
existing `_write_tiny_iedb_csv` convention) and all five passed against the
UNFIXED code -- a false-negative regression suite. The real bug specifically
needs a non-empty category (`"Antigen Processing"`, `"Assay"`, `"Antigen
Presenting Cell"`) to expose the missing `" | "` separator; blank categories
let the old single-string candidates match trivially via the field-only
fallback pass. Rebuilt the header from real category/field text measured
directly off the registered IEDB file, keyed by real column index, with
CEDAR's extra column insertable at its exact reported index (27) -- this is
what makes the "before" run fail with the issue's own literal symptom
language rather than a synthetic proxy for it.

Fixing the fail-closed guard's blast radius took more than the four
`_COLUMN_NAMES` entries: 21 of 33 existing scanner tests relied on three
shared CSV-writing helpers that left most columns unnamed, silently accepting
whatever `_FALLBACK_INDICES` guessed. Extended each helper to name every key
(placed past the columns each helper's rows actually populate, so no existing
`row[N]` assignment needed to change) rather than softening the guard to
tolerate unnamed columns -- the guard's whole point is that an unnamed column
must not resolve silently.

Full combined-suite run (`pytest -n 5 tests`, the same invocation
`test.sh --all` and `deploy.sh` use): 1646 passed, 1 skipped, twice in a row
-- 1641 from before plus the 5 new tests, zero regressions.

# Issue #467 — mhcgnomes version-floor tripwire

## Objective

An installed `mhcgnomes` below the floor `pyproject.toml` declares fails the
suite with 200+ scattered `AttributeError`s across 15+ files (e.g.
`Species.compatible_with` missing below 3.39.0), with nothing connecting any
one traceback back to the actual cause. Reported from a release install of
3.33.4 shadowing a correctly-locked 3.64.2 in a shared environment's
site-packages. Add a tripwire that turns 200+ cryptic failures into one
message naming the declared floor, the installed version, and the resolved
import path.

Deliberately scoped to the tripwire the issue actually asks for, not to
modifying the shared environment itself: that environment is used by other
concurrent sessions and several sibling repos, and a version bump there is
someone else's call, not a fix this repo's code can make unilaterally. The
per-worktree `uv run` workaround already documented on the issue remains the
correct way to get a conforming environment; this tripwire is what makes the
*next* person who hits a shadowing install able to diagnose it in one line
instead of reverse-engineering it from a curation.py traceback.

## Design

- `tests/mhcgnomes_floor_check.py`: pure, independently-testable functions
  (`declared_floor`, `floor_violation_message`, `check`) mirroring the
  existing `tests/xdist_cache.py` pattern -- parsing/comparison logic lives
  outside conftest.py so it has a public surface to unit-test.
- `tests/conftest.py`'s new `pytest_configure` hook calls `check()` once,
  before collection, and `pytest.exit()`s the whole session with the
  message if it fails -- no per-test overhead, no chance of 200 confusing
  failures burying the one line that explains them.
- Version comparison via `packaging.version.Version` (already a transitive
  dependency), not string comparison -- `"3.9.0" > "3.54.0"` lexically,
  backwards, so a naive string floor check would have the wrong sense for
  any single-digit-vs-two-digit minor version pair.

## Plan

- [x] `tests/mhcgnomes_floor_check.py`: parse the floor, compare, message.
- [x] Wire into `tests/conftest.py`'s `pytest_configure`.
- [x] Unit tests: parsing (real file + synthetic + missing-declaration),
      message content, numeric-not-lexical comparison, the real
      environment's happy path, and the `pytest_configure` wiring itself.
- [x] End-to-end proof, not just a mock: a fake `mhcgnomes==3.33.4` package
      shadowed onto `PYTHONPATH`, reproducing the exact reported scenario,
      confirmed the whole session exits on one message instead of running
      into 215 failures.
- [x] Format, lint, targeted + full combined-suite run; PR, CI, merge,
      deploy from clean main.

## Review

End-to-end proof, not just a mock: built a fake `mhcgnomes` package declaring
`__version__ = "3.33.4"` and shadowed it onto `PYTHONPATH` ahead of the real
3.64.2 install -- an exact reproduction of the reported scenario. The whole
session now exits immediately with:

```
Exit: installed mhcgnomes 3.33.4 is older than the floor this project
declares in pyproject.toml's `alleles` extra (3.54.0).
  Resolved from: .../fake_mhcgnomes_shadow/mhcgnomes/__init__.py
  ...
```

instead of collecting and running into 215 scattered `AttributeError`s.

10 new unit tests cover the parsing (real pyproject.toml + a synthetic
string + a missing-declaration failure mode), the message content, numeric
vs. lexical version comparison, the real environment's happy path, and the
`pytest_configure` wiring itself (via `pytest.exit.Exception`).

Full combined-suite run (`pytest -n 5 tests`, the same invocation
`test.sh --all` and therefore `deploy.sh` use): 1641 passed, 1 skipped --
1631 from before plus the 10 new tests, zero regressions.

Not attempted: upgrading the shared virtualenv itself. That environment is
shared across concurrent sessions and several sibling repos; a version bump
there is a decision for whoever owns those other workloads, not something
this repo's code can safely make unilaterally. The tripwire is the part of
#467 that is actually this repo's to fix.

# Consolidate the isolated-curation test fixtures (#473 follow-up review)

## Objective

`/code-review` on #473 (the fixture-cache-leak fix) found no correctness bugs but
confirmed four real cleanup findings in the fix itself: an unguarded `finally`
that could mask a real test failure, redundant `try/finally` ceremony around a
`yield`-fixture pytest already tears down unconditionally, the same ~13-line
docstring and clear/yield/clear block duplicated across two files, and two
pre-existing tests now carrying manual cache-clear calls the fixture itself
already guarantees. Fix all four together.

## Design

- New shared `_isolated_curation_root` fixture in `tests/conftest.py`: copies the
  packaged curation YAML into an isolated temp tree, monkeypatches
  `curation._data_path` / `cell_name_parser._registry_path`, and clears every
  curation cache on setup and teardown via a plain `yield` (no `try/finally` --
  pytest already runs a fixture's post-`yield` code unconditionally, so wrapping
  it adds a masking risk with no added guarantee).
- `test_builder.py`'s `isolated_curation` and `test_cache_current.py`'s
  `curation_referencing_uncached_asset` become thin fixtures depending on the
  shared base, each owning only the fake `pmid_overrides.yaml` content and
  monkeypatches specific to their own tests.
- Removed the now-redundant manual `cache_clear()` calls in
  `test_cache_tracks_new_attribution_reference` and
  `test_curation_change_rebuilds_stored_evidence`.

## Scope note

Five more tests in `test_curation.py`/`test_exclude_from_ms.py` hand-roll a
narrower, single-file `_data_path` monkeypatch with their own correctly-scoped
`try/finally` cache_clear -- already safe, unlike the two fixtures this PR
touches. Left alone: a DRY improvement there would touch several already-correct,
unrelated tests for a purely stylistic gain, disproportionate to the risk.

## Plan

- [x] Shared `_isolated_curation_root` fixture in `tests/conftest.py`.
- [x] Both fixtures refactored to thin wrappers; unused imports removed.
- [x] Redundant manual cache-clear calls removed from the two dependent tests.
- [x] Verify the exact confirmed regression pair from #474 still passes.
- [x] Format, lint, targeted tests, full combined-suite run twice; PR, CI,
      merge, deploy from clean main.

## Review

`ruff check --fix` removed one now-unused `Path` import in `test_cache_current.py`
as a direct consequence of the refactor (the shared fixture owns that copy logic
now); no other changes needed to satisfy lint.

Verified in order: both fixtures' own tests (78 total, unchanged pass count) still
pass; the exact two-test pair that reproduced #474's regression still passes;
`pytest -n 5 tests` (the combined invocation that originally exposed the leak)
passed 1631/1631, twice in a row.

# Fixture cache leak found while deploying 1.62.4

## Objective

The 1.62.4 deploy (merge of #472) failed its own `./test.sh --all` gate with 23
failures, all in `test_curation.py`/`test_exclude_from_ms.py`, none touching code
`#472` changed. GitHub Actions CI on the same commit was green on all four Python
legs. Root-caused and fixed before retrying the deploy.

## Root cause

`deploy.sh` runs `test.sh --all`, which mixes integration and non-integration
tests in one `pytest -n 5` invocation with no `-m` filter. CI's Python 3.11 job
keeps them in two separate `pytest` invocations (`-m "not integration"` then
`-m integration`), so this bug had no way to surface there.

`tests/test_cache_current.py`'s `curation_referencing_uncached_asset` fixture (added
for #448) monkeypatches `curation._data_path` to an isolated single-PMID YAML tree
for the duration of one test, and never clears any curation cache. That was safe
when written: nothing in `observations_cache_is_current()`'s call graph touched
`curation.load_pmid_overrides()`'s `lru_cache`. #471 changed that —
`supplement.load_supplementary_manifest()` now calls `ms_excluded_pmids()`, which
calls `load_pmid_overrides()` — and `_source_fingerprints()` (part of the
predicate's call graph) reads the supplementary manifest. So the fixture's test
now populates the REAL, process-global `load_pmid_overrides` cache with its fake
single-PMID data, and nothing clears it afterward. Under `-n 5` with the full
~1630-item collection, whichever worker draws this test early in its queue then
serves every subsequent real-data curation test on that worker from the fake
cache for the rest of the run — reproduced deterministically (though the exact
failing set varies run to run, since xdist's dynamic scheduling isn't) down to a
minimal two-test repro:
`test_cache_current.py::test_observations_predicate_never_downloads` followed by
`test_curation.py::test_pmid_mono_allelic_override`.

`test_builder.py`'s pre-existing `isolated_curation` fixture has the identical
shape (patches `_data_path`, no fixture-level teardown) and has stayed safe only
because its three current tests each remember to clean up manually. Same bug
class, same fix.

## Plan

- [x] Bisect: confirmed via a minimal two-test repro, not a full-suite guess.
- [x] Harden both `_data_path`-patching fixtures to `yield` + unconditional
      `curation._clear_curation_caches()` teardown, so the next function that
      gains an indirect `load_pmid_overrides()` dependency can't reopen this.
- [x] Verify the exact confirmed pair passes; verify both fixtures' existing
      tests still pass; re-run the full `test.sh --all`-equivalent invocation.
- [x] Format, lint; PR, CI, merge, retry the deploy from clean main.

## Review

Root cause confirmed with a minimal two-test repro before touching anything:
`tests/test_cache_current.py::test_observations_predicate_never_downloads` followed
by `tests/test_curation.py::test_pmid_mono_allelic_override`, run together with no
xdist, fails identically to the deploy log. The fix (both fixtures now `yield` +
unconditionally clear curation caches in `finally`) turns that pair green.

Verified against the actual failure mode twice: `pytest -n 5 tests` (deploy.sh's
exact invocation, no `-m` filter) passed 1631/1631 both times, where main at
`c4fa4cc` failed 23 (first run) and 7 (second run, verbose) -- the varying failure
set across runs is itself evidence this was xdist scheduling exposing a real
process-global cache leak, not a fixed collection-order bug.

Not a production bug: `curation._clear_curation_caches()` already lists
`ms_excluded_pmids` (added correctly in #444/#466). The gap was purely in two test
fixtures that monkeypatch `_data_path` without a teardown, which stayed invisible
until #471 gave `load_supplementary_manifest()` an indirect path to
`load_pmid_overrides()` that didn't exist when the #448 fixture was written.
CI never caught it because the workflow keeps integration and non-integration
tests in two separate `pytest` invocations; `deploy.sh`'s `test.sh --all` runs
them together in one `-n 5` pass, which is exactly the condition needed to expose
a same-worker cache leak.

# Issue #471 — post-merge review findings on #466 (exclude_from_ms)

## Objective

`/code-review` on merged PR #466 surfaced 11 findings: a validation gap that lets
`exclude_from_ms` silently no-op, two QC signals that never learned about the new
filter, a supplementary-data contradiction with no cross-check, a second code path
that bypassed the filter entirely, two cleanup items, two test-coverage gaps, and
two doc corrections. Fix all of them in one release.

## Plan

- [x] `curation.ms_excluded_pmids()`: reject a non-boolean `exclude_from_ms` value
      loudly instead of silently treating it as "not excluded".
- [x] `qc.cross_reference()` / `curation_plan()`: skip excluded PMIDs so they stop
      reading as `yaml_only` gaps and inflating priority.
- [x] `qc._default_curated_mhc_samples()`: skip excluded PMIDs in the token audit.
- [x] `supplement.load_supplementary_manifest()`: raise if a manifest entry's PMID
      is also curated `exclude_from_ms: true` — the two curations disagree about
      whether the study is an elution experiment, and silently dropping the
      hand-vetted rows a few build steps later hid that.
- [x] `report._run_report_from_csv()`: apply the same MS-scoped exclusion the built
      corpus gets, so `--from-csv` stops diverging from `hitlist report`.
- [x] `builder._drop_masked_rows()`: shared skeleton for `_drop_short_mhc2_rows` and
      `_drop_excluded_from_ms`, closing the missing `.head(5)` cap on the latter.
- [x] Structural schema-guard test: a field not self-declared unread must be
      referenced somewhere outside its own `PMID_ENTRY_FIELDS` declaration, with a
      synthetic negative case proving the check isn't vacuous.
- [x] Restore the `"#444" in description` assertion dropped from the replacement test.
- [x] `tasks/todo.md`: fix the stale `(1.62.0)` heading to `(1.62.3)`, and note
      format/lint passed in #444's own review section, matching the prior entry.
- [x] Format, lint, full non-integration and integration tests; PR, CI, merge,
      deploy from clean main.

## Review

All 11 findings fixed with direct regression coverage for every new branch, not just
incidental coverage from existing tests.

**Validation (#1).** `ms_excluded_pmids()` now rejects a non-boolean `exclude_from_ms`
with a `ValueError` naming the PMID, instead of `is True` silently treating `1` or
`"true"` as "not excluded".

**QC signals (#2, #3).** `cross_reference()` (and therefore `curation_plan()`, which
consumes it) and `_default_curated_mhc_samples()` (the token audit's default sample
source) both skip `ms_excluded_pmids()` now. Verified against the real corpus: nothing
raises, and a direct unit test proves an excluded study's curated arm no longer reads
as a `yaml_only` gap while an unrelated study's genuine gap still does.

**Supplementary contradiction (#4).** `load_supplementary_manifest()` raises if any
entry's PMID is also curated `exclude_from_ms: true` — the two curations disagree
about whether the study is an elution experiment, and the alternative (silently
dropping the hand-vetted rows after `scan_supplementary` already logged them as
added) is exactly the kind of silent contradiction #436 and #444 both warn about.
Verified clean against the real packaged manifest (21 entries, 11 exclusions, zero
overlap) before adding the guard.

**Report path parity (#5).** `--from-csv` now applies the same MS-scoped exclusion the
built corpus gets via a new `_drop_excluded_ms_rows`, careful to only drop the
MS-classified subset — `df` here is the raw mixed scan, unlike the builder's
already-split `obs`, so a binding row for an excluded PMID stays either way.

**Cleanup (#6, #7).** `_drop_short_mhc2_rows` and `_drop_excluded_from_ms` now share
one `_drop_masked_rows` skeleton, which closes the missing `.head(5)` cap on the
exclusion loop as a side effect of not duplicating it.

**Test coverage (#8, #9).** Added the structural half of the schema guard: any
`PMID_ENTRY_FIELDS` entry not self-declared unread/informational must have its exact
name referenced somewhere outside its own declaration, with a synthetic negative case
(`_fields_claiming_a_reader_without_one`) proving the check actually catches an unwired
claim rather than passing vacuously. Restored the dropped `"#444" in description`
assertion.

**Documentation (#10, #11).** `#444`'s review section now notes format/lint passed,
matching the prior `#462` entry's convention; its heading cites the version it actually
shipped as (1.62.3, not 1.62.0 — three other PRs landed and bumped the version between
when that section was drafted and when it merged).

Full non-integration suite: 1580 passed (0 new failures). Targeted suite across every
changed module plus all new tests: 407 passed, 1 skipped (a real-corpus integration
test that predates artifact_version 5 locally — pre-existing, unrelated to this PR).
Coverage confirmed on every new branch: the validation raise, the contradiction raise,
`_drop_excluded_ms_rows`'s three return paths, and both new QC skip checks are all
exercised by a test, not just reached incidentally.

# Issue #454 — one duplicate-key-rejecting loader for every curation YAML

## Objective

`UniqueKeyLoader` (#450) guarded `pmid_overrides.yaml` and `condition_vocabulary.yaml`;
the other nine packaged YAML files still loaded through plain `safe_load`, where a
duplicated key silently discards the first value. All eleven parse clean today, so this
is a guard-rail change with no data edits and a byte-identical corpus.

## Design

- New leaf module `hitlist/curation_yaml.py` holding `UniqueKeyLoader` and
  `load_curation_yaml(path_or_traversable)`. It imports only PyYAML, so the light
  registry/download modules use it without pulling in the curation stack, and
  `conditions` drops its lazy import that dodged the `curation` cycle.
- `hitlist.curation.UniqueKeyLoader` stays importable under its original name.
- Every `yaml.safe_load` / `yaml.load` call in the package (12 sites, 9 modules) routes
  through the helper. A test greps the package for any direct PyYAML parse outside the
  leaf module, and a parametrized test loads every packaged YAML through the guard.

## Plan

- [x] Leaf module + re-export; route all twelve call sites.
- [x] Tests: rejection, path/traversable inputs, packaged-file sweep, bypass guard.
- [x] Document the guarantee in `docs/curation-process.md`.
- [ ] Format, lint, tests; bump the version, PR, CI, merge, deploy from clean main.
# Issue #468 — the warm-up deadline stops paying for process startup

## Objective

`_supervise_prefetch_tasks` started its wall-clock budget before building a **spawn**
pool, so the cost of starting a fresh interpreter was charged to a deadline that exists
to bound *network* stalls (#402, #407). On a loaded machine that could consume the whole
phase without a single fetch attempted, and it made
`test_prefetch_supervisor_terminates_blocked_inflight_call` depend on runner load — it
failed on the Python 3.11 leg of #465, a PR that touches nothing in `mappings.py`.

## Plan

- [x] Start the clock once the pool exists, so the budget measures warm-up work.
- [x] Regression that fails on the old code with the exact CI message: a context double
      whose `Pool` takes longer than the deadline, asserting the work was still attempted.
- [x] Prove the in-flight timeout test is deterministic again.
- [ ] Format, lint, full tests; PR, CI, merge, deploy from clean main.

## Review

The new test reproduces the CI failure verbatim against the unfixed supervisor
(`assert 'timed out' in '    prefetch deadline of 0s exhausted; skipping 1 proteome(s).'`)
and passes with the clock moved. Behaviour on either path was already correct — every
proteome is reported unavailable — so no corpus or artifact effect.

# Issue #448 — public, quiet cache-validity predicates

## Objective

Give library consumers (tsarina, presto) a side-effect-free way to ask whether the
built `observations.parquet` / `binding.parquet` set and the `peptide_mappings.parquet`
sidecar are current, so they can announce a rebuild before spending ten minutes on it
instead of buffering `build_observations()`'s status block and inferring afterwards.
`observations.is_built()` / `mappings.is_mappings_built()` stay existence-only.

## Design

- `hitlist.observations.observations_cache_is_current() -> bool | None`
  - `None` when no IEDB/CEDAR source is registered: validity is unknowable, and
    `build_observations()` would raise rather than answer.
  - Otherwise the same verdict `build_observations(force=False)` reaches before
    deciding to skip: artifact version, source + curation fingerprints, parquet
    fingerprints. Nothing printed, nothing written.
  - Must not download. `_curation_fingerprints` resolves `peptide_attributions`
    assets through `packaged_or_fetched`, which fetches a missing externalized CSV
    on a wheel install. The predicate resolves through a new
    `downloads.packaged_or_cached` instead and reports the cache stale when an
    asset is absent — which is what a build would conclude too, since fetching
    stamps a fresh mtime.
- `hitlist.mappings.mappings_cache_is_current(*, release=112, fetch_missing=True,
  use_uniprot=False, flank=DEFAULT_FLANK) -> bool`
  - Same contract check as `build_peptide_mappings(force=False)`; keyword defaults
    mirror the builder's, with a test that pins them together.
  - Returns `bool` only: the sidecar is either stamped against the observations on
    disk or it is not, so there is no unknowable case to encode as `None`.
- Both names join `_PUBLIC_API`; README documents them beside `is_built`.

## Plan

- [ ] `downloads.packaged_or_cached`: local-only twin of `packaged_or_fetched`.
- [ ] Thread `fetch_missing_assets` through `_curation_fingerprints`,
      `_source_fingerprints`, `_cache_is_valid`; default unchanged for the builder.
- [ ] Add the two predicates, export them, document them.
- [ ] Tests: `None` without sources; True/False on seeded caches; silent on stdout;
      never fetches; mapping defaults pinned to `build_peptide_mappings`.
- [ ] Smoke-test against the local `~/.hitlist` cache (artifact_version 3 vs code 4).
- [ ] Bump to 1.62.0, format, lint, test, PR, CI, merge, deploy from clean main.

# Flat experimental-condition columns

## Objective

Revise `tasks/condition-model-spec.md` around one consistent, flat condition schema that
curators can fill from papers and that reaches individual rows of the unified training table.
Keep source evidence, unknown values, and ambiguous peptide attribution explicit. Implement,
verify, and ship the resulting change through a versioned PR and PyPI release.

## Plan

- [x] Audit existing condition strings, metadata readers, and representative primary sources.
- [x] Write the flat schema, paper-extraction contract, migration, and acceptance criteria.
- [x] Implement validation, categorical annotations, and propagation through every export.
- [x] Curate the existing condition vocabulary and source-verified representative samples.
- [x] Verify combinations, controls, missing information, ambiguous assignments, and projection.
- [x] Run format, lint, tests, and a before/after corpus comparison; inspect the final diff.
- [ ] Bump the version, open the PR, verify CI, merge, and deploy from clean main.
- [ ] Record results and review the next relevant open issues.

## Review

Shipped as #450. 23 flat condition columns declared once in `hitlist/conditions.py`
and spliced into the loader schema, the samples row, the empty-frame schema, the
expression-anchor projection, the observation join and the training defaults.

**Curation.** All 761 arms annotated from the 167 distinct condition strings, written
into the YAML as reviewed data (`curated_text`), every pre-existing value byte-identical.
19 arms across four studies upgraded to `primary_source` with section-level locators.

**What the pilot caught.** Reading the papers disproved three curated facts that no
consistency check could see: Stopfer 2021's biopsies are snap-frozen not fresh, and
its `condition_control: untreated` asserted a therapy status the paper never states;
Javitt 2019's A549 genotype carried `HLA-B*07:02` where both the paper and IEDB's own
rows say `HLA-B*18:01`. Lorente 2019's arms are correct but incomplete (5 profiled,
2 curated) — filed as #452 rather than expanded here.

**Attribution.** The tie guard now compares curated `condition_id` instead of the
coarse category, so 184,811 rows stop being assigned an arm they were never entitled
to. `_candidates_disagree_on_arm` deliberately stays on the category — that gate asks
a different question, and identity there cost 7,629 correct discriminations.

**Filed, not worked around.** #451 (class-pool path leaves ungrouped rows blank
instead of `pmid_ambiguous`; the fix moves 1.23M rows and needs its own verdict pass),
#452 (Lorente arm split).

Corpus: 4,439,321 rows before and after. 1,532 tests pass including integration.

---

# Arm attribution cluster — #442 / #366 / #359 / #364 / #362

## Release split

Approved plan: three sequenced PRs, each version-bumped and deployed before the next branches.

1. **1.58.9 — #442 QC check, study-level schema guard, #362 close-out.** Code and docs only;
   the corpus must come out byte-identical.
2. **1.59.0 — #359 + #364.** `sample_group` curation vocabulary plus group-aware attribution,
   then the arm curation that depends on it.
3. **1.59.1 — #366.** `arm_resolution` accounting across all 28 ambiguous studies.

## What the investigation changed

Re-measuring on the current corpus before writing code contradicted two of the five issues.

- **#362 is already fully delivered**, not partly. Its two proposals (carry the matched sample
  onto observations; denormalize the APM block) are live, and its remaining "consumer caveat" —
  that a WT control inherits the study panel's gene flags — describes pre-#353 behavior.
  Verified on the Shapiro HAP1 panel: `HAP1 wildtype` reports `apm_genes_perturbed=""` and
  `apm_perturbed="false"` while `study_apm_perturbed` stays True. Documenting that caveat would
  have described a bug that no longer exists, so PR 1 pins the correct behavior with a test and
  closes the issue instead.
- **#359's premise is false.** A subagent ran the counterfactual end-to-end: making PMID
  29242379's arm labels symmetric yields `""`, not `pmid_ambiguous`, because the class-pool path
  has no `_consensus_meta` fallback; and the TIL/meningioma rows stay unattributed because
  `assay_comments` is blocked wholesale when candidate arms disagree. Labels alone make the
  output strictly worse, so PR 2 needs `sample_group` and a matcher change.
- **The asymmetry bug is a live mis-attribution, and wider than filed.** PMID 29242379's 3,919
  "attributed" rows land on the untreated arm because `ovarian` matches `source_tissue = Ovary`
  and appears on only that one of six candidates. PMID 30833945 has the identical pathology via
  `lung`. 8,595 rows are confidently attributed to a control arm on no evidence.
- **#366's biggest study is unresolvable by construction.** PMID 33858848 is 255,179 of the
  450,704 ambiguous rows; its arms are per donor and its evidence records tissue. Orthogonal
  axes, unrecoverable from IEDB.

## Steps

- [x] Re-measure all five issues on the current corpus; run the #359 counterfactual.
- [x] PR 1: `qc.sample_attribution_audit` + CLI, `PMID_ENTRY_FIELDS` guard, #362 close-out test.
- [x] PR 1: PR #445, CI green, merged, deployed 1.58.9, wheel verified; #442 and #362 closed.
- [x] PR 2: `sample_group` + group-aware attribution; curated all four studies.
- [x] PR 2: PR #446, CI green, merged, deployed 1.59.0; #359 and #364 closed.
- [x] PR 3: `arm_resolution` across all 31 ambiguous studies.
- [ ] PR 3: PR, CI, merge, deploy 1.59.1, close #366.

## Review

### PR 1 — #442 + #362 + the study-level guard (1.58.9)

`qc.sample_attribution_audit()` reports the 232 profiled arms that reach zero observation rows,
bucketed by whether the study attributes anything at all: 137 `label_mismatch_candidate` (the
join demonstrably works in that study, so the label is the suspect) and 95 `study_unattributed`
(a different failure, triaged per study). Deliberately not wired into `run_all` or
`curation_plan` — `sample_label` is synthesized by the export join, so this is the only check
needing the full enriched table, and either rollup would make every `hitlist qc` pay that build.

**#444 found while adding the guard.** `exclude_from_ms` is documented as excluding a study from
the MS index and is set on 11 studies, every one curated as *not* a mass-spec elution experiment
— yeast display, peptide microarray, refolding crystallography, computational tools. Nothing
reads it. 40,355 rows / 33,101 unique peptides from 6 of those studies are in the corpus and
reach the enriched export. Three more study-level keys are unread: `donors` (11 studies),
`samples` (2), `tissues` (1). `PMID_ENTRY_FIELDS` now declares all 37 permitted top-level keys
against their readers and the loader rejects the rest; the unread ones are declared as UNREAD
citing #444 rather than described as if they worked. `samples`/`tissues` renamed to
`n_samples`/`n_tissues` per the count-suffix rule.

Removing those 40,355 rows is a deliberate corpus change and ships separately.

### Review fixes folded into PR 1

`/code-review` on the branch returned 12 findings, 8 of them real defects in the #373 provenance
work already published as 1.58.8. Each was verified against the corpus before fixing.

- **`effective_override` skipped the `rules` level.** PMID 27846572's `primary fibroblasts`
  exported `cell_line` / `study` while the study's rule sends every Direct Ex Vivo fibroblast row
  to `healthy` — exported provenance contradicting the classification the build applied, on
  131,252 observation rows across 14 studies that carry both `rules` and `ms_samples`. Rules match
  per row, so a sample-level value cannot resolve them; the new origin `study_conditional` says
  the inherited value is a default a rule may supersede, instead of asserting finality.
- **`_consensus_meta` kept arm-specific claims on arm-less rows.** Arms of one study routinely
  agree on `override` (PMID 34129938 marks all six `cell_line`), so consensus preserved
  `origin="sample"` on rows whose `sample_label` it had just blanked. Latent only because those
  PMIDs have no rows; it would have broken `test_no_unattributed_row_carries_a_sample_level_override`
  on the next corpus refresh. Study-origin values still survive — those are deposit properties.
- **`_SAMPLE_PROVENANCE_COLUMNS` / `_TRAINING_DEFAULTS` not extended** — `--with-expression-anchors`
  silently dropped all six new columns, and binding rows got NaN where every other MS-only column
  gets `""`.
- Plus: a join docstring asserting the opposite of the join's behavior, an un-coerced legacy
  `notes` beside three coerced siblings, and a test using `.` as a regex stand-in for a literal `+`.

Three design findings taken: `note` → `sample_note` on observations (it collided with the
study-level `note` key *and* the `notes` column), `sample_override` dropped from the 4.4M-row join
as derivable from the other two, and a new test tying every declared `MS_SAMPLE_FIELDS` key to an
exported column — the old one only checked descriptions were non-empty, so "declared" could have
become a synonym for "accepted and ignored", the exact failure #373 exists to prevent.

Corpus effect: 13 sample rows move `study` → `study_conditional`; zero changes to observation row
identities. Gates: format, lint, 1,432 tests, build smoke.

### PR 2 — #359 + #364 (1.59.0)

`sample_group` names the sample *system* an arm belongs to. Attribution resolves the system
first — admitting IEDB's narrative fields, because identifying a system is what they do reliably
— then the arm within it, where those fields stay blocked. Opt-in per study and enforced
all-or-nothing at load, so a half-curated study cannot silently fall back.

The plan's counterfactual held: symmetric labels alone would have produced `""`. What makes the
study attributable is the group stage plus a `_consensus_meta` fallback on the class-pool path,
which the allele path already had.

| study | before | after |
|---|---|---|
| 29242379 Chong | 3,919 attributed, all to the wrong arm | 11,450 arm-resolved, 70,878 system-resolved |
| 30833945 Javitt | 4,676 attributed to the wrong arm | 7,524 system-resolved |
| 32938616 Faridi | 37,643 condition-only, panel-level | 37,643 line+condition, 54,526 line-resolved |
| 27371725 Nagarajan | 1,334 rows with no arm | all 1,970 reach a curated arm |

Corpus-wide: unattributed 1,630,156 -> 1,493,039; `discriminated` down exactly 8,595, the false
control-arm attributions and nothing else; `elution_conditions` unchanged at 74,304. Row
identities identical, and only the four curated PMIDs moved.

Two source findings changed the curation from what the issues assumed. **Nagarajan's classical
restrictions are NetMHC predictions**, not measurements: the study acid-eluted the whole cell
surface with no allele-specific pulldown, then assigned H2-Kb/H2-Db by NetMHC and Qa-2a by
Rankpep. Only Qa-1b was experimental, and no Qa-1b rows are in the corpus — so the entry sets
`restriction_evidence: predicted`, the vocabulary #415 exists for. The paper's real title covers
classical MHC, and IEDB was right that the cells are bone-marrow-derived dendritic cells, not the
curated "splenocytes". **Faridi ran three lines**, not a panel; LM-MEL-53's HLA type is stated
nowhere in the paper, so it stays class-only rather than inheriting LM-MEL-44's on the strength of
being the same patient.

Implementation note: the token scorer drops tokens under three characters, so `LM-MEL-44` and
`LM-MEL-33` both reduced to `mel` and tied — the digits that distinguish them were invisible. The
group selector now tries an alphanumerics-only containment test first (`lmmel44` inside
`lmmel44melanocyte`), per-row factual fields before narrative ones, falling back to token scoring.
That recovered all 37,643 elution-resolved rows, which the first attempt had cut to 14,617.

### PR 3 — #366 (1.59.1)

Every one of the 584,966 ambiguous rows now carries a recorded reason. Zero unexplained.

| verdict | rows | means |
|---|---|---|
| `axis_mismatch` | 255,179 | curated arms and recorded metadata on different axes (33858848: donor vs tissue) |
| `curation_gap` | 145,279 | a per-row discriminator exists; the only verdict marking real work |
| `no_row_discriminator` | 129,982 | measured — all four per-row fields take one distinct value across the study |
| `multi_arm_evidence` | 54,526 | the evidence positively places the peptide in more than one arm |

**75% of the ambiguity is settled**: a property of what was deposited, not of how carefully
anyone curated. That is the answer #366 wanted, and the reason a five-value vocabulary beat the
planned three — the data showed three genuinely different kinds of unresolvable, and calling
32938616's "peptide really was in both arms" case *unresolvable* would have been wrong.

Two guards stop a verdict becoming a way to stop looking. `arm_resolution_note` must carry the
measurement, and a note without a verdict is rejected at load. And a test re-measures every
`no_row_discriminator` study against the corpus: if a refresh gives one a varying per-row field,
the verdict is stale and the study is re-audited rather than silently trusted.

`hitlist qc sample-attribution --actionable-only` hides findings in settled studies, so the #442
audit distinguishes "unresolvable" from "unexamined".

Purely additive: one new column on the samples export, no pre-existing sample or observation
value changed.

---

# Sample-curation conservation and primary-source audit — #438 / #437 / #436 / #373

## Release split

The audit turned one issue into four with a hard dependency order, so it ships as three PRs
rather than one. Each bumps the version and deploys before the next branches from main.

1. **1.58.6 — #438 + #437, inventory conservation.** Every raw `ms_samples` record must survive
   loading and reach the sample export. This is foundational: #436 and #373 both reason about
   per-sample records, and today five of them do not exist as far as the loader is concerned.
2. **1.58.7 — #436, verified curation corrections.** Data-only fixes to the four studies whose
   sample curation source verification disproved.
3. **1.58.8 — #373, sample-level `override` / `note` semantics + schema guard.** Enabling a
   sample-level override before #436 would promote wrong curation into row classification, so
   this lands last.

## Specification

### PR 1 — #438 + #437 (1.58.6)

- `load_pmid_overrides()` rejects duplicate `pmid` identifiers before building its mapping,
  naming every duplicated key. Today a dict comprehension silently keeps the last entry.
- Consolidate the two duplicate pairs into one entry each, preserving both the study/sample
  metadata of the earlier block and the `source_organism` / `species` curation of the later one.
  Keep the exact `species` strings the scanner currently reads so no observation row changes.
  Correct the two wrong study labels the duplicates introduced (verified against PubMed:
  33460454 is Gastaldello 2021, not "Owen 2021"; 28188227 is Barnea 2017, not
  "Alvarez-Navarro 2018").
- `generate_ms_samples_table()` keeps `n_samples: 0` / `profiled: false` records instead of
  dropping them, exporting `profiled="false"` and a null `n_samples`.
- The observation metadata join excludes unprofiled samples explicitly, so restoring the
  metadata cannot manufacture a peptide observation or an arm match. (Binding needs no change:
  `generate_binding_table` never joins `ms_samples`.)
- Regressions: raw-YAML-to-loader and loader-to-export inventory conservation, duplicate
  rejection, unprofiled round-trip, join exclusion, and unchanged observation row identities.

### PR 2 — #436 (1.58.7)

- SKMEL5 +/- binimetinib replaces the A375 +/- trametinib curation for PMID 34497125.
- Add the MC38 idAdpgkG IFN-gamma / doxycycline / dTAG-13 arms for PMID 34129938.
- Add the WT/mock, TAP1-KO/mock, WT/H37Rv, TAP1-KO/H37Rv THP-1 conditions, primary human
  macrophages, and the Alg8-pulsed splenocytes for PMID 39438697.
- Correct the C1R note for PMID 27846572 (Caron 2015, not Bassani-Sternberg 2015) and add the
  missing T2 sample. Preserve each arm without inventing allele typing.
- Record source URLs, tables/figures, deposit file names, and affected corpus counts in an audit
  document; compare evidence identities before and after.

### PR 3 — #373 (1.58.8)

- Export original `note`, `classification`, and `reason` separately, retaining legacy `notes`.
  Carry the note and provenance override into MS/training evidence through the existing sample
  join. A sample override takes precedence only on a resolved sample attribution. Explicit null
  clears study-level override; absent sample override retains the normal PMID/rule behavior.
  Export the effective override value and its origin so null, inheritance, and ambiguity remain
  distinguishable. Preserve existing raw evidence fields and identities.
- Validate sample keys during YAML loading against an explicit mapping of consumed fields and a
  guard covering every current YAML key; no silently accepted typos.

## Steps

- [x] Enumerate 748 raw sample records: five lost to duplicate PMID replacement, then four
      explicit unprofiled records dropped from the 743 retained by the loader.
- [x] Verify the paper and deposit identities; identify and file additional curation defects.
- [x] PR 1: failing inventory/duplicate/unprofiled tests, then the loader, YAML, and export fixes.
- [x] PR 1: corpus identity comparison and gates.
- [x] PR 1: opened #439, CI green, merged, deployed 1.58.6, verified the published wheel.
- [x] PR 2: finished the source audit; applied and verified the curation corrections.
- [x] PR 2: corpus comparison, gates, PR #441, CI green, merged, deployed 1.58.7, wheel verified.
- [x] PR 3: override/null/inheritance/ambiguity tests, then the metadata and schema changes.
- [ ] PR 3: corpus comparison, gates, PR, CI, merge, deploy 1.58.8.

## Review

### PR 1 — #438 + #437 (1.58.6)

Eight new regressions in `tests/test_sample_inventory.py` failed first: duplicate PMIDs present
in the packaged YAML, no duplicate rejection in the loader, five sample records lost between the
file and the loaded mapping, two consolidated entries missing their samples or their provenance,
and four unprofiled records missing from the export.

Two existing tests pinned the behavior this PR reverses and were rewritten rather than deleted:
`test_ms_samples_no_zero_n` asserted every exported count was positive — true only because the
records it describes were being dropped — and `_KNOWN_CHIMERIC_SAMPLES` gained the two
HLA-B27-transgenic-rat arms, which are genuine chimeras (rat host and proteome, human HLA
transgene) that only became visible once 28188227's duplicate blocks were consolidated.

Verified against the full local corpus, before (ca5648c) versus after:

| | before | after |
|---|---|---|
| sample export rows | 739 | 748 |
| observation rows | 4,439,321 | 4,439,321 |

The nine added sample rows are exactly the five records the duplicate PMIDs discarded (three
Tasmanian devil, two transgenic rat) plus the four explicitly unprofiled records. No sample row
was removed, the column set is unchanged, and no field on any pre-existing row changed.

Observation `pmid`, `peptide`, `mhc_restriction`, `source_organism`, and `species` are identical
row-for-row across all 4,439,321 rows — the consolidation deliberately kept the exact `species`
strings the scanner already read, so no provenance fill moved. The only observation change is
metadata gained: 33,959 Tasmanian-devil rows that were previously unattributed now discriminate
into their three curated arms, including the IFN-gamma arm and its control-arm flags. PMID
28188227's 59,778 rows stay unattributed, correctly — both curated arms carry the same imprecise
`mhc: HLA-B*27`, so nothing distinguishes WT from ERAP1-KO and the join declines to guess.

The join guard is load-bearing rather than defensive bookkeeping: attribution path 3c matches
`attributed_sample_label` against `sample_label` with no allele involved and overrides every
heuristic above it, so a curated per-row label colliding with an unprofiled arm would attribute
real peptides to a sample the paper says was never profiled.

README corpus counts were stale independently of this change (159 PMIDs / 633 samples / 446
typed) and are now recomputed: 215 / 748 / 579, covering 96.0% of observations.

Gates: format, lint, 1,397 tests, and the packaged-build smoke tests all pass. CI, merge, and
publication are pending.

### PR 2 — #436 (1.58.7)

Source verification is complete for all four studies; the findings are recorded on #436 and
summarized in `docs/pmid-curation.md`. Three failure modes recurred: a plausible cell line
substituted for the real one (and its genotype carried along), a perturbation axis collapsed to a
single "unperturbed" arm, and a citation inverted.

Sample export 748 -> 755: seven removed (two phantom Liepe arms, two wrong-line Stopfer arms,
three replaced Leddy arms), fourteen added. Five surviving rows changed fields, all intended:
Liepe's GR-LCL and C1R genotypes, Pollock's two rows gaining study-level APM flags from the new
`perturbations:` block, and Stopfer's biopsy row gaining a `source`.

Observation rows stay at 4,439,321 with `pmid`, `peptide`, `mhc_restriction`, `source_organism`,
and `species` identical row-for-row. **Exactly 111 rows changed attribution**, and they are the
finding of this PR: T2's rows were being attributed to the phantom `JY (EBV-LCL)` sample by
`allele_exact` — JY's curated `HLA-A*02:01` matched them at the highest-confidence tier — so a
TAP-deficient hybridoma was labeled an EBV-LCL. A curated sample the paper never mentions is not
a harmless extra row; it competes for real evidence. They now attribute to T2, still
`allele_exact`.

Two existing tests were coupled to the corrected labels and were updated, not weakened:
`test_real_mixed_species_studies_keep_their_per_sample_species` (renamed arms; the per-sample
`species:` it guards is intact) and the stale docstring on
`test_resolver_skips_apc_when_cell_name_varies`, which explained the unassigned B-cell rows as
GR-LCL/JY ambiguity. Those 11,733 rows carry a six-allele `mhc_restriction` rather than a single
allele, so no allele path can fire; they were unassigned before and after, and JY was never a
candidate for them.

### PR 3 — #373 (1.58.8)

`override` (13 samples) and `note` (3) reached no consumer at sample level. Both are now
exported, along with `classification` and `reason` as their own columns beside the unchanged
legacy `notes`.

The resolution keeps four cases distinct rather than collapsing them to a value:
`sample` (the arm's own claim), `sample_null` (a curator considered this arm and decided
against one), `study` (inherited), `none` (nobody curated one). `sample_null` and `none`
produce the same empty value and mean different things; PMID 34497125 is the shape that
matters — two `cell_line` arms beside a patient-biopsy arm explicitly marked null.

`curation.MS_SAMPLE_FIELDS` declares all 22 permitted keys against what reads each, and the
loader rejects an undeclared key. `curation.OVERRIDE_VALUES` does the same for the override
vocabulary, which previously existed only in `classify_ms_row`'s branch chain and a YAML header
comment, so a misspelled override fell through to default classification. Study, rule, and
sample levels are all validated.

Verified before and after: sample export 755 -> 755 with six columns added, none removed, and
every pre-existing column byte-identical; observations 4,439,321 rows with `pmid`, `peptide`,
`mhc_restriction`, `source_organism`, `species`, `sample_label`, `sample_attribution`,
`perturbation`, and `is_control_arm` all identical row-for-row. Purely additive.

Origins across the 755 samples: study 513, none 229, sample 12, sample_null 1.

Two limits stated rather than papered over. The classification flags stay build-time and
PMID/rule-driven — `classify_ms_row` runs in the scanner, before any sample attribution exists,
so a sample-level override cannot feed them without recomputing flags in the export and letting
it disagree with the raw index. And `origin == "sample"` appears on no observation row today,
because all 13 override-bearing samples belong to PMIDs with zero rows in the current corpus;
the deterministic sample-export tests cover those cases, and the corpus test asserts the
negative — no unattributed row may carry a sample-origin value.

Filed #442 while verifying: 232 of 684 profiled curated samples are never attributed to any
observation row, including PMID 31844290's `ccRCC Pat9`, whose caveat is the one #373 exists to
surface.

### Audit carried into PR 3

The three PMIDs carrying sample overrides have zero rows in the current local indexes. The
two note-bearing studies contain 295,895 MS rows in total; Liepe 2016 also has 90 binding rows.
Source verification already disproved the SKMEL5/A375 and binimetinib/trametinib curation,
and found omitted MC38 and THP-1 perturbation arms.

Consolidating the duplicates also corrected two study labels that named the wrong first author:
33460454 is Gastaldello 2021 (Immunology, 10.1111/imm.13307), not "Owen 2021", and 28188227 is
Barnea 2017 (MCP, 10.1074/mcp.M116.066241), not "Alvarez-Navarro 2018". Both verified against
PubMed. 28188227's invented `title` was replaced with the published one.

---

# PR 3 specification — #426

- Preserve the full-load source-species selection in projected loads by reading both raw
  source-organism inputs before the derived filter is evaluated.
- Verify MS and binding with full output, peptide-only output, and explicit derived-column
  output. Include blank/null source organisms, populated fallback species, a conflicting fallback
  that must lose to the primary value, and unresolved rows.
- Ship as 1.58.3 after the gene-query PR has merged and deployed. Run the required gates on the
  final rebased branch; merge only after all CI checks pass, then deploy from clean main.

---

# PR 4 specification — #427

- Forward allele-set and provenance filters through the shared observations/binding loaders,
  preserving the export's comma-separated list input handling, and delete the duplicate filters.
- Verify canonical/bare/case-variant alleles, list and comma-separated inputs, combined provenance
  filters, and explicit empty inputs across raw loaders, MS/binding exports, and training modes.
- Ship as 1.58.4 after the preceding PRs have merged and deployed. Run format, lint, and the full
  default suite on the final branch, require every CI check, and deploy from clean main.

---

# Data consistency release series — issues #424–#427

## Priority and acceptance contract

1. **#424 / v1.58.1 — current curation in persisted data.** Include the scanner's curation YAML
   inputs and peptide-attribution CSVs in observations cache fingerprints. Cover cell-line
   metadata as well as study overrides, tissues, and monoallelic hosts. A curation-only edit must
   invalidate the cache and a normal rebuild must persist the edited value, including consecutive
   builds in the same interpreter. Existing metadata without these inputs must rebuild once.
2. **#425 / v1.58.2 — complete gene-query results.** Preserve OR semantics for mixed gene-symbol
   and Ensembl-ID queries across MS, binding, and training. Apply the same gene selection when
   expanding source-protein mappings; do not reintroduce unrelated mappings through a shared
   peptide. Keep explicitly supplied low-level name/ID filter semantics clear.
3. **#426 / v1.58.3 — projection-independent species filters.** A source-species filter must
   select the same evidence whether or not the output projects its derived column or fallback
   inputs. Exercise both indexes, missing source organism, and a nonmatching control row.
4. **#427 / v1.58.4 — one allele-set filter contract.** Route export filtering through the
   loader's normalization and validation so aliases, canonical inputs, and invalid empty queries
   behave consistently in MS, binding, and training exports.

## Execution and verification

- [x] Recheck the reported defects, current main, existing PRs, and release scripts.
- [x] PR 1: add failing cache/rebuild regressions; implement #424; review; format, lint, test;
      bump version; open PR; require all CI checks; merge; deploy from clean main; verify PyPI.
- [x] PR 2: add mixed-query and mapping-expansion regressions; implement #425; run all gates;
      bump version; open PR; require CI; merge; deploy from clean main; verify PyPI.
- [x] PR 3: add projected-filter regressions; implement #426; run all gates; bump version;
      open PR; require CI; merge; deploy from clean main; verify PyPI.
- [x] PR 4: add loader/export parity regressions; implement #427; run all gates; bump version;
      open PR; require CI; merge; deploy from clean main; verify PyPI.
- [x] Review remaining related Hitlist/ecosystem issues by dependency and data-quality impact.

Each PR has its own feature branch and patch release. The actual `deploy.sh` publishes the
version already in `hitlist/version.py`, so each bump is made explicitly in its PR. Reproduction
fixtures must be independent of the developer's data cache. Required release validation includes
`./test.sh --all` through the deployment script; never infer CI success from local tests.

## Review

- PR 2 design: keep the low-level mapping/observation filters conjunctive, and resolve the
  export's combined gene-query axis through one internal union loader. Use that same helper
  for evidence selection and source-protein expansion, deduplicating mappings that match both
  name and ID. Explicit peptide filters intersect the selected gene peptides, including empty
  intersections. Tests cover both evidence kinds, same-gene aliases, unknown queries, and shared
  peptides whose unrelated source-protein mappings must remain excluded.
- PR 1 implementation fingerprints the three curation YAMLs, cell-line registry, and every
  referenced peptide-attribution CSV with content hashes. Missing fingerprint entries invalidate
  old metadata automatically. Builds clear the file-backed curation caches and derived results.
- The two-build regression exposed #428: nested unbounded caches defeated bounded-cache eviction
  and retained stale classifications after `cache_clear()`. Removed all four redundant wrappers;
  source category and restriction evidence now both change on the next normal build.
- PR 1 validation: all seven new regressions pass; format and lint pass; `./test.sh` passes
  1,205 tests with one expected warning. The earlier focused builder/curation/smoke run passed
  290 tests. PR #429 passed every CI check, merged, and shipped as 1.58.1. The deployment's
  complete suite passed 1,229 tests, and both wheel and sdist hashes match the PyPI artifacts.
- PR 2's mixed-query regression also exposed #430: an untyped empty peptide IN predicate cannot
  bind to a large-string parquet column. The shared loader now types its peptide value set so
  unmatched queries and empty intersections return zero rows. Targeted export/loader tests pass
  201 tests. Format and lint pass; `./test.sh` passes 1,232 tests with one expected warning.
  PR #431 passed all CI jobs, merged, and shipped as 1.58.2. Its complete release suite passed
  1,256 tests. Both distribution hashes match the published PyPI artifacts.
- PR 3 adds the raw `species` fallback to the columns read for species-axis filters. All six
  new full/projected regressions pass, and the focused observations suite passes 45 tests.
  The branch is rebased on PR #431 and bumped to 1.58.3. Format and lint pass; the default suite
  passes 1,238 tests with one expected warning. PR #432 passed every CI job, merged, and
  shipped as 1.58.3. Its complete release suite passed 1,262 tests, and both distribution
  hashes match the published PyPI artifacts.
- PR 4 deletes the duplicate export membership filters and forwards both allele-set and
  provenance arguments through the loaders. All 42 new parity regressions pass; the combined
  export suite passes 200 tests. The branch incorporates PR #432's species fix and is bumped
  to 1.58.4. Format and lint pass; the default suite passes 1,280 tests with one expected
  warning. All seven original review reproductions pass on the combined series. PR #433 passed
  every CI job, merged, and shipped as 1.58.4; all 1,304 release tests passed and both PyPI artifact
  hashes matched the local builds. Final verification is also recorded in the PR description.

## Next priorities after this series

1. **#386 — export species-filter parity.** Expose the loader's source-species, host-species,
   and chimeric filters through exports. The projection fix in this series is its foundation.
2. **#373 — sample-curation schema validation.** Reject or implement ignored sample `override`
   and `note` fields, with explicit provenance semantics. Avoid accepting curation that has
   no effect on the output.
3. **#357 — expression coverage.** Make missing DepMap expression data visible and provide
   an explicit acquisition path; resolved sample identifiers alone do not establish coverage.
4. **Evidence-dependent curation.** Keep #359 and #366's ambiguous experimental arms unresolved
   until source evidence distinguishes them. Resolve pirl-unc/mhcgnomes#190 before extending
   the affected mouse curation in #364; track helper-gene parsing in mhcgnomes#191 separately.

The related mhcgnomes and openvax/pyensembl backlogs were reviewed. Parser correctness is upstream
of affected curation; unrelated parser display, packaging, and optional FASTA work do not block
the four releases above.

---

# Bug review — 2026-09-05

## Scope and approach

Review the current main snapshot (`8fd02b3`) for reproducible correctness defects, with emphasis
on public filtering/export contracts, sample attribution, and cached build artifacts. Use the
existing tests as a baseline and isolated temporary fixtures to verify suspected failures. This
is a findings review; production fixes and their release workflow are a separate follow-up.

## Steps

- [x] Read repository guidance, lessons, recent history, and package/test configuration.
- [x] Run formatting, lint, and the existing test suite to establish the baseline.
- [x] Trace public API and CLI paths through filtering, joins, and artifact reuse.
- [x] Reproduce actionable defects and check for existing GitHub issues.
- [x] File confirmed new bugs with exact reproduction and impact, as required by AGENTS.md.
- [x] Record verification results and report prioritized findings with source locations.

## Review

- Reviewed main `8fd02b3` / Hitlist 1.58.0. No production code changed; the local review branch
  contains only this task record. No PR/release is part of this findings-only review.
- `./format.sh` made no changes; `./lint.sh` passed; `./test.sh` passed all 1,198 default-suite
  tests with one expected backend-exception warning. Integration tests were excluded by the
  script's default selection. This is local baseline validation, not a new CI run.
- Seven isolated assertions reproduced four defects, independently covering both MS and binding
  paths where applicable. Reproductions use temporary fixtures and no network. The full local
  reproduction file is `/private/tmp/hitlist_bug_review_20260905.py`; each issue also contains a
  self-contained reproduction whose output was executed and verified before filing.
- **P1 — [#424](https://github.com/pirl-unc/hitlist/issues/424):** observations cache fingerprints
  omit the main curation YAML inputs. Changing `restriction_evidence` from experimental to
  predicted leaves the cache valid, permitting stale persisted scientific annotations.
- **P2 — [#425](https://github.com/pirl-unc/hitlist/issues/425):** mixed gene-symbol/Ensembl-ID
  export queries use AND instead of the documented OR semantics. Two individually matching
  genes yield zero rows when queried together; mapping expansion repeats the same conjunction.
- **P2 — [#426](https://github.com/pirl-unc/hitlist/issues/426):** projecting only `peptide`
  omits the raw `species` dependency of a `source_species` filter, dropping rows that need its
  documented fallback when `source_organism` is blank.
- **P2 — [#427](https://github.com/pirl-unc/hitlist/issues/427):** export allele-set filters omit
  normalization that the raw loaders perform. `A*02:01` matches the loader but silently returns
  zero exported rows where canonical `HLA-A*02:01` succeeds.

---

# Issues #418 and #419 — assay routing and provenance CLI parity

## Goal

Keep biochemical purified-MHC stability measurements out of the MS-elution index, and make every
MHC allele-set provenance value emitted by Hitlist valid at each CLI export entry point.

## Premise check

- The registered 2026-03-30 IEDB source has 222 rows for PMID 36423003: 146
  `cellular MHC/mass spectrometry` ligand-presentation rows and 76
  `purified MHC/direct/radioactivity` half-life rows.
- The existing qualitative-only classifier routes 66 non-`Positive` half-life rows correctly but
  leaks the 10 plain-`Positive` half-life rows into `observations.parquet`. They are eight
  BoLA-6*013:01, one BoLA-1*023:01, and one BoLA-2*012:01 row; the issue's three-DRB3/153-MS count
  does not match the current registered source or built corpus.
- The same mechanism affects 942 plain-`Positive`, purified-MHC half-life rows corpus-wide (646
  fluorescence and 296 radioactivity). Classification must be based on the structured method and
  response, not a PMID or allele allow-list.

## Design

- Extend `is_binding_assay` with optional `assay_method` and `response_measured` inputs, preserving
  source compatibility for existing two-argument callers.
- Treat a half-life response measured on purified MHC as binding evidence. Keep the existing
  qualitative/comment rules unchanged in this PR, and explicitly retain cellular-MHC mass
  spectrometry ligand-presentation rows.
- Pass the already-scanned structured fields into the classifier. Bump the observations artifact
  contract so existing cached parquets rebuild instead of retaining stale routing.
- Promote the complete MHC allele-set provenance tuple to one public constant, use it for the
  scanner contract and all three CLI `choices`, and export it through the lazy top-level API.
- Add scanner/classifier regressions for both assay types and end-to-end parser coverage proving
  every emitted provenance value is accepted by `export ms`, `export binding`, and
  `export training`.
- Bump to 1.58.0: this changes which public evidence index contains existing rows and adds a public
  vocabulary constant plus a new accepted CLI value.

## Steps

- [x] Reproduce both reports and audit #418 against the registered raw-source snapshot.
- [x] Correct #418's stale row-count/allele premise on the issue.
- [x] Add failing classifier, scanner, and CLI regressions.
- [x] Implement structured assay routing, cache invalidation, and shared provenance choices.
- [x] Verify targeted raw-source output and real rebuilt index counts.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [ ] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- The fix keys on a source-defined assay signature, not PMID or allele identity. An isolated full
  rebuild moved all 942 matching assay IRIs from MS to binding, left zero matching rows in MS, and
  confirmed every moved IRI in `binding.parquet`.
- PMID 36423003 now has 146 cellular mass-spectrometry rows and 76 purified-MHC half-life rows in
  their respective indexes. The current IEDB source already carries correct structured metadata,
  so no upstream issue is warranted.
- `MHC_ALLELE_PROVENANCE_VALUES` is now the public contract used by all three CLI parsers; the
  parser regression exercises every value for MS, binding, and training exports.
- Artifact contract version 2 forces existing cached parquets to rebuild with the corrected
  routing. Hitlist is bumped to 1.58.0 because existing evidence moves between public indexes and
  a public vocabulary/CLI value is added.
- `./format.sh` and `./lint.sh` passed; `./test.sh` passed 1,198 tests with one expected warning;
  `tests/test_build_smoke.py` passed 2/2.

---

# Issue #416 — preserve study-specific THP-1 HLA typing

## Goal

Stop offering HLA-A*24:02 and HLA-B*35:01 as candidate presenters for PMID 35051231 when that
study explicitly typed and analyzed its THP-1 sub-line as homozygous HLA-A*02:01,
HLA-B*15:11, and HLA-C*03:03. Preserve the heterozygous typing for PMID 33392160, whose authors
used DSMZ ACC-16 and reported that genotype.

## Source finding

- Nicholas 2022 S1 Table repeats A*02:01, B*15:11, and C*03:03 in both class-I haplotypes for
  THP-1. The Results explicitly call the three loci homozygous and assign 6,499 peptides to the
  A*02:01 and B*15:11 motifs.
- Ghosh 2020 reports A*02:01/A*24:02, B*15:11/B*35:01, and C*03:03 for its THP-1 culture and
  states that the line came from DSMZ ACC-16.
- Cellosaurus CVCL_0006 preserves both homozygous and heterozygous typing records and cites a
  paper specifically documenting the THP-1 HLA discrepancy. These are study/sub-line-specific
  facts, not two spellings of one canonical genotype.

## Design

- Replace only the three Nicholas THP-1 sample genotypes with the study's homozygous class-I
  typing while retaining its reported class-II typing unchanged.
- Keep Ghosh/DSMZ and other independently sourced THP-1 entries heterozygous unless their own
  source says otherwise. Do not create an allele alias or globally rewrite THP-1.
- Clarify the PMID notes and the shared `cell_lines.yaml` entry: the registry owns canonical line
  identity, while `ms_samples[].mhc` owns study-specific typing. Nicholas gives no catalogue
  number, so do not invent a second Cellosaurus accession or sub-line name.
- Add regressions at the curation and observation-join boundaries proving that PMID 35051231
  excludes A*24:02/B*35:01 and PMID 33392160 retains them.
- Bump the patch version to 1.57.2.

## Steps

- [x] Inspect issue #416 and verify Nicholas S1/Results, Ghosh Results, DSMZ, and Cellosaurus.
- [x] Add failing study-specific typing and observation-join regressions.
- [x] Correct the three Nicholas samples and document the source-specific registry contract.
- [x] Verify affected real-PMID outputs and the corpus-wide sample-ploidy audit.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [x] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- The correction is confined to the three Nicholas THP-1 arms. The Ghosh/DSMZ sample remains
  heterozygous, and the shared cell-line registry now explicitly directs consumers to the
  study-level genotype.
- A real-data join over PMID 35051231's B*15:11 observations contains neither A*24:02 nor B*35:01
  in exact or class-pool sample metadata; `sample_ploidy_audit()` remains empty corpus-wide.
- No upstream issue is warranted: Nicholas, Ghosh, DSMZ, and Cellosaurus faithfully expose the
  divergent source/sub-line typings; the defect was Hitlist's choice to substitute a global
  superset for Nicholas's reported genotype.
- `./format.sh` and `./lint.sh` passed; `./test.sh` passed 1,192 tests with one expected warning;
  `tests/test_build_smoke.py` passed 2/2.
- PR #422 merged as `bb11d78`; Hitlist 1.57.2 was uploaded to PyPI as both wheel and sdist.

---

# Issue #414 — separate the measured A19 genotype from predicted restrictions

## Goal

Represent both source facts from PMID 36423003 without converting one into the other: 2824TP's
measured A19 genotype contains `BoLA-6*014:02`, while the paper and supplement assign three retained
peptides to `BoLA-6*014:01` by prediction.

## Source finding

- Results 3.2 and the independent Vasoya et al. A19 definition both give the genotype as
  `BoLA-2*016:01 BoLA-6*014:02`.
- Table 3 and Supplementary Data 2.1 report predicted restrictions as `BoLA-6*014:01`; the
  supplement contains 23 such predictions, all from the two 2824TP runs, and no `*014:02`
  prediction. IEDB faithfully retains three final-table rows, so this is not an IEDB ingestion bug.
- Supplementary Data 2.1 also contains 16 `BoLA-2*008:01` predictions from 2123TP, but none survive
  into the final Table 3 / IEDB set. Keep it as measured sample typing without inventing an
  observation.

## Design

- Correct the 2824TP `ms_samples[].mhc` genotype to `BoLA-6*014:02`.
- Register the paper's three peptide-to-2824TP mappings through the existing
  `peptide_attributions` mechanism.
- Generalize scan-time attribution so a curated mapping can label an already allele-resolved row.
  Preserve its reported restriction, allele set, and `exact` allele-set provenance; only add the
  source-backed sample label. Emit one row per attributed sample if a future resolved peptide maps
  to more than one.
- Let the observation join use the existing `curated_sample_label` path. The public `sample_mhc`
  must show `*014:02`, while `mhc_restriction` remains the paper's predicted `*014:01` and
  `restriction_evidence` remains `predicted`.
- Do not create an allele alias or claim that `*014:01` and `*014:02` are equivalent molecules.
  Keep the unexplained source-level mismatch explicit in the study note and #414.
- Bump the patch version to 1.57.1, since this corrects existing sample typing and attribution
  without adding a new public schema field.

## Steps

- [x] Inspect the article, independent A19 definition, and all relevant supplementary workbooks.
- [x] Correct #414's issue record with the primary-source finding.
- [x] Add failing curation, scanner, and observation-join regressions.
- [x] Implement resolved-row sample attribution and correct the PMID curation/data asset.
- [x] Verify the real three-row output and the corpus-wide sample-ploidy audit.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [x] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- Primary-source checking showed that IEDB faithfully represents this paper's own prediction
  table, so no upstream IEDB issue is warranted. The unresolved mismatch is within the published
  source: its measured A19 genotype uses `*014:02`, while its retained predictions use `*014:01`.
- The three predictions now retain the exact resolved molecule and `predicted` evidence while a
  separate curated label selects 2824TP, whose public sample genotype contains `*014:02`.
- A real scan of the 7.77 GB IEDB export found exactly the three expected PMID rows with this
  separation. The corpus-wide sample-ploidy audit remains clean.
- Release gates are clean: formatting and lint passed, the full suite passed 1,190 tests with one
  expected warning, and the two build-smoke tests passed against regenerated artifacts.
- PR #421 merged as `c43937b`; Hitlist 1.57.1 was uploaded to PyPI as both wheel and sdist.

---

# Issue #415 — restriction evidence is separate from allele-set provenance

## Goal

Expose whether a named MHC restriction was experimentally isolated, implied by a monoallelic
system, computationally predicted, or not established. Keep this independent from
`mhc_allele_provenance`, which answers only where the candidate allele set came from.

## Design

- Add a categorical `restriction_evidence` column with four values: `experimental`,
  `monoallelic`, `predicted`, and `unknown`.
- Infer `experimental` only for resolved restrictions in binding assays and `monoallelic` only
  for resolved restrictions whose existing sample classifier proves a monoallelic system.
  Everything else remains `unknown` unless explicitly curated.
- Support PMID-level defaults plus condition-matched `restriction_evidence_rules`, so mixed
  studies can describe one evidence-generating method without relabeling unrelated rows.
- Curate PMID 36423003's resolved cellular-MHC/MS allele assignments as `predicted`; its
  class-only row and purified-MHC half-life rows must not inherit that claim.
- Carry the axis and an exact-value filter through the canonical observation, binding, export,
  training, and CLI paths. Do not change `mhc_allele_provenance` values or semantics.
- Treat the newly discovered purified-MHC index leak as separate issue #418 rather than hiding
  it in evidence curation.

## Steps

- [x] Add failing unit and scanner regressions for inferred, curated, mixed-study, and unresolved
      evidence states.
- [x] Implement and validate the evidence vocabulary and conditional curation API.
- [x] Persist `restriction_evidence` in scanner and supplementary rows; categorize it in builds.
- [x] Add load/export/training/CLI filters and document the new schema contract.
- [x] Curate PMID 36423003 and verify real-corpus counts by evidence and assay family.
- [x] Bump the minor version and update changelog/release-facing version surfaces.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [x] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- `mhc_allele_provenance=exact` remains a structural statement: the source row named a concrete
  allele. `restriction_evidence=predicted` can now accompany it without laundering a predictor's
  assignment into an experimental observation.
- The only inferred positive labels are mechanically safe: resolved binding-assay restrictions
  are `experimental`, and resolved rows already proven monoallelic are `monoallelic`. All other
  rows default to `unknown` unless a validated PMID rule says otherwise.
- PMID 36423003 uses a method-and-response rule. Raw-source verification separated resolved
  cellular-MS predictions from the class-only row and the purified-MHC half-life assays.
- That verification exposed two unrelated pre-existing defects, filed as #418 (binding-index
  leakage) and #419 (CLI rejects `peptide_attribution` provenance). Neither is hidden or folded
  into this PR.
- Focused checks passed 10 tests; the complete impacted curation/scanner/supplement/load/export
  suite passed 460 tests.
- Release gates are clean: formatting and lint passed, the full suite passed 1,185 tests, and the
  two build-smoke tests passed against regenerated artifacts.

---

# PR #417 review fixes — class-safe and heterodimer-safe attribution

## Goal

Resolve every review finding without weakening the new sample-MHC attribution contract:
peptide support must stay within the target MHC class, serotype matching must work for either
chain of a reported class-II heterodimer, typed nonmatching serotypes must not become unknown
support, and merged HLA-DM measurements must not be labeled as control samples.

## Design

- Infer the query MHC class from the requested allele or serotype and use it when the caller did
  not supply `mhc_class`; also gate every summary row against that target class so monkeypatched,
  legacy, or explicitly broad observation frames cannot contribute opposite-class evidence.
- Make the sample join symmetric for class-II pairs: retain exact full-restriction matching as
  first priority, then try each normalized observation-side heterodimer component. Carry the
  effective join key through ambiguous-candidate resolution and match-type provenance so a DQ8
  beta-chain expansion can match a full DQA1/DQB1 observation without rewriting the observation.
- Derive peptide-summary allele and serotype evidence from the full restriction plus all of its
  components. A target beta chain can therefore match a full pair exactly, and DQ8 can match the
  pair through its DQB1 component.
- Define `unknown_allele` only when the attributed sample provides neither exact-allele typing nor
  serotype typing. A known, nonmatching serotype is negative evidence for this query, not unknown.
- Replace each of the four MAPTAC samples whose deposited peptides merge `-DM` and `+DM` with two
  truthful experimental-arm samples. Their shared-allele observation join is intentionally
  ambiguous and therefore blanks arm/APM/control metadata via the existing consensus path.
- Treat all five defects as local: no upstream issue is warranted unless implementation exposes a
  dependency behavior that prevents component-aware matching rather than merely requiring it.

## Steps

- [x] Add focused regressions for cross-class summary leakage, DQ8 pair joining and summary
      support, nonmatching-serotype exclusion, and merged HLA-DM arm metadata.
- [x] Implement target-class and heterodimer-component matching with truthful provenance.
- [x] Split the four merged MAPTAC HLA-DM conditions into explicit `-DM` / `+DM` sample arms.
- [x] Run targeted tests and inspect the affected real-PMID outputs.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`.

### Second review: preserve precision through fallback paths

- [x] Reproduce noncanonical serotype parsing and multi-sample class-pool summary behavior.
- [x] Canonicalize every parsed serotype with its own `to_string()` representation before catalog
      lookup, for both whole fields and tokens inside mixed fields.
- [x] Represent class-pool exact alleles and reported serotypes separately; propagate both through
      the fallback without converting inferred serotype members into reported exact alleles.
- [x] Add focused unit and end-to-end regressions for spelling variants, allele joins, and
      `class_only_sample_serotype` summary provenance.
- [x] Re-run real-corpus checks, `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [ ] Update the version/PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- All five findings were local Hitlist defects; no upstream issue was warranted. The DQ8 catalog's
  beta-chain members are sufficient once Hitlist applies its own component-aware matching contract
  symmetrically to observation pairs.
- Peptide summaries now infer and push down the target class, retain a row-class backstop, filter
  mixed-sample genotypes to that class, derive serotypes from both chains of a pair, and reserve
  `unknown_allele` for samples with neither exact nor serotype typing.
- The observation join aliases a full class-II restriction only from a serotype member. A
  regression proves that two fully known pairs do not become a match merely because they share
  one chain.
- PMID 31495665 now has separate `dm-` and `dm+` samples for each of the four alleles whose peptide
  sets were merged during ingestion. Real merged observations resolve as `pmid_ambiguous` with
  blank `condition_category`, `apm_perturbed`, and `is_control_arm` rather than false controls.
- Real-corpus checks confirmed PMID 34433824's DQA1/DQB1 rows join to the DQ8 sample as
  `serotype_expansion`; its DQ8 peptide summary is nonempty; PMID 35051231 contributes no class-I
  row to a DRB1*11:01 query; and PMID 28467828 contributes no row to an unrelated DR4 query.
- Verification: 16 focused review regressions passed; `./format.sh` and `./lint.sh` passed;
  `./test.sh` passed 1,171 tests with one expected warning; `tests/test_build_smoke.py` passed 2/2.
- The second review's six focused cases now pass. Parsed serotypes use mhcgnomes' canonical
  representation for catalog lookup, while class pools serialize source-reported exact molecules
  and serotypes rather than expanded join candidates. The synthetic multi-sample regression now
  reports `class_only_sample_serotype`, never `class_only_sample_allele`.
- Final verification after both review rounds: the curation/export suite passed 362 tests;
  `./format.sh` and `./lint.sh` passed; `./test.sh` passed 1,176 tests with one expected warning;
  and `tests/test_build_smoke.py` passed 2/2.

---

# Issues #380, #381, and #374 — truthful sample-MHC attribution

## Goal

Fix the curated-sample MHC attribution defects in dependency order: one documented
sample-MHC candidate API (#380), correct per-sample genotypes for PMID 36423003 (#381),
and eliminate the declared-class/typed-allele contradictions (#374).

## Steps

- [x] Inspect the sample-join implementation, current YAML, corpus counts, and paper methods.
- [x] Implement and document the centralized sample-MHC attribution-candidate API.
- [x] Add a general audit for samples whose `mhc` pools several genotypes.
- [x] Curate PMID 36423003 and the remaining #374 samples from primary sources.
- [x] Add focused unit and invariant tests.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`.
- [x] Bump the version and open a PR.

## Review

### What the verification changed

Three of the four premises in the issues were wrong, and checking first saved
implementing them:

- **#374 group 1 (HLA-G declared class I) was already fixed** — all three
  721.221-HLA-G transfectants declare `non-classical` today. No work needed.
- **#374 group 2 was real but mis-framed.** The eleven `I+II` samples were not
  contradictions: every study *did* profile both classes, and the class-II alleles
  were simply missing from the curation. The fix was to finish the typing from each
  paper's own table, not to weaken the declaration to `I`.
- **#381's allele table was incomplete and its acceptance criteria wrong.** The corpus
  holds 13 class-I BoLA alleles, not 6, and its class-II sample has three real DRB3
  genotypes. Curating all 13 onto one sample would pool eight animals and would also
  report a NetMHCpan prediction as an observation.

### The generalization

The #381 bug — an `mhc` field holding a union across samples rather than one
genotype — is a *class* of defect, not one entry. `qc.sample_ploidy_audit` detects it
without threshold tuning: a diploid donor carries at most two alleles per locus, so
three is proof of pooling. It found six samples; all six were wrong, and all six are
fixed here from primary sources. The audit now guards the corpus in CI.

Notably it also guards against doing #381 *wrong*: the pooled 13-allele curation the
issue asks for would fail it.

### Deliberately not done

- **Predicted-vs-observed restriction** (#415). `mhc_allele_provenance` has no value
  meaning "predicted", and 155 of PMID 36423003's rows resolve `exact` from a
  NetMHCpan <2%-rank assignment. This is not one study's problem — IEDB populates
  elution restrictions by inference routinely — so it needs a schema axis and a
  corpus-wide sweep, not a patch here.
- **THP-1 class-I typing conflict** (#416). Two primary sources disagree; the
  heterozygous DSMZ form is kept and the conflict filed rather than guessed.
- **BoLA-6*014:01 vs *014:02** (#414). IEDB and the paper disagree on one allele of
  one line. IEDB's value is curated so its rows still attribute, discrepancy recorded.

### Sample-count changes

| PMID | Before | After |
|---|---|---|
| 36423003 | 2 | 9 (8 per-line class-I + 1 locus-level class-II) |
| 32350084 | 2 | 26 (19 EBV-LCL + 7 K562) |
| 26768311 | 2 | 10 (5 allotypes x 2 conditions) |
| 31495665 class II | 2 | 14 (6 single arms + 4 alleles x 2 HLA-DM arms) |

---

# Comprehensive modality correctness — issues #382, #376, #396, #399

## Program goal

Make MHC identity and source-protein attribution explicit, correct, and auditable across MS
elution, binding assays, curated sample metadata, and peptide mappings. Ship the work in two
dependency-ordered PRs: the shared MHC identity contract first, then Ensembl IG/TR mapping
coverage and provenance.

## Phase 1 — contextual MHC identity and validation (#382, #376, #396)

### Design

- Introduce one cached, documented MHC annotation resolver that accepts the raw restriction,
  source-reported class, and optional curated species context. It returns the normalized
  restriction, resolved species and provenance, canonical class and provenance, plus explicit
  correction/conflict flags.
- Treat curated species as a parsing constraint when it can parse the designation; fall back to
  an explicit designation's unconstrained species for legitimate engineered-MHC systems. Cache
  keys include the species context. A compatible generic result such as `Bos sp.` may be refined
  to `Bos taurus`; an incompatible unconstrained guess is recorded as a context disagreement.
- Derive class only from actual molecules (`Allele`, `Gene`, `Pair`). Derive semicolon candidate
  sets component-wise when every resolved component agrees. Class-only, serotype-only, and
  unparseable restrictions retain the normalized source-reported class.
- Store `mhc_class_reported`, `mhc_class_source`, `mhc_class_corrected`,
  `mhc_species_source`, and `mhc_species_context_disagrees` on both MS and binding rows. Refresh
  these fields after donor-set promotion so the stored restriction and provenance cannot drift.
- Add a cross-modality MHC-token audit covering MS, binding, and curated sample MHC. Known source
  errors and parser gaps carry distinct statuses/reasons; any new unrecognized token fails the
  build. Expose the audit through the Python QC API, bare `hitlist qc`, and a dedicated CLI command.
- Version the observations artifact contract so existing parquets rebuild once instead of
  silently preserving the old schema and wrong classifications.
- Print build summaries for class corrections and incompatible contextual-species corrections.

### Verification

- [x] Unit-test contextual parsing, explicit-species fallback, class derivation/fallback,
      donor-set behavior, and correction flags.
- [x] Scanner-test the Bos contextual case, Caja/Mamu correction, class-only fallback, and
      post-promotion donor-set fields for both source classifications.
- [x] Unit-test known-invalid, parser-gap, sentinel, and unknown-token QC behavior across MS,
      binding, and curated sample inputs; add a real-corpus staleness/new-token guard.
- [x] Test artifact-version invalidation, schema columns, build summaries, CLI routing, and docs.
- [x] Bump the patch version; run targeted tests, format, lint, and the complete test suite.
- [x] Open a PR closing #382, #376, and #396; require all CI jobs, merge, deploy, and verify PyPI.

## Phase 2 — immunoglobulin/TCR mapping coverage (#399)

### Design

- Include Ensembl's coding IG/TR biotypes (`IG_V/D/J/C_gene`, `TR_V/D/J/C_gene`) alongside
  `protein_coding`; continue excluding pseudogenes and document the germline-only boundary.
- Carry source-gene biotype through `ProteomeIndex`, long-form peptide mappings, mapping schema,
  filters/exports, and artifact-version metadata so IG/TR attribution is distinguishable from a
  conventional protein-coding match.
- Keep `ProteomeIndex.from_ensembl(biotype="protein_coding")` as an explicit compatibility mode;
  make the new plural `gene_biotypes=` API and the mapping worker's task contract explicit.
- Test index construction and mapping with protein-coding, IG, TR, pseudogene, duplicate-sequence,
  cache round-trip, process-worker, and legacy-artifact cases. Quantify recovered current-corpus
  mappings before release.

### Verification

- [x] Implement and verify the expanded Ensembl index contract and mapping provenance.
- [x] Bump the patch version; run all required gates and corpus coverage comparisons.
- [ ] Open a PR closing #399; require all CI jobs, merge, deploy, and verify PyPI.

## Review section

- `resolve_mhc_annotation()` now owns normalization, contextual species resolution,
  molecule/donor-set class derivation, source fallback, and persisted provenance. Scanner and
  supplementary ingestion both use it before filtering and refresh it after set promotion.
- The registered 4.4M-row corpus has exactly five reviewed exceptional tokens: four
  `invalid_source` values (`HLA-B23`, `HLA-DR7A`, `HLA-DR3A`, `HLA-DR1B`) and one parser gap
  (`HLA-Cw16`). The audit finds no unrecognized token; its integration test pins both growth and
  stale allowlist entries.
- Observations artifact contract v1 forces a one-time rebuild for the new schema. Build output
  reports class corrections, species-context conflicts, and token-audit totals before writing.
- Verification: 338 affected non-integration tests passed; the dedicated corpus audit passed;
  the supplementary suite also passes under Python 3.9; format and lint passed; full
  `./test.sh --all -rs` passed 1,170 tests with zero skips and one expected warning.
- Phase 1 shipped in PR #412 as v1.55.7; every CI job passed and the wheel and sdist were
  verified on PyPI.
- Phase 2 centralizes the translated Ensembl policy as conventional `protein_coding` plus the
  eight coding IG/TR gene biotypes. Both gene and transcript records must satisfy the policy;
  pseudogenes remain excluded. `gene_biotype` now survives index metadata, worker normalization,
  sidecar filtering, and exploded training exports. Mapping artifact v2 forces a clean rebuild.
- The Ensembl 112 audit finds 420 translated IG/TR proteins. Against the current registered human
  corpus they produce 15,808 long-form mappings and recover 4,451 unique peptides with no prior
  human-proteome match (MS: 1,745 class I and 2,836 class II unique peptides).
- Phase 2 verification: targeted proteome/mapping/export tests passed 220 tests; format and lint
  passed; `./test.sh --all -rs` passed 1,175 tests with zero skips and one expected warning.

---

# Issue #410 — deterministic Alpizar resolver regression

## Goal

Remove the last full-suite skip without weakening the regression. The test must exercise the
public observations-export path against a small, version-controlled Alpizar-shaped fixture rather
than depending on whichever IEDB snapshot happens to be registered on the developer machine.

## Diagnosis and design

- PMID 27920218 is present in the current build (8,144 rows). The stale test selected zero rows
  because IEDB replaced the old 515 literal `HLA class I` restrictions with explicit
  semicolon-separated candidate-allele sets.
- Keep the biological behavior under test: ambiguous C1R rows must route to B*40:02, B*39:01, or
  the pooled sample from their antigen-processing text. Exercise both the historical class-only
  representation and the current allele-set representation.
- Use a temporary observations parquet plus a minimal synthetic PMID override and call
  `generate_observations_table()`. This covers the real class-pool orchestration and candidate
  scorer while remaining independent of the installed corpus.
- Correct the newly exposed provenance error: a class-pool candidate selected from row-level
  discriminator text must report `sample_attribution=discriminated`; `sample_match_type` remains
  `pmid_class_pool` because the restriction itself was not an exact allele match.
- Update the shipped Alpizar curation note to document the IEDB representation change.

## Steps

- [x] Replace the conditional full-corpus Alpizar test with the deterministic public-API fixture.
- [x] Correct and test class-pool discriminator provenance.
- [x] Update the Alpizar curation note and bump the patch version.
- [x] Run targeted tests, `./format.sh`, `./lint.sh`, and `./test.sh --all -rs`.
- [x] Review the diff, open a PR closing #410, merge, deploy, and verify PyPI.

## Review section

- The paper was never absent: the current corpus has 8,144 Alpizar rows. IEDB changed the 515
  ambiguous restrictions from `HLA class I` to candidate-allele sets, making the old filter stale.
- The replacement writes four small observation rows to a temporary parquet and exercises
  `generate_observations_table()` with an isolated Alpizar-shaped override. It covers the old
  class-only form, both current single-transfectant sets, and the current pooled set.
- Class-pool scoring now records `sample_attribution=discriminated` while correctly retaining
  `sample_match_type=pmid_class_pool`; the latter describes restriction-level evidence, whereas
  the former describes the sample-selection mechanism.
- Targeted export tests pass (117 passed, 17 integration tests deselected). Format and lint pass.
  The complete corpus suite passes 1,154 tests with zero skips and one expected warning.

---

# Issue #406 follow-up — isolate direct prefetch-worker tests

## Goal

Prevent direct unit calls to the child-only prefetch entry point from leaking its data-directory
override into later xdist tests. The full integration suite should retain only genuine
corpus-dependent skips.

## Steps

- [x] Scope `_prefetch_worker` test doubles and data-directory mutation to a monkeypatch context.
- [x] Add a regression assertion that the parent test process state is restored.
- [x] Bump to 1.55.5; run format, lint, targeted mixed-order tests, and `./test.sh --all -rs`.
- [x] Ship a follow-up PR, merge, deploy, and verify PyPI.

## Review section

- Direct `_prefetch_worker` tests now emulate the disposable child-process boundary with a nested
  monkeypatch context and assert that `_override_data_dir` is restored after each call.
- The mixed-order regression (`test_mappings.py` followed by `test_observations.py` in one worker)
  passes all 86 tests; the full suite passes 1,153 with only one legitimate corpus-dependent skip
  (`Alpizar 2017 not present in this build`). Format and lint pass. Version bumped to 1.55.5.

---

# Issues #402, #404, #405 — bounded/offline-safe mapping builds and artifact contract

## Goal

Make peptide-mapping builds terminate predictably, obey the documented no-fetch policy, and
rebuild sidecars whenever the code or parameters that define their contents change. Remove the
new timeout environment variables: safety deadlines are internal invariants, while legitimate
caller choices remain explicit function/CLI arguments.

## Design

- Replace the parent loop's before-call stopwatch with a supervised, killable child process.
  The parent submits one canonical at a time to a single-child process pool, records the in-flight
  name before dispatch, and waits only until one fixed absolute phase deadline. If the child does
  not answer, terminate it, report the named canonical, and skip it plus the unattempted tail.
- Return explicit prefetch outcomes. Workers may only receive UniProt/Ensembl tasks whose required
  local cache warm-up succeeded; a failed/timed-out fetch is not retried silently in a worker.
  This keeps the existing "failure is tolerated" contract without moving the same hang elsewhere.
- Replace `_per_canonical_mapping_worker`'s positional tuple protocol with a documented, picklable
  `MappingTask` value object. Keep index construction, peptide mapping, output normalization, and
  coverage accounting in one worker entry point so unit and real process-pool tests exercise the
  same API across cache/network policies and peptide lengths.
- Remove `HITLIST_PREFETCH_BUDGET`, `HITLIST_DOWNLOAD_TIMEOUT`, and their float parsers. Keep the
  socket timeout and prefetch deadline as finite positive internal constants. Tests may pass an
  internal deadline argument directly; users do not configure safety correctness through process
  state.
- Honor `fetch_missing=False` (#405): reuse existing cached artifacts, but do not launch the
  prefetch worker or any network download for an uncached proteome. Log and record skipped tasks.
- Add a monotonic peptide-mapping artifact version plus behavior-defining parameters (Ensembl
  release, UniProt search policy, flank width, seed length, and output schema) to the metadata.
  Legacy or mismatched metadata is stale.
- On an observations-cache hit with `build_mappings=True`, invoke the mappings builder so it can
  validate/rebuild only the sidecar. Do not rescan observations, bulk proteomics, or expression.
- Keep the pre-call progress line from #403 and make deadline/failure messages unconditional when
  they explain omitted output.

## Implementation and verification

- [x] Add failing regression tests for an actually blocked in-flight prefetch, explicit failed and
      unattempted outcomes, no worker retry, and `fetch_missing=False` network isolation.
- [x] Introduce and document `MappingTask`; expand direct and process-pool worker contract tests.
- [x] Implement supervised prefetch and remove the timeout environment-variable APIs/tests.
- [x] Add mapping artifact contract metadata and cache-validation tests, including legacy metadata
      and each behavior-defining parameter.
- [x] Add a builder early-return regression proving stale/missing mappings rebuild independently.
- [x] Identify all four default-suite skips; remove any state-dependent skip that masks a unit-test
      branch, or document why the integration/dependency skip is intentional.
- [x] Bump the patch version and update user-facing documentation/comments.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`; inspect the diff and test behavior.
- [x] Isolate the unrelated default-suite cache/multiprocessing flake found during final
      high-concurrency verification (#406), then rerun all required gates.
- [x] Push a PR linking #402, #404, and #405; check every CI job.
- [x] Merge, update clean `main`, run `./deploy.sh`, and verify the released version on PyPI.

## Review section

- Replaced the pre-call-only stopwatch with a single-child supervisor that names every in-flight
  request, enforces one absolute 900-second warm-up deadline, terminates a blocked call, and marks
  the current/unattempted proteomes unavailable. Mapping workers are structurally cache-only, so
  the same network operation cannot escape the deadline as an on-demand retry.
- Removed `HITLIST_PREFETCH_BUDGET` and `HITLIST_DOWNLOAD_TIMEOUT`. The finite socket timeout and
  warm-up deadline are internal safety constants; invalid/exhausted internal test deadlines fail
  closed. `fetch_missing=False` now reaches resolution, primary UPID fetches, and PMID overrides.
- `_per_canonical_mapping_worker` now accepts a documented `MappingTask` and returns a named
  `MappingResult`. One implementation builds one seed index, maps every peptide length, produces
  one normalized frame, distinguishes unavailable from zero matches, and preserves the full
  coverage denominator. Direct, pickle, corrupt-cache, and real process-pool cases cover the API.
- Mapping metadata now carries artifact version, Ensembl release, UniProt/fetch policy, flank,
  seed length, and schema. Legacy/mismatched/incomplete artifacts rebuild; observations cache hits
  still validate the independently cached mappings sidecar.
- The four default skips were two tests conditional on a developer's local observations cache and
  two tests conditional on the optional, non-PyPI `cancerdata` package. The first pair now uses an
  isolated empty data directory; the second injects a fake provider and separately tests the
  actionable missing-provider error. Latest default run: 1,130 passed, 0 skipped.
- `./format.sh`, `./lint.sh`, and `./test.sh -rs` pass. Version bumped from 1.55.3 to 1.55.4.
- Final high-concurrency review exposed #406: bulk/proteome tests depended on real user cache state,
  and one multiprocessing regression required a sandbox-forbidden Manager socket. The PR now
  isolates those caches per test and uses spawn-safe result files instead of a Manager service.

---

# Issue #46 — multi-axis species model (PR 1: schema + filters)

Scope (user-approved): **Schema + filters**, detection via **genus-aware heuristic + audit**.
Defer: effector_organism, mhc_donor_individual, build-time axis validation, per-PMID override curation.

## Design
New per-row columns derived purely from existing `host`, `source_organism`, `mhc_species`:
- `host_organism`  — normalize_species(host)            (clean binomial)
- `source_species` — normalize_species(source_organism) (clean binomial; disambiguates from data-`source`)
- `engineered_mhc` — bool: genus(mhc_species) != genus(host_organism), both animal genera
- `xenograft`      — bool: genus(source_species) != genus(host_organism), both animal genera
- `chimeric`       — bool: engineered_mhc | xenograft

Heuristic detail (kills false positives found in audit):
- Compare at **genus** level so `Sus scrofa`≈`Sus sp.`, `Mus musculus C57BL/6`≈`Mus musculus` don't flag.
- "Animal genera" = genera that appear in `mhc_species` (MHC-bearing => animal). This excludes
  virus/bacteria sources (SARS, Mtb, vaccinia → not xenograft) AND immunization models
  (chicken-OVA-in-mouse: Gallus not MHC-bearing in corpus → not xenograft). Principled, self-adapting.
- Unparseable host strings (e.g. "B6.ERAAP null") → genus not in set → conservatively non-chimeric.

## Steps
- [ ] curation.py: add `species_genus(s)` helper (genus token of normalize_species; "" if empty)
- [ ] curation.py: `compute_species_axes(df, animal_genera=None)` — adds the 5 columns; idempotent
      (skips columns already present); derives animal_genera from df.mhc_species if None.
- [ ] builder.py: call compute_species_axes(obs)/(binding) before _atomic_write_parquet.
- [ ] observations.py: add `source_species=`, `host_species=`, `exclude_chimeric=` to
      load_observations / load_ms_observations / load_binding / _load_peptide_index.
      Compute axes post-load when columns absent (old parquet) so filters work without a rebuild.
- [ ] qc.py: `audit_species_axes()` — counts chimeric/engineered/xenograft + lists suspicious rows.
- [ ] tests: unit-test species_genus + compute_species_axes; integration-test the three filters via a
      small temp parquet. Cover FP cases (Sus scrofa/Sus sp., pathogen-source, strain suffix) and
      TP cases (HLA-tg rat, dog-tumor-in-mouse).
- [ ] version bump; format/lint/test; PR.

## Review section

Discovered the heuristic was already half-built: `is_chimeric_system` + `is_engineered_mhc`
existed in curation.py (tested), and export.py already materializes `is_chimeric` /
`is_engineered_mhc` in the observations EXPORT. Gap filled by this PR:

- curation.py: added `is_xenograft(source, host, mhc)` — host-axis counterpart of
  is_engineered_mhc. 3-arg (needs mhc) to avoid flagging heterologous-antigen studies
  (foreign protein on native host cells, host genus == mhc genus) as xenografts.
- observations.py: `_attach_species_axes()` derives host_organism / source_species /
  is_chimeric / is_engineered_mhc / xenograft at LOAD time (no rebuild needed; same pattern
  as is_non_peptide_ligand). Added `source_species=` / `host_species=` / `exclude_chimeric=`
  to load_observations / load_ms_observations / load_binding / load_all_evidence + the shared
  _load_peptide_index. Registered the 5 derived columns in _DERIVED_COLUMN_DEPS.
- qc.py: `species_axis_audit()` — groups chimeric rows by (host, source, mhc) triple with
  severity (info = clean engineered/xeno; review = chimeric w/ populated host, neither).
- tests: is_xenograft unit tests; loader filter/column integration tests (FP cases: pathogen
  source, substrain, heterologous antigen; TP: HLA-tg rat, dog-tumor-in-mouse); qc audit tests.

Real-corpus numbers (load-time derived, no rebuild): is_chimeric 1.64%, engineered 1.58%,
xenograft 0.32%; exclude_chimeric drops 72,813 / 4.44M rows. host-human (4.12M) ≠ source-human
(3.49M) — the conflation #46 set out to fix is now expressible.

Deferred (noted in PR): effector_organism, mhc_donor_individual, build-time materialization +
axis validation, per-PMID chimeric override curation, adding `xenograft` to the export schema.
897 passed, 2 skipped. lint/format clean.

## v1.49.x — curated sample metadata self-consistency (#372/#374/#375/#379)

Shipped: per-sample `species` honored (2 mouse samples were exporting as human); HLA-G
transfectants moved to non-classical; 3 unparseable `mhc` tokens fixed; species-inference traps
pinned to explicit forms; class filters normalized at both boundaries so `non-classical` is
reachable end-to-end (was 18 samples / 0 observations); zero-match filters return an empty frame
instead of raising KeyError; `_mhc_class_matches` unified with `_sample_class_tokens`.

Review round: fixed a real bug in `species_compatible` (compared raw strings before resolving, so
`"Gallus gallus (chicken)"` vs `"Gallus gallus"` was False), removed a dead `try/except ImportError`
on a hard dependency, derived the parquet spelling set from the alias table, cached
`normalize_mhc_class_token`, split the typo guard from the allele-join guard, added staleness
assertions to both allow-lists, and updated the CLI help + curation doc for the non-classical
vocabulary.

Deferred (filed): #380 serotype/locus `mhc` values never reach the allele join; #381 PMID 36423003
has real BoLA alleles in IEDB but is curated class-only; #382 species inference is pinned only in
curated YAML, the ingest path still misclassifies; #374 remainder (11 `I+II` samples need their
class-II genotypes read out of the papers).

## v1.51.0 — adopt mhcgnomes' species API (#383)

Shipped: floored `mhcgnomes>=3.39.0` (CI installs latest, so an unpinned floor is what let a green
local run ship a red CI); deleted `curation.species_compatible` in favour of
`Species.compatible_with`; replaced the trap-pinning tests with the real invariant — no curated
`mhc` token may resolve with `species_source == "inferred"`. That guard found 4 chicken `BF2*`
tokens resolving by cross-species inference (PMIDs 18612635, 36695776), now pinned with `Gaga-`;
inferred tokens 4 → 0. Patr-AL is `Ib` upstream so its allow-list entry is gone (contradictions
12 → 11).

Review finding #4: the source-vs-MHC species invariant reached only the test suite. The samples
table now exports `mhc_species` and `species_axes_agreement`, and the guard test asserts on the
column rather than re-deriving it — they were briefly two implementations and disagreed on 19
serotype/locus rows. `_SAMPLE_PROVENANCE_COLUMNS` extended so the `--with-expression-anchors`
variant carries them too. Corpus: 651 agree, 35 undeterminable (12 of them `mhc: unknown`), 2
disagree — both engineered chimeras (#46), correct as curated and now visible.

Next: #380 (serotype/locus values never reach the allele join), #381 (PMID 36423003 has real BoLA
alleles in IEDB but is curated class-only), #382 (species inference pinned only in curated YAML;
the ingest path still misclassifies), #374 remainder (11 `I+II` samples need class-II genotypes
read out of the papers).

---

## Serotype provenance and the unreachable half of the serotype map (#455, #458)

Reported from tsarina, which moved its `--serotype` filter onto this column and
found that two different facts share one name, and that six specificities never
appear at all.

### The two problems

1. **#455 — 41,478 rows carry no serotype that should.**
   `mhcgnomes.data.serotypes["HLA"]` spells alleles two ways: 915 entries use
   the compact `C*0304` form, 11 use `C*15:02`. The 11 are the hand-curated
   rows its generator cannot reproduce (mhcgnomes#156). `_build_allele_to_serotypes_map`
   keyed by whatever the table held while `allele_to_all_serotypes` looked up a
   compact key it built itself, so those rows were unreachable and Cw12, Cw14,
   Cw15, Cw16, Cw17 and Cw18 were absent from every annotation. Worst of these
   is Cw16: curated in deliberately from WHO's `hla_nom.txt` (mhcgnomes#153),
   and discarded here by a key format.

2. **#458 — `serotypes` mixes primary data with a projection.**
   35,257 human rows are serologically typed studies where the serotype *is*
   the observation and no molecule was measured. 1,630,309 rows name a molecule
   and get a serotype computed from it. Both spell the result identically. The
   marker was `allele_resolution == "serological"`, which is an inference the
   consumer has to know to make, from a column named for resolution.

### Plan

- [x] Normalize both sides of the reverse map through one key helper, and
      assert every table entry stays reachable.
- [x] Delete `_build_allele_to_serotype_map`, unused since `allele_to_serotype`
      started delegating to the plural form.
- [x] Add `serotype_source` (`reported` / `derived` / `donor_set` / empty) to
      `MhcAnnotation`, so the distinction is a column rather than a deduction.
- [x] Record `mhcgnomes_version` in `observations_meta.json`: the derived
      columns are only as current as the library that computed them.
- [x] Bump the observations artifact version so every existing index rebuilds
      its derived columns.
- [ ] Thread `serotype_source` through the loader, exports and CLI the way
      `restriction_evidence` (#415) is threaded.
- [ ] Document both axes in the README.
- [ ] Tests: reachability of the whole table, the three source values, the
      artifact-version invalidation, and the new filter.
- [ ] Rebuild the local index and confirm the derived columns regenerate.
- [ ] Version bump, three gates, PR, deploy.

### Deliberately not in this PR

`#456` (retired designations such as `B*44:01` never resolving to their current
name, because `parse()` runs with `use_allele_aliases=False`) is a semantic
decision about whether `mhc_restriction` may stop being what the paper
reported. It stays open for a call rather than being bundled here.

# PR #462 review fixes and release

Specification: [pr-462-release-spec.md](pr-462-release-spec.md).

- [x] Confirm current PR, review findings, dependency data, and release scripts.
- [x] Preserve species in both serotype lookup directions; add regressions.
- [x] Correct worker sizing and interpreter selection; make continuation tests deterministic.
- [x] Measure corpus impact and validate the declared dependency floor.
- [ ] Run format, lint, and all tests; record review results.
- [ ] Update PR description, push, verify CI, and merge.
- [ ] Deploy from clean main and verify PyPI publication.
- [ ] Review the next relevant issue group.

## Review

Species-aware annotation, inverse lookup, and the shared public query normalizer
now agree across the catalog (#463, #449). The comparison in
`pr-462-serotype-impact.json` changes 27 observation and 464 binding rows;
human annotations are unchanged. Artifact version 4 forces regeneration on the
next build. The dependency floor remains mhcgnomes 3.54.0.

The test runner uses free/speculative memory, a single worker when probes fail,
and Python's own pytest module. The continuation test no longer depends on
process startup fitting into five seconds (#440); actual process termination
remains covered.

Format/lint pass with the locked Ruff version. The isolated environment passes
68 targeted query/runner checks and all 252 curation tests at the mhcgnomes floor.
Full local validation and CI are in progress before merge and deployment.

---

## #444 — honor `exclude_from_ms` (1.62.3)

- [x] Give the flag a reader: `curation.ms_excluded_pmids()`, registered in
      `_clear_curation_caches` so a rebuild sees YAML edits.
- [x] Drop the rows in `builder._drop_excluded_from_ms`, called on `obs` only.
- [x] Bump `_OBSERVATIONS_ARTIFACT_VERSION` 4 → 5 so stale caches rebuild.
- [x] Regression suite in `tests/test_exclude_from_ms.py`.
- [x] Re-describe the key in `PMID_ENTRY_FIELDS` and `docs/pmid-curation.md`.
- [ ] Merge, then republish the CI corpus so the integration assertion goes live.

### Review

The flag was documented, curated on 11 studies, and read by nothing. It now has
exactly one reader, and the drop runs next to `_drop_short_mhc2_rows` — after the
supplementary merge, so it covers IEDB, CEDAR and supplement in one place.

Measured by running the shipped helper over the real 4.4M-row corpus:

| | before | after |
|---|---:|---:|
| MS observation rows | 4,480,783 | 4,440,428 |
| MS peptides | 1,341,351 | 1,310,337 |

40,355 rows go (0.90% of the corpus), from 6 of the 11 studies; the other 5
contribute only binding rows. 31,014 peptides leave the corpus entirely — they
were never MS-observed, only yeast-displayed, microarrayed or predicted. The
remaining 2,087 are also seen in genuine elution studies and stay.

What the dropped rows actually were: 12,490 `purified MHC`, 12,154
`High throughput multiplexed assay`, 6,285 and 5,608 fluorescence-based
`purified MHC` variants, 2 `x-ray crystallography`. All 40,355 carried
`is_binding_assay = False`, which is why IEDB's own flag never caught them and
the MS/binding fork put them on the MS side. Chen 2019 is the sharp case: 3,816
of its rows are labelled `cellular MHC/mass spectrometry` by IEDB, but the paper
is the MARIA prediction tool. Only a curator reading the paper catches that, and
one did.

**Scoped to MS evidence.** `binding.parquet` keeps all 472,497 rows from these
studies, including Wendorff 2020's 418,890 microarray measurements. `binding` is
the complementary mask of the same scan and is never derived from `obs`, so the
filter cannot reach it by construction; a test pins it anyway.

The integration assertion at `generate_observations_table()` skips against a
corpus built before this change, because a build-time filter is a property of the
artifact and asserting it against the old artifact tests nothing. The CI corpus
(`ci-corpus-v1`) predates it, so that test skips in CI until the corpus is
republished; the unit tier covers the drop logic unconditionally.

**Local suite is not a clean signal right now.** 215 tests fail identically on
this branch and on clean main at 6ed972f: the shared virtualenv has mhcgnomes
3.33.4 in site-packages, below hitlist's own `>=3.54.0` floor, shadowing the
3.64.2 sibling checkout. `Species.compatible_with` does not exist at 3.33.4. The
failure sets diff clean, and this branch adds 6 passing tests. Filed separately
as #467.

Format/lint pass with the locked Ruff version — `env -u VIRTUAL_ENV uv run
./format.sh` / `./lint.sh` against the lockfile's mhcgnomes 3.64.2, unaffected
by the shared venv issue above (#467). CI's four Python legs are the clean
signal for the full suite; all passed on 8886b37.

## #483: split `test.sh --all` into two pytest processes (1.62.11)

`deploy.sh`'s combined `--all` pass had OOM-killed the machine three times in
one day, even at a single xdist worker. `test.sh` now runs non-integration and
integration tests as two separate `python -m pytest` invocations (CI has done
this since #272/#274) instead of one process carrying both. Each pass
re-probes available memory independently and gets its own per-worker budget —
`PER_WORKER_GB` (2.5, unchanged) for the light pass, a new
`INTEGRATION_PER_WORKER_GB` (5) for the integration pass.

**Real end-to-end run, not just `--collect-only`.** Ran the new script for
real: light pass 1620 passed (13m55s), integration pass 42 passed / 1 failed
(8m42s). No OOM kill, despite the machine being genuinely memory-starved by
two unrelated repos' test suites (`pirlygenes`, `vaxrank`) running the whole
time — system-wide available memory sat under 0.2GB for 56 of the sampled
10s ticks during the run. That contention is real evidence the split
survives non-ideal conditions, but it also means this isn't a clean
before/after: I don't have a same-methodology peak-RSS number for the old
single-process `--all` to compare against, only the earlier session's
differently-measured 4.6-8.7GB range from the OOM-killed attempts (killed
before reaching whatever their true peak would have been, so not a ceiling
either).

**Process RSS trace (own descendant PIDs, summed):** light pass peaked at
~3.6GB transiently, mostly running ~200-350MB; integration pass reset to
~80MB at the fresh-process boundary, then climbed to ~11.6GB peak while
building/holding the observation corpus fixture. Take the integration peak
as an upper bound, not a clean physical-memory number — summed RSS across a
coordinator + one xdist worker double-counts any pages both processes have
resident from the same mmapped fixture file (`tests/xdist_cache.py`, #262).
The one clean, methodology-independent result: it finished both passes
without being killed.

**Unrelated integration-test failure found, not fixed here**: the fresh
run's integration pass failed
`test_current_corpus_mhc_token_allowlist_is_complete_and_not_stale` — the
test's hardcoded allowlist still expects `HLA-DR3A`/`HLA-DR7A`/`HLA-DR1B` as
`invalid_source` exceptions, but the currently-registered corpus no longer
has them. Reproduces cleanly on `origin/main` HEAD with no local diff to
either file, so it's pre-existing data/test drift, unrelated to this PR.
Filed as #484 rather than fixed in-flight, since diagnosing which side (test
allowlist vs. corpus) is stale is its own task.

`tests/test_test_script.py` previously only asserted the old single-pass
`--all` behavior (and never covered the plain, non-`--all` path at all).
Rewrote it to assert two independent invocations for `--all` (own marker,
own worker count from its own per-worker budget, right cov flags each), plus
a new test for the previously-uncovered default path, plus a test that
extra CLI args reach both passes. Confirmed both new `--all` tests fail
against the pre-#483 script (only one invocation ever appears) and the
default-path test still passes against it, so the rewrite isn't vacuous in
either direction.

## #483 (continued): preflight memory guard, so a low-memory run aborts instead of getting killed later (1.62.13)

The two-pass split (1.62.11/1.62.12) mitigated one failure mode but left the
issue's other two asks open: no preflight guard at the low end, and no clear
failure signal when the OS kills the process. This PR adds the preflight
guard; the retry-once-after-delay idea from the issue is still not done.

`worker_count()`'s `mem_cap` used to hard-floor at 1 in the `awk` step
itself, so a critically-low reading could never actually reach 0 by the time
`workers < TEST_SH_MIN` ran — the "floor to TEST_SH_MIN and proceed anyway"
behavior the issue complained about was baked in twice over, not just once.

Removed that inner floor so `mem_cap` can be 0. After the full pipeline runs
(CPU cap, `TEST_SH_MIN` floor, `TEST_SH_MAX` ceiling), compare the *final*
worker count against `mem_cap` -- not `TEST_SH_MIN` directly, which the first
draft of this fix got wrong: a case with `TEST_SH_MIN=10, TEST_SH_MAX=2` and
enough memory for exactly 3-4 workers used to abort under that draft (`3 <
10`), even though the final, TEST_SH_MAX-clamped count of 2 was perfectly
safe. Comparing against the post-clamp `workers` instead fixes that -- it
only aborts when the number of workers that would *actually run* needs more
memory than is available, after every floor and ceiling has already applied.

Caught this by hand-deriving expected worker counts (page counts × page size
÷ per-worker GB, with `int()` truncation) for each existing parametrized
case in `tests/test_test_script.py` rather than trusting the numbers already
there -- one of the four existing cases (`10_000` speculative pages) turned
out to only work under the *old*, incorrect version of this guard, and
silently masked a real design bug once actually verified against a hand
computation instead of copied from a neighboring case.

`available_bytes` probe failing (the existing macOS/Linux fallback) still
bypasses the guard entirely and proceeds at `mem_cap=1` -- no evidence of
scarcity, nothing to abort on.

Verified: all new/changed cases in `tests/test_test_script.py` fail against
the pre-guard script and pass against the new one. Real (non-stub) dry run
against this machine's actual `vm_stat`/`sysctl` output at 6GB available
proceeds normally with 2 workers, confirming no false-positive abort under
ordinary conditions.

Still open on #483: `deploy.sh` retrying the test step once after a short
delay before giving up, since the underlying pressure (other apps, not
hitlist) can resolve on its own.

## #488: pmhc --predictor mhcflurry crash + best-allele narrowing not restricted to the queried genotype (1.62.14)

Found running a real user query (4 cancer-testis-antigen genes x a
6-allele patient typing, `--predictor mhcflurry`) -- two real bugs in the
same code path, both fixed here since both block the same real command.

**Crash on >6 unique (peptide, allele) pairs.** `_predict_mhcflurry` called
`Class1PresentationPredictor.predict(alleles=[[a] for a in ...])` -- a
list-of-single-element-lists. That's neither of the two shapes mhcflurry's
`alleles` argument actually accepts (a flat list of <=6 allele strings as
one shared genotype for every peptide, or a dict of sample_name -> alleles
paired with `sample_names`). Falls into the flat-list branch, and the
*number of scored rows* gets checked against the 6-allele-genotype limit --
crashes on essentially any real query. Fixed by keying the dict by allele
(one-allele "sample" per unique allele) and using each row's own allele as
its `sample_names` entry. Verified manually against the real mhcflurry
package with 8 distinct alleles before writing the regression test.

**One atypical-length peptide crashed the whole batch.** A real 16-mer
MS hit made `predict()` raise (`Class1PresentationPredictor`'s affinity
model only supports lengths 5-15) and abort scoring the other 32 peptides
in the same query. `_score_and_narrow_to_best_allele`'s own docstring
already promised "peptide-length mismatch" degrades to NaN gracefully --
`_predict_mhcflurry` just didn't implement that promise. Fixed by filtering
to the predictor's `supported_peptide_lengths` before calling `predict()`
and leaving out-of-range rows NaN.

**Best-allele narrowing wasn't restricted to the queried allele set.**
Separately, real output showed `best_predicted_allele` values (e.g.
HLA-B*07:02, HLA-B*08:01) that weren't among the 6 alleles the query asked
about at all. `_score_and_narrow_to_best_allele` scores a row's *entire*
recorded ambiguity set (every allele any matched study's genotype carried)
and picks the global best -- `query()`'s own `alleles` filter (the specific
individual's genotype) was never threaded into the narrowing step, only
into the initial corpus-level pushdown. Added an `allowed_alleles`
parameter: when the caller supplied a specific allele filter (`--mhc-allele`
or `--sample`), restrict each row's candidate set to the intersection with
it before scoring, falling back to the full set only if that intersection
is empty. Re-ran the original real query after the fix: every
`best_predicted_allele` in the output is now one of the 6 queried alleles.

Filed #488 with the root-cause writeup; both fixes + narrowing landed here.
Also filed #491 (separate, not fixed here): `--mhc-allele` should split a
single space/comma-joined token and gain a `--mhc-alleles` alias -- the
same real query first failed with 0 rows because all six alleles were
passed as one quoted shell argument.

## #493: surface whether a peptide also occurs in an un-queried gene (1.62.15)

User asked whether `pmhc` checks for a candidate peptide (from a queried
CTA gene panel) also occurring in some other, un-queried protein -- a real
TCR-T target-selection safety question, since a peptide shared with a
normal-tissue gene is a very different risk than one unique to the tumor
antigen.

Turned out the underlying data already existed and was already being
loaded on every `pmhc` query: `load_observations`'s auto-attach mechanism
(#238) joins `gene_names`/`gene_ids` from `peptide_mappings.parquet`
whenever the caller requests them, which `query()` always does (needed to
split multi-gene rows, one row per gene). But right after exploding into
one row per (peptide, gene), the "final precise gene filter" drops every
row whose gene isn't in the caller's query -- silently discarding the fact
that the SAME peptide also matched a gene the caller didn't ask about. The
code's own pre-existing comment already named this exact scenario ("e.g. a
KRAS-attributed peptide that also matches NRAS") without ever surfacing
it.

Added `other_genes` (semicolon-joined genes besides the ones queried whose
protein also contains this peptide, empty if unique to the query) and
`n_source_genes` (total distinct genes, including queried ones) by
capturing each peptide's full multi-gene mapping before the precise filter
runs. Deliberately doesn't try to classify "harmless same-family paralog"
vs. "unrelated gene" -- checked a real query's output by hand and every
flagged gene turned out to be a CT-antigen-family paralog (CTAG1A/CTAG1B
alongside CTAG2, the SSX family alongside SSX1/SSX2, XAGE1A alongside
XAGE1B), not a genuine off-target, but that judgment call belongs to the
reader, not something to hardcode against a possibly-stale gene-set
snapshot.

**A real correctness gotcha caught by testing, not review**: the
per-donor-row consolidation `_score_and_narrow_to_best_allele` does after
predictor narrowing (`_collapse_rows_sharing_narrowed_allele`) uses an
explicit `agg_spec` for its `.groupby().agg()` call -- any column not
named there is silently dropped, the same trap `mhc_species` and the
`_line_ids`/`_donor_ids`/`_donor_type_ids` columns already had (per their
own comments). Verified by writing the fix, then deliberately removing
only the `agg_spec` addition and confirming exactly one test failed (the
consolidation-path test) while the two `query()`-level tests still passed
-- proof the fix was targeting the right, specific bug rather than a vague
"add it everywhere and hope."

Also: caught mid-implementation that I'd been editing these files directly
in the shared main checkout instead of a worktree -- broke the project's
own golden rule #1. Recovered via `git stash` in main + `git worktree add`
+ `git stash pop` in the new worktree, without touching main's history;
nothing had been committed yet so nothing was actually at risk, but noting
it since it's exactly the mistake [[feedback_worktree_when_concurrent]]
warns about.

## #491: pmhc --mhc-allele accepts a pasted genotype string; --mhc-alleles alias (1.62.16)

`--mhc-allele "A*03:01 A*02:01 B*14:02 B*44:02 C*08:02 C*05:01"` -- six
alleles quoted as one shell argument, which is how anyone pasting a typing
report's genotype line would naturally write it -- silently matched nothing
and returned zero rows. `action="extend", nargs="+"` only separates what the
shell already split into distinct argv tokens; a single quoted string
arrives as ONE token and got matched literally against `mhc_restriction`.
No error, just a plausible-looking empty result.

Added `_split_allele_tokens`, applied where `_pmhc` reads the parsed value:
each raw token is split on commas and whitespace and flattened, so quoted,
comma-joined, and repeated-flag forms all work and compose. Added
`--mhc-alleles` as a plural alias (argparse keeps `dest="mhc_allele"` from
the first option string, same as the existing `--protein`/`--gene` pair on
this subparser -- verified rather than assumed).

Deliberately scoped to `pmhc`. The `data`/`report` subcommand's own
`--mhc-allele` help text already claims "Space-separated, comma-separated,
or repeated" without implementing it either, but that's a separate
pre-existing mismatch on a different code path and isn't what #491 asked
for.

Tests cover the helper directly (pass-through, space-joined, comma-joined,
mixed, empty) plus two CLI-level tests that drive real argparse through
`main()` with a stubbed `pmhc_query.query`, so the alias's dest resolution
and the plumbing in `_pmhc` are both exercised, not just the pure function.
All seven fail against the pre-fix CLI (the alias ones with argparse's own
"unrecognized arguments" error).
## #483 (final): deploy.sh retries the test gate once after a delay (1.62.17)

Last of #483's three asks. The two-pass split (1.62.11) and the preflight
memory guard (1.62.13) shipped earlier; this is the retry.

The memory pressure behind a failed or OS-killed `./test.sh --all` during a
deploy is usually ordinary desktop app usage on the shared machine, not
anything test.sh did, and it often clears within a couple of minutes --
observed repeatedly across this session's deploys. One retry after a delay
turns a class of deploy failure that currently needs a human to notice and
re-run into one that resolves itself.

Deliberately ONE retry, not a loop: a second consecutive failure propagates
for real rather than masking a genuine break behind indefinite retrying.
`DEPLOY_TEST_RETRY_DELAY_SECONDS` (default 120) is the knob, matching
test.sh's existing env-var tunable convention.

Implementation leans on `set -e`'s existing semantics rather than fighting
them: `if ! ./test.sh --all; then ... ./test.sh --all; fi`. A command in an
`if` condition is exempt from `set -e`, so the first failure is caught; the
retry is the last statement in the block and is NOT in a conditional, so
`set -e` aborts the script with the retry's own exit code if it also fails.
Verified both branches by simulation before writing the real tests.

First tests deploy.sh has ever had. It calls `./lint.sh` and `./test.sh` by
relative path, so the tests copy the real deploy.sh into a tmp dir beside
fake versions of those, with fake python/twine on PATH so the build and
upload steps can never touch anything real. Three cases: transient failure
then success (proceeds, exactly 2 calls), two failures (aborts, exactly 2
calls -- not a loop, and never reaches the build step), and first-try pass
(1 call, no retry noise). The two retry cases fail against the pre-fix
script; the pass-through case correctly passes either way.
## #497: other_genes correctness fixes from code review of #493 (1.62.18)

Code review of the just-shipped #493 work found several real defects. The
serious one, reproduced live against the repo's own fixture before fixing:

**Ensembl-ID queries false-flagged every co-queried sibling.**
`other_genes` subtracted `names`, the raw resolved query set.
`resolve_gene_query` fills only `ids` for an Ensembl ID and leaves `names`
EMPTY, so an ID-based query subtracted nothing:

    by SYMBOL  (["NRAS","KRAS"]):  other_genes = "" , ""      correct
    by ENS ID  (same two genes):   other_genes = "NRAS","KRAS"  false alarm

Exactly what an existing test pins as must-not-happen, just reached by the
other input shape. The same subtraction was also too WIDE: `names` is
HGNC-alias-expanded, so a gene whose approved symbol collides with one of
those aliases got silently erased from `other_genes` -- a false negative in
the one field whose whole job is to surface cross-reactivity, and strictly
worse than a false positive because the reader sees an empty cell and stops
looking.

Fixed by subtracting the symbols that actually survived the precise gene
filter instead of the raw query set. Covers both input shapes, drops nothing
to alias collisions, and on an unfiltered scan the set is empty by
construction so the field degrades to "every other gene", the only thing it
can mean when nothing was asked for.

Also from the same review: `other_genes` had no truncation while
`format_table` computes column widths once across the whole result, so one
peptide hitting a paralog family (PRAMEF, CT45A, GAGE) padded every other
row out to match -- now truncates like `pmids` does. `n_source_genes` was
documented and in `_empty_result` but never rendered in the default table.
My own comment calling `other_genes` a "peptide-level constant" was false
(it subtracts the row's own gene, so it's (peptide, gene)-level; `"first"`
aggregation is safe only because `gene_name` is in `group_cols`, and the
comment now says exactly that). Docstring covered only the filtered case.
Row-wise `apply(axis=1)` and a per-row lambda vectorized; `observed=True`
added to the one groupby in the file missing it.

**One review finding deliberately NOT taken.** It proposed fixing
`n_source_genes`'s undercount by counting Ensembl IDs instead of symbols.
Measured against the real 5.9M-row mapping table first: blank `gene_id` on
114,703 rows vs blank `gene_name` on 29,376 -- counting IDs would lose ~4x
MORE loci. Neither field alone is a complete key, and name/id pairing is
already unreliable by the time query() sees it (two independent
`_join_unique` calls upstream, then padded and exploded). A sound count has
to be derived at mapping time. Filed as #496 with the measurements; the
docstring now states the limitation ("at least this promiscuous, never at
most") rather than shipping a change that looks like a fix but isn't.

Tests added for the two branches that had none and where the semantics
actually differ: Ensembl-ID queries (fails against the shipped code) and
the unfiltered whole-corpus scan.

## #496: count source LOCI, not gene symbols, for n_source_genes (1.62.19)

The follow-up deliberately deferred out of #497. `n_source_genes` counted
distinct gene SYMBOLS off the exploded `gene_names` view, which silently
dropped every locus with no HGNC symbol -- and always in the reassuring
direction, so a reader concluded a candidate peptide was more gene-specific
(a safer TCR-T target) than it is.

**Measured before choosing a key**, against the real 5,875,577-row mapping
table, because the obvious fixes are both wrong:

| tier | rows |
|---|---|
| Ensembl ID present | 5,760,874 |
| no ID, symbol present | 106,470 |
| neither | 8,233 |
| blank `protein_id` | 0 |

So "just count gene_id" loses ~4x more loci than counting symbols does.
And a naive symbol-when-no-ID fallback double-counts, because **5,487
symbols appear BOTH with and without an ID** -- they'd land in one bucket
under the ENSG and another under the bare symbol.

New `mappings.locus_keys` therefore keys on the Ensembl ID, falls back to
the symbol *resolved to its ID first* (learned from the rows carrying both,
so the 5,487 collapse correctly), then the bare symbol when the corpus
never pairs it with an ID, and finally `protein_id`, which is never blank.
The last tier over-counts a nameless multi-isoform locus rather than
dropping it -- the safe direction for a promiscuity signal, and documented
as such.

Computed in `annotate_observations_with_genes` where the frame is still one
row per (peptide, protein) and gene_name/gene_id are genuinely row-aligned,
then wired through `_GENE_DERIVED` / `_DERIVED_COLUMN_DEPS` like the other
four derived columns, so every consumer of observations gets it rather than
just `pmhc_query`. That also removes the second, query-time code path that
was answering the same question differently.

**Corpus-wide impact:** of 1,282,910 peptides, 25,622 (2.00%) change count.
All 25,622 are increases; zero decrease, which is the evidence the
symbol->ID resolution didn't introduce double-counting. `AAAAAAPPPST` goes
from **0 to 2** -- the old count claimed it came from no gene at all.

`other_genes` still can't NAME an unnamed locus, so the docstring now points
out the tell: `n_source_genes` exceeding the `other_genes` entries plus the
row's own gene means there are unnamed loci behind it.

## #502: centralize "what is one sample"; --by-gene n_samples was ~always 1 (1.62.20)

`pmhc --by-gene` reported **1 sample for CTAG2 across 17 references** —
impossible on its face. `gene_distribution` counted
`nunique(attributed_sample_label)`, and that field is blank on 4,295,716 of
4,440,124 rows (**96.7%**), so every uncurated row collapsed into one
bucket: **13** distinct samples corpus-wide against **2,244** with the PMID
fallback. CTAG2 now reads 14, SSX2 1 -> 5.

The deeper problem wasn't the arithmetic, it was that `query()` had already
solved this properly — PMID fallback plus cell-line/donor decomposition —
so the repo published one column name under two definitions that had
drifted by two orders of magnitude. Same shape as #496's "two code paths
answering one question".

**Centralized rather than merely shared.** New `hitlist/sample_identity.py`
is the canonical home, with a fully public API (`SAMPLE_IDENTITY_COLUMNS`,
`SAMPLE_IDENTITY_ID_COLUMNS`, `sample_identity_ids`,
`add_sample_identity_columns`, `count_distinct_ids`, `count_samples`). The
attached columns are public too — `line_id` / `donor_id` / `donor_type_id`,
previously underscore-prefixed, which wrongly signalled callers shouldn't
rely on them when they are exactly the shared contract.

Also fixed the cell-line tier's missing last-resort fallback: a
`src_cell_line` row carrying neither a line name nor a mono-allelic host
got an empty ID and vanished from the count, where the donor tier had used
a PMID fallback all along (21 rows / 4 PMIDs / 19 peptides).

**Drift guard**, per the repo's no-private-interfaces rule: a test scans
every module for sample counts taken off the raw label and fails naming the
offending `file:line`. Verified it actually fires by reintroducing the bug
in `export.py` and watching it point at `export.py:4219`.

**Deliberately NOT unified:** `samples.py::sample_peptidomes` is a third
definition, grouping raw scanner output by
`(pmid, antigen_processing_comments)`. It runs one pipeline stage earlier,
before curation attaches `src_cell_line` / `cell_line_name` /
`monoallelic_host`, so it *cannot* use this definition. Documented in the
module docstring so nobody unifies it later and breaks it.

One test lesson worth recording: the cross-path agreement test initially
PASSED against the broken code, because the existing fixture has every
`attributed_sample_label` curated, so the old count agreed by accident.
Rewrote it on an unlabelled fixture where it fails pre-fix with
`assert 1 == 3` — the CTAG2 symptom in miniature. A green test that cannot
fail is worse than no test.
# Priority release prerequisite: phase-local memory retry (#526)

The pending clean-main 1.62.26 deployment reproduced #526: all 1,676
regular tests passed, the integration memory guard refused at 1.62 GiB,
and free memory recovered above 8 GiB during the retry delay. The current
deployment then reruns all regular tests instead of retrying the refused
integration preflight. Prepare the already-reviewed #527 patch directly
on main as a release prerequisite if this blocks publication again.

Keep the 2.5/5 GiB budgets and one retry. Only preflight refusals can retry;
actual pytest failures must stop immediately. Retain serial fallback guards,
test each phase independently, and preserve the reserved version 1.62.35.
Do not incorporate unrelated curation from the old stack.

- [x] Preserve the old #527 branch in a named backup.
- [x] Apply only its phase-retry implementation and script regression tests.
- [ ] Run format/lint/script tests, full tests, and supported-version CI.
- [ ] Review, merge and deploy before correctness PRs only if needed to
      resolve their release blocker; otherwise retain as the next foundation.

Restacked review: only deploy.sh, test.sh, their tests, version and planning
notes differ from main. All 20 script tests pass (51.48 s); format and lint
pass. Full local validation and final CI are still required.

# Retry memory preflight at the test phase boundary (#526)

Deployment currently repeats an already-passed 11-minute regular suite when
the separate integration process cannot start under its 5 GB memory guard.
Keep both phases and their budgets, but retry only the current phase's memory
preflight, once, inside a single invocation. Actual pytest failures must fail
immediately. A new deployment invocation must always run both phases again;
there is no persisted success token or cross-revision reuse.

Add an explicit `test.sh --retry-memory` option used by deploy.sh. Keep the
existing deploy delay environment setting, forwarding it to the test runner.
Probe again after the delay and cap retries per phase. The same guard must
apply to the actual single worker when pytest-xdist is absent; otherwise
serial fallback bypasses both memory protection and its retry path.

- [x] Reproduce phase replay and unguarded serial fallback with command stubs.
- [x] Implement bounded preflight-only retries, preserving the memory budgets.
- [x] Cover regular/integration refusal and recovery, exhausted retries, real
      test failures, new invocations, and serial/xdist worker counts.
- [ ] Run format.sh, lint.sh, test.sh, CI and self-review.
- [ ] Bump 1.62.35, open its own PR, merge and deploy from clean main.

Review: seven new resource-control regressions fail before the change; all 20
test/deployment script cases pass afterward (37.50 s). Format/lint pass.
Self-review confirms unchanged 2.5/5 GB budgets, one preflight retry per phase,
fresh probes after each wait, actual one-worker accounting without xdist,
immediate propagation of real test failures, no build after failed tests, and
no successful-phase state persisted between invocations. Existing deploy delay
configuration is preserved. Full tests/CI and release remain pending.
#358 RNA curation remains uncommitted and will use the next version when its
verified primary inputs and complete profile QC are ready.

# Retired allele identities without rewriting reported typing (#456)

## Priority release plan for genotype correctness (#528, #514)

The user elevated mutation-token fabrication and genotype-contradicting
attribution as urgent correctness defects. Their implementation exists in
PRs #530 and #531, but the old stack makes them wait for unrelated changes.
Retain #529's allele-identity handling as the correctness prerequisite.
The pending main release reproduced #526, so place its isolated phase-retry
repair first: #527 -> #529 -> #530 -> #531. Preserve the reserved
versions 1.62.35 through 1.62.38; patch versions may skip unshipped
numbers. Other PRs must be rebased and bumped beyond the priority releases
before they land. Preserve named backups of all original branch tips.

No scientific values should be changed by restacking. Re-run mutation
ownership and downstream join regressions, missing-tissue genotype tests,
and corpus audits on the new bases. Retain uncertainty for coarse typing,
missing loci, and predicted restrictions. Recheck latest development
dependencies without modifying an environment while tests are running.
Complete format, lint, full tests, final CI, review, merge and clean-main
PyPI deployment for each PR in dependency order. Never weaken memory guards
or report an unreleased draft as fixed for installed users.

- [x] Back up branch tips and restack the three priority PRs on main.
- [x] Verify regressions, current curation impact, and supported Python versions.
- [ ] Run format/lint/full tests and final-head CI on the restacked PRs.
- [ ] Publish pending main release, then merge/deploy #529, #530, and #531.
- [ ] Rebase and renumber the remaining queue after the priority releases.

Priority review: the new stack contains only allele identity, mutation
tokenization, and genotype-consistency changes. All 80 combined regressions
pass on Python 3.9 and 3.12. The mutation audit preserves every candidate
and precision category across the current main-based 772 sample records
and 376 distinct MHC fields, with unchanged YAML. No curated candidate
currently carries a mutation. The genotype audit again changes zero values
over 12,888 patterns representing 3,644,028 observations from 102 studies.
The identity audit still changes only B*44:01 on 7,127 rows across the full
5,333,255-row vocabulary. All six development dependency heads remain current.
Full local suites and final CI remain pending. With memory restored, the
pending clean-main 1.62.26 deployment has begun its serial full test gates.

IPD-IMGT/HLA 3.65.0 Deleted_alleles.txt (2026-07-14), HLA00317, confirms
B*4401 was a sequence error identical to B*44:02:01:01 (March 1994).
The audited 832-token vocabulary changes only B*44:01, on 7,127 rows.
Preserve the source's normalized restriction and field depth. A separate
cached identity resolver applies mhcgnomes aliases only when the gene or
first two allele fields change, truncating any added fields to the original
depth. Keep mutations on their original chains; require mhcgnomes 3.64.3,
which fixes upstream #193. Never enable aliases in the reported-value parser.

Use this resolver for derived serotypes, candidate sets, sample matching,
loader filters and pMHC aggregation/prediction candidates. Refresh stored
candidate identities before set filtering and refresh serotypes only for
restrictions containing a renamed allele. Preserve raw restrictions in
loaders and expose their distinct reported spellings in aggregate results.
Avoid deriving experimental certainty from an alias or expanding a genotype.

- [x] Read primary IPD retirement evidence and audit all restriction tokens.
- [x] Fix, review, merge and publish upstream mutation preservation (#193).
- [x] Add failing regressions for aliases, typing depth, mutations and chains,
      raw provenance, stored-index projection/filtering, aggregation and scoring.
- [x] Implement one derived identity helper and route relevant consumers to it.
- [x] Audit both complete indexes; assert only the documented identity changes.
- [ ] Run format, lint, focused checks, full tests and final CI; self-review.
- [ ] Bump 1.62.36, open its own PR, merge and deploy after the preceding releases.

Review: 37 initial regressions failed before the change. Format/lint and 463
focused tests pass on Python 3.12; 461 cases also passed on Python 3.9 before
adding the final two duplicate-set/projection regressions. All 929 catalog
members remain reachable. Current catalog memberships remain preferred over
memberships inherited from a retired designation. No genotype is expanded.

The 5,333,255-row audit covers both complete indexes and all 832 tokens.
Only B*44:01 changes identity (7,127 observation rows); all raw restrictions
and all parsed mutations retain their identity/chain. Canonicalizing a stale
catalog member B*15:112 -> B*15:11 additionally restores B15 membership on
63,990 observation rows. The primary HLA nomenclature broad/split table lists
B75 under B15: https://hla.alleles.org/pages/antigens/broads_and_splits/.
A direct old-index loader comparison verifies 71,117 serotype updates, 7,127
candidate updates, unchanged candidate counts and zero reported-name changes.
No binding rows change. Stored current names also refresh the affected
catalog membership, including under projection and serotype filters.

The latest six development dependency heads were checked; pyensembl was
updated to 2.10.17 at d4abbf26, and mhcgnomes 3.64.3 b276be94 is now the
required floor. The lockfile changes only that dependency and its requirement;
uv lock --check and pip check pass. The full feature suite was canceled to
reserve memory for the active clean-main release; it is not counted as a pass.
Full tests and final CI remain required.

Independent problems found and filed: mhcgnomes#196 (missing broad/split
memberships), #197 (missing documented retirement), and hitlist#528 (sample
field tokenization fabricates alleles from mutation labels). The resolver
preserves mutations; it cannot restore information already lost by that
separate tokenizer. Upstream #193 is fixed and published. RNA curation #358
remains independent and is reserved for 1.63.0.

# Parse mutant sample genotypes as molecules (#528)

Review follow-up, 2026-09-23: the same whitespace split remains in
`extract_allele_tokens` and `mhc_species_of`. The first invents HLA-E*76C
from B*08:01 E76C mutant; the second falsely assigns a mouse mutant both
human and mouse species. File the reproduced sibling defect and include
these consumers in the shared molecule segmentation fix. Move semicolon
handling into that shared helper. Preserve reported allele names in token
extraction, derived identities in sample matching, and coarse species
resolution. Audit all current sample fields for token/species changes as
well as candidate changes; reject malformed mutation tails consistently.

- [x] File the sibling-helper defect with exact reproductions (#537).
- [x] Add failing extraction/species regressions and share segmentation.
- [x] Repeat current-curation audit and focused tests on Python 3.9 and 3.12.
- [ ] Require final supported-version CI after this review follow-up.

Whitespace splitting can turn `E76C mutant` into an invented HLA-E allele,
while stripping the mutation from its actual B allele. Preserve complete
parseable molecules/pairs before splitting genotype lists. For mixed fields,
consume the longest parseable molecule/serotype/locus span, using mhcgnomes
instead of an HLA-only token regex. Keep whole-field coarse names and existing
list handling. Mutation tokens or a mutant marker left outside a successfully
parsed molecule must raise an explicit input error, never fall back to a
wild-type genotype. Do not mistake a compact genotype allele for a mutation;
use mhcgnomes' Mutation parser and test actual curated field vocabulary.

Require mhcgnomes 3.64.4 once upstream #198 passes CI and ships: whole-pair
alpha selectors must retain chain ownership before this tokenizer trusts
those parsed results. Test against the isolated upstream branch until then;
do not replace a dependency underneath an active release test process.

- [x] Add failing regressions for single mutants, mutated pairs, mixed lists,
      selectors, invalid mutation tails, and ordinary multi-allele genotypes.
- [x] Implement molecule-aware segmentation with explicit malformed-input errors.
- [x] Audit every curated sample MHC field before/after and preserve raw YAML.
- [x] Require the shipped upstream fix, update the lockfile and dependency envs.
- [ ] Run format, lint, full tests and final CI; review and fix findings.
- [ ] Bump 1.62.37, open a separate PR, merge and deploy in order.

Review: format/lint and 459 focused tests pass; all 15 new cases also pass on
Python 3.9. Tests exercise the export join for separate mutant and wild-type
arms, including full class-II pairs, in addition to segmentation and invalid
mutation rejection. Every candidate and precision category is unchanged over
all 775 curated sample records (376 distinct field values), with identical
raw YAML SHA-256. No mutation-bearing curated sample field currently exists;
the regressions use explicitly synthetic inputs.

Requires the published mhcgnomes 3.64.4, whose chain-selector fix passed
17,171 tests and CI on Python 3.9–3.12. Both isolated hitlist environments
use its latest merged development commit 5565acd. The lockfile only changes
mhcgnomes and its requirement; uv lock --check and pip check pass. The full
test entry point was refused before pytest by its unchanged memory guard
(0.22 GiB available; 2.5 GiB required), so full local validation remains
pending. Final CI, merge, and clean-main deployment are also required.

Priority-stack review: shared segmentation now also protects allele extraction
and sample species (#537), including class-II chain ownership, mouse mutants,
true mixed-species genotypes, haplotype/coarse species and malformed tails.
All 333 curation, mutation and identity checks pass on Python 3.9 and 3.12;
format and lint pass. The repeat audit covers the 772 samples on the direct
main-based priority stack, with 376 distinct fields: candidates, precision,
extracted molecules and species are all unchanged. Raw YAML SHA-256 is
7380766616315777df7593946a7637a30c7afe21cf5fabaa84ef4e08771e74d0.
All mutation examples remain synthetic. Final CI and the full local suite
are required again after this review change. Reserve 1.63.0 for RNA #358.

# Respect reported genotype evidence in class-pool attribution (#514)

The class-pool text scorer currently ignores the observation's restriction,
so a generic display-label word can select a sample with a disjoint reported
genotype. Re-read Ritz 2017 (PMID 28834231, PMC5846733): both cell lines were
SSO/SSP typed; MAVER-1's six alleles and homozygous HEK293's three alleles are
explicitly distinct. Reproduce the missing-tissue failure before editing.

Validate the proposed class-pool text/group winner without changing text
scoring weights: when observation and sample contain precise, reported
molecules at the same loci, reject a disjoint named winner. Unknown,
coarse, or serological typing must not be treated as complete exact evidence,
and derived/predicted candidate sets must not be used as reported typing.
Respect class-II chain versus complete-pair precision and mutation identity.
Missing loci are unknown, and an untyped candidate cannot win merely by
surviving exclusions. Genotype evidence may select a sample only when exactly
one candidate overlaps and every alternative is explicitly disjoint. Include
the restriction and its evidence category in row-discriminator
keys so rows with identical text but distinct typings cannot share a winner.
Leave curated per-row sample labels authoritative and keep raw data intact.

- [x] Reproduce the paper-grounded missing-tissue case and precision edge cases.
- [x] Implement the conservative winner check and use precise row keys.
- [x] Audit changed attributions against the stored corpus and primary sources.
- [ ] Run format/lint/full tests and final CI; review all behavioral changes.
- [ ] Bump 1.62.38, open a separate PR, merge and deploy after #530.

RNA #358 remains independently in progress and will use 1.62.39.

Review: three source-grounded export regressions fail on the base. Initial
testing exposed two unsafe generalizations, both corrected before review:
filtering candidates before text scoring changed relative token weights and
could promote an untyped cell line; treating absent loci as negative typing
would reject DRB3/DRB4 observations from samples typed only at DRB1. The final
check preserves scoring, handles gene coverage as uncertainty, and uses the
existing species-ancestry relation for Calu/DLA naming. Pair-vs-chain identity,
mutation identity, retired names, serotypes, partial alleles, unknown fields,
predicted restrictions and identical-text/different-genotype rows are tested.

Audit all 12,888 distinct attribution input patterns across 3,644,028 stored
observations in 102 multi-sample studies: zero existing attribution changes.
Raw source YAML and restrictions are unchanged. The Ritz paper explicitly
reports the two SSO/SSP-typed genotypes and HEK293 homozygosity; the missing-
tissue regressions reconstruct that source evidence. No full-corpus column
copy is added: restriction/evidence are part of the existing unique-row keys.
Full local tests and final CI remain required before merge, with clean-main
PyPI publication afterward.

Final focused validation: format/lint and 201 export/curation tests pass;
all 22 new cases also pass on Python 3.9. The audit remains unchanged after
using the existing restriction/evidence columns directly as row keys.

## PR #533 release-runner restack (1.62.40)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


# Attribute exact deposited elution conditions (#512)

Re-read Stopfer 2020 (PMID 32488085, PMC7265461), Figures 4/6 and the
Supplementary Data 3/5 file maps: separate DMSO, 1/10 uM palbociclib, and
10 ng/mL IFNG arms were profiled at 72 hours. IEDB contains seven exact
treatment-enumeration statements and seven independent MDA-MB-231
quantification statements. Of 12,277 melanoma observations, 8,390 name
one treatment (5,528 IFNG, 2,171 high-dose and 691 low-dose palbociclib);
3,887 name multiple arms, not simultaneous combination treatment.

Add a validated PMID-level YAML map from exact assay-comment strings to
the existing condition IDs supported by each unambiguous statement. Map the
three single-treatment statements; the four multiple-arm statements remain
unmapped and cannot select an arm, even if a caller supplies fewer candidates. Preserve
the unchanged comment text as source evidence; never use substring matching
for doses. Validate that keys are nonempty strings and values are nonempty
lists of distinct condition IDs declared by this study's profiled samples.
The existing elution-condition selector will use an exact curated match
when available, and return a winner only when exactly one candidate ID is
supported. Use it in both allele-match and class-pool tie-breaks. Keep raw
observations and the existing generic narrative admission rules unchanged.

- [x] Add failing source-comment/dose-boundary/multi-arm and schema tests.
- [x] Implement the validated mapping and both selection paths.
- [x] Review all seven statements and curate only the three unambiguous ones.
- [x] Audit all 15,821 observations and all 14 distinct comments before/after.
- [ ] Run format/lint/full tests and final CI; review, merge and deploy 1.62.40.

Review: 473 expanded curation/export tests pass and both the class-pool and
ambiguous exact-allele paths select source-supported condition IDs. The
audit covers 36,584 rows across Stopfer, Ritz, and Schellens: exactly 8,390
Stopfer rows acquire an arm, 3,887 multi-arm rows stay ambiguous, all 3,544
MDA quantification assignments are unchanged, and every value in the other
two studies is unchanged. Raw peptides, restrictions, comments, cell names,
and sample groups are identical. The categorical vocabulary expands only
to accommodate newly assigned metadata. Full local tests and final CI remain
required before merge, followed by clean-main PyPI publication.

Found and filed #532: filtering to one row can remove sample-group identity
because discriminator variability is computed within the selected query.
The unchanged base reproduces this with two Stopfer rows versus one; the
source-mapping tests exercise the complete multi-line study context. This
separate query-stability fix follows #512; no generic narrative guard is
relaxed here. RNA #358 remains independent and will use 1.63.0 once its
source data and validation are complete.

Final review: all 28 regression/schema cases pass on Python 3.9 and 3.12;
the 13 public-export regressions fail against the unchanged #531 base.
Format, lint, and whitespace checks pass. Full local tests and final CI
remain pending in the serial validation queue.
# Restack the next attribution fixes after the priority releases

Keep the reviewed #512 -> #534 -> #532 changes behind #528 and #514.
The old branches still include the unrelated queue that was removed from
the priority stack. Back up all three tips, then apply only their own
commits onto the final #531 head and preserve versions 1.62.40–1.62.41.
The independent RNA work remains reserved for 1.63.0.

This is a topology repair, not a new curation pass. Preserve the source-
verified dose statements, cohort restrictions, and query context rules.
Compare scientific source/test/data diffs against the old heads; investigate
any change beyond removal of unrelated ancestor commits. Repeat focused
checks and the recorded source audits on the new bases before release.
Keep memory-heavy audits and full suites serial with the active deployment.

- [x] Preserve old branch tips and restack only each issue's own commits.
- [ ] Review conflicts, confirm scoped diffs, and rerun focused validation.
- [ ] Repeat source/corpus audits and final supported-version CI.
- [ ] Run format/lint/full local tests, review, merge and publish after #531.

## PR #536 release-runner restack (1.62.41)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


# Keep peptide-to-patient attribution within its source cohort (#534)

Primary-source review confirms that Sarkizova 2020 (PMID 31844290,
PMC7008090) contains independent monoallelic 721.221 and patient-tumor
experiments. Supplementary Data 2 (publisher MOESM4, SHA256
46bf469653a4e252825f185e1d2946a717aace0d9e894cb0c0e81903100bbfca)
contains patient sample directories, including MEL_13240_005 and its IFNG
arm for NAPWAVTSL. Peptide overlap does not transfer that patient identity
to a separately recorded monoallelic assay. Reconcile the existing CSV
against the workbook without changing its valid patient assignments.

Declare `peptide_attribution_restrictions` in PMID YAML: an optional
validated nonempty list of exact reported restriction strings to which the
registered peptide map applies. For Sarkizova, allow only the deposited
`HLA class I` patient-cohort restriction. Match before donor-set promotion;
do not infer cohort eligibility from derived monoallelic flags or require
source classification to be enabled. Unscoped Connelley maps retain their
source-verified exact-allele behavior. Preserve the public peptide map API.

Repair derived labels in existing observation/binding artifacts at load:
for scoped studies, retain genuinely narrowed `peptide_attribution` rows,
but clear labels attached to out-of-scope exact restrictions. Remove only
the duplicate donor copies of those invalid rows, keyed by their original
PMID, assay IRI, peptide, and restriction. Retain raw evidence and valid
patient splits. Require a rebuild if a stale affected row has no source
assay identity; never guess which repeated rows are independent assays.
Projection must pull the repair inputs and then return exactly the requested
columns, including a one-column peptide query. Bump the artifact version
so a normal rebuild regenerates the corrected scan rather than skipping.

The current MS artifact has 89,723 mislabelled monoallelic rows representing
47,641 source assay/peptide/restriction records: 42,082 excess donor copies.
Its 54,682 valid patient-attribution rows must be unchanged. Audit the
complete affected study and both index modalities, then check scanner output
from original source rows and export metadata, including HLA-G. This fix
must precede the class-roster change in #535/#532.

- [x] Reconcile patient CSV pairs against original Supplementary Data 2.
- [x] Add failing mixed-cohort scan, old-index repair, projection, and schema tests.
- [x] Implement validated source scope, scanner guard, and derived-index repair.
- [x] Audit affected study and preserve Connelley exact-allele attribution.
- [ ] Run format/lint/full tests and CI; review, merge and deploy 1.62.41.

Review: the CSV exactly matches all 39,624 original workbook peptide–patient
pairs (zero missing or unsupported pairs) and remains unchanged. Across the
stored study, 89,723 false labels become 47,641 unlabelled independent source
records, removing only 42,082 excess donor copies. Their retained evidence
is identical; all 54,682 valid patient rows and all 146 Connelley MS rows
are identical before and after, including their exports. HLA-G rows now
export the corresponding 721.221 transfectants and HLA-G typing. Re-scanning
the original IEDB/CEDAR rows independently gives the exact same result as
repairing the old scan, with all 222 Connelley scan rows unchanged.

The initial regression run failed 14 cases on the unchanged base. Format,
lint, and 391 expanded tests pass; the additional public-export regression
also passes. All 64 final scanner/repair cases pass on Python 3.9. The local
full suite and final CI remain required; memory guards currently prevent
release validation. Latest development HEADs for mhcflurry, mhcgnomes,
pyensembl, datacache, gtfparse, and serializable were rechecked and match the
isolated test environment. This PR remains a draft until its gates pass.

## PR #535 release-runner restack (1.62.42)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


Restack review: stable patch IDs for each scientific source, data and test
delta are identical before/after: #512 4a8b5368, #534 68097bf1, and #532
4300627a. The sole conflict was between appended planning sections; both
were retained. Format and lint pass on all three new branches. Fresh CI and
repeat corpus/source audits remain pending, with full local suites kept
serial behind the active 1.62.35 deployment.


# Make sample attribution stable under output filtering (#532)

The same deposited observation must retain the same sample/group/condition
when a caller asks for one peptide, one restriction, or another narrower
output. The current variance guard measures the selected query, so removing
other rows can erase a known cell identity or admit narrative after a factual
field stops varying. Preserve the full-study boilerplate guard and the
existing unfiltered attribution behavior.

For queries with row filters, read a compact context containing only PMID,
restriction, class, and the four discriminator fields from the complete MS
artifact. Stream projected parquet batches and collapse duplicate patterns;
cache against path, nanosecond mtime, and size so rebuilds invalidate it.
Restrict the compact context to queried curated multi-sample studies. Do not
load peptides, expression, or other wide payloads for the context.

Use full-context restriction pairs when constructing sample join aliases.
Evaluate ambiguous-allele discriminator variation and winners against the
context; apply winners only to selected output rows. For the class-pool
variance gate, retain only context rows that the preceding allele/single-
sample paths leave without sample MHC, exactly as the current full-query
path does. Reuse the same candidate and winner dictionaries. Never add
context observations to the returned frame or change scoring weights.

Keep the complete curated sample roster when filtering observation class:
class selection must not turn a multi-sample study into a single-sample
study and change its fallback provenance or matched-sample count.

Tests must use a real temporary parquet and public filters, not simulate a
filtered loader that also hides the full study. Cover grouped cell identity,
exact-allele ties, ungrouped samples, treatment-dose attribution, unchanged
multi-arm ambiguity, and genuine constant narrative. Compare every relevant
metadata field between full and one-peptide exports; verify row counts,
raw values, and context-cache invalidation. Audit unfiltered output against
the base and a representative filtered query for each multi-sample study.

- [x] Add public-query regressions and demonstrate base failures.
- [x] Implement bounded full-study discriminator context without changing curation.
- [x] Finish memory review and all filtering paths after the curation prerequisite.
- [x] Audit full versus filtered attribution across the corpus.
- [ ] Run format/lint/full tests and CI; review, merge, deploy 1.62.42 after #534.

Review: 10 public-query regressions fail against the original base and two
class-roster cases fail before retaining the complete roster. Rebased onto
the source-verified cohort repair in #536; added explicit old-index HLA-G
checks for peptide, allele, and class queries. Format/lint and 275 expanded
export/repair tests pass after rebase. All 18 final query-context cases pass
on Python 3.12 and all 38 query-context/repair cases pass on Python 3.9.
Full local tests and final CI remain required before merge and clean-main
deployment.

The repeated audit uses freshly reconstructed patterns after correcting the
cohort data: 11,850 patterns represent 3,601,946 observations from 102 studies.
Every unfiltered value is identical to the corrected base. Of 132 narrow
queries (66 known and 66 ambiguous), the base's 24 known-sample discrepancies
disappear and all ambiguous cases stay unchanged. Every class-filtered
pattern matches the complete export. Before this change class filtering
changed matched-sample counts on 1,506,759 represented rows and attribution
provenance on 38,252, while all sample labels now agree after the separate
cohort repair. No scoring weights, source YAML, or stored artifacts change
in this PR. The compact context contains seven columns, streams 10,000-row
batches, and caches two file revisions without retaining peptide payloads.

## PR #516 release-runner restack (1.62.43)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #516 priority queue restack (1.62.43)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #289 specification — reusable manual-dataset guidance

IEDB/CEDAR now support automatic fetching, but HPA and DepMap still reach the
reported hard-coded CLI message. Make both `fetch` and missing `get_path`
manual-dataset errors name the download URL, expected file and registration
dataset key without choosing a downstream application's command prefix.
Keep exception types and behavior unchanged. Patch bump to 1.62.43; inspect
both real error paths in an isolated empty registry, run format/lint/test,
review CI, merge and deploy.

- [x] Update both manual-dataset error paths and verify the resulting messages.
- [ ] Run required gates, review CI, merge, deploy and verify PyPI.

Review: both actual error paths for absent `depmap_rna` retain the download
URL, exact expected filename and dataset key, with their original exception
types. Format/lint pass. No parser, registry or download behavior changed.

## PR #517 release-runner restack (1.62.44)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #517 priority queue restack (1.62.44)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #509 specification — class-scoped fixture lifetime

Two long-peptide class-scoped fixtures are instance methods despite not
using their instance. Pytest 9.1 deprecates this lifetime mismatch. Define
them as ordinary module functions with class scope and distinct names,
without changing their constructed indexes. Re-plan after Python 3.9 CI
showed that its classmethod descriptor lacks the metadata pytest expects;
module functions avoid that version-dependent descriptor behavior. Verify
the existing mapping tests with PytestRemovedIn10Warning promoted to an
error, then format/lint/test, review CI, merge and deploy 1.62.44.

- [x] Reproduce the warning as an error and correct fixture binding.
- [ ] Run required gates, review CI, merge, deploy and verify publication.

Review: the existing suite produced 12 setup errors with the deprecation
promoted to an error; all 30 tests now pass under that same warning policy.
Format/lint pass. Scope, construction and test assertions are unchanged.
The change follows pytest's documented classmethod fixture migration.

## PR #518 release-runner restack (1.62.45)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #518 priority queue restack (1.62.45)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #508 specification — explicit boolean mapping

Replace the two deprecated pandas no-silent-downcasting option contexts
with nullable-boolean conversion before filling missing mapped values,
then preserve the public plain-bool dtype. Check object, string and
categorical inputs, missing values, homogeneous categories and empty
frames. Use warnings-as-errors to verify the existing export tests, and
compare both flags over the actual restriction vocabulary. Run required
format/lint/test and CI; review, merge and deploy 1.62.45.

- [x] Reproduce warning/failure boundaries and fix both conversions.
- [x] Verify outputs/dtypes and existing export behavior.
- [ ] Complete full gates, merge, deploy and verify PyPI.

Review: all 15 dtype/missing-value cases fail before the fix with the
deprecated option warning treated as an error. The complete non-integration
export suite now passes under that policy (176 tests). Compared every output
column of the old/new training-default paths over all 1,304 distinct
restriction strings in observation and binding artifacts plus missing data,
for object, string and categorical inputs: zero differences. Plain boolean
dtypes and non-default row indexes are preserved. Format/lint pass.

## PR #519 release-runner restack (1.62.46)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #519 priority queue restack (1.62.46)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #306 specification — finish source-axis consistency

PRs #309/#312 already introduced canonical source_species and documented the
two raw inputs. Residual defect: an explicit `unidentified` source blocks a
valid legacy-species fallback in filters/flags, while pMHC's independent
warning coalesce treats it as resolved. Centralize missing-source handling
in the existing shared coalesce, preserving the original raw columns and
source_organism precedence for two valid inputs. Treat blank, unknown and
unidentified (case/whitespace insensitive) as missing, and keep unknown
source output blank. Use the coalesce for the pMHC warning. Ensure projected
chimeric/engineered/xenograft flags read the legacy fallback too. Verify
MS/binding filtering and narrow projections, categorical/null inputs, and
warning agreement; audit real raw-pair changes. Ship 1.62.46 after gates.

- [x] Reproduce source-sentinel and projection inconsistencies.
- [x] Share coalescing and complete derived-column dependencies.
- [ ] Audit affected corpus inputs, test, review CI, merge and deploy.

Review: seven regression cases fail before the fix. All 255 focused
observation, pMHC and export-species tests now pass; format/lint pass.
Audit covered 4,440,428 observation rows and 892,827 binding rows: 72,329
and 371 respectively change only canonical source_species from the literal
unidentified to blank. Raw fields are preserved, all three biological
flags are unchanged, and unresolved-source warning counts are unchanged.
The tests additionally cover valid legacy fallback, source precedence,
source-only query rows and narrow projections absent from this corpus.

## PR #521 release-runner restack (1.62.47)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #521 priority queue restack (1.62.47)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #452 specification — five ERAP2 experimental lines

Re-read Lorente 2019 (PMID 31530632 / PMC6823859), Experimental
Procedures and Results. Replace the two aggregated arms with parental
polyclonal WT, unedited process-matched WT1/WT2, and ERAP2 knockout
KO1/KO3. Record three independent biological preparations per line and
retain the shared B*40:02 transfectant / ERAP1 Hap8 context. Distinguish
parental untreated and process-matched control roles without claiming
the unedited controls are gene knockouts. Use unique condition IDs and
valid comparator references. Audit all 10,319 IEDB rows: three comments
distinguish WT-only, KO-only and both, but never identify a clone. Preserve
that ambiguity and state the actual clone-resolution limit; do not assign
WT-only rows to parental WT or any KO row to a guessed clone. Check all
other study entries are semantically unchanged. Release 1.62.47 after
source regressions, corpus audit, format/lint/full tests, CI and review.

- [x] Read primary methods and audit the deposited row descriptions.
- [x] Curate all five lines and honest attribution limits.
- [ ] Verify controls, replicates, observed-row impact and required gates.
- [ ] Review, merge, deploy and verify PyPI.

Review: all four source/attribution assertions fail against the old two-arm
curation; 86 focused curation, condition, group and arm-resolution tests
pass after the fix. Format/lint pass. All 10,319 observed rows retain their
raw B*40:02 restriction and blank clone/condition fields; they gain C1R
system attribution and the arm_not_recorded verdict. The scoped YAML
rewrite verifies that every other entry is semantically unchanged.
Primary methods expose a separate cellular-genotype versus selected-ligand
restriction ambiguity, filed as #520; this PR records that limitation
explicitly without silently changing the existing attribution contract.

Release ledger: #501 / PR #505 shipped 1.62.21. #504 / PR #506 shipped
1.62.22 after 1,659 regular and 43 integration tests. Both releases have
verified PyPI wheel and sdist artifacts. Remaining PRs still need their
full local gates, final review, merge and deployment.

## PR #523 release-runner restack (1.62.48)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #523 priority queue restack (1.62.48)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #357 specification — fetchable, usable DepMap expression

Use the pinned primary DepMap 24Q4 release (Figshare 27993248.v1), not
the changing portal landing page. Register the actual gene matrix,
transcript Profile matrix and their model/profile mapping companions.
Provide an explicit opt-in `data fetch depmap` bundle that downloads and
builds line_expression.parquet; normal package installation/build remains
independent of downloading multi-GB expression matrices. Keep each source's
normalization, citation and profile provenance accurate.

Primary-file audit disproves two assumptions in the current parser: the
transcript file is OmicsExpressionTranscriptsTPMLogp1Profile.csv (4.17 GB),
its rows are ProfileID, and its headers are SYMBOL (ENST...), not the
reverse. Stream rows and select registered systems before melting so the
whole release is never expanded in memory. Join profiles through the
release mappings, select source-appropriate profiles without summing
replicates, and reconstruct TPM from log2(TPM+1). Preserve compatibility
with already-registered supported files. Model metadata contains five of
the six listed lines; HEK293 is absent and must not be fabricated. HAP1
has a stranded RNA library, separately relevant to #358. The actual
checksum-verified gene matrix contains 19,193 HAP1 values: library
strandedness must not be used to infer matrix membership. Select the
default RNA profile and check whether its row is actually present.

The resolver must report exact/parent/family RNA only when its source has
rows for that key. Missing optional downloads retain lower-tier fallback.
Cache availability against the built artifact's identity so installing or
rebuilding data becomes visible in the same process. Test explicit source
availability, missing-data fallback, exact official header shapes, profile
selection, bounded parsing and a fake-download-to-peptide-origin end-to-end
path. Audit the real selected rows and source profile IDs. Run all required
checks, review CI, merge and deploy 1.62.48.

- [x] Inspect primary release metadata, README, model/profile maps and CSV prefixes.
- [x] Reproduce fetch/parser/availability failures and implement the complete path.
- [ ] Validate real data and focused/full test gates.
- [ ] Review, merge, deploy and verify PyPI.

Review: four regressions fail before the fix. All 293 expanded export, CLI,
dataset and expression tests pass; 98 expression/fetch tests pass after the
final profile-selection correction. Format/lint pass. Downloaded all five
primary files and verified their sizes and published MD5s. The real build
produces 1,502,002 rows, including 19,193 genes and 227,188 transcripts for
each of six lines (HAP1 included), with one release-default ProfileID per
line. Five requested lines now resolve to actual exact-line RNA; HEK293
retains tissue fallback because it is absent. HAP1 registry wiring belongs
to #358. The audit script completed in 51 seconds; the external timing
wrapper alone returned an error because sandbox policy disallows its
sysctl query. Its full JSON report and built index were verified separately.
Wrong legacy placeholder citations discovered during the audit are filed
as #522 and will be corrected separately from this download/parser change.
Full local tests, final CI, merge and deployment are still required.

## PR #524 release-runner restack (1.62.49)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #524 priority queue restack (1.62.49)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #522 specification — verified RNA-source provenance

Check each stored PMID against the original article and read the intended
papers' methods. Essletzbichler HAP1 is PMID 25373145, SRP044391,
TopHat alignment with mapping-weighted gene counts reported as FPKM;
remove the invented kallisto/TPM claim. Pearson is PMID 27841757;
its own cohort is 18 donor B-LCLs, while JY is an external validation
transcriptome reprocessed with kallisto 0.42.5. Retain only that supported
JY association and remove unsupported HHC coverage. Kaabinejadian
PMID 35154160 profiles HLA ligands, not host RNA; retire the falsely
attributed RNA source and its C1R/721.221 anchor references. These are
empty placeholders, so actual RNA values and fallback resolutions must
remain unchanged. Document source evidence and availability explicitly;
do not convert FPKM to TPM without a complete quantified feature set.
Add source-fact regression assertions and verify all registry links,
packaged data and existing fallback behavior. Run format/lint/full tests,
final CI, review, merge and deploy 1.62.49 in its own PR.

- [x] Re-read primary methods, provenance and data-access statements.
- [x] Correct or retire unsupported placeholder metadata.
- [ ] Verify unchanged values/fallbacks and all required gates.
- [ ] Review, merge, deploy and verify PyPI.

Review: all three primary-source regression checks fail before the change;
101 provenance/expression/fetch tests pass afterward, with format/lint.
Every registered line was resolved against an empty optional-data cache and
against the full 1,502,002-row expression index before/after: all anchors,
row counts and hashes are identical. Source IDs referenced by remaining
anchors all exist. No numerical expression file changed. Full tests and
final CI remain required before merge; deployment remains required after.

## PR #525 release-runner restack (1.62.50)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #525 priority queue restack (1.62.50)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.

# Phosphoantigen restriction filtering (#230)

The issue correctly identifies BTN3A1 rows as unsuitable for peptide-MHC
consumers, but its extracellular "presenter" description is outdated. Yuan et
al., Nature 2023, PMID 37674084, DOI 10.1038/s41586-023-06525-3, reports
intracellular phosphoantigen binding by BTN3A1 and cooperation with BTN2A1;
the CRISPR screen also implicates BTN3A2/BTN3A3 (Figure 1 / Results). These
proteins are not classical peptide-presenting MHC molecules. Restrict the
new allow-list to BTN3A1/2/3 and BTN2A1; do not infer the whole BTN family.

Existing indexes can carry a stale false `is_non_peptide_ligand` flag. Recompute
this derived classification from restriction strings at load time, including
projection and opt-in reads, so upgrading the rule actually affects old indexes.
Keep raw rows available through the existing explicit opt-in.

- [x] Read the primary paper's mechanism, structural results and Figure 1.
- [x] Audit current BTN rows and baseline flag counts.
- [x] Add failing helper and stale-index/projection/opt-in regressions.
- [x] Add the narrow sourced rule and refresh stored derived flags on load.
- [x] Compare the complete restriction vocabulary and validate filtering.
- [ ] Run format.sh, lint.sh, test.sh and final CI; review the complete diff.
- [ ] Bump 1.62.50, open a separate PR, merge and deploy after preceding releases.

Review: seven regression cases fail before the fix. All 504 non-integration
curation/loader/export tests pass; format/lint pass. The complete 5,333,255-row
audit changes only two observation and six binding rows, all HLA-BTN3A1 records
for IPP/HMBPP from PMID 23872678 (DOI 10.1038/ni.2665, original abstract and
Figures 4–6 checked). Existing non-peptide counts remain 304 observations / 218
binding rows; the new totals are 306 / 224. Real old indexes now exclude these
eight rows by default and preserve all eight with opt-in, with accurate flags.
Raw compound names and reported restrictions are retained. CI exposed a stale
integration-test whitelist; it now independently includes the same four specific
BTN genes. Full gates remain pending. RNA host curation #358 remains independent
while checksum-verified inputs are acquired.

# #565 — pan-HLA-II elutions need the cell's whole class-II typing

PMID 33592498's three CIITA-transduced class-II arms listed a DRB1 pair as
their experimental candidates while the elution used HB245, which the paper's
Methods call an anti-pan HLA-II antibody. Every DP, DQ and DRB3/4 ligand in
those elutions therefore matched no candidate of its own sample. The cells'
class-II typing is known (Table 1) and already recorded in `mhc_genotype`, so
the fix is to make the candidates that typing and say so with
`mhc_basis: sample_typing` — the locus gap was created by the curation, not by
the assay. Write DP and DQ as alpha/beta pairs, since that is how both the
paper and the deposited restrictions name them, and the join keys on the
deposited string. Do not widen any sample whose primary source was not read.

- [x] Verify HB245's specificity, the DP/DQ ligand claim and each line's
      class-II typing against PMC8724627 (Methods, Results, Table 1).
- [x] Expand the three arms' candidates to the cell's class-II typing and set
      the basis; extend the study's `hla_alleles` pool to match Table 1.
- [x] Measure the attribution delta against origin/main for this study's rows.
- [x] Curate `elution_condition_ids` so the now-shared class-II alleles reach
      the line that was actually eluted.
- [x] Audit the corpus for the same pan-antibody / subset-of-loci shape.
- [x] Add regression tests, bump the patch version, run format/lint/test.

Review: the paper is explicit — "Anti-pan HLA-I (HB95) and HLA-II (HB245)
antibodies", the highlight "CIITA induces peptide presentation across the
HLA-DP, -DQ, and -DR allotypes", and per-line motif deconvolution naming the
DP and DQ combinations. Table 1 matches the curated `mhc_genotype` exactly,
including the unknown second DPA1/DPB1 slot for HROG02.

Measured over this study's 55,628 observation rows (the join is keyed per
PMID, so the study's own rows reproduce what the full export computes for
them): `sample_match_type` changes on 15,772 rows, all `pmid_class_pool` →
`allele_match`, leaving none in the pool; `sample_mhc` on 39,739;
`sample_attribution` on 26,376; `mhc_basis` on 39,646; `mhc_restriction` on 0.
Expanding the candidates alone would have been worse than the bug for 1,457
rows: HLA-DRB4*01:03 is typed in both HROG02 and RA, and the arm tie-break
scores tokens of length 3 or more, so "RA" never matches and those rows
first-picked HROG02. The deposited elution statements resolve them — the same
per-row discriminator the study's `arm_resolution` verdict was filed against.
Measured on the shipped code: 0 rows contradict their deposited elution
statement, down from 571 on main. Final distribution over the study's 55,628
rows -- match type 55,136 allele_match / 492 pmid_class_pool; attribution
30,803 allele_exact / 17,348 elution_conditions / 6,985 pmid_ambiguous / 492
elution_conditions_excluded. Reaching 0 took a third change the review
caught — the statement map is only consulted for an *ambiguous* allele key, so
an allele typed in one line alone skipped it and handed 492 parental-statement
class-II rows to that line's CIITA arm as `allele_exact`. The exporter now
vetoes an arm its own deposited statement excludes.

Audit: outside this study no sample's candidate list drops a locus of the
assayed class that its own `mhc_genotype` reports. The only other pan-class-II
elution with a locus gap is PMID 27869121's melanoma metastasis arm, whose
`mhc` is the class designation `"HLA class II"` — no typing to narrow rather
than a typing narrowed too far. The mono-allelic C1R / 721.221 / K562 panels
list one molecule under a pan-class-I or pan-class-II pull by design and carry
`mhc_basis: selected_restriction`.

Follow-ups, not fixed here: the paper reports 165/651/83 HLA-II peptides in the
parental lines and IEDB carries 585 class-II rows under the parental-only
statements, but no parental class-II arm is curated, so those rows reach the
CIITA arm or no arm at all — filed as #567. The `ip_antibody` string glosses HB245 as IVA12;
ATCC lists IVA12 as HB-145, and the paper names no clone, so the gloss is
unverified — the pan-HLA-II specificity it stands on is the paper's own words.

# Prioritized donor/specimen qualification — 2026-10-16

Spec: [human-lineage-spec.md](human-lineage-spec.md).

- [x] Inspect published cohort gaps and primary Atlas source documentation.
- [x] File the primary donor-source gap and define the bounded artifact contract.
- [x] Pin source tables through datacache and inspect source/sample semantics.
- [x] Implement the all-tissue donor-aware reader without changing existing scope.
- [ ] Add regressions, publish the frozen donor cohort and independently verify it.
- [ ] Format, lint, full tests, final-head CI, merge, clean-main deploy and public checks.
- [ ] Update #680/#678 and proceed to assay-context applicability in Presto #126.


## Primary donor cohort review

All 967,449 raw source hits independently verify against the native donor and
peptide tables; 17 artifacts pass. The 15/3/3-donor split is disjoint by donor and
reported sample group, with 94,920 shared peptide pair entries explicitly reported.
The first donor summary failed and is preserved under #703. The successful v2
changes only donor summary typing plus recipe/execution metadata; every other
artifact remains byte-identical. Thirty focused tests and formatting/lint pass.
Publication, complete local/CI validation, merge and PyPI verification remain gates.

# T-cell and processing evidence availability audit (#723)

## Specification

Measure what the public IEDB and CEDAR T-cell assay exports actually contain
before connecting new target families to the native training interface. Treat
the published source export as the unit of evidence: pin its bytes, record its
header contract, source and date, and derive counts from distinct assay IDs
rather than duplicated join rows. Separate measured positive, negative and
unknown results; method/readout families; restriction resolution; contributor
studies; and the availability of experimental context and quantitative units.
Do not turn unobserved MS peptides into T-cell negatives or infer a unique
specimen/run from donor genotype. Preserve Atlas's existing donor labels and
leave unresolved specimen/acquisition links explicitly unknown.

This first PR will publish a reproducible availability report and a narrow
source acquisition/audit recipe. The report decides which evidence families can
support a subsequent frozen training export and records source gaps without
claiming measured processing or receptor-specific supervision prematurely.
After source coverage is known, implement the first eligible native functional
export with independent tests for target routing, context, units and censoring.

## Plan

- [x] Inspect official IEDB/CEDAR export endpoints and schemas; record source
      versions, provenance, file hashes and distinct assay identity semantics.
- [x] Audit source-backed T-cell endpoints, outcomes, restriction resolution,
      study coverage and contextual field availability; inspect processing
      evidence separately.
- [x] Build a deterministic, source-pinned audit with readable aggregate reports
      and independent verification, keeping raw assays out of Git.
- [x] Add focused regression tests for source parsing, counting and unknowns;
      document residual Atlas specimen/run uncertainty and the next export gate.
- [ ] Review the diff and version bump; run format, lint and full tests; open a
      PR, require final-head CI, merge and deploy from clean main.

## Review

IEDB has 578,529 unique T-cell assay IDs in 12,357 references. CEDAR has
157,263 IDs, all shared with IEDB, so the union stays 578,529. IEDB reports
363,584 negative and 214,945 positive-tier outcomes. Source method/readout
fields support distinct availability counts for multimer, functional and
other T-cell assays; none contain direct-processing keywords. This is not a
claim that no processing assays exist in other entities or study supplements.
Only 135,751 names have simple two-field HLA syntax; the evidence-code table
shows many such names come from prediction or have undetermined provenance.
The raw exports are excluded from Git; two content-pinned source ZIPs and a
deterministic aggregate report are committed. An independent streaming probe
reproduced both source assay/study/outcome/context counts and the complete
CEDAR-to-IEDB assay-ID overlap. Four focused tests, format and lint pass.
`./test.sh` initially met the #483 memory preflight several times; after free
memory recovered, the unmodified full gate passed: 3,143 tests at 89%
coverage. Final-head CI, merge and clean-main PyPI deploy remain pending.

# Row-level T-cell assay metadata from the pinned audit (#723) — 2026-10-21

## Specification

Make the exact IEDB/CEDAR source snapshot behind the availability report
accessible through a public streaming Hitlist API and a reproducible local
Parquet/JSON export. Verify both source hashes before yielding any rows. Use
IEDB as the one-row-per-assay source of the union, and verify that every CEDAR
ID occurs there; keep the CEDAR membership as a flag, not a second assay row.
Preserve source assay/reference IDs, exact reported qualitative outcome,
method/readout, quantitative value/units/inequality, experimental context,
restriction evidence code and coarse restriction-name resolution. Do not
conflate multimer, functional and other assay families. Keep physical specimen
and receptor identities explicitly unknown. This row-level interface excludes
antigen sequences, antigen IDs/source organisms and allele names; it is an
assay metadata export, not an epitope-specific training target or a change to
`generate_training_table`.

## Plan

- [x] Inspect the source schema, existing audit, source acquisition and public
      API conventions; validate export column names against the pinned files.
- [x] Implement source-pinned streaming reader and deterministic local export
      with source manifest and CEDAR membership deduplication.
- [x] Add focused fixture tests for hash/schema failures, exact outcomes,
      context, unknowns, deduplication, and export receipt.
- [x] Run against the real pinned source snapshots and reconcile row counts
      with the availability audit; document usage and limitations.
- [ ] Run format, lint, full tests; review diff, bump patch version, create PR,
      merge after final-head CI, and deploy from clean main.

## Review

The new local Parquet export has 578,529 rows, including 157,263 CEDAR
memberships. IEDB and CEDAR match on every projected assay field and reference
ID for their shared IDs. The output has 27 columns, occupies 4.0 MB, and has
SHA-256 `5929eb6120b32207df89b58de4245ef6bf9eacbaad32cbb3cd25e2bd2088db62`.
It was built from the two catalog-pinned ZIPs at `/private/tmp/iedb_tcell_full_v3.zip`
and `/private/tmp/cedar_tcell_full_v3.zip`; the local output is
`/private/tmp/hitlist-tcell-metadata-20261020`. Ten focused tests and the
3,152-test standard suite pass on the pre-version-bump implementation. The
export intentionally omits antigen/allele identities and remains separate
from peptide-specific training labels.

# Human-origin peptide functional evidence (#723) — 2026-10-21

## Specification

The pinned IEDB/CEDAR rows contain epitope, assay antigen, outcome and
reported MHC restriction fields. Add a source-pinned local export for the
bounded human-origin subset: unmodified canonical linear peptides, tested as
the same peptide, with a cellular functional readout. Retain exact source
qualitative result, method, readout, context, reference and restriction
evidence. Represent allele assignment conservatively: an exact HLA name is
still a reported restriction, and a narrow evidence-code subset can be
flagged for experimental restriction support without asserting that each row
independently established the restriction. Preserve unknown specimen
and receptor lineage. Exclude multimer/binding, processing, nonhuman and
ambiguous antigen-format rows. Keep the raw export local, not in Git. Do not
change the existing metadata API or imply that all 578,529 assays are
peptide-specific training examples.

## Plan

- [x] Confirm source links and measure eligible source subsets without
      disclosing sequences.
- [x] Implement a deterministic streaming functional export and manifest
      with explicit target and MHC assignment status.
- [x] Add focused tests for source filters, outcome fidelity, restriction
      status, CEDAR deduplication and atomic failure.
- [x] Run the real pinned sources and reconcile counts; document usage and
      unresolved donor/specimen linkage.
- [ ] Review diff; bump version; run format, lint and full tests; open PR,
      merge after final-head CI and deploy from clean main.

## Review

The pinned source export yields 47,593 one-per-assay human-origin peptide
cellular outcomes, including 29,075 CEDAR-overlap IDs. These span 2,334
reference IDs and 18,051 distinct peptide sequences. Exact source outcomes
are 25,416 Negative, 18,675 Positive, 2,630 Positive-Low, 678 Positive-High
and 194 Positive-Intermediate. 2,158 rows have a single named HLA plus one
of the two selected experimental restriction evidence codes. These source
codes may be applied across related assays in a reference, so the status is
not a row-level independent restriction measurement. Specimen IDs remain
unknown. Eight focused tests pass; format and lint pass. The full `./test.sh`
preflight initially refused to start with 0.27 GB free, below its 2.5 GB
single-worker budget. Retry when unrelated memory pressure recedes.
The local Parquet result is 595,072 bytes with SHA-256
`9870d110d8b1df04f8119175338ad2f1ddf7b8c155a28fad34d491037697c501`;
the source ZIPs and raw rows are not committed.

# Standardized human-origin functional training bundle (#723, #729) — 2026-10-21

## Specification

Turn the existing bounded human-origin peptide functional assay export into a
frozen, model-facing bundle while retaining the source-faithful assay rows.
Write `assays.parquet` with every currently eligible source assay,
`training.parquet` with one standardized assay-level qualitative-response
example per unambiguous row, `excluded.parquet` with source IDs and exclusion
reason, and a versioned manifest with pinned source hashes, output hashes and
coverage counts. Keep target meaning tied to the source method/readout;
retain original qualitative tiers and quantitative strings, units and
inequalities. Type subject counts only where source strings are valid whole
numbers. Flag qualitative/subject-count disagreement rather than silently
using a binary label. The standard MHC input is populated only when a single
two-field HLA name has one of the selected experimental restriction codes;
retain the raw reported name and evidence code separately. It remains a
source-supported candidate, not independent row-level proof. Preserve
reference ID as a study grouping key and specimen/receptor identities as
unknown. Do not create a holdout split or claim specimen independence.

The bundle stays within the existing human-origin, direct-peptide cellular
scope. Other source organisms and assay families remain outside this target.
The training examples are assay records, not individual people or MS labels.

## Plan

- [x] Inspect existing training bundle contracts and audit the pinned source
      for label/count and missingness patterns; file the discovered problem.
- [x] Implement deterministic bundle writing and verification, with stable
      schema and complete provenance.
- [x] Add focused tests for routing, exclusions, typed counts, MHC input
      gating, exact source preservation, tamper rejection and atomicity.
- [x] Run the real pinned sources, reconcile every source assay to either the
      training or excluded table, and document the consumer contract.
- [ ] Review diff; bump patch version; run format/lint/full tests; open PR,
      merge after final-head CI, and deploy from clean main.

## Review

The pinned source bundle contains 47,593 evidence assays, 47,421 typed
training rows and 172 excluded rows with a single observed reason:
qualitative result versus responding-subject-count disagreement (#729).
Training rows have 22,134 positive and 25,287 negative assay-level labels;
2,155 retain a candidate single-HLA input under the selected evidence rule.
The 47,421 rows cover 2,333 reference groups. Family coverage is 14,241
activation/proliferation, 4,970 cytotoxicity and 28,210 soluble-factor
release examples. The training Parquet SHA-256 is
`e9be7fd7fc9998549c0465f6b0fb05aaeb50965188b5fd34b5293cde58076d5a`.
The bundle verifier recalculates routing from every source-faithful assay row
and passes on the real pinned source. Focused tests pass; full project gates,
PR and release remain pending.
Default datacache acquisition and explicit local ZIP inputs produced identical
training and evidence Parquet hashes.

# Human-origin peptide multimer evidence (#723) — 2026-10-21

## Specification

Add a second, distinct assay-level training task for qualitative peptide–MHC
multimer binding to tested T cells. Use the same pinned IEDB/CEDAR ZIPs and the
same human-origin, unmodified canonical, directly tested linear-peptide filter
as the functional bundle. Require source method `multimer/tetramer` and readout
`qualitative binding`. Preserve the 8,432 eligible source assays as local
evidence, route the 9 assays whose qualitative result disagrees with reported
positive-subject count to an explicit exclusion table, and expose the other
8,423 through the existing typed training schema with a distinct task name.
Keep MHC assignment conservative, retain original reported restriction and
assay context, and make study grouping and unknown specimen/receptor lineage
visible. This is a cell-binding assay outcome, not a cellular functional
response, peptide–MHC affinity measurement, or receptor-sequence label.
Do not add pathogen-origin rows or combine task labels in one table.

Expose the common human direct-peptide source projection through a documented
public iterator used by both task-specific exports, rather than sharing a
private predicate between modules. Implement a separate versioned local
bundle/API so the existing functional bundle remains byte-for-byte stable.
The new verifier must check pinned source
receipts, file hashes, schemas, source-to-training/exclusion routing, and
coverage counts. Test the real pinned files without committing raw sequences.

## Plan

- [x] Audit source eligibility, readout semantics, count disagreements, and
      MHC-name/evidence coverage without displaying peptide sequences.
- [x] Add multimer evidence and training writers using the shared validation
      and row schema, with an independent task and manifest.
- [x] Add focused tests for family separation, count exclusions, MHC gating,
      source parity, verifier tampering, and atomic failure.
- [x] Run the pinned source ZIPs, reconcile all rows and document the output.
- [ ] Bump version, run format/lint/full tests, open PR, merge after final-head
      CI, and deploy the verified artifacts from clean main.

## Review

The pinned source has 36,268 multimer assays in IEDB; 8,432 meet the bounded
human direct-peptide filter and all report qualitative binding. All 8,432
source rows appear in the evidence table; 8,423 enter the separate training
task (3,129 positive and 5,294 negative) and 9 are excluded for reported
qualitative/subject-count disagreement (#731). The 8,423 training rows cover
660 source reference groups; 1,899 have a candidate single HLA under the
selected restriction-evidence rule. CEDAR duplicates 7,625 of the evidence
IDs and adds no independent rows. The training Parquet SHA-256 is
`d108664e7129136cb83c897f2ac19018062ca2229bbb6f61ebfbf3bd3fb07eda`.
The verifier passes on the real pinned files. Rebuilding the existing
functional bundle after the public-iterator refactor produced byte-identical
evidence, training and excluded Parquet files compared with version 1.82.16.
The public API accepts an assay-family name or collection and the general
bundle writer/verifier select `functional` or `multimer` explicitly. Twelve
focused tests, `./format.sh`, `./lint.sh`, and all 3,167 standard tests pass.

# Hitlist 2.0 evidence catalog — PR 1 of 6 (#734)

Spec: [v2-evidence-system-spec.md](v2-evidence-system-spec.md). Parent: #733.

- [x] Verify current main and its T-cell exports; create the parent and six
      native GitHub sub-issues with sequential PR boundaries.
- [x] Write the architecture and PR 1 specification before implementation.
- [x] Add the immutable, versioned catalog and strict introspection interface.
- [x] Declare current artifact grains, field roles, null meanings and task
      requirement contracts without modifying any assay data.
- [x] Add focused contract tests and no-corpus import verification.
- [x] Document the catalog, pinned baselines and migration sequence.
- [x] Run format, lint and full tests; review diff and bump patch version.
- [x] Open PR, pass final-head CI, merge and deploy from clean main.

CI discovery: every Python unit matrix leg fails when a training bundle has
no contributor PMIDs. Record this independently, fix the empty mapped-column
dtype at its source, assert the serialized empty-list value, then repeat the
full gate before merging PR 1.

- [x] File the empty-contributor bundle bug (#741) and link its fix from PR 1.
- [x] Fix the dtype and verify both empty and legacy bundle cases locally.
- [x] Verify the fix in the final-head CI matrix.

## Review

The catalog declares 14 logical evidence/context types, six task contracts,
five T-cell assay families, source population overlap and generation-scoped
count baselines. Existing source rows and label exporters are untouched.
Six focused catalog tests and 46 assay/T-cell tests pass under a clean,
worktree-local locked Python 3.12 environment. Its one editable install reports
version 1.82.18 consistently. Format and lint pass. The first local
`./test.sh` invocation was refused by its memory guard: 1.37 GiB free against
the required 2.5 GiB for one worker. A later full run passed: 3,177 passed,
one skipped, 89% coverage. The five CI unit legs initially exposed #741 in
empty and legacy bundles. The source fix casts the mapped PMID field to a
string dtype before filling `[]`, and the two affected test paths now pass
locally. Final-head CI passed build, lint, all five Python unit legs,
integration and coverage. PR #740 merged as `95a693a`; the clean-main
release workflow #39789897982 passed both test phases and produced verified
artifacts. PyPI 1.82.18 serves the exact wheel and sdist hashes. The local
`./deploy.sh` run passed unit tests but its integration memory preflight
refused at 9.71 GiB free against the 24 GiB macOS requirement, so the
repository's clean-main release artifact workflow provided the full gate.

# Hitlist 2.0 MHC observation graph — PR 2 of 6 (#735)

Spec: [pr2-mhc-evidence-spec.md](pr2-mhc-evidence-spec.md). Parent: #733.
PR 1 shipped as 1.82.18; this PR targets 1.82.19.

- [x] Pin and verify the public v4 corpus; inspect exact schemas, identities,
      field coverage and contributor relationships before implementation.
- [x] Write the PR 2 specification and identify unresolved design risks.
- [x] Implement one content-bound logical package with typed MHC results,
      measurements and source links; preserve all native fields and contributors.
- [x] Publish a complete native-field disposition map and explicit MHC-role,
      method, endpoint and null-status policies.
- [x] Reconcile every v4 result and contributor link by partition/source and
      provide a machine-readable report of differences and exclusions.
- [x] Add offline fixtures and corpus verification, bump 1.82.19, run format,
      lint and full tests; review count and labeling parity.
- [x] Open PR, pass final-head CI, merge and deploy from clean main.

## Review

The hash-pinned v4 package reports 5,327,796 distinct result IDs across
4,486,991 MS, 814,268 binding and 26,537 other assay rows, with no ID
collisions. All 9,381,101 contributor links resolve, and every result has a
link. These structural checks preceded the v2 package build.

The first full v4 conversion exposed native `pmid: int64` against the typed
text reference field. Preserve the integer in the original Parquet and
convert explicitly for the typed representation; keep the regression in the
offline fixture before rerunning the corpus conversion.

The 1.82.19 conversion now passes on the full pinned v4 source. The package
manifest in `tasks/mhc-v2-evidence-results/` hashes 12 files and reports
5,327,796 typed result/measurement pairs, 9,381,101 preserved contributor
links, 23 source snapshots, zero orphan links and zero missing source links.
An independent rowwise check of all 5,327,796 source rows found zero changed
IDs, peptide/modification fields, MHC text, source IDs, assay class/outcome,
reported value/unit/inequality or quality flags. All three partition/source
counts equal v4. Fifteen focused catalog/package tests pass. The local editable
install reports 1.82.19 consistently. `./format.sh` and `./lint.sh` pass;
`./test.sh` passes 3,188 tests with one skip and 88% coverage. The final
source-link audit requires registered datasets, positive source row locators,
source record IDs and raw payloads on every contributor link; it produced
the identical manifest hash as the previous build. Final-head
CI, merge and PyPI deployment were pending at that review point.

PR #742 passed build, lint, all five Python unit legs, integration and
coverage, then merged as `e7c74d9`. The clean-main release workflow
#39806405579 passed the full corpus gate and bound both distributions to that
commit and the pinned v4 manifest. Their hashes and license metadata passed
verification; PyPI 1.82.19 serves the exact wheel and sdist hashes. Local
`./deploy.sh` passed lint and all 3,189 unit tests but its macOS integration
memory guard refused at 8.64 GiB free against 24 GiB, then again at 7.33 GiB;
the hosted clean-main workflow supplied the full integration gate.

# Hitlist 2.0 biological context graph — PR 3 of 6 (#736)

Spec: [pr3-context-evidence-spec.md](pr3-context-evidence-spec.md).
Parent: #733. PR 2 shipped as 1.82.19; this PR targets 1.82.20.

- [x] Audit the hash-pinned cohort and curation source revisions before code.
- [x] Specify the physical package, relation grains, null semantics and
      conservation rules.
- [x] Fix historical MHC package verification across catalog evolution (#743).
- [x] Materialize curated arms, concurrent factors, MHC roles, scoped protocol
      claims, assay context/lineage links and the curation amendment ledger.
- [x] Preserve all native cohort and curation source assets, map every field,
      and reconcile all observation, source and arm links.
- [x] Add focused fixtures and a full pinned-corpus conversion report; bump
      1.82.20, format, lint, and review the diff.
- [x] Pass the full test gate locally or on final-head CI.
- [x] Open PR, pass final-head CI, merge and deploy from clean main.

## Review

Preimplementation audit: all 4,071,463 cohort assay IDs are in the 5,327,796
v4 MHC results, and all 71 shared native fields match cell for cell by ID.
The cohort adds 90 fields at that grain. All 2,042,689 assigned condition
rows map to 489 current curated arms without an orphan; 318 curated arms have
no linked cohort row. The cohort used a historical PMID registry. Current
curation keeps all 807 arm keys and adds 35 source-backed field changes on 27
arms; eight changed arms reach 23,064 cohort rows. Both curation revisions and
the original 161-column cohort table must survive the migration. No donor,
specimen or acquisition identity is established by these exports.

The audited full package manifest has SHA-256
`0d16dd1b9fd0708087d9f04df5c4c57c0f3166f48db513d41816c9bda2685e6b`.
It retains all 5,327,796 MHC results, 4,071,463 bounded cohort contexts and
7,646,178 cohort contributor links. All observation, contributor and historical
arm references resolve. Its 807 arms and 216 studies per curation revision,
2,339 factor claims, 2,074 MHC claims, 342 protocol claims, 168 lineage claims,
88 control links and 35 amendments match the independent source audit.
Independent comparison of all 94 typed context/identity projection fields
found zero mismatches across the 4,071,463 source rows. Focused tests pass
(22), and `./format.sh` and `./lint.sh` pass. The wheel/sdist build contains
the historical curation YAML, new module and license. The local `./test.sh`
attempt was blocked before collection by the existing #483 macOS memory
preflight (0.74 GiB available; 2.5 GiB required); final-head CI and a retry
remain necessary.

PR #744 passed the complete final-head CI gate, including all five Python
unit legs and corpus integration, and merged as `95dc872`. Clean-main release
workflow #39816816221 passed its full gate on that exact commit. The tested
wheel and sdist passed provenance, license and metadata checks; PyPI 1.82.20
serves the identical SHA-256 hashes.

# Hitlist 2.0 supporting modalities — PR 4 of 7 (#737)

Spec: [pr4-supporting-evidence-spec.md](pr4-supporting-evidence-spec.md).
Depends on PR 3 (#744) merging and publishing 1.82.20. The new external
quantitative-source acceptance cases are assigned to dependent PR 5 (#746).

- [x] Audit pinned v4 bulk, expression and mapping source schemas, grains,
      endpoint population and source-specific quantity scales.
- [x] Specify source preservation, typed identities, protocol scopes, material
      relation precision, per-family required fields and conservation checks.
- [x] Implement the versioned supporting evidence package and catalog entries
      without losing the parent package or native source bytes.
- [x] Add focused regressions and independent full-corpus parity/coverage report.
- [x] Bump version, run format/lint/full tests, review diff and final-head CI.
- [x] Open PR after PR 3 ships; merge and deploy from clean main before PR 5.

## Review

Pinned source audit: `bulk_proteomics.parquet` has 2,269,288 rows, split into
2,047,003 Bekker-Jensen peptide detections without intensity, 156,211
Bekker-Jensen protein abundance rows with log2 intensity, and 66,074 CCLE
protein rows with a distinct normalized log2 abundance. The table spans 19
observed source/grain/enzyme/fraction/enrichment combinations. The legacy
Trypsin/P display label contradicts the reviewed MaxQuant search rule (#745),
so typed search semantics and raw source text must remain separate.
`line_expression.parquet` has 42,909 gene rows in two profiles (GM12878 and
HAP1), and no populated transcript ID. The anchor registry lists 21 lines,
but names and fallback tiers do not create measured rows or paired specimens.
`peptide_mappings.parquet` has 6,205,013 occurrence rows across 1,145,721
mapped peptides. The three native MHC partitions have 1,506,973 nonempty
peptide strings, of which 1,506,664 remain eligible after removing flagged
non-peptide-ligand rows. Thus 360,943 eligible peptides have no retained
mapping; all remain assay evidence. These counts are source/sequence counts,
not independent subjects or training admission.

The final 1.82.21 package manifest has SHA-256
`70f1cdb2124aea18f1ec4051dfecb58c73c34324dd06cb8d0b225b2b6a9fdb9c`.
It hashes the complete context predecessor, 11 pinned supporting source assets,
and all typed relations. The source and typed row counts reconcile for all
2,269,288 bulk, 42,909 gene-expression and 6,205,013 mapping rows. A second
streaming check found zero copied-field mismatches across 19 bulk fields,
11 expression fields and all 12 mapping fields. A separate measurement replay
checked all bulk endpoints/representations and both expression representations
on every source row. Focused tests pass (16), as do `./format.sh` and
`./lint.sh`. An isolated editable installation reports only this checkout at
1.82.21. The wheel includes the new module, pinned source receipt and license;
large externalized bulk CSVs remain ingestion assets. The local `./test.sh`
attempt stopped before collection at the existing #483 memory preflight (0.25
GiB free versus 2.5 GiB required for one worker); the final-head CI full test
gate remains required before merge.

The final-head PR #747 CI passed build, lint, all five Python unit legs,
integration and coverage on the exact merged code. PR #747 merged as
`7cd7186741478af6c2c2c46c45b70f6707627aec`; the 1.82.21 clean-main
release workflow #39828013399 passed on that exact merge. Its tested wheel
and sdist passed provenance, license and metadata checks. PyPI 1.82.21 serves
the identical SHA-256s: wheel
`3ab0984b47f27126644b0cf765cf778435065cc9134f68708b64e98098816a92`
and sdist
`0edc97a1c791da3398ad6898c3304c74d842468255a12a18de4dfce944a626e6`.

# Hitlist 2.0 external quantitative MS — PR 5 of 7 (#746)

Spec: [pr5-external-quantitative-source-spec.md](pr5-external-quantitative-source-spec.md).
Depends on PR 4 (#747) merged and PyPI 1.82.21 published. Develop the next
source generation on merged main; do not merge until publication completes.

- [x] Audit the pinned 17 original source assets, six workbooks, community SDRF,
      table/panel grains, 32 contrasts and independent quantity reconciliation.
- [x] Specify source-cell and typed-measurement identities, material/partner
      precision, derivation inputs, protocol scopes and conservation gates.
- [x] Implement the generic external-source generation and PXD057588 adapter,
      preserving the parent generation and all original source bytes.
- [x] Add offline ambiguity tests, all-cell/quantity replay and a pinned result
      report; keep related unqualified sources explicit as deferred contracts.
- [ ] Bump 1.82.22, run format/lint/full tests, review diff and final-head CI.
- [ ] Open PR after 1.82.21 ships, merge and deploy from clean main before PR 6.

## Review

Initial source audit: the 17 pinned files total 11,532,567 bytes; all staged
originals match the published SHA-256 and size receipt. Six workbooks contain
35,772 primary sequence rows, 309,126 positive intensity cells, 200,724
reported log2 fold-change cells under 32 contrasts, and 62,911 sequence-block
records across all primary and additional panels. The community SDRF has 72
rows × 45 positional columns and covers six of 50 raw files. The separate
Presto scalar replay verified every fold change within 1e-8 log2, but no
training admission or exact cross-assay specimen pairing follows from that
arithmetic audit. Source methods distinguish MHC elution from global digest
quantities and table-specific normalization; the same nine TMT channel labels
have eight different sample-name mappings between Tables 9 and 11.

The final 87-file package manifest in
[external-v2-evidence-results](external-v2-evidence-results/README.md) has
SHA-256 `daff7930e193c6dbde1fe84c065b024187232147fa006d9348e297766e95e69c`.
An independent streaming replay checked all 851,830 source cells, 62,911 panel
records, 35,772 primary rows, 309,126 intensity results, 200,724 fold-change
representations, 802,896 derivation inputs and 3,240 SDRF cells with zero
projection mismatches. Maximum fold-change error was 2.55e-14 log2. Embedded
source assets have independent inodes from ingestion inputs; issue #748 tracks
older predecessor hardlink behavior. Sixteen focused tests and format/lint pass.
The isolated editable install is 1.82.22 and `uv pip check` passes. The full
local `./test.sh` gate passes: 3,214 passed, one skipped, 88% coverage in
5m17s. Final-head CI and clean-main publication remain pending.

# Hitlist 2.0 T-cell evidence — PR 6 of 7 (#738)

Spec: [pr6-tcell-evidence-spec.md](pr6-tcell-evidence-spec.md).
This branch is rebased onto PR 5's exact merged main commit. Open PR 6 after
1.82.22 publishes.

- [x] Verify both pinned T-cell ZIP hashes and independently replay the existing
      availability and bounded training counts.
- [x] Audit the full IEDB/CEDAR field intersection and the CEDAR-only mutation
      field; file the preservation gap as #750.
- [x] Implement one verified successor generation with native ZIP snapshots,
      typed assay/measurement/context/contributor relations and field disposition.
- [x] Materialize the 578,529-row availability and 55,844-row bounded training
      eligibility funnel with explicit exclusions and source-linked claims.
- [x] Independently replay all source rows, family counts, CEDAR overlap,
      candidate statuses and source fields; add focused offline fixtures.
- [x] Bump 1.82.23 and run format/lint/full local tests on the rebased branch.
- [ ] Run final-head CI; review, merge, deploy clean main and verify PyPI exact
      hashes before PR 7.

## Review

The pinned ZIPs match their 2026-10-20 receipts. The existing metadata export
replays 578,529 IEDB assays and 157,263 shared CEDAR IDs. The bounded exports
contain 47,593 direct functional and 8,432 multimer source rows; 172 and nine
respectively are currently excluded for qualitative/subject-count disagreement,
leaving 47,421 and 8,423 training examples. Of these, 4,054 carry conservative
candidate single-HLA assignments, without specimen IDs. A separate first-fail
funnel accounts for every IEDB assay and recovers all 56,025 direct candidates.
All 157,263 shared IDs match on the 158 common source columns. CEDAR has one
additional `Epitope | Mutation` column, populated on 30,639 overlap rows,
including 17 current training IDs. Preserve this contributor claim (#750).

The final 99-file package manifest in
[tcell-v2-evidence-results](tcell-v2-evidence-results/README.md) has SHA-256
`4c3275ca6da9a9e1de084fcc1a8d1243d4e67b1e9c3ed4d6307cbcce5c3738e3`.
It embeds both exact original ZIPs and the verified external predecessor.
Independent source replay compares all 578,529 assay rows with the prior
metadata export, all 55,844 examples with the prior functional/multimer
bundles, 735,792 source-contributor locators, 30,639 mutation claims, and
every reported outcome, antigen and eligibility key; mismatches are zero.
Two offline fixtures cover source disagreement, pooled antigen exclusion,
unknown HLA and distinct functional/multimer labels. The source mutation
field remains linked to CEDAR without changing the 17 affected examples.
The nested-predecessor verifier regression, `./format.sh` and `./lint.sh`
pass; 29 focused catalog, T-cell, external and availability tests pass. The
isolated editable install is 1.82.23 and `uv pip check` passes. The first
`./test.sh` attempt stopped at the existing #483 preflight (0.19 GiB free
versus 2.5 GiB required). The retry passes all 3,218 tests with one skip and
87% coverage in 1m58s. After rebasing onto merged PR 5, the source tree
is byte-identical to the audited head; format/lint and `./test.sh` pass again
with 3,218 tests, one skip and 87% coverage in 2m42s. Final-head hosted CI
remains pending.

# Hitlist 2.0 supporting numeric measurements — PR 7 of 8 (#751)

Spec: [pr7-supporting-numeric-spec.md](pr7-supporting-numeric-spec.md).
Branch from PR 6's exact merged commit `ca0c729`; merge only after its
1.82.23 wheel and sdist are published and hash-verified.

- [x] Audit every native bulk-protein and expression numeric column by source,
      grain and missingness; preserve peptide detection as nonquantitative.
- [x] Append one immutable, independently snapshotted successor generation
      with typed per-representation measurements and source-row lineage.
- [x] Register the new measurement relations and field-level source
      dispositions; verify parent hashes, schemas, counts and source identities.
- [x] Replay every populated native numeric cell independently, add small
      offline fixtures for null, zero, multiple scales and source alignment.
- [x] Bump 1.82.24 and pass format/lint/focused offline tests.
- [x] Review final diff and pass final-head hosted unit, integration, coverage,
      build and lint CI. Local full tests stopped at the unchanged #483 memory
      preflight; hosted full-corpus integration passed.
- [x] Merge and deploy from clean main; verify exact PyPI wheel/sdist hashes
      before PR 8.

## Initial review

Pinned v4 native bulk has 2,047,003 Bekker-Jensen peptide-detection rows and
222,285 protein rows: 156,211 Bekker-Jensen and 66,074 CCLE. No peptide row
has a populated source numeric quantity. All 156,211 Bekker-Jensen protein
rows have log2 intensity and abundance percentile; all 66,074 CCLE protein
rows have normalized log2 abundance and percentile. That yields 444,570
source numeric protein cells. Each of 42,909 gene-expression rows has TPM and
log2 TPM, yielding 85,818 expression cells, including 3,771 measured zeros
on each scale. The source percentile ranges approximately 0–1. The typed
result files preserve all source-row identities but currently omit these
numeric cells, as filed in #751. An independent snapshot for the successor
will also address the final-generation inode risk described in #748.


The 103-file successor package has manifest SHA-256
`560770376121758b3c18679c3bf4db78bc89f9e95cc72785a7314fe35167cb1e`
and exact predecessor manifest SHA-256
`4c3275ca6da9a9e1de084fcc1a8d1243d4e67b1e9c3ed4d6307cbcce5c3738e3`.
The writer checked all 2,312,197 native source-row alignments and emitted
444,570 distinct bulk-protein plus 85,818 expression numeric representations.
An independent streaming replay found zero mismatches for every source value,
source field, result ID and native locator. All 3,771 HAP1 TPM zeros survive.
The complete package verifier passed, and none of the predecessor files share
inodes with the new snapshot. Thirteen focused numeric/catalog/T-cell checks
and format/lint pass. The first full local `./test.sh` attempt stopped at
its existing #483 preflight (0.09 GiB free versus 2.5 GiB required); no
worker or memory threshold was bypassed. The isolated editable install
reports 1.82.24 consistently; `uv pip check` passes for all 48 packages.
The 1.82.24 wheel/sdist build, license metadata and `twine check` pass;
large source assets are absent from the wheel. Final-head hosted unit,
integration, coverage, build and lint jobs passed. PR 7 merged as `b124ecb`
with the same tree as its validated head. Its clean-main local deploy preflight
stopped at 0.13 GiB free versus the 2.5 GiB floor; the hosted clean-main
release workflow passed the full suite and build on `b124ecb`. Two independent
Actions artifact downloads matched the release manifest, and PyPI 1.82.24
serves exact wheel SHA-256
`32bd91de1d40dc2c671dfc4df3ab9d83f863bb0cc48e9d75d5785e843e5a5e69`
and sdist SHA-256
`37cd0197ee66a04c91246f60ee1f266dbf6ebc6967bb49ce2e286931c3cea3f6`.
The release verifier's direct artifact ZIP endpoint remains affected by #755;
the successful workflow, sole retained artifact and byte-identical double
download supplied the independent publication check.

PR 6 merged in #753 and PyPI 1.82.23 serves exact wheel SHA-256
`b7c265994d43bed66ab614d6ef5edbc4cbf56eb127d0523b3472bdb08e0627e9`
and sdist SHA-256
`1bae77fc2a8268a1f375067762ebb5e6d2da980b51b110d7fe649f46783fff6b`.
The clean-main workflow passed; the release verifier's direct artifact-ZIP
retrieval hung (#755), so two independent Actions downloads, the run's exact
commit/artifact metadata and matching manifest/distribution hashes supplied
the publication provenance check.

# MAPP source generation (#752)

Plan: tasks/mapp-source-plan.md.
- [ ] Preserve pinned original source quantities, context, confidence and mapping alternatives.
- [ ] Integrate canonical external package/catalog and verify real conservation.
- [ ] Run format/lint/full tests, publish reviewed release and consumer evidence.

MAPP first shared generation completed in43.29s:10,673peptides,149,959evidence rows,362,882quantity representations,540,034derivation edges. Independent original TSV/workbook-XML replay passes in9.46s. Preserved first launcher path-type and source-NaN failures;74,573PEP NaN values now explicit.16focused tests and format/lint pass. Full normal test suite running. Reference occurrence relations/query/recipe and reviewed deployment remain pending; do not close752. Concurrent numericPR756 reserves1.82.24; source PR uses1.82.25 and will rebase before merge. First generation records development1.82.24 and immutable original bytes; do not relabel it as a final release.

MAPP verification: full normal suite passes3,224tests/10warnings in356.83s;16focused final-catalog tests and format/lint pass. Independent111-file inventory and original TSV/XML replay pass. Filed757for missing parallel-proteome run metadata/independent precursor estimates. Keep752open for reference integration and full recipe handoff. Worktree creation triggered repository metadata-refresh hook for existing editable1.82.2; pip check reports no broken requirements. Subsequent git writes disable hooks to avoid shared-environment churn.

## MAPP reference integration

Plan: tasks/mapp-reference-plan.md. Current main numericPR756 merged into branch; version1.82.25 retained.
- [x] Materialize seven reference relations with exact alternative occurrences and negative candidates.
- [x] Verify all full sequences, aliases, occurrences and original source claims independently.
- [x] Verify byte-identical original eleven MAPP typed tables and PXD057588 source artifacts.
- [ ] Publish reviewed release and downstream endpoint handoff.

Actual schema2 package:120files,62,984proteins,133,414accessionclaims,37,721occurrences,64,538sourceclaims,26,992resolutions,21,346coverage rows. Producer54.80s, independentBioPython/Aho-Corasick/graph replay20.22s, peakRSS0.987GiB. Initial test guard refused0.77GiB; normal retry admitted one worker at3.89GiB.21focused tests and format/lint pass. Broader endpoint/recipe remains739; precursor pairing757. No fit or concentration claim.

Reference software closure: normal full suite passes3,236tests/10warnings in320.81s with one worker and intact memoryguard. Format/lint and21focused tests pass; independent complete source/reference replay passes. Indexed-corpus integration CI remains separate. No source code changed after validation.

## MAPP biological recipe (stacked on758)

Plan: tasks/mapp-biological-recipe-plan.md. This is a scoped shared recipe foundation for739, not the whole unified major release.
- [ ] Registered recipe, source verification, complete dispositions and contributors.
- [ ] Explicit parent/occurrence policy, connected-component holdouts and quantities.
- [ ] Fixtures, real independent admission replay, format/lint/full tests.
- [ ] Publish and hand off immutable recipe output to the registered Presto experiment.

MAPP recipe verification: both shared exports independently replayed; normal full suite passes 3,243 tests / 10 warnings in 327.51 s, format/lint and strict docs pass. Nine local biological fits and independent checkpoint/prediction replay complete; uniform remains the held-out reference. New data request #759; precursor issue #757 remains open. Draft recipe PR and complete unified query release remain separate.

## 2026-10-23 MAPP run repeatability and representation

Previous goal turn: progress. Plan: mapp-run-repeatability (Presto) / tasks/mapp-run-recipe-plan.md (Hitlist).
- [x] Shared complete-run recipe with all original observations/status and fixed source partitions.
- [x] Independent original-source replay and local software checks.
- [x] Registered run-repeatability diagnostics, then evidence-led fixed representation comparison.
- [ ] Publish complete artifacts, report and canonical registry; verify remote readback.

Review: shared source and every run/feature/metric/checkpoint replay pass; all 186 archived fitting files and nine recipe files match original bytes. Baseline repeats (r=0.914); treatment-change repeatability is weak (r=0.172). Thirty regularized fits complete in 25.81s; all allocated weights train, but none beats uniform. Hitlist b94289d: 20 focused tests and strict docs pass; normal local full-suite memory preflight refused twice. Hosted Python 3.10–3.14 unit CI and integration/coverage/lint/docs all pass (3.11: 3,247 passed/2 optional skips; integration: 96 passed). Calibration data request #762 filed and read back exactly; precursor #757 and perturbation #759 remain open. Complete artifact and report publication is the remaining closure step; full concentration goal remains active.

## 2026-10-23 MAPP integration with Hitlist 2.0

Previous goal turn: progress (30 fits/report/data issue published). Plan: tasks/mapp-v2-integration-spec.md. Preserve original experiment inputs while integrating source758/recipe760 with concurrent main release761.
- [x] Immutable successor of current reference package; canonical query bindings.
- [x] Canonical registered distribution recipes without a parallel public API.
- [x] Independent real-source/recipe conservation and focused software checks.
- [ ] Final integrated-head full CI, publication, review and release.

Review:124 predecessor files have independent inodes and unchanged bytes;33 MAPP files, all18 query relations/1,829,008 rows and all original three recipe targets/splits independently conserved.55 focused tests, format/lint and strict docs pass. Normal full-suite memory preflight refused before workers (0.67GiB available/2.5required); no full-suite pass claimed. Integration execution is1442c03; final import-order correction is recorded separately. Source snapshot dependency763 merged;2.0.2 is reserved for this integration. Concentration/response/precursor gaps remain752/762/757/759.

# Hitlist 2.0 evidence store and recipes — PR 8 of 8 (#739)

Spec: [pr8-evidence-store-spec.md](pr8-evidence-store-spec.md). Base is merged
PR 7 commit `b124ecb`; publish its 1.82.24 distributions before PR 8 merges.
The October 22 mouse-reference acceptance amendment in #739 extends this PR
before the 2.0 merge; the first implementation commit is `361c3c7`.

- [x] Audit all materialized files and declared fields; add explicit physical
      selectors, aliases, readiness and missing-field disposition.
- [x] Add MHC and external numeric measurement evidence entries; validate
      source identities, endpoints, grains and population counts.
- [x] Implement verified `EvidenceStore` catalog/query and named relationships
      with strict predicates, no silent label or context promotion.
- [x] Implement declarative recipe materialization with eligibility decisions,
      exact source lineage, uncertainty, coverage and hash-bound outputs.
- [x] Add one CLI surface, examples, migration notes, and 2.0.0 version bump.
- [x] Retire the public v1 `hitlist export training` command so the evidence
      recipe is the sole CLI training path; retain audited source export code.
- [x] Append a byte-preserving, hash-verified mouse-reference successor
      generation with typed source mappings, exact occurrences, parents,
      parent-support sets, species roles and result-context lookup.
- [x] Extend the catalog, query bindings, recipe and audit to expose those
      relations without multiplying labels or promoting project claims.
- [ ] Replay all original and new pinned baselines and adversarial fixtures;
      run format/lint/full tests, final-head CI, review, merge and exact-artifact
      PyPI deployment.

## Review

The preflight on PR 7's 103-file package found 42 original materialized
evidence types, all with existing physical files. Of the original catalog's
declared field placements, 232 were not same-named physical columns. Four
additional typed relations are now registered: MHC measurements, external
quantity measurements, primary external peptide rows and source panels. The
pre-reference catalog had 46 materialized and four contract-only types. Its 46 physical
bindings validate against real schemas; all typed columns have a native query
name. Across declared fields, 367 are direct physical, 12 source constants,
73 related through named links, 68 unavailable at the claimed grain and 43
contract-only. The real-package audit matches all 25 pinned baselines,
including the two-revision arm/study counts, and gives 64 source/endpoint
groups and 33 T-cell eligibility groups. Ten adversarial fixtures pass.
A bounded real-package MHC recipe reconciles all 4,486,991 source rows into
20 accepted candidates and 4,486,971 explicitly limit-omitted rows; its
output verifier checks all files, decisions, contributor and lineage rows.
Those 20 MHC candidates retain 40 separate IEDB/CEDAR source-contributor
records. A real T-cell functional example for shared assay ID 450 emits one
accepted peptide label and two contributor rows, one per provider, with no
extra assay. No provider link is unresolved in either bounded replay.
Recipe YAML uses the repository's duplicate-key guard, and recipes include
available direct source identifiers even when not explicitly selected. The
pre-reference audit replay retained its original audit hash. Before the
amendment, the required full unit gate passed twice: 3,222 passed, one skipped,
86% coverage; format/lint and the 2.0.0 wheel/sdist checks also passed. The
1.x README is preserved in `docs/legacy-1x.md` with a clear
historical warning; the project landing page now documents the 2.0 interface.
The October 22 shared-reference amendment arrived after these checks. Its
ten output hashes, independent replay receipt and 18,126,470-byte FASTA hash
match the pinned source. The new 124-file package independently clones every
predecessor file, adds seven typed relations and retains distinct inodes. The
53 materialized types validate against actual schemas; all 32 baselines match.
The reference audit preserves 35,772 source rows, 30,587 sequences, 52,277
exact positions, 25,758 reference parents, 26,449 support sets, five scoped
mouse role claims and one context row per 309,126 original result. The source
parent statuses are 35,747 exact, 24 missing IDs and one resolved ID without
the peptide. A bounded numeric recipe preserves 20 measurement candidates and
their 20 source peptide links, explicitly filters 200,724 fold-change rows,
and marks the other 309,106 intensities as limit-omitted. Table 10 measured
macrophage versus tumor-partner roles and unresolved specimens remain explicit.
The final audit JSON records 431 physical, 21 constant, 105 related, 68
unavailable and 43 contract-only field placements; its SHA-256 is
`02dd24ac68ebbd68067437cfd906e04f2f059221c588ce0765326475676aa731`.
The old `hitlist export training` CLI parser and dispatch are removed; the
source export implementation remains available for historical audits. Its
old helper tests and the new CLI rejection test pass. The first final full
suite attempt under the filesystem sandbox passed 3,209 tests but failed 18
environment-bound checks: 12 localhost HTTP fixtures could not bind, and six
`test.sh` tests inherited the explicit unknown-memory fallback. The rerun
with normal macOS probing and localhost access passed all 3,227 tests at 85%
coverage before the CLI retirement. Final format/lint/full tests on that
small follow-up diff, CI and deployment remain.

# PR 9: immutable evidence-generation snapshots (#748)

Spec: [pr9-evidence-snapshot-spec.md](pr9-evidence-snapshot-spec.md). Base is
the merged and PyPI-published 2.0.0 release (`0dfa170`). This is a 2.0.1
patch release for package-byte safety; it does not change assay semantics.

- [x] Reproduce source and predecessor hardlink mutation failures.
- [x] Add one clone-first, independent-inode snapshot primitive and migrate
      every evidence-package writer to it.
- [x] Prove fallback, target-exists and source/predecessor mutation behavior;
      compare original source hashes and package reconciliation.
- [x] Bump 2.0.1, run format/lint/full tests, review, open PR, pass final-head
      CI, merge and deploy from clean main; verify exact public artifacts.

## Review

Both new regressions fail against 2.0.0 before the fix: appending to a v4
`observations.parquet` input changes the published MHC package's native
file, and appending to the MHC predecessor changes the published context
package's native binding file. This proves the shared-inode mechanism rather
than relying only on a hypothetical failure. All seven generation writers now
use one clone-first snapshot helper for imported input and predecessor files.
Five focused snapshot and
mutation tests pass after the change; the full set of 39 focused package
tests passes. Each package writer still verifies its manifest and
reconciliation, so file hashes and typed row counts remain part of the
acceptance check. The full repository gate passes: 3,227 tests, 85% coverage,
with one worker and an explicit 0.75 GB memory budget after the normal 2.5 GB
preflight refused at 1.03 GB free. Its peak resident set was 0.88 GiB.
Format and lint pass. An isolated editable install reports the expected 2.0.1
metadata; both 2.0.1 distribution artifacts build, pass `twine check`, and
the wheel contains the new snapshot helper. PR #763 passed the complete
Python matrix, integration and coverage checks, merged as `dcd33aa`, and the
clean-main release run 39908123658 passed 3,226 unit tests plus 96 integration
tests. PyPI 2.0.1 serves the tested wheel and sdist SHA-256 values
`3f1ec6bb0898a198e922001d72e4b015ad8682b322246f5409c1a12abdf1956b`
and `3192df2afb1aaac972d43dba9523bab576525602c2843b1b3498a56619295722`.

# PR 10: bounded release-artifact verifier (#755)

Spec: [pr10-release-verifier-spec.md](pr10-release-verifier-spec.md). Base is
the merged and PyPI-published 2.0.1 release (`dcd33aa`). This is an
independent 2.0.2 patch release for the publication gate.

- [x] Reproduce the unbounded direct artifact-ZIP request in a failing test.
- [x] Add bounded exact-run artifact retrieval and retain the trusted receipt
      plus downloaded-distribution hash checks.
- [x] Cover timeout, failure, corruption, mismatch and artifact replacement.
- [ ] Bump 2.0.2; run format/lint/full tests, build, review, PR CI, merge and
      deploy from clean main; verify exact public artifacts.

## Review

The new bounded-download regression fails against 2.0.1 at the exact direct
`gh api .../zip` call, before any implementation change. The replacement
verifier limits metadata calls to 30 seconds and the exact-run, exact-name
artifact download to 120 seconds. It rechecks the artifact ID, rejects
missing, extra or symlinked files, compares the trusted and local receipts,
and checks downloaded distribution hashes against the receipt. All 26
focused release tests pass, including simulated timeout, partial cleanup,
failed download, corrupt receipt, modified wheel and artifact replacement.
The new verifier also passed against the real 2.0.1 release artifact from
successful run 39908123658 and clean main `dcd33aa`. Format and lint pass;
66 release/deploy/corpus tests pass. An isolated editable install reports
the expected 2.0.2 metadata, and the wheel and sdist build, pass `twine check`
and retain exact license metadata. Local `TEST_SH_MAX=1 ./test.sh` stopped at
its memory preflight (0.37 GB available versus the conservative 2.5 GB
one-worker budget), before pytest started. The hosted PR matrix and corpus
integration will supply the full test gate. PR CI, merge and release remain.

# PR 12: first-class multimer source adjudication (#766)

Spec: [pr12-multimer-adjudication-spec.md](pr12-multimer-adjudication-spec.md).
Base is merged and PyPI-published 2.0.3 (`d27415e`); the successor release is
2.0.4. PR 11 (#768) passed its complete clean-main release gate in run
39922883083 and PyPI serves its exact wheel and sdist hashes. The historical
SIGTERM cause remains open under #754.

- [x] Review all nine IEDB assays against the pinned typed package and their
      primary papers/supplements; preserve unresolved source meaning.
- [x] Specify one assay-linked review grain, source anchors, schema-8
      inheritance, query links and a zero-change label reconciliation.
- [x] Add a controlled, versioned review input and strict source/eligibility
      validation, with adversarial tests first.
- [x] Append independently snapshotted schema-8 package, register its typed
      review relation in EvidenceStore and preserve schema-6/7 access.
- [x] Build and verify the real successor package; reconcile all predecessor
      hashes, 578,529 assays, 8,432 direct multimer candidates and 55,844
      training examples, including nine retained exclusions.
- [ ] Bump 2.0.4, run format/lint/full tests and focused real-package checks;
      review the diff, open PR, pass final-head CI, merge, run clean-main
      release and verify exact PyPI hashes.

## Review

The review ledger contains nine IEDB source-assay claims at
`source_assay_review` grain, keyed to three primary publications/supplements.
Its SHA-256-pinned YAML is copied into the package and materialized as one
queryable Parquet relation. The verifier rejects changed raw qualitative or
subject-count cells, antigen/MHC assertions, missing IEDB contributors,
accepted training rows and changed eligibility. Schema-8 review queries join
to the original assay, outcome, antigen and eligibility without multiplying
the source assay. Schema 7 remains readable and marks the review relation
`not_in_generation`. A source-registry inheritance failure found by the real
package replay was fixed and covered by a focused regression. The Hombrink
source URL was corrected from the landing page to the publisher's direct PDF
endpoint after a fresh download matched its pinned SHA-256; the final package
and distributions were rebuilt from those reviewed bytes.

The real schema-8 successor has 126 hashed files: all 123 predecessor files
match by SHA-256 and occupy distinct inodes, plus the predecessor manifest,
review YAML and nine-row Parquet relation. The original T-cell assay,
outcome, eligibility, training and contributor files are byte-identical.
`EvidenceStore.open` verifies both generations. The full new audit has 54
materialized types, four contract-only types and all 33 baselines matched.
The before/after source counts are 578,529 assays, 735,792 contributor rows,
8,432 direct-peptide multimer candidates, 8,423 accepted multimer examples,
55,844 total training examples and nine unchanged exclusions.

Format/lint pass; 22 focused catalog/store/adjudication tests pass. An
isolated editable install reports 2.0.4 and both distributions build with the
review input and module, pass Twine checks and retain exact license metadata.
The required local `./test.sh` stopped before collection at its conservative
memory preflight: 0.75 GiB free versus the 2.5 GiB one-worker minimum. Full
hosted PR CI and clean-main release gates remain pending.

## 2026-10-23 shared MS-calibration source

Plan: tasks/ms-calibration-source-spec.md. Previous goal turn: progress (published original-source audit and exact issue762 handoff). Build on verified6e275fe in this isolated branch; preserve earlier experiments and shared environment.
- [x] Source adapter, nominal panel/amounts, distinct reporter forms and context.
- [x] Canonical typed-query exposure with source-scoped availability and safe joins.
- [ ] Original-source and predecessor conservation, software checks, report and publication.

The availability/concentration goal remains active. This source is an observation-response anchor candidate; no intracellular concentration or processing-effect label is supplied.

Pre-materialization review:55 focused tests, format/lint and strict docs pass. Whole-Methods review corrected injection-volume metadata to reported2µL reference protocol and addedData13/14/15 for site instruments, stage-specific gradients and64-window settings.13 typed relations and8 pinned source assets (seven workbooks plus article XML). Source-only implementation; real-source/parser conservation and all previous package/recipe checks remain next.

Source integration attempt1 stopped at a real cross-workbook reporter-label difference. Plan amended before code: preserve32 distinct labels, restrict raw/normalized comparisons to Data4, keep source conflict and failed execution. No output generation published.

Real-source closure: corrected04f7369 and independent verifier pass, preserving157 previous files and allthree MAPP recipes.32 distinct aggregation/form labels retained.56 focused tests, format/lint and strict docs pass; normal full runner refuses0.65/2.5GiB before pytest. Hitlist762 updated with protocol context and aggregation-equivalence acceptance. Review/release, hosted CI and bounded canonical raw-S4 admission remain.

## 2026-10-23 bounded MS nominal-response recipe

Plan: tasks/ms-calibration-response-spec.md; Presto raw-s4-nominal-response plan.
- [x] Compile source-bound conditional raw-S4 examples with complete exclusions/provenance.
- [x] Expose and verify through canonical catalog/materialize/CLI APIs; preserve old recipes.
- [x] Verify original-source admission independently, then run the registered local response diagnostic.

No free-concentration, processing supervision or SWATH-to-MAPP transfer claim. Full availability goal remains active.

Raw-S4 review: independent XML admission checks all3069 raw slots;1974 positives become224 summaries/22 observed peptides. All295 local fits and validation/test outputs independently replay (maxprediction delta7.93e-11). Context RMSE0.8084 versus mean1.1796 and shared-penalty sequence model1.2762; no detector-transfer/concentration claim. Full report: https://github.com/pirl-unc/presto/blob/9f8e4e0/experiments/2026-10-23_0359_codex_raw-s4-response/README.md.61 focused tests/format/lint/docs pass; normal full local runner refuses0.30/2.5GiB before pytest. Previous ad60a8f full hosted CI passes; current96b9232 CI is separate. Review/merge/release remain;2.0.3 is unreleased.


## 2026-10-23 evidence successor compatibility (#771)

Plan: [evidence-successor-compat-spec.md](evidence-successor-compat-spec.md).
- [x] Identify and file schema-8 collision with the newer adjudication release.
- [x] Merge current main without losing either evidence family.
- [x] Write schema-9 successors with verified reference/adjudication predecessors.
- [x] Regress family dispatch, source registry, snapshot conservation and tamper rejection.
- [x] Register and independently verify a real combined source/recipe generation.
- [x] Run required format/lint/tests/docs and publish draft with exact readback.

Review: executed code0219827 preserves all126 predecessor files,9 review claims and56 appended-source files in a183-file combined generation. Independent direct-Parquet/hash replay checks32relations/1,853,840mixed-grain rows and allfour unchanged recipe outputs; both historical schema8 families remain readable. New package schema9/catalog12, version2.0.5 candidate. Normal localtest.sh passes3,317tests/10warnings in321.51s with unchanged memoryguard, one worker; format/lint/70focused checks/strict docs pass. First sandbox memory probe and initial standalone-recipe baseline mismatch are retained in the report. Source code was not changed after validation. No new biological fit. Review/hostedCI/merge/release remain separate gates.

Report: https://github.com/pirl-unc/presto/blob/82feaf2/experiments/2026-10-23_0551_codex_adjudication-source-conservation/README.md. Combined manifest42317a5efb7d642be5b8265131e6a6f7963e469adda946b780dac4d6aebb05b0. Fix is on existing draft767; controls source admission759 remains next. Broader availability goal stays active.

## 2026-10-23: reconcile MAPP PR #760 with released 2.0.4

Plan: [mapp-main-204-merge-spec.md](mapp-main-204-merge-spec.md).

- [x] Merge current main and resolve the schema-8 family collision.
- [x] Preserve MAPP relations/recipes and source-reviewed adjudications with
      accurate generation-specific bindings and registry.
- [x] Verify both historical package families and real MAPP-only successors.
- [x] Reconcile all three recipe targets/contributors with the prior generation.
- [x] Run format/lint/test/docs, review CI, merge #760 and release 2.0.5 with
      verified PyPI artifact hashes.
- [x] Start #767 integration on the released MAPP version as the next serial PR.

### Review

Schema-9/catalog-11 successors built from both verified parent kinds. Independent
SHA-256 checks match all 126 reviewed-parent and 123 reference-parent files,
with separate inodes. The review-based package exposes 72 materialized relations,
9 review claims and 10,673 MAPP peptides, with all 33 baselines matched. The
reference-based package correctly omits review availability. All 18 MAPP typed
relations match 1,829,008 prior logical rows; all three recipe data inventories
match previous targets, contributors and admissions byte for byte. Focused
catalog/review/successor/recipe tests pass (45), as do format, lint and strict
docs. Full local `./test.sh` passes: 3,290 passed, one skipped (85% total
coverage). Final-head hosted CI passes. PR #760 merged as `cfbb43f`; the
clean-main release run 39941299674 passed and PyPI 2.0.5 wheel/sdist hashes
match the CI receipt (`08460a9a0b1d0f5b56fdf3f4ffb335bbe8f7a90e01812e09ab68b2c81b8bc110`,
`d63eeb63f34f1917d428a592f1c2270a88868d60e2cf94380364c04e2c7b6c0c`).
See [mapp-205-results/README.md](mapp-205-results/README.md).

## 2026-10-23 release #767 after MAPP 2.0.5

Plan: [ms-calibration-206-release-spec.md](ms-calibration-206-release-spec.md).

- [x] Confirm published 2.0.5 and live #767 head, then merge main.
- [x] Bump 2.0.6; correct predecessor terminology and calibration guide.
- [x] Replay historical and real combined packages and all four recipes.
- [x] Run format, lint, full tests, strict docs and distribution checks.
- [ ] Review final-head CI, merge and publish exact 2.0.6 artifacts.

## Review

The 2.0.6 candidate preserves all 126 adjudication predecessor files, all 183
files in the prior combined source generation, and all four registered recipe
semantic payloads. The released 2.0.5 MAPP package remains readable. The full
local test suite passes (3,319 tests, 84% coverage). The wheel and sdist pass
Twine, license, and calibration file-inventory checks. Hosted release gates
remain before merge and PyPI publication; see
[ms-calibration-206-results/README.md](ms-calibration-206-results/README.md).

## 2026-10-23 controlled MAPP source and signal (#759)

Previous goal turn: progress. Plan: tasks/mapp-controlled-source-spec.md.
- [ ] Shared two-panel adapter, source-linked metadata, aliases and quantities.
- [ ] Reference alternatives, canonical query and run-level recipe.
- [ ] Registered real-source generation and independent verification.
- [ ] Normal format/lint/tests/docs; source/recipe conservation.
- [ ] Training-only signal check and frozen biological comparison contract.
- [ ] Publish source/experiment evidence and update759; retain broader goal.

Review: pending. Captured signal does not directly measure free concentration.

## Shared complete-case controlled response

Plan: tasks/mapp-controlled-contrast-spec.md; frozen scientific contract is the
Presto controlled-mapp-exact-sampled-fit specification.
- [x] Canonical contrast materialize/verify implementation, base recipe preserved.
- [ ] Fixture/source replay and required format/lint/test/docs.
- [ ] Publish real 59/14/14 membership and contrast evidence with biological fit.

Review: implementation in progress; no new biological fitting yet.

## 2026-10-23 release #772 after calibration 2.0.6

Plan: [mapp-controlled-207-release-spec.md](mapp-controlled-207-release-spec.md).

- [x] Confirm PR #767 merged and merge its exact main commit.
- [x] Confirm exact Hitlist 2.0.6 published before readying PR #772.
- [x] Bump 2.0.7 and retain historical catalog-11/12 readers.
- [x] Rebuild real controlled successor and replay all old and new recipes.
- [x] Run local formatting, lint, strict docs and distribution gates.
- [x] Pass the full local test suite under its unchanged memory guard.
- [x] Review final-head CI, merge and publish exact 2.0.7 artifacts.

## Review

The 2.0.7 real package preserves all 183 prior combined files and nine review
claims, and its source and both new recipes pass independent replay. All four
previous recipe semantic payloads remain byte-identical. Distribution gates
and 3,330 local tests pass; the hosted release gate and exact PyPI artifact
hash verification passed; see
[mapp-controlled-207-results/README.md](mapp-controlled-207-results/README.md).


## Shared quantitative digestion source #774 (parent #770)

Plan: tasks/digestion-source-spec.md.
- [x] Source-faithful typed adapter and canonical catalog bindings.
- [x] Fixtures, real source reconciliation and prior-source conservation.
- [x] Complete independent original-source row/cell/slot checks on the 2.0.8
      reviewed package.
- [x] Run final-head CI and review the source-only #773 PR.
- [x] Merge #773; leave all-substrate arithmetic and the grouped kinetic
      recipe to #775.
- [x] Publish the exact 2.0.8 release artifacts to PyPI.

Review: ongoing; full concentration goal remains active.

PR #773 merged as `89e45ed` after all exact-head checks passed. The clean-main
2.0.8 release run `39979099617` passed; both published PyPI distributions
match its SHA-256 and size receipts exactly. Source admission #774 is shipped.

Digestion review: draft PR773 at977adc2, all18 assets/11 typed relations/250 package files verified; all218 prior source payloads and33 baselines conserved. Independent original-source decoding/typed replay passes.37 focused and3,341 normal guarded tests pass; format/lint/strict docs pass. Published Presto1503d67 source and synthetic kinetic reports. All-substrate quantitative arithmetic, training recipe and biological fitting remain pending, as do final-head review/CI/merge/deployment. No concentration claim.

Review update: the 2.0.8 review head adds a 391-member typed inventory, named
workbook cells, explicit 152 blank-run Skyline dispositions and source-unit/
representation labels. A 251-file real package at manifest SHA-256
`ad62414d511b74a6aa1e4da457fdff68d9f4527874d1a2e385c65e5b98df41e8`
preserves all218 prior files exactly at separate inodes. The independent
Hitlist-free round-trip verifies the five typed source tables, all source
members/workbook rows/constructs, original MS1 and raw values, run/time links
and 280,664 final slots; details in
[digestion-source-208-results/README.md](digestion-source-208-results/README.md).
Format/lint/37 focused tests/strict docs and distribution gates pass. The
current local full test runner refused at0.68GiB free versus2.5GiB required;
hosted final-head test and integration checks are still required.

## Contiguous digestion time-course recipe

Plan: tasks/digestion-contiguous-recipe.md. Based on merged PR773 source review.
- [x] Implement pure source-linked curve compilation and frozen component grouping.
- [x] Register canonical materialize/verify/discovery with fixtures.
- [x] Freeze and run real source admission, independently verify targets and groups.
- [ ] Preserve exclusions, complete full software gates and ship the candidate.

Review: the 251-file reviewed source package remains byte-identical after
materialization. All 17 original condition arithmetic checks pass; every one
of 50,372 curve dispositions and 30,566 admitted curves passes an independent
Hitlist-free replay. The median train/validation/test repeated-label temporal
correlations are 0.951/0.776/0.960 on the selected panel. The output is MS1
peak area in source arbitrary units, not molar concentration. See
[digestion-recipe-209-results/README.md](digestion-recipe-209-results/README.md).
No new biological model fit.

PR #776 is open as a draft. Local format/lint, 3,351 full tests,
strict docs, distribution gates and both independent real-source verifiers pass.
Final-head hosted CI, review, merge and PyPI publication are still required.

Release update: PR #776 merged as `3a7844f`; clean-main run `39985180560`
passed and exact wheel/sdist hashes were verified on PyPI as Hitlist 2.0.9.
Issue #775 is closed. The broader source-selection, RF1, MS-response and
biological-fit questions remain in #770.

## Binding assay protocol context (#721)

Plan: [binding-assay-context-210.md](binding-assay-context-210.md).

- [x] Inspect pinned v4/cohort schemas and source-study coverage; pin one
      primary article and its literal method claims.
- [x] Implement a portable reviewed-evidence successor with assay-grain
      review links and study-scope protocol claims.
- [x] Audit all corrected-cohort binding/IC50 rows and independently verify
      real package conservation, source anchors and query behavior.
- [ ] Run all required local and hosted gates, review, merge and publish 2.0.10.

Review: 25,896 corrected-cohort binding assays contain 8,084 literal IC50
results over 198 populated PMIDs plus 543 PMID-unknown rows; only 560 IC50
rows have a populated assay comment. PMID 28481943 contributes 910 cohort
rows plus 28 same-study influenza controls in full v4. Its published method
reports ranges and alternatives that remain at their reported scope. The
256-file real successor and independent replay conserve all 251 predecessor
files and 814,268 original binding result IDs. All 21 protocol claims match
source excerpts. Format/lint/36 focused tests pass. The default local full
test runner refused at 2.18GiB free versus its 2.5GiB preflight requirement;
the hosted gate is still required. No binding labels or donor/replicate counts
changed. See [binding-assay-context-210-results/README.md](binding-assay-context-210-results/README.md).

Release update: PR #777 merged as `fc25dba`; clean-main run `39994432709`
passed, and exact wheel/sdist hashes were verified on PyPI as Hitlist 2.0.10.
Issue #721 remains open for the other binding studies.

## Gomez-Zepeda original result and run lineage (#778 / #779)

Plan: [gomez-source-lineage-211.md](gomez-source-lineage-211.md).

- [x] Pin both original supplemental archives and reconcile all five peptide unions.
- [x] Implement immutable schema-11 successor with all original S3/S5 rows and
      source-run peptide memberships at explicit grains.
- [x] Register typed evidence and query bindings, with source-specific fields.
- [x] Independently replay source rows, membership and links against the real
      source archives and verify predecessor conservation.
- [x] Run format/lint/test and hosted gates; review, merge and deploy 2.0.11.

Review: the independent replay checks all 594,916 original CSV rows, 590,615
source-run peptide memberships, 68,552 exact derived links and all 256
unchanged predecessor files. All 51,491 out-of-length rows remain with
explicit null links. The 262-file schema-11 package has manifest SHA-256
`ade958582dd2eb580f4e02187cd7445621e72fc07ec294a97624a09ba24a2659`.
JY and Raji each contain large WT/S overlap, so #780 will repair arm labels
after #779 preserves source rows. The first full local test pass found one
catalog-version assertion still pinned to 15; it is updated to 16 and the
full required gate is rerunning.

Release update: PR #781 merged as `41efb7d`; clean-main run `40004210180`
passed. PyPI Hitlist 2.0.11 exactly matches the tested wheel SHA-256
`e817336cc14ed82081b6f8c9c8a75f3aa6cf8b93a05b9814b23b5c12d1fcec23`
and sdist SHA-256
`316eae7adc504ef98bcd131df37042268c84e77372b067f85bebd85f7eaa9498`.

## Gomez-Zepeda source-backed arm projection (#780)

Plan: [gomez-arm-projection-212.md](gomez-arm-projection-212.md).

- [x] Audit original conditions, source-arm cardinality and current wrong labels.
- [x] Implement the schema-12 source membership, arm-support and unique
      projection relations without changing predecessor files.
- [x] Publish the nine-arm source vocabulary and Gomez-only authoritative
      assay-context projection; predecessor context and all assay/training
      counts remain unchanged.
- [x] Independently replay all source-to-arm decisions and before/after labels
      on the pinned schema-11 candidate (the full copied successor is deferred
      locally because the 5.7GB predecessor plus copy exceeds available disk).
- [ ] Run required local/hosted gates, review, merge and publish 2.0.12.

Review: implementation is complete locally. Frozen anchors: 33,156 one-arm,
10,840 two-arm and 24,556 three-arm derived peptides; 723 cross-material
plasma errors; 355 current JY-WT assignments with no WT source membership.
The source replay produces 590,615 membership-arm rows and nine explicit arms;
the source-specific context projection has 68,552 rows (33,156 unique and
35,396 multi-arm). Local format/lint pass; the repository test runner refused
to start at 0.57GB free versus its 2.5GB preflight requirement, so hosted
checks are required before merge.

Release update: PR #783 merged as `d29e17f`; clean-main workflow `40052322873`
passed the complete release gate. Hitlist 2.0.12 is on PyPI with wheel SHA-256
`2fc18bd279aca523d1b8e75e9b651fdc8cfdd56368a4dd6b953dfdcff44ac8e6` and sdist
SHA-256 `a606e63b1c3f91ad2d5cea5bcaec8dc45c8f8f3345861f8bf9570a7c22227511`.

## Gomez-Zepeda prediction panels (#782)

Plan: [gomez-prediction-panel-213.md](gomez-prediction-panel-213.md).

- [x] Preserve MOESM8/S5 and MOESM10/S7 prediction rows at source grain.
- [x] Add explicit predicted-vs-unknown restriction projection without
      rewriting native assay labels or dropping alternatives.
- [x] Independently replay all 394,952 prediction rows and 68,552 joins.
- [ ] Run required gates, review, merge and publish 2.0.13.

Review: source archives and row counts are pinned in the implementation plan.
The real schema-11 native inventory replay linked 322,513 prediction rows and
resolved exactly 50,761 named rows as `predicted`, leaving 17,791 `unknown`.
PR #784 merged as `3d96612`; clean-main release workflow `40056724704` passed
and Hitlist 2.0.13 was published to PyPI. Wheel SHA-256 is
`fb2f08af0ec5d74c421ef246ebe42d1b01c3de5f9f2cf4215908623e891775e6`; sdist
SHA-256 is `a4b8b7fcdd77de27ebcb4520be9eb562aa1da8d50b6f06d89b70036b00fd8f47`.

## Binding assay context availability (#786)

Plan: [binding-context-availability-786.md](binding-context-availability-786.md).

- [x] Add schema-14/catalog-19 audit relation over every retained binding row.
- [x] Preserve native values and represent source-scope/unknown status without
      broadcasting study claims or converting IC50/KD.
- [x] Run required gates, review, merge and publish 2.0.14.

Review: the real corrected-cohort replay produces 25,896 unique rows, with
8,084 literal IC50 and 3,232 KD (~IC50) rows; 910 rows are in the reviewed
PMID 28481943 study scope and 24,986 remain unreviewed. Focused tests pass
(47). PR #787 merged as `6245d98`; clean-main release workflow `40062604133`
passed the complete gate. Hitlist 2.0.14 was published to PyPI with wheel
SHA-256 `1b6f060e58fff5249d305b56d87ee55118b11c0d1ce6c6b2dd77d17708667124`
and sdist SHA-256
`42eb07754831ad4a7fea4da87a92bb95dc1fbc7be6610d9e8a0f9cb0d9aeb74e`.

## Homogeneous native peptide measurements (#789 / #788)

Plan: [tmab-native-measurements-789.md](tmab-native-measurements-789.md).

- [x] Pin and audit Dasgupta 2014 and Gelman 2013 native-peptide workbooks.
- [x] Materialize common measurement, protocol, source-row and lineage
      relations without dropping unidentified, censored or reused entries.
- [x] Register the package in the catalog/query/recipe contract and run the
      required gates, review, merge and publish 2.0.15.

Review: PR #790 merged as `f4d3a6d`; clean-main release workflow `40066109622`
passed the complete gate. Hitlist 2.0.15 was published to PyPI with wheel
SHA-256 `0b29abe8c3d4c4addc61d5b656eac2085c9fa842a45a8dd44c3a6b79fb9d3840`
and sdist SHA-256
`b3304a026264f9ccc4bed8c5166714e78385f37f92b0b22c22a2e2863ef22ab2`.

## TMAB reuse, identity and protocol provenance corrections (#792 / #791)

Plan: [tmab-provenance-corrections-792.md](tmab-provenance-corrections-792.md).

- [x] Make cross-paper reuse candidates source-independent while preserving
      source-specific rows and unresolved specimen independence.
- [x] Separate tentative/modification identity from numeric exactness and
      correct acquisition and exposure/wash provenance.
- [x] Correct Gelman heatmap sequence projection and run the local required
      gates.
- [ ] Open the PR, pass hosted gates, merge and publish 2.0.16.

Review: implementation is complete locally. The rebuilt package preserves
10,564 source rows, 61,908 measurements, 33 protocol contexts and all prior
censoring/value-status counts. It records 971 conservative possible
cross-paper candidate groups (1,942 source rows) without asserting specimen
independence; tentative/unresolved identities are deferred. Local format,
lint and full tests pass (3,389 tests).

## TMAB semantic corrections (#791 / #792)

- [x] Implement tasks/tmab-semantic-corrections-791.md on the isolated correction branch.
- [x] Verify focused regressions, canonical query fields and the full original workbooks.
- [ ] Finish full repository tests, review and publish the correction PR/release.
- [ ] Qualify the canonical native-response recipe separately before fitting.

Review: 43 focused tests pass; the final regression file gives 14 failures and
one pass against pinned f4d3a6d. Format and lint pass. Full ./test.sh completed successfully (3,403 tests).
The real-source replay preserves 10,564 rows and 61,908 measurement slots and
recovers all 830 independently identified candidate reuse groups (1,660 rows).
There are 7,933 exact and 4,662 censored treated/control source candidates;
these are not independent training counts. The first replay's 1,242 whitespace
comparison failures are retained and explained by a separate successful closure:
existing projected labels trim whitespace while raw rows remain literal. No
production data or parser rule changed to resolve that comparator error.
Evidence: Presto experiments/2026-10-24_0156_codex_tmab-semantic-replay.


## Reconciled native response successor (#791 / #793)

- [x] Merge concurrent main7649ffe while preserving its raw provenance additions.
- [x] Retain strict sequence/peptidoform parsing, full compound names, handling contexts and per-slot possible reuse.
- [x] Add v3/schema17/catalog22 and the registered exact/bounded response recipe, version2.0.17.
- [x] Pass format/lint,54focused and3,414full local tests plus independent full-source/reference/target/split/weight replay.
- [ ] Push the resolved merge, pass new-head hosted gates, merge and release2.0.17.

Review: see tmab-upstream-reconciliation.md and Presto0259 admission. PR794 merged independently and its2.0.16release-build passed; preserve issue792closure and do not treat prior PR793 checks as new-head evidence. No APC concentration qualification is claimed.

## Standalone T-cell and processing evidence package (#795 / #723)

Plan: [tcell-processing-package-795.md](tcell-processing-package-795.md).

- [x] Audit current T-cell and controlled-processing package/registry paths
      and freeze source/count/hash anchors.
- [x] Define the schema-18/catalog-23 common package contract while keeping
      functional, multimer, receptor, processing and MAPP families separate.
- [x] Register typed family projections, availability/no-direct-processing
      results and replay tests.
- [x] Run hosted gates, merge and deploy 2.0.18.

Review: the standalone replay preserves 55,844 T-cell examples, 578,529
assay rows, 192,114 MAPP quantity results and 362,882 MAPP measurements.
Aliases query functional response, pMHC multimer binding, processing
measurement and MAPP products without merging their scientific families.
The branch is rebased on the strict TMAB response successor (schema 17 /
catalog 22).

Release update: PR #796 merged as `fda1eff79d7a77f195e6a5f3a4c5abbee5cfd7c4`;
clean-main release workflow `40079964064` passed. Hitlist 2.0.18 is on PyPI
with wheel SHA-256
`eece32b864313a5bddd45113a48717d7e86da37ea4d55fb5c3f5be487c7ea065` and sdist
SHA-256 `8177479c633b5e870a89a4bc682d53b2c8e208e8f7cf5fb1c0cd3366ab36fafb`.

## Standalone proteasome processing package (#797 / #770)

Plan: [digestion-processing-package-797.md](digestion-processing-package-797.md).

- [x] Freeze the schema-9/10 source appendage, relation schemas and replay anchors.
- [x] Define schema-19/catalog-24 family and availability ledgers without
      relabeling MS response as molar production or APC processing.
- [x] Copy all source assets and typed relations into a standalone envelope;
      expose aliases and preserve run/construct/context/conflict joins.
- [x] Run hosted gates, merge and deploy 2.0.19.

Review: the real package replay verifies schema 19/catalog 24 and preserves
388 runs, 5,792,969 MS1 rows, 545,664 raw quantities, 53,413 final vectors,
280,664 final values and 2,284 conflicts. EvidenceStore exposes 14 bindings;
aliases return those source counts and five family/five availability rows.
Hosted build/lint/unit/integration/coverage checks passed. PR #799 merged as
`e8faa7fd437d9f8b47d67a4c7836ad3df073f948`; clean-main release workflow
`40084695481` passed and Hitlist 2.0.19 is on PyPI with wheel SHA-256
`56c5dd7fdcbc474aeac18bdee614b32f4d64fa677add4225bd2d2a29136a787b` and sdist
SHA-256 `4a7f167b7e10807b51a4184905e916cb790f15030e2df8aa4d9842406b759f3d`.

## PXD009360 precursor source contract (#801 / #800)

Plan: [precursor-source-contract-801.md](precursor-source-contract-801.md).

- [x] Create parent #800 and child PR issues #801, #802 and #803.
- [x] Freeze hash-pinned PXD009360 source assets, archive members and
      unresolved run/pairing requirements.
- [x] Implement schema-20/catalog-25 source-contract package and local replay (PR #804; merged at `6416616a9b496a4706d9bd161242a247ace103d2`).
- [x] Verify clean-main release run `40089253494`, upload 2.0.20 to PyPI, and
      record wheel `bdebbe850890f92cfed5b9925ce5f5f03c38cb6958e574519d02a1f7889ddcaa`
      and sdist `aabab8b9a0dc30853ca13d52247940d110d05ca5dec5a3a1185d138ab257ede7`.
- [x] Materialize complete precursor quantities in child PR #802 (schema-22/catalog-27 package; PR #808 merged at `897c98cbd514b03ea3156733f48e1e2d5354a534`).
- [x] Verify clean-main release run `40094966489`, upload 2.0.21 to PyPI, and
      record wheel `e21b3019c6c9798174ded0f4be36e66eed1c621f3b78ceb14b02f26515a75f58`
      and sdist `a981791ef83ff85c93aaa45a368307f1e8a9784281f32e4411d24d37ec45650a`.
- [x] Qualify the precursor recipe and pairing scope in child PR #803 (PR #810
      merged as `ba4774217e76ebbfd3a11e99b7deeb08e03583b1`). The real replay
      retains 2,423,504 quantity rows, emits four source-relation qualifications
      and three unresolved pairings, and admits zero training rows.
- [x] Verify clean-main release run `40103757775`, upload 2.0.24 to PyPI, and
      record wheel `eaf5f6f9f407dcec32ab8d5c37119d710d4a679950fcc3b74abc55b7123e5110`
      and sdist `4cff9737acb1e0921867a0dfb95a3488658ef686fee8914ca0657b890e878400`.

# Quantitative HAP1 admission — issue 726

Previous goal turn: progress (source arithmetic verified and issue updated).
Plan: [hap1-quantitative-726.md](hap1-quantitative-726.md).
- [x] Revalidate current main and existing issue; pin actual source semantics.
- [x] Implement immutable typed source/context/quantity/reference package.
- [x] Connect the package to EvidenceStore and test source-faithful query behavior.
- [x] Build on original assets and independently verify all admitted quantities.
- [ ] Run format/lint/tests, bump version, publish and track merge/release.
- [ ] Register the downstream biological comparison after source admission.

Review in progress: format/lint and 52 focused package/catalog/query tests pass.
The full test script first could not access its memory probe in the sandbox;
with access restored, the real probe found 0.22 GiB free against its 2.5 GiB
minimum and refused to launch. No worker-capacity override was used. Hosted/full
suite and merge/release remain pending.

Original-source build v1 correctly stopped on repeated SDRF column names before
quantity projection. Positional metadata preservation and its regression test fix
that issue. Build v2 completes source replay in 379.92 seconds, peak 1.163 GiB:
2,391,692 supplement cells, 558,216 deposited protein cells, 607,928 DIA rows,
40,964 FASTA entries and 60,447 occurrence/unmapped records. There are 83 declared
sample groups because DDA library generation includes additional WT/TAP1/TAP2
biological labels beyond the 36 quantitative elution and 36 proteomics groups.
Independent original-cell verification passes all 2,391,692 supplement cells,
558,216 protein cells and 607,928 DIA rows, with source/sample/run links and
59,726 emitted exact occurrences checked (721 source peptide rows have no match).
The checker took 96.90s and peaked at 0.987 GiB. It checks each emitted match,
not independently the exhaustiveness of the reference search. The actual package
query exposed the existing all-null optional-join bug now tracked as #805. Three
regressions fail before the fix; 55 focused tests pass afterward. The original
package now supports nullable context queries without losing unknown rows.
No biological model fit yet; full-suite validation and release remain pending.

## HAP1 response recipe and biological baselines

Previous status-only goal turn: no progress. Revalidated shared source CI40088549411 succeeds at2f4bbe0, including integration and Python3.10–3.14 units.
- [x] Freeze response/precursor/split and local baseline contracts before fitting.
- [x] Implement registered Hitlist recipe, source-conserving dispositions and meaningful tests.
- [x] Materialize and independently verify real HAP1 response dataset.
- [x] Run local complete-precursor controls and sequence baselines, training-only grouped CV.
- [x] Independently verify, publish report/artifacts and specify physical follow-up from evidence.

HAP1 response review: registered recipe materialization succeeds against the original
schema20 source (manifest a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309).
All 2,391,692 supplement cells retain dispositions; 220,271 canonical positive KO
responses and 129,728 complete-precursor responses across all partitions. Independent
consumer verification reproduces every response, precursor group and dependency split.
Original source bytes/NA values are unchanged. Source replay+recipe424.19s/1.802GiB;
independent typed-source closure114.83s/1.109GiB. Presto development fit finds10.0%
inner-CV RMSE improvement from simple sequence features; this is not concentration
validation or an untouched test. No outer validation/test model score was produced.

Format/lint pass.43 focused tests pass. Full guarded ./test.sh runs with3workers
and passes3,439 tests/394warnings in642.44s (wrapper658.47s). Initial sandbox memory
probe failed; the host-probed launch used7.95GiB free and retained the guard. A first
focused command named a nonexistent singular test file and collected0tests; corrected
plural filename was verified from rg before the successful43-test run.

[ ] Upstream merge/release remains pending: main concurrently added the distinct
PXD009360 precursor_context family at schema20/catalog25/version2.0.20 (PR804).
Publish this recipe as a stacked draft until source PR806's schema/version integration
is resolved; do not claim this work is released.

## Integrate HAP1 and concurrent precursor context

Spec: tasks/hap1-main-integration-spec.md. Keep both explicit package families;
their pinned schema20 envelopes remain distinguishable by family and revision.
- [x] Inspect current main6416616 and source/recipe exact-head CI (all source/recipe checks pass).
- [x] Merge main, retain both catalogs/dispatch paths and advance live catalog/version.
- [x] Verify namespace regression, old HAP1 package and required test suite.
- [ ] Integrate the stacked response PR, publish/release through the upstream workflow.

Integration review: both explicit schema20 families coexist without relabeling
pinned source generations. Live catalog26 and package version2.0.21 retain all
upstream precursor-context support and HAP1/null-join behavior. Format/lint pass.
The focused integration suite passes45 tests after updating a stale current-catalog
version assertion (initial44pass/1fail preserved in /private/tmp/hitlist-hap1-merged-focused.log).
Required full guarded tests and original-package reopen remain pending until the
concurrent local biological fits finish; hosted checks will validate this new head.

Stacked integration: retained both documentation/task sections and both package
families; recipe consumer advances to2.0.22. Original pinned source/recipe hashes
remain unchanged. New-head format/lint/focused and full tests pending.

Full integration validation found five additional stale live-catalog assertions:
Python3.14 CI40093682648 and local guarded ./test.sh both fail only those assertions
(local3437pass/5fail,168.19s). Update those to live catalog26; immutable source
schema/catalog assertions remain unchanged. Preserve both failure logs and rerun
format/lint/focused/full tests. This corrects the missed integration test scope.

Corrected source integration validation: format/lint and24 targeted tests pass;
full guarded ./test.sh passes3,442tests/392warnings in303.37s with1worker after
memory probe3.42GiB. Original pinned HAP1 package reopens under2.0.21/catalog26,
replaying all typed relations from original assets in51.46s; manifest88cad10d…
remains unchanged and contextual query succeeds. Source/schema20/catalog25 is
preserved. New-head hosted CI and upstream merge/release remain pending.

Stacked2.0.22 integration retains the validated source fixes. Format/lint and
49focused source/recipe/query tests pass; final recipe full suite and pinned recipe
replay now pending. The earlier source full suite and actual source reopen passed.

Recipe full-suite preflight refused before collecting tests:1.83GiB available versus
2.5GiB minimum. No guard override or hidden partial suite. Source3,442tests and
recipe49focused tests pass; recipe full-suite/actual pinned-recipe replay and release
remain pending. Publish integration as draft and obtain new-head hosted checks.

Successor-main897c98c integration: source consumer2.0.22/live catalog27 preserves
HAP1schema20/catalog25 and upstream precursorv2/schema21/catalog26. The current
recipe branch advances the precursor reader to v3/schema22/catalog27. Format/lint,
70focused tests and guarded full ./test.sh pass (3,443tests/393warnings,162.99s,
2workers after7.15GiB memory probe). Original HAP1 package again replays/query-opens
with unchanged manifest88cad10d… in53.08s. Upstream's older precursorv1 envelope
requires its historical consumer; this observed compatibility limit is explicit.

The earlier integrated recipe865acb8 subsequently passed3,446tests/394warnings
in119.32s after a7.92GiB probe allowed3workers. Main then advanced again. The
current recipe integrates source a3b80a8 and becomes2.0.23/live catalog27;
new-head focused/full checks and pinned recipe replay are now required.

Final integration validation for recipe 2.0.23: format/lint and 50 focused tests
pass. Guarded ./test.sh passes 3,447 tests with 393 warnings in 168.41 seconds;
the real memory probe allowed two workers with 5.39 GiB available. The original
pinned response package replays under this consumer in 22.06 seconds: all
2,391,692 dispositions, 220,271 responses and original precursor/split relations
match. Source and recipe manifest hashes remain 88cad10d… and a1eec392….
Hosted checks, merge and PyPI deployment remain pending; do not claim release.

## PXD021013 relative MS response (#762)

Specification: [proteometools-ms1-spec.md](proteometools-ms1-spec.md).

- [x] Revalidate current main, existing issue and public source audit; preserve shared editable environment.
- [x] Specify source, homogeneous context, quantities, uncertainty and target before implementation.
- [x] Implement subset-capable source package and EvidenceStore registration.
- [x] Implement registered relative-response recipe and dependence groups.
- [x] Verify meaningful fixtures and real source conservation in the experiment registry.
- [x] Run format.sh, lint.sh and test.sh: 3,458 passed; strict documentation build passes.
- [ ] Publish PR/version and deploy after full release gates.
- [ ] Run prospective grouped biological observer evaluation in Presto.

Review: local source/recipe implementation verified against 320,222 original cells and
1,147 unchanged measured targets. The corrected intended-product graph has two
pilot components; 106 shared off-planned background IDs remain audited. Local suite
3,458 passed in 325.27 seconds; peak process RSS 926,728,192 bytes. Full integration
release gates and PyPI deployment remain pending. No new predictive gain claimed.

## Nanaware2024 quantitative lymph/DC evidence

Spec: tasks/nanaware-lymph-quantities-spec.md.

- [x] Inspect recovered primary assets and distinguish native human lymph, DC presentation and uncertain mouse digest material.
- [x] Freeze typed source contract before implementation.
- [x] File source owner with measurement/sample gaps: Hitlist817.
- [x] Implement source snapshots, typed relations, EvidenceStore integration and integrity checks.
- [x] Verify fixtures and actual source independently; format/lint and3466full unit tests pass;43focused source/catalog tests pass.
- [ ] Publish PR with version bump; merge and deploy after checks.
- [ ] Register an endpoint-specific recipe and prospective biological comparison before fitting.

Source review: actual materialization and independent121346cell/32995quantity/3430reference/23455identification replay pass. Initial full suite:3459pass/7fail, all failures are expected global catalog-version assertions28versus29; updated only those assertions, preserving historical package versions. Focused source tests pass8cases. Source issue: https://github.com/pirl-unc/hitlist/issues/817. Full required rerun/release remains pending.

Required full rerun passes3466tests in294.19seconds; maximum processRSS870432768bytes. Actual source quantity/reference bytes are unchanged. Structural readiness finds raw accession spelling differences and DC reported-length inconsistencies, retained for recipe review rather than silently corrected by the source importer. This source-only task is not complete until its PR is merged and PyPI release verified; endpoint recipe remains follow-up.

## MAPP material-linkage qualification

Specification: [mapp-material-linkage-qualification.md](mapp-material-linkage-qualification.md).

- [x] Audit current deposited MAPP coverage and distinguish MAPP from adjacent
      tryptic projects.
- [x] Establish that the primary paper supplies a publication-level parallel
      MAPP/proteome comparison but not an acquisition/specimen/aliquot crosswalk.
- [x] File/link the material-graph, missing-PXD009438 and parallel-tryptic
      recovery work items (#820, #821, #822, #823; #823 is linked to #757).
- [x] Add verified common material-context/link relations and real MAPP facts.
- [ ] Audit/admit PXD009438 without inventing patient/run identities.
- [ ] Recover an exact PXD009360 mapping or publish a pinned negative-evidence
      audit; never use run order or correlation as a surrogate.

Review: current coverage is PXD009092 (10,673 peptide rows, 149,959 evidence
rows, 18 runs), PXD007263 (3,146/27,196/12) and PXD009508 (6,060/19,581/18):
19,879 deposited peptide rows,196,736 evidence rows and48 MAPP acquisition runs
before deduplication. PXD009438 is the unadmitted fourth captured-fragment MAPP
project. PXD009360/PXD009181/PXD009112 are tryptic panels. The paper's parallel
HEK293 statement is scoped context, not a per-run or physical-material join.

Material-graph review: catalog30 / package version2.0.29 add the shared
`material_contexts` and `material_links` relations. Replayed PXD009092 has25
contexts/24 links, while PXD009508+PXD007263 have42/40; all physical IDs remain
null. Source adapter verification and the final rebased guarded suite pass:3,480
tests,393 warnings in173.10seconds with two workers. `./format.sh` and `./lint.sh`
also pass. One initial suite failure was a stale expected catalog value29 in the
import-isolation test; it now expects the intentional live value30.

## Goldberg2025 canonical MAPP admission

- [x] Freeze source, context and candidate-export specification.
- [x] Implement shared source/query interfaces and explicit dispositions.
- [x] Verify fixtures and original-source replay.
- [x] Register baseline candidates and qualification axes.
- [ ] Complete checks, publication/release and experiment registry.

### Goldberg2025 admission review

Original source and registered candidate export independently pass. All394,732
peptide-cell decisions retained;94,034baseline review candidates, including19,058
omitted publisher cells. Source/publisher/pooled values and material precision
reconcile. Format/lint and3,489unit tests pass; full-corpus CI, PR/merge and verified
release remain pending. No model fit or predictive improvement in this family.

## Goldberg2025 baseline prediction

- [x] Freeze cohort, dependence, feature, optimizer and numerical gates.
- [x] Implement shared composition recipe and verify original-source cohort.
- [x] Qualify exact/sampled gradients and feature support.
- [x] Fit all frozen arms and seal covariate predictions.
- [x] Evaluate held-outs once and independently reconstruct outputs/metrics.
- [ ] Publish results and complete dependency release checks.

### Baseline composition review

Shared recipe and original-source replay pass:40,701measurements across274/74/110
train/validation/test components. All394,732baseline decisions retained; no
publisher filtering or missing-negative inference. Thirteen focused tests, format
and lint pass; full unchanged guarded rerun passes3,493tests/392warnings in314.19s.
First full attempt had16disk-reserve failures while temporary source materialization
occupied space; its log is retained and the guard was not relaxed. Presto2244
independently verifies targets,529,113predictions,allmetricsandbootstrap. Exact/RL
beatuniform; primarysequence-versus-length5%gatefails ontest(4.334%). No APC
concentration claim. PR/rebased checks/release still pending.

## PBMC observed composition and frozen transfer

- [x] Freeze source, confidence, duplication, component, endpoint and transfer policy.
- [x] Register canonical PBMC observed-composition recipe and meaningful tests.
- [x] Verify source and cohort independently with original quantities/metadata.
- [x] Enforce support before prediction: both cohorts fail; no models scored.
- [ ] Complete shared-code checks/publication and experiment closure.

Review: twenty-eight focused source/composition tests, format and lint pass. Independent
Presto2315 replay checks692,452original cells, all13,790decisions and395admitted
quantities/weights; max arithmetic error1.78e-15. CTRL6components/27groups/234cells;
LUPUS7/20/161; both fail the frozen20component gate. No threshold relaxation,
prediction/fit or biological transfer claim. Initial fixture reciprocal-group
omission was fixed before admission; source validation was not weakened. Full
local suite remains gated by2GiBdisk headroom; latest observed~1.42GiB. Hosted
full tests and publication/release remain required before this change is done.


## PXD058121 intact-degradomics source (Hitlist831)

- [x] Review original source audit and freeze tasks/intact-degradomics-source-spec.md.
- [x] Implement standalone source package and common material/query registration.
- [x] Add source-exception fixtures and run format/lint/45 relevant tests; full hosted suite pending.
- [x] Independently verify all real-source values and links through public queries.
- [ ] Publish PR, pass full hosted gates, merge and verify package release.

Implementation in progress: the standalone adapter, typed tables, pinned asset
contract and public catalog/store bindings now exist on the feature branch.
Format and lint pass; six focused source-fixture tests pass (one existing pandas
deprecation warning). Verification caught and corrected prefix replacement that
corrupted acquisition labels, and relationship cardinalities that the public query
API rejected. Repeated views and publisher conflicts now reconcile in the fixtures.
These are software checks, not a real-source admission or biological fit. Public
package/catalog regression checks, original-data replay, documentation and hosted
full tests remain pending. The source audit is Presto2338; feature base is
mergededf061a2/version2.0.32, with proposed source release2.0.33.

Real-source verification: Presto family2026-10-25_0021_codex_intact-canonical
builds in13.37s/659.4MiB and independently replays in12.25s/946.7MiB. All410,349XML
cells/325,718populated cells,44,457quantities,17,862input masses,8,138assignments
and public joined contexts agree with originals. Publisher3,427links retain one
conflict; nine repeated views remain non-independent. The initial independent
verifier omitted substrate-only label grammar; its failure and explicit coverage
amendment are retained. Canonical bytes did not change. No training or predictive
gain. Full hosted gates, review, merge and verified release remain open.

Hosted CI at7ece088found six stale global catalog-version assertions (32versus31), with3,499unit tests passing on Python3.10. Update those global assertions only; historical package schema/catalog contracts remain unchanged. Runtime source/data bytes are unchanged. Run all six affected test files before pushing and require fresh hosted full gates.
The six affected files now pass27tests/one pandas warning in0.35s; format/lint pass. All frozen runtime hashes remain unchanged. Fresh hosted gates required.

## Intact-degradomics kinetic forecast recipe

- [x] Freeze tasks/intact-kinetic-recipe-spec.md against the prospective Presto model plan.
- [x] Implement source-bound registered recipe and all dispositions/context.
- [x] Verify targets, support and blinding with meaningful fixtures and independent originals.
- [ ] Publish, pass full hosted checks, merge and verify release.

Review: 32 focused source/recipe/interface tests pass; format/lint pass. A fixture
initially omitted the original F2 40-minute R1 extra column; corrected before
admission without changing production behavior. Presto0046 independently verifies
993 targets (626 train/181 validation/186 test), all 8,138 dispositions, weights
and source lineage; maximum target error8.88e-16. Sixteen/64-draw sampled gradients
and exact training both fail the biological comparator; this does not invalidate
the original measurements or change the recipe. Full local suite remains deferred
below the unchanged2GiBfree-space prerequisite. Full hosted tests and release are
required. Source PR832 passed all unit/integration/docs/lint checks and merged as
18bbf81d; its2.0.33 release build40149903223 is running separately.

# Native PSMD2 source admission (#837)

- [x] Create isolated durable feature checkout from authoritative main.
- [x] Read source qualification and freeze the source/query implementation spec.
- [x] Implement the shared source adapter, typed relations and metadata contract.
- [x] Add meaningful fixtures and independently replay the real source package.
- [x] Run format, lint and the full default suite with unchanged resource guards.
- [ ] Complete a versioned PR, verified merge and tested PyPI release.

Spec: tasks/native_psmd2_source_spec.md. The native adapter, catalog/query bindings
and source-fidelity fixtures are implemented; verification and release are pending.
Presto family 0424 independently qualifies the original reference and quantities.

Review: 33 focused checks pass, including nine new source-fidelity fixture tests.
Presto family 0519 independently reconciles all 255,635 quantities,73,113 references,
38,641 origins,292,449 workbook cells and228,749 parent alternatives through the
public store; all 16 primary keys and 28 relationships pass. Dependency components
remain 1,689. All 3,533 default non-integration tests pass in 179.10 seconds. Format and lint
also pass. Integration release gates, merge and PyPI publication remain pending.

## Native same-parent contrast recipe (#837)

- [x] Freeze tasks/native-contrast-recipe-spec.md before computing source effects.
- [x] Implement registered recipe, all dispositions, lineage and source binding.
- [x] Verify invariants through meaningful fixtures and independent source replay.
- [ ] Run required software gates and complete versioned PR/tested release.
- [ ] Consume the released recipe in the separately registered biological fit.


Native contrast review: format/lint, 36 focused tests and all 3,544 default tests
pass (184.61s pytest; 189.59s wall). Initial full-suite preflight refused 2.33GB;
retry retained unchanged guards and passed with 5.01GB available/two workers.
Presto family0612 independently replays all10,647decisions,228contrasts and2,736
contributors. Support: train55components/158contrasts, validation8/38, test18/32;
82parent sequences across81components. No fitted predictor or held-out effect
summary. SourcePR839 merged as e755be4 with the identical tree as bfbaa67 base;
source release40167325306 remains live. Recipe PR/release and later fit pending.

## PXD024809 observation source — #845

- [x] Create isolated branch from current main and write source contract before code.
- [x] Complete independent scored/aligned/library lineage audit before freezing compiler schema.
- [x] Implement source package, catalog/bindings/store/provenance and meaningful corruption fixtures.
- [ ] Independently reconcile the original source and complete required format/lint/test gates.
- [ ] Publish patch-version PR, pass hosted checks, merge and verify tested PyPI deployment.

Spec: tasks/dilution_observer_spec.md. No calibration training or biological concentration claim yet.

Implementation review: candidate2.0.40 has15shared tables,17pinned originals,
explicit cross-namespace links and15source-supported named protocol claims.
18focused tests pass, including tamper rejection after manifest rehash, large
IDs, ambiguous/decoy/missing states, source-exact protocol claims and public joins.
format.sh/lint.sh pass. Two full-suite preflights refused at0.25/0.08GiB free plus
speculative memory; no tests ran and no guard was weakened. psutil.available's
13.9GiB includes inactive pages and did not establish admission. Original-source
materialization/replay and full-suite/publication gates remain pending. Preserve
both refusal logs; proceed with a reviewable draft while awaiting real headroom.

Continuation: branch7c9fa84 is published as draftPR846. Hosted Tests40182792458
and Docs40182792527 are live. The latest local admitting memory read is0.063GiB;
no full-suite or source build was launched. Static review found six existing
assertions of the global catalog36; update those to37 without changing the
historical package envelopes, then run the six affected files and source tests.
Keep original-source replay and required full gates pending.

Catalog correction review: all48tests across the six existing catalog files and
new dilution source tests pass in1.63s (one existing pandas warning). Format and
lint pass. Production source modules and fixed data contract are unchanged.

Hosted Tests40182919894/Python3.11 found one additional stale subprocess-output
assertion (catalog string36 rather than37):3568passed,2skipped,1failed in142.55s.
Fix that assertion as part of the same global catalog migration; historical
package catalog constants remain unchanged. Re-run the catalog and source files
before publishing a fresh head. Docs40182919914 and lint already pass.

Final catalog correction:26catalog/source tests pass in1.68s; format/lint pass.
CI failure log40182919894/Python3.11 is retained. New-head full checks required.


## PXD058121 organ baseline-change recipe

- [x] Freeze source/endpoint contract before numeric admission.
- [x] Implement registered F3 compiler and shared package/replay support.
- [x] Verify synthetic edge cases and unchanged F2 output.
- [ ] Run format/lint/test, open PR, merge and verify release.

Spec: tasks/intact-organ-transfer-spec.md. Source issue: #831.
