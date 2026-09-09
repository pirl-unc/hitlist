# Flat experimental-condition columns

Status: implementation handoff. Supersedes the nested proposal on `spec/condition-model`.
No implementation or PR has been created. This document specifies the intended change;
primary-source verification and the acceptance checks below remain implementation work.

## Outcome

Expose a consistent block of condition columns on sample, peptide-observation, and unified
training tables. A consumer must be able to select a documented treatment, genetic change,
control type, or combination using columns, without parsing prose or traversing nested objects.

Use the existing flat `ms_samples` records as the authoring format. Add scalar columns to those
records; do not introduce `line -> sample -> conditions -> modifiers`, allele-axis templates,
per-kind parameter dictionaries, or a separate object hierarchy. Existing sample grouping and
acquisition defaults remain useful and should be reused.

The unit of a curated row is one reported sample group under one defined experimental condition
and measurement context. It may summarize documented replicates. It is not necessarily a
physical specimen, a single MS run, or an individual peptide detection.

## Evidence already inspected

- The current YAML contains 215 study entries, 761 sample records, and 167 distinct condition
  strings, counted directly on main at `735014f`.
- Existing strings contain multiple simultaneous factors, sequential treatments, baseline
  engineering, vehicle controls, mixed groups, and records explicitly not profiled by MS.
- `generate_ms_samples_table()` already produces a flat table. The observation join and
  `generate_training_table()` provide the path to the user's unified dataset.
- The exporter currently compares coarse `condition_category` values when deciding whether
  candidates represent different conditions. This is insufficient for distinct conditions
  sharing a category, including different durations of the same treatment.
- Existing category and APM classifiers inspect agents and targets and apply precedence rules.
  Reading a modifier kind alone cannot replace them.

These counts measure existing curation, not completeness of the papers. Do not describe all
167 strings as independently verified against primary sources.

## Flat schema

Declare one ordered column registry in a small condition module. Use it for validation,
empty-frame schemas, export propagation, training defaults, and documentation. All categorical
values are strings. Empty string means unknown/not reported; it never means untreated.

### Identity and interpretation

| Column | Meaning / permitted values |
|---|---|
| `condition_id` | Persistent short ID within a study. Assigned once in curation; never derived at export time from a label, row position, category, or mutable metadata. |
| `condition` | Original curated condition wording, retained for audit and compatibility. |
| `condition_status` | `annotated`, `partial`, `mixed`, `unreported`. Describes completeness of the categorical annotation of this record, not confidence in peptide attribution. |
| `condition_evidence` | `curated_text` or `primary_source`; blank when nothing has been annotated. |
| `condition_reference` | Source URL/accession plus a usable section, figure, table, sheet, or sample identifier. Required for `primary_source`. |
| `condition_control` | Explicit role: `untreated`, `vehicle`, `mock`, `empty_vector`, `non_targeting`, `isotype`, `positive`, `other`, or blank. |
| `condition_control_for` | Semicolon-separated condition IDs within the same study, only when a comparison is documented. |
| `condition_combination` | `single`, `simultaneous`, `sequential`, `unspecified`, or blank. Describes relationships between documented interventions. |

`(pmid, condition_id)` identifies a curated context row. Export the same pair unchanged; no
second redundant identity field is needed. A label may change without changing this identity.
Never reuse a retired ID for a different context.

### Independent categorical factors

| Column | What to extract |
|---|---|
| `condition_knockout_genes` | Named genes explicitly knocked out/deleted. |
| `condition_knockdown_genes` | Named genes explicitly knocked down; do not relabel knockdown as knockout. |
| `condition_overexpression_genes` | Named genes explicitly overexpressed. |
| `condition_genetic_variants` | Reported gene/variant designations, preserving the available precision. |
| `condition_transfection` | Reported introduced gene/construct or coarse `unspecified` when only transfection is stated. |
| `condition_transduction` | Reported introduced gene/construct or coarse `unspecified`. |
| `condition_cytokines` | Canonical identities of the documented cytokine exposures. |
| `condition_drugs` | Canonical compound names; retain a reported drug class when the compound is unnamed. |
| `condition_infection` | Reported organism designation, preserving uncertainty; no inferred infection from a vector name. |
| `condition_stimulation` | Other documented stimuli or activation/expansion contexts. |
| `condition_antigen_exposure` | Coarse exposure such as `peptide_pulse` or `cross_presentation`. |
| `condition_background` | Explicit intrinsic genetic/functional background, separate from an intervention introduced in this study. |
| `condition_mhc_context` | Reported context such as `monoallelic`, `soluble_mhc`, or `mhc_coexpression`. |
| `condition_culture` | Reported medium/culture category, including `standard_culture` when that is all the source states. |
| `condition_material` | Reported state such as `direct_ex_vivo`, `cultured`, `fresh`, `frozen`, or `biofluid`. |
| `condition_labeling` | Documented labeling context such as `SILAC`; distinct from biological treatment. |

Use an explicit YAML vocabulary/alias table for normalization, not gene/drug/organism lists
embedded in Python. Validate gene designations independently of membership in the APM subset.
Unknown reported entities must remain representable after review; never discard a source fact
merely because an APM filter has no corresponding flag.

For several known values in one column, use sorted, unique, semicolon-separated tokens, e.g.
`ERAP1;ERAP2`. This means both apply to that row, not one or the other. Do not place a union of
alternative conditions into these cells. These are multi-valued categoricals: consumers split
tokens for membership/multi-hot encoding rather than treating a combination as a new agent.

Use `none` only where absence is explicitly supported for that particular field. Do not fill
every missing column with `none` when a source says `unperturbed`: background engineering,
culture, and measurement context may still be present. `unspecified` means an intervention
exists but its target is unnamed; blank means its presence/absence is not established.

`annotated` means the condition facts available in the reviewed text are categorized. It does
not mean all possible experimental variables were reported. `partial` means a known fact is
not yet representable or resolved. `mixed` means the record combines alternative conditions;
keep only facts established for every contributing condition and preserve the source wording.

### Fields already represented elsewhere

Keep `sample_label`, `sample_group`, `species`, `source`, `mhc`, `mhc_class`, `n_samples`,
`profiled`, `reference_proteomes`, provenance overrides, and acquisition fields. Do not copy
their values into redundant new columns. Existing study defaults remain overridable per row.

Do not reinterpret `mhc`: it remains the currently curated typing/candidate context. MHC
genotype, experimentally documented expression, IP scope, and per-peptide restriction evidence
are different claims. This PR must not infer one from another or silently rename their meaning.

Fine protocol parameters are not required for categorical completeness. Preserve available
details in `condition` and source references. Do not introduce one ambiguous `dose`/`duration`
column for a combination containing several agents. A later quantitative schema requires its
own demonstrated use case and unambiguous agent/parameter association.

## Example table

Illustrative reduced projection; omitted columns still exist in the exported schema. These
examples describe the representation and are not a substitute for source verification.

| pmid | condition_id | sample_label | condition_control | condition_knockout_genes | condition_cytokines | condition_drugs |
|---|---|---|---|---|---|---|
| 31530632 | wt | C1R baseline | untreated | none | | |
| 31530632 | erap2_ko | C1R ERAP2 KO | | ERAP2 | | |
| 30833945 | cytokines | A549 cytokine-treated | | | IFNG;TNF | |
| 34497125 | vehicle | SKMEL5 vehicle | vehicle | | | DMSO |
| 34497125 | meki | SKMEL5 treated | | | | binimetinib |

Author those same fields directly on `ms_samples` entries. Do not put them under a nested
`modifiers`, `treatments`, or `categoricals` key. Include `condition_control_for` only after
verifying the comparison. Keep existing labels during migration.

## Standard paper-to-table extraction contract

For each study, use one worksheet/table with the columns above plus the existing sample and
measurement columns. Every row must be tied to the authors' actual sample/group identifier.

1. Enumerate all profiled source groups from Methods, figure/table legends, supplementary
   sample sheets, and deposit metadata. Record unprofiled comparison groups separately using
   `profiled: false`; do not count them as peptide evidence.
2. Fill only explicitly supported columns. Normalize spellings using the shared vocabulary,
   retain original wording, and attach a precise source locator. A curator or extraction tool
   should emit the same flat records and pass the same validator.
3. Record controls from the experiment's comparisons. Baseline, untreated, and control are
   separate concepts. Do not derive control role from an empty treatment list.
4. Record every simultaneous factor. For sequential protocols, record the known categorical
   factors, mark `sequential`, and retain order in the original wording. Do not invent order
   from the order of YAML keys or alphabetically serialized tokens.
5. Split alternatives into actual rows only when the source identifies them. A label such as
   `various combinations` must remain `mixed`/`partial` until the combinations are recoverable.
6. Independently establish whether peptide evidence identifies those rows. A paper describing
   an experiment is not proof that its deposited peptide list preserves the condition axis.
7. Validate and report both sample-annotation coverage and peptide-attribution coverage. Do
   not report the first as evidence that the second improved.

Before claiming confidence in primary-source extraction, execute and document a pilot covering:

| Pattern | Primary source to inspect | What the pilot must demonstrate |
|---|---|---|
| Genetic perturbation and baseline | [Lorente 2019, PMID 31530632](https://pmc.ncbi.nlm.nih.gov/articles/PMC6823859/) | Separate material background, genetic intervention, comparator, and peptide attribution. |
| Multiple cytokines | [Javitt 2019, PMID 30833945](https://pmc.ncbi.nlm.nih.gov/articles/PMC6387973/) | Both factors survive in the same row without inferring individual effects. |
| Vehicle versus drug | [Stopfer 2021, PMID 34497125](https://pmc.ncbi.nlm.nih.gov/articles/PMC8449407/) | Vehicle exposure and control role coexist as explicit columns. |
| Multi-gene study panel | [Shapiro 2025, PMID 40113210](https://pmc.ncbi.nlm.nih.gov/articles/PMC12090245/) | Each row receives its own perturbation, not the study-wide union. |

Also inspect existing monoallelic, sequential, mixed-condition, and unprofiled records. Record
unavailable full texts or ambiguous supplemental identifiers as limits, not verified sources.
The [SDRF-Proteomics model](https://sdrf.quantms.org/quickstart.html) is a useful reference for
keeping sample properties, experimental factors, and sample-to-file relationships distinct;
implementing the full standard is outside this change.

## Peptide attribution and unified exports

- Carry the column block through `generate_ms_samples_table`, expression-anchor sample
  exports, `generate_ms_observations_table`, and `generate_training_table`, including empty
  results, binding-only results, and projected outputs.
- Binding rows receive blank MS condition fields, never invented untreated/control values.
- On a resolved observation, export the matched `(pmid, condition_id)` and its condition data.
  Preserve existing evidence identifiers and `sample_attribution` explaining how it matched.
- On unresolved observations, leave `condition_id` blank. Preserve only fields all eligible
  candidate contexts agree on. Missing on any candidate is not agreement with a known value.
  Never union alternative treatments into a combined-treatment claim.
- A pooled measurement, positive evidence for multiple conditions, and an ambiguous assignment
  are distinct. Retain the existing `arm_resolution` explanations and original evidence.
  Do not expand one unresolved detection into multiple asserted detections.
- Coarse category equality is insufficient to resolve a tie. Distinct context IDs cannot be
  first-picked because their categories agree. Continue resolving sample systems from factual
  source metadata; condition-specific claims require condition-specific discriminators.
- Narrative/label matching must not turn wording changes into condition assignments. Test
  renamed display labels against unchanged explicit source identifiers and resolved IDs.
- Include `condition_id` in candidate metadata throughout every attribution path. Keep source
  references for the facts distinct from evidence that maps an observation to those facts.

Use categorical dtypes for the added low-cardinality columns where the existing exporter does
so. Do not add a Python loop over millions of peptide rows; annotate the small sample table and
propagate through the established joins. Preserve the existing narrow-projection contract.

## Compatibility and classifier migration

This PR is additive. Keep `condition`, `condition_category`, `perturbation`, `is_control_arm`,
`arm_resolution`, and existing APM columns with their documented legacy meanings. The new
`condition_control` explicitly distinguishes experimental controls from the old baseline flag.

Do not delete the prose classifiers merely because flat columns exist. First implement and
compare equivalent structured derivations on the whole curated population, including aliases,
inhibitors, combination precedence, and intrinsic backgrounds. Keep compatibility readers if
necessary; new columns must read explicit curation, not silently fall back to guessing a
category from arbitrary prose. Unknown future strings need review and an explicit status.

Stable engineering and a transient intervention may both appear in an experimental comparison.
Do not change a material's identity based on whether its parental comparator appears in a paper.
Do not retire `sample_group` in this flat design: no new nesting has replaced its function.

## Implementation sequence

1. Add the condition column registry, YAML normalization vocabulary, and validation. Reject
   undeclared fields, invalid statuses, malformed multi-value cells, duplicate IDs, and dangling
   control references. Reject duplicate YAML keys before ordinary dict construction loses them.
2. Assign persistent IDs and curate the existing sample records in place. A mechanical
   migration can suggest annotations from the 167 existing strings, but its output must be
   explicit, reviewed data. Mark these `curated_text`, not `primary_source`. Preserve all 761
   records and existing labels. Unknown material properties remain unknown.
3. Execute the primary-source pilot, recording concrete locators and corrections. File any
   confirmed curation/code defects as issues and link them from the implementation PR.
4. Propagate the shared columns through all exports and repair identity-based tie handling.
5. Add focused regressions and a corpus audit reporting sample coverage, partial/mixed records,
   observation attribution, and every intentional change to existing values.
6. Run `./format.sh`, `./lint.sh`, and `./test.sh`; run required integration/release checks.
   Bump the version, open the PR, check CI, merge, and deploy from clean main per AGENTS.md.

## Acceptance criteria

- All 761 existing sample records survive. Each has a persistent unique context ID and an
  explicit annotation status; any newly discovered split is separately documented.
- Every exported dataset has the same declared condition columns. No nested objects or JSON
  treatment blobs are necessary to filter the condition categoricals.
- Primary-source pilot records include precise locators. The audit distinguishes these from
  mechanical normalization of existing curation, and identifies any remaining extraction gaps.
- A two-cytokine condition preserves both factors; a knockout-plus-treatment condition preserves
  both axes; a knockdown does not become a knockout; intrinsic background remains distinguishable.
- A vehicle control retains both the vehicle and its control role. Unreported conditions never
  become untreated, wild type, or known absence through defaults.
- Mixed alternatives do not become a fabricated simultaneous condition. Sequential order is
  neither inferred from token order nor claimed when the source omits it.
- Different conditions sharing one coarse category remain distinct; an unresolved peptide has
  no asserted condition ID; shared factual metadata can survive ambiguity without unioning arms.
- Stable source-ID assignments survive display-label changes, and all reference joins use the
  persistent key where available. Existing source labels remain usable during migration.
- Binding rows, empty tables, expression-anchor exports, and narrow projections have meaningful,
  consistent defaults. Existing evidence identity, filters, and row counts are conserved except
  for independently verified and documented attribution corrections.
- Corpus comparison reports changes by study and reason. Do not require byte-identical newly
  extended files or silently accept changes to pre-existing scientific annotations.
- Required local checks and PR CI pass before merge; the merged release is published to PyPI.

## Handoff state

Branch: `feat/flat-condition-columns`, based on main `735014f`.
Files changed so far: this spec, the task plan, and the lesson capturing the flat-table preference.
No package code/data changes, tests, commits, PR, merge, or deployment have been performed.
The primary-source pilot is required work, not a completed validation claim.
