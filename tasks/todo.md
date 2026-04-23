# Stražar 2023 Ingestion Plan

## Goal

Ingest Stražar et al. 2023 (PMID 37301199) as supplementary HLA-II mono-allelic data, using the existing supplementary-manifest pipeline plus a curated PMID override that records the study-wide metadata and mono-allelic method.

## Work Plan

- [x] Confirm supplementary file contents and study structure
  Deliverable: identify which workbook/sheet contains the peptide-level HLA-II ligand table, the allele naming format, whether peptides are already allele-restricted per row, and whether any sample splits or conditions need separate CSVs.
  Verification: local inspection notes reflected in the curated CSV shape and manifest comments.
  Result: the peptide payload is `mmc3.zip` → `ST_ligands_merged.csv`; hitlist ingests only the current-study `dataset == "internal_aff_clean1and2"` slice, not the merged public-training rows.

- [x] Add PMID 37301199 override in `hitlist/data/pmid_overrides.yaml`
  Deliverable: study label, title, note, allele summary, `mono_allelic_method`, acquisition metadata if recoverable from the paper/supplement, and `ms_samples` entries that enumerate the mono-allelic HLA-II panel with class II MHC strings.
  Verification: override loads cleanly in tests and sample export surfaces the new study metadata.
  Result: added 42 class II mono-allelic samples with exact HLA strings, peptide counts, Strep-tag II method metadata, and Orbitrap Exploris 480 / Spectrum Mill acquisition details.

- [x] Add supplementary peptide payload(s) and manifest entry
  Deliverable: one or more CSVs in `hitlist/data/supplementary/` plus `hitlist/data/supplementary.yaml` entries with correct defaults for human class II mono-allelic transfectants.
  Verification: `scan_supplementary()` includes PMID 37301199 rows with expected class II / mono-allelic metadata.
  Result: added `hitlist/data/supplementary/strazar_2023_hla2.csv` with 308,418 peptide-allele rows and a manifest entry that documents the source filtering.

- [x] Add or extend tests for the Stražar ingestion
  Deliverable: assertions covering manifest presence, class II row loading, mono-allelic classification via `mono_allelic_method`, and basic size / allele expectations for PMID 37301199.
  Verification: targeted pytest coverage passes locally.
  Result: added tests for paired class II allele resolution, Stražar method-based mono-allelic classification, supplementary load counts, and exported sample metadata.

- [x] Run required repo verification
  Deliverable: `./format.sh`, `./lint.sh`, and `./test.sh` all pass after the ingestion.
  Verification: command exit status 0 for each script.
  Result: all three passed; `./test.sh` finished with `186 passed, 1 skipped`.

## Review

- Review finding 1: the Stražar supplement does not live in the small XLSX tables; the usable peptide payload is `mmc3.zip`, and the published CSV/ZIP bundle merges current-study data with IEDB and Abelin 2019. **Handled** by ingesting only `internal_aff_clean1and2`.
- Review finding 2: paired class II restrictions like `HLA-DQA1*01:03/DQB1*06:03` were previously classified as `unresolved`, which broke `mono_allelic_method` overrides for DQ/DP pair strings. **Fixed** in `classify_allele_resolution()`.
- Review finding 3: the full verification path is expensive because `test.sh` scans the large supplementary CSVs and observation/export joins under coverage. **Verified** anyway per repo policy: `./format.sh`, `./lint.sh`, and `./test.sh` all passed.

# PR 26 Follow-up Plan

## Goal

Make the observations export reliable enough for large-scale peptide + MHC + experimental-metadata extraction, with explicit fixes for current correctness gaps and a path to higher-confidence sample joins.

## Work Plan

- [ ] Fix mono-allelic filtering in `generate_observations_table()`
  Issue: `--mono-allelic` / `--multi-allelic` currently checks the wrong column name, so filtering is inert.
  Deliverable: normalize on one boolean field name (`is_monoallelic` or an exported alias) and add API + CLI coverage that proves row counts change as expected.
  Verification: targeted unit test for both flags and one CLI test covering `hitlist export observations --mono-allelic`.

- [ ] Fix supplementary species propagation
  Issue: supplementary rows currently default `species` to empty string, and rows without explicit allele assignments also end up with empty `mhc_species`, causing them to disappear under species filters.
  Deliverable: derive species consistently from manifest defaults and/or host metadata so all JY rows remain human even when `mhc_restriction` is blank.
  Verification: test that `scan_supplementary()` returns human species for PMID 38480730 and that species-filtered observations retain the expected supplementary rows.

- [ ] Expand cache invalidation to supplementary payload files
  Issue: `build_observations()` fingerprints the supplementary manifest but not the CSV payloads, so edited supplementary data can leave a stale cache in place.
  Deliverable: include every referenced supplementary file in the cache fingerprint.
  Verification: unit test for fingerprint contents plus a cache-validity test that flips when a supplementary CSV fingerprint changes.

- [ ] Make the observations join explicit about match quality
  Issue: the current join is best-effort and silently falls back from allele match to PMID-only match, which makes downstream consumers over-trust sample-level metadata.
  Deliverable: add join provenance columns such as `sample_match_type`, `matched_sample_count`, and optionally `matched_sample_index` or `sample_id`.
  Verification: tests covering exact allele match, genotype containment match, single-sample fallback, and unmatched rows.

- [ ] Tighten sample matching for multi-sample PMIDs
  Issue: matching by `mhc_restriction in sample.mhc.split()` is not enough when one PMID contains multiple samples with overlapping genotypes, shared alleles, or mixed class I/class II arms.
  Deliverable: rank candidates deterministically using additional evidence already present in metadata, such as MHC class, mono-allelic status, sample type, and curated sample identifiers where available.
  Verification: fixture-based tests for at least one ambiguous multi-sample PMID and one mono-allelic PMID.

- [ ] Add a stable sample identifier to exported sample metadata
  Issue: joined rows currently inherit free-text sample labels, which are not ideal for downstream joins or debugging.
  Deliverable: assign a reproducible `sample_id` in `generate_ms_samples_table()` and carry it into the observations export.
  Verification: tests asserting uniqueness within PMID and persistence across repeated exports.

- [ ] Separate observation-level allele assignment from sample-level genotype
  Issue: supplementary peptides with blank `mhc_restriction` still belong to a human sample, but they do not have peptide-level allele evidence; downstream users need that distinction made explicit.
  Deliverable: add fields such as `has_peptide_level_allele`, `sample_mhc`, and possibly `allele_assignment_source`.
  Verification: tests for blank-allele supplementary rows showing retained sample metadata but false/empty peptide-level assignment fields.

- [ ] Broaden packaging and runtime tests for parquet export
  Issue: `hitlist export observations -o *.parquet` adds a parquet dependency path that should be exercised explicitly.
  Deliverable: add CLI or unit coverage for parquet output and document the engine dependency expectations.
  Verification: test writing a parquet export in CI or, if that is too heavy, a narrower unit test for the output branch plus dependency note in docs.

- [ ] Add end-to-end supplementary build coverage
  Issue: the current tests mostly validate `scan_supplementary()` in isolation, not that supplementary rows survive the full build and export flow.
  Deliverable: add an integration-style test around `build_observations()` with a small synthetic supplementary fixture.
  Verification: assert that the built observations table contains supplementary rows, dedupes against scanner data correctly, and preserves provenance columns.

- [ ] Document trust levels for downstream users
  Issue: the new export looks training-ready, but some rows are still heuristic joins rather than assay-resolved sample assignments.
  Deliverable: update docs to distinguish high-confidence rows (mono-allelic exact match, unique sample fallback with curated genotype) from heuristic rows.
  Verification: README or task doc update reviewed alongside code changes.

## Recommended Order

1. Fix correctness bugs that make current output wrong: mono-allelic filtering, supplementary species propagation, cache invalidation.
2. Add join provenance so the current heuristic behavior becomes inspectable.
3. Improve ambiguous multi-sample matching and stable sample identifiers.
4. Extend integration coverage and document trust levels.

## Completed

- [x] Fix mono-allelic filtering in `generate_observations_table()`
  - Fixed column name: `is_mono_allelic` → `is_monoallelic` in export.py
- [x] Fix supplementary species propagation
  - Added `normalize_species()` to curation.py using mhcgnomes Species.get()
  - Replaced ad-hoc `_species_from_host()` in supplement.py with `normalize_species()`
  - JY peptides without allele assignment now correctly get `mhc_species="Homo sapiens"`
- [x] Expand cache invalidation to supplementary payload files
  - `_source_fingerprints()` now fingerprints every CSV referenced in supplementary.yaml
- [x] Species normalization across codebase
  - `normalize_species()` handles: "human", "homo_sapiens", "Homo sapiens (human)", etc.
  - Applied at: export.py, observations.py, scanner.py, indexer.py, supplement.py
  - `--species human` now works identically to `--species "Homo sapiens"`
- [x] Vectorize `generate_observations_table()` join
  - Replaced `iterrows()` over 4.9M rows with pandas merge operations
  - Test suite: 20+ minutes → under 2 minutes

## Review

- Review finding 1: mono-allelic filter checks `is_mono_allelic` / `src_mono_allelic`, but current observations use `is_monoallelic`. **FIXED.**
- Review finding 2: supplementary rows for PMID 38480730 currently produce 5,179 rows with empty `mhc_species`, which breaks `--species Homo sapiens`. **FIXED.**
- Review finding 3: cache invalidation currently fingerprints `supplementary.yaml` but not the referenced supplementary CSVs. **FIXED.**

# Unified Training Export for Presto (2026-04-23)

## Goal

Add a single integrated training-export surface that exposes hitlist's pMHC evidence in a form suitable for model training, including Presto, without changing curation inputs and without introducing a new persistent "training index" sidecar. The built indices should keep their current semantic boundaries, be built together through the existing build flow, and support clear export methods for different downstream use-cases.

## Design Constraints

- [x] Preserve current index boundaries
  Deliverable: continue to treat `observations.parquet` as MS/elution evidence, `binding.parquet` as in-vitro binding evidence, and existing mapping/sample indices as their own canonical stores.
  Verification: no new stored `training.parquet` or parallel index family was added to the build graph.

- [x] Add one high-level unified export surface
  Deliverable: add a single public API entry point for training-oriented exports that composes the existing indices instead of bypassing them.
  Verification: `generate_training_table(...)` now lives in `hitlist/export.py`, composes the canonical indexes, and is covered by targeted unit tests.

- [x] Keep API and CLI semantics aligned
  Deliverable: the same conceptual export should be reachable through both Python and `hitlist export ...`, with the CLI flags mapping directly onto the API behavior.
  Verification: `hitlist export training` and `_export_training(...)` map directly onto `generate_training_table(...)`, with CLI helper coverage for the flag mapping.

## Proposed API Shape

- [x] Add `generate_training_table(...)` to `hitlist/export.py`
  Deliverable: one function that can export MS rows, binding rows, or both; preserve `evidence_kind` in the unified result; reuse existing exported columns where available.
  Verification: added training-export tests for unified `both` mode and invalid evidence-mode rejection.

- [x] Support two representation modes with one semantic contract
  Deliverable: default compact mode keeps one row per evidence row with stable evidence/sample context plus optional aggregated mapping columns; `explode_mappings=True` returns one row per `(evidence row, peptide mapping)` for flank-aware training consumers such as Presto.
  Verification: compact unified export keeps one row per evidence row; exploded export test confirms per-mapping row expansion plus flank columns.

- [x] Join peptide mappings through the canonical mapping index
  Deliverable: when requested, attach `protein_id`, `gene_name`, `gene_id`, `position`, `n_flank`, `c_flank`, `proteome`, and `proteome_source` from `load_peptide_mappings(...)`, filtered only to peptides present in the export.
  Verification: exploded export joins against `peptide_mappings.parquet`; tests assert per-peptide multi-mapping survives the export.

- [x] Preserve pMHC-relevant context instead of re-flattening it away
  Deliverable: unified rows keep assay/sample metadata already present in observations and binding exports, including MHC restriction/genotype context and measurement metadata when available.
  Verification: unified export tests confirm MS rows keep sample provenance/context while binding rows are tagged `sample_match_type="not_applicable"` instead of receiving fake sample context.

## Proposed CLI Shape

- [x] Add `hitlist export training`
  Deliverable: new subcommand under the existing export CLI, not a separate script.
  Verification: added the subcommand to `hitlist/cli.py` and CLI helper coverage for its argument wiring.

- [x] Keep one build command for underlying indices
  Deliverable: continue using the normal build path for observations/binding/mappings rather than adding a dedicated training-build command.
  Verification: the new export reads only from `observations.parquet`, `binding.parquet`, and `peptide_mappings.parquet`; no new build target was introduced.

- [x] Expose clear use-case filters
  Deliverable: training export supports clear evidence selection (`ms`, `binding`, `both`), shared pMHC filters, and a flag for mapping explosion.
  Verification: API + CLI helper tests cover `include_evidence`, `explode_mappings`, and shared pMHC filters.

## Work Plan

- [x] Implement `generate_training_table(...)` and any small export helpers needed to align observation/binding columns cleanly.
- [x] Wire `hitlist export training` into `hitlist/cli.py` with direct, unsurprising flags.
- [x] Add API and CLI tests for compact, exploded, and evidence-selective exports.
- [x] Update README / docs so downstream users understand index boundaries versus export surfaces.
- [x] Bump the package version per repo policy.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`.

## Review

- [x] Review question 1: does the new export create a second canonical representation of the same index data, or is it clearly a composed export surface?
  Answer: it is a composed export surface only. The canonical stored indexes remain observations, binding, and mappings.

- [x] Review question 2: are the semantic boundaries between observations, binding, mappings, and training export obvious from both the Python API and CLI?
  Answer: yes. `hitlist export observations` and `hitlist export binding` remain the evidence-family-specific exports; `hitlist export training` is the only unified training-facing surface and requires no extra index.

- [x] Review question 3: can Presto consume the export directly, including a flank-aware per-mapping mode, without requiring additional ad hoc joins?
  Answer: yes. `generate_training_table(..., explode_mappings=True)` / `hitlist export training --explode-mappings` produces one row per `(evidence row, peptide mapping)` with `evidence_kind`, `evidence_row_id`, mapping identity, and flanks.

## Review

- Review finding 1: the cleanest way to avoid a new sidecar was to keep the canonical storage split exactly where it already belongs (`observations.parquet`, `binding.parquet`, `peptide_mappings.parquet`) and add one composed export on top. **Implemented** via `generate_training_table(...)` and `hitlist export training`.
- Review finding 2: a unified export still needs to make non-applicable binding fields explicit instead of silently dropping them or fabricating sample metadata. **Handled** by normalizing the mixed schema and tagging binding rows with `sample_match_type="not_applicable"`.
- Review finding 3: full-dataset mapping explosion cannot rely on a giant parquet `IN (...)` predicate for every peptide. **Handled** by using push-down filters for small peptide sets and falling back to a full mapping scan plus in-memory peptide filtering for large exports.
- Review finding 4: repo verification passed after the integration work. `./format.sh`, `./lint.sh`, and `./test.sh` all succeeded; `./test.sh` finished with `286 passed, 2 skipped`.

# PR 138 Review Fixes (2026-04-23)

## Goal

Address the review findings on the unified training export without changing its overall architecture: keep the single composed training surface, but tighten its filter semantics, allele-resolution semantics, and projected identity contract.

## Work Plan

- [x] Make exploded mapping exports respect gene filters
  Issue: filtering evidence rows first is not enough; shared peptides can re-expand into unrelated mapping rows when `explode_mappings=True`.
  Deliverable: carry resolved mapping filters into the exploded mapping load so `gene`, `gene_name`, and `gene_id` constrain the mapping rows as well as the evidence rows.
  Verification: added a regression test where a shared peptide maps to `PRAME` and `MAGEA1`; `gene_name="PRAME"` now returns only the PRAME mapping row.

- [x] Fix `has_peptide_level_allele` semantics on unified exports
  Issue: the training export currently treats any non-empty `mhc_restriction` as allele-level, which incorrectly marks serological and class-only binding rows as resolved.
  Deliverable: derive the flag from allele resolution where available, falling back to the non-empty heuristic only when no resolution metadata exists.
  Verification: added regression tests for both `generate_observations_table()` and `generate_training_table()` covering `four_digit`, `serological`, and `class_only` restrictions.

- [x] Preserve stable evidence identity under column projection
  Issue: projected training exports keep `evidence_kind` but drop `evidence_row_id`, even though exploded outputs need a stable regrouping key.
  Deliverable: projected exports keep both `evidence_kind` and `evidence_row_id` whenever the row id exists.
  Verification: added a regression test for `generate_training_table(columns=["peptide"])` asserting that both identity columns survive projection.

- [x] Capture the review correction in lessons
  Deliverable: create/update `tasks/lessons.md` with the pattern that new composed exports need adversarial tests for post-filter expansion and projected identity.
  Verification: created `tasks/lessons.md` and added three review-driven lessons for composed exports.

- [x] Re-run required verification
  Deliverable: `./format.sh`, `./lint.sh`, and `./test.sh` all pass after the fixes.
  Verification: all three passed; `./test.sh` finished with `290 passed, 2 skipped`.

## Review

- Review finding 1: exploded mapping exports were only filtered at the evidence-row stage, so shared peptides could re-expand into unrelated genes. **Fixed** by resolving gene filters once and applying them to the mapping load as well as the evidence load.
- Review finding 2: `has_peptide_level_allele` was using a string-presence heuristic that mislabeled serological and class-only restrictions as allele-resolved. **Fixed** by switching to resolution-aware logic with a conservative string fallback only when resolution metadata is absent.
- Review finding 3: projected training exports dropped `evidence_row_id`, which made regrouping exploded rows impossible for narrow projections. **Fixed** by preserving both `evidence_kind` and `evidence_row_id` in projected exports.
- Review finding 4: required repo verification passed after the fixes. `./format.sh`, `./lint.sh`, and `./test.sh` all succeeded; `./test.sh` finished with `290 passed, 2 skipped`.
