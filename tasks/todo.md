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
