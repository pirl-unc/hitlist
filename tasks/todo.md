# Marcilla Validation Leak Fix (2026-04-23)

## Goal

Prevent the two PMID `24366607` HLA-B*40:02 acid-strip / flow-cytometry IC50 validation rows from being exported as part of the Marcilla mono-allelic MS sample.

## Work Plan

- [x] Confirm the exact leakage path in the built data
  Deliverable: identify the non-elution PMID `24366607` rows and show why they currently join onto the `C1R-HLA-B*40:02` sample.
  Verification: local inspection isolates the offending rows and their assay-comment signature.
  Result: confirmed that PMID `24366607` has `2318` current MS rows, of which exactly `2` (`GSFSRFYSL`, `GEFSRFYSL`) are acid-strip / reference-peptide / IC50 validation observations that would join onto the curated `C1R-HLA-B*40:02` sample by PMID + allele.

- [x] Route the validation rows to the binding-assay path
  Deliverable: update the central binding-assay classifier so Marcilla-style acid-strip / reference-peptide / flow-cytometry IC50 rows no longer remain in the MS index.
  Verification: targeted unit coverage proves the exact assay-comment pattern classifies as binding.
  Result: extended `is_binding_assay()` to recognize competitive-binding validation language (`acid stripped`, `reference peptide`, `IC50`, `β2m`) so the Marcilla validation rows will rebuild into `binding.parquet` instead of `observations.parquet`.

- [x] Add regression coverage for the fix
  Deliverable: test coverage that would fail if these competitive-binding validation rows ever return to the MS path.
  Verification: targeted pytest passes before the full suite.
  Result: added `test_is_binding_assay_competitive_ic50_comment()` with the exact Marcilla assay-comment pattern; local spot-check against the current built PMID shows the new classifier picks out exactly the two leaking peptides.

- [x] Add explicit `ms` names for MS-only public API entry points
  Deliverable: additive aliases for the Python APIs that return MS-only data, so callers can choose names that make the modality obvious without breaking existing imports.
  Verification: small regression tests cover the aliases and the README prefers the new names.
  Result: added `hitlist.export.generate_ms_observations_table()` and `hitlist.observations.load_ms_observations()` as backward-compatible aliases; README examples now prefer the explicit `ms` names and alias tests cover both.

- [x] Run required repo verification
  Deliverable: `./format.sh`, `./lint.sh`, and `./test.sh` all pass on this branch after the fix.
  Verification: zero exit status for all three commands.
  Result: `./format.sh`, `./lint.sh`, and `./test.sh` all passed; full suite result was `286 passed, 2 skipped`.

## Review

- Review finding 1: the Marcilla curation itself was not the real bug; the leak came from the global assay classifier missing competitive-binding validation language in otherwise elution-oriented PMIDs. **Fixed** in `is_binding_assay()`.
- Review finding 2: using plain `flow cytometry` as a binding keyword would have been too broad. **Avoided** by keying the new pattern to competitive-binding terms (`acid stripped`, `reference peptide`, `IC50`, `β2m`) instead.
- Review finding 3: the Python API surface was still using MS-implicit names (`generate_observations_table`, `load_observations`) even though those functions are modality-specific. **Improved** by adding explicit `ms` aliases while retaining the existing names for compatibility.

# Bulk Proteomics Fallback Bug (2026-04-23)

## Goal

Unblock full-suite verification by fixing the bulk-proteomics loaders so an unreadable built `bulk_proteomics.parquet` warns and falls back to packaged source data instead of crashing.

Linked issue: [#144](https://github.com/pirl-unc/hitlist/issues/144)

## Work Plan

- [x] Confirm the failure mode
  Deliverable: show that the failing tests die inside `_load_parquet_or_none()` on `pd.read_parquet(...)` rather than in the bulk filter logic itself.
  Verification: full-suite traceback captured from `./test.sh`.
  Result: `10` bulk-proteomics tests failed on rebased `main` with `pyarrow.lib.ArrowInvalid` raised from `hitlist.bulk_proteomics._load_parquet_or_none()`.

- [x] Patch the loader fallback
  Deliverable: if the built parquet exists but is unreadable, warn and return `None` so the CSV/YAML fallback path executes.
  Verification: targeted regression tests cover both peptide and protein loaders against an invalid fake parquet file.
  Result: `_load_parquet_or_none()` now catches unreadable-parquet exceptions, emits a `RuntimeWarning`, and falls back to packaged sources.

- [x] Re-run required verification on the rebased branch
  Deliverable: `./format.sh`, `./lint.sh`, and `./test.sh` all pass after the fallback fix.
  Verification: zero exit status for all three commands.
  Result: `./format.sh`, `./lint.sh`, and `./test.sh` all passed on the rebased branch; full suite result was `296 passed, 2 skipped`.

## Review

- Review finding 1: the original full-suite failure was not caused by the Marcilla leak fix; it was an independent bulk-proteomics loader bug on current `main` when a built parquet is unreadable. **Addressed** by making the loader honor its documented fallback contract.
- Review finding 2: the first fallback regressions used the full packaged CSVs and pushed `pytest --cov` into an avoidable memory spike. **Fixed** by switching those regressions to small synthetic fallback fixtures that still exercise the corrupt-parquet path and filter semantics.
- Review finding 3: required repo verification passed after the fallback fix. `./format.sh`, `./lint.sh`, and `./test.sh` all succeeded; `./test.sh` finished with `296 passed, 2 skipped`.

# HLApollo PMID 24366607 Curation Plan

## Goal

Curate missing HLApollo study PMID `24366607` (`Marcilla_C1R_B4002_2014`) in `pmid_overrides.yaml`, add regression coverage, bump the patch version, and open a PR for the change.

## Work Plan

- [ ] Confirm the exact upstream shape of PMID `24366607`
  Deliverable: verify the built hitlist observations for this PMID line up with the source paper's `C1R-B*40:02` mono-allelic phosphopeptidome.
  Verification: local query of `~/.hitlist/observations.parquet` confirms row count, allele, MHC class, and source fields.

- [ ] Add PMID `24366607` to `hitlist/data/pmid_overrides.yaml`
  Deliverable: study label, title, note, `override`, `mono_allelic_host`, and one `ms_samples` entry for the `C1R-HLA-B*40:02` transfectant.
  Verification: `load_pmid_overrides()` loads the new PMID and exported sample metadata shows the expected mono-allelic sample.

- [ ] Add regression coverage
  Deliverable: test(s) proving PMID `24366607` exports as a single class I mono-allelic `HLA-B*40:02` sample and does not regress to an ambiguous / pooled representation.
  Verification: targeted pytest coverage passes locally and remains green in the full suite.

- [ ] Bump patch version for the PR
  Deliverable: increment `hitlist.version.__version__`.
  Verification: version file diff shows a patch bump only.

- [ ] Run required repo verification
  Deliverable: `./format.sh`, `./lint.sh`, and `./test.sh` all pass.
  Verification: zero exit status for all three commands.

- [ ] Publish the change as a PR
  Deliverable: commit, push, and open a draft PR linked to issue `#127`.
  Verification: branch, commit, and PR URL recorded below.

## Notes

- Built observations already show PMID `24366607` as `2318` IEDB rows / `2317` unique peptides, all `HLA-B*40:02`, almost entirely labeled `C1R cells-B cell`, which matches the intended single-sample curation shape.
- The closest local precedent is PMID `27920218` (`Alpizar 2017 — HLA-B phosphopeptidome`), which already uses `override: cell_line` and `mono_allelic_host: "C1R"` for `C1R-HLA-B*40:02`.

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
