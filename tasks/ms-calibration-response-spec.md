# Raw-S4 nominal-response recipe

Prospective spec, October23,2026; extends draft#767 without changing source bytes/schema/version2.0.3. Previous goal turn made progress: independent source admission verified and published at Presto50dbfad. Parent experiment plan: Presto experiments/agents/codex/plans/raw-s4-nominal-response.md. Issue#762 owns data limits.

## API and outputs

Add `ms_calibration_raw_s4_nominal_response_v1` as a registered response recipe behind the same EvidenceStore.materialize / CLI / verify-recipe envelope. Keep existing registered distribution recipes byte-compatible. Use a distinct output_kind `registered_response` so arbitrary peak area is never labeled a probability distribution. Catalog advertises availability only when every required calibration relation exists. Reject unknown arguments/recipe IDs and immutable destination collisions.

Keep implementation small: a calibration response compiler plus its envelope/verifier in one module; add explicit dispatch at existing recipe/catalog boundaries. No plugin framework or rewrite of prior MAPP semantics. New functionality remains part of the unreleased2.0.3 PR; no environment install or version claim.

Query complete native calibration S4 measurement, nominal standard/load, reporter, slot, protocol-parameter and assertion tables. Export input source rows (both representations for conservation) with exact IDs, all raw-slot dispositions, per-positive contributor links and per-sequence/site examples. Source cells remain exact provenance pointers into the hash-bound immutable source package. Full source replay compares exports against original package tables when source_package is supplied; standalone verification checks the frozen contract, source manifest, recomputation, grains and summary arithmetic. Source inventory hashes alone are insufficient.

## Scientific contract

RawS4 only; preserve3069 raw slot decisions and quantify normalized rows as excluded by endpoint scope. Positive finite observations are eligible conditionally on being reported. Nonpositive/empty/absent responses remain excluded with original status and reason; they are not zeros/nondetections. Exclude raw standards with multiple reporter forms, explicit aggregation suffixes, or isotope annotations missing/different from the nominal panel. Count multiplicity among raw reporter IDs only (the curve-only SMD2.sum4 is not a raw alternative). Preserve every applicable reason rather than first-reason overwriting.

Unique raw reporter per nominal standard still does not certify form-specific molarity. Keep original annotations, sequence, nominal fmol/unit, source group as audit-only, site/slot, source-cell IDs and source protocol scope. No inferred acquisition/day/specimen, form fraction, saturation/censor cutoff or APC context. Never use normalized measurements or source curve points as extra examples.

Summaries: median log10 peak area within reporter×site, min/max log10 response, count and exact contributing measurement IDs. Minimum one reported positive slot; missing slots remain in dispositions. Each summary gets weight1 in a prospective peptide×site analysis; do not weight by number of repeated source slots. Retain nominal log10 load as a covariate, not assumed linear denominator. Supply response-independent peptide grouping: sort all nominal panel sequences by SHA256(recipe_id + ':' + sequence); rank modulo5 defines fold_id. All sites and source slots for one sequence share its fold; folds are metadata, not a fitting split assertion.

## Verification before biological diagnostic

Unit fixtures exercise absent/empty slots, multiple forms, missing isotope annotations, raw-only multiplicity despite a curve-only alternative, aggregation exclusions, repeated-site summary arithmetic, all-source dispositions, deterministic group splits, immutable materialization, source absence, unknown arguments and malicious content edits with recomputed file hashes. Source-linked verification must reject mutated source targets even if the output is internally consistent. Re-run affected existing recipe/catalog tests and full normal test.sh with intact memory guard, format/lint/docs; hosted CI on the new head.

Register a new Presto experiment before actual recipe materialization/fitting. Independently recompute all original XML rawS4 exclusions, nominal joins, summaries and folds without importing this compiler. Freeze that admitted recipe, then run fixed nested peptide-grouped ridge diagnostics (load/site baseline versus length/AA fractions) with full held-out prediction dumps and independent metric/coefficient replay. No training before admission passes; no neural/RL comparison on an assay lacking processing trajectories.

Pre-source-run checks:61 focused tests pass1.05s, format/lint pass255 files, strict docs pass. Normal test.sh retains its default memory budget and refuses before pytest (see /private/tmp/hitlist-ms-response-full-tests.log). Tests include rehashed internally consistent target tampering rejected when the original source is supplied. Actual original-source qualification and fitting remain next.

Execution review: clean96b9232, original-source admission and all295 nested grouped fits independently verified in Presto0359. The source supports22 observed peptides, not all23 form-eligible sequences; APPPGGWPLQPR remains an unobserved rawS4 peptide, not zero. Context baseline wins; shared sequence regularization shrinks context too, so a future separate-penalty diagnostic is an untested mechanism question. Full result: https://github.com/pirl-unc/presto/blob/9f8e4e0/experiments/2026-10-23_0359_codex_raw-s4-response/README.md. Source and model contracts were not changed after outcomes.
