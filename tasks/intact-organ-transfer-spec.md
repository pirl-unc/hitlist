# PXD058121 organ time-course baseline changes

The existing registered temporal recipe selects F2 only. Add a separately
registered F3 endpoint, `pxd058121_organ_baseline_change_v1`, without changing the
F2 contract, table schemas, row decisions, weights or arithmetic. Presto's frozen
transfer family 2026-10-25_1655 has verified 44 F3 acquisitions and no recorded F2
acquisition overlap. Numeric F3 admission and model scoring have not occurred.

## Fixed endpoint and source contract

- Original `organs` views, Brain/Liver/Spleen, 0/30/60/90/120 minutes only.
- Exclude unresolved raw-acquisition links, F2 acquisition/raw-file overlap and
  source-backed `repeated_acquisition_source_views` connecting F2 and F3, including
  conflicting-value claims. Preserve all assignment dispositions. Outside-F3
  assignments have no interpreted quantitative values in this recipe.
- Apply the existing positive finite, valid span, single reported/replayed mass
  assignment, <=10 ppm, explicit unmodified and input-ID/intensity-agreement gates.
  Exclude every eligible duplicate sequence/acquisition mass; require exactly one
  qualified intact parent in the same acquisition. Do not repair absent IDs.
- Target: log10(product/intact) at t minus the mean of that same organ/parent/span's
  eligible nominal-zero log ratios. Baseline eligibility cannot depend on future
  values or coverage. Retain every baseline assignment/measurement/normalizer ID,
  original value, log ratio and replicate label in a baseline table. Preserve the
  distinction between absent and zero, and between replicate labels and physical
  biological independence. No imputation or censoring invention.
- Splits are supplied `baseline`, primary `test` (30/60) and descriptive
  `extrapolation` (90/120). There is no fitted/validation partition. Weighted target
  cells give equal mass to organ, then distinct peptide curve, time, replicate.
  Support passes only if each fixed organ has >=20 supported primary sequences and
  >=40 primary cells. Support failure does not drop organs or change the contract.
- Preserve target values only in decisions/examples; covariates contain supplied
  baseline value/IDs but no later product value, normalizer value, log ratio or
  target change. Absolute quantities and APC concentrations remain unidentifiable.

## Implementation and verification

1. New small endpoint compiler and public catalog/materialize/verify dispatch.
   Share the existing immutable snapshot/replay packaging machinery through private
   implementation hooks; retain the F2 public signatures and default behavior.
2. Meaningful synthetic source fixtures: uneven replicate/time support, per-run
   gain and per-peptide fixed-response cancellation, baseline-only/future-only
   sequences, ambiguous/modified/conflicting masses, missing parent, raw/reuse
   conflicts, and no future values in covariates. A synthetic fixture is a software
   check, not predictive biological evidence.
3. Verify complete output schema/row replay, including rehashed target or baseline
   tampering. Compare F2 compiler output with the unchanged base implementation.
4. Run required format/lint/test scripts with one local worker and normal memory
   admission. No shared Python installation changes or main-branch edits.
5. Bump 2.0.40 to 2.0.41, open linked issue/PR, then merge/release only with passing
   required checks. A release is not complete until PyPI deployment is verified.

The upstream recipe is independently useful; Presto owns its frozen 21 predictors,
prospective quality thresholds, model comparisons and biological interpretation.
