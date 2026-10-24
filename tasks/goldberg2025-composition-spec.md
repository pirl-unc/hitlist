# First prediction test on the new baseline MAPP quantities

Codex / GPT-6, October24,2026. Prospective; no fitting or grouped outcome inspection
has occurred. Prerequisite: independently verified Goldberg original package and
registered baseline candidates from the2210 admission family. Source quantity and
confidence counts are already known from admission; they are not held-out results.

## Hypothesis and scope

Baseline quantities may contain sequence-dependent variation that cancels from
response ratios. First establish that a parsimonious sequence score predicts
held-out captured-peptide composition before adding a processing simulator. Test
direct differentiation and genuine discrete REINFORCE on identical observed sets,
features and objectives. Do not change the full availability/concentration goal.

This endpoint is relative measured MAPP signal conditional on observed peptides
from a supplied source accession. It is not molarity, a complete free-peptide
distribution, or evidence about unobserved short/nonbinding products. A549 is not
a professional APC. Full original parent sequences and exact precursor pairing
remain unavailable. The learned score combines processing, capture and detector
effects; no latent part is separately identified by this experiment.

## Frozen cohort and dependence policy

- Original six control acquisitions only. Use canonical candidate dispositions,
  never the publisher-selected subset or pooled Table3 identities.
- Require source peptide PEP in[0,.01), per-run `By MS/MS`, canonical sequence,
  blank reverse/contaminant flags and source-human-only parent annotations.
  This is a declared confidence policy, not a new experiment-level FDR guarantee.
- Require exactly one distinct supplied parent accession for the composition
  group. Preserve exclusions; source annotation does not prove the actual origin.
  Peptide-level intensity may aggregate forms; retain that limitation and do not
  apply a chemistry-specific detector correction without separate qualification.
- Group by supplied accession × acquisition; at least three distinct sequences
  per group. Positive reported quantities only, no missing-as-negative conversion.
- Build dependence components over all original peptide sequences and all parent
  labels before eligibility filtering, including excluded bridges. Repeated runs
  and source IDs for a sequence remain together. Hash sorted component members
  with salt `goldberg2025_baseline_v1`; modulo10 buckets0–5train,6–7validation,
  8–9test. Require at least100/20/20 components and20groups per held-out split.
  Stop and report failed support instead of changing this split.
- Retain prior-study peptide/accession overlap as a descriptive provenance axis;
  do not use historical outcomes for fitting or tune earlier inspected held-outs.

## Inputs, outputs and methods

One peptide sequence is the predictive input. Features: log length,20amino-acid
fractions and20N-/20C-terminal indicators (61raw coordinates). Use fitting-only
scaling and supported SVD coordinates after within-group centering; remove null
directions. No parent sequence, inferred species, guessed machinery, MHC, TCR or
assay-identity embedding. Known source accession/run defines the conditional set,
not a trainable identifier. Other Presto modules are not trained in this family.

One scalar log10 signal score per peptide. Predicted centered log signal subtracts
the group's mean predicted score. Predicted fractions are softmax(ln10×score).
Targets are centered log10 original intensity and within-group intensity fractions.
Normalization conditions on each observed set; it cannot predict its missing mass.
Give components equal weight, then accession/acquisition groups equal weight within
components, then peptides equal weight for centered-log error.

Fixed comparisons:
1. Uniform fractions / zero centered score.
2. Length-only and61-feature ridge: weighted centered-log MSE; alpha.001/.01/.1/1,
   select by validationRMSE only. No test selection.
3. Same supported61-feature linear score trained by exact weighted reverseKL
   q||observed fractions +.01mean(coefficients²),500full-batch float64Adamsteps,
   lr.02, zero initialization, final checkpoint. One exact fit suffices because
   it is deterministic; do not represent repeated identical fits as uncertainty.
4. Identical model/objective/update budget with64fresh candidate-identity samples
   per group/update and leave-one-out reward baseline, seeds42/43/44. Stop gradient
   through sampled log(q/p) rewards; keep the regularizer analytic. No clipping,
   standardization, sample reuse or referenceKL in this first biological test.
   All candidate scores are computed in both arms; this does not test score-saving
   simulator-only training. Report score evaluations and actual local runtime.

No pretraining, restarts, feature expansion or checkpoint selection. Primary
prediction gate: selected sequence ridge improves lengthRMSE≥5% on validation
and the once-only test, with positive testR² and nonconstant predictions. The
matched exact/RL comparison reports both reverseKL and centered-log errors; any
distribution success must beat uniform on both held-outs. RL seed spread is not
biological uncertainty. No claim of RL benefit from tiny single-cohort differences.

## Verification and closure

Register a new timestamped experiment before admission/launch. Independently
reconstruct original targets, groups, exclusions, component split, weights,
features and predictions. Qualify exact versus finite-difference gradients and
sampled-score mean against the exact gradient before fits. Record every allocated
parameter's support/update; output intercepts that cancel must not be allocated.
Freeze all final covariate predictions before opening test outcomes; inspect test
once and do not tune against it. Preserve every condition, failed gate, checkpoint,
per-example validation/test dump and log. Report per-acquisition/component errors,
prediction spread, bootstrap component uncertainty and source-confidence coverage.

Local CPU, one numerical thread,6GiB peak RSS,20minutes per fit,2hours family and
10GiB outputs. No cloud or shared-environment installs. If support, numerics or
resource gates fail, preserve evidence and re-plan before extending this family.
Success is a measured-signal baseline only; next steps toward the full goal still
require independently constrained processing, selection and detection and an APC
population with appropriate measurements.

## Implementation precision, frozen before cohort construction (22:37 UTC)

The shared Hitlist recipe is `goldberg2025_observed_baseline_composition_v1`.
It reuses the existing baseline-candidate compiler, retains its complete decisions,
and adds composition decisions, examples, label-free covariates, dependency nodes
and group summaries. No private Presto source curator. A duplicate sequence within
one eligible accession/acquisition group stops admission rather than being selected
or summed by its measured value.

Dependency nodes are literal `peptide:`+sequence and `parent:`+each supplied parent
label. Connect every original peptide row, including excluded bridges. Sort every
component's nodes; JSON encode with separators(',',':') and ensure_ascii=True.
Component ID is SHA256 of `goldberg2025_baseline_v1:` followed by that JSON; split
is int(component_id,16)%10. No accession alias, homology or specimen identity is
inferred. All group support gates refer to admitted components/groups.

Use natural log length. Feature means/scales are weighted fitting-only moments;
constant scales become1. Center standardized features within each observed group,
then SVD the sqrt(training_weight)-weighted training design. Retain singular
values >1e-10 times the largest; project without whitening. Ridge objective is
weightedMSE+alpha*mean(beta²). The exact and sampled arms use the identical supported
sequence design and .01mean(beta²), Adam betas(.9,.999),epsilon1e-8. No intercept.
For padded categorical sampling, masked positions have zero probability and never
become observations. Log softmax uses ln10 times the log10 score.

Before fitting, check exact autograd versus independent analytic gradients at
zero, a fixed[-.15,.15] coordinate ramp and a seed53N(0,.1) point; max absolute
error<=1e-10. Central finite differences epsilon1e-5 require<=1e-6. For each point,
256independent64-sample REINFORCE estimates require relative mean L2 error<=.10
(denominator floor1e-8) and maximum absolute coordinate error/estimated standard
error<=5 (SE floor1e-8). Fixed qualification seed9017. Preserve any failed gate.

Fit code reads training and validation label files only. All covariate predictions,
checkpoint hashes and validation-only ridge choices are sealed before the once-only
test evaluator reads its label file. Save exact archived predictions; independent
metric reconstruction uses those exact bytes for ranks and separately checks
prediction arithmetic tolerances to avoid spurious rank disagreements on ties.
Uniform correlations are undefined. Bootstrap paired test-component RMSE gains
with2000resamples,seed20261024; this quantifies component variation, not independent
study uncertainty. Record both exact and RL score evaluations, updates and runtime.
