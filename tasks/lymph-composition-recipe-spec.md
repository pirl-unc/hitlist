# Independent lymph/DC composition and matched exact/REINFORCE comparison

Registered October 24, 2026 by Codex / GPT-6, before recipe membership counts,
target distributions, new fitting or held-out scores. This extends the active APC
availability objective; success here is necessary evidence, not completion.

## Hypotheses and endpoints

1. A frozen independently evaluated measured-MS response model transfers to these
   original positive PEAKS area measurements.
2. Full-parent processing improves conditional peptide composition beyond that
   observer and simple sequence regression.
3. Exact differentiation and fresh discrete REINFORCE, with the same physical
   model, initialization and reverse-KL objective, reach comparable predictions.

Human native extracellular lymph and mouse DC I-Ab-eluted material are separate
cohorts. DC database-search is primary; library-search is a separately fitted
secondary sensitivity cohort reusing acquisitions, never an independent replicate.
No pooled species inference, inferred KO or new pMHC affinity claim. DC prediction
combines processing, MHC selection and observation and cannot identify free APC
concentrations. Human lymph is not an intracellular APC pool.

## Canonical data recipe (implemented in Hitlist, not a private Presto curator)

Use only pinned source manifest
6e82a782b0511ba56425855f0764c4c40127e715fdb10b10e6c12118ab539c4c.
The existing original-cell/reference audit remains binding. Preserve all source
observations, quantity-cell dispositions, sample/context/source identifiers and
reported length/start discrepancies; derive positions by exact sequence occurrence.

- Scope area, not intensity; retain missing/zero/invalid values as dispositions,
  never absent negatives or numerical zeros. No new confidence threshold: these
  are the published table's identifications, with unqualified common FDR.
- Primary chemistry matches the frozen observer: no numeric modification except
  one +57.02 (+/-0.02 Da) on each C, and no other declared modification. Canonical
  non-C unmodified peptides qualify. This is a chemistry restriction, not a
  claim that modified products do not exist. Preserve excluded forms.
- Original-reference matching retains the declared accession token/isoform.
  Candidate origins include exact occurrences in matching original accession
  references and original same-study deposited peptide-evidence alternatives.
  Collapse identical full sequences, never distinct sequences. Require one
  canonical full parent sequence for this identifiable within-parent endpoint;
  sum all exact occurrences. Preserve ambiguous/missing origins and exclusions.
- Before outcomes or filtering, connect every original parsed peptide to its
  supplied accession and candidate full-parent sequences, including excluded
  rows, both DC algorithms, shared peptides and all candidate origins. Components
  determine all cohorts' common splits; no peptide/full-parent overlap is allowed.
- Choose one feature per block/sequence/parent by greatest finite source -10lgP,
  then numeric charge ascending, then stable observation ID. Choice uses no area
  and is the same across channels. No charge/form sum, best-intensity selection,
  per-channel replacement or claim of total peptide quantity.
- A group is block/sample/full-parent with >=3 distinct selected positive peptide
  features. Output measured log10 area centered within that group and area
  fractions. Predictions are centered/normalized over the same observed set;
  no target mean enters features. This conditions on observed positive features.
- Split SHA256('nanaware2024_composition_v1:' + component_id), integer modulo10:
  0..5 train,6..7 validation,8..9 test. No reseeding, cohort switching or support
  relaxation. Pilot minimum per cohort:20 train/5 validation/5 test independent
  components. Failed cohorts receive a no-fit report. This is a small pilot,
  not the previous macrophage study's 100/25/25 confirmation design.

Weights: equal components, equal parent/sample groups within component, equal
peptides within group for log10 MSE. Reverse KL weights groups by the same
component/group weights. Multiple channels do not multiply independent parents.
No train/validation/test unit conversion differences or target clipping.

## Models and fixed fitting contract

Local float64 CPU, one numerical thread, no environment change/cloud. No pretrained
biological processing, MHC/TCR/species modules. Frozen observer is1702sequence ridge
alpha0.1 with its exact archived feature masks, scales and coefficients; no use of
its old held-out labels. Its calibrated coefficient control is learned on new
training only; report fixed coefficient1 and train-only scalar calibration.

Controls: zero centered prediction; length/length-squared ridge; sequence ridge
(length, length-squared,20fractions,20first/20last indicators), penalties
0.001/0.01/0.1/1 selected on validation MSE; frozen MS coefficient1; calibrated
MS scalar (unconstrained scalar, sign reported, not a pure detector calibration).
Training-only centering/scaling and rank masks; omit unidentifiable directions.

Physical model: continuous unit parent supply, independent irreversible cuts,
common removal fixed1. Bond rate exp(rho + left/right AA coefficients); center
the20indicator effects separately via19contrasts each (39coordinates with rho).
All contiguous1..L products exist mathematically; calculate queried steady-state
amounts analytically and sum occurrences without materializing O(L^2) spans.
No peptide-specific offsets. Initialrho=log(.01), residue effects0. Frozen observer
log10score either absent or added with coefficient1. Keep its missing terminal
effects and cross-chemistry/instrument limitations explicit.

MSE arms: uniform and sequence processing, each with/without frozen observer.
500full-batch Adamsteps,lr.02,betas(.9,.999),eps1e-8, penalty0.01mean(residue²),
no rho penalty, no dropout/restarts/schedules/clipping. Use final checkpoint;
validation evaluates controls and gates, never selects physical steps.

Matched distribution arms: sequence processing+fixed observer, exact reverse
KL(q||measured area fractions) versus fresh REINFORCE64 per group with leave-one-out
reward baseline; seeds42/43/44, same500steps/Adam/penalty/initialization. q is normalized
over admitted observed peptide candidates, not full intracellular support. Reward
is logp-logq; expected score gradient of negative reward equals exact reverse-KL
gradient. Compute all candidate scores in both arms (no claimed score savings).
No group standardization, clipping, reuse or reference KL in this biological
reference; prior synthetic factorial evidence does not establish biological gains.

## Qualification, outcomes, resources and stop rules

Before real fits: test source feature choice independent of area, original
reference alternatives, excluded-row dependence, chemistry, missingness, exact
coordinates despite bad annotation, target normalization, deterministic split,
atomic/tamper rejection. Independently reconstruct real targets from original
workbook cells and candidate memberships. Existing steady-state math tests plus
independent scalar formula and exact expectation of the discrete policy gradient
must agree <=1e-8. Real-data gradient probe256groups of64samples at fixed initial
state must have relative mean error<=0.15 or absolute error<=0.01; failure stops
sampled fitting, not a change of objective or sample count.

Admission/eachfit cap600seconds/3GiB; total fitting3600seconds, outputs<=1GiB,
retain>=2GiBdisk. Preserve failures and freeze any implementation-only correction.
No new samples or altered eligibility in response to predictive scores.

Freeze all checkpoints and test covariate predictions before test outcomes.
Report validation/test weightedMSE/RMSE/MAE/R², Pearson/Spearman, outputspread,
forward/reverseKL,TV, component/sample strata, runtime/score evaluations, gradient
error, every seed and source-linked prediction. Affinity/binder/nM metrics do not
apply. Primary physical success requires >=5% validation and test RMSE gain over
both selected sequence ridge and calibrated MS control, plus positive testR².
Observer transfer requires >=5%gain over length on both splits. Small-cohort
uncertainty must be explicit; no general APC claim even if gates pass.

Independently replay checkpoints/predictions/metrics; publish report, exact source
and launch snapshots, errors and canonical registry. Keep old tests closed.

## Implementation checklist

- [ ] Shared registered recipe, source/selection tests and isolated versioned PR.
- [ ] Actual admission and independent cell/dependence verification.
- [ ] Numerical qualification, controls and exact/sampled fits on supported cohorts.
- [ ] Sealed held-out evaluation and independent prediction reconstruction.
- [ ] Report/registry/publication and source release follow-through.
