# Shared contiguous digestion time-course recipe

Prospective admission contract, 2026-10-23; source arithmetic family1057 passed
all17 conditions. This is a candidate named recipe, not a biological fit.
Use the common Hitlist EvidenceStore/registered-recipe API. Isolate from ongoing
PR773 review edits; keep old source package/producer pins and later reconcile
any upstream source-schema update without silently relabeling the generation.

## Endpoint and source semantics
Recipe ID: digestion_contiguous_timecourses_v1. Target: original positive,
background-corrected final MS signal, in arbitrary units, at a physical nonzero
digestion time. All source states and reasons stay queryable. Never claim
intracellular/molar concentration or infer negatives from missing products.
No new intensity normalization in the data recipe. Output raw targets, exact
parent sequence/span, known time, source condition/protocol and biological
replicate label, plus original final-vector/slot contributors and raw/filtered
baseline evidence. Filtered technical indices are source row-number assignments;
they are not assured paired acquisition identities.

## Eligibility (freeze before curve outcomes)
- Exact canonical20 substrate and peptide; no additional MHC-length/affinity cut.
- Source productType must be PCP, with no splice alternative. Independently find
  every I/L-equivalent contiguous occurrence in the literal construct; require
  exactly one. Retain all mappings and exclusions in a disposition ledger.
- Group peptide aliases by I/L-normalized sequence within condition. Never
  multiply quantities by alias, annotation, mapping or repeated vector slot.
  Collapse duplicate time values only when numerically consistent to1e-10
  relative/max(1,abs(value)); disagreeing aliases exclude the affected curve.
- Preserve original final-vector selection as a source condition. This recipe
  does not independently validate the original search/synthesis-error removal.
- Exclude RF1/unresolved quantity-metadata time conflicts at condition scope.
- Require actual zero-time values in the filtered source for each admitted
  peptide/biological/technical trajectory; do not accept postprocessor mock0.
  These are observed filtered baselines, not guaranteed substrate-only controls.
- Do not turn forced final time0 into a loss target. Nonzero-time absent/NaN,
  nonpositive or nonfinite values stay exclusions/missing states.
- Retain quantitative trajectories only when at least3 distinct positive times
  are available and at least2 source biological labels share at least3 positive
  times for the peptide/condition. Expose other times/missingness, do not fill.
  The first repeated-curve diagnostic uses the declared common positive times.
- Keep Ffh ratios distinct conditions with measured protocol context, sharing
  one construct group. Do not invent residual enzyme activity or APC state.

## Grouping and portable export
Before outcome eligibility, connect all constructs sharing any I/L-normalized
source final peptide, including excluded/annotation-only bridges, and identical
full constructs. Fixed5fold assignment: sort component hashes lexicographically
and assign round-robin folds0..4. Fold0=test,1=validation,2..4=train for first
prospective diagnostic. No salt/seed search; report empty/small partitions as a
failed support gate, not a reason to resplit. This limits exact product/parent
leakage, not homology or study/specimen independence. All conditions and biological
labels for a construct remain together. Do not inspect held-out temporal fit or
repeatability before a fitting protocol is frozen.

Exports: contract, source manifest, components, all curve dispositions/reasons,
source-linked positive observations/trajectories, canonical query tables,
summary counts by condition/split and recipe manifest hashes. Include selected
subset and missing values explicitly; no outcome-based holdout reassignment.
Reuse canonical materialize/verify infrastructure and independently rebuild
from original source identities to verify targets, duplicate handling, groups
and exclusions. Preserve the typed source and prior recipes.

## Observation model for later fitting (not launched here)
First test conditional temporal shape: a supplied measured reference-time value
calibrates each peptide/replicate's unknown MS response. Model expected fragment
copies are unnormalized mass-conserving independent-cut expectations. Predict
ratios across time from the same peptide and declare the supplied reference
measurement at evaluation; this is not zero-shot abundance prediction.
Log-ratio residuals avoid domination by arbitrary MS amplitudes, but nonlinear
Monte Carlo plug-in objectives are biased. Freeze exact, complete-partition
score-estimator and explicitly named exact-denominator/objective hybrid variants
only after training-support/signal/gradient probes. Report finite-sample bias and
rare-event variance; do not call a hybrid an RL-only experiment or claim score
savings from a full exact objective. Include no-change, shared-time and constant
cut-rate controls; assess low-rate identifiability/variation before interpreting
biological rate parameters. Sequence-to-rate function must be parsimonious and
coverage-audited. Full APC/machinery/transport/clearance/MS calibration remain.
