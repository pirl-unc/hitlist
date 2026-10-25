# PXD018722 native response recipe and first biological screen

Prepared after canonical source replay 0519, before computing response targets,
choosing a reference peptide, assigning partitions or fitting a predictor. This
specifies the next recipe/fit under `native-psmd2-canonical-and-contrast.md`; it
does not replace the full precursor-relative APC availability objective. No new
training is authorized by a software-only test pass: the canonical source and
recipe must pass their required tested releases first.

## Hypothesis and observable

PSMD2 knockdown changes the relative pattern of products from the same parent in
a sequence-dependent way that transfers to other parent dependency components.
The direct endpoint is a conditional, observed native-extract signal contrast.
It is not an absolute intracellular amount or a claim of generalized APC behavior.

For eligible peptide i and a fixed same-parent reference j, compute each run's
`log10(raw_intensity_i) - log10(raw_intensity_j)`. The target is the mean over the
three KD runs minus the mean over the three control runs. Do not pair Cont1 with
KD1: numbered labels do not establish pairing. Produce one target per nonreference
peptide, retaining all six individual log ratios and twelve measurement IDs.

Shared parent abundance and run loading cancel within a run under a multiplicative
observation model. A fixed peptide-specific response cancels between conditions.
Condition-dependent extraction, suppression, modification mixtures and clearance
do not necessarily cancel. This is an explicit modeling assumption and a limit on
interpretation, not a new calibration claim. The three runs per arm do not imply
three independent biological subjects or justify a standard error divided by sqrt(3).

## Canonical recipe, with source dispositions

Implement `pxd018722_native_contrast_v1` through Hitlist's shared registered-recipe
interface, building on the released `native_peptidomics` source. Reuse generic
snapshot, manifest and verification helpers. No Presto-only source curator.
Record the exact source package digest, recipe version and source contract.

Admit only original peptide-sequence raw intensity, never LFQ, total intensity,
publisher duplicates, feature sums or a mixture of observation grains. Every
source peptide receives all applicable exclusion reasons, even if an earlier
reason already excludes it. Preserve all six source values and IDs in the audit.

Primary eligibility is fixed before any response summary:

1. Standard amino-acid sequence within the deposited 8–25 residue search support;
   source reverse and contaminant flags absent; finite PEP in [0, 0.01].
2. At least one deposited-reference origin; all origins literal, all explicitly
   human by original taxonomy, no engineered role, exactly one distinct complete
   parent sequence. Preserve multiple reference records or repeated spans of that
   same sequence; do not choose a preferred accession or location.
3. Every declared parent token resolves, and all of its complete reference
   sequences agree with the unique sequence above. Unresolved or contradictory
   parent claims are excluded from supervised examples but remain in the source
   dependency graph. This is stricter than the descriptive 0424 feasibility stratum;
   report the resulting loss of support without relaxing the policy.
4. All six acquisition-level raw values are finite and strictly positive. Zeros
   and missing values remain unqualified observations, not negative concentration
   labels. This conditions evaluation on observed peptides; it cannot establish
   recovery of the missing intracellular distribution.
5. A parent must have at least two eligible peptides. Choose its reference by the
   lowest SHA256 of `pxd018722-native-reference-v1|parent_sequence_sha256|peptide`;
   use no signal value, effect size or fitted score in reference selection.

Keep all original dependency components formed before eligibility. Require
consistency of one component per admitted parent; assign a split to the complete
component, never to a peptide or contrast. The graph already includes ambiguous
and excluded source claims. Store modified-form positive-support and matching-type
annotations for each example; these qualify the endpoint, not automatic exclusions
chosen after a result. Any further sensitivity cohort needs its own frozen policy.

Output source dispositions, admitted peptide contexts/quantities, parent/reference
choices, one contrast per nonreference peptide, source-wide component membership,
partitions, within-split weights and explicit interpretation assertions. Preserve
all origins and individual source links needed to recompute the target. Verification
replays eligibility, reference selection, component splits, target arithmetic and
weights from the pinned source, including all excluded peptides.

## Splits, dependence and support gate

Use integer SHA256 of `pxd018722-native-split-v1|dependency_group_id` modulo 20:
0–13 training, 14–16 validation, 17–19 test. Do not search seeds or reassign groups
to improve balance. Keep both endpoints of each shared-reference contrast in the
same component. Overlapping/repeated sequences and alternatives remain grouped.

Require at least 40 training, 8 validation and 8 test dependency components,
with at least 80/16/16 contrasts respectively. A failed support gate is a reported
source limitation, not permission to tune the split or include ambiguous parents.
The previous 81-component/228-contrast descriptive stratum is feasibility evidence,
not an expectation that this stricter contract must pass.

Within each split, give components equal total weight, parents within a component
equal weight, and contrasts within a parent equal weight. Acquisitions and repeated
reference peptides do not multiply independent-example weight. The study and the
six acquisition environments remain shared across all splits; this tests transfer
to parent components within a study, not specimen/study independence. Source audits
precede these partitions, so do not describe them as an untouched external blind
test or claim homology-disjoint transfer without separate evidence.

## Small direct model and prospective selection

Compare exactly three models: zero contrast; a log-length contrast; and a linear
sequence contrast. Use the existing parsimonious sequence feature definition from
family0143 (log length, composition and termini), pinning its exact source/hash in
the fitting registration. Do not add identity, parent abundance, species prediction,
MHC, TCR or unused assay heads. This subset has one species and one perturbation.

Every learned prediction is `g(features_i) - g(features_j)`, with no intercept.
Scale feature differences by their training-only weighted RMS, leaving zero-RMS
columns zero. Do not center the difference matrix: centering would introduce a
hidden intercept and break the antisymmetric contrast. Fit weighted squared error
plus alpha times squared coefficients, with weights summing to one.

Choose alpha from [0.01, 0.1, 1, 10, 100] using three inner training-only component
folds, assigned by a separately pinned hash namespace. Repeat transforms inside
each fold; select the smallest mean weighted loss, breaking exact ties in favor of
larger alpha. Refit on the full training split only. Validation assesses the frozen
conditions; it does not tune further features, thresholds, alpha bounds or source
eligibility. Evaluate all three fixed conditions on test once and retain all results.

Before fitting, register exact feature code, data/recipe/checksum, inner-fold rule,
numeric solver, environment, resource/runtime caps and output paths in a timestamped
experiment. Unit fixtures must test precursor/load cancellation, equal/opposite
contrast predictions, correct unpaired arm averaging, no shared-reference leakage,
training-only scales/alpha choice, weights and missing/zero handling. Independently
recompute fitted coefficients and every held-out prediction afterward.

## Evaluation, decision and route back to exact versus RL

Save validation and test predictions with source/parent/component IDs, target,
prediction and weight. Report weighted loss and RMSE, MAE, Pearson, Spearman,
prediction/target spread, signed-change accuracy with its denominator, and each
component's contribution. A 500 nM cutoff is inapplicable to this endpoint.
Use 2,000 dependency-component bootstrap draws with a preregistered seed for paired
RMSE-gain uncertainty; no acquisition-level bootstrap claiming biological replication.

Advance the physical representation comparison only if the sequence model improves
validation RMSE by at least 5% over both zero and length and its prediction standard
deviation is at least one quarter of target standard deviation. Call test transfer
supported only if the same gain holds on test and the paired component-bootstrap
95% interval excludes zero. Report weaker outcomes as uncertain/negative; do not
redefine success or tune the inspected test. Preserve zero and length references.

For a useful representation, the next independently registered family compares
exact fragment expectations and discrete complete-partition sampling under this
same contrast observation model. Both retain short/nonbinding/unobserved fragments
in the physical support; queried expectations need not materialize every possible
span. There is no supplied processing time in steady-state cells: exposure is
dimensionless unless separately calibrated. Specify baseline REINFORCE, clipping,
group standardization, sample reuse and KL objectives explicitly, with equal scoring
budgets, gradient error, recovery/error metrics and actual runtime. Distinguish an
exact-forward gradient hybrid from a simulator-forward model and quantify any
finite-sample bias introduced by log-of-expectation objectives.

This contrast constrains relative changes. Baseline abundance distributions,
clearance/time calibration, MS observation bias and professional-APC validation
remain required for the full availability goal. Passing this screen cannot replace
those requirements; failing it must change the next scientific hypothesis.

## Implementation boundary

Use native_contrast_schema.py for the fixed recipe contract and typed outputs,
native_contrast_compiler.py for pure source-table compilation, and native_contrast.py
for the shared materialize/verify API. Snapshot only required canonical tables with
independent copy-on-write files, retaining the original package manifest and source
assertions/material context. No new reference download or duplicate raw inputs.

Outputs: all-peptide dispositions, admitted parents, contrast examples, label-free
covariates, original measurement contributors and all-source components. One label
per peptide/reference pair, not nine cross-condition pseudo-replicates. Weight by
component, then parent, then contrast. A failed support gate stays in the summary
and prevents fitting; it does not invalidate a faithful source-decision package.

Tests must cover original quantity grain, zero/missing, parent ambiguity, reporter
exclusion, hash-based reference/splits independent of values, precursor/run/peptide
response cancellation, six-run averaging, lineage, weights, no-overwrite and tamper
replay. Complete format/lint/default suite, real-source recipe replay and the tested
release before using the output in a biological fit. Base is source PR839 commit
bfbaa67; revalidate main and bump from 2.0.37 to 2.0.38 before publication.
