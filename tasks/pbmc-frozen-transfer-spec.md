# Prospective frozen A549-to-primary-PBMC captured-signal transfer

Codex/GPT-6, October24,2026. Prior original-source counts/headers/search parameters
are known; no PBMC endpoint cohort or model performance has been inspected. The
2244A549test is closed. This family makes no parameter updates.

## Hypothesis and scope

The modest sequence signal learned from A549baselineMAPP may transfer to a new
study of primaryPBMCs. Compare the identical frozen exact-gradient and discrete
REINFORCE-trained models to uniform and the frozen direct ridge controls. PBMCs
are a mixed immune-cell population, not a purified APC subtype. Captured observed
signal is the endpoint; no free-pool, molarity, missing-mass or disease inference.
Source publicationPBMC versus repositoryblood-serum claims remain explicitly
conflicting. Clinical source labels are not physical specimen or aliquot IDs.

## Source qualification and registered cohort

OriginalPXD009438 assets already present in /private/tmp/hitlist-pxd009438-source-20261024/assets.
Bind all seven assets to mergedHitlistPBMC receipts and verify original archive
SHA256/SHA1. Use the shared source adapter and EvidenceStore/registered recipe;
no private Presto source curator. Preserve original985peptide rows,8851evidence
rows and all13790peptide/run decisions. Independently replay original source fields,
run aliases, quantities and metadata after materialization.

Primary cohort: literalCTRLsource acquisitions only. Secondary: literalLUPUS
source acquisitions, separately weighted and reported. No clinical normalization,
participant joining or cross-modal/precursor pairing. Per-cell eligibility:
positive finite original reported-run intensity; canonical sequence; explicitly
blank original peptide reverse/contaminant flags; peptidePEPin[0,.01); per-run
Identificationtype exactlyByMS/MS; at least one positive direct-MS/MS evidence;
all linked evidence intensities reported(numeric count equals source count);
exactly one distinct supplied parent accession. Human is project/search scope;
per-parent organism origin and full original Human_UP_2016_02FASTAremain unverified.
This differs fromA549's source-header human-only axis and must be disclosed.

Within accession/acquisition, exclude every otherwise eligible row of a duplicated
sequence rather than selecting by quantity/confidence or summing ambiguous rows.
Retain groups with>=3distinct remaining sequences. Graph all original sequence/
parent labels, including excluded and both clinical-label groups, before filtering.
No aliases, homology or exact-specimen identities inferred. Components define
weights and uncertainty; there is no new train/validation partition because every
model was frozen on the separateA549study. This is external test only.

Require>=20components,>=20groups and>=100measurements inCTRLbefore primary
evaluation. The secondaryLUPUScohort uses the same support gates independently.
Preserve failures; no resplitting, threshold relaxation, source pooling or refit.

Targets: originallog10intensity minus observed group's unweighted mean, and
intensity divided by its group total. Equalcomponents, equalgroups withincomponent,
equalpeptides within group for error. Normalization conditions on the observed set.
Peptide totals can aggregate forms; no frozen detector correction is applied.

## Frozen predictors and sequencing

Freeze2244models.json and input/prediction seals before cohort admission. Use only
uniform, validation-selected length_ridge_0.001 andsequence_ridge_1, exact, and
reinforce_42/43/44. Original61features, means/scales and supportedSVDprojections/
coefficients are unchanged. Center each target group's transformed inputs/scores;
softmax(ln10*score) gives target observed-set fractions. No output intercept, new
normalization calibration, model choice or prior outcome use. All other Presto
branches remain untrained in this family. No additional sampling during inference.

Admission writes label-free inputs and separately sealedCTRL/LUPUSlabels. Predict
all eligible covariates, hash predictions and model fingerprints, then evaluate
labels once. Independent NumPy/SciPy versus Torch inference and archived-byte
metrics/prediction reconstruction;1e-10prediction/coefficient/continuous metric
tolerance. Rank metrics recomputed from exact archived prediction values.

Primary gate: sequence-vs-length RMSEgain>=5%, positive sequenceR2and nonconstant
outputs inCTRL. Distribution gate: each exact/RLmodel beats uniform reverseKL in
CTRL. SecondaryLUPUSmetrics descriptive. No test-selected model/seed or ensemble.
Report weightedRMSE/MSE/MAE/R2/SD andreverseKL, unweightedPearson/Spearman,
per-acquisition/component strata, primary pairedcomponentbootstrap2000resamples
seed20261024, and feature-range/source-peptide/accession overlap. At most these
predeclared seven models are evaluated. Null correlations for uniform. No nM or
binder metrics; no synthetic biological labels, pretraining or optimization.

## Resources and closure

LocalCPUone numerical thread,4GiBRSS,20minutes per stage,1hourfamily. Read-only
source audit and boundedCoWappend may add<=128MiBnew physical bytes while keeping
a hard1GiBfree-space floor. Existing3.8GiBreference generation is snapshotted with
verifiedCoWsupport; all original paths remain independent. Parent verification is
read-only. Monitor disk/RSS/time, abort and preserve receipts if a cap fails.
Do not run the full local suite below the separately registered2GiBheadroom gate.
No cloudtraining, installations, newRAWdownload or deletion of other work.

Shared recipe ships through Hitlist with source/support/status uncertainty intact.
Before closure preserve code/config/env, source and model hashes, all support/
qualification failures, predictions, metrics, overlap, runtime and independent
verification. Update canonical experiment log and existing#822/#825as appropriate.
Full APC concentration/availability goal remains active regardless of this narrower
transfer result. An external transfer failure ends this frozen family without
adapting to these inspected target labels.

## Implementation details frozen before admission

Component IDs are SHA256 of `pxd009438_composition_v1:` followed by compact,
sorted-node JSON (node strings `peptide:<literal sequence>` and
`parent:<literal supplied accession>`). Group IDs are `pxd009438:composition-group:`
followed by SHA256 of compact JSON `[parent_accession, channel_id]`. No split
assignment is made; cohort is the literal CTRL or LUPUS label and split is test.
Rows are stably sorted by measurement ID inside groups. Counts/weights are computed
separately per cohort even when a component occurs in both cohorts. Recipe outputs
include all decisions, all source component nodes, groups, examples and label-free
covariates; source flags, acquisition material precision and parent-origin
uncertainty remain attached. The source records/table snapshot and source generation
manifest are bound into the recipe envelope, and output verification replays the
complete derivation. This freezes identifiers/ordering, not a new eligibility rule.
