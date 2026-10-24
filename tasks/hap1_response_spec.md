# HAP1 response recipe and biological baseline, frozen before fitting

## Question and boundary
Test peptide-dependent knockout response beyond condition and measured precursor
controls. This is relative MHC-I elution, not free-peptide concentrations. HAP1 is
not a professional APC. No inference of causal cleavage effects from all KOs.
Previously inspected MAPP/native/macrophage held-outs remain closed.

## Canonical Hitlist recipe
Registered recipe `hap1_machinery_response_v1`, revision
`shapiro2025_hap1_quantitative_v1`, pinned source manifest
88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8.
Use existing EvidenceStore materialize/verify/describe interfaces. Snapshot typed
source tables and source contract/manifest. Stream every supplementary quantity
cell into an explicit disposition. Preserve sources, metadata, all candidate
origins and contributor identifiers; no Presto source importer.

Primary response: one positive mmc6 KO peptide biological-replicate intensity
minus the mean log10 intensity of at least two positive WT replicates for the
same literal peptide. No control cross products, missing-value imputation,
synthetic negatives or bounds. B2M is background only, WT reference only. Any
peptide positive in any B2M replicate is excluded from primary responses. Keep
all original evidence and flag this as a conservative prespecified policy.
Only standard 20-residue sequences with candidate full-sequence groups each
supported by at least one human_target reference are admitted. Preserve decoy/
standard/nonhuman alternatives; unsupported groups defer the primary endpoint.
Noncanonical quantities are deferred until independent source replay. Peptide
identification is source-reported, not newly validated by this recipe.

Before eligibility filtering, union every mmc6/mmc7 literal peptide, all reported
parent claim tokens, every exact candidate full-parent sequence hash and deposited
protein inference group (leading Protein plus Indistinguishable Proteins). Literal
accession tokens only; do not merge isoforms by stripping suffixes. Component ID is
SHA256 of sorted node identities. Split SHA256(recipe_id+component_id) modulo20:
0..13 train,14..16 validation,17..19 test. No reshuffling or outcome-based balancing.
This does not establish independence of specimens/clones/studies or homologs.

Precursor covariates: source mmc8 MaxLFQ only. Map exact literal reference accessions
through deposited leading+indistinguishable protein groups. Each candidate parent
must have exactly one supporting measurement group; ambiguous or missing groups
are flagged and deferred for precursor-controlled fitting. Require >=2 positive WT
and >=2 positive KO cells per group. Never fill supplemental NA with deposited
positive quantities. Group response = mean(log10 KO)-mean(log10 WT). Keep every
contributor and all group membership claims. Distinct candidate parent sequences
receive equal weight; average their log10 group responses for the direct baseline.
Repeated groups are allowed across candidate parents, explicitly not independent
quantities. This is condition-level protein response, not paired aliquots or molarity.

Recipe exports all eligible elution responses with complete/missing precursor status,
components, candidate parents/origins, per-parent precursor inputs, all cell dispositions
and full source snapshots. Baseline cohort uses complete precursor responses for
ALL models. Within each fitted/evaluated subset, weights average condition, component,
peptide, then replicate. Do not transform targets or average target replicates.

## Initial experiment (development only)
Local CPU, float64 NumPy/SciPy, one numerical thread, no installs/cloud/Modal.
Five deterministic component folds on outer training only: SHA256 of
`hap1_baseline_inner_v1:`+component ID modulo5. Outer validation/test remain unscored,
reserved for a separately frozen physical exact-vs-sampled comparison. Do not load
outer outcome arrays into baseline fitting. No pretraining or biological synthetic data.

Controls: zero response; condition mean; condition plus fixed precursor coefficient1;
condition plus fitted nonnegative common precursor coefficient gamma (profile out
condition means). Fit gamma from weighted within-condition covariance/variance.
Direct sequence model adds condition-specific linear sequence effects to the frozen
fold's calibrated precursor offset. Features: length,20 AA fractions,20 N-terminal
and20 C-terminal one-hots. Equal hierarchical weights; train-only feature mean/SD,
omit SD<=1e-12. Center sequence and residual within condition so intercept is free.
For each condition solve weighted MSE + lambda*||beta||^2 with condition-normalized
weights; three fixed lambdas [.001,.01,.1]. No peptide/protein identity parameters,
optimizer, restart, epoch selection or hyperparameter changes. All transforms,
calibration and coefficients refit within each fold and full outer training.
If a fold lacks a condition needed in evaluation, stop with explicit support error.

Metrics: weighted MSE/RMSE(log10ratio), MAE,R2; pooled Pearson/Spearman; per-condition
and per-component errors; within-condition prediction SD. No nM/binder metrics.
Save all source-linked full-training/inner-validation predictions and coefficients.
Candidate development gate: best of the three fixed sequence penalties improves
pooled inner-CV RMSE >=2% versus best analytical control and weighted within-condition
prediction SD >.001 in every condition. Passing is feature-signal evidence and
selects a follow-up specification, not a blind test or biological qualification.
Failing means stop feature expansion on this inspected training cohort and examine
observation/source support before sampling. Report all conditions regardless of gate.

## Verification/resources/closure
Meaningful synthetic unit fixtures for reference/denominator/precursor ambiguity,
B2M exclusion, missing-vs-zero, component leakage, source-conservation and package
tampering. They are software tests, never biological training examples.
Independent source/response arithmetic and split/contributor verification before fit.
Independent saved-prediction metrics and coefficient reconstruction afterward.
Bound recipe build+replay <=900s and4GiB; entire baseline fit <=300s and2GiB. No guard
bypass; retain failed attempts, no changed biological settings after inspecting outcomes.
Freeze executed code/git dirty state/environment before every launch. Store summaries,
coefficients, per-example predictions, verification receipts and comparison figure in
experiment directory; update README/canonical log/tasks. Report limitations and source
release status separately from experimental numerical closure. Full APC goal stays active.
