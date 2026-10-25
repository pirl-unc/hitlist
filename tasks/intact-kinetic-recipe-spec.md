# PXD058121 measured kinetic forecast recipe

Implement `pxd058121_product_substrate_forecast_v1` through the existing registered
EvidenceStore recipe interface, without changing historical source packages.
This follows the frozen Presto `intact-kinetic-forecast.md` model specification.

Decision universe: every source assignment, including outside-kinetics views.
Retain source row, quantity, acquisition/context and span identities; unknown
physical material/replicate confidence remains explicit. The recipe selects the
F2 kinetic view once, positive finite quantities with valid substrate spans,
one reported/replayed assignment, mass error <=10 ppm, explicit unmodified
fields, and exact input-mass intensity agreement. All rows in residual duplicate
sequence/acquisition groups are excluded. No best-quantity choice or sum.

Use a unique admitted full-substrate row from the same acquisition as denominator;
missing/ambiguous denominators exclude that acquisition's products. Parent rows
remain explicit denominator-only decisions. Targets are log10(product intensity)
minus log10(parent intensity); do not divide first or manufacture zero labels.
At least two distinct admitted fitting times are required per product sequence,
based only on0/20/40/60/80minutes.100minutes is validation;120minutes is test.
Missing late values stay missing. Equal peptide/time/replicate-cell weights within
each split. Support requires20product sequences/40observations on each held-out.

Export all decisions, admitted examples, a label-free covariate table and
trajectory-support records; retain exactly all source tables used to compile
these. Source manifest/contract/table digests bind the output. Verification
recompiles from frozen source tables, checks summaries and every typed output,
and rejects changed target/weight/disposition values even after rehashing.
Keep failed staging/replay evidence. Refuse existing destinations.

Register recipe discovery, materialization and verification; version2.0.34 with
catalog32 unchanged. Meaningful tests: within-acquisition denominator linkage;
missing/ambiguous normalizer exclusion; no late-only trajectory eligibility;
training-only minimum support; exact targets and hierarchical weights; no label
fields in covariates; duplicate masses, replayed ambiguity and source conflicts;
unsupported cohort retains reasons and fails support; rehashed tampering fails.
Independent original-source/target/split replay in Presto precedes training.

Run format/lint and focused recipe/interface tests. Full local suite remains
below the2GiBfree-space requirement; require hosted full checks before merge and
verified release. No data fitting or biological performance is asserted here.
