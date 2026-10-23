# Soh digestion arithmetic and bounded time-course recipe (2.0.9 candidate)

Source: Soh et al. 2024, MSV000092631 / PXD044463. This review advances #775
under #770. Hitlist 2.0.8 is published from the exact clean-main release run
`39979099617` with both PyPI file hashes verified. The source package is the
independently reviewed 2.0.8 candidate
`/private/tmp/hitlist-v2-digestion-208-unit-candidate`, manifest SHA-256
`ad62414d511b74a6aa1e4da457fdff68d9f4527874d1a2e385c65e5b98df41e8`.
It has 251 hashed files. The recipe materialization reads that verified package
and never changes it. The portable candidate at
`/private/tmp/hitlist-digestion-209-review-candidate` has manifest SHA-256
`5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615`.
The candidate is local evidence, not a released training artifact.

## Independent source arithmetic

`python scripts/verify_digestion_arithmetic.py PACKAGE REPORT_DIR` reads the
original nested MS1 archive and source row text without importing Hitlist. It
replays precursor row reduction, raw peptide/run quantities, technical and
biological means, I/L coalescence, source mock zeros, baseline/noise subtraction,
censoring and final vector slots in all 17 conditions. All 17 pass: 545,664 raw
quantities and 280,664 final slots; maximum relative error over all checked
numeric layers is 9.30e-15. All 33 source files in the package match the
manifest, and the discrepancy log has zero rows. See
[arithmetic-summary.json](arithmetic-summary.json) and
[arithmetic-conditions.json](arithmetic-conditions.json). The 2.0.8
[original-source round-trip](../digestion-source-208-results/README.md)
independently checks the source ZIP/workbook/FASTA/RData decoding and typed row
correspondence. Together these establish original-input and quantitative
arithmetic fidelity; they do not establish the search, synthesis-error removal,
product-selection or biological interpretation as correct.

## Bounded recipe and conservation

The registered `digestion_contiguous_timecourses_v1` recipe uses the common
`EvidenceStore.materialize`/`verify_recipe_package` interface. It keeps every
one of the 53,413 final source vectors in 50,372 curve dispositions and embeds
the exact five input Parquet tables, original source manifest, all source slot
contributors, context and output hashes. Every source vector belongs to exactly
one disposition. The Hitlist-free
`scripts/verify_digestion_timecourses.py` recomputes every decision, I/L span,
zero-time support count, source status, target, source grouping and split from
the embedded source tables. It passed on all 50,372 dispositions and 30,566
admitted curves; see [independent-verification.json](independent-verification.json)
and [recipe-summary.json](recipe-summary.json). The source package manifest
remains the same SHA-256 after materialization. A forged target with a repaired
file hash fails both portable and independent verification in the focused test.

The endpoint is reported positive, background-adjusted biological-mean MS1
peak area at nonzero physical digestion time, in
`source_MS1_peak_area_arbitrary_units`. No new normalization, affinity or MHC
length filter is applied. Forced zero, absent/censored, annotation-only,
ambiguous mapping, spliced product and RF1 conflict rows retain their source
identity and explicit exclusion reason. An extra zero-time cardinality check
excludes one additional already-ineligible curve relative to the initial
recipe candidate; admitted counts are unchanged. The accepted panel remains
conditional on the deposited final table, not a complete set of potential
digestion products or prospective MS detectability labels. The paper's
additional positive terminal-time selection has not been independently
identified in the deposition and is not silently claimed for these targets.

Fifteen literal constructs define 15 frozen leakage groups; the three Ffh
ratios share one construct. The fixed split assigns nine groups to train, three
to validation and three to test, yielding 23,054 / 3,832 / 3,680 admitted
curves. Five of 17 conditions have no admitted curve: CaM, IF2 and UBE2S lack
the recipe's observed zero-time support; Ffh 1:25 lacks repeated common
positive times after that filter; RF1 retains its unresolved quantity versus
metadata time conflict. No excluded curve is turned into a negative target.

## Repeated-label temporal signal and limits

The independent verifier reports pairwise Pearson correlation across common
nonzero times after log10 transforming each curve's positive MS1 signal. Median
within-peptide temporal correlations are 0.951 for 11,527 train label pairs,
0.776 for 1,916 validation pairs and 0.960 for 1,840 test pairs. The fractions
with positive correlation are 0.891 / 0.837 / 0.975. These are descriptive
same-study repeatability checks on a source-selected panel, not held-out model
predictions. Before curve admission, the verifier also compares the two literal
filtered technical labels at their common positive times using log10(1+MS1
area). Median train/validation/test correlations are 0.958/0.904/0.922 over
40,400/5,199/7,542 nonconstant literal pairs; three further pairs are
constant. This is source-row repeatability, not verified acquisition pairing
or independent biological replication. There are only 15 independent
construct groups, no external
study, no demonstrated physical pairing of filtered technical indices, and no
known peptide-specific MS response factor. Do not interpret arbitrary MS1 area
as molar fragments or APC intracellular availability. Prospective fitting,
sequence-to-rate transfer and an observation likelihood remain #770 work.

## Software verification

`./format.sh`, `./lint.sh`, 10 focused recipe/independent-tamper tests, real
materialization, both independent source/recipe checks, strict docs, the
wheel/sdist plus Twine/license checks and the normal full test runner pass.
The full run has 3,351 passed, 84% coverage and 10 existing warnings. Hosted
final-head unit/integration CI, merge and exact PyPI publication remain release
gates. The local review output and this report are not PyPI publication claims.
