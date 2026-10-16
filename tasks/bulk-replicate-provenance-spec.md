# Source-backed bulk experiment counts — #699

## Problem and evidence

The general bulk peptide API estimates the number of replicates from the maximum
observed peptide detection count. A maximum does not establish the experiment
inventory. Its current fixtures encode that assumption. The existing curated
PXD004452 scope registry records exact MaxQuant experiment labels, acquisition-run
counts, summary.txt SHA-256 and source links. It explicitly marks the HeLa 46-fraction
pool unresolved because legacy aliases collapsed distinct experiments. Experiment
labels and fraction/acquisition runs do not establish biological replication.

## Design

1. Project experiment counts from the existing reviewed scope registry. Reuse its
   exact fractionation protocol mapping, including legacy pH evidence, and validate
   unique scope IDs and experiment-list cardinality. Never infer totals from observed
   detections or a user-filtered subset. Do not change the registry's scientific claims.
2. Preserve original numerator/denominator fields as reported values. Add explicit
   n_experiments_possible, n_experiments_detected, experiment IDs, acquisition-run
   counts, an unknown biological-replicate count, count unit, source reference/hash,
   and status. Keep n_replicates_possible/detected as documented compatibility names
   for the experiment counts. A known inventory of three pooled experiments can
   coexist with an unknown per-peptide numerator and unknown reproducibility fraction.
3. Available, finite, integral counts within the curated total are usable. Unresolved
   aggregate/scope, missing, non-integral, negative or oversized counts stay unknown
   with an explicit status. A supplied min_reproducibility must be finite in [0,1]
   and exclude unknown fractions even at zero. Omission preserves all observations.
4. Apply the same projection on CSV fallback, old/new parquet queries and new bulk
   parquet builds. Keep non-Bekker-Jensen evidence outside this curation and reject
   unsupported evidence for reproducibility filtering. Preserve row order, identities,
   non-count quantities, pH provenance and raw CSV bytes. Keep the existing scoped
   PXD013455 dataset, historical artifacts and detectability fitting unchanged.
5. Update docs and source-valid fixtures, bump 1.81.0, format/lint and run meaningful
   regressions plus full CI/integration/release gates. Review, merge, run deploy.sh
   from clean main and independently verify the public PyPI distribution.

## Actual evidence deliverable

Freeze a clean data-only recipe; stream the pinned complete packaged peptide CSV
through datacache and publish the annotated table, per-scope counts, filtered-row
counts, original/curated numerator and denominator comparisons, citations, registry,
code, hashes, and row/non-count preservation receipt. Verify public downloads and
independently recompute curated totals and valid fractions from the registry and
reported counts. Recover only the small pinned original summary.txt member via
datacache range reads and independently check experiment labels and run inventory.
No new model fit or biological-independence claim.
