# Contributor storage and scan cost — #63, second pass

## Objective and baseline
Continue the open 2× full forced-build target after 2.1.1/2.1.2 shipped a measured
9.1% reduction. Start from main ff5247f (2.1.3), which also includes #872's
supplementary correction. Compare both variants from this same base so scientific
output changes from another PR cannot be mistaken for performance regressions.
The original checkout and shared environment remain untouched.

## Investigation and candidate selection
The retained sampling profile attributes most scan overhead to lossless provenance
capture (JSON construction and SQLite records/assay/edge inserts), followed by CSV
parsing and cached row annotations. Contributor publication additionally spends time
on bounded graph traversal, payload decoding and Arrow conversion. Refresh a profile
on current main, then test isolated changes on identical real source prefixes.
Evaluate bounded per-column encoding reuse, reduced scratch-index/graph overhead,
avoiding repeated pure annotation setup, and replacing retained per-row dictionaries
with compact tuples while preserving DataFrame inference and first-seen columns. Prefer a simple, demonstrated gain;
reject cache growth, deferred constraint failures, or worse CPU/memory tradeoffs.
Use small focused measurements to select a candidate before four full builds.

## Invariants
Preserve all input cells and exact original-fields/raw-row JSON strings, source
record identifiers, missing/short-row handling, Unicode and escapes, output order,
row-group boundaries, schemas, relationships, cycle and multiple-donor semantics.
Reject duplicate source identities at record time. Keep bounded scratch space,
byte/row capture/output limits, decoded-cache limits, disk reserve checks, cleanup,
and atomic publication. Preserve all curation and scientific classification rules.
Do not weaken force-build behavior, reference cache policy or network audits.

## Verification
- Regression fixtures target affected public behavior, including oversized rows,
  duplicate identities, graph branches/cycles and storage exhaustion.
- Compare released and candidate real-prefix outputs byte for byte with matching
  dependency/input pins and fresh processes on one allocation; record CPU/RSS.
- For a selected candidate, run baseline/candidate/candidate/baseline complete forced
  builds, using the same warm pinned raw/reference files and cold derived indexes.
  Independently reread all retained outputs and validate metadata allowances.
- Re-measure test.sh separately. Preserve ordinary memory admission and record
  optional-data/dependency skips without attributing them to platform.
- Run format.sh, lint.sh, test.sh, pip check and editable metadata validation.
  Bump the version, pass final-head CI, merge via PR, deploy exact clean-main tested
  artifacts and verify both actual public PyPI downloads.
- Keep #63 open unless evidence establishes its target; record remaining measured
  stages and continue investigation while useful work remains.

## Review
Pending current-base profiling and candidate selection. Prior completed evidence
remains in docs/serial-build-performance.md and tasks/full-build-63-results/.

Current 500,000-row profile: 66.33 s scanning (19.18 s capture, 11.14 s CSV
iteration, 10.32 s scanner self, 5.30 s assay annotation, 4.51 s DataFrame
construction) and 13.88 s contributor publication. Local bounded diagnostics
support testing consecutive-cell JSON reuse and WITHOUT ROWID record storage;
the wider covering edge index used more scratch and had no clear gain. Keep the
existing edge index. Tuple buffering is a memory reduction hypothesis to measure.

## Independent-source diagnostic before architectural changes

The first same-allocation 500,000-row ABBA comparison passed exact byte checks:
58.96 s baseline versus 54.77 s candidate mean, with parent peak RSS 2.77 versus
2.26 GiB. This is a prefix result, not a full-build ratio. The lower row-buffer
memory makes independent source scans worth evaluating before adding concurrency.

Run a separate, bounded diagnostic over pinned IEDB and CEDAR prefixes. Each
source has its own fresh Python process, capped collector and output files;
compare serial source scheduling with two simultaneous children in alternating
order. Include subprocess startup, capture, publication and output writes in the
clock. Retain dependency/input pins, CPU and sampled tree memory, network-audit
markers, child completion, and actual output hashes. No production concurrency
until this proves useful; the diagnostic deliberately excludes the additional
cross-source graph/partition merge a production implementation would require.
If selected, write a separate resource/merge design before implementation: preserve
one total scratch budget, per-source duplicate identity, deterministic partitions,
source snapshots, worker cancellation, and atomic failure of the whole build.
