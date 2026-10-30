# Serial build optimization: 2.1.0 to 2.1.1

The serial scanner and contributor-publication changes in 2.1.1 reduce mean full
build time from 1,647.49 to 1,498.19 seconds (27.5 to 25.0 minutes): 9.1% less wall
time, or a 1.10× speed ratio, across four alternating runs on one allocation.
CPU time falls 8.9%. All seven Parquet artifacts are byte-identical across every
run. Mean parent peak RSS rises from 30.70 to 30.98 GiB; this is not a memory
optimization. The 2× whole-build target in
[#63](https://github.com/pirl-unc/hitlist/issues/63) remains open.

The [earlier performance report](reliability-performance.md) describes the changes
already shipped in 2.1.0. Its separate-host and uncontrolled-network timings are
historical evidence and are not combined with this experiment's speed ratios.

## What changed

CSV capture now encodes each source cell once and reuses that encoding in the two
existing JSON representations. The scanner binds the column projection once per
file and reuses the projected fields. Contributor publication builds Arrow columns
from bounded tuple batches and caches the small set of relationship JSON strings.
Nested and typed supplementary payloads retain the general JSON encoder.

The CSV encoder does not cache source values across rows. Existing scratch, batch, decoded-cache
and atomic-publication limits remain. The changes preserve source cells, stable
identifiers, row order, curation flags, schemas and contributor relationships.

## Controlled full-build method

The baseline is released 2.1.0, commit
`5f988fc01d7c0470fb369cc7f73f154a76f63ceb`. The measured candidate's production code
is commit `2b850a39a4c0cfa814eb12b852cc742576d5d287`. Producer archive hashes,
installed versions, instrumentation hashes and individual measurements are retained
in `tasks/full-build-63-results/`.

All four full builds run on one non-preemptible allocation in baseline, candidate,
candidate, baseline order. Each uses a fresh process and output/cache directory,
the same pinned dependencies, eight allocated CPUs, 64 GiB RAM, 512 GiB scratch,
two mapping workers and single-threaded BLAS/OpenMP settings. Both producers scan
serially; the harness's historical `requested_scan_workers=4` field has no effect
on either producer. The whole experiment has a four-hour deadline and no retries.
The full-build image uses Python 3.12.1, pandas 3.1.0 and Arrow 25.0.1. The CPU-model
probe returned unknown; the comparisons share the actual allocation, not merely
matching resource requests.

Inputs are the same 14,719,407,889 bytes of IEDB/CEDAR exports, 26 packaged assets,
supplementary inputs and pinned human Ensembl 112 references used in the earlier
report. Preparation verifies/copies inputs and builds the raw reference index
outside timing. Timed forced builds use warm source/reference files and cold
Hitlist mapping indexes. This is the complete pinned raw-source build with limited
mapping-reference coverage; it is not a complete multispecies-reference benchmark.

An inherited Python audit hook rejects socket connection and DNS attempts during
each timed build, including spawned Python interpreters. A caught attempt still
invalidates the measurement. This is a Python network audit, not an OS firewall.
Input preparation and the test suite are outside that audit's scope.

CPU totals include the parent and exited/waited children. Parent peak RSS is a
process-lifetime high-water mark. Stage memory is sampled parent RSS; process-tree
sums can count shared pages repeatedly. Nested stage measurements must not be
added to their parents. Joining and generation publication also contribute time
outside the explicitly instrumented stages.

## Full-build measurements

The unprofiled experiment `issue63-full-pair-1` completed successfully. Dependency
and input pins match across all runs; all four timed builds have installed Python
audit markers and no attempted-access logs. These are two repetitions per variant,
not a hardware-independent performance guarantee.

| Run | Variant | Wall (s) | CPU (s) | Parent peak RSS (GiB) | Sampled tree peak RSS (GiB) |
|---|---|---:|---:|---:|---:|
| 1 | Baseline 2.1.0 | 1,638.90 | 1,643.62 | 30.73 | 36.47 |
| 2 | Candidate 2.1.1 | 1,485.89 | 1,492.63 | 30.96 | 36.85 |
| 3 | Candidate repeat | 1,510.48 | 1,510.55 | 31.00 | 37.07 |
| 4 | Baseline repeat | 1,656.08 | 1,653.80 | 30.68 | 42.73 |

The historical tree-RSS columns in this report are retained for audit, but are not qualified
for memory comparisons. [#876](https://github.com/pirl-unc/hitlist/issues/876)
subsequently demonstrated that this runtime can list child thread IDs as descendants;
the original sampler counted the same address space repeatedly. Parent-process and
waited-child high-water marks, timings and output hashes are independent of that bug.
New measurements deduplicate thread-group IDs.

The parent-memory increase is about 0.28 GiB (0.9%). Tree-memory measurements vary
more and include shared-page double counting; they do not establish a memory win.

The following stage figures are means of the two runs for each variant. RSS is the
mean of each stage's sampled parent peak, not a memory allocation delta. Smaller
nested Parquet writes are retained in the machine-readable receipts.

| Stage | Baseline wall (s) | Candidate wall (s) | Baseline CPU (s) | Candidate CPU (s) | Baseline RSS (GiB) | Candidate RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| IEDB scan | 537.56 | 470.63 | 538.18 | 474.23 | 23.96 | 24.21 |
| CEDAR scan | 453.44 | 398.61 | 452.78 | 397.00 | 30.54 | 30.98 |
| Contributor publication | 216.39 | 179.71 | 213.59 | 177.48 | 21.35 | 22.08 |
| Protein mappings | 120.58 | 119.49 | 121.83 | 120.69 | 23.06 | 23.92 |
| Bulk proteomics | 29.85 | 28.92 | 29.80 | 28.88 | 23.42 | 24.09 |
| Supplementary scan | 23.33 | 23.70 | 23.10 | 23.59 | 20.84 | 21.55 |
| Line expression | 0.38 | 0.38 | 0.37 | 0.36 | 23.40 | 24.07 |

The two source scans and contributor publication remain the largest measured
stages. The uninstrumented joining and generation work must remain explicit when
planning the next optimization; the named stages do not explain every second.

## Why scanning remains serial

A six-run, same-allocation diagnostic compared a 500,000-row source prefix in
baseline, serial, parallel, parallel, serial, baseline order. Both output files
were byte-identical across all six runs, dependencies matched, and the Python
network audit recorded no attempts.

| Variant | Wall times (s) | Mean wall (s) | Mean CPU (s) |
|---|---|---:|---:|
| Released baseline | 69.465, 68.568 | 69.016 | 66.755 |
| Serial encoding/projection | 60.036, 59.855 | 59.946 | 57.390 |
| Parallel annotation/capture | 65.806, 65.205 | 65.505 | 103.335 |

The serial candidate lowered mean prefix time by 13.1%. Parallel capture was
slower than that simpler path and substantially more expensive in CPU time, so
the worker implementation was removed. These prefix measurements do not establish
a full-build speed ratio.

## Output verification and exclusions

The final verifier re-read every retained output file, checked its actual hash and
size against the build receipt, and confirmed byte identity for all seven Parquet
artifacts in all four runs. It also validated generation source/output receipts and producer
versions. Only named location, timestamp and producer-version metadata differences
are allowed and individually recorded. The deterministic README is reconstructed
and hash-verified; the original README was not retained by the benchmark harness.

| Artifact | Rows | Columns |
|---|---:|---:|
| Observations | 4,486,991 | 71 |
| Binding | 814,268 | 71 |
| Other assays | 26,537 | 71 |
| Contributor links | 9,381,101 | 11 |
| Protein mappings | 6,205,013 | 12 |
| Bulk proteomics | 2,269,288 | 57 |
| Line expression | 42,909 | 18 |

## Test-suite measurements

The separate experiment `issue63-test-pair-1` completed all four default test.sh
runs on one allocation with Python 3.12.16, identical dependencies, fresh caches
and two admitted workers under the unchanged 2.5 GiB per-worker guard. Baseline
runs pass 3,734 tests; candidate runs pass 3,744. Every run has two skips and ten
expected warnings. The candidate includes ten additional regressions.

| Run | Variant | Script wall (s) | CPU (s) | Waited-child peak RSS (GiB) | Sampled tree peak RSS (GiB) |
|---|---|---:|---:|---:|---:|
| 1 | Baseline | 260.70 | 504.15 | 0.534 | 10.92 |
| 2 | Candidate | 264.04 | 508.78 | 0.537 | 11.03 |
| 3 | Candidate repeat | 253.70 | 489.93 | 0.538 | 10.98 |
| 4 | Baseline repeat | 252.11 | 484.76 | 0.534 | 10.93 |

Mean wall time is 256.41 versus 258.87 seconds (candidate 1.0% higher); mean CPU
is 494.46 versus 499.36 seconds. This does not establish a suite speedup. The
waited-child high-water mark and sampled tree sum are different memory metrics;
neither should be substituted for the other or described as physical RAM usage.

The test candidate is `ff604fd2f2f5bb4d8054ce98b24ccb3eac7525fa`: identical production
code to the full-build producer, with benchmark cleanup and its regression fixture
hardened. Local test.sh separately passes 3,745 tests (one skip).
Both measured production and harness revisions pass the hosted Python/pandas
matrix, 97 integration tests and the release-build gate. Their verification
receipts are separate from these paired suite measurements.

## Excluded attempts and cancellation diagnostics

The first scan experiment failed in the harness's mixed-PMID Arrow conversion;
the corrected projection matches the builder and preserves every original cell
in contributor data. Two later experiments were interrupted and are excluded from
repeatability claims. Their initiating cancellation cause remains unknown;
[#869](https://github.com/pirl-unc/hitlist/issues/869) retains the evidence.
Background submission now uses a durable function-call ID, checkpoints, bounded
child cleanup and failure receipts. A synchronous remote call can be cancelled
after its caller disconnects even when the app is detached; this is a plausible
mechanism for those interruptions, not proof of their initiating actor.
See [Modal's invocation documentation](https://modal.com/docs/guide/function-invocation-methods#durability).

## Reproduction

Create both producer archives from their exact commits using
`git archive --format=tar.gz`. From the instrumentation checkout, with the pinned
requirements and authenticated Modal in an isolated environment:

```sh
HITLIST_BENCHMARK_BASELINE_ARCHIVE=/path/to/baseline.tar.gz \
HITLIST_BENCHMARK_ARCHIVE=/path/to/candidate.tar.gz \
  python -m modal run --detach scripts/reliability_modal.py \
  --label UNIQUE_LABEL --paired --background
```

Retain the printed function-call ID and query its completion. A successful local
submission is not a completed benchmark. `experiment.json` remains unsuccessful
until all four builds, dependency checks, network audits and output comparisons
finish. Full artifacts are retained in the project's `presto-data` volume under
`hitlist-reliability-20261028/<run-label>/`.

For the separate test-suite experiment, set `HITLIST_BENCHMARK_TEST_IMAGE=1` and
replace `--paired` with `--test-pair`. This uses a Git-equipped Python 3.12 Bookworm
image, the same alternating order, fresh caches and `TEST_SH_MAX=2` with unchanged
memory admission. Do not run another benchmark concurrently on the allocation.
