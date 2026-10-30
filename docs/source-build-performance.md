# Source scanning and contributor storage

This continues [#63](https://github.com/pirl-unc/hitlist/issues/63) from released
2.1.3 (`ff5247f`), after the [first serial optimization](serial-build-performance.md).
The controlled full-build comparison reduces mean wall time from 979.99 to
680.47 seconds: **30.6% less time, or 1.44× faster**. All seven Parquet outputs
remain byte-identical. The issue's 2× whole-build target remains open.

## Candidate changes

The scanner retains row tuples instead of one dictionary per row. It reuses an
assay annotation only when the immediately preceding row has the same four input
fields. Contributor capture reuses the previous row's encoded cells within its
existing byte cap; oversized rows clear that cache. The disposable SQLite records
table uses its primary key as its storage index. Original source cells, row order,
identifiers and scientific classification rules remain unchanged.

Serial results also construct owned empty columns: a zero-row Arrow view can
otherwise retain its source’s full string buffers. This corrects
[#879](https://github.com/pirl-unc/hitlist/issues/879) while preserving schema and
metadata. The parallel IPC path already transports empty schemas without those
source buffers.

The builder can scan IEDB and CEDAR in two independent processes. It consumes their
results in the existing source order and uses the same partition normalization as
the serial path. Each child writes private Zstandard-compressed Arrow tables and a closed
contributor graph. The parent imports payload blocks, records and edges before existing
cross-source deduplication and generation publication.

`HITLIST_SCAN_WORKERS=1` selects serial scanning. Larger values are capped at two
source processes; the default is two, with `HITLIST_BUILD_WORKERS` as a fallback
cap. Two sources, two available CPUs and conservative memory admission are all
required. Unknown memory selects serial scanning. The current admission threshold
is 28 GiB per source plus 4 GiB parent headroom, increased for source files larger
than the qualified inputs. Available memory is capped by the container's remaining
allocation, including the process’s resolved nested cgroup and visible ancestor
limits. CPU affinity and ancestor quotas also cap admission. This is a capacity
estimate, not a memory reservation against other applications. The qualified
full-corpus runs below retain this conservative threshold.

The existing contributor scratch budget covers the parent graph, both child
graphs, IPC files and bounded worker metadata together. SQLite page limits and
aggregate IPC write limits enforce separate reservations. The parent reclaims a
reservation only after removing its files. If a child exhausts a reservation before
any import, the builder cleans up both children and explicitly retries serially.
Other errors abort publication. Existing published data remains intact.

## Controlled full-build comparison

The complete `a71f2e9` candidate and baseline `ff5247f` run in
candidate/baseline/baseline/candidate order on one eight-CPU, 64 GiB allocation.
Both candidates select two source workers, with no serial retry. Input hashes,
dependency versions and network-audit checks match the recipe below. Every timed
Python interpreter has an audit marker; no attempted-access logs were recorded.

| Run | Variant | Wall (s) | Aggregate CPU (s) | Parent peak RSS (GiB) | Sampled tree peak RSS (GiB) |
|---|---|---:|---:|---:|---:|
| 1 | Candidate | 677.23 | 898.15 | 10.82 | 32.98 |
| 2 | Baseline | 978.20 | 982.39 | 31.04 | 31.04 |
| 3 | Baseline repeat | 981.77 | 991.87 | 31.05 | 31.05 |
| 4 | Candidate repeat | 683.71 | 908.98 | 10.95 | 33.92 |

Mean aggregate CPU falls from 987.13 to 903.57 seconds (8.5%). Mean parent peak
RSS falls from 31.05 to 10.89 GiB, but work moved into children: mean sampled tree
peak rises from 31.05 to 33.45 GiB (7.7%). A lower parent peak therefore does not
establish lower total memory. Tree RSS sums processes and can double-count shared
pages; it is not physical-memory usage. Machines that do not meet the 60 GiB
minimum admission threshold retain serial scanning; this 1.44× result qualifies
the admitted parallel path on the pinned corpus.

The following are means of the two runs for each variant. CPU includes the parent
and waited children. Memory is each stage's sampled tree peak, not the cumulative
process high-water mark or an allocation delta.

| Stage | Baseline wall (s) | Candidate wall (s) | Baseline CPU (s) | Candidate CPU (s) | Baseline RSS (GiB) | Candidate RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Source ingestion | 531.36 | 275.27 | 532.20 | 497.36 | 31.05 | 33.45 |
| Contributor publication | 123.95 | 117.41 | 122.61 | 116.04 | 22.32 | 6.54 |
| Protein mappings | 78.24 | 77.48 | 79.01 | 78.19 | 29.51 | 13.95 |
| Supplementary scan | 20.06 | 19.16 | 19.90 | 19.15 | 21.57 | 5.01 |
| Bulk proteomics | 14.79 | 13.66 | 15.06 | 13.73 | 24.36 | 9.87 |
| Line expression | 0.26 | 0.26 | 0.26 | 0.26 | 24.33 | 9.83 |

The baseline source row sums the two serial scanner calls. The candidate row
includes the complete source stage, including child startup, partition normalization,
compressed transfer and graph import. Consequently these stage boundaries differ;
the whole-build clock includes all work for both. Joining, other serialization and
generation work remain outside the listed stages. Source ingestion, contributor
publication and this remaining work are the next profiling targets. Do not multiply
this ratio by an earlier experiment's ratio to claim a combined improvement.

The verifier rereads every retained output and checks its hash and size against
its build receipt. All seven Parquet artifacts are byte-identical across all four
runs, preserving values, counts, row order, curation flags and schemas. Generation
source/output receipts are validated independently. The deterministic README is
reconstructed and hash-verified; its original file was not retained.

| Artifact | Rows | Columns |
|---|---:|---:|
| Observations | 4,486,991 | 75 |
| Binding | 814,268 | 75 |
| Other assays | 26,537 | 75 |
| Contributor links | 9,381,101 | 11 |
| Protein mappings | 6,205,013 | 12 |
| Bulk proteomics | 2,269,288 | 57 |
| Line expression | 42,909 | 18 |

## Controlled prefix evidence

All comparisons use pinned real-source bytes and dependencies, fresh interpreters,
retained output hashes, and the inherited Python network audit. Four runs alternate
baseline/candidate/candidate/baseline on one allocation. The independent-source
experiment alternates serial/parallel/parallel/serial on one allocation. They are
separate experiments; absolute timings must not be combined across allocations.
Receipts are in `tasks/contributor-storage-63-results/`.

| Comparison | Serial/baseline mean (s) | Candidate mean (s) | Scope |
|---|---:|---:|---|
| Storage + tuple rows | 58.96 | 54.77 | 500,000 IEDB rows, capture and publication |
| Plus adjacent annotation reuse | 57.96 | 53.21 | Same bounded IEDB workload |
| Independent source processes, corrected sampler | 73.54 | 37.49 | 500,000 rows per source; excludes cross-source merging |
| Actual merged builder source path | 115.57 | 76.79 | 500,000 rows per source; includes IPC, graph import and cross-source dedup |

The second IEDB comparison reduces wall time 8.2% and mean parent peak RSS from
2.77 to 2.26 GiB. Its observation and contributor outputs are byte-identical.
The independent-source experiment preserves both outputs for both sources across
all four runs. Mean sampled tree peak RSS is 2.43 GiB serial versus 4.54 GiB
parallel. Shared pages across distinct processes can still be counted repeatedly;
this is not a physical-memory measurement.

The independent-source diagnostic excludes production graph import and partition
merging. The actual merged builder source path preserves all four output files across
four runs and reduces mean wall time 33.6%. The full-build result above is a
separate experiment; no whole-build gain is inferred from this prefix.

## Memory sampler correction

[#876](https://github.com/pirl-unc/hitlist/issues/876) identified a runtime where
procfs lists child thread IDs beside their process leader. The original sampler
counted that address space once per thread. A live probe measured one child five
times. The corrected sampler deduplicates thread-group IDs while still traversing
each task's descendants; its result exactly matches the independent parent-plus-
child RSS calculation. The original and corrected probe receipts are retained.

Earlier tree-RSS measurements are explicitly unqualified in the historical
reports. Their parent high-water marks, output hashes and wall/CPU measurements
are independent of the overcount. The new source comparison uses the corrected
sampler for both schedules. No cross-experiment timing change is attributed to
the sampler correction.

## Final-code test-suite comparison

The separate `issue63-pass2-test-pair-4` experiment uses the complete `a71f2e9`
candidate. All four ordinary `test.sh` runs pass, with identical dependencies,
fresh caches, and two workers admitted by the unchanged memory policy. Baseline
2.1.3 passes 3,754 tests; candidate 2.1.4 passes 3,793. Each has two skips and ten
expected warnings. The candidate includes 39 additional passing regressions.

| Run | Variant | Script wall (s) | Aggregate CPU (s) | Sampled tree peak RSS (GiB) |
|---|---|---:|---:|---:|
| 1 | Candidate | 271.01 | 543.56 | 2.31 |
| 2 | Baseline | 259.67 | 503.80 | 2.22 |
| 3 | Baseline repeat | 251.11 | 485.02 | 2.22 |
| 4 | Candidate repeat | 263.13 | 525.58 | 2.30 |

Mean wall time increases from 255.39 to 267.07 seconds (4.6%). Mean aggregate CPU
increases from 494.41 to 534.57 seconds (8.1%). Mean sampled tree peak RSS is
2.22 versus 2.30 GiB. These are complete suites with different test inventories;
the measurements do not isolate the added tests' cost. No suite speedup is claimed.
The earlier `e88891a` comparison and its 3.3% increase are retained as historical
evidence; its absolute timings must not be combined with this allocation.

All five hosted unit configurations and 97 integration tests pass for `a71f2e9`.
Its separate release gate runs `test.sh --all`: 3,794 unit tests pass with one skip,
then all 97 integration tests pass. The release environment includes mhcflurry;
the ordinary unit matrix's missing optional mhcflurry and Ensembl data explain its
two skips. The final local test attempt refused before starting because free memory
was below the unchanged threshold; that refusal is retained. Hosted verification
supplies the complete final-code test runs. Checks on later evidence-only commits
and clean-main release builds are separate from these benchmark measurements.

## Full-transfer capacity correction

The first full candidate using uncompressed IPC exhausted the IEDB worker's
3.2 GiB transfer reservation and correctly retried serially. That experiment was
cancelled and is excluded from parallel timing claims; its completed baseline
alone cannot establish a speed ratio. [#880](https://github.com/pirl-unc/hitlist/issues/880)
records the failure. A read-only live inspection of the first 4 KiB of its log
identified the retry; the observer action is retained with the excluded attempt.

The corrected candidate uses [Arrow IPC compression](https://arrow.apache.org/docs/python/generated/pyarrow.ipc.IpcWriteOptions.html)
with Zstandard and no compression-thread fan-out. The total scratch budget and
all reservations remain unchanged. Regression cases cover both compressible
transfers that fit only after compression and incompressible transfers that must
still fail into the bounded serial retry.

The replacement full experiment runs candidate/baseline/baseline/candidate on one
allocation, checking the candidate first to expose capacity failures early. Each
candidate must select two source workers without a serial retry before entering
the comparison. The replacement full-build and final-code suite comparisons both
pass; their complete receipts are retained separately from the cancelled attempt.

## Reproduction and scope

The final comparison uses baseline `ff5247f787ce79fbb640a00bcbe702fb7b7910cb`
and candidate `a71f2e9bf7a01e0b33d751a3134a88f024f4b055`. Run the committed harness
with a configured Modal account and the pinned files at the volume paths recorded
in `tasks/corpus-qualification-sources.json` and
`tasks/reliability-benchmark-reference.json`:

```sh
git archive --format=tar.gz --output=/tmp/hitlist63-baseline.tar.gz \
  ff5247f787ce79fbb640a00bcbe702fb7b7910cb
git archive --format=tar.gz --output=/tmp/hitlist63-candidate.tar.gz \
  a71f2e9bf7a01e0b33d751a3134a88f024f4b055
export HITLIST_BENCHMARK_ARCHIVE=/tmp/hitlist63-candidate.tar.gz
export HITLIST_BENCHMARK_BASELINE_ARCHIVE=/tmp/hitlist63-baseline.tar.gz
python -m modal run --detach scripts/reliability_modal.py \
  --label unique-full-label --paired --candidate-first \
  --require-source-workers --background
HITLIST_BENCHMARK_TEST_IMAGE=1 python -m modal run --detach \
  scripts/reliability_modal.py --label unique-test-label \
  --test-pair --candidate-first --background
```

Each label must be unused. The printed durable function-call ID and the completed
`experiment.json`, rather than the submission command's exit, establish completion.
The four full builds share one nonpreemptible allocation with eight CPUs, 64 GiB
memory and 512 GiB scratch; the suite uses a separate allocation. Requirements are
hash-locked, including pandas 3.1.0 and Arrow 25.0.1. Both variants use two mapping
workers. A request for four source workers exercises the candidate's two-worker
cap; the baseline still scans sources serially.

Every build has fresh outputs and a cold Hitlist mapping index. Raw source files
and the human Ensembl 112 reference are prepared before timing; acquisition,
verification and reference indexing are excluded. Other reference coverage is not
claimed. The inherited Python audit rejects Internet-socket and name-resolution
attempts during timing, including spawned interpreters; it is not an OS network
sandbox. Output qualification hashes every retained Parquet byte and validates
generation receipts. Only specifically recorded location, time and producer-version
metadata differences are allowed.
