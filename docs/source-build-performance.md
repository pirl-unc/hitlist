# Source scanning and contributor storage

This continues [#63](https://github.com/pirl-unc/hitlist/issues/63) from released
2.1.3 (`ff5247f`), after the [first serial optimization](serial-build-performance.md).
Full-build qualification is pending. Prefix results below do not establish the
issue's 2× whole-build target.

## Candidate changes

The scanner retains row tuples instead of one dictionary per row. It reuses an
assay annotation only when the immediately preceding row has the same four input
fields. Contributor capture reuses the previous row's encoded cells within its
existing byte cap; oversized rows clear that cache. The disposable SQLite records
table uses its primary key as its storage index. Original source cells, row order,
identifiers and scientific classification rules remain unchanged.

The builder can scan IEDB and CEDAR in two independent processes. It consumes their
results in the existing source order and uses the same partition normalization as
the serial path. Each child writes private Arrow tables and a closed contributor
graph. The parent imports payload blocks, records and edges before existing
cross-source deduplication and generation publication.

`HITLIST_SCAN_WORKERS=1` selects serial scanning. Larger values are capped at two
source processes; the default is two, with `HITLIST_BUILD_WORKERS` as a fallback
cap. Two sources, two available CPUs and conservative memory admission are all
required. Unknown memory selects serial scanning. The current admission threshold
is 28 GiB per source plus 4 GiB parent headroom, increased for source files larger
than the qualified inputs. Available memory is capped by the container's remaining
allocation, including the process’s resolved nested cgroup and visible ancestor
limits. CPU affinity and ancestor quotas also cap admission. This is a capacity estimate, not a memory reservation against other
applications. Full-corpus resource qualification is still pending.

The existing contributor scratch budget covers the parent graph, both child
graphs, IPC files and bounded worker metadata together. SQLite page limits and
aggregate IPC write limits enforce separate reservations. The parent reclaims a
reservation only after removing its files. If a child exhausts a reservation before
any import, the builder cleans up both children and explicitly retries serially.
Other errors abort publication. Existing published data remains intact.

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
four runs and reduces mean wall time 33.6%. Complete forced builds remain required
before making a whole-build claim.

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
