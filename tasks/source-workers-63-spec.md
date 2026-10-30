# Independent full-source workers — #63

## Decision gate
The independent-prefix diagnostic halves source-stage wall time with similar CPU
and exact per-source output bytes. #876 fixes a demonstrated tree-RSS overcount;
repeat resource measurements before final admission. This is not yet a whole-build
speed claim: the production path must include merging all graph data/partitions.

## Scope
Parallelize only the builder's independent IEDB and CEDAR scans, at most two worker
processes. Keep the public scanner's direct-call behavior serial. Do not partition
one source, change source classification, cache force-build results, or reinterpret
observations. Use the same serial helper for scan/partition normalization so dtype,
categorical order and row order stay identical. The parent consumes sources in the
existing IEDB-then-CEDAR order regardless of completion order.

## Admission and ownership
Unknown capacity, fewer than two CPUs, a single source, or an explicit workers=1
selects the serial path. Use Linux MemAvailable capped by cgroup v1/v2 remaining
memory, resolving process membership and every visible ancestor mount limit
(including hybrid v1/v2 hierarchies), or macOS free+speculative pages. Admit two sources only with conservative
per-source budgets (at least28GiB each, increased for source files larger than the
qualified corpus) plus4GiB parent headroom. Reconfirm against the full benchmark;
never force workers past measured capacity. Honor HITLIST_SCAN_WORKERS as a cap.

Each worker owns a fresh process, collector, scratch subdirectory and Arrow IPC
outputs. No per-row Python values or payloads cross a multiprocessing queue. Pass
only source paths, data-directory resolution and explicit resource quotas. Spawn
processes so inherited SQLite handles/global mutable state cannot leak. Parent
cancellation or any worker failure terminates/reaps every owned worker before
scratch cleanup. A normal worker error aborts the atomic build.

## One total scratch budget
Preserve the caller's existing collector limit and free-space reserve. Partition
that limit into five page-aligned reservations: parent graph, two worker graphs,
and two per-worker aggregate IPC quotas. Each SQLite database enforces its own
reservation with max_page_count. A bounded write sink enforces each IPC reservation
across all output files, checking disk reserve before writes. Their sum must never
exceed the original collector budget.

After workers exit successfully, read each source's IPC tables into owned Arrow
buffers and delete those IPC files before reclaiming that reservation for the
parent. Merge its captured graph, then delete its snapshot before reclaiming its
graph reservation. Never grow a parent quota before the corresponding files have
been removed. Restore the original parent quota only after all reservations are
released. Before any import, a worker reservation exhaustion can explicitly retry
serially after terminating/reaping workers and removing their owned files; do not
retry other failures or any failure after parent graph mutation.

## Snapshot and merge
A worker flushes bounded payload blocks, commits/closes its disposable database,
and moves that file into its owned result directory. Sources retain original file
paths/hashes and logical row locators. The parent checks source identity, attaches
the private snapshot, copies payload blocks with a block-ID offset, copies records
with that same offset, and copies edges. Source record IDs remain unchanged and
unique; source scans already deduplicated independently. Per-scan assay lookup
scratch is not needed after that source completes. Existing cross-source and
supplementary deduplication still run in the parent with the complete graph.

Detach before removing a snapshot. Never expose partial results as public data.
Source rehash and contributor publication continue through the existing collector
write path, preserving its generation-level atomicity and corruption checks.

## Verification
Compare serial versus actual parallel builder source ingestion on fixtures with
same assay IRIs across sources, duplicate rows, missing IRIs/PMIDs, multiple donor
labels and empty modality partitions. Require exact contributor bytes and identical
frames/dtypes/categorical order. Cover quotas, capacity-only serial retry, malformed
worker output, arbitrary failure, source mismatch, snapshot corruption, cancellation
and child cleanup. Existing graph oracle/cycle/storage tests still apply.

Before selecting the implementation, compare complete forced builds using identical
pinned inputs/dependencies and corrected memory sampling, including startup, IPC,
merge and output verification. If the merged path loses the diagnostic benefit,
remove it. Pass required local/hosted checks, version bump, PR merge and exact tested
PyPI artifact publication; keep the 2x target open unless actually established.

## Admission review — #878

The initial draft checked only root controller files. A synthetic nested service
with 96 GiB remaining at the root but 30 GiB at its parent demonstrates that this
can admit workers incorrectly. Resolve mount-relative membership and all visible
ancestor limits for both memory and CPU; also cap by CPU affinity. Unknown
hierarchies select serial execution. This is an unshipped draft correction.
