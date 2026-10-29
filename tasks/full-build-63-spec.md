# Controlled full-build performance — issue #63

## Objective and baseline

Reduce the full forced build's dominant measured costs without changing source
records, observation identities, curation flags, graph relations, row ordering,
public schemas or source/reference availability. Start from released 2.1.0,
`5f988fc01d7c0470fb369cc7f73f154a76f63ceb`. The previous 2.0.44 comparisons remain
historical: separate hosts and unintended reference downloads prevent a reliable
2× claim. Measure new gains against 2.1.0, and do not silently change the baseline
to obtain the issue's target. A 2× result must hold across repeated comparable
unprofiled full builds, not a component microbenchmark or a favorable single run.

## Investigation and implementation decisions

1. Reuse pinned 14.7 GB IEDB/CEDAR sources, 26 packaged assets and prepared human
   Ensembl 112 references from the existing benchmark. Keep their limited species
   coverage explicit. Source acquisition and reference preparation stay outside
   timing; outputs and the Hitlist mapping index start cold for each forced run.
2. Profile the released scanner and contributor publication again. Existing full
   profiles predate the block codec and direct mapping changes. First use the
   bounded real-source diagnostic to locate current row-level work, then confirm
   relevant mechanisms in a full profile when needed. Profiled runs cannot supply
   speed ratios. Existing annotation/classification caches are not speculative
   optimization targets.
3. Choose changes from current evidence. Candidate mechanisms include redundant
   JSON construction/serialization, repeated SQL work and graph traversal,
   Python-to-Arrow conversion, and independent source scans. Preserve exact JSON
   bytes unless a separately justified public contract change is required; this
   task does not authorize dropping contributor fields or changing semantics.
4. Prefer bounded batching and eliminating redundant work before concurrency.
   If concurrency is justified, bound workers and memory, keep deterministic
   source/row order, preserve per-scan duplicate handling and contributor graph
   ownership, and fail the entire atomic build when any worker fails. Never
   disable memory admission or scratch-capacity guards.
5. Use a feature branch and isolated environment; do not touch the shared Python
   environment or the user's original checkout. File newly discovered defects
   on the owning repository and link them from the PR. #867 and #868 are separate
   follow-ups, not prerequisites for this performance implementation.

## Controlled experiment

Run paired baseline/candidate fresh processes on the same allocated remote
container with identical pinned dependencies, source bytes, warm references,
thread limits, worker counts and resource allocation. Alternate order across
repeats to expose cache/order effects. Retain source/archive/recipe hashes,
package versions, CPU/platform information (including unavailable probes),
wall/parent-plus-child CPU, parent peak RSS, and sampled process-tree RSS with
its shared-page caveat. Cap remote CPU, memory, disk and runtime explicitly.
Network attempts during the timed build must be detected; preparation may fetch
only the existing pinned inputs. Do not time source preparation or package
switching as build work. Avoid simultaneous unrelated benchmark jobs.

Retain all outputs and validate their actual bytes and generation metadata using
the strict existing comparison. Every differing byte needs an explanation; do
not sort away differences or drop fields to make comparisons pass. Run at least
two complete measurements of each variant for the final comparison. Measure the
default test.sh separately under matching worker/cache settings; new tests are
expected and their count differences must be disclosed. No suite speedup is
claimed solely from full-build gains.

## Verification and shipping

Add targeted regressions for each changed invariant and meaningful failure mode.
Run format.sh, lint.sh, test.sh, strict documentation, editable-install metadata
and pip check. Complete the hosted Python/pandas matrix, full integration and
release gates. Bump the version for the PR (initially 2.1.1, recheck main before
landing), merge only after final-head checks, deploy the exact clean-main tested
artifacts, and verify both actual public PyPI downloads. Record measured outcomes
and limitations in the report and issue. Close #63 only when its acceptance
criteria are supported, rather than treating effort or elapsed time as evidence.

## Review

The released 100,000-row diagnostic took 21.338 seconds under sampling: scan
18.62, contributor capture 4.42 (2.70 in JSON), raw field access 1.51, and
contributor publication 2.54 seconds. Its producer archive SHA-256 is
c56de402e4ab6263405859f14b6e0c722db1d59224a59c1001a9912605a9c087.
Initialization dominates the annotation-cache misses in this prefix; those are
not evidence for another annotation cache.

A five-repeat, alternating-order 10,000-row diagnostic compared exact JSON
bytes. Median baseline serialization was 0.32458 seconds; encoding each string
cell once and assembling the two existing JSON views was 0.19338 seconds.
Caching short escaped strings was slower (0.28241 seconds) and is rejected.
These component results do not establish a whole-build speedup.

The first candidate binds each CSV column layout once, encodes each cell once
using the same standard-library JSON string encoder, reuses projected fields,
and writes bounded contributor batches from tuples. Nested/typed supplementary
JSON retains the general encoder. All 70 scanner/provenance tests and format/lint
pass, including exact JSON/Parquet equality for Unicode, controls, unmapped and
short rows, donor labels and capture/output batch boundaries.

The next candidate separates pure row annotation from ordered CSV capture.
The parent retains source registration, original row numbers, immediate record
insertion and per-scan assay deduplication. Spawned workers annotate bounded
batches; the parent consumes results in submission order and inserts the emitted
observation edges. At most two batches per worker may be in flight, with row and
byte limits per batch. Workers receive the parent's resolved data directory.
Full scans must keep the serial path, explicit worker controls, conservative
memory admission and whole-build failure propagation. Regression comparisons
cover cross-batch duplicate IRIs, filtered first occurrences, donor expansion,
unmapped/short rows, ordered frames and exact contributor Parquet bytes.

The annotation refactor passes all 88 focused scanner/provenance/worker tests,
including actual spawned workers from a threaded parent, explicit cache context,
whole-operation failure, input handle closure and scratch cleanup. Direct scans
default to serial; the builder selects at most four workers, reserving 32 GiB
for the measured parent peak and 2.5 GiB per additional worker. Low/unknown
capacity keeps the serial path. Batches cap at 1,024 rows or 4 MiB of input values;
an oversized row stands alone. These admission estimates still need validation
against the full-run memory receipts.

The first serialization candidate's 100,000-row contributor Parquet is identical
to the baseline (SHA-256 46d897c45c234a13188f1f2da846f9c3e8e9b4272949e60d41d7d2b88114d7dc,
7,588,242 bytes). Its sampled scan took 20.49 s versus the earlier 18.62 s on a
different allocation. Do not infer a regression or gain from those profiled,
separately hosted runs; the unprofiled same-allocation experiment decides which
changes remain.

The first alternating 500,000-row diagnostic (`issue63-paired-scan-1`) is excluded:
its baseline scanned and wrote contributors, then the new harness failed to
serialize the raw mixed PMID column. The builder already normalizes that column
to numeric nullable Int64 before Arrow conversion. The harness now uses that
same projection, retains every original source cell in contributors, and tests
integer, missing and nonnumeric PMID inputs. This is a harness defect, not a
production regression or an accepted timing result.

The local test.sh preflight initially could not read the macOS capacity probe
inside the sandbox. The same command outside it admitted two workers at 7.72 GiB
available with the unchanged 2.5 GiB per-worker budget. No memory guard was lowered.

The admitted local run passed 3,761 tests (one skip, ten expected runtime/user
warnings) in 367.41 pytest seconds / 372.07 script seconds. The subsequently
added harness PMID regression also passes in the seven-test harness subset;
final-head complete checks remain required.

The second paired diagnostic was cancelled by the remote runner after its first
baseline and while its candidate process was finishing. The trigger is unknown;
there is no OOM or timeout attribution. The candidate's own report completed
(four annotation workers, 57.69 parent CPU seconds and 26.80 waited-child CPU
seconds), but the alternating experiment did not complete and cannot establish
a repeatable speed ratio. Retain its partial receipt and logs. Parent work
remains the main CPU cost, so annotation-only concurrency is not yet an accepted
optimization; assess serialization/capture placement before a full-corpus claim.

The third alternating 500,000-row diagnostic completed successfully. Baseline
wall times were 68.751 and 68.879 s; candidate times were 62.724 and 61.817 s
(1.105× ratio of means, 9.5% less elapsed time). All non-Hitlist package versions
match and both output files have identical bytes across all four runs. The
inherited Python socket audit recorded no network attempts. Candidate CPU rose
from about 66.1 s to 85.1 s including waited children, and sampled tree RSS rose
from about 2.7 GiB to 4.1 GiB. This narrow diagnostic does not establish a full-
build gain or justify shipping the annotation-only concurrency yet. Profile the
parent on the same 500,000-row scope before choosing a revised boundary.

## Revised worker boundary

The 500,000-row parent sampling diagnostic attributes 35.03 of 104.13 s to
record encoding/capture, 13.18 s to CSV iteration, 10.39 s to assay deduplication,
6.92 s to the per-value input-size generator, and 5.51 s to observation inserts.
These profiled numbers explain placement, not a speed ratio. Move exact CSV JSON
encoding and bounded scratch-block compression to the existing annotation
workers. The parent keeps CSV parsing, peptide filtering, per-scan assay
deduplication, source snapshots, graph ownership and ordered output assembly.

Worker batches carry one source layout and raw rows plus original row numbers,
record IDs and the parent's duplicate decision. Duplicate rows are still captured
but do not enter annotation. Workers return bounded compressed payload blocks,
record locators, annotations and observation edges. Parent imports blocks and
records, then edges, in submission order; SQLite remains parent-only. Scratch
block boundaries may change, but every public JSON and Parquet byte must match.
Use guarded executemany for record and edge insertion. Direct record/CSV APIs
retain immediate duplicate rejection; worker-path duplicate record rejection is
at ordered batch ingestion, before scan returns or anything is published.

Preserve the 128-row/1 MiB scratch-block caps and the 1,024-row/4 MiB worker-input
caps, at most two batches per worker. The raw-row byte estimate includes tuple,
list, string and locator storage; a single oversized row remains isolated. Keep
serial fallback and all worker admission limits. Extend regressions to cover
non-ASCII/control/unmapped cells, pending serial payloads before/after imported
blocks, duplicate records, capacity exhaustion during import, filtered duplicate
IRIs, cross-source boundaries and worker failure. Re-run the same unprofiled
alternating diagnostic before committing to full measurements.

The revised boundary passes 98 focused tests. A further targeted-peptide
regression checks filtering before assay deduplication and original row numbers.
The required full local test.sh attempt refused at 0.13 GiB free versus its
unchanged 2.5 GiB guard; no tests started in that attempt.

`issue63-paired-scan-4` was also cancelled, after one baseline and one revised
candidate had completed according to the remote lifecycle log. The local stream
was incomplete. Its receipt remains failed and cannot supply a repeated ratio.
Track cancellation diagnostics in #869. Modal's invocation documentation says
synchronous remote calls are cancelled after caller disconnect even when an
app is detached, while spawn calls are durable. That is a plausible mechanism,
not proof of this initiating cause. Add explicit background submission with a
retained function-call ID, start/per-run checkpoints, signal diagnostics and
process-group cleanup. Verify signal cleanup in a real subprocess regression.
Reference: https://modal.com/docs/guide/function-invocation-methods#durability

Before a full-corpus experiment, compare baseline/serial-candidate/parallel-
candidate/parallel-candidate/serial-candidate/baseline on one allocation. This
tests whether concurrency improves on the simpler serial changes; keep all
outputs and apply identical equality/network/dependency checks. Separately run
test.sh in an alternating pair on one Git-equipped allocation with fresh caches,
identical dependencies and TEST_SH_MAX=2; test networking is not audited as build
networking because tests intentionally exercise local sockets and audit failures.

## Accepted production direction

The durable six-run experiment completed, with exact bytes across all six runs
and matching dependency versions. Baseline times: 69.465, 68.568 s; serial
candidate: 60.036, 59.855 s; parallel capture: 65.806, 65.205 s. Means are 69.016,
59.946 and 65.505 s respectively. Serial uses 57.39 CPU s on average, compared
with baseline 66.755 and parallel 103.335 including children. Parallel is slower
than the simpler serial path and substantially more costly; reject it. Remove
worker scheduling/admission/capture code and its implementation-specific tests.
Retain exact CSV token reuse, single field projection, relationship serialization
caching and tuple-based contributor output. The final serial encoder uses
list(map(escape, cells)); no source-value caching is introduced.

The 13.1% lower mean wall time is a scan-prefix result, not a full-build claim.
Measure the reduced final producer against released 2.1.0 on a single allocation
with the same four-run full-build design. The 2× target remains unproved and #63
must stay open unless the final evidence establishes it. Removing rejected
concurrency is part of the result, not a reason to discard its receipts.

The reduced producer is commit 2b850a39a4c0cfa814eb12b852cc742576d5d287,
archive SHA-256 768dac42458709ffd1e302c8aa6fbb7a927be58e67b2c5e3351f43c1894254e2.
Its local test.sh passed 3,745 tests, one skip and ten expected warnings; the
unchanged guard admitted one worker at 3.14 GiB available (ceiling two). Pytest
took 392.77 s and the script 396.08 s. This verifies the code; it is not a
paired suite measurement. Format/lint, strict MkDocs, editable metadata and
pip check also pass. Draft PR #870 has all five hosted unit jobs, 97 integration
tests and the separate release-build gate passing on this producer. Final-head
checks and complete repeated full measurements remain required. Signal cleanup
now tolerates the child exiting between poll and signal, always commits receipts,
and the regression publishes its PID atomically before the test signals it.
