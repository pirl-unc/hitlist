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
