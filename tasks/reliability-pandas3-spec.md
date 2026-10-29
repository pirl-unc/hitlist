# Reliability, performance and pandas 3 baseline — 2026-10-28

## Requested outcome

Resolve Hitlist issues #815, #844, #754, #176 and #63. Require pandas >=3
and Python >=3.11 across package metadata, lockfile, runtime assumptions, CI
and current documentation. Preserve scientific source values, labels, identifiers,
row multiplicity, provenance, historical artifacts and the complete release gate.
The original checkout's uncommitted work and the shared Python environment stay
outside this work; implement on the isolated reliability feature branch.

## Runtime contract (#815)

Remove the deprecated pandas infer_string option access entirely. Depend on the
supported pandas 3 string behavior and verify warning-as-error imports and real
scanner/string-dtype behavior. Remove obsolete Python-3.10 dependency branches,
update the supported Python matrix and docs, and check declared requirements,
the resolved lockfile and isolated installed environments. Verify the pandas 3.0
floor and the latest mutually compatible stable release. Preserve package-import
filesystem safety and lazy public APIs.

## Release integrity (#844)

Reconstruct the 2.0.38 incident and the replacement 2.0.39 release from immutable
GitHub commits, successful workflow artifacts/receipts and public PyPI files.
Do not overwrite, delete or silently bless conflicting historical bytes. Record
what can actually be established and strengthen reusable publication checks where
the incident exposes a gap. Test rejection of mismatched version/source/artifacts,
partial publications and unverified files. The new release must use only the
successful clean-main artifact pair and verify actual public downloads afterward.

## Release termination (#754)

Inspect the original attempt and retry, job/run metadata, retained logs and the
already shipped #768 mitigation. Distinguish SIGTERM/exit 143 from proof of OOM;
do not manufacture a historical sender when the runner retained no such evidence.
Complete a bounded incident investigation and improve durable failure diagnostics
as needed: stage/test progress, exit/signal status, peak memory, cgroup/pressure
and runner events. A failed or partial test gate must never produce an approved
release. Preserve existing memory admission and full test coverage.

## Measured build performance (#176, #63)

Use real pinned IEDB/CEDAR sources, supplementary inputs and proteome/reference
assets. Existing tiny smoke tests are correctness checks, not scale benchmarks.
Locate the existing isolated corpus/Modal source cache rather than redownloading
or rebuilding large data on the memory-constrained shared workstation. Use bounded
remote CPU/memory/time, separate outputs, verified inputs and no mutation of
historical corpus generations.

Record a current baseline for the full forced build and its observation, mapping,
bulk and expression stages: unprofiled wall/CPU time and Linux peak RSS, plus a
separate cProfile pass for hot functions. Keep instrumentation overhead separate
from speed comparisons. Select optimizations from measured hot spots, then repeat
on identical source bytes, dependency versions and resource/concurrency settings.
Target the requested 2x full-build speedup; report measured results rather than
extrapolating microbenchmarks. Compare complete output schemas, row counts,
identities, source values, curation flags and provenance using bounded reads;
document expected metadata/version differences separately. Record comparable
test-suite timings as well as build timings. Add focused regressions for changed
behavior and publish reproducible profiling scripts and receipts.

Measurement revision: the initial 100,000-row cProfile diagnostics reported
impossible cumulative relationships (scan less than its record() children;
thread waits exceeding elapsed time), including in a standalone interpreter.
Retain those runs as invalid diagnostics. Use a separate sampling-profiler pass
with pyinstrument 5.1.3 and cross-check its duration against independent wall
time. Keep all speed comparisons unprofiled with the same runtime dependency
pins; the added optional profiling tool is not imported during those builds.

## Delivery

Run format.sh, lint.sh, focused tests, test.sh, dependency/lock/install checks,
supported-Python CI and the full release gate. Bump the release version on the
feature branch; review the final diff and measurements, merge through PR, deploy
from clean main and compare public PyPI bytes with the tested artifacts. Update
each issue with its own evidence and disposition; do not close profiling issues
based on a synthetic benchmark or label the historical SIGTERM as proven OOM.

## Selected storage optimization

The valid 100,000-row sampling diagnostic placed provenance record capture at
10.01 s of the 26.66 s scan (diagnostic durations, not a speed ratio). On 10,000
identical real payloads, median compression time over three repetitions was
0.3315 s for per-row zlib level 1 and 0.0266 s for 128-row zstd level-1 blocks.
Including a conservative 24-byte reference per row, compressed payload storage
was 8,716,064 versus 889,246 bytes. Every decoded payload matched exactly.

Implement bounded disposable SQLite payload blocks: at most 128 rows or 1 MiB
per block, with a single oversized source row isolated, and a 4 MiB decoded-block
LRU. Store each row's byte offset/length and retain its existing stable ID and
exact JSON serialization. Keep graph traversal, public contributor columns,
relationships, schema version, scratch cap and atomic publication unchanged.
Reuse the canonical JSON encoder and bounded escaped dataset-prefix cache.
Full before/after builds and complete output reconciliation remain required;
these microbenchmark results do not establish a whole-build speedup.

## Interrupted baseline and measurement revision

The first full baseline was preempted by its Modal host after completing the
IEDB scan (1136.66 s wall, 1113.69 s process CPU) and 69% of CEDAR. It is an
incomplete attempt, excluded from full-build ratios. Its output directory is
retained; a restarted attempt cannot overwrite it or count it as complete.
Subsequent full runs use Modal's non-preemptible CPU setting with the same
8-CPU/64-GiB/90-minute bounds. New runs also record input/code/recipe pins and
CPU model, sample parent/tree RSS per stage, and collect separate mapping-worker
sampling sessions for diagnostic runs. Tree RSS is a sum that can double-count
shared pages; it is not a physical-memory claim. The first candidate already
started under the original preemptible configuration and remains an explicitly
identified exploratory run.

The first isolated test-suite timing is invalid: the benchmark image omitted the
`git` executable required by release-artifact fixtures (3,666 passed, 26 setup
errors). Add git to the benchmark image and repeat both variants with fresh
caches. No product test or admission guard is bypassed. A spawned-worker smoke
check confirms the diagnostic wrapper returns the original cache-miss result
and retains its sampling session on worker shutdown.

The fresh Git install exposed 404s from the legacy image builder's Debian 11
security repository (Debian 11 LTS ended August 2026). Both test measurements
use `HITLIST_BENCHMARK_TEST_IMAGE=1`, a Python 3.12/Bookworm image with Git and
GNU time; the full-build pair retains its original identical image. This does
not change workspace-wide Modal settings. Source: https://www.debian.org/releases/bullseye/

Dependency review: request pandas' `pyarrow` extra so the chosen pandas release
also constrains its compatible Arrow floor (pandas 3.0 requires Arrow 13; 3.1's
extra requires Arrow 16). An unconstrained separate Arrow requirement would
allow an existing older installation to survive an upgrade and fail at runtime.
The benchmark pins already use Arrow 26; their installed versions are unchanged.

The candidate's separate 100,000-row diagnostic retains the same 100,000 source
links and byte-identical contributor Parquet (SHA-256
46d897c45c234a13188f1f2da846f9c3e8e9b4272949e60d41d7d2b88114d7dc).
Its sampled duration is 19.674 s versus 30.851 s for the baseline diagnostic;
record capture accounts for 3.79 s versus 10.01 s. These diagnostic runs locate
changed costs, but do not establish the full-build speedup or its noise floor.

## Full worker profile: mapping aggregation and unintended downloads

The complete baseline sampling run isolates 431.74 s in `map_peptides`, including
242.24 s creating pandas groups and 63.37 s selecting DataFrame columns. Compute
source counts and flank agreement while each peptide's hits are already grouped
in the lookup loop. Preserve row order, occurrence multiplicity, distinct-protein
counts, terminal empty flanks, metadata, column order and pandas dtypes. Keep
aggregate columns in parallel lists to avoid enlarging every mapping-row dict.
Verify against the original pandas aggregation on shared/repeated/terminal hits.

The other worker's profile proves `fetch_missing=False` does not reach the
Ensembl branch: lazy `Genome.db` attempts HTTP downloads. The optional symbol
backfill has the same lazy path. Add an explicit keyword policy to
`from_ensembl` (default preserves direct callers), forward the worker policy,
and reject unavailable required local files before lazy feature access. Use
PyEnsembl's public offline availability inspection, retaining cache hits where
available. Backfill skips uninstalled releases using the same inspection. Test
actual missing-cache instances with a network recorder and successful installed
fixtures. Correct the earlier benchmark's offline claim; no unavailable
reference may silently count as a successful zero-hit search.

Run focused checks and another pinned full candidate build, then reconcile all
outputs and report timing variance. The two completed baseline builds took
2307.36 and 3320.58 s, so a single before/after ratio is not a reliable speedup
estimate. The full sampling run is diagnostic and excluded from timing ratios.

## Complete-suite pandas deprecation cleanup

The local guarded test.sh passed 3,734 tests but exposed three additional
pandas 3.1 deprecation sites: filling an all-missing float modifications column
with a string, and two deprecated DataFrame.drop(inplace=True) calls. Declare
the textual column's string dtype before filling and assign drop results.
Verify the existing bulk/flanking/export contracts and the full unit suite with
FutureWarning promoted to errors. Include this final runtime revision in a
separate pinned full candidate measurement/output comparison.

The first focused warning check accidentally selected integration-marked export
tests as well as unit tests. It was interrupted after 82 passes; it is not a
completed check. A simultaneous test.sh correctly refused admission at 0.19 GiB.
Repeat focused checks with `-m 'not integration'`, and preserve test.sh's normal
admission policy. Make FutureWarning originating from Hitlist code an error in
the shared pytest configuration so both supported pandas branches retain the
migration contract without suppressing third-party warnings.
