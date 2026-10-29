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
