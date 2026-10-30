# Issue #63, second-pass evidence

Baseline: released 2.1.3, main `ff5247f787ce79fbb640a00bcbe702fb7b7910cb`.
Earlier full/test producer: `e88891a`, version 2.1.4; full qualification was
superseded after the IPC capacity failure. See
[the report](../../docs/source-build-performance.md) for boundaries and results.
Final measured producer: `a71f2e9bf7a01e0b33d751a3134a88f024f4b055`, version 2.1.4.

- `baseline-profile*`: 500,000-row sampling profile; diagnostic timings only.
- `scan-1*`: completed ABBA prefix comparison of tuple rows, previous-cell JSON
  reuse and WITHOUT ROWID records; exact observation/contributor bytes.
- `scan-2*`: completed ABBA prefix comparison adding adjacent annotation reuse.
- `source-pair-1*` / `source-summary.json`: independent-source diagnostic before
  the RSS correction. Its tree-memory values are unqualified; see #876.
- `memory-probe-*`: observed duplicate thread IDs and corrected process sum.
- `source-pair-2*`: repeated independent-source diagnostic with corrected RSS;
  excludes the production graph/partition merge.
- `merged-sources-1-experiment.json`: actual source ingestion, IPC, graph merge
  and cross-source dedup. All four final Parquet outputs match across four runs.
- `superseded-experiments.json`: intentionally cancelled full/test runs after
  review identified nested-cgroup issue #878. Partial results are not qualification.
- `admission-probe.json`: corrected membership/mount resolution on the actual
 64 GiB runtime. Local regression tests cover nested and hybrid hierarchies.
- `local-validation.json`: complete local test.sh pass, format/lint and isolated
  metadata/dependency checks for earlier `e88891a`; separate from paired suite timing.
- `test-*`: completed earlier `e88891a` suite comparison, before the empty-schema
  and IPC-compression corrections.
- `final-test-*`: four complete ordinary test.sh runs for `a71f2e9` and baseline,
  identical dependencies, normal two-worker admission and individual summaries.
- `final-producer-ci.json`: successful current-code matrix, integration, docs and
  release gates; later evidence-head and clean-main checks remain separate.
- `full-*`: complete candidate/baseline/baseline/candidate builds for the final
  producer, exact seven-Parquet comparisons, input/output receipts, GNU time
  resources, per-stage means and independently collected network-audit inventory.
  Both candidates use two source workers without a serial retry.
- `execution-identifiers.json`: durable app/call IDs. A local submission's exit
  does not establish remote completion. Running experiments remain unqualified.

Remote artifacts remain under `presto-data/hitlist-reliability-20261028`.
Full comparisons require identical source/dependency pins, zero recorded Python
network attempts and exact retained Parquet bytes. Named location/time/version
metadata allowances are recorded individually. Sampled tree RSS counts each
thread-group once but can still count shared pages across distinct processes.


`ipc-capacity-*` retains the full candidate's transfer-reservation failure and
intentional cancellation. `final-local-memory-refusal.txt` records a local test.sh
preflight refusal before any tests started. The replacement full qualification
passes with actual parallel selection and no serial retry; old partial results
remain excluded.
