# Issue #63 experiment receipts

Baseline producer: released Hitlist 2.1.0, commit
`5f988fc01d7c0470fb369cc7f73f154a76f63ceb`. All results here use the dependency and
input pins in the repository. Profiles are diagnostic; only complete unprofiled
experiments can supply timing ratios. The 500,000-row scan prefix is not a full
build.

- `baseline-scan*`: released 100,000-row profile and input receipt.
- `serialization.json`: five alternating repetitions of exact JSON encoding on
  10,000 real rows; short-string token caching was slower and rejected.
- `serialization-scan*`: serial candidate `8b2eafd`, profiled on a different
  allocation; no cross-allocation speed ratio is claimed.
- `paired-scan-3*`: completed same-allocation baseline/candidate/candidate/baseline
  experiment for annotation-only workers (`ecb778c`). All output bytes and
  non-Hitlist package versions match; zero Python socket-audit attempts. The
  9.5% lower mean wall time costs more CPU and sampled tree memory.
- `parallel-parent*`: parent sampling profile for the annotation-only candidate,
  locating encoding, parsing and database work that still run in the parent.
- `failed-pair1.log`: harness PMID-to-Arrow failure; excluded from timing evidence.
- `interrupted-pair2*`, `interrupted-pair4*`: incomplete experiments, excluded from
  repeatability claims. #869 tracks the cancellation evidence and diagnostics.
- `local-memory-refusal.txt`: test.sh refused before tests started; its ordinary
  admission threshold was preserved.

Public Parquet equality includes every raw source cell, contributor relation,
curation flag, identity, schema and row order. Network checks cover Python socket
and DNS audit events, including spawned interpreters; they are not an OS firewall.
Sampled process-tree RSS may double-count shared pages.
