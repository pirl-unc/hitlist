# Spec: OncoRef CTA Policy Selection (#852), Export Deprecation Notice Fix (#893), and Build-Worker Parent-Only SIGTERM Cleanup (#881) (v2.1.24)

## 1. `#852` — Explicit OncoRef CTA policy choice in evidence bundles

- Keep `cta_policy="legacy"` as the default on `resolve_expression_table()`,
  `write_cta_evidence_bundle()`, and `hitlist export cta-evidence --cta-policy`
  so upgrading `oncoref` never implicitly replaces the historical 624-gene
  `oncoref.cta` provider (`definition="strict"` / `"extended"`).
- Support explicit owner-policy selectors:
  - `"legacy"`: `oncoref.cta` (`strict` / `extended`)
  - `"target_sets"`: `oncoref.target_sets` + `oncoref.cta_policy.manifest()` (OncoRef 1.20.0 contract; `CTA` targets vs `Unsafe` only — `Neither` is neither target nor default forbidden)
  - `"cta_selection"`: `oncoref.cta_selection` (`results`, `profiles`, `sequence_dispositions`, `contract`, `coverage`; OncoRef 1.22–1.36 contract, preserving independent profile attributes such as normal maxima, tumor replication, adult Disposable pattern, and developmental evidence; `CTA` targets vs `Unsafe` only — `Other` and `Unassigned`/`Unassessed` are never folded into `Unsafe`)
  - `"target_selection"`: `oncoref.target_selection` (`results`, `metadata`, `definition`; OncoRef 1.37+ rebuilt set-based catalog; `CTA` targets vs `Unsafe` only — `Cancer Antigen only`, `Reproductive Antigen only`, and `Other` are never folded into `Unsafe`)
- Reject `definition="extended"` when `cta_policy != "legacy"` and raise a clear
  `ImportError` if the installed `oncoref` version lacks the requested policy module.
- Record a structured `cta_policy` receipt inside `manifest["expression"]["cta_policy"]`
  (`selector`, `provider`, `policy_id`, `reference_id`, `source_snapshot_id`,
  `oncoref_version`, `counts`, `n_cta_genes`, `n_unsafe_genes`, `unsafe_only_forbidden`,
  `other_is_not_forbidden`, `observed_peptide_blacklist_independent`, and `unsafe_gene_ids`
  for non-legacy policies).
- Preserve `cta_reference.parquet` with canonical `Ensembl_Gene_ID`, `included_in_definition`,
  and for policy-backed runs `policy_category`, `policy_classification`, `policy_id`,
  `reference_id`, and independent profile attributes (serialized as parquet-safe columns).
- For non-legacy policies with an explicit `Unsafe` set, annotate `peptides.parquet`
  with `has_policy_unsafe_match` (only `Unsafe` genes, never `Other`/`Neither`) and
  `policy_categories` while keeping donor-resolved observed-peptide blacklists completely
  independent and hard exclusions.
- Extend `verify_evidence_bundle()` to validate recorded `cta_policy` metadata,
  `cta_reference.parquet` policy/reference identity and `CTA`/`Unsafe` disjointness,
  and replay `_peptide_summary()` consistently while preserving 100% backward
  compatibility with archived bundles that predate `cta_policy`.

## 2. `#893` — Conditional deprecation notice for promoted top-level CLI commands

- Add `via_export: bool | None = None` to `hitlist.cli_export.run_export(args, cmd)`
  defaulting to `getattr(args, "command", "export") == "export"`.
- Emit `Note: hitlist export <cmd> is deprecated; use <replacement> instead.` ONLY when
  invoked via `hitlist export ...` (`invoked_via_export is True`).
- Pass `via_export=False` from `hitlist.cli_samples` (`_handle_samples`,
  `_handle_proteomics`, `_handle_expression`).
- Add unit tests verifying silent stderr (no deprecation note) for `hitlist samples`,
  `hitlist proteomics`, and `hitlist expression`, while preserving deprecation notes for
  `hitlist export samples`, `hitlist export bulk-proteomics`,
  `hitlist export line-expression`, and all other legacy `hitlist export` subcommands.

## 3. `#881` — Build-worker cleanup after parent-only `SIGTERM`

- Add `hitlist/_worker_lifecycle.py` defining:
  - `ParentTerminated(SystemExit)` (default exit code `128 + signal.SIGTERM`, i.e. `143`),
    which inherits from `BaseException` so `except Exception:` cannot swallow it while
    every `finally:` block and context manager unwinds.
  - `graceful_sigterm_scope()` context manager that installs a `SIGTERM` handler ONLY
    when running on the main thread, on platforms with `signal.SIGTERM`, AND only when
    the existing `SIGTERM` handler is `signal.SIG_DFL` (preserving custom caller signal
    handlers, `SIG_IGN`, and non-main-thread behavior; restoring `SIG_DFL` on exit).
  - `terminate_process_pool(pool, timeout=5.0)` helper to promptly terminate and join
    owned `ProcessPoolExecutor` worker processes on cancellation (`BaseException`).
- Apply `graceful_sigterm_scope()` around:
  - `hitlist._source_scan._parallel_sources` (ensuring `_stop_workers(jobs)` and
    `TemporaryDirectory` cleanup run on parent-only `SIGTERM`).
  - `hitlist.mappings._prefetch_proteomes_supervised` (fixing the `aborted` bug where
    cancellation via `KeyboardInterrupt` or `ParentTerminated` previously called
    `pool.close(); pool.join()` instead of `pool.terminate(); pool.join()`).
  - `hitlist.mappings.build_peptide_mappings` parallel `ProcessPoolExecutor` loop
    (terminating owned mapping workers promptly on cancellation instead of waiting for
    `shutdown(wait=True)`).
- Document the graceful-termination contract and uncatchable termination limits
  (`SIGKILL`, OOM-killer, container/host power loss) in `hitlist/_worker_lifecycle.py`
  and `docs/` (where build/CLI behavior is documented).
- Add bounded real-subprocess regressions in `tests/test_worker_sigterm.py` (no corpus
  or large allocations):
  1. Parent running `_parallel_sources` emits worker PID receipts -> test sends
     `SIGTERM` ONLY to parent PID -> verifies parent exits with `143`, every owned
     worker PID exits promptly, owned `hitlist-scan-*` scratch directory is removed,
     and a pre-existing published generation remains untouched.
  2. Parent running mapping worker pool (`_prefetch_proteomes_supervised` and
     `ProcessPoolExecutor` via `terminate_process_pool`) emits worker PID receipt ->
     parent-only `SIGTERM` -> verifies child exits promptly.
  3. Caller custom `SIGTERM` handler, `SIG_IGN`, and non-main-thread execution are all
     preserved untouched.
