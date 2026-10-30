# Build profiling and output validation

This report covers the changes shipped in 2.1.0. The subsequent
[serial build comparison](serial-build-performance.md) uses released 2.1.0 as its
baseline and measures both variants repeatedly on one allocation.

The final forced build took 1563.96 seconds (26.1 minutes); the mapping-fix revision
took 1593.01 seconds. Baselines took 2307.36 and 3320.58 seconds (38.5–55.3 minutes).
All seven Parquet artifacts are byte-identical across these runs, including all source
payloads, curation flags, schemas and row order.

These observations do not establish a reliable 2× whole-build speedup. Host variation is
substantial, the CPU model was unavailable, and earlier runs attempted unintended
reference downloads. The final build's 32.2% lower wall time relative to the faster
baseline is an observed comparison, not a promised speedup on other hardware. The
broader 2× target in #63 remains open.

## Scope and reproduction

The baseline is `dfe15bcb1a5c75a2e1ef225c424e9de89df37d0b` (2.0.44). The initial
block-storage candidate is `6c4765e` (2.1.0). A later candidate, `66716ba`, also removes
per-peptide DataFrame grouping and checks local Ensembl reference availability before
lazy feature access (#865). The final production revision `4e5a696` also removes
remaining pandas 3.1 deprecations (#866). All four candidate runs have complete output
comparisons against the baseline. The later dependency-extra correction selects pandas'
supported Arrow floor; all paired benchmark environments use Arrow 25.0.1. Local unit
verification uses Arrow 26.0.0; it is a separate environment. All installed package
versions and producer archive hashes are in the receipts.

Each full build uses the same 14,719,407,889 bytes of pinned IEDB/CEDAR exports, 26
verified packaged-data assets, supplementary inputs and pinned human Ensembl 112 raw
reference files. Export dates are unknown. Inputs and software pins are in
`tasks/corpus-qualification-sources.json`, `tasks/reliability-benchmark-reference.json`
and `tasks/reliability-benchmark-requirements.txt`.

Preparation copies/verifies sources and indexes the raw Ensembl reference outside
timing. The timed forced build function has fresh outputs, warm raw/reference files and
a cold Hitlist mapping index. GNU time also records the enclosing fresh Python process
in each resource receipt. The intended cache-only policy was not enforced for Ensembl in
the baseline or initial candidate: full sampling proves attempted reference downloads in
mapping and optional gene-symbol backfill (#865). Their wall times therefore include
this uncontrolled network work. The input receipt string describing missing references
as not downloaded records the intended policy, not a verified property of those runs.
This is a full raw-source build with explicitly limited mapping reference coverage. Only
human has matched mapping rows. This is not a qualified biological training release or a
measurement of a complete multispecies reference installation.

Full builds use Python 3.12.1 with identical pinned dependencies apart from Hitlist,
including pandas 3.1.0 and Arrow 25.0.1. Full jobs request/limit 8 CPUs and 64 GiB RAM,
use two mapping workers, cap runtime at 90 minutes, and preserve separate output
directories. The recorded CPU-model probe returned unknown, so equal resource requests
do not establish identical hardware. Subsequent full jobs use non-preemptible capacity
after the first baseline was preempted. `candidate-1` completed on the original
preemptible configuration and is retained as exploratory evidence. Interrupted attempts
never count as totals.

Create producer archives with `git archive --format=tar.gz COMMIT`, then run from the
profiling checkout, with its isolated environment and authenticated Modal:

```sh
HITLIST_BENCHMARK_ARCHIVE=/path/to/producer.tar.gz \
  python -m modal run --detach scripts/reliability_modal.py --label UNIQUE_LABEL
```

Add `--profile` for a separate 10-ms Pyinstrument diagnostic, including worker sessions.
Do not use profiled runs for speed ratios. Initial cProfile samples had invalid
cumulative relationships; they are retained but excluded. Existing MHC
annotation/context/classification caches were inspected; no redundant cache was added on
the strength of initialization costs in a sampling profile.

Use `--test-suite` with `HITLIST_BENCHMARK_TEST_IMAGE=1` for `./test.sh` timings. Both
test variants use Python 3.12.16/Bookworm, two test workers, fresh caches and identical
dependency versions apart from Hitlist. The first test image lacked Git; that failed
attempt is excluded. The legacy image's fresh Git installation also failed on Debian 11
security-repository 404s. The supported test image avoids changing workspace-wide Modal
settings or altering the full-build comparison.

## Complete run measurements

Times are seconds. CPU is parent plus exited/waited children. Peak RSS in this first
table is the parent process lifetime high-water mark. The first candidate used
preemptible capacity; later candidates and both complete baselines used non-preemptible
capacity. Candidate 3 adds direct mapping aggregation and local reference checks;
candidate 4 adds the final pandas deprecation cleanup.

| Run | Variant | Wall | CPU | Parent peak RSS (GiB) |
|---|---|---:|---:|---:|
| baseline-2 | Baseline | 2307.36 | 2178.85 | 30.20 |
| baseline-3 | Baseline repeat | 3320.58 | 3069.57 | 30.25 |
| candidate-1 | Blocks (exploratory) | 2250.94 | 2210.20 | 30.72 |
| candidate-2 | Blocks repeat | 2511.01 | 2412.96 | 30.76 |
| candidate-3 | Direct mapping | 1593.01 | 1575.28 | 30.73 |
| candidate-4 | Final pandas cleanup | 1563.96 | 1556.16 | 30.72 |

## Stage measurements: faster baseline and final candidate

Stage RSS is the sampled parent-process peak, not an allocation delta. Nested Parquet
writes are included in their parent stages and must not be added again. Receipts also
retain process-tree RSS sums; those can double-count shared pages and are not
physical-memory totals. Their descendant counting is additionally affected by
[#876](https://github.com/pirl-unc/hitlist/issues/876), which found child thread IDs
counted as separate address spaces on this runtime; retain those original receipts
for audit, not memory comparisons. Parent RSS and timing measurements are separate.
The full build includes additional joining, deduplication and
generation publication work beyond the named stages below.

| Stage | Baseline wall | Final wall | Baseline CPU | Final CPU | Baseline RSS (GiB) | Final RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| IEDB scan | 731.45 | 521.37 | 717.78 | 518.77 | 23.29 | 23.82 |
| CEDAR scan | 597.61 | 435.39 | 587.05 | 430.62 | 30.20 | 30.72 |
| Protein mappings | 367.49 | 107.73 | 323.87 | 108.83 | 23.57 | 23.08 |
| Contributor publication | 283.15 | 214.09 | 278.09 | 211.42 | 21.00 | 21.47 |
| Supplementary scan | 34.42 | 22.73 | 33.81 | 22.69 | 20.50 | 20.94 |
| Bulk proteomics | 18.76 | 23.64 | 18.49 | 23.75 | 22.96 | 23.50 |
| Line expression | 53.03 | 0.37 | 1.09 | 0.36 | 22.94 | 23.48 |

Mapping completed in 107.73–122.94 seconds in the later candidates versus 367.49–474.10
seconds in the baselines. Its parent-plus-child CPU fell from 323.87 to 108.83 seconds
in the displayed pair. This agrees with the sampled cost of millions of tiny pandas
groups. Optional gene-symbol filling fell to 0.34–0.37 seconds after unavailable
references were skipped before lazy downloads.

## Full sampling findings

The separate full diagnostic recorded 3382.98 seconds in the parent profile and retained
individual worker sessions. Inclusive durations must not be added:

- Scanner: 1947.47 s, including 836.98 s capturing contributor records.
  The latter includes 333.13 s in zlib compression and 309.27 s serializing JSON.
- Contributor publication: 353.60 s.
- Mapping worker: 431.74 s in `map_peptides`, including 242.24 s iterating
  tiny pandas groups and 63.37 s selecting DataFrame columns.
- Optional gene-symbol fill: 68.39 s, including 68.27 s entering lazy reference
  downloads. A second mapping worker independently exposes the same policy bug.

`classify_ms_row`, `resolve_mhc_annotation` and PMID/species context already use caches.
Their sampled parent costs were 3.69, 2.96 and 0.03 s respectively; adding another cache
would not address the dominant costs.

The mapping change accumulates distinct-protein counts and flank agreement while lookup
hits are already grouped by peptide. It preserves repeated positions, terminal empty
flanks, row order and pandas dtypes. Cache-only workers now reject unavailable local
Ensembl data before lazy feature access; optional symbol filling uses the same
availability check. Direct callers retain their existing default download policy unless
they explicitly disable it.

## Selected changes and output conservation

The 100,000-row diagnostic identified record capture as a material scanner cost. A
controlled 10,000-payload codec experiment compared three repetitions per variant.
Median per-row zlib-1 encoding took 0.3315 seconds; bounded 128-row zstd-1 blocks took
0.0266 seconds. Storage including conservative row references was 8,716,064 versus
889,246 bytes. All decoded payloads matched. These are component measurements, not
predictions of whole-build speed.

The implementation bounds each payload block at 128 rows or 1 MiB and the decoded LRU at
4 MiB. A single oversized source row is isolated. Stable identifiers, original JSON
strings, graph traversal, relation/label handling, public schema, scratch limits and
atomic publication are preserved. No data or curation flag is dropped to obtain a
smaller scratch representation.

The full comparison verifies actual retained file hashes, generation source and output
receipts, producer versions and all metadata values. It records individual
location/time/version differences. The deterministic README, which the initial benchmark
did not retain, is reconstructed and verified against its recorded hash. For all four
candidates and the repeated baseline, the seven data files are byte-identical, including
schemas, ordering, nulls, source payloads, curation flags and every column:

| Artifact | Rows | Columns |
|---|---:|---:|
| Observations | 4,486,991 | 71 |
| Binding | 814,268 | 71 |
| Other assays | 26,537 | 71 |
| Contributor links | 9,381,101 | 11 |
| Protein mappings | 6,205,013 | 12 |
| Bulk proteomics | 2,269,288 | 57 |
| Line expression | 42,909 | 18 |

The default `./test.sh` measurements pass 3,692 baseline and 3,715 candidate tests (two
skips each). End-to-end script wall times are 394.12 and 402.76 seconds; the candidate
includes additional regressions. This single pair establishes no test-suite speedup.
Hosted CI separately runs the full integration suite and complete release gate. The
final local unit verification separately passes 3,735 tests (one skip), with both
FutureWarning and DeprecationWarning promoted to errors; these local results are not
used as a paired speed measurement.

Machine-readable evidence is in `tasks/reliability-results/`. Full output files, logs
and diagnostic sessions are retained in the project's `presto-data` volume under
`hitlist-reliability-20261028/<run-label>/`. Use `--compare-before LABEL --compare-after
LABEL --label UNIQUE_COMPARISON_LABEL` to repeat the strict full-output verification. A
byte difference fails the check and requires investigation; the verifier does not
silently sort or drop fields.
