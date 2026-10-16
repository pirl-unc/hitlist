# Separate pinned corpus reconstruction — Hitlist #680 / #685 / #678

## Inputs and isolation

Build with the published Hitlist 1.76.0 producer (main 1c11cb6) and a frozen,
mutually compatible environment recorded with hashes. Never use the local
artifact-v5 cache or remote September cache as current training evidence. Never
change user checkouts, existing shared caches, MHCflurry or unrelated workloads.
Use a new namespace on the user's existing Modal presto-data volume. CPU only.

The two raw source CSVs have separately verified content pins in
`tasks/corpus-qualification-sources.json`. Their export dates are unverified;
filesystem modification times do not establish export dates. Upload directly
from the existing files into content-addressed transport paths, without making
another 15 GB local copy or overwriting remote files. Validate remote SHA-256 and
size before registration/use. The existing remote exports have different sizes
and cannot substitute for these pins. Raw exports remain private in the user's
workspace; publish derived qualified artifacts, following existing corpus policy.

All reusable acquisitions, registered input identities and output generations use
datacache. Modal volume transfer is transport into that shared storage. Stage
sources/reference receipts and validate them before dependent builds. Pin every
packaged curation/supplement/bulk/expression input through the exact producer wheel
and its source fingerprints; external packaged-manifest supplements must pass
Hitlist's existing hash-pinned asset contract. Record any omitted optional inputs
(e.g. unregistered DepMap matrices); do not silently imply full expression coverage.

## Bounded stages

Use an isolated Linux Python 3.12 image; install frozen requirements and run pip
check. Capture installed versions, source code, options, UTC timestamps, CPU/RAM,
runtime and peak resource observations. No GPU or model fitting in this build.
One job at a time, at most 8 CPUs / 64 GiB RAM and a 90-minute deadline per stage.
Modal requires at least 524,288 MiB of ephemeral disk for this function; the first
200,000 MiB request was rejected before execution. Request the provider minimum,
while retaining the 16 GiB contributor scratch cap and 1 GiB free-space reserve.
Persist each completed atomic generation and stage receipt to the shared volume;
failures must leave previous complete outputs intact and retain actionable logs.
Do not bypass contributor disk/memory safeguards or use stale-cache opt-outs.

1. Validate/register raw pins, acquire/record packaged-manifest supplement inputs,
   then build observation/binding/other-assay/contributor/bulk/expression indexes
   in the separate root. Preserve original and reviewed source/cell/MHC/condition
   axes and explicit endpoint/unit/censoring fields.
2. Acquire and hash the Ensembl release-112 human reference before mapping. Reuse
   only verified compatible references. Build mappings with fetch_missing=False
   and use_uniprot=False; this intentionally qualifies human-reference coverage
   first, with other reference gaps and unmapped rows explicitly reported. Do not
   fetch a broad uncontrolled collection of species references during qualification.
3. Pin the final complete generation, publish the current portable corpus with
   manifest v2 through the released snapshot tool, then independently verify all
   outputs and public downloads. CI migration waits until a real complete corpus
   is available and its integration tests pass; #685 stays open until then.
4. Freeze a human-HLA training cohort with original observation/contributor IDs,
   full reported MHC sets versus resolved restrictions, cell types/conditions,
   assay policies, sources and mapping ambiguity. Publish actual membership and
   output-support coverage; do not count mappings/generated rows as independent
   measurements. Publish readable TSV previews/summary tables alongside parquets.
5. Audit exact peptide overlap, source duplicates, known reused experiments and
   reviewed donor/specimen identities. Keep known reuse together. Unknown axes
   remain inconclusive. The default claim is peptide-disjoint within this captured
   corpus; any stronger independence claim requires its verified restricted cohort.

## Qualification and release

Record downloadable assets, SHA-256, source age limitations, label policies,
partition assignments and leakage/coverage audits. The Presto consumer then runs
its canonical preflight, bounded fitting and held-out baselines against those exact
pins. Passing build/tests alone is not training readiness. Keep #678 open while
scientific qualification or claimed-output evidence is missing. Every code change
lands via a versioned PR with format/lint/full tests, merge and PyPI verification.
This branch will carry reproducibility scripts, actual qualification receipts and
any reviewable workflow migration; independent defects get their own issues/PRs.

## Reference filesystem correction

The remote reference stage downloaded and parsed Ensembl-112, but first SQLite
publication failed because Modal Volumes reject hard links (datacache issue #125).
Keep downloads shared through datacache; build the annotation/sequence indexes on
local POSIX scratch. Publish the complete closed reference cache as an immutable,
hash-verified datacache bundle. Mapping stages consume verified local scratch
copies of that bundle; do not weaken atomic/no-clobber publication or precreate an
empty public database. Preserve the failed stage receipt and original downloads.
