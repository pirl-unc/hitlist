# PXD024809 observation source — Hitlist #845

Base: current main4f4e398 (2.0.39), isolated branch
`codex/pxd024809-observation-20261025`. No shared-environment installation.
The scientific source0813 audit and registered scoring-lineage0844 audit live in
the Presto modular worktree. Implement only after the source headers and exact
join rules are independently reconciled; preserve all unresolved claims.

## Public surface and bounded scope

Add `write_dilution_observer_package(source_directory, destination, *, assets)`
and `verify_dilution_observer_package(directory)`. Use the existing independent
copy-on-write `snapshot_file`, streaming Parquet sinks, deterministic IDs and
whole-source/typed-replay verification. Register a `dilution_observer` package
family/revision with a distinct envelope, logical catalog entries, exact physical
bindings, provenance links and EvidenceStore dispatch. Reuse CLI query/catalog
handling. Do not create a second data/metadata system or a training recipe yet.

The source receipt pins original publication/project/inventory, aligned quantity
TSV, ten scored-run TSVs and master library. Validate safe relative filenames,
declared SHA256/bytes, unique identities and roles; never overwrite a destination.
Original assets remain immutable snapshots with independent inodes. Preserve all
headers and source row/cell literals. Unknown or changed formats fail explicitly.

## Required grains

Use shared field names and schemas where meanings agree. Keep source-specific
columns under their literal source representation when no exact alias exists.

- Assets and original records: one pinned source file and one ordered original
  table row/XML/JSON assertion. Positional headers/cells and source locators survive.
- Protocol assertions: processing/acquisition/instrument/LC/matrix/spike stage,
  nominal grid, library construction, and explicit unknown parameter/material
  claims. Contradictory twofold versus1:10 assertions remain separate/queryable.
- Nominal standards and levels: full intended ten-standard panel, original heavy
  isotope/position form, five solution-concentration levels with exact units and
  source coordinates. Missing purity/lot/injection amount is not filled in.
- Acquisitions: original filename/source run identifier and filename-derived
  nominal-level/index relation, with explicit precision. Never infer a specimen
  or independent biological replicate from a filename.
- Aligned quantities: one original precursor/run slot, preserving zero/NA,
  intensity, source feature/form/group IDs, charge, RT/bounds, m_score and ranks.
  Do not call an aligned quantity an independently identified signal.
- Scored features: one original per-run feature/candidate row, retaining target/
  decoy state, source feature and run IDs, sequence/form/charge, raw intensity,
  scores, transition aggregation and all original cells. Feature IDs are text;
  values larger than2^53 must never pass through floating-point conversion.
- Library transitions and form memberships: preserve every target/decoy transition
  and original precursor/form identity; report each intended standard's target,
  decoy-only, absent or unresolved membership. An absent query is not a negative.
- Quantity/feature links: source-ID/run/form/charge-checked relationships with
  original locators, unmatched/ambiguous states and exact/rounded/different
  quantity/score comparisons. Do not join by nearest RT/intensity/sequence.
- Material context/links: reuse shared context schemas to distinguish JY HLA eluate
  from post-isolation standards. Source HLA species/class/alleles describe matrix
  origin, not standard restriction. Preserve sample/aliquot/recovery uncertainty.

The final concrete column schema and identity tolerances must be frozen after
0844's structural audit and before compiler execution. Preserve parsed numbers
alongside raw literals; no source-layer imputation, filtering or censor thresholds.

0844 namespace finding: library TransitionGroupId is not the scored/aligned
precursor namespace. Preserve both namespaced identifiers. Exact full molecular
form/charge/target-state membership can relate the library to those records,
subject to independent replay and ambiguity handling. Numeric-ID equality must
not be used as a library provenance link; the initial failed comparison remains
in the scientific audit. Aligned-to-scored feature IDs are a separate relation.

## Verification and publication

Small synthetic fixtures must exercise missing/zero/absent standards, distinct
isotope forms/charges, large feature IDs, duplicate/ambiguous IDs, decoys, source
conflicts, unknown sample identity, altered original bytes and tampered typed rows
even after manifest rehash. Exercise public EvidenceStore queries and provenance
joins; tests are software checks, not biological calibration evidence.

On original files, independently replay all source records, membership/feature
joins, raw/typed quantities and context claims. Register the materialization and
replay under Presto experiments before execution. Keep resource limits20min/stage,
3GiB RSS,1GiB reserve+32MiB margin and50ms process monitoring. Use the package's
clone-aware snapshots rather than recopied inputs per fit.

Run required format.sh, lint.sh and test.sh without relaxing memory guards.
Freeze a patch-version bump, publish the PR, pass hosted gates, merge and finish
tested PyPI publication. Do not call the implementation done before deployment.
No local shared environment update. A quantitative calibration recipe is a
separately specified later task after likelihood/selection semantics qualify.

## Concrete compiler contract after independently verified family0844

Previous goal turn was progress: Presto50a2af4 publishes the source-lineage audit,
all70closure entries verify against committed bytes, and Hitlist845 has the
verified findings. Presto CI40179955279 remains live at this continuation's start.

Use a pinned source revision `pxd024809_dilution_observer_v1`, package envelope28
and catalog37. Source assets are the17 nonempty admitted original files: project,
article, inventory, both pipeline files, aligned table, ten scored tables and
master library. The failed empty API response is historical retrieval evidence,
not an original data asset. Optional `assets` maps exact admitted basenames to
local paths; it cannot override expected hashes, roles, schemas or the panel.
Snapshot all sources independently. Validate exact headers, widths and hashes.

Typed tables, with deterministic source-scoped primary keys:

1. `assets`: original filename, role, URL, digest, size, license/provenance status.
2. `source_records`: one original TSV row or addressed metadata assertion,
   source asset/locator/line and ordered literal headers/cells JSON. Original
   metadata bytes are also retained unchanged.
3. `assertions`: subject/context, original-source pointer, predicate, literal value,
   assertion status and details. Preserve full project/protocol fields and source
   paragraphs, alongside explicit qualified matrix/species/HLA/method claims and
   scoped unknowns. Keep the dilution conflict as two reported claims.
4. `nominal_standards`: all ten protocol sequences, intended exact heavy form and
   heavy-leucine position; no invented assay observation or restriction label.
5. `nominal_levels`: five original Table1 factor/sample/value/unit claims, raw
   numeric concentration plus parsed value. They are nominal solution amounts.
6. `acquisitions`: exact run/filename/run ID, source record, filename-derived level
   and DIA index, material context, unknown physical specimen/aliquot status.
7. `library_precursors`: namespaced library group and exact form/charge/decoy,
   literal group label, transition count and first original record.
8. `library_transitions`: every original transition ID/precursor link, original
   record, product m/z, charge and library intensity (raw plus parsed).
9. `scored_features`: every source feature/run/precursor ID as text, full form,
   charge/decoy, raw/parsed intensity, RT/bounds, rank, m_score and d_score; exact
   form-library link with explicit missing/ambiguous status. All other fields
   remain queryable in the lossless source record.
10. `aligned_quantities`: every original aligned slot, source IDs/form/charge/group
    label, raw/parsed intensity, RT/bounds/scores/ranks and literal states; optional
    form-library link. Label missingness remains separate from chemical identity.
11. `feature_links`: one disposition per aligned slot (or one per candidate when
    duplicate IDs make the link ambiguous), candidate count, optional scored ID,
    exact ID/run/form/charge/target crosschecks and an acceptance status. No fallback
    based on RT, intensity or similar sequence.
12. `comparisons`: six named raw/scored field comparisons for each unambiguous ID
    match, with raw strings, exact decimal delta and the audit's fixed tolerance
    label. Equal numbers do not create a confidence or censor label.
13. `standard_memberships`: intended standard to every exact library form at each
    charge, with target/decoy state; one explicit absent membership if none exists,
    never fabricated per-run zero measurements.
14–15. Shared `material_contexts` / `material_links`: project, post-isolation
    standard panel, JY HLA eluate matrix and acquisitions. Validate the existing
    material graph; no inferred specimen or aliquot identities.

The source protocol describes DIA1 and DIA2 with different window definitions.
Preserve the full protocol and filename-derived method index. Do not label the
two acquisitions per nominal level exchangeable technical or biological
replicates. Protocol reports workflow default settings; retain that report
separately from unqualified executable configuration and actual per-run values.

Stream library rows/source records through existing8192-row sinks. Hold compact
library and scored-feature indices only; no raw transition-array accumulation.
Use Decimal for source numeric states/comparisons and keep invalid/nonfinite/raw
literals, rather than changing source zeros or NAs. IDs never use numeric parsing.
The compiler must preserve ambiguous IDs as relations; tests explicitly exercise
same-form alternative charges, decoys and differing source namespaces.

Verification recompiles from pinned originals and checks schemas, all batches,
counts and immutable contract, even if a caller rehashes altered typed tables.
Fixtures monkeypatch only the pinned source contract, identifying themselves as
synthetic software checks. Real-source materialization/replay is separately
registered before execution; it must reconcile audit0844, not merely re-run the
same compiler. Required public query joins cover original record/asset,
acquisition, library precursor, aligned/scored links and material contexts.
