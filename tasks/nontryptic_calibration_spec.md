# PXD024917 known-dose calibration evidence — Hitlist849

Base85ff2c81, isolated branch codex/pxd024917-calibration-20261025.
Presto1816 independently verifies S1/S3/S6/S7 and identifies source contradictions.
Presto1857 is retrieving only the two exact transition exports named by S7.
The published source screen is closed; no model fit is authorized by this spec.

## Public data contract

Add a source package `nontryptic_calibration`, revision
`pxd024917_nominal_ms2_v1` only after exact
export schemas are qualified. Reuse independent source snapshots, deterministic
source-scoped IDs, Arrow tables, the shared EvidenceStore/catalog and material
context graph. No environment installs, main edits or Presto-only parser.

The first implementation unit can proceed before the pending export downloads:
`nontryptic_workbooks.py` reads source worksheets without assigning biological
semantics. Its returned sheets retain name/state/dimension/merged ranges and all
stored cells, including formatting-only cells. Each cell retains its coordinate,
Excel type, literal cached value, displayed string text, attributes, formula and
formula attributes, and a reserialized XML element. Original asset bytes remain
the authority for lexical XML and styles. Missing cells remain absent; numeric
literals never pass through float, so large IDs and exact decimals survive.
Do not evaluate formulas, treat a formula's absent cache as zero, promote row
order into identity, or include phonetic annotations in displayed string text.
Reject ambiguous ZIP members, sheet IDs/relationships/cell coordinates, external
or escaping sheet paths and oversized expanded archives. This reader is internal
source infrastructure, not a qualified package or model-facing recipe.

Implementation order:

1. Preserve workbook source sheets/cells with literal OOXML, original value,
   cached formulas, types/styles, original coordinates and absent-cell semantics.
   A formatting-only cell is distinguishable from a cell with a reported value.
2. Materialize source panel, reported isotope forms and isotope assertions, source
   protocols and nominal amount/signal measurements. Keep S3C transition counts
   and S1 predicted affinities distinct from signal or binding observations.
3. Add exact transition-export tables/lineage only after1857 confirms their
   actual headers, grain and identities. Do not guess fields or C/D/E run links.
4. Integrate package/catalog/public query and provenance relations. Freeze pinned
   asset hashes and the concrete schemas before original-source materialization.
5. Exercise synthetic software fixtures, independent original-source replay and
   all required format/lint/test gates. Bump patch version for the PR.

## Known source semantics to preserve

- S3A: free1H peptide after purification;18 sequences,71 dose rows,213 reported
  positive signals. GLFDQHFRL[+7] has no0.1-fmol row. Do not manufacture a zero.
- S3B: embedded2H hipMHC before IP;9 sequences,45 dose rows,135 actual slots,
  including9 blanks. Missing values are not zeros or inferred detection bounds.
- Amount fmol and summed targeted product-ion signal are different quantities.
  Preserve raw units and values beside any explicit unit conversion.
- C/D/E are ordinal source slots until source evidence establishes acquisition
  identity. All doses/columns for a peptide form one dependence group for a later
  sequence-generalization recipe; no specimen identity comes from a run number.
- S6A valine4.007-Da claim conflicts with S3/S6B nominal+6. S6A omits methionine,
  but S6B E99 supplies an M isotope assignment. Preserve every original assertion,
  affected form and conflict disposition; no silent mass or atom correction.
- S7 hipMHCspiikein spelling differs from PRIDE inventory hipMHCspikein. Preserve
  both names and the precision of their study-inventory correspondence.
- Matrix MHC/HLA selection describes the calibration design, not measured
  affinity, sample-specific restriction or native intracellular concentration.
- Reported AAA/HPLC free-peptide and ELISA embedded-complex assignment methods,
  extraction/spike stage, trigger settings, transition selection, acquisition,
  instrument and source scope remain homogeneous protocol assertions.

## Verification and limits

Tests cover omitted rows versus blank cells versus zero, duplicate identities,
formula/cached-value distinction, unusual source types, tampered raw bytes,
tampered typed values even after rehash, original form contradictions and public
provenance joins. Fixture measurements are synthetic software tests, not model
training or biological evidence. Preserve failed attempts and unknown metadata.

Original numerical materialization must use a separately registered Presto family
with20min/stage,3GiB RSS/startup free+speculative memory,1GiB reserve+32MiB margin,
one numerical thread,50ms monitoring and OS peak check. Local software tests use
the repository's existing test.sh memory admission; do not weaken either guard.
Public PyPI publication currently requires explicit user approval after automatic
review rejected2.0.41; do not repeat or bypass that action for a later version.

## Reader review and independent software verification

Local test attempt1 refused before pytest (2.02GiB versus the unchanged2.5GiB
budget). This is not a test failure or a passing test. Hitlist440 documents why
inactive memory is excluded; preserve that policy. Hosted software CI may verify
the source reader separately from local scientific source qualification.

Review found two silent-loss boundaries in the unpublished reader: a document of
the wrong XML type can become an apparently empty sheet, and duplicate value,
inline-string or formula elements select only their first occurrence. Reject
those inputs. Resolve sheets only through worksheet relationships, validate the
workbook/shared-string/worksheet root types and unique sheets/sheetData containers,
and retain empty-but-valid sheets. Extend synthetic regression fixtures to cover
all these boundaries, including a valid absolute package target.

Freeze this internal reader as a separately reviewable versioned draft PR linked
to849, without claiming source ingestion or consumer training. Run required local
format/lint and retain local test refusal; use the existing hosted test workflow
without relaxing its checks. Read results for the exact new head. Keep the rest of
849 open until original exports, typed source tables and recipe are qualified.

## Qualified source boundaries — 2026-10-26

Presto family1857 has now retrieved the two exact official exports and replayed
42,941 stored cells. The pinned reader passes original-source verification.
The source-schema prerequisite for typed curation is satisfied; quantitative
reconstruction and acquisition identities remain distinct, incomplete claims.

- DirectSpike has 2,592 transition rows, 432 peptide/form/replicate groups and
  six product ions per group. Its separate N-column dose legend must not become
  row-aligned metadata. Preserve 1,296 absent Concat values as missing, not errors.
- hipMHC has 1,092 rows in 252 groups: 140 have six product ions plus precursor;
  112 have precursor only. Preserve the fragment/precursor distinction explicitly.
- All 54 export forms match a unique S6B assertion using peptide sequence,
  labeled positions and total reported nominal shift. This correspondence does
  not repair the S6A isotope-atom conflict. Keep both reported forms and assertions.
- Source-reported S3 signal and derived transition sum must be separate quantities.
  Only 144/213 S3A ordinal candidates match within relative 1e-6; maximum relative
  discrepancy is 46.63%. Product-only S3B sums match 105/117 numeric candidates,
  versus 1/117 with precursor included. Do not choose a numerical matching subset.
- At 100 fmol, S3B values occupy C/D with E blank, while the export labels are
  1/3 with no label2. Do not promote an ordinal column into a physical acquisition.
  A candidate correspondence table may retain form/dose/replicate evidence, but
  exact acquisition_id, specimen_id and aliquot_id remain null without support.

Implement reusable tables for original source sheets/cells, standards, reported
forms/assertions, nominal amounts, observation protocols, S3 source slots and
reported measurements, export transitions, and candidate lineage. Reuse existing
catalog field meanings and the material-context/assertion framework. Include raw
strings/coordinates beside parsed numbers; keep S1 predictions and S3C counts
separate from assay signals. The next recipe can declare a source-reported,
protocol-specific sequence/dose response endpoint with peptide-grouped folds,
without asserting calibrated isotope molarity, complete sum reconstruction,
native MS1 transfer or APC concentration supervision. Freeze that recipe separately.

## Implementation specification — source compiler and package

Use a pinned `nontryptic_calibration.json` contract for the six workbooks and
reviewed source assets. The compiler consumes only these original assets, never
Presto's derived JSON. It returns Arrow-schema-constrained relations and writes
them under `nontryptic_calibration/typed/`. IDs are source-scoped hashes of exact
file/sheet/coordinate or explicitly reported identifiers. No source IDs pass
through floating point. Every raw numeric literal is retained beside its parsed
value; formulas cannot silently become measured values.

Relations and grains:

- `assets`, `source_sheets`, `source_cells`: original asset, sheet and stored cell;
  preserve empty/formatted cells, formula/cache/type/style and original XML.
- `standards`: each S1 panel peptide, gene labels, length and predicted affinity
  as annotation, explicitly not measured binding.
- `isotope_assertions`, `reported_forms`: S6A residue claims and S6B forms;
  store original atoms/shift, labeled positions and S6A arithmetic consistency.
  Conflicts and missing residue assertions are retained rather than repaired.
- `protocol_assertions`: addressed source claims or reviewed interpretations,
  scoped to project/direct/embedded protocol; raw asset and locator are required.
- `source_slots`: sheet/column ordinal with null acquisition/specimen/aliquot IDs.
- `nominal_amounts`, `measurements`: each original S3A/B dose row and each C/D/E
  slot; retain all 348 slots, including nine blanks, without creating omitted rows.
- `transition_counts`: S3C counts with their own grain and dose units.
- `export_forms`, `form_correspondences`, `transition_groups`, `transitions`:
  reported export form, unique-or-ambiguous source-form candidate, explicit
  form/replicate group and original transition row. Precursors and products remain
  different signal kinds. Duplicate rows are flagged, never silently deduplicated.
- `aggregate_signals`: complete sums for the two declared modes, with missing,
  invalid and zero counts; empty product sets have null sums, not zero.
- `measurement_candidates`: form/dose correspondence only; a suffix/column
  equality can be a diagnostic flag but is never an acquisition identity.
- `material_contexts`, `material_links`: project and the two observation designs,
  using the existing shared precision-qualified context schemas.

Validation is layered. Tiny synthetic workbooks first exercise actual source
layout, missing versus zero versus omitted rows, formula rejection, isotope
conflicts, precursor separation, deterministic identities and ambiguous joins.
Then the package wrapper snapshots pinned source bytes and verifies every typed
table by replay. Integrate package-family dispatch, catalog fields/relations,
EvidenceStore bindings and public query tests; bump the current catalog version
without rewriting historical package versions. The existing PR patch version
2.0.42 already covers this unfinished source feature. A guarded original-source
materialization is registered separately before executing the compiler on the
biological files. No response recipe or model training is implicit in the compiler.

## Original-source correction: panel length assertion

Guarded Presto0236 inspect-1 independently confirms S1 B19 sequence AMLGTHTMEV
has ten residues, while E19 explicitly reports9 (numeric, no formula). The source
compiler wrongly treats a contradictory annotation as a fatal schema error.
Retain calculated peptide_length plus reported_length_raw, reported_length_value,
reported_length_status and length_consistency_status. Do not overwrite either
claim, shorten the sequence, drop its measurements or promote formula caches.
Expose raw/status/consistency fields homogeneously in the catalog. Test matching,
conflicting, missing, nonnumeric, fractional, zero and formula length assertions.
Freeze the changed implementation separately from the failed attempt, then run
original verification with an independent check of every new length field and
exactly this one source conflict. Existing numerical/identity gates remain fixed.
