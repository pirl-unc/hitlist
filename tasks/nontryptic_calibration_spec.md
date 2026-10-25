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
