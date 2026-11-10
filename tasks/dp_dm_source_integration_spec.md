# Integrate source-qualified HLA-DP/DM immunopeptidomes

Date: 2026-11-10. Codex / GPT-6. Isolated branch based on Hitlist43e2266 (2.1.9).
Issue: https://github.com/pirl-unc/hitlist/issues/895.
Source recovery and independent verification are in the Presto branch
codex/mhc2-conditions-20261110, family2026-11-10_1454_codex_dp-dm-source-recovery.

## Verified source and intended behavior

Use original PXD017154 processed MSF databases, not converted mzIdentML pass flags.
All32 original checksums pass;32 read-only SQLite query exports preserve native
confidence, protein mappings and modifications. Primary supplemental Methods p2
resolves DP401/402 to DPA1*01:03 and DP10 to DPA1*02:01. The original mzIdentML
Sample fields independently match all32 acquisition/sample labels in the SDRF.
The methods specify high-confidence PSMs; native TargetFPRHigh=.01 and native
ConfidenceLevel=3 implement that criterion. All69,327 such rows are preserved.

The conservative source candidate subset additionally requires an unmodified
canonical peptide of10-512 residues, a unique visible rank-1 identification,
complete native protein references, and no occurrence in the high-confidence
no-DP control. This gives21,547 peptide-by-biological-sample rows /10,143 distinct
sequences across10 DP samples. There are also17 distinct control peptides;70
ordinary sample rows overlap these controls. These are not measured affinity
negatives. This subset is not an exact reconstruction of the paper's quantity
analysis and does not imply all excluded matches are false.

## Implementation

- Reuse `scan_supplementary` and its existing peptidoform/provenance contract.
  Export one source-pinned compressed CSV per biological sample, so the existing
  file/peptide/MHC identity never collapses different samples. Preserve original
  file hashes, native PeptideID/SpectrumID and acquisition support in contributor
  fields and a source asset manifest; do not multiply training rows by PSM count.
- Curate PMID33025005 with10 named DP biological samples and exact heterodimers.
  Preserve the shared CD74/CD80/DP-engineered HeLa background. Encode the
  experimental additional HLA-DM transduction as present vs explicit none relative
  to that background; do not imply absence of all previous transductions. Preserve
  source references and biological versus technical replicate counts.
- Keep no-DP control records in separate evidence/companion artifacts; never put
  them in the default positive ligand table. Do not make intensity or affinity
  targets from Mascot scores, spectral counts or nondetection.
- Register the source through the existing supplementary manifest. Retain row
  has_ptm=False only when derived from native modification tables. Original
  high-confidence modified/unmapped/alternative-rank evidence stays recoverable.
- Check package-data inclusion for compressed CSVs and keep the source package
  small. Do not replace the shared editable install or touch another worktree.

## Verification and publication

Trace source rows through `scan_supplementary`, provenance registration, sample
attribution and condition joining into Presto's common MS adapter. Test exact
counts by sample, complete alpha/beta identity, source-qualified chemistry,
explicit DM presence/absence, control exclusions, and no acquisition inflation.
Verify all serialized candidate rows against the independently replayed source
bundle, then run format.sh, lint.sh and test.sh under repository rules. Bump the
package version in the PR; publish through the repository's normal release flow.
Search for concurrent updates before choosing the final patch version.

No biological training or comparator claim is made by this data integration.
Presto must still apply original-source/sequence overlap controls and freeze its
new treatment/control comparison before fitting. Existing frozen cohorts and
checkpoints stay reproducible. The binding/MS/processing objective remains open.

## Artifact locator

Modal volume presto-checkpoints:
`2026-11-10_1454_codex_dp-dm-source-recovery/source-recovery-v1.tar.gz`
SHA256 b8ecc4413dde2f76a271d81f6f8359580d11ea794bfd61f0ae13813888a1abcc.
Original native files:
`2026-11-10_1454_codex_dp-dm-source-recovery/native-v1/` (6,397,444,096 bytes).
Local source root: /private/tmp/presto-dp-dm-source-20261110.

Implementation detail: register the shared original accession in the existing
lineage table; keep specimen/donor identity unknown rather than turning A/B
source labels into proven independent physical specimens. Actual per-peptide
acquisitions remain recoverable in native PSM contributor fields. The first
local test launch could not read the sandboxed memory probe; retry the same
guarded test command with normal host visibility, without changing its guard.

The first complete default suite ran 3,820 tests: 3,729 passed, 91 failed.
One new test read packaged source data without its required integration marker;
move that source-census check to the integration phase. Existing failures include
mhcgnomes 3.70.2 parser/serotype changes and its deprecated cytosolic spelling,
plus stale editable Hitlist metadata (2.1.4 versus imported 2.1.9). Preserve the
failure log, prove baseline behavior, and qualify in an isolated release
environment; do not downgrade packages or alter the shared editable environment.
Version 2.1.11 is reserved for this PR because pending PR894 owns 2.1.10.

## Publication review — 2026-11-10

The source integration and common export verification pass on all 21,547 rows.
Original native PSM support, protein annotations and acquisition/sample fields
are compared field-for-field with the recovered source. Presto resolves every
exact heterodimer and retains DM absent (12,375) / present (9,172). No affinity
target or inferred specimen identity is introduced.

`./format.sh`, `./lint.sh`, isolated-environment `pip check`, and wheel asset
verification pass (all 12 source assets match). The full local default suite
had 3,729 passes / 91 failures; 9 existing failures reproduce on clean base.
Follow-up: https://github.com/pirl-unc/hitlist/issues/897. The new real-source
census check now has the required integration marker. Subsequent guarded
isolated test launches stopped before pytest for insufficient host memory;
no guard was lowered. Full post-fix pytest/integration qualification remains
open. Do not merge/release this source PR on the strength of the adapter audit.

Version 2.1.11 reserves the next patch after pending PR894's 2.1.10. No PyPI
release occurred. Presto's overlap audit embargoes 8,338 rows against protected
presentation/Atlas identities; 13,209 clear those specific checks, but affinity
and other reserved cohorts remain unchecked. These are source candidates, not
independent TRAIN additions or an improvement in predictive performance.
