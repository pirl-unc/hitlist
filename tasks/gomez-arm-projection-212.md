# Gomez-Zepeda source-backed arm projection (#780)

## Objective and predecessor

Replace the erroneous single-arm projections for PMID 38480730 with an
authoritative source-backed context while retaining every historical assay,
source result, membership and package artifact. The predecessor is released
Hitlist 2.0.11, package schema 11/catalog 16, manifest SHA-256
`ade958582dd2eb580f4e02187cd7445621e72fc07ec294a97624a09ba24a2659`.
Every one of its 262 files remains byte identical in the schema-12 successor.

The primary article XML has SHA-256
`3311abf23d895b7f37f18cd8dbf5ea3a273d785d6c34db24b2aaa0b5a1d97260`.
Methods section `Sec24` (Experiment design) distinguishes pooled wild-type
method-development material from the final JY/Raji WT and transfected
cultures and states three LC-MS injection replicates per sample. Section
`Sec15` (Cell transfection) documents distinct S1 and S2 constructs. The
successor embeds this XML and anchors arm claims to those source sections.

## Source conditions and material identities

1. Preserve nine reported arm identities across five materials: JY WT/S1/S2,
   Raji WT/S1/S2, HeLa, SK-MEL-37 and commercial plasma. `sample_group`
   identifies the five materials. No culture, sample or acquisition label is
   promoted to a donor or specimen identity.
2. S3 HeLa, SK-MEL-37 and plasma each identify one material/arm. S3 JY is
   untransfected JY at method-comparison scope; it is not asserted to be the
   same physical culture as an S5 WT culture.
3. For S5, use `ConditionGenotype` to distinguish JY_WT, JY_S1, JY_S2,
   Raji_WT, Raji_S1 and Raji_S2. Verify every original source row against
   `Condition`, `Genotype` and the `BioReplicate` prefix. `Condition` alone
   collapses S1/S2 and is insufficient.
4. A source result is a positive detection in one reported acquisition.
   Absence from another run is not a negative result. A derived peptide may
   retain several measured arm memberships.

## Frozen reconciliation anchors

- Original S3/S5 result rows: 594,916; source-run memberships: 590,615;
  linked derived observations: 68,552; out-of-length source rows: 51,491.
- Derived arm cardinality: 33,156 peptides have one supported arm, 10,840
  have two and 24,556 have three. This yields 128,504 peptide-arm links.
- HeLa 10,356, SK-MEL-37 11,491 and plasma 1,397 are single-material arms.
  The multiple-arm evidence occurs in JY and Raji.
- Current errors: 574 plasma rows are assigned SK-MEL-37 and 149 are assigned
  JY; 674 are blank. All 1,397 have plasma-only source support.
- Of 8,408 JY rows currently assigned the old WT/titration arm, only 1,233
  have unique WT support; 6,820 also occur in transfected arms and 355 have
  no WT source membership.
- `attributed_sample_label` is null for all 68,552 rows. Remove the stale
  curation note claiming it is populated.

## Schema-12/catalog-17 contract

1. Add `gomez_source_membership_arm` at one original membership-to-source-arm
   mapping. Record the exact membership, source asset/row scope, reported
   condition, material, arm ID and mapping reason.
2. Add `gomez_derived_arm_support` at one derived observation × supported arm,
   including source phases, memberships and result counts. Add
   `gomez_derived_arm_projection` at one derived observation with material,
   supported-arm count, nullable unique arm and status `unique_source_arm` or
   `multi_arm_source_evidence`. Query multi-arm support as its own relation so
   a join cannot multiply training weight silently.
3. Add a source-specific nine-arm vocabulary with all arms grouped by
   material, explicit no-transfection claims only for source-established WT
   arms, distinct S1/S2 transfection claims, and primary-source references.
   The historical PMID YAML remains preserved as predecessor curation; the
   source-backed relation is the authoritative Gomez arm projection.
4. Publish a Gomez-only revised assay-context projection while retaining the
   complete historical context table in the predecessor. The projection has
   exactly 68,552 Gomez observations. A study row receives one arm only when
   source support is unique; otherwise it retains material identity and
   explicit multi-arm status with arm-specific labels cleared. Predicted HLA
   never discriminates material or arm.
5. Bind schema-12 `EvidenceStore` queries to the new source-arm and context
   relations; keep schema <=11 bindings historical. Do not change assay or
   training counts or contaminant admission.

## Verification and release

Independently replay original ZIPs and predecessor typed files. Verify exact
parent inventory/hashes, every source condition mapping, all nine arms, all
68,552 derived IDs, the one/two/three-arm counts, correction of all 723 plasma
errors and all 355 unsupported JY-WT assignments, field equality for every
other PMID, unchanged assay/training counts, and no fabricated negatives.
Report before/after counts by source file, material, arm cardinality,
attribution and training admission. Run `./format.sh`, `./lint.sh`,
`./test.sh`, strict docs and distribution gates. Review and merge the
version-bumped PR, then run the clean-main release and verify both PyPI files.
