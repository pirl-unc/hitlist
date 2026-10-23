# Binding assay protocol evidence (#721, first source-reviewed release)

## Objective and pinned inputs

Make assay conditions discoverable at their reported scope through the same
`EvidenceStore` package and query surface as MHC results. Preserve every
predecessor file and assay result. Start with the complete corrected human
cohort's binding/IC50 population; source-review the large open-access Sidney
et al. 2017 study (PMID 28481943) without turning study-wide ranges into
per-assay constants or reported IC50 into measured KD.

Pinned inputs:

- Public v4 MHC manifest SHA-256
  `2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7`;
  binding partition has 814,268 assay rows.
- Corrected human cohort manifest SHA-256
  `b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`;
  25,896 binding rows include 8,084 literal IC50 rows over 198 PMIDs plus
  PMID-unknown rows. These are assays, not independent experiments.
- PMID 28481943 / DOI 10.1371/journal.pone.0177140, Europe PMC full-text XML
  `PMC5421785/fullTextXML`, SHA-256
  `948816f8584a2779082048ad161bf12d8c8f0ebae2c2a70315d9d22efcaf65aa`.
  Its Methods section `sec003` explicitly reports a purified MHC,
  radiolabeled competitive assay, 0.1–1 nM tracer range, room-temperature or
  37°C incubation for two days, six competitor concentrations spanning
  100,000-fold, and at least three independent experiments. Tracer identity,
  exact per-assay temperature, pH, replicate IDs and uncertainty are not
  supplied there. The article states an approximation to KD under specified
  conditions; the observed endpoint remains IC50.
- PMID 28481943 has 938 matching v4 binding assays, all reported IC50/nM,
  including 910 in the corrected human-source-organism cohort and 28 influenza
  peptide controls outside it. Retain all 938 where the study-wide method
  applies; do not redefine the original cohort to include the controls.

## Design

1. Add a general, hash-bound reviewed-evidence successor over the current
   schema-9 combined package. Its manifest identifies the predecessor and
   preserves every predecessor file with a separate inode; verification
   checks the complete inherited inventory and new files. Historical package
   readers continue to work. The new package is portable and needs no
   query-time sidecar. Reject unsupported or ambiguous predecessor envelopes.
2. Materialize an assay-result-grain `binding_assay_context` table with one
   stable row per v4 binding assay and an explicit review state. Link it to
   `mhc_binding` by observation ID as an optional one-to-one relation; link a
   reviewed row to a protocol ID. Keep source assay ID, PMID, original endpoint,
   reported value, unit and inequality as validated anchors, never replacement
   measurements. An unreviewed row has no invented protocol details.
3. Materialize `binding_protocol_claim` at one claim per predicate and source
   scope. Record literal/range/alternative values, units, applicability,
   evidence basis, article URL, asset hash and XML section locator. Distinguish
   study-wide facts from ranges or alternatives whose exact per-assay value is
   unresolved. Represent unreported fields distinctly from explicit absence
   and not applicable; do not infer absence from silence.
4. Review PMID 28481943 against both the article and exact IEDB anchors.
   Record all supported conditions, the locus-specific capture antibodies,
   and the independent-experiment minimum. Preserve the paper's conditional
   KD-approximation statement as a protocol interpretation, not a new KD
   measurement or training target. Do not assign a temperature, tracer
   identity/concentration, peptide purity or physical replicate to a specific
   assay when the paper supplies only alternatives or ranges.
5. Audit all 25,896 corrected-cohort binding rows by endpoint/method/study and
   all 8,084 IC50 rows by source, comments and reviewed-condition coverage.
   Compare the new review with the frozen cohort by exact observation IDs and
   all original measurement fields, including the 28 same-study v4 controls
   outside the cohort. Publish machine-readable counts and source-linked
   examples. No observation or contributor counts change.
6. Expose both new relations in the catalog/query CLI and document their
   grain, fields, null states and join precision. Support future studies by
   adding reviewed claims to a revisioned source ledger, with no paper-title
   inference or arbitrary sampling.

## Verification and release

- Fixture tests reject wrong source hashes, stale IEDB anchors, duplicate
  observation IDs, unsupported claim status, per-assay promotion of a study
  alternative, missing predecessors, byte changes and forged claim values.
- An independent reader of original XML and v4/cohort Parquet replays all
  reviewed links, counts and claims in the real successor; compare every
  predecessor file digest and verify all catalog queries/joins.
- Run `./format.sh`, `./lint.sh`, `./test.sh`, strict docs, source/release
  distribution checks. Review exact-head CI, merge a version-bumped PR, run
  the clean-main release gate and verify both PyPI artifacts.

This PR establishes one source-reviewed study and complete audit/unknown
coverage. #721 stays open for the other IC50 studies and richer primary
source/supplement curation. The present source does not give per-assay pH,
tracer identity or full concentration bounds.
