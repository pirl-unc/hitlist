# Corrected corpus generation

Released producer **Hitlist 1.78.0** rebuilt all source indexes and peptide mappings
in a separate datacache generation. Build recipe commit
`fb64a0c199d91d07f8f7e1fd0da1189b91488a08` is preserved at
`experiment-corrected-corpus-20261016`. The original v3 corpus and human baseline
remain unchanged. Run ID:
`a6933c31825caf35475ba38ae243551bb815d65ba03ee93d9ddb768869e4b5a0`.

All six stages succeeded with the pinned Linux dependencies, eight CPUs, 64 GiB
RAM, 512 GiB scratch and no GPU. Source hashes are unchanged: two raw IEDB/CEDAR
exports and 26 supplementary assets. Raw export dates remain unverified. Only the
previous human Ensembl 112 reference bytes/indexes were reused, with complete
receipt/artifact verification and matching cache-library versions. Observations
and peptide mappings were independently rebuilt.

| Stage | Seconds | Peak runner RSS (bytes) | Receipt SHA-256 |
| --- | ---: | ---: | --- |
| Environment | 1.88 | 220,585,984 | `c35aee65c53aec92263398e834fcb0128d75776b2919089530970e9d4c1998a9` |
| Sources | 71.89 | 239,017,984 | `0466162b040d6f03651b0bda17745babdac3277f48818206167dfa23d97d0663` |
| Indexes | 2,795.44 | 32,390,795,264 | `04190777afcfab4aaec17e3c0c544bea61f1fc072bb44cbbde1038e56bcf8cef` |
| Reference | 44.41 | 223,924,224 | `6b3cd59bb79213745d998fd6ebaa3d9d67e24cb23ef633b680fe89dfd8f75592` |
| Mappings | 706.32 | 5,372,858,368 | `727eb4bf4ca52e12ad5ed5a486df3c672308e7e573c0ffd9a6af3a10158aefe5` |
| Snapshot | 17.89 | 229,474,304 | `41e7a01a86e1382cc60c9008488035dd9035d289156db717c412b87fe81ed140` |

RSS excludes worker processes. Each stage's JSON here includes the receipt and
remote file pins; canonical receipt bytes reproduce the listed hashes. Public
`build-receipts.tar.gz` also contains the independently downloaded original
receipts, logs and exact recipes. These measurements do not describe model fitting.

## Changes from the original generation

The corpus contains 4,486,991 MHC-MS observations, 814,268 binding rows and 26,537
other assay rows; mapping covers 1,506,664 unique peptides. Comparison of all 71
stored observation fields verifies identical peptide/source/assay/provenance keys
and row order. Exactly 79,406 rows from PMID 32350084 changed in one field:
`restriction_evidence`, from `unknown` to `predicted`. Measurements and all other
stored observation fields remain equal. Binding, other-assay, contributor, bulk
proteomics, expression and peptide-mapping parquet bytes are identical to v3.

The comparison preserves every changed cell with citable source/assay IDs, PMID,
reference title and reference IRI. Start with comparison-summary.json and the
field/study counts; the complete changed-cell TSV stays in datacache and the public
release. The comparison streams 8,192 rows at a time and fails on schema, row-count
or identity/order changes. Nulls/NaNs compare equal; values in the TSV use JSON
encoding so missingness is distinct from literal strings.

Source-arm attribution and class-II pair interpretation also operate during
training export. Their actual effect belongs to the separately pinned corrected
human cohort, not an inference from this raw-index comparison.

## Downloads and limits

[Complete corpus v4](https://github.com/pirl-unc/hitlist/releases/tag/ci-corpus-v4)
contains the ten native portable files plus the comparison, build receipts,
qualification notes and publication hashes. Native manifest SHA-256:
`2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7`.
All 18 public assets were independently downloaded through datacache and
hash/size-verified, and the native complete-generation inventory passed its
consumer verifier. [public-verification.json](public-verification.json) records
all durable URLs and hashes. The first immediate post-publication request returned
a transient GitHub 404; the unchanged canonical URLs then verified successfully.

This is a reproducible software corpus, not a qualified training split. Mapping
is limited to human Ensembl 112; other missing references remain explicit.
Optional DepMap matrices are absent. Unknown experimental origins, donors,
specimens and acquisitions remain unknown. Positive elution is not detectability
supervision. #678/#680 remain open for endpoint and independence qualification.

Format/lint and 20 focused qualification/comparison/recovery tests pass. The local
full-suite memory guard refused insufficient capacity; it was not bypassed.
Full CI, final-head review, merge and clean-main PyPI distribution remain required.
