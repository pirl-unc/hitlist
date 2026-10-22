# Biological context evidence: pinned full-source reconciliation

The [Hitlist 2.0 plan](https://github.com/pirl-unc/hitlist/issues/733),
[context subissue](https://github.com/pirl-unc/hitlist/issues/736), and
[MHC package release](../mhc-v2-evidence-results/README.md) define the source
and migration boundary. This report is for the verified 2026-10-16 v4 corpus,
the published corrected human-cohort bundle, the cohort-time 1.78.0 curation,
and the current 1.82.20 curation. The accompanying [manifest](manifest.json)
contains exact hashes, source access/license statuses, source-field
dispositions, relation keys and the machine-readable count audit.

## Conservation and grain

| Relation or source | Verified rows | Grain |
| --- | ---: | --- |
| Parent MHC assay results | 5,327,796 | Retained assay results across all three MHC partitions |
| Corrected human-cohort assay context | 4,071,463 | Same result IDs as a bounded MHC subset |
| Corrected cohort contributor links | 7,646,178 | Original source record to bounded cohort result |
| Curated studies | 216 per revision | Source study, retained twice for provenance |
| Curated arms | 807 per revision | Study plus condition key, retained twice |
| Condition-linked cohort results | 2,042,689 | One cohort assay row with a curated arm key |
| Control-to-comparison links | 88 across both revisions | Explicit curated comparator assertion |
| Concurrent arm factor claims | 2,339 across both revisions | One source field/token/revision |
| Arm MHC role claims | 2,074 across both revisions | One reported candidate, genotype or context claim |
| Scoped protocol claims | 342 across both revisions | Study default or arm override |
| Reviewed lineage axis claims | 168 | One arm and identity axis |
| Curation amendments | 35 | One changed native field on one arm |

The package retains every parent MHC native and typed file, all four native
cohort assets, both PMID curation YAML revisions and the reviewed lineage
YAML. The cohort's own manifest pins the *exact* four MHC input file hashes in
the parent package. Its 4,071,463 result IDs all resolve to parent MHC rows;
the other 1,256,333 MHC results remain present without a cohort context row.
Every cohort result has a source contributor, no contributor points to a
missing cohort result, and no assigned condition points to an unregistered
historical arm. No assay is counted twice when a context, contributor,
factor or lineage claim is joined.

## Label and context checks

An independent keyed comparison of all 71 native v4 fields across the
4,071,463 cohort results found zero differences, including nulls. An
independent rowwise comparison of the typed assay-context projection found
zero differences across its 90 extended cohort fields, three contributor
summary fields and observation ID. The cohort-time and current PMID curation
revisions retain the same 807 arm keys; 35 source-backed cells change on 27
arms, and eight changed arms link to 23,064 results in this bounded cohort.
Both old and new values remain inspectable through the source YAML and typed
amendment relation.

`sample_attribution` remains on every cohort result. The 48,975 condition
links marked `class_pool` retain a coarser `class_pool_context` status; they
are not relabeled as direct sample statements. A blank intervention remains
unreported, while a curated `none` is an explicit absence. When an arm has
simultaneous gene, infection or cytokine claims, each remains a separate
factor under that arm's combination status.
Dose and duration remain in raw condition text when no structured source
field establishes them; this package does not infer numeric exposures.
Presenting-sample candidates, selected restrictions, independent cellular
genotype and MHC capture context keep distinct roles. Protocol values retain
study or arm scope and do not create a run ID. The 11 study-level `donors`
source entries remain in native curation; without row-level identity evidence
they do not create donor IDs.

All 4,071,463 cohort assay rows have unknown donor, specimen and acquisition
identity. Reviewed experimental origins resolve on 57,188 rows and remain
unknown on 4,014,275. The typed lineage relation records the reviewed arm
claims and their citations; the native lineage registry remains packaged.
These row counts do not count independent donors, specimens or acquisitions.

## Reproduction

The package writer accepts a verified PR 2 MHC package and the four-file
corrected human-cohort bundle. It checks source hashes and schemas, validates
curated conditions and lineage, streams the large assay and contributor
relations with bounded memory, verifies all ID links, and atomically publishes
one self-contained generation. `verify_context_evidence_package` rechecks its
inventory, hashes, parent MHC manifest, cohort source manifest, typed schemas
and row counts. The source assets are ingestion inputs; the output package
contains them under one manifest. The source license status remains
`unverified` where the source did not establish a license.
