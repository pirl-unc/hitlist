# PR 3 specification: material, condition and protocol evidence

Parent #733, issue #736, verifier defect #743. PR 2 merged and published as
1.82.19. This PR targets 1.82.20 and precedes the bulk/expression migration.

## Source inputs and row grains

- Use the published, hash-pinned v4 MHC package as the assay keyspace: one `provenance_id` per assay result. Do not duplicate its 5,327,796 MHC rows.
- Use the corrected human cohort `training.parquet` and `identities.parquet` as a separate, verified assertion source. Both have 4,071,463 rows at the human observation grain. `training.parquet` has 161 columns: the 71 native v4 columns plus 90 context, protocol, lineage, and quality fields; `identities.parquet` has a 27-column identity/lineage projection. There are 2,042,689 nonempty `condition_id` claims. Join to v4 by exact `provenance_id` after checking uniqueness and all biologically material fields for discordance; any unmatched or conflicting row gets an explicit audit outcome. The source lacks condition IDs on 2,028,774 rows, which must remain unresolved.
- Use `pmid_overrides.yaml` as a versioned curation source. A material arm ID is namespaced by study PMID or submission ID plus its curated `condition_id`; no globally unique arm identity may be inferred from a bare condition name. Keep every `ms_samples` entry, including unprofiled arms, as a curated material/arm assertion. Keep the original YAML in the package by hash.
- Keep **both** the cohort's historical curation YAML and current curation YAML as distinct source revisions. The cohort manifest pins SHA-256 `5223fc63...` at commit `95c3216`, while current main is SHA-256 `f563a8f1...`. The later changes add explicit control absences, soluble-MHC capture context, and source citations. Join cohort rows to historical arm assertions first; expose newer assertions as versioned amendments with a change ledger, never by silently rewriting the cohort source.
- Use `specimen_lineage.yaml` and the human cohort lineage fields as provenance for experimental origins. Donor, specimen and acquisition IDs stay unknown where so reported. Do not treat assay IDs, PMIDs, sample labels, line names, or allele sets as such IDs.
- Give each condition factor a stable source-backed claim ID, arm key, type, value, status, source precision, citation, and optional dose/duration/order when reported. One arm may have multiple simultaneous factors. Blank is unreported, `none` is sourced explicit absence, `unspecified` is applied but unnamed, and `mixed` means alternative conditions rather than simultaneous factors.

## Relations

- `studies`: study-level source and citation context.
- `material_arms`: one curated arm (including unprofiled). Owns display label, source and condition status; distinct from specimens.
- `arm_factors`: one asserted factor token per arm, including knockout, knockdown, overexpression, variants, transfection, transduction, cytokine, drug, infection, stimulation, antigen exposure, culture, background, material state, labeling and explicit control. Keep the raw scalar field and source locator alongside the token. Preserve combination semantics on the arm.
- `arm_mhc_claims`: distinct roles for selected restriction, sample candidate set, independent cellular genotype (with named cell and locus completeness), and capture/expression context; no automatic promotion to peptide restriction.
- `protocol_claims`: study or arm scoped capture/IP, instrument, acquisition mode, fragmentation, labeling, search engine, FDR and quantification method. Per-arm override references the exact overriding source claim and study default remains separately available. The public v4 reported assay method/readout remain on the assay and can be linked to a protocol context without inventing a run.
- `assay_arm_links`: each v4 assay result gets at most one resolved arm link from an exact published cohort `provenance_id` claim or independently validated direct source mapping. Ambiguous, pooled, missing and source conflict have explicit statuses and raw evidence. Sample labels and paper-level defaults are never sufficient for a unique arm link.
- `lineage_claims`: preserve reviewed origin IDs and explicit unknowns per axis. Separate lineage context from assay link and material arm.

## Physical package and initial typed schema

Augment one verified PR 2 MHC package into one new atomic generation. Keep
its native and typed MHC files at their existing paths; add a `cohort_native/`
directory with the four hash-pinned native cohort assets and its manifest,
plus the historical and current curation YAML snapshots. Retain the exact PR
2 manifest as a versioned predecessor and make the new top-level manifest
inventory cover every file. The eventual catalog/query API opens this one
top-level generation; callers do not supply any sidecar path to query it.

Initial tables:

1. `typed/curated_studies.parquet`: one row per source study and curation
   revision, preserving the reported study label, source context and arm
   resolution explanation without assigning row-level material identity.
2. `typed/curated_arms.parquet`: one row per `(curation_revision, PMID,
   condition_id)`; include the YAML study/arm ordinal, source hash, raw label,
   condition text/status/evidence/reference, control role, combination and
   profiled status. The arm key is a curated condition key, not a donor or
   specimen identity.
3. `typed/arm_factors.parquet`: one row per source field/token and revision,
   preserving raw scalar value, token order or simultaneous membership,
   explicit absence and `unspecified` statuses, source scope and citation.
   Mixed alternative conditions are not treated as simultaneous.
4. `typed/arm_mhc_claims.parquet`: separate selected restriction, sample MHC
   candidates, cellular genotype, locus completeness and MHC expression or
   capture context. Preserve reported text and evidence role.
5. `typed/protocol_claims.parquet`: study defaults and arm overrides as
   separate claims with scope and citation. Instrument, IP antibody,
   acquisition mode, fragmentation, labeling, search engine, FDR and
   quantification are named fields. A study default is not called a per-run
   observation.
6. `typed/assay_context.parquet`: one row per cohort assay, keyed to the PR 2
   `observation_id`, with cohort row index, attributed arm key/status/basis,
   cell type and line, source tissue/species context, the protocol fields at
   exported precision and every lineage axis/status. It must carry the
   original cohort `sample_attribution`; downstream recipes set their own
   link precision threshold. No `material_id` or `run_id` is fabricated.
7. `typed/curation_amendments.parquet`: the 35 source-backed changed cells
   between cohort-time and current curated arm revisions, with old/new raw
   value, field, arm key, citation and revision hashes. Absent new values do
   not delete old claims.
8. `typed/lineage_claims.parquet`: one reviewed arm/axis claim from the
   cohort's lineage registry, including explicit unknown axes, source IDs and
   citation. The native registry remains intact.
9. `typed/arm_control_links.parquet`: one source-backed comparator relation
   per control/target arm and revision. A generic `untreated` role never
   creates a comparator link by itself.

The remaining cohort columns are still source queryable through the native
training table and a complete field disposition map. Do not assign a typed
endpoint or biological role to a legacy derived classification without
reviewing its meaning. A later request may project native fields explicitly
with its provenance path; it must not silently drop them.

## Data quality and conservation

- Exact file hashes of all source inputs. Catalog lists concrete relations, grains and required references. A package generation is atomic and self-contained; no runtime caller-managed sidecar path.
- Complete source-field disposition map for all 161 cohort training fields, the 27 identity fields, and every declared curated study/sample/condition/lineage key. Each maps to typed relation or an explicit retained source field and reason. Reject a new unreviewed key.
- All source rows and claims remain in the package. Round-trip values, including empty string versus null, native IDs, citation, precision and source scope. Assay counts remain unchanged.
- Reconciliation reports link cardinality, source cohort overlap/mismatches, resolved/unresolved/ambiguous arms, factor counts by type, protocol field coverage and lineage status by axis. Flag any evidence disagreement without replacing the older claim.
- Offline tests cover simultaneous KO plus infection, explicit `none` control, mixed alternatives, study default overridden by arm, genotype versus candidates, selected restriction versus peptide restriction, ambiguous arm, unprofiled arm, duplicate source IDs and hostile schema/hash changes.

## Preimplementation checks and remaining risks

1. Verify all 4,071,463 cohort provenance IDs are present in the v4 package and that one source ID maps to one row of the same evidence kind. Compare peptide, MHC, PMID and assay ID where available.
2. Determine whether the cohort was built from the exact pinned v4 generation or a reproducible subset with changes beyond the corrected fields. Preserve conflicting claims if any.
3. Inspect existing export provenance for `condition_id` and `sample_attribution`; a label appearing in the v4 row is not by itself arm proof. Find source of the 2,042,689 cohort links.
4. Count YAML study/sample keys, condition tokens and protocol fields, including studies without PMIDs and duplicate labels, before fixing table keys.
5. Confirm source licensing/access metadata for the cohort and curated YAML without inventing rights.
6. Fix #743: PR 2's verifier currently requires equality with the installed
   catalog schema version, so a future catalog bump would invalidate a valid
   historical package. Bind verification to the package's declared supported
   schema and own field map instead.

## Pinned source checks completed 2026-10-22

The corrected human cohort `identities.parquet`, `manifest.json`, and
`source_contexts.tsv` were downloaded from the tagged public release and
independently matched to their committed SHA-256 receipts. Every one of the
4,071,463 cohort `provenance_id` values appears in the pinned v4 MHC keyspace;
the remaining 1,256,333 MHC results lie outside that human cohort. Exact
keyed comparison across all cohort rows finds zero differences in peptide,
reported MHC, PMID (including null), or assay IRI. The cohort can therefore
contribute source-backed context to existing MHC results without changing their
assay grain or measurements. Its 27-column native schema has no donor,
specimen or acquisition IDs resolved. The curated YAML currently has 216
studies and 807 sample arms, each with a `condition_id`; 37 study keys and 45
sample keys appear. The `donors` study key appears in 11 studies but is
documented as unread in the current loader (#444), so it remains a source
assertion and cannot populate individual donor IDs without reviewed links.
The release's `training.parquet` also matches its receipt and contains the
full 161-column human assay/context table. Therefore PR 3 must ingest or
retain this file, not try to recreate the 90 additional fields from the much
narrower 27-column identity projection alone.
The cohort manifest's four v4 input file hashes exactly equal the pinned v4
MHC source file hashes. Its `specimen_lineage.yaml` hash equals current main,
but its `pmid_overrides.yaml` hash does not. `git show 95c3216` reproduces the
cohort's pinned curation hash. The 55-line net diff to current main consists
of reviewed control absence and soluble-capture amendments; source revisions
must remain separately auditable in PR 3.
All 2,042,689 cohort condition assignments resolve to 489 distinct
`(PMID, condition_id)` arms in the current curated YAML; there are zero
unregistered pairs and zero assigned conditions without PMID. The other 318
curated arms remain registry assertions without a linked row in this bounded
human cohort. Include them as arms, without manufacturing assay links.
Of the linked rows, 48,975 carry `sample_attribution=class_pool`; keep that
attribution basis with the relation and do not silently call it a direct
material/lineage observation. The other attribution categories likewise remain
queryable, so later recipes can set their own evidence threshold.
An arm-by-arm diff finds 807 stable arm keys in both revisions, with 35
changed cells on 27 arms: 16 `condition_mhc_context` additions of
`soluble_mhc`, eight explicit knockout absences, three explicit knockdown
absences, and four pairs of improved evidence class/reference. Eight of those
27 changed arms link to 23,064 rows in the bounded cohort. A versioned
amendment relation can show every current gain without rewriting the 1.78.0
training source or inflating assay counts.
A full keyed comparison of **all 71 native v4 fields** across all 4,071,463
cohort training rows found zero differing cells (including nulls). The 90
extra cohort columns are additive at that exact assay grain. This is a
stronger conservation check than comparing only identity, MHC and peptide.

## Verification and release

Use offline fixtures to exercise status, revision, factor combination,
genotype/MHC-role separation, protocol inheritance, source-schema drift,
duplicate IDs, and hash tampering. On the pinned full inputs, independently
reconcile all 5,327,796 MHC assay IDs, 4,071,463 cohort rows, 7,646,178
cohort contributor links, 807 arm keys per revision, 2,042,689 condition
assignments, and the 35-cell curation amendment ledger. Report field coverage
and null/uncertainty by axis; audit all 161 native cohort fields and exact
contributor payload retention. A changed input generation must fail its pinned
contract or produce a separately reviewed reconciliation, never silently
adjust expected counts. Bump 1.82.20, run `./format.sh`, `./lint.sh`,
`./test.sh`, review the diff, require final-head CI, merge and deploy from
clean main before starting PR 4.
