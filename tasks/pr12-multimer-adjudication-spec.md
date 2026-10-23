# PR 12: first-class multimer source adjudications (#766)

Objective: append the nine reviewed pMHC multimer assay interpretations to the verified EvidenceStore package without changing source outcomes, contributor rows, eligibility, or training examples. This is an information-only generation, version 2.0.4 after 2.0.3 deploys.

## Source basis

- PMID 10618411 / IEDB 1646138: Kotzin et al. Figures 3–4 describe group-level negative gp39–DR4 staining and three weak individual synovial signals. Different aggregation levels; no peptide-level binary target.
- PMID 30568303 / seven IEDB IDs: corrected Nature Supplementary Information PDF SHA-256 `e755519a29e08dc1378e09f1d95f94ac4bc893494df7330b99f6e76bba7a2631`, Table 2, PDF page 15. Evaluable/response counts exactly match IEDB's zero responders; its `Positive` qualitative tier remains unexplained. Preserve Table 1 pretest as a distinct population/condition.
- PMID 25589627 / IEDB 2445272: AACR Figshare Supplementary Table 3 PDF SHA-256 `fa6758ca9d58cf4691dc9b6a0f17575ec88215fb023998a730cb7abe8b264058`. HNF4G-1M column has no positive frequency; measured lines follow one/two multimer pull-down/expansion rounds, while IEDB says Direct Ex Vivo. Preserve both preparation claims and mark unresolved.

## Data contract

- One immutable adjudication row per source assay ID, with stable claim ID, assay family, original IEDB qualitative/count values, primary publication PMID/DOI/URL, exact figure/table locator and pinned source hash where available, reviewed endpoint/population/preparation scope, interpretation code, unresolved alternatives, and training effect `retain_excluded`.
- Source and adjudication are distinct grains. Link nine reviews to existing T-cell availability, outcome, eligibility and contributors by exact source_assay_id. Keep raw columns in their original tables; duplicate a few here only as validated audit anchors.
- Append schema-8 generation to verified schema-7 package using independent copy-on-write snapshots of every predecessor file and an exact parent manifest. Materialize an `assay_adjudication` relation in the EvidenceStore catalog with generic fields that other assay families can use later. Source PDFs stay at publisher URLs; do not redistribute them.
- Reject unknown IDs, duplicate reviews, changed source counts/qualitative, missing source links, changed parent file hashes, or any original training eligibility shift. Require before/after reconciliation: 578,529 T-cell assays, 8,432 multimer candidates, 8,423 multimer examples, 55,844 total examples, nine reviewed exclusions, 0 lost/added source assay rows.

### Minimal schema shape

`assay_adjudication` is a relation at `source_assay_review` grain. The stable
`claim_id` is derived from source-assay ID and a named review revision. The
physical row includes `source_assay_id`, `assay_family`, reviewed source
identity and locator, literal source outcome/count audit anchors, distinct
reviewed endpoint/population/preparation/reagent scope, a controlled
interpretation code, open questions, review status and training effect. Do
not use an inferred peptide-level label field. Peptide, MHC, effector-cell and
contributor values are reached through existing one/many links so their
original assertion precision is retained. Each review's publisher source
asset is identified by URL and hash; no publisher PDF is copied into the
package.

Use a versioned, repository-controlled review input with strict field and
identity validation. Freeze the review input bytes in the generation as a
source receipt. Add all logical fields to the common catalog, one physical
binding, and relationships to availability, reported outcome, antigen claim,
eligibility and source contributors. A direct request by `source_assay_id`
must return the review and allow explicit joins without duplicating T-cell
assays. Historical schema-6/7 packages must report this relation as absent.
Replace the catalog's one-off schema-7 source-population gate with an explicit
minimum package schema on each evidence declaration; this makes future
generations extensible without adding special cases.

### Generation and verification

Extend the schema-7 verifier to accept an embedded predecessor manifest and
verify its listed files without claiming the successor's new files are part
of that predecessor. For a root schema-7 package, keep the exact inventory
check. The schema-8 writer verifies the schema-7 input, snapshots all 123
listed predecessor files plus its manifest, writes the typed review relation,
and records exact file hashes and a parent-manifest hash in a new root
manifest. The schema-8 verifier checks exact inventory, all hashes, predecessor
verification, typed row schema and identities, all source-result and
eligibility anchors, every relationship key, and the frozen counts. Check
independent inodes for the new generation to preserve immutable input bytes.
The review input is interpretation metadata, not another measured assay.

The eight `Positive`/zero rows remain unresolved because the publication
does not explain IEDB's qualitative tier. The Kotzin negative/three row
describes a cohort-level negative with three weak individual signals; it is
not a peptide-level negative target. The GAPVAC Table 2 counts describe
vaccine-induced responses; keep Table 1 pre-immunogenicity separate. The
Hombrink source measures expanded/pulled-down lines, whereas IEDB's
`Direct Ex Vivo` concerns the starting cell preparation. None of these
reviews authorizes training admission or source-field correction.

## Verification

- Red tests for claim table validation, source/eligibility join, all-nine coverage, and EvidenceStore query/join. Tests should catch a contradictory source count and an unreviewed `Positive` admission.
- Build and verify actual successor package from `/private/tmp/hitlist-v2-reference-200-final`, preserving parent and output hashes. Query all nine reviews and original outcome/eligibility rows through EvidenceStore; compare baseline audit and source contributor counts.
- Run format, lint, test, full CI, PR review, merge, clean-main release, exact artifact verification and PyPI hash comparison. Record unresolved status honestly; do not promote any of the nine into training.
