# PR 2 specification draft — MHC assay results and source graph (#735)

This PR follows the merged and published PR 1 catalog. Bump 1.82.18 to 1.82.19, then format, lint, full test, PR, merge, and PyPI publication. It must not change the legacy source corpus or call any row training eligible.

## Grain and ID

One v2 observation is one existing retained assay result in exactly one of the three v4 MHC partitions. Use its existing `provenance_id` as the source-backed observation key; assert uniqueness within and across partitions rather than inventing an assay/donor identity. Source assay IRI remains a separate identifier and can repeat. Contributor links use source-record IDs and can be many-to-one. A deduplicated result still exposes every original IEDB, CEDAR, or supplementary source record.

## Atomic evidence package

Publish a self-contained, content-bound v2 evidence generation containing a
manifest, typed MHC observation/measurement tables, the contributor relation,
and all original v4 source assets. Hard-link immutable input files into private
staging when possible and copy across filesystems; publish by atomic directory
rename after verification. The manifest makes source and derived files one
logical package. The query layer in PR 6 must resolve registered relations
without caller-managed sidecar paths. A partial write never becomes current.

The PR 1 catalog describes logical types but does not yet carry dataset-level
license/access or concrete source release hashes. Add those to the generation
manifest in this PR (and to the catalog's source descriptor if required for
introspection), using `unknown` where a source did not establish them. Do not
turn an unverified upstream export date into a release claim.

Do not copy all 71 native columns into a new wide table just to avoid an apparent
sidecar. Preserve the exact native Parquet files and contributor
`original_fields`/`source_row_values` under the same evidence package, keyed by
stable row identifiers and content hashes. Typed projections add meaning;
original cells remain inspectable and can be round-tripped. A schema or hash
mismatch fails the build, rather than dropping or coercing a field.

Ship a complete 71-column field disposition map: each native column is typed
here, explicitly deferred to the material/protocol migration in PR 3, or
retained as source-only with a reason. `peptide_modifications`, `has_ptm`,
species axes, sample tags and curator annotations remain discoverable through
the package even if PR 2 does not normalize their final semantics. Reject any
new source column without an explicit disposition; no silent projection.

## Typed projection

For each result retain: evidence family, native partition and source population; source assay/reference IDs; peptide and MHC values at their reported precision; method/readout and modality basis; raw qualitative tier, outcome/admission status; raw quantitative text, parsed value, original unit, censoring/inequality; source result/grain IDs. Keep `tested_mhc`, `candidate_mhc_set`, reported `mhc_restriction` and independently evidenced peptide restriction as separate claims. Unreported, not applicable, explicit absence, unresolved and conflicting statuses must not collapse. Do not infer donor/specimen/acquisition IDs from assay rows, sample labels or candidate sets.

Project peptide modifications/PTM, non-peptide ligand status and potential
contaminant status as typed fields. These are eligibility inputs, not new
assays or inferred negative labels.

Core native-to-logical bindings: `provenance_id` -> observation ID;
`assay_iri` -> source assay ID; `reference_iri`/`pmid` -> source reference;
`source` plus `supplementary_file` -> source origin; `peptide` -> peptide
sequence; `qualitative_measurement` -> reported result;
`quantitative_measurement` -> raw numeric text; `quantitative_value` -> parsed
value; `measurement_units` and `measurement_inequality` -> reported qualifier;
`assay_method`, `response_measured`, `assay_modality` and its source -> method,
readout, modality and classification basis. Keep the raw native names in the
field map for round-trip and avoid implying that `response_measured` alone
defines a normalized quantitative endpoint.

For this PR, keep the native `mhc_restriction` text as `reported_mhc` on every
row. Emit `tested_mhc` only for binding rows whose current evidence is
`experimental`; keep `candidate_mhc_set` separate and preserve its source
precision. Leave peptide-level restriction unassigned in this migration unless
a directly supported source claim is added with its evidence link in PR 3.
`restriction_evidence=monoallelic` or `predicted` does not turn a candidate
into an experimentally established peptide-level restricting allele.

No polarity or family promotion: the 23,537 negative/unestablished MS rows, 1,560 structural rows, and 1,440 non-MS ligand rows stay in `mhc_other` and remain queryable. Quantitative binding endpoints such as IC50, KD and half-life retain their distinct source endpoint and unit. Unknown methods retain the source text and an explicit unknown classification.

## Verification

- Stream the frozen v4 native Parquets with bounded memory. Assert 4,486,991 MS, 814,268 binding, 26,537 other rows in the pinned generation and report per-source counts (IEDB/CEDAR/supplements). Assert no source/observation link is orphaned or silently collapsed.
- Reconcile each v2 observation ID and source assay ID to one native row. Assert modality and outcome classification parity unless a source-linked change ledger explains each difference.
- Round-trip every original stored value via the pinned native asset, including nulls, and every contributor raw JSON cell and relationship. Never equate an assay count with independent biological samples.
- Offline fixtures: IEDB/CEDAR duplicate, supplementary overlap, positive and negative MS, structural, non-MS ligand, censored affinity/stability, empty method, unresolved MHC, null versus literal empty text. Validate unknown fields and malformed schema fail loudly.
- Full v4 integration on the corpus-capable runner; publish a machine-readable reconciliation plus a readable report with changed IDs/cells, exclusions and source/partition breakdowns. A changed count requires explicit reviewed evidence and a baseline update, not an adjusted expected number.

## Risks to resolve before code

Check whether `provenance_id` is unique across all three native partitions and whether every retained row has it in the exact v4 artifact. Check the v4 manifest's source row/field hashes and how `source_row_values` map to reported assay cells. Check that source assay IDs remain nullable without making result IDs nullable. Determine whether current `mhc_restriction` reflects a tested molecule, sample candidates, or a restriction claim in each family; preserve raw text and provenance when the role cannot be established.

## Pinned v4 inspection completed 2026-10-22

The hash-pinned public v4 bundle installed and verified locally. Metadata reports
4,486,991 / 814,268 / 26,537 rows, with 5,327,796 distinct `provenance_id`
values and no duplicates across partitions. Its 9,381,101 contributor links
have zero orphan IDs, and every retained observation has at least one link.
The exact `source` partition counts match #735. Each MHC partition has the same
71-column native schema. These are artifact facts, not biological independence
or peptide-level restriction claims.

`ms_outcome_status` alone cannot select positive presentation: `mhc_other`
contains 2,995 `positive_result` rows because structural and non-MS ligand
assays can be positive within their own methods. Family and modality are
required in the admission rule. Binding contains both positive and negative
qualitative results. The v4 sample rows also show `mhc_restriction` and
`mhc_allele_set` can have the same text even when `restriction_evidence` is
`unknown`; this must not become a measured peptide-level assignment.
All 5,327,796 rows have a peptide and assay IRI. The 535,026 curated
supplementary MS rows have blank `assay_method`/`response_measured` and are
admitted by a curated ligand-table basis; do not fill in a reported instrument
or method. Only 261,715 of 814,268 binding rows have a parsed quantitative
value, so quantitative binding eligibility must be narrower than binding
evidence availability. These coverage figures come from the pinned v4 files.
The MS partition has 4,466,022 distinct assay IRIs across 4,486,991 results,
so assay IRI cannot serve as the row key. Binding and other partitions happen
to have one assay IRI per retained row in this generation; the schema must not
assume that remains true.
