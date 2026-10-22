# PR 6 draft — typed T-cell evidence generation (#738)

Prerequisite: PR 5 merged and published as 1.82.22. Branch from that exact
main commit, bump to 1.82.23, and append to its verified external generation.
The two pinned October 20 source ZIPs are 45,120,781 and 14,506,291 bytes,
with SHA-256s in `hitlist/data/tcell_audit_sources.yaml`; local originals at
`/private/tmp/iedb_tcell_full_v3.zip` and
`/private/tmp/cedar_tcell_full_v3.zip` match. Their mutable public URLs are
ingestion references, and the output bundles the exact source bytes.

## Identity and native preservation

One observation per IEDB assay ID, 578,529 total; CEDAR's 157,263 shared IDs
are contributor records, not extra observations. Keep source reference ID,
source row number, exact raw double-header field map, and original ZIP/CSV
bytes. Preserve all 161 IEDB and 162 CEDAR columns through the raw source
snapshots; CEDAR alone has an `Epitope | Mutation` field and source-specific
IRI columns. The mutation field is populated on 30,639 CEDAR assays. An
independent full-row comparison found all 157,263 shared IDs identical across
their 158 common columns; CEDAR-only fields still carry additional information.
Seventeen of the existing 55,844 bounded training IDs have a populated CEDAR
mutation field, so the typed projection must retain it rather than losing it
through the IEDB-only projection. Keep its source meaning and eligibility
implications explicit; do not silently re-label or exclude those assays (#750).
Declare these differences instead of assuming identical schemas.
Preserve all native fields and
declare field disposition: typed projection, source-only/raw, derived, or
unknown. Reject duplicate IDs, incomplete source rows, changed hashes and
unexpected CEDAR-only IDs. Do not silently drop a row for missing peptide,
MHC, donor or numeric measurement.

## Typed result and context

Store source-reported assay method, response/readout, five qualitative tiers,
quantitative raw string, parsed numeric value when valid, unit and inequality,
subject counts, antigen object/epitope relation, effector/APC names and culture
conditions, in-vitro process, TCR name, and MHC restriction name/evidence.
Retain a raw-field relation or locator for fields outside the typed surface.
Group the 578,529 assays into the five audited family counts without claiming
five interchangeable targets. A negative functional response is distinct from
negative MHC presentation and from a negative multimer cell-binding result.

Keep direct peptide sequence, antigen/pool object, reported MHC, and
conservative candidate single-allele assignment separate. No donor/specimen
identity is invented. Source-reported subject counts are claims about subjects,
not independent assay observations. Preserve disagreements between qualitative
result and subject counts as parallel claims and an ambiguity reason.

## Bounded target projections

Reuse the existing source-faithful direct human peptide qualification and
training-bundle policies, yielding the pinned 47,421 functional and 8,423
multimer examples. Explicitly retain excluded/ambiguous reasons for every
availability row, including wrong antigen format, missing identity, modified
peptide, nonhuman source, unsupported readout, subject-count disagreement and
unresolved restriction where the target needs it. The 55,844 trainable rows
are a subset of availability, never a new independent assay count. If the
source-frozen counts differ, stop and investigate before publication.

Preflight on the exact pinned ZIPs: the existing bounded source projections
contain 47,593 functional rows and 8,432 multimer rows. Training excludes 172
and nine respectively, all for reported qualitative/subject-count disagreement;
the retained counts are 47,421 and 8,423. Candidate single-HLA assignment
appears on 2,155 and 1,899 retained examples respectively, 4,054 total.

An independent first-failure funnel over every IEDB assay recovers exactly
56,025 direct human peptide candidates: 511,628 fail the human-origin gate;
4,723 fail antigen linear-peptide type; 2,602 fail direct-epitope relation;
2,258 fail canonical peptide sequence; 1,167 route to the other assay family;
126 fail epitope linear-peptide type. The order is part of the reported
first-failure policy; retain all raw reasons as parallel source facts where
multiple gates fail. The candidate family counts are 28,266 soluble release,
14,357 activation/proliferation, 4,970 cytotoxicity and 8,432 multimer.

## Packaging and verification

Package schema 5/catalog schema 6; embed exact parent manifest and every
predecessor file. Add typed assay, measurement, antigen/restriction, source
contributor and qualification relations. Use stable source IDs, assay IDs and
measurement IDs with key and foreign-key checks. Bundle the original ZIPs with
independent inode snapshots and an exact hash inventory. Independently replay
all 578,529 original CSV rows, CEDAR overlap and family counts, plus the
55,844 bounded examples and their source outcomes. Offline fixtures cover
negative functional versus negative MS, multimer versus activation, peptide
pool versus single peptide, unresolved HLA and conflicting subject counts.
Publish a machine-readable manifest and readable field-disposition/coverage
report. Run format, lint, full tests, final-head CI, merge, clean-main deploy
and PyPI exact-hash check before PR 7.
