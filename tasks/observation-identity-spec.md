# Donor-aware training observation identity (#614)

## Scope and contract

Fix the current collision before building contributor retention (#622) and
experimental-origin/specimen audits (#616). The builder deliberately preserves
rows sharing an assay when their `attributed_sample_label` differs. Training
exports must preserve that distinction through filtering, protein mapping
expansion and narrow projections. This change does not assign biological
specimen identities, infer independence or change observation counts/weighting.

Keep the old assay/reference/positional identifier in `evidence_source_id`.
For rows without an explicit `attributed_sample_label`, `evidence_row_id` stays
byte-for-byte identical to its current value. For explicitly attributed rows,
derive a versioned readable compound identifier from the evidence kind, original source
identifier and attribution key, before mapping expansion. Prefer the persistent
curated `(pmid, condition_id)` looked up directly by the original attributed
label, provided that label uniquely names an arm. Otherwise use the original
label scoped to its PMID and an explicit label-fallback namespace. Percent-escape
the source locator and attribution component so delimiters remain unambiguous.
Example: `ms:attributed:v1:http://www.iedb.org/assay/7578387|arm:31844290:mel3_13240_006`.
Never use a
heuristic export sample assignment, line name, genotype, sample grouping key or
row number to distinguish donor attributions on an identified assay.

The lookup uses the complete curation rather than observed multiplicity, so
peptide/source filters, row order and one-donor subsets cannot change a key.
Renaming the display label (and updating its source attribution) while retaining
the curated condition ID preserves observation identity. Duplicate/ambiguous
curated labels fall back to the source label instead of choosing the first arm.
Missing labels, including categorical nulls, keep the old identity. Older indexes
missing assay identifiers retain the existing documented reference/positional
fallback limitations; those keys cannot certify unique observations. This PR
does not silently invent a recoverable original locator for old indexes.

Both `evidence_row_id` and `evidence_source_id`, with `evidence_kind`, survive
projection. Protein mapping alternatives share these IDs, while independently
attributed observations sharing a peptide retain distinct IDs. Ordinary MS and
binding export behavior, canonical indexes and builder deduplication stay as-is.
Document the attributed-ID migration and the distinction between observation,
source assay, curated arm and biological specimen in the public API/README.

## Verification and release

Plan revision after user feedback: replace the opaque SHA-256 observation IDs
with inspectable compound IDs. Keep hashes for artifact integrity, as planned
under #622/#616. Verify delimiter escaping and pin a readable compatibility value.

1. Add failing public-export fixtures for two donors on a shared assay with two
   protein mappings each; compare compact, expanded, filtered and projected
   results. Include an independent assay sharing the peptide and a non-split row.
2. Verify order invariance, unchanged non-attributed IDs, persistent-arm display
   renames, original-label fallback, ambiguous curation and categorical nulls.
3. Add a corpus regression for PMID 31844290 / SLLQHLIGL: six observations, two
   source assays, six observation IDs, three donor labels. Run only this narrow
   real-corpus query locally; full integration validation runs in CI to avoid
   workstation memory pressure (tasks/lessons.md, 2026-09-29).
4. Run format, lint, focused tests and the editable-install verifier after the
   patch version bump. Inspect the diff and pass the full supported-Python CI.
5. Merge the PR, dispatch the clean-main release workflow (which runs format,
   deploy.sh, lint and test.sh --all), verify its artifact manifest against
   clean main, publish the tested wheel/sdist and verify PyPI hashes.
6. Record release evidence and assess the next dependency block across Hitlist
   #622/#616 and MHCflurry #444. Keep their broader acceptance criteria open.
