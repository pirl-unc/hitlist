# Canine sample curation in YAML — #672

## Problem and contract

The new canine importer hardcodes tissue groups and source-specific donor,
developmental and health annotations in Python. Tissue categories belong in
YAML under the repository rules. Inferring the adult Labrador cohort from a
plasma/urine label also hides a biological assignment in the format adapter.

Move the reviewed facts into each source's `canine_bulk.yaml` entry: tissue
group mappings, named complete sample contexts, a default context, and explicit
source-sample-ID assignments for the five adult-fluid experiments. The adapter
only extracts sample IDs/tissue labels and applies those curated values.
Unmapped tissues keep a blank group. Keep donor identities unresolved; no
new biological claims, pool expansion, input acquisition or SQL view changes.

All ten normalized sample columns must remain identical for the 38 Pfizer
pools and 18 Labrador samples. Schema-1 bundles remain readable using their
frozen metadata and database without consulting the changed live catalog.

## Implementation and verification

1. Show regressions for catalog-driven tissue/context values and a plasma-like
   label that must not silently acquire adult demographics.
2. Move the existing facts into YAML, including exact adult source IDs 10478,
   10479, 10480, 10481 and 10829. Keep import operations source-format-only.
3. Check catalog context completeness and references in the curation tests.
   Verify context metadata survives in the frozen manifest.
4. Recurate both byte-pinned real sources to new temporary directories; compare
   samples by their unique IDs with the previously verified 1.71.0 bundles.
5. Bump to 1.71.2, refresh editable metadata, run format/lint/full unit tests,
   pandas 2/3 focused tests, strict docs and complete release CI.
6. Publish 1.71.1 before merging this PR. Merge the fully green final head,
   validate clean-main artifacts, publish and compare public PyPI bytes.

## Review

Both initial regressions failed before implementation. The 61 canine/reference
tests pass on pandas 2 and 3; format, lint and strict docs pass. Real-source
recuration preserves every field in all 38 Pfizer pools and 18 Labrador samples,
matched by unique sample ID. Both old bundles also pass offline verification.
No SQL view or bundle schema changes were needed. Full local unit and release
CI receipts are recorded in the task review and PR.
