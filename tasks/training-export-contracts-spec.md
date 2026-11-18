# Stable peptide-index export contracts — #912 and #914

Base: main 78bd28b, Hitlist 2.1.12. Candidate: 2.1.13, recheck before landing.

## Required behavior

- Projected mhc_class_label_severity and mhc_class_label_suspect must be
  recomputed from the same peptide/class inputs as a full read, even when
  stale fields exist on disk. Preserve the current biological thresholds and
  inline-PTM handling. Filters and projected results must agree, and helper
  dependencies must not escape the requested columns.
- An empty query must retain every supported requested derived field with a
  meaningful stable dtype. Cover stored-empty inputs and zero hits from
  restriction, gene, peptide, source, serotype and post-load filters across
  observations, binding and other-assay loading.
- Preserve stored/source fields and their dtypes; retain the existing behavior
  for genuinely absent unsupported fields. Explicitly distinguish those from
  supported computed fields. Do not fabricate source evidence for nonempty
  legacy inputs whose required inputs are absent.
- Every result path must reach the same derivation/projection finalization.
  Prefer typed empty predicates over sentinel peptide values or premature
  returns. Avoid mappings/corpus reads merely to manufacture an empty schema.
- Keep public filter normalization policy (#913) and scientific threshold
  changes outside this first release.

## Verification

Reproduce #912 and #914 first with small fixtures through guarded test.sh.
Compare projected/full results, including the historical class-II 33-residue
old-implausible flag, category/string storage, stored/absent annotations,
quality exclusion predicates and helper-column trimming. Empty results must
have stable column order and dtypes; verify a zero-result gene lookup does not
trigger a second mappings read. Exercise all public loaders and representative
combined predicates rather than mirroring implementation details.

Use an isolated locked environment, then format, lint, full test.sh and hosted
corpus/release gates. No test memory-guard bypass. Audit package/lock metadata
after the patch bump, review the final diff and ship via a new PR. Record
actual outcomes and separate source/training claims from software validation.
