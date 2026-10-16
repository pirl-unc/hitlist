# Replan after actual corpus integration failure — #694

The v3 integration run completed with 94 passes and two curation failures.
Keep the failing tests unchanged. Do not merge #692 or call the corpus qualified
while the source re-audit is unresolved. Independent Presto PXD fitting continues;
it uses a different pinned study and is unaffected. The running #693 export is a
baseline from producer 1.76.0 and must retain the discovered limitations.

PMID 32350084: inspect all 79,406 captured rows and 196 distinct elution comments.
The comments name individual cell lines or explicit sets of lines. Compare names
with the primary article/tables and existing 26 curated samples. Use existing
source-backed statement/group mechanisms when evidence supports attribution;
preserve multi-line ambiguity and do not equate a motif-predicted restriction with
the complete cellular allele set or donor identity. Record residual unresolved
rows and their reasons rather than stamping the entire study resolved.

PMID 33981976: the captured 11,049 rows include 11,043 blank comments and six rows
with a treatment-effect statement. The prior `no_row_discriminator` verdict
described a different 11,043-row capture. Check the original article/supplements
and exact source assays before deciding what the six statements identify.
Blank comments do not establish untreated controls; an effect statement is not
automatically exclusive evidence of one arm. Record unresolved treatment scope.

Keep downloadable source acquisitions and full audit tables in datacache, pin
their hashes, and include small actual-source regression fixtures in Git. Replay
every affected row with source IDs retained; compare counts, attribution and
missingness before/after. Repeat format/lint/full tests and the unchanged full
integration suite on v3. Update the PR around its final publication plus curation
scope and link #694. No data-generation metadata is rewritten retrospectively.

The full replay exposed #696: equivalent complete class-II pairs in reversed
chain order fail the reported-typing compatibility test. Fix that exclusion
boundary by matching both permutations while retaining locus/species/allele-depth
and mutation checks; different partners must still contradict. The original
index classified this study's explicitly motif-predicted restrictions as unknown.
Add the source-backed predicted-evidence curation for future rebuilds, and report
replay of the captured evidence separately from refreshed classification. Use
the established `unperturbed — DMSO vehicle control` convention so solvent control
and drug arms remain distinct coarse treatment categories; the direct condition
control field remains vehicle. Re-measure every row before claiming resolution.
