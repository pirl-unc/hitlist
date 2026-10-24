# Native peptide response recipe with explicit model references (#791 / #788)

Specification before implementation; branch reconciled with upstream PR794 / 7649ffe. Publish as
2.0.17 after the source correction release; never mutate older source packages.

## Implementation choice

Add one registered recipe `tmab_native_response_v1` through existing
EvidenceStore.materialize and verify_recipe_package dispatch. The recipe accepts
one explicitly hash-pinned human reviewed FASTA model reference with provenance.
It snapshots that reference into its immutable output and reuses shared
read_reference/exact_occurrences. This avoids another package family/schema and
keeps reusable curation in Hitlist. Native source catalog fields remain unchanged.

Export original source assertions, all reference occurrences and used full protein
records, sequence/parent/reuse components, condition/peptidoform groups, individual
exact or bounded log-ratio observations with all provenance contributors, all
source dispositions and explicit hierarchical weights. Protein descriptions and
gene names are assertions, never accession IDs or chosen measured origins.

## Fixed semantics

- Read all four source relations and source rows through the public query API.
  Require corrected TMAB v3. No source edits or new label interpretation.
- Use canonical sequence identities even for deferred tentative rows to build
  conservative split components before eligibility. Unknown sequences remain
  explicit. Reference mapping retains all exact positions and candidate parents;
  no I/L substitutions, inferred polymorphisms or unique-parent assignment.
- Join identical sequences, all shared full reference sequences, source gene/name
  assertions and possible reuse into components before filtering. Use sorted
  source-row membership for stable component IDs and a fixed SHA256(recipe_id +
  component_id) modulo20 split:0..13train,14..16val,17..19test. Report no homology,
  donor/specimen or external-study independence. Shared-control label proxies
  remain explicit and may cross these peptide/parent splits; they are not known
  physical specimens or extra biological replicate counts.
- Candidate copies with different source experiment labels or contradictory
  cell/treatment/dose/time values are all deferred. Matching copies retain every
  provenance contributor, one effective observation and unchanged target/qualifier.
- Only existing candidate_exact/candidate_censored treated/control slots train.
  Controls/averages/unknown/tentative/unsupported/nonpositive/missing/overlap/error
  records remain in dispositions. No MHC length/affinity filter or new negatives.
- Response group is peptide form plus normalized literal experiment label and
  typed condition fields. Charge states and reported ratios remain observations
  within that group; their total training mass is one group, not independent
  specimen mass. Loss weights within each split average condition -> component ->
  form -> effective observation. Never average bounds into exact labels.
- Preserve all model-reference alternatives, including unsupported full-protein
  residues. Sequence-response admission and full-parent model support are separate
  flags. A model consumer must explicitly handle alternatives/support.

## Tests and real-data qualification

- [ ] Miniature canonical source fixture: repeated cross-paper values, different
  labels, charge rows, mixed exact/bounds, tentative/unknown identities and controls.
- [ ] Reference fixture: multiple exact origins, repeated positions, source
  descriptions that are not accessions, parent sharing through deferred rows.
- [ ] Verify grouping/split leakage invariants, per-split nested weight sums,
  source conservation, no fabricated exact values, reference hashes and tampering.
- [ ] Register a Presto family before full canonical materialization; independently
  reconstruct source candidates, occurrences, groups, split components and weights.
- [ ] Run format/lint/test, hosted gates, review and publish; maintain source PR793.
- [ ] Freeze/run the next biological baseline and matched exact/sampled fits on
  these recipe outputs; no source audit substitutes for predictive results.

## Review

Pending. Reference ratios are observed native-extract responses, not absolute
concentrations, cell-machinery activity labels, genetic loss or APC validation.
