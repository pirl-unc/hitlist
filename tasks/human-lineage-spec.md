# Primary human donor lineage — 2026-10-16

The user prioritizes donor/specimen independence, source lineage and assay-context
applicability before further fitting. This PR restores a directly reported donor
axis for the benign HLA Ligand Atlas; Presto #126 remains the next separate PR.

## Evidence and boundaries

The published 9,053-row historical cohort includes 1,113 Atlas observations
(876 train, 116 validation, 121 test). Its IEDB tissue-level rows cannot uniquely
identify donors. The source release 2020.12 explicitly maps each sample-hit row
to peptide_sequence_id, donor, tissue and HLA class. Use that primary relation,
not a reconstruction from genotype or sequence matching. See
https://hla-ligand-atlas.org/data and DOI 10.1136/jitc-2020-002071.

A source-reported donor/tissue/HLA-class group is a reported sample group, not
proof of a unique physical specimen, aliquot or raw acquisition. Those axes
remain unknown unless separately evidenced. Do not add inferred donor identities
to historical IEDB observations or silently replace their frozen inputs.

## Implementation contract

1. Pin all three primary tables with exact compressed hashes/sizes and provenance
   through datacache. Record primary documentation/paper URLs and the CC-BY-4.0
   license. Preserve complete raw rows, logical row positions and table links.
2. Extract reusable Atlas reading into one all-tissue source API; keep the existing
   essential-tissue reader's behavior intact as an explicit filter. Preserve source
   typing independently of peptide restriction. Add donor/source/sample-group IDs
   and an evidence basis; physical specimen and acquisition identities stay unknown.
3. Validate reference integrity, class, row identity and source immutability. Frozen
   files must be discoverable/shared through datacache. Add source-backed fixtures
   and meaningful malformed-source regressions.
4. Register a clean, immutable audit recipe before executing it. Publish full native
   observations, donor/sample summaries, label policy and a source registry. Assign
   complete donor groups deterministically to train/validation/test using a fixed
   label-blind hash ordering (15/3/3 for 21 donors). Audit donor, sample-group and
   peptide overlap; physical specimen/acquisition independence stays inconclusive.
   This is a prospective source cohort, not an evaluated model benchmark. Never
   mix it with the old IEDB cohort as if independent evidence.
5. Include a readable historical-source gap report that preserves all old IDs and
   partitions. Candidate donor matches, if inspected, remain candidates, not IDs.
6. Review, version, format/lint/full tests, final-head CI, merge, clean-main deploy,
   public download/hash/regression verification. Preserve old published artifacts.

## Validation and limits

- All source sample-hit rows retained once; raw fields and lookups reproduce.
- Donor partition coverage is complete; each source donor belongs to one split.
- Peptide overlap is measured independently and is not required to be zero for
  this donor-only claim. Report positivity-only supervision and unknown physical
  specimens/acquisitions; no broad training-readiness or model-quality assertion.
- Existing essential-tissue counts and rows stay identical on the same sources.
- Publish the executed recipe, exact version/commit, runtime and memory, hashes,
  source tables, output tables and an independent verification receipt.
