# Historical references, then canine evidence

## Scope and release order

1. Finish #658: correct the historical UniProt flat-file taxonomy parser, recover
   the complete 2015_10 human reference, publish its immutable managed asset and
   catalog entry, validate the original observed peptide coverage, merge and
   deploy. Preserve 2026_03 as the pinned default and keep study search-reference
   identity (#654) distinct from sequence compatibility.
2. Address #660 using its existing local patch where recoverable. Add explicitly
   reviewed, offline supplementary inputs and curate the six original canine
   worksheets, preserving every contributor and donor/IP identity. Keep human
   HCT116 host/DLA monoallelic evidence separate from endogenous dog tumors.
3. Address #661 with an explicit species/reference/candidate-policy contract.
   Preserve human defaults and old bundle replay; require canine evidence for
   canine candidate admission and normal-tissue counterevidence. Record a more
   detailed contract after inspecting the current APIs and source manifests.

## Historical reference implementation

The current extractor skips an OX taxonomy line when evidence annotations occur
between the numeric taxon ID and its semicolon. UniSave E9PBK2 entry version 24,
current in 2015_10, reproduces this. Parse the numeric taxon with an explicit
boundary and allow evidence annotations, including wrapped OX lines; do not
admit taxon prefixes, organism-host OH lines, or names containing HUMAN alone.
Keep accession, sequence version, gene and exact amino-acid sequence intact.

Use the real UniSave entry as a provenance-bearing regression fixture and cover
annotated, unannotated, wrapped and wrong-taxon records through archive extraction.
Re-run the bounded archive builder on GitHub Actions: never retain the complete
31.8 GB source archive locally. Retain all official size/MD5 and human count
checks (20,196 reviewed canonical; 128,790 unreviewed canonical), validate the
21,935 reviewed isoforms seen in the prior complete source traversal, and record
the final asset SHA256, byte size, counts, source checksums and builder checksum.

Independently inspect the output FASTA, count unique accessions and sequence
types, compare representative historical entries and audit the 661,142 observed
Bekker-Jensen peptide strings without treating a match as proof of the original
searched database. Publish only verified compact FASTA/receipt/license assets.
Add 2015_10 alongside 2026_03, test both fetch/cache orders and explicit paths,
and demonstrate a real managed download with checksum-verified reuse.

## Verification and release

Bump to 1.68.1. Run format.sh, lint.sh, focused regressions and test.sh without
weakening memory guards. Use full CI if local memory refuses the complete suite.
Require final-head CI, merge by SHA, run deploy.sh from clean main through the
existing release-build workflow, verify its distributions and compare downloaded
PyPI bytes to the tested artifacts. Only then begin canine implementation.

## Boundaries

No vaccine assembly. No implicit human CTA membership, orthology admission,
human normal-tissue substitute, inferred isoform abundance or all-allele labels
for canine multiallelic samples. No modifications to unrelated local work or
shared-environment dependency resolution.
