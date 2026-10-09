# Concrete species-bundle contract — #661

Build on the reviewed offline curation in #663. Keep schema-1 human bundles and
`write_cta_evidence_bundle` unchanged. Add `write_species_evidence_bundle` and
species-kind dispatch in `verify_evidence_bundle`; no automatic index, download,
OncoRef membership or human Atlas load is permitted.

## Inputs and public boundary

The public writer accepts a local JSON input manifest (default conventional
`species-evidence.json`) and a fresh destination directory. The manifest pins
relative local files by SHA256 and byte size: complete protein FASTA, a Canvax-
compatible schema-1 expression JSON (reference/occurrences/samples/contributions),
candidate sequence groups, scoped Hitlist observation and contributor Parquets,
and explicit normal-tissue evidence. It captures a versioned policy and normal
source descriptor. Paths remain local and frozen; files may be absent only when
an explicitly empty normal input declares missing canine coverage.

The reference descriptor contains taxon, assembly accession, annotation release,
source version and asset hashes, with a canonical JSON SHA256 key matching the
Canvax contract. Require the protein FASTA digest in those asset hashes. Every
occurrence/sample must match the reference key; expression sample taxa match the
reference. Candidate groups identify exact full-sequence SHA256s, empirical
species-expression admission, policy ID and supporting allocation identities.
Every group must resolve to supplied occurrences and supporting expression;
no human-orthology membership is imported. Preserve all occurrence alternatives,
not only selected candidate loci. Full FASTA hits without annotation remain
explicit unresolved mappings.

Use the existing source/host/MHC fields and contributor schema. Observation
identities must be unique and have full contributors. Attach only the explicit
reviewed arm lineage, then freeze it. Missing I/L discrimination is a qualifier
on reported-string support, not proof of exact molecular identity. Exact matches
to reported observation strings do not transfer MS support to nested windows.

## Derived evidence

Map exact observed or explicitly requested peptides over every protein in the
supplied complete reference. Stream one protein at a time; bound input bytes,
protein residues, candidate peptide count and mapping rows. Keep every position
and every source occurrence, including noncandidate and unannotated hits.
I/L-equivalent mapping, when requested, is separately labeled and never becomes
exact-string support. Reference specificity is explicitly scoped to the supplied
reference, not a safety claim.

Retain raw expression contributions and sample namespaces. Versioned policy
fields explicitly name allowed reproductive tissues, required normal tissues,
measurement unit, allowed-tissue minimum, normal-tissue maximum and minimum
normal donor coverage. Never apply TPM thresholds to another unit or mix source/
assay/quantifier namespaces. Gene/promoter support remains at its reported level;
only transcript evidence can establish full sequence-group abundance. Ambiguous
allocations retain lower/upper bounds and all coding alternatives; missing is
not zero. Evaluate reproductive support and normal counterevidence separately,
including other loci encoding the identical full sequence. Report missing normal
coverage and unresolved isoform support independently of candidate membership.

Normal MS data have an explicit target taxon, source/version, coverage declaration
and source records. Reuse positive-MS and nonmalignant heart/brain/lung, two-
distinct-donor, allele-independent sequence exclusions. Reject a human normal
source descriptor for a canine reference; audit unknown tissues/donors, non-MS,
cell lines and malignant tissue as exclusions. Preserve a genuinely empty canine
normal snapshot as missing evidence, never a negative result or human fallback.

## Output and replay

Freeze all supplied files under stable names, along with policy, lineage,
complete mappings, group expression audit, peptide summaries, presentation and
excluded observation tables, tissue audit/risk and forbidden sequences. Canonical
ordering and JSON serialization are deterministic. Hash every artifact, publish
the manifest last and reject an existing destination. Verify hashes and replay
all derived relationships using only frozen inputs and policy; do not consult
live biological registries or original source paths during replay.

Tests must cover wrong references, identical testis/heart proteins, incomplete
annotation, missing normal coverage, gene versus transcript support, mixed units,
ambiguous allocations, human-host DLA, unassigned tumor restrictions, exact versus
I/L matching, nested windows, contributor completeness, distinct donor counting,
resource bounds, deterministic output and offline replay. Run existing human
bundle tests unchanged. Use the smallest real scoped source replay with a full
reference once the contract passes; no synthetic result is a biological finding.
