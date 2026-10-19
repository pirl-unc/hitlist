# Species-scoped evidence bundles — #661 design review

This follows the offline observation/contributor boundary in #660. Keep the
existing human `write_cta_evidence_bundle` API and schema-1 verification intact.
Introduce an explicit `write_species_evidence_bundle` mode, with a separate kind
and versioned input contract, and dispatch it through `verify_evidence_bundle`.
No OncoRef human alias resolution, human Ligand Atlas loading, global mapping
rebuild or network access occurs in the new mode.

## Frozen inputs

Require a manifest identifying taxon, assembly accession, annotation release,
source version and exact SHA256/size for a complete local protein FASTA. A
canonical reference key binds every occurrence, expression sample and policy
result to this exact reference. Reject mismatched keys/taxa rather than silently
coercing names. The current Canvax reference, occurrence, sample and contribution
JSON contracts are useful interchange inputs; Hitlist must not depend on Canvax
or duplicate its vaccine/design responsibilities.

Retain every source occurrence with protein, transcript and gene IDs, source
provenance and exact full-sequence identity. Candidate sequence groups are an
explicit empirical canine discovery input, with membership, policy version and
supporting expression records. Human CTA membership and orthology are not an
admission basis. A candidate list must never restrict the self-proteome used for
mapping. Quarantined/incompletely annotated proteins in a full supplied FASTA
remain mapping background and unresolved evidence, not silently omitted rows.

Expression samples retain taxon/reference, study, donor, specimen, library,
assay, quantifier, source, unit, tissue, health and relevant strata. Contributions
retain original allocation identities, all compatible coding occurrences,
gene/promoter/transcript scope, lower/upper bounds and original provenance.
Missing remains missing. Reject conflicting duplicate allocations; do not sum
reprocessed derivatives or gene estimates with transcript estimates. Unit-
mismatched measurements cannot pass a TPM threshold and remain in the audit.
Gene/promoter measurements do not create positive full-isoform abundance.

Require an explicit versioned reproductive-restriction policy and its declared
normal panel/coverage. Keep restricted-in-observed-panel distinct from complete
normal-panel coverage. Normal expression at another locus encoding the identical
protein is counterevidence at the shared sequence group, even if a testis locus
passes the candidate screen. Preserve raw evidence and uncertainty alongside any
caller-reviewed membership; evidence absence cannot become a safety verdict.

## Peptides, presentation and tissue exclusions

Use exact observed or explicitly supplied peptide strings, keeping I/L distinct.
Map them across the entire supplied reference in bounded batches, retaining every
protein hit, position and annotated source occurrence. Match against the FASTA
first; a partial annotation table cannot hide noncandidate proteins. Report exact
and I/L-equivalent results separately if the latter are requested. An ambiguous
I/L identification does not establish exact stereochemical sequence evidence.
Do not infer MS support for nested windows or copy observations onto all candidate
alleles. Keep donor/sample typing and peptide restriction separate.

Accept a frozen scoped observation table and its contributor graph from #660 or
other reviewed Hitlist sources. Reuse positive-MS admission, identity, lineage and
source/host/MHC species columns. Preserve human-host DLA observations as a
separate evidenced context; they do not establish endogenous canine translation.
Retain excluded observations with reasons and exact contributor coverage.

Require an explicit canine normal-tissue MS input, including an explicit empty
snapshot with a coverage/missingness declaration when no such data are available.
Reuse the nonmalignant heart/brain/lung, >=2 distinct donor, allele-independent
sequence exclusion mechanism after validating its taxon and provenance. Do not
load human normal data implicitly. Unknown donor/tissue, tumor, cell-line,
non-MS and unresolved-sequence observations remain auditable exclusions. Carry
expression counterevidence separately from empirical normal-MS blacklist status.

## Portable output and replay

Freeze the reference descriptor/bytes, occurrences, discovery membership,
expression samples/contributions/policy, observations, contributors, lineage,
complete peptide mappings, normal-tissue audit, summaries and forbidden sequence
list. Publish a checksum-bearing manifest last into a new directory. Canonical
ordering and JSON encoding produce deterministic files; no absolute source path
or clock time determines content identity. Verify hashes and recompute identity,
mapping, expression-link and tissue-count relationships using only frozen inputs.
Expose input-byte, reference-residue, peptide-batch and output-byte limits; fail
before publishing if a bound or completeness contract cannot be established.

## Acceptance sequence

Start with an intentionally small offline fixture containing: a testis candidate
and adult-heart locus encoding the identical sequence, a noncandidate protein,
a missing normal measurement, gene and transcript evidence, mixed measurement
units, wrong-reference rows, endogenous dog MS, human-host DLA MS, unassigned
restriction, an I/L alternative and a nested window. Verify all matches and
counterevidence survive, unknowns remain unknown, every contributor is linked,
serialization is deterministic, and verification works with all networking and
live curation providers disabled. Existing human bundle/replay tests must remain
unchanged and passing. Then replay a bounded real canine source slice with a
complete supplied reference and explicit coverage; do not claim raw-spectrum
reanalysis, unique protein attribution or vaccine assembly.
