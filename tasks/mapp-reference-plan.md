# Reference alternatives for shared MAPP records

October22,2026; Codex/GPT-6. Previous goal turn was progress: shared quantity importer in Hitlist draft758, verified complete source generation, metadata issue757. NumericPR756 has merged; first merge current main into isolated Hitlist branch, preserving both contracts/version changes. Full objective remains biological availability modulo precursor abundance; no synthetic or source audit alone completes it.

## Model-input evidence contract

Add a reusable sequence-reference materializer, called by the MAPP adapter, for explicit references and peptide identities. Reference generation, accession identity, full sequence, exact occurrence and measured origin are distinct. Preserve complete modern isoform-inclusive human2026_03 FASTA and historical2023canonical DAT with verified hashes and source metadata. Original Human_UP_2016_02.fasta remains unavailable. Never strip isoform suffixes, infer secondary aliases from another reference, collapse distinct full sequences or reinterpret I/L or noncanonical letters.

Typed relations: references; reference proteins and raw header/record metadata; reference-scoped primary/secondary accession claims; all exact peptide occurrences (including overlapping and additional source-unlisted parents); source parent claims and their resolutions, including missing IDs and resolved sequence-negative candidates; one coverage row per peptide/reference. Each occurrence has literal one-based inclusive coordinates and immediate flanks. Leading source coordinates/flanks have separate comparison statuses. Repeated positions and multiple alias resolutions survive. No join duplicates quantity records, assigns biological origin or supplies parent-mixture weights. Component IDs, if generated, link through shared full-sequence identity within a reference and constrain splits only; they do not claim homology/specimen independence.

Use a generic six-residue-prefix scan (or shorter fixed minimum for small fixtures), with direct substring confirmation; no approximate matching. Protein sequence checksum and accession resolution remain inspectable. Keep complete reference records in the local source package. Only receipts/code are committed. Source adapter schema advances for new relations; immutable prior source generation remains verifiable with its frozen adapter snapshot.

## Verification and experiment

Meaningful fixtures: overlapping/nested spans, terminal flanks, duplicate sequences under multiple accessions, aliases resolving to multiple proteins, missing and isoform-sensitive IDs, resolved-but-sequence-absent candidates, coordinate shifts/mismatches, leading-versus-listed claims, no matches, unsupported reference formats and invalid/duplicate identities. Check quantity-table hashes before/after references.

Register real shared generation before executing. Compare every actual match/coordinate and source resolution against previously independently verified BioPython+Aho-Corasick audit outputs, plus independently check emitted full protein sequences and aliases against original sources. Recover the same37,721occurrences across bothreferences and64,538source accession claims. Counts alone are insufficient: compare exact row keys/values. Preserve runtime, code/inputs/hashes, all statuses and qualification limitations; no predictive split or fit. Run normal format/lint/full tests and document next recipe step. Publish to existing draft758 and Presto131; keep the full goal and source/query issues open until their acceptance criteria actually hold.

- [x] Merge current upstream safely and freeze reference contract.
- [x] Implement reference relations and source adapter integration.
- [x] Test edge cases and verify real source/reference generation.
- [ ] Publish verified inputs and advance biological recipe integration.
