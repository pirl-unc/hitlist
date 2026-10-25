# Native intracellular peptidomics: source-qualified evidence for #837

Base: main 8008fd713bfdb8f340bd635cc830803662c9a186, isolated branch
`codex/native-psmd2-source-20261025`. Presto source families 0344 and 0424 supply
verified original bytes and independent replay; no trained model or concentration
calibration is implied. Preserve existing checkouts and shared Python.

## Smallest useful public contract

Add a `native_peptidomics` evidence family with source-qualified `native_*`
relations, using the existing EvidenceStore/catalog, Arrow sinks, material graph,
source snapshot helper and original OOXML reader. Dataset-specific interpretation
belongs in a pinned source contract, not in downstream Presto code. Do not fold
native extraction into canonical MAPP capture or MHC-elution relations. Keep the
interfaces capable of further native sources without claiming TMAB migration is
already implemented.

One public writer takes caller-provided original assets and a new destination;
one verifier checks pinned hashes, contract, schemas, counts and deterministic
replay. No implicit network or environment installation. Source snapshots prefer
independent filesystem clones. Package generation is atomic and refuses overwrite;
failed staging is retained. Add a new envelope/catalog version without reinterpreting
older packages. Patch release after the existing 2.0.36; recheck main before bump.

Core grains and required relations:

- Assets: original URL/member/range provenance, SHA256, size, role and rights status.
  Include deposited FASTA, workbook, original peptides/evidence/modification tables,
  summary/parameters, design/mqpar, project/inventory/article and the retained ZIP
  ranges/directory that establish member provenance. Do not claim full ZIP SHA1.
- Source records/cells: every original text row and XML/workbook cell, coordinates,
  raw and resolved representation, formula/style attributes and source view.
- Material contexts/links: existing shared schemas. Project, control/KD arms and
  six acquisitions are known; specimen/aliquot identities and biological pairing
  are unresolved. Never manufacture replicate independence from numbered labels.
- Acquisitions: source dataset, original run, experiment label, condition/context
  IDs and raw labels. Reconcile the summary, evidence, design and mqpar mappings.
- Peptides: source-qualified identity, sequence, native ID, PEP, reverse/contaminant
  flags and original annotations. Publisher and original text are views of the
  same entity, not additional training observations.
- Forms/features: original modification-specific IDs and modified sequences,
  charge, source evidence type, acquisition and all original ID relationships.
  Preserve MSMS, MULTI-MSMS, MULTI-MATCH and MULTI-SECPEP separately.
- Quantities: one source entity, acquisition and quantity kind per row; original
  literal and numeric value/status, scale/unit, normalization/form scope, source
  record and source field. Raw sequence intensity and LFQ intensity remain distinct;
  feature and modification-specific intensities have their own entity grains.
  Zero has unknown observation cause; missing is distinct and neither is a
  negative concentration label. No automatic sum of rounded exports replaces a
  supplied source quantity.
- References/origins: all 73,113 original records with full sequences, header,
  accession, checksum and explicit construct disposition. Enumerate all literal
  and separate I/L-equivalent spans across the full deposited reference, including
  repeated locations and candidates absent from the leading/declared parent list.
  P22222/NS3CCNB1_HUMAN is an engineered reporter despite its OX=9606 tag. Preserve
  such origins as exclusions/ambiguity; do not make them optimization targets.
  Retain all source-declared parent alternatives and unresolved contaminant IDs.
- Assertions/reconciliations: source-scoped species, HEK293T cell context, PSMD2
  knockdown, native extraction, preparation uncertainty, enzyme/search configuration,
  matching settings, protein versus peptide FDR and source discrepancies. No numeric
  residual activity, complete KO, professional APC status or calibrated precursor
  abundance is supplied. Keep source gene-date defects and directly corresponding
  original text annotations; do not reverse-engineer gene names from dates.

Public field descriptions and relationships must explain entity/quantity grain and
source scope. A query must be able to recover both the original raw intensity and
why it cannot be called an unmodified-peptide molar concentration.

## Source evidence that implementation must reproduce

0424 verifies 10,647 original peptide rows, 40,679 evidence rows and 10,935 form
rows. All 74,529 publisher raw-intensity cells match the original text exactly.
The 13 gene discrepancies are date conversions, with actual original text names
available. Mass/PEP/score representation differences are retained separately.
All ID memberships and six run mappings reconcile. Sums of rounded feature/form
intensities are not exactly equal to all source sequence intensities; maximum
relative residuals are 8.052e-5 and 7.550e-5. Preserve source values and residuals.
Raw and LFQ differ in 30,170 cells. A normalization algorithm is not proved by a
column name or those residuals.

Full-reference mapping has 38,641 peptide spans, 403 I/L-only alternatives, 159
peptides without a deposited-reference match, 53 with a reporter origin and 582
with additional unreported accession origins. Twenty-four declared accession
tokens absent from the FASTA are contaminant-prefixed. The article's reference
count differs from the deposited 73,113; preserve both claims, use the deposited
reference for source replay. Search-specific length 8–25 is distinct from the
general minimum-length field 7. Matching between runs is enabled.

Build dependency components before any supervised selection using all reference
sequence alternatives and all declared parents. Do not fit or materialize a
training recipe in the source adapter. The descriptive complete-acquisition
stratum suggests 82 parents/81 dependency components and 228 reference contrasts;
that is feasibility evidence, not validation or the final source contract.

## Verification and release

Add small synthetic software fixtures for multiple origins/repeated spans, I/L,
misleading reporter taxonomy, source-only contaminant accessions, source zeros,
matching-only evidence, raw/LFQ separation, modification aggregation, unpaired
numbered labels, date-corrupted annotations, changed source bytes and tampered
typed projections. Cover public query fields and relationship closure, not just
internal parsing. Source-data replay must independently reconcile every quantity,
origin, disposition and original row/cell, including retained discrepancies.

Run `./format.sh`, `./lint.sh`, and the full `./test.sh` with at least 2 GiB free
headroom; do not relax the provenance reserve. Use source paths/PYTHONPATH in the
isolated checkout without replacing the shared editable installation. Open a
versioned PR linked to #837, inspect CI, merge only verified code, then perform a
clean tested release and verify the published wheel/sdist. Only afterward consume
the canonical package for a registered biological fit. Source admission and
later recipe release are explicit separate milestones; #837 remains open until
its acceptance items are genuinely covered.
