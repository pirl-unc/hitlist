# Offline canine ligand curation — #660

## Deliverable and source scope

After the historical-reference release, create a separate 1.69.0 feature branch.
Extend the existing supplementary scanner with explicit reviewed `entries`, a
local `directory`, and `allow_download=False`. Supply a reproducible offline
curator for PMID 42199926's six original XLSX supplements. Preserve the current
packaged-corpus behavior and contributor model; no new corpus is downloaded or
substituted automatically. The independently inspected source reports 16,359
8–30-aa observations, including 3,580 human-host DLA observations and 12,779
canine tumor observations (12,181 distinct strings, four dogs).

Primary sources: PMID 42199926 / DOI 10.1016/j.isci.2026.115975, Results and STAR
Methods, Data S1–S6, PXD074485. Original workbooks mmc2.xlsx–mmc7.xlsx have been
independently checked against their article MD5s. Their source URLs, exact sizes,
SHA256s, MD5s, selected worksheet names, row counts, sample contexts and source
sections belong in YAML, not biological constants embedded in Python.

## Scanner contract

- Both packaged and explicit entries undergo the existing excluded-from-MS
  contradiction check. Explicit input must not bypass curation validation.
- Explicit entries are caller-reviewed MS data. Require a local file for each;
  never resolve a custom file to an unrelated global asset with the same name.
  Offline mode never accesses the network, including on absent files. Preserve
  the packaged scanner's existing on-demand behavior when called without inputs.
- Validate an optional supplied CSV SHA256 before parsing. Caller-supplied local
  inputs fail on missing peptide columns instead of silently losing a source.
- Preserve all original CSV columns and the complete manifest in contributor
  records. Carry a reviewed `attributed_sample_label` from the entry defaults
  into observations so existing sample attribution and lineage attachment can
  reach the exact arm. Keep synthesized source identity and duplicate lineage.
- No row automatically receives experimental allele support because it has a
  class label or belongs to a genotyped tumor. Default source/host/species and
  existing MHC identity/restriction resolution remain separate.

## Workbook curator

Provide `curate_canine_ligands(directory, output_dir)` with no network path.
Read a packaged reviewed YAML profile; verify all six local assets before
opening workbooks. Use openpyxl read-only iteration, import it only for the
curator, and document its installation. Read only the three DLA sheets in Data
S1 and the `All Peptides` sheet of each tumor workbook. Exclude human-TAA
comparison sheets entirely. Validate the selected headers, integer reported
length, 8–30 canonical amino-acid letters, identification method and expected
row count. Fail visibly on a changed source or malformed selected row.

Write deterministic per-arm CSVs, scanner manifest and checksum-bearing receipt
into a fresh staged output directory, publishing only after every source passes.
Preserve source URL/hash, worksheet and one-based original row, reported protein
accessions (not verified unique mappings), method, sample, IP antibody, study
FDR threshold/scope, raw deposit and source database description. Keep individual
q-values, spectrum IDs, exact searched FASTA identities and I/L discrimination
unknown. Summary-table strings remain exact reported strings; do not resolve
I/L or infer nested windows. No timestamps or absolute input paths in content
identity. The wheel includes metadata and synthetic tests, not original files.

## Curation and identity

Curate eight `ms_samples` with stable condition IDs and exact labels:
three HCT116 DLA transductants, Lola H58A, Lola BB7.6, 163828A BB7.6,
Lily H58A and Bogey BB7.6. Human HCT116 has a human peptide source/host and
introduced canine DLA; establish that mechanism through the existing independent
species-context registry. Canine tumors have endogenous dog source and host.
Only the three transductants receive the monoallelic restriction-evidence rule.
Keep tumor peptide restrictions class-only and unassigned. RNA typing is sample
context, not observed peptide restriction; preserve reported typing/provenance
without claiming all typed alleles presented every peptide. Locus ambiguity
(e.g. DLA-88L in Bogey) must remain explicit if not structurally resolvable.

Add reviewed donor/specimen links to the existing lineage registry for the four
named dogs. Both Lola IP arms link to the same donor and tumor specimen. The IPs
have separate experiment origins; neither worksheet names nor peptide labels
establish a raw acquisition or an independent dog. Human cell-line arms do not
invent human donor/specimen identities. Source and condition IDs must survive
scanner -> observation/sample join -> lineage attachment with contributor links.

## Verification and release

- Generated tiny XLSX fixture with original-style headers and a deliberately
  tempting human-TAA sheet: only selected observations enter the scanner.
- Missing files, checksum mismatch and malformed selected rows fail without any
  network call or published partial output. Repeated exports are byte-identical.
- Verify species axes, independent transfectant context, 3 monoallelic vs 5
  unassigned arms, exact sample attribution, four dog donor IDs and shared Lola
  donor/specimen with distinct experiments.
- Confirm original workbook/sheet/row/protein accessions and CSV digest survive
  the contributor graph. Duplicate observation rows retain every contributor.
- Re-run the curator and scanner on all six real assets locally, assert all
  16,359 contributors and independently audited counts. This is a scoped source
  validation, not a full corpus rebuild or raw-spectrum reanalysis.
- Format, lint, focused curation/export/provenance tests, unchanged `test.sh`
  guard, full final-head CI, merge, clean-main deploy and PyPI byte verification.

## Next dependent work — #661

Use a new explicit species-scoped evidence-bundle mode with frozen canine taxon,
complete protein reference, all occurrence mappings and a versioned empirical
reproductive-expression policy. Preserve the human API/defaults and old replay.
Accept evidence at its reported gene/promoter/transcript level with units,
missingness and ambiguity bounds; do not derive isoform abundance or import
human CTA membership. Retain noncandidate matches and normal-tissue expression
as counterevidence, especially identical testis/heart protein sequences. Canine
normal-tissue evidence is an explicit input, including missing coverage and
exclusion reasons; human Ligand Atlas is never an automatic replacement.
Define and test the concrete bundle schema after #660's portable observation
and contributor boundary is established. Hitlist exports evidence only; Tsarina
and Vaxrank remain responsible for vaccine assembly.

## Additional primary-source details

The selected headers are `Peptide, Length, Accession, Found By`, except mmc5
which reports `Peptides` plural; record the exact accepted header per source.
STAR Methods describes the DLA-88*501:01 transduction construct as
`DLA-88*501:01 (AA21 P > L)`. Preserve that reported construct description and
its provenance beside the reported worksheet allotype. Do not silently turn a
worksheet allotype name into proof of an unmodified construct sequence, or
invent a coordinate convention absent from the supplied source. Flow cytometry,
Western blots, motif predictions and human-TAA comparison sheets do not enter
the curated MS observation table.

Use the explicit source label `HCT116 HLA-I KO` for the curated transductant
inputs and add only that engineered variant to the monoallelic host registry.
Ordinary HCT116 cells are not an HLA-null host. Include HCT116 and its explicit
KO spelling in the PMID/allele-scoped restriction rule for compatibility with
reviewed downstream CSVs, while the curated importer carries the exact engineered
context. Do not add a whole-paper allele pool: it would populate class-only tumor
rows from unrelated transductant alleles.
