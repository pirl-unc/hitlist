# Curation sanity pass — 24 September 2026

Issue [#554](https://github.com/pirl-unc/hitlist/issues/554); proposed release 1.62.57.

## Scope and limits

Screened all 215 curated studies / 775 original sample records for knockout,
knockdown and cytokine text without corresponding flat columns, suspicious
control roles, MHC parsing/ploidy, and cell-line versus primary/clinical source
overrides. Heuristic matches were investigated, not automatically rewritten:
H-2Kd is not a knockdown; non-targeting siRNA is not a gene knockdown; a cytokine
used only in a functional T-cell assay is not an MS treatment.

Re-read the primary sources for the four corrections below and compared all
34,547 deposited observations for those studies against main e61806f. This is
a broad consistency screen plus targeted source review, not a claim that all
215 papers, every genotype, or every clinical sample has been source-verified.
The packaged inventory becomes 789 samples; study count stays 215.

## Source-supported corrections

| Study | Correction | Measured exported impact |
|---|---|---|
| Chen 2020, PMID 32161166 | Both HeLa arms carry endogenous HLA-A/B/C KO followed by B*51:01 transduction. Control shRNA is non-targeting; ERAP1 is knocked down, not knocked out. Two independent MS experiments per arm. | 2,650 control-only and 2,544 ERAP1-KD-only observations resolve using exact deposited statements. All 6,315 shared observations retain blank arm IDs. Shared HLA knockout metadata is retained across all 11,509 rows. |
| Venema 2021, PMID 33717175 | Replace the natural patient-genotype comparison with WT and CRISPR ERAP2-KO arms of one SAG-transduced EBV-LCL. Restore the reported six-allele genotype, cultured material and two biological replicates. | Remove the unsupported single unperturbed-patient assignment from all 7,850 observations. Their deposited metadata never identifies WT versus KO, so no row acquires an invented knockout assignment. |
| Bourne 2022, PMID 35561310 | Separate DB, SU-DHL-4 and SU-DHL-6; each has DMSO, IFNG, IFNG+DAC, IFNG+TAZ and IFNG+DAC+TAZ arms. Record the paper's individual class-I/DR genotypes and doses; DMSO is a vehicle. | 2,584 observations acquire exact arm IDs, including 2,561 IFNG observations; 3,300 retain line identity with ambiguous treatment. Another 1,537 class-II rows remain study-ambiguous because of matcher issue #556, including 618 with single-arm statements. |
| Ritz 2017, PMID 27862975 | THP-1 comparison sample is a cancer cell line and HEK293 is a noncancer transformed cell line; neither inherits healthy-biofluid provenance. | Two sample overrides change. Exported attribution for all 7,767 deposited serum/plasma observations is unchanged. The pre-existing shared-material attribution error is separately filed as #556. |

Sources and precise locations:

- [Chen, PMC7196583](https://pmc.ncbi.nlm.nih.gov/articles/PMC7196583/): CRISPR
  and lentiviral methods, Results, Figure 3. IEDB calls the HeLa silencing siRNA;
  the paper explicitly specifies lentiviral shRNA. Raw evidence text is retained.
- [Venema, PMC7950316](https://pmc.ncbi.nlm.nih.gov/articles/PMC7950316/): Cell
  Lines, ERAP2 KO Generation, Cell Culture and HLA-Peptide Immunopurification,
  Figure 1, first Results section. Genotype is A*03:01/A*29:02/B*40:01/B*44:03/
  C*03:04/C*16:01; WT and KO pellets were combined after differential labeling.
- [Bourne, PMC9327544](https://pmc.ncbi.nlm.nih.gov/articles/PMC9327544/): Cell
  culture and sources, Drug treatments, Figure 5. Use Figure 5's 125 nM DAC,
  1 µM TAZ and 100 ng/mL IFNG; treatment is renewed after 48 h for another 48 h.
  Do not infer drug-only MS arms from the broader flow-cytometry experiment.
- [Ritz, PMC5557337](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557337/): Figure 2
  and comparison text distinguish cultured-cell HLA from donor serum/plasma sHLA.

## MHC checks

The built evidence snapshot contains 4,440,428 MS and 892,827 binding rows.
`mhc_token_audit()` checked the available restriction, host-typing and serotype
fields plus curated sample typing. No new unrecognized designation or parser
gap was found. The sole finding is the already-reviewed source defect HLA-B23
in 1,693 host-typing rows of PMID 29557506; its intended allele remains unknown.
No curated sample exceeds two alleles at one locus, before or after this pass.
These checks establish syntax/ploidy consistency, not biological correctness.

The new Venema and Bourne typings come from explicit source statements. No
study-wide union is substituted for one cell's genotype, and reported allele
strings and peptide evidence are not rewritten. The cellular-genotype versus
selected-ligand-restriction contract remains tracked in
[#520](https://github.com/pirl-unc/hitlist/issues/520).

## Remaining work found by this pass

- [#555](https://github.com/pirl-unc/hitlist/issues/555): verify the full sample
  roster behind PMID 32502341's primary-ALL/healthy claims versus its deposited
  EBV-LCL evidence; add the missing HeLa-CIITA Mock arm in PMID 36215666 and verify
  its class-I typing; distinguish natural ERAP backgrounds from KO in PMID
  28063628. It also records primary-cell/PBMC override leads in 26154972,
  27846572 and 28228285.
- [#556](https://github.com/pirl-unc/hitlist/issues/556): fix the matcher that
  loses a constant cell identity after other systems leave the unmatched pool;
  protect shared serum/plasma evidence from being assigned to plasma. The latter
  affects 3,859 Ritz rows. These require attribution changes, not invented data.

## Verification and release review

All seven new regression cases fail with main's curation and pass with the
corrected data. The focused curation/condition/inventory/QC suite passes 113
tests. Format and lint pass. Runtime implementation and raw evidence are
unchanged; the data diff is confined to the four named studies.

Full test/CI and publication results are recorded in `tasks/todo.md`. Source
classification stored in an existing observation index requires a rebuild;
the curation fingerprint already triggers that on the next observation build.
The before/after counts above describe exports over the same deposited rows,
not a claim that the user's cached indexes have been rebuilt in place.
