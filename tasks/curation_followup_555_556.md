# Source and attribution follow-up — 24 September 2026

Addresses [#555](https://github.com/pirl-unc/hitlist/issues/555) and
[#556](https://github.com/pirl-unc/hitlist/issues/556), following release 1.62.57
(base b095964). Proposed release: 1.62.58. Six YAML study records change;
the inventory remains 215 studies and grows from 789 to 794 sample records.

## Source decisions

| PMID | Source-backed correction | Deposit/export result |
|---|---|---|
| 32502341, Pfammatter | Cultured healthy-donor EBV B-LCL and patient-derived B-ALL 10H080 expanded in mice replace primary ALL/healthy B-cell claims. TMT is analytical labeling, not a drug. Typing remains unspecified. | All 10 deposited observations explicitly describe B-LCL and resolve to its TMT preparation. No deposited B-ALL observations. |
| 36215666, Sarango | HeLa-CIITA Mock, control siRNA and TAX1BP1-siRNA are distinct MS arms in both classes. Two aliquots/biological samples per arm, not five technical LC-MS runs. Withdraw unsupported exact class-I typing; retain DRB1*01:02. | 3,632 Mock, 1,156 siCTRL and 2,233 siT6BP observations resolve; 7,882 shared observations retain blank arm identity. All 21 deposited statement variants are mapped explicitly. |
| 28063628, Martin-Esteban | Four natural lymphoblastoid backgrounds: 6370 (ERAP1 Hap2/ERAP2+), 15510 (Hap1/ERAP2−), 10151 (Hap2/ERAP2−), P50 (Hap10/ERAP2+). Three independent peptide preparations each. No experimental knockout or knockdown. | All 3,454 observations name multiple lines and remain arm-ambiguous. Remove erroneous study APM intervention and cancer classification. Selected B*27:05 is retained, not presented as a full genotype. |
| 26154972, Caron | Primary healthy PBMCs receive their own healthy override. Methods name JY, Jurkat and C1R transfectants, not B721.221. | Correct override provenance on 4,617 PBMC observations; revise the generic label on 9,328 cell-line observations. Existing build-time PBMC classification was already correct. |
| 27846572, Liepe | Primary fibroblasts receive a healthy override and cultured-material annotation; GR-LCL receives its own EBV override. | Attribution identity unchanged. Metadata changes on 15,347 observations, including conservative blank override where candidate sources disagree. IEDB's fibroblast “Direct Ex Vivo” field is retained verbatim. |
| 28228285, Abelin | Split primary fibroblasts from the validation cell-line group; give fibroblasts and PBMCs healthy overrides. These are previously published comparison datasets, not new specimens. | Roster count changes from 18 to 19 on all 27,102 observations. No observation switches condition ID. A pre-existing 2,972-row attribution conflict is separately documented in #558. |

Primary source locations:

- Pfammatter: [original supplement](https://doi.org/10.1021/acs.analchem.0c01545.s001),
  Figures S1–S5, and [PRIDE PXD017918](https://www.ebi.ac.uk/pride/archive/projects/PXD017918).
  The publisher's main full text was not available. The authors' matching
  [US20230158132 methods](https://patents.justia.com/patent/20230158132), Examples
  1 and 3, identify 10H080 expansion in NSG mice and the same TMT experiment.
  The patent's additional clinical cohort is not imported into this paper's roster.
- Sarango: [PMC9724678](https://pmc.ncbi.nlm.nih.gov/articles/PMC9724678/),
  Figures 3/EV2, Cell transfections and Immunopeptidome methods. Other autophagy
  receptor interventions and IFNG readouts in functional assays are not MS arms.
- Martin-Esteban: the author's [deposited thesis](https://repositorio.uam.es/bitstreams/a074744b-58e1-4795-8d14-761911c1d75f/download),
  chapter A4 methods, PDF page 104; preceding A3 methods describe the patient-derived
  6370/10151/15510 lines. The source supports noncancer lymphoblastoid lines;
  no unsupported EBV subtype or complete HLA genotype is asserted.
- Caron: [PMC4507788](https://pmc.ncbi.nlm.nih.gov/articles/PMC4507788/),
  “Blood samples, cell lines and synthetic peptides.” Healthy PBMCs were isolated
  by density gradient and frozen; the cell lines were cultured.
- Liepe: [institutional accepted manuscript](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/77732670/Liepe_et_al_main_text_Science_final_3.pdf),
  with fibroblast data reused from reference 6,
  [Bassani-Sternberg, PMC4349985](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349985/),
  Cell culture and Results.
- Abelin: [PMC5405381](https://pmc.ncbi.nlm.nih.gov/articles/PMC5405381/),
  Figure 3C–G and accompanying validation comparison; fibroblasts from
  Bassani-Sternberg and PBMCs from Caron.

## Resolver and complete-corpus impact

Discriminator variance now uses the whole study/class context, including rows
already matched by allele. Exact distinguishing label tokens that name multiple
samples veto a token-score winner. Explicit mapped statements that cannot identify
one candidate preserve ambiguity instead of falling back to narrative guessing.
A genotype-rejected text guess retains curated consensus rather than erasing it.
The unfiltered variance frame retains only distinct discriminator patterns (#562):
at most 6,603 across the curated corpus, rather than millions of repeated peptide
rows. Comparing all 14,120 exported patterns before/after this compaction produces
identical results; the filtered context path is unchanged.

The audit collapsed every attribution input combination into 14,120 patterns
weighted by their observation counts: 4,265,105 observations from the 191 curated
PMIDs represented in this snapshot (all 215 curated PMIDs were included in the scan). The remaining 175,323 MS observations have no curated candidates. It compared
all common exported fields before and after, not just nonblank labels. This is an
attribution audit over one fixed snapshot, not a revalidation of every source paper.

Beyond the six source studies, only these studies change:

- Bourne 35561310: recover SU-DHL-6 identity on all 1,537 previously study-ambiguous
  class-II observations. Of these, 618 identify one treatment and 919 retain
  ambiguous treatment within SU-DHL-6.
- Ritz 27862975: 3,859 observations explicitly naming both serum and plasma lose
  an unsupported plasma-only assignment. Single-material assignments are preserved.
- Gloger 27600516: 156 observations explicitly listing multiple melanoma lines
  lose an unsupported Mel-624 assignment. This is the same shared-label scoring bug.

Raw restriction and evidence columns are unchanged. Separate before/after calls to
`classify_ms_row` over all six studies change only 3,454 Martin-Esteban cancer flags
and 10 Pfammatter primary-healthy-tissue flags. Cached build-time flags require the
normal curation-fingerprint-triggered rebuild; the user's indexes were not rewritten.

## Follow-up boundaries

- [#558](https://github.com/pirl-unc/hitlist/issues/558): 1,901 Abelin observations
  report A*02:04 and 1,071 report B*44:02, explicitly on B721.221 transfectants;
  the roster contains A*02:03/B*44:03. The existing matcher wrongly sends these to
  generic validation cells. Verify source typing before changing allele claims.
- [#559](https://github.com/pirl-unc/hitlist/issues/559): 34 Sherman chicken rows
  retain ambiguous attribution because BF2*2101 and BF2*021:01 normalize differently.
  Preserve evidence and resolve nomenclature from primary sources before adding aliases.
- [#520](https://github.com/pirl-unc/hitlist/issues/520) remains the foundational
  genotype-versus-selected-restriction contract; this PR does not redefine it.

## Verification

Baseline regressions reproduce both #556 defects; all eight new source checks
also fail against the previous YAML. The full-suite inventory checks exposed
[#561](https://github.com/pirl-unc/hitlist/issues/561): they incorrectly split or
discarded a class-only designation beside a typed allele. The production build QC splitter had the same defect, caught by the packaged-data
build smoke test. Both checks now preserve the explicit class component, keep
rejecting malformed tokens, and use no exception list or invented genotype; the runtime
parser already supports it and the regression verifies its sole join allele is DR.
 New source checks exercise
single/shared HeLa arms in both MHC classes, all eight ERAP combinations, B-LCL versus
xenograft provenance, and four primary-material overrides. Query-context regressions
cover full, peptide-filtered and projected exports. The exhaustive audit checks
all 365 distinct source patterns from the nine affected studies individually:
every filtered query and projected result agrees with the full-context export. Format and lint pass; full-suite,
query-audit and final-head CI results are recorded in `tasks/todo.md` and the PR.
