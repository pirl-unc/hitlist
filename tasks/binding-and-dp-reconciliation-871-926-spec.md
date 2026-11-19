# Binding Method Conflict Review (#871) & PMID 35251023 DPB1*19:01 Reconciliation (#926)

Date: 2026-11-19
Target release: `hitlist==2.1.22`
Issues: `#926`, `#871` (unblocks `pirl-unc/presto#131`)

## 1. `#926` — Reconcile extra `HLA-DPA1*02:02/DPB1*19:01` pair in `PMID 35251023`

### Primary-source audit
- **Publication**: Laghmouchi et al. 2022, *Frontiers in Immunology* (`PMID 35251023`, `PMC8888658`, DOI `10.3389/fimmu.2022.831822`).
  - Full-text XML (`PMC8888658.xml`, `113,249` bytes, SHA-256 `c650f06dd56be6d786bb497c81ee0f5ea66ba319a883d59c3d29cd7c36f7381d`) explicitly reports **13** K562 cell lines (median `3,131` peptides, range `2,172–4,449`) with `0` mentions of `19:01`.
  - Supplementary Appendix PDF (`DataSheet_1...pdf`, Figshare `34077722`, DOI `10.3389/fimmu.2022.831822.s001`, `1,010,928` bytes, SHA-256 `4ce9351b45bc32390dacf0e53c578050f8ce697d31adb29dc8b598f89295ff42`): Supplementary Table 1 (page 2) and Supplementary Figures 1–4 list only those **13** constructs; no supplementary peptide table is published.
  - PRIDE dataset `PXD030591` (`34` files, `16` `.msf` search outputs, `17` `.raw` files, plus `PXD030591_community_annotated.sdrf.tsv`, `12,861` bytes, SHA-256 `23895b42403e8cfa7166dc565d98a18047f426f559d43bd7668c0da17850d835`) covers only the **13** published alleles (`DPB10101`–`DPB11701`), with an omitted run slot `L2172688` between `L2172687a` (`01:01`) and `L2172689a` (`03:01`) and `0` `DPB1*19:01` files.
- **Deposited `ci-corpus-v4` inventory** (`observations.parquet`, manifest SHA-256 `2318bc53...`):
  - `45,961` total rows under `http://www.iedb.org/reference/1040629` (`source="iedb"`, `0` `cedar`/supplement contributors) across `14` distinct restrictions.
  - `HLA-DPA1*02:02/DPB1*19:01`: **`1,322` observations** (`1,280` unique peptides), `provenance_id` `observation:record:iedb:row:4521146` .. `observation:record:iedb:row:4568080`, `assay_iri` `http://www.iedb.org/assay/18870849` .. `http://www.iedb.org/assay/18917801`.
  - Not an alias/duplicate of any published pair (maximum unique-peptide overlap is `393/1,280` with `HLA-DPA1*02:02/DPB1*05:01`, `307` with `09:01`, `305` with `17:01`; `0` other PMIDs in `ci-corpus-v4` carry `DPB1*19:01`), consistent with an unverified 14th construct or author-correspondence peptide list curated by IEDB without public paper/PRIDE documentation.

### Implementation
1. Ship pinned reconciliation receipt at `hitlist/data/evidence_revisions/pmid_35251023_dpb1_19_01_reconciliation.json`.
2. Update `arm_resolution_note` on `pmid: 35251023` in `hitlist/data/pmid_overrides.yaml` citing `#926`, exact counts (`1,322` rows / `1,280` unique peptides), `PXD030591` 13-allele inventory, and the receipt, while keeping `ms_samples` strictly at the `13` primary-source constructs (never inferring a 14th arm).
3. Add unit and `ci-corpus-v4` integration regressions in `tests/test_apc_intervention_curation.py`.

## 2. `#871` — Review ELISA-referencing `PMID 12135423` KD panels tagged as competitive fluorescence

### Primary-source & corpus audit
- **Cited protocol**: Sylvester-Hvid et al. 2002, *Tissue Antigens* 59(4):251–258 (`PMID 12135423`, DOI `10.1034/j.1399-0039.2002.590402.x`), *"Establishment of a quantitative ELISA capable of determining peptide - MHC class I interaction."*: peptide-dependent refolding of denatured recombinant MHC-I + β2m measured by quantitative ELISA (not competitive displacement and not fluorescence).
- **Exact `ci-corpus-v4/binding.parquet` (`814,268` rows) scope** (`27,891` rows across `22` IEDB references; live IEDB still retains identical tags as of 2026-11-19):
  1. **20 DTU direct submissions** (`reference/1019353` `1000446`, `1019354` `1000447`, `1019507`–`1019524` `1000450`–`1000467`; **`18,723` rows**, `743` in `corrected-human-cohort`, `16,319` in Presto's admitted concentration TRAIN cohort; examples `1715424`, `1715425`, `1715426`): tagged `purified MHC/competitive/fluorescence` + `dissociation constant KD` (`nM`) with submission-level general comment `"In general, this data has been generated using a quantitative biochemical peptide-HLA interaction assay (see PMID: 12135423)..."` -> `source_method_conflict_submission_scope`.
  2. **1 DTU direct submission** (`reference/1014017` `1000422`; **`9,150` rows**, `8,367` in Presto's TRAIN cohort; example `1607390`): tagged `purified MHC/direct/fluorescence` + `dissociation constant KD (~EC50)` (`nM`) with the same truncated `PMID 12135423` comment -> `source_detector_conflict_submission_scope`.
  3. **1 literature study** (`reference/1021191`, `PMID 21169544`, Tang et al. 2011; **`18` rows**; example `1810181`): tagged `purified MHC/direct/fluorescence` + `dissociation constant KD (~EC50)` (`nM`) citing `Sylvester-Hvid, et al. 2002. Tissue Antigens. 59: 251-258. [PMID: 12135423]` -> `source_detector_conflict_study_scope`.
- **Safeguards**:
  - Original `assay_method`, `response_measured`, `quantitative_measurement`, `measurement_units`, `measurement_inequality`, and `assay_iri` are preserved verbatim (`KD` is never relabeled `IC50`).
  - `20,267` LOCI (`purified MHC/direct/fluorescence` + `LOCI` comments) and other unreviewed `competitive/fluorescence` or `KD` rows remain `not_source_reviewed`.
  - If upstream IEDB later updates `assay_method` or `response_measured` on any of the 22 reviewed references, rows are flagged `upstream_tag_changed_requires_rereview` rather than double-corrected.
  - Any newly appearing uncataloged reference citing `PMID 12135423` with a `fluorescence` method is flagged `uncataloged_citation_requires_review`.

### Implementation
1. Ship pinned review contract `hitlist/data/evidence_revisions/binding_assay_method_reviews.json`.
2. Extend `hitlist/binding_assay_context.py` (keeping frozen schema-10 package verifier untouched) with `binding_method_review_contract()`, `review_binding_assay_methods()`, `summarize_binding_method_reviews()`, and `METHOD_REVIEW_COLUMNS`, re-exported lazily in `hitlist/__init__.py`.
3. Pass `METHOD_REVIEW_COLUMNS` cleanly through `qualify_binding_measurements` and `prepare_affinity_training_data` (`hitlist/model_training.py`) without changing exclusion counts or breaking existing tests.
4. Add representative-row unit tests (including drift detection, LOCI non-conversion, and uncataloged citation flagging) plus `ci-corpus-v4` integration verification in `tests/test_binding_assay_context.py`.
5. Update `docs/model-training.md` and `hitlist/version.py` (`2.1.22`).
