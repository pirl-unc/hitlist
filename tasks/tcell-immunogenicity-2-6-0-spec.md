# Spec: `hitlist==2.6.0` — T-Cell Immunogenicity Modalities, Pin/Parity Reconciliation, SPR Separation & Reference MHC Genotype Registry (#947; foundational for #943 & #944)

## Goals

1. **Fix `hitlist.tcell_assays` live ZIP pin drift & asynchronous IEDB/CEDAR parity failures (#947 Bug 1):**
   - Add `source_verification_mode: Literal["pinned", "reconcile"] = "pinned"` to `tcell_assays`, `tcell_peptides`, `tcell_functional`, `tcell_multimer`, and `tcell_training` (plus `--verification-mode` CLI flag).
   - `"pinned"` preserves exact existing strict byte-pin and row-by-row parity checks.
   - `"reconcile"` computes actual `(sha256, size_bytes)` for both ZIPs without aborting on pin drift, records `pin_status`, `actual_sources`, `n_parity_diffs`, `parity_diff_assay_ids_sample`, `n_cedar_only_assays`, and `cedar_only_assay_ids_sample` in structured manifest provenance (`source_reconciliation`).
   - Update `verify_human_tcell_training_bundle()` to support verifying reconciled bundles against `evidence_manifest["actual_sources"]` while keeping `"pinned"` verification strict.

2. **Separate non-cellular TCR–pMHC biophysical assays out of cellular T-cell families (#947 Bug 2):**
   - Update `hitlist.tcell_assays.assay_family(method, response, *, separate_biophysics=True)` so cell-free recombinant TCR–pMHC assays (SPR, BLI, X-ray crystallography / 3D structure, ITC, MST, `dissociation constant KD`, `association constant KA`, `off rate`, `on rate`) route to `"tcr_pmhc_biophysics"` instead of `"other_tcell_assay"`.
   - Keep `tcell_evidence.py` (frozen `tcell_2026_10_20` v5 package) calling `assay_family(..., separate_biophysics=False)` so historical v5 checksums stay intact.

3. **Foundational Curated Reference MHC Genotype Registry (`hitlist/data/reference_mhc_genotypes.yaml` + `hitlist/mhc_genotypes.py`)** (shared foundation for #947, #943, and #944):
   - Curate verified reference MHC genotypes for canonical human/rodent cell lines (`HAP1`, `HeLa`, `JY`, `Raji`, `A549`, `HEK293T`, `MCF-7`, `K562`, `C1R`, `721.221`, `T2`, `Jurkat`, `GRANTA-519`, `ECN90`, `THP-1`, `A-375`, `624-mel`, `Panc 06.03`, `Panc 10.05`, `31227ABO`, `AWELLS`, `BER`, `EL-4`, `P815`, `A20`, `RMA`, `B16-F10`, `NIT-1`) with explicit `mhc_null_class_i` / `low_endogenous_class_i` / `tap_deficient` safeguards.
   - Curate canonical inbred and HLA-transgenic rodent strains (`C57BL/6`, `BALB/c`, `NOD`, `SJL`, `C3H`, `CBA`, `DBA/2`, `FVB`, `Lewis`, `HLA-A*02:01 Tg`, `HHD` / `HLA-A*02:01 B2m-/- H2-Db-/-`, `HLA-B*07:02 Tg`, `HLA-A*24:02 Tg`, `HLA-DRB1*04:01 Tg`, `HLA-DRB1*01:01 Tg`, `HLA-DRB1*15:01 Tg`, `HLA-DQB1*03:02 Tg`).
   - Provide `resolve_condition_mhc_genotype()` exposing `mhc_allele_set`, `mhc_allele_set_size`, `mhc_allele_cardinality` (`monoallelic` / `multi_allelic_bag` / `class_only_unknown`), `mhc_allele_set_provenance`, `mhc_species`, `host_species`, and `is_chimeric_mhc_host`, while strictly separating predicted restriction (`MHC binding prediction`) from experimental single-allele restriction.

4. **Multi-Modality, Context-Rich, Multi-Species T-Cell Immunogenicity Bundle (`hitlist/tcell_immunogenicity.py`)**:
   - Keep frozen `hitlist-human-tcell-training-v1` backward-compatible.
   - Add `iter_tcell_immunogenicity_assays()`, `write_tcell_immunogenicity_bundle()`, and `verify_tcell_immunogenicity_bundle()` (`hitlist-tcell-immunogenicity-v2`, `schema_version=2`) exposing:
     - Granular `immunogenicity_modality` (`pmhc_multimer`, `th1_cytokine`, `th2_cytokine`, `th17_cytokine`, `treg_regulatory_cytokine`, `other_cytokine`, `cytotoxicity`, `degranulation`, `proliferation`, `activation_aim`, `other_cellular_tcell`) + `cytokine_or_readout` + `assay_method`
     - Normalized `ordinal_intensity` (`Negative`, `Positive-Low`, `Positive-Intermediate`, `Positive-High`, `Positive-Ungraded`), `ordinal_intensity_rank`, `(n_subjects_tested, n_subjects_positive)`, `reported_response_frequency_pct`, and numeric `quantitative_value`
     - Biological context: `in_vivo_process_type`, `disease_context`, `in_vitro_process_type`, `effector_repertoire_type` (`endogenous_polyclonal` vs `tcr_clone_or_transduced`), `effector_tcr_name`, `antigen_delivery_mode` (`exogenous_peptide`, `endogenous_processed`, `whole_protein_or_organism`, `other`), `apc_lineage`, `antigen_category`, `cedar_mutation_raw`, and `wt_peptide`
     - Condition-level `mhc_allele_set` + `mhc_allele_set_size` + `mhc_allele_cardinality` + `mhc_allele_set_provenance` alongside strict experimental `mhc_allele`, plus `mhc_species`, `host_species`, `is_chimeric_mhc_host`, across human and non-human species (`species_filter="all"` by default)
     - Clean separation of cell-free `tcr_pmhc_biophysics` rows (excluded from cellular training rows and optionally written to `biophysics.parquet`).
