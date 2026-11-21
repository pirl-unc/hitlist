# Spec: `hitlist==2.8.0` — Structured Perturbed MS Contrasts, Orthogonal Cell Lineage Ontology & Biophysical Stability/Kinetic Covariates (#945, #946)

## Goals

1. **Orthogonal APC vs Tumor / Healthy Tissue Lineage Ontology (`hitlist/data/cell_lineage_ontology.yaml` + `hitlist/cell_lineage.py`, #945):**
   - Separate four orthogonal biological axes: `cell_lineage`, `malignancy_status`, `tumor_type`, and `activation_state`.
   - Distinguish APC subtypes (`moDC`, `cDC1`, `cDC2`, `pDC`, `macrophage`, `monocyte`, `b_cell_primary`, `ebv_lcl`) from tumor lineages (`melanoma`, `nsclc`, `crc`, `breast_carcinoma`, `rcc`, `ovarian`, `gbm`, `aml`, `cll`, `dlbcl`, `mcl`, `mm`, `pdac`, `cervical_carcinoma`) and `engineered_monoallelic_host` lines (`721.221`, `K562`, `C1R`, `T2`).
   - Provide `classify_cell_lineage()` and vectorized `attach_orthogonal_cell_lineage(df)`.

2. **Structured Perturbation Contrast & APM Expression Provenance (`hitlist/perturbation_schema.py`, #945):**
   - Normalize curated `pmid_overrides.yaml` arms and observation frames into explicit `(arm_id, control_arm_id, intervention_type, target_gene, variant_or_allele, dose, dose_unit, exposure_duration_hours, baseline_engineering, vehicle)` records using canonical HGNC ortholog groups via `hitlist.apm_orthologs`.
   - Standardize expression provenance tiers (`study_matched_measured`, `reference_cell_line_ccle_depmap`, `reference_cell_line_encode`, `reference_tissue_gtex_hpa`, `reference_primary_immune_dice_monaco`, `parent_line_proxy`, `unknown`) via `tag_expression_provenance()`.

3. **Biophysical Stability & Kinetic Covariates + Unit/Censoring Audit (`hitlist/stability_context.py` + `hitlist/model_training.py`, #946):**
   - Extract structured covariates (`assay_protocol`, `assay_temperature_c`, `assay_ph`, `catalyst_context`) distinguishing `spa_37c`, `dsf_tm`, `cd_tm`, `bfa_cell_surface_decay`, `fluorescent_peptide_exchange`, `spr_kinetics`, `bli_kinetics`, `t2_rmas_stabilization`, and `none` / `hla_dm` / `hla_dm_do` / `tapbpr` catalysis.
   - Standardize kinetic rate units (`off_rate` -> `s^-1`, `on_rate` -> `M^-1 s^-1`) alongside `half_life` (`min`) and `thermal_stability` (`celsius`) in `qualify_binding_measurements()` and `prepare_stability_training_data()` (supporting `KINETIC_STABILITY_ENDPOINTS` while keeping default `STABILITY_ENDPOINTS` backward-compatible).
   - Provide `qualify_stability_and_kinetics()` and `audit_stability_normalization()` for provenance-rigorous stability audits across human and non-human alleles.
