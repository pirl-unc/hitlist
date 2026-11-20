# Spec: APC/Drug Perturbation, HLA-DM Arm Recovery & Atlas Prediction Provenance (#901, #902, #853, #855) (v2.2.0)

## 1. `#853` — Abelin 2019 MAPTAC Class-II HLA-DM Arm Recovery (`PMID 31495665`)
- Pin primary workbook `1-s2.0-S1074761319303632-mmc2.xlsx` (`SHA-256 78b551cecd395d8650c9696ebdb34f78066f351b83f27d043e0e57880420adbe`, `998,734` B, sheets `DataS1A` and `DataS1B`).
- Package checksum-pinned `hitlist/data/supplementary/abelin_2019_maptac_class2_dm_arms.csv.gz` (`45,636` condition-resolved rows across 14 Class II MAPTAC columns: 8 paired `dm-`/`dm+` arms totaling `13,317` rows including `2,133` shared peptides + 6 single-arm DRB1 alleles totaling `32,319` rows) with exact `(source_column, source_row)`, `attributed_sample_label`, and complete heterodimer restrictions (`HLA-DPA1*01:03/DPB1*06:01`, `HLA-DQA1*01:02/DQB1*06:04`).
- Package companion `hitlist/data/supplementary/abelin_2019_maptac_class2_source.yaml`.
- Replace the merged `abelin_2019_maptac_class2.csv` entry in `hitlist/data/supplementary.yaml` with `abelin_2019_maptac_class2_dm_arms.csv.gz` so merged historical rows are never double-counted alongside recovered condition-resolved rows.
- Update `hitlist/supplement.py` so per-row `attributed_sample_label` is part of `synth_iri`, preventing shared peptides across `dm-` and `dm+` in one CSV from colliding.
- Wire `condition_control_for` and `condition_background: "MAPTAC_tagged_HLA_transfection"` on `PMID 31495665` in `pmid_overrides.yaml`, explicitly distinguishing no added HLA-DM co-transfection (`endogenous_dm_status="unknown"`) from genetic DM knockout.
- Expose `abelin_dm_source_contract()` and `prepare_hla_dm_contrast_data()` converging Abelin (`31495665`) and Meurer (`33025005`) on the shared material/condition contrast schema without pairing `37301199` DM-only rows as a control.

## 2. `#855` — HLA Ligand Atlas Prediction-Derived Allele-Association Provenance (`PMID 33858848`)
- Add scoped `restriction_evidence_rules` (`Assay Method: cellular MHC/mass spectrometry`, `Response Measured: ligand presentation` -> `evidence: predicted`) to `PMID 33858848` in `pmid_overrides.yaml`, preserving `unknown` for multi-allele donor sets (`536,571` rows) and `experimental` for any separately measured binding assay while marking all `365,042` single-allele named MS rows (`203,992` Class I + `161,050` Class II, including the `24,399` DPA1/DPB1 & DQA1/DQB1 rows) as `predicted`.
- Pin `hitlist/data/evidence_revisions/hla_ligand_atlas_33858848_prediction_provenance.json` (`PMC8054196.xml` Methods: `NetMHCpan-4.1`, `SYFPEITHI`, `NetMHCIIpan-4.0`, `MixMHC2pred`, thresholds, donor typing provenance) and expose `atlas_prediction_provenance_contract()`.

## 3. `#902` — Flucloxacillin Condition Membership & Peptide Adduct Identity (`PMID 33633747`)
- Update `PMID 33633747` in `pmid_overrides.yaml` with `arm_resolution: multi_arm_evidence`, `sample_group: "B721-5701"`, `condition_control_for: "cells_flucloxacillin"`, primary-source dose/duration (`150 ug/mL FLX`, 5 days, duplicate cultures), and exact `elution_condition_ids` reconciling `1,142` FLX-only, `226` untreated-only, and `1,122` shared both-arm rows while retaining `3` `Filler Assay - delete.` and `3` blank sHLA rows unattributed.
- Pin `hitlist/data/evidence_revisions/flucloxacillin_33633747_adduct_contract.json` (`PMC7900192.xml` Table 1 confirmed `+453.06 Da` FLX-haptenated lysine sites `KAAKLKEKY@K4`, `RTKKVGIVGKY@K4`, `TAAQITQRKW@K9`; `Table_2.xlsx` SHA-256 `e462c09b...`; T-cell vs MS separation).
- Expose `flucloxacillin_adduct_contract()` and `prepare_flucloxacillin_evidence_views()` returning both an unmodified-only training subset and a modification-aware evidence view without collapsing molecular forms.

## 4. `#901` — DLBCL (`PMID 35561310`) Supplement Control Recovery & Conditional Comparator Provenance
- Pin `hitlist/data/evidence_revisions/dlbcl_35561310_supplements.json` (`advancesADV2021006069-suppl3.xlsx` SHA-256 `dba6c6af...`, `10,507` Class I rows; `suppl5.xlsx` SHA-256 `3ff0e09f...`, `5,203` Class II rows across all 15 sheets including `DB_untreated` `7` Class I + `20` Class II; literal `lengtg` & `Query #:z` header variants; NetMHCpan 4.0 `predicted` restriction provenance; explicit `no_public_intensity_or_replicate_export` gap; `#930` conflict links).
- Wire same-line `condition_control_for` edges on `PMID 35561310` (`DMSO -> IFNG`, `IFNG -> IFNG+DAC;IFNG+TAZ;IFNG+DAC+TAZ`) in `pmid_overrides.yaml` so added-drug effects compare against same-line `IFNG`, never relabeled as drug-alone.
- Expose `dlbcl_supplement_contract()`, `load_dlbcl_supplement_rows()`, and `reconcile_dlbcl_supplements_with_iedb()`.
