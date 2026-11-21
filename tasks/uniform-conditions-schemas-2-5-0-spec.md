# Specification: Uniform Condition Columns & Canonical Assay/Schema Representation (`v2.5.0`)

## Goal

1. Ensure curated experimental conditions live in the **right structured columns** — separating vehicle solvents from active drugs, baseline/host engineering from intrinsic genetic background, structured dose/duration from free-text prose, normalizing casing duplicates, fixing misplaced non-condition text and missing structured fields in `pmid_overrides.yaml`, and aligning `condition_category` / `is_control_arm` with structured columns.
2. Eliminate column-name drift across assay families and training/export APIs by introducing a canonical assay schema contract (`hitlist/schema_conventions.py`) and harmonizing `model_training.py`, `export.py`, `apc_contrasts.py`, and `drug_dm_qualification.py`.

## Design

### A. Structured Condition Columns (`hitlist/conditions.py`, `hitlist/data/condition_vocabulary.yaml`, `hitlist/condition_categories.py`, `hitlist/qc.py`)
1. Add 5 first-class columns to `CONDITION_FIELDS`:
   - `condition_vehicle` (`MULTI_VALUE_CONDITION_COLUMNS`, treatment role): solvent/carrier vehicles (`DMSO`, `PBS`, `ethanol`, `water`, `saline`), strictly distinct from active pharmacological drugs in `condition_drugs`.
   - `condition_baseline_engineering` (`MULTI_VALUE_CONDITION_COLUMNS`, `ENGINEERING_CONDITION_COLUMNS`): baseline/parental genetic engineering of the host system (`TAPBPR_KO`, `CD74_transduced`, `CD80_transduced`, `HLA-DP_transduced`, `MAPTAC_tagged_HLA_transfection`, `HLA-DPA1_transduced`, `HLA-DPB1_transduced`, `ACE2_transduced`, `TMPRSS2_transduced`), strictly distinct from intrinsic biological background in `condition_background`.
   - `condition_dose`, `condition_dose_unit`, `condition_duration_hours` (single-value exposure parameters, treatment role): numeric-string dose + canonical unit (must appear together) and numeric-string duration in hours.
2. Validation invariants in `conditions.py`:
   - Reject pure solvent vehicles (`DMSO`, `PBS`) in `condition_drugs` (directing curation to `condition_vehicle`).
   - Reject `condition_control: untreated` coexisting with an active non-`none` intervention column.
   - Require `condition_dose` and `condition_dose_unit` as a pair; validate positive finite numbers for `condition_dose` and `condition_duration_hours`.
   - Add case-canonical aliases in `condition_vocabulary.yaml` (`ERAP1_Hap1 -> ERAP1_hap1`, `ERAP1_Hap2 -> ERAP1_hap2`, `ERAP1_Hap8 -> ERAP1_hap8`, `ERAP1_Hap10 -> ERAP1_hap10`, unit casing aliases `ug/ml -> ug/mL`, `ng/ml -> ng/mL`, `µM -> uM`, and `condition_vehicle` aliases).
   - Enforce case-insensitive uniqueness on `condition_background` tokens across `pmid_overrides.yaml` in QC/tests.
3. Align `condition_categories.py` & `export.py`:
   - Add optional `condition_columns` parameter to `categorize_condition(perturbation, *, condition_columns=None)` (100% backward-compatible when omitted).
   - Vehicle/untreated control arms with no active intervention classify consistently as `unperturbed` (`is_control_arm="true"`), eliminating the split between `"unperturbed — DMSO"` and `"DMSO vehicle control"`.
   - Structured fallback classifies arms with curated `condition_drugs`, `condition_cytokines`, `condition_stimulation`, `condition_antigen_exposure`, `condition_infection`, `condition_transduction`, or `condition_transfection` instead of dumping them into `other_perturbation`.
   - Expand `_NON_APM_PATTERNS` drug regex with curated compounds (`azacitidine`, `5-azacytidine`, `binimetinib`, `entinostat`, `gemcitabine`, `BET degrader`, `dTAG-13`).

### B. Curated Data Fixes in `hitlist/data/pmid_overrides.yaml`
- Migrate `DMSO`/`PBS` from `condition_drugs` to `condition_vehicle` + `condition_drugs: none` on PMIDs `35561310`, `34497125`, `23328677`, `30779550`, and backfill `condition_vehicle: DMSO` + `condition_drugs: none` on PMIDs `32488085`, `33981976`.
- Populate structured `condition_dose`, `condition_dose_unit`, `condition_duration_hours` on clear single-agent arms (PMIDs `32488085`, `34497125`, `23328677`, `35561310` DMSO).
- Move engineered host constructs out of `condition_background` into `condition_baseline_engineering` on PMIDs `30484775` (also adding `condition_knockout_genes: TAPBPR`), `33025005`, `31495665`, `35251023` (removing `untreated` + `condition_transduction` contradiction), and `34171305` (adding missing `ACE2_transduced;TMPRSS2_transduced`).
- Canonicalize `ERAP1_Hap1`/`ERAP1_Hap2`/`ERAP1_Hap10` -> lowercase `hap` on PMID `28063628`.
- Fix PMID `35051231` (`thp_1_macrophage_uv_mock_uninfected` missing `condition_stimulation: PMA`; mock arms missing `condition_infection: none` + `condition_control_for`) and PMID `39438697` mock arms (`condition_infection: none` + `condition_control_for`).
- Backfill missing `condition_mhc_context` on PMIDs `37301199`, `28228285`, `31495665` (Class-I), `23739916`, `28871256`, `29475511`, `18612635`, `28188227`, `26740625`.
- Clean non-condition text in `condition` on PMIDs `33123756`, `27397612`, `42199926`.

### C. Canonical Assay Schema Unification (`hitlist/schema_conventions.py`, `model_training.py`, `export.py`, `apc_contrasts.py`, `drug_dm_qualification.py`)
1. New module `hitlist/schema_conventions.py` (exported in `hitlist/__init__.py`):
   - Defines canonical column names per assay family (`presentation_ms`, `binding_affinity`, `stability`, `tcell`, `perturbation_contrast`, `bulk_proteomics`, `line_expression`) and an explicit `COLUMN_SYNONYM_MAP`.
   - Provides `canonical_assay_columns(assay_type)` and `normalize_assay_frame(df, assay_type=None, *, keep_aliases=False)`.
2. `hitlist/model_training.py`:
   - Propagate all `CONDITION_COLUMNS` into `prepare_ms_training_data().accepted`.
   - Include canonical `mhc_restriction` on accepted affinity/stability/MS tables (alongside `allele` where MHCflurry/Presto expect it) and align `excluded` tables with `mhc_restriction`, `mhc_class`, `measurement_value`, `measurement_unit`, and `measurement_inequality` (plus `raw_measurement_unit` alongside `raw_measurement_units`).
3. `hitlist/export.py`:
   - Include `"condition"` in `generate_observations_table` `meta_cols` and `_CATEGORICAL_COLUMNS` so `condition` is available consistently across `ms_samples`, `observations`, and `expand_condition_memberships`.
4. `hitlist/apc_contrasts.py` & `hitlist/drug_dm_qualification.py`:
   - Remove `- {"DMSO"}` hack in `apc_contrasts.py`; expose canonical `condition_control_for` (keeping `is_comparator_for` alias) and `condition_vehicle`.
   - Rename `background` -> `baseline_treatment` in `load_dlbcl_supplement_rows()` and `dlbcl_35561310_supplements.json` and expose canonical `comparator_condition_ids` across DLBCL & HLA-DM contrasts; read `condition_baseline_engineering` in `prepare_hla_dm_contrast_data()` (with fallback to `condition_background`).
