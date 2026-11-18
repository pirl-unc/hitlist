# APC Intervention, Statement Membership & Engineered HLA-DP Curation — #899, #900, #856, #593

## Goal

Make APC genetic/pharmacological intervention studies and engineered MHC-II arms directly
usable for model training (Presto and MHCflurry) through Hitlist's shared curation,
export, and `context_evidence` surfaces without inventing missing controls, collapsing
shared detections into opposing labels, or treating MS nondetections as affinity
negatives.

## Scope & Architectural Design

### 1. Unlock `context_evidence` Curation Evolution (`hitlist/context_evidence.py`)

`write_context_evidence_package` currently rejects any curation revision whose study or
arm count differs from `pmid_overrides_1_78.yaml` (`len(historical_arms) != len(current_arms)`),
and `verify_context_evidence_package` assumes `2 * n_curated_studies_per_revision` and
`2 * n_curated_arms_per_revision` rows in `curated_studies.parquet` and `curated_arms.parquet`.

- Remove the `len(historical_arms) != len(current_arms)` guard in `write_context_evidence_package`.
- Preserve `n_curated_studies_per_revision` and `n_curated_arms_per_revision` (recording the
  immutable `cohort_1_78` counts checked against `PINNED_COHORT_COUNTS`) while adding explicit
  `n_historical_curated_studies`, `n_current_curated_studies`, `n_curated_studies`,
  `n_historical_curated_arms`, `n_current_curated_arms`, and `n_curated_arms` fields to
  `reconciliation`.
- Update `verify_context_evidence_package` to check `inventory.get("n_curated_studies", 2 * inventory["n_curated_studies_per_revision"])`
  and `inventory.get("n_curated_arms", 2 * inventory["n_curated_arms_per_revision"])`, keeping
  existing frozen packages verifiable verbatim while allowing `pmid_overrides.yaml` to split
  or add arms.
- In `_amendments`, fall back to `old.get("condition_reference") or ""` only when `current`
  is empty (removed/split historical arm), leaving existing modified-arm amendments unchanged.

### 2. Statement Membership, Explicit Nondetection & Comparison Projection (`hitlist/curation.py`, `hitlist/export.py`)

- **Negated-clause guard in `_arms_in_elution_conditions`**: Ignore negated fragments
  (`but not ...`, `not from ...`, `excluding ...`) when parsing generic elution enumerations
  so `"eluted from X, but not from untreated cells"` never adds `"control"` to positive arms.
- **Structured `elution_condition_ids` schema**: Extend `elution_condition_ids` in
  `curation.py` and `StatementMap` in `export.py` so each statement value may be either:
  - `[condition_id, ...]` (existing shorthand for `{"detected": [...], "not_detected": []}`), or
  - `{"detected": [condition_id, ...], "not_detected": [condition_id, ...]}` where `detected`
    is non-empty, both lists contain distinct profiled `condition_id`s of the study, and
    `set(detected).isdisjoint(not_detected)`.
- **Exported per-observation membership columns**:
  Keep `generate_observations_table` and `generate_training_table` at one row per source
  observation while adding two dictionary-encoded (`category`) columns:
  - `supported_condition_ids`: `condition_id` when a single arm resolved; or sorted `;`-joined
    compatible `detected` arms when a mapped statement positively places the observation in
    multiple arms (`"and"` / shared detection); `""` when unmapped/disjunctive (`"or"`),
    blank, or statement-vetoed.
  - `excluded_condition_ids`: sorted `;`-joined compatible `not_detected` arms when a mapped
    statement explicitly states nondetection in comparator arm(s); `""` when unstated or
    statement-vetoed.
- **Declared condition-comparison projection (`expand_condition_memberships`)**:
  Provide `expand_condition_memberships(df)` in `hitlist.export` to project an exported
  observations/training frame into long-form `(source_observation_index, condition_id, membership_status)`
  rows (`membership_status` in `{"detected", "explicit_not_detected"}`) joined with curated
  arm condition columns while retaining `provenance_id` / `evidence_row_id` so shared
  detections and explicit nondetections can be consumed without duplicating source-row counts.

### 3. Primary-Source Study Curations (`hitlist/data/pmid_overrides.yaml`, `hitlist/data/species_contexts.yaml`, `tests/data/engineering_drift_baseline.yaml`)

- **PMID 30484775 (`#899`, Ilca et al. 2018, PMC6307860)**:
  Replace mixed `cells_tapbpr_manipulation` with four source-backed `HeLaM-TAPBPR-KO` + `IFNG`
  arms (`helam_tapbpr_ko_ifng_tapbpr_wt`, `helam_tapbpr_ko_ifng_tapbpr_oloop`,
  `helam_tapbpr_ko_ifng_tapbpr_og30l`, `helam_tapbpr_ko_ifng_tapbpr_m29`), separating
  `condition_background: "TAPBPR_KO"` + `condition_cytokines: "IFNG"` from lentiviral rescue
  constructs (`condition_transduction`) and loop variants (`condition_genetic_variants`),
  retaining the two `\ufffd` deposited statement keys verbatim, and updating
  `species_contexts.yaml` (`human_helam_tapbpr_immunopeptidome`).
- **PMID 31222486 (`#900`, Koumantou et al. 2019)**:
  Curate `DG013A` (`1 uM, 6 d`) primary-source metadata, comparator link, and exact statement
  map (`1,198` untreated without false absence under DG013A; `417` DG013A detections
  without comparator exclusions because specificity includes enrichment (#917);
  `1,589` shared in both).
- **PMID 23328677 (`#900`, Cronin et al. 2013)**:
  Fix `raji_dmso_vehicle_control` (`condition_control: "vehicle"`, `condition_drugs: "DMSO"`,
  `condition_control_for: "raji_entinostat_hdac_inhibitor"`), map the `16` DMSO-only and `10`
  entinostat-only statements with explicit `not_detected` comparators, and leave the `158`
  disjunctive `"DMSO or Entinostat-treated"` rows unmapped/unresolved (`axis_mismatch`).
- **PMID 32265295 (`#900`, Lorente et al. 2020)**:
  Replace pooled `cells_hla_b_27_05_wt_vs_erap2_ko` with `b27_c1r_wt`
  (`condition_knockout_genes: "none"`, `condition_mhc_context: "mhc_transfectant"`) and
  `b27_c1r_erap2_ko` (`condition_knockout_genes: "ERAP2"`,
  `condition_mhc_context: "mhc_transfectant"`), mapping `412` WT-only, `278` KO-only, and
  `6,258` shared rows.
- **PMID 29930882 (`#900`, Gravett et al. 2018)**:
  Link `hct_116_untreated_constitutive` as control for `hct_116_gemcitabine_treated` and map
  the `87` gemcitabine-exclusive rows and `35` (`34 + 1`) shared presence-and-absence rows.
- **PMID 30779550 (`#900`, Murphy et al. 2019)**:
  Replace generic cancer-cell-line placeholders with the four documented biological systems
  (`HCT 116`, `MDA-MB-468`, murine `EL4`, and murine `C57BL/6` EL4-recipient splenocytes, each
  with untreated/vehicle and doxorubicin arms under `sample_group`), map the `2,038`
  `"terated with doxorubicin"` HCT 116 rows, keep all blank/system-only rows `group_ambiguous`
  (`arm_not_recorded`), and add a `Direct Ex Vivo` conditional rule so mouse splenocytes do
  not inherit `override: cell_line`. Final source review (#922) adds distinct cultured
  and subcutaneous-tumor EL4 arms; EL4-only statements cannot assert cultured material.
- **PMID 35251023 (`#856`, Laghmouchi et al. 2022, DOI `10.3389/fimmu.2022.831822`)**:
  Replace generic `cell_lines_multiple_hla_dp_genotypes` with the 13 retrovirally transduced
  HLA-null K562 single-HLA-DP heterodimer arms (`condition_mhc_context: "mhc_transfectant;monoallelic"`,
  `condition_transduction: "HLA-DPA1;HLA-DPB1"`, `ip_antibody: "B7/21"`) listed in
  Supplementary Table 1. The deposit's additional DPA1*02:02/DPB1*19:01 restriction is
  retained without an asserted matching construct (#921); the earlier claim that all
  45,961 rows are qualified by 14 primary-source constructs is withdrawn.
- **Engineering drift cleanup (`#593`)**:
  Update PMID `28514659`'s three 721.221 sHLA transfectant arms to
  `condition_mhc_context: "mhc_transfectant;monoallelic;soluble_mhc"` and separate PMID
  `36796642`'s cultured K562 transfectants (`condition_material: "cultured"`) from primary
  tumor resections (`condition_material: "direct_ex_vivo"`), removing `36796642` from
  `tests/data/engineering_drift_baseline.yaml` in lockstep.

## Verification Gates

1. Replay all modified PMIDs (`30484775`, `31222486`, `23328677`, `32265295`, `29930882`,
   `30779550`, `35251023`, `28514659`) against `ci-corpus-v4` `observations.parquet` and verify
   exact row-count conservation and expected `condition_id`, `supported_condition_ids`,
   `excluded_condition_ids`, and `sample_attribution` distributions.
2. Verify `engineering_drift_audit()`, `sample_ploidy_audit()`, and `mhc_token_audit()` pass.
3. Add comprehensive unit and source-fixture regressions in `tests/test_context_evidence.py`
   and `tests/test_apc_intervention_curation.py`.
4. Pass `./format.sh`, `./lint.sh`, `./test.sh`, hosted CI, merge, and deploy `v2.1.11` to PyPI.
