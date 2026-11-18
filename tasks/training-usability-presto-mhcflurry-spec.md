# Training Data Usability for Presto and MHCflurry — Specification (v2.1.18)

Refs: `pirl-unc/hitlist#706`, `pirl-unc/hitlist#931`, `pirl-unc/hitlist#678`, `pirl-unc/hitlist#680`, `openvax/mhcflurry#494`, `pirl-unc/presto#34`, `pirl-unc/presto#138`, `pirl-unc/presto#140`.

## Goal

Make Hitlist exports and verified `training_bundle` artifacts directly usable by Presto and MHCflurry training pipelines without consumer-side ad-hoc endpoint regexes, unit conversions, silent inequality drops, arbitrary multi-mapping flank selection, or conflation of purified-MHC binding assays with unresolved tissue-donor observations.

All raw source fields, historical bundle verifiers, and existing split-audit verdicts remain unchanged. Every task-specific training preparation view conserves 100% of input candidate rows across `accepted` + `excluded` with machine-readable reason codes and retained provenance.

## Scope & Architecture

### 1. Assay Preparation & Biological-Independence Axis Applicability (`#706`)

In `hitlist/lineage.py`, `hitlist/split_audit.py`, `hitlist/training_bundle.py`, `hitlist/export.py`, and `hitlist/evidence_bundle.py`:

- Add `classify_assay_preparation(assay_method, evidence_kind="", *, is_ms_observation=False)` returning `(assay_preparation, assay_preparation_basis)` strictly from explicit preparation statements and assay modality — never inferred from numeric endpoint names (`IC50`, `KD`, etc.):
  - `"purified_mhc"` (`"explicit_purified_mhc_method"`): `purified mhc` without conflicting cellular/lysate/MS evidence.
  - `"cellular_mhc"` (`"explicit_cellular_mhc_method"`): `cellular mhc` without conflicting preparation tokens.
  - `"lysate_mhc"` (`"explicit_lysate_mhc_method"`): `lysate mhc` (cell-lysate MHC assembly/binding from biological cells).
  - `"cell_associated_ms"` (`"native_elution_evidence"`): positive MS/elution observations without conflicting `purified mhc` claims.
  - `"unknown"` (`"conflicting_assay_preparation"` or `"unspecified_assay_preparation"`): contradictory preparations, multiplexed/generic `"binding assay"`, or blank non-MS methods.
  - Support generated negative/control rows (`generated_id` + `parent_assay_contexts` JSON list of `{"method": str, "evidence_kind": str}`): inherit unanimous non-unknown parent preparation with basis `"generated_parent_consensus"`, otherwise `"unknown"` (`"generated_parent_unresolved_or_conflicting"`).
- Add `attach_lineage_applicability(frame, *, copy=True)` (and `include_applicability: bool = False` on `attach_lineage` so frozen `species_evidence` bundle replays keep their exact schema), populating `APPLICABILITY_COLUMNS`:
  - `assay_preparation`, `assay_preparation_basis`
  - `experimental_origin_applicability`, `specimen_applicability`, `donor_applicability`, `acquisition_applicability`
  - Values in `("applicable_resolved", "applicable_unresolved", "not_applicable", "applicability_unknown")`:
    - `experimental_origin` and `acquisition` are always applicable (even purified recombinant MHC preparations can share protein preparations, experiments, or runs).
    - `donor` and `specimen` are `"not_applicable"` for `"purified_mhc"` (unless an explicit reviewed ID is resolved), `"applicable_resolved"` / `"applicable_unresolved"` for `"cellular_mhc"`, `"lysate_mhc"`, and `"cell_associated_ms"`, and `"applicability_unknown"` for `"unknown"` preparation when unresolved.
  - Original `experimental_origin_status`, `specimen_status`, `donor_status`, and `acquisition_status` remain untouched.
- Wire `include_applicability=True` in `generate_training_table` (and matching `evidence_bundle` concatenation), keeping `_project_training_columns` narrow-projection columns unchanged while retaining `APPLICABILITY_COLUMNS` in `training_bundle.IDENTITY_COLUMNS`.
- Extend `audit_splits` (`hitlist/split_audit.py`) and `training_bundle._coverage`:
  - Preserve existing `POLICIES` verdicts unchanged (`not_applicable` on `donor`/`specimen` never grants an overall independence `"pass"`).
  - Report `by_assay_preparation` per partition and per-axis applicability denominators (`n_applicable_observations`, `n_applicable_resolved_observations`, `n_applicable_unresolved_observations`, `n_not_applicable_observations`, `n_applicability_unknown_observations`) in `coverage[axis]` and `applicability_status` in pairwise `overlaps[axis]`.
  - Gate `manifest["coverage"]["applicability"]` on `"assay_preparation" in identities.columns` so `verify_training_bundle` and `verify_evidence_bundle` continue accepting historical pre-applicability schema-1 bundles (`human-cohort-corrected-v1`).

### 2. Task-Specific Presto & MHCflurry Training Preparation API (`hitlist/model_training.py`)

Exported lazily from top-level `hitlist` (preserving `generate_training_table not in hitlist.__all__` per 2.0 contract):

- `TrainingTableSelection(accepted: pd.DataFrame, excluded: pd.DataFrame, summary: dict)`
  - Invariant: `len(accepted) + len(excluded) == summary["n_input_candidate_rows"]`, with `n_*` unit-suffixed counts throughout.
  - Accepts either a verified `training_bundle` directory (`Path | str`, calling `verify_training_bundle` before reading and recording `bundle_manifest_sha256` in `summary`) or an in-memory `pd.DataFrame`.
- `qualify_binding_measurements(df: pd.DataFrame) -> pd.DataFrame`:
  - Non-destructively adds:
    - `binding_endpoint_family`: `"ic50"`, `"kd_approx_ic50"`, `"kd_approx_ec50"`, `"kd"`, `"ec50"`, `"ka"`, `"half_life"`, `"thermal_stability"`, `"on_rate"`, `"off_rate"`, `"qualitative_binding"`, `"mhc_binding_unspecified"`, `"other_binding"`, `"not_binding"`.
    - `standardized_value` (`float64`), `standardized_unit` (`"nM"`, `"min"`, `"celsius"`, `""`), `unit_conversion_status` (`"canonical_unit"`, `"linear_converted"`, `"nonlinear_or_unsupported_unit"`, `"missing_unit"`, `"no_numeric_value"`, `"not_applicable"`). Never exponentiates `log10(...)` units and never converts `IC50` into `KD`.
    - `censoring_status` (`"exact"`, `"unreported_operator_numeric"`, `"left_censored_lt"`, `"left_censored_lte"`, `"right_censored_gt"`, `"right_censored_gte"`, `"operator_without_value"`, `"no_numeric_value"`, `"unrecognized_inequality"`, `"not_applicable"`) and `normalized_inequality` (`"="`, `"<"`, `">"`, `""`).
- `prepare_affinity_training_data(source, ...) -> TrainingTableSelection`:
  - Produces MHCflurry- and Presto-ready molar affinity rows (`allele`, `reported_mhc_restriction`, `allele_completeness`, `peptide`, `measurement_value` in `nM`, `measurement_inequality` in `{"=", "<", ">"}`, `measurement_type`, `measurement_kind`, `measurement_source`, `censoring_status`, `unit_conversion_status`, `assay_preparation`, `donor_applicability`, `specimen_applicability`, provenance & lineage columns).
  - Explicit options for `endpoints` (e.g., `("ic50",)` for strict literal IC50 vs molar affinity set), `blank_inequality_policy` (`"treat_as_exact"` vs `"exclude"`), `allow_censored`, `mhc_class`, `species`, `scorable_class_i_only` (uses `class_i_prediction_scope` two-field protein identities for MHCflurry `Class1AffinityPredictor`), `allow_single_chain_class_ii`, `allow_modified_peptides`, and peptide length bounds.
- `prepare_stability_training_data(source, ...) -> TrainingTableSelection`:
  - Produces Presto-ready `half_life` (`min`) and `thermal_stability` (`celsius`) supervision tables cleanly separated from affinity, with full censoring, provenance, and exclusion accounting.
- `prepare_ms_training_data(source, ...) -> TrainingTableSelection`:
  - Collapses protein-mapping expansions to observation grain (`1 row per evidence_row_id`), preventing multi-gene weight inflation and arbitrary flank selection (`pirl-unc/presto#34`):
    - Computes `n_protein_mappings`, `n_gene_mappings`, `n_distinct_flank_pairs`, and `flank_status` (`"not_mapped_in_input"`, `"unmapped"`, `"unique_mapping"`, `"concordant_flanks"`, `"expression_disambiguated"`, `"ambiguous_flanks"`).
    - Blanks `n_flank` and `c_flank` when `flank_status == "ambiguous_flanks"` while retaining `candidate_flank_pairs`, `candidate_protein_ids`, and `candidate_gene_ids`.
  - Qualifies presenting MHC candidate bags (`candidate_alleles`, `n_candidate_alleles`, `candidate_bag_status`, `candidate_bag_provenance`):
    - Distinguishes `"monoallelic_single"`, `"experimental_single"`, `"predicted_single_restriction"` (excluded unless `allow_predicted_restrictions=True`), `"sample_candidate_bag"`, and `"study_pool_candidates"` (excluded unless `allow_study_pools=True`).
    - Supports `scorable_class_i_only=True` and `max_alleles_per_bag` (e.g., `max_alleles_per_bag=6` for MHCflurry `Class1PresentationPredictor`).

### 3. Documentation & Regression Verification (`#931`)

- Fix stale 1.x imports and removed `hitlist export training` / `hitlist data build` commands in `docs/training-provenance.md` and `docs/index.md` (`#931`).
- Add a dedicated guide `docs/model-training.md` (wired into `mkdocs.yml`) showing end-to-end Presto and MHCflurry workflows, applicability audits, flank disambiguation, and bundle verification.
- Verify against synthetic unit cases, `ci-corpus-v4` (all 814,268 binding rows), and `human-cohort-corrected-v1` (reproducing exact `2,533` purified-MHC vs `81` cellular-MHC human IC50 counts and offline historical bundle verification).
