# APC/Cytokine Contrasts (#904) & DLBCL Dose-Unit Conflict Preservation (#930)

Date: 2026-11-19
Target release: `hitlist==2.1.23`
Issues: `#904`, `#930` (unblocks `pirl-unc/presto#155`, `pirl-unc/presto#157`)

## 1. `#930` — DLBCL (`PMID 35561310` / `PMC9327544`) dose-unit conflicts & assay-specific scope

- **Verified primary sources**:
  - `PMC9327544.xml`: `119,741` bytes, SHA-256 `78f96efc3dbc9412ee3cb3fc3249271efe54ba84c6ec2f3b338804f83e64b444`.
  - Supplemental PDF `advancesADV2021006069-suppl6.pdf`: SHA-256 `bfae2561be9bf6e031cdc3d420d52ef00a30908435d3bf60d0e4a821cece06f1` (Supplemental Figure 10, PDF page 11 / printed page 43).
- **Literal statements preserved in `hitlist/data/evidence_revisions/dlbcl_35561310_treatment_conflicts.json`**:
  1. `Figure 5` (immunopeptidomics MS, primary interpretation for the 15 `ms_samples` arms): `125 nM` decitabine, `1 uM` tazemetostat, `100 ng/mL` IFN-gamma (`48 h` then refreshed `48 h`).
  2. `Drug treatments` section (Methods): `125-2000 nmol/mL` decitabine (`1 nmol/mL == 1 uM == 1000 nM`, NEVER normalized as `nM`), `312.5-5000 nmol/mL` tazemetostat (`1000x` unit discrepancy against `125 nM` / `1 uM`), `1-100 ng/mL` IFN-gamma.
  3. `Supplemental Figure 10` (PDF page 11 / printed page 43): `100 ug/ml IFNg` (`1000x` unit discrepancy against Figure 5's `100 ng/mL`, NEVER silently repaired to `ng/mL`).
  4. `Figures 2-3` (RT-qPCR transcript assays): `100 nM` decitabine, `1 uM` tazemetostat, `10 ng/mL` IFN-gamma (distinct `transcript_expression` endpoint scope, kept strictly separate from `peptide_detection`).
- Retains the 15 existing `ms_samples` arms and `elution_condition_ids` in `pmid_overrides.yaml` while annotating `#930` conflicts on DLBCL rows without blocking usable Figure 5 peptide-detection evidence.

## 2. `#904` — Separating isolated cytokine/stimulus contrasts from differentiation cocktails & non-APC populations

### A. Primary-source curation & declarative token-list statement matching
1. Add validated `elution_token_condition_ids` support to `hitlist/curation.py` (`PMID_ENTRY_FIELDS`, `validate_elution_condition_ids`) and `StatementMap` in `hitlist/export.py`: matches a pinned prefix + closed comma-separated token map (`None` on any unknown token; never guesses).
2. **`PMID 32983136` (Marino 2020 / `PMC7485268`, `34,606` rows)**:
   - Map all 7 exact deposited tokens (`CD14`, `ImDC`, `ImmDC`, `MaDC`, `CD19`, `CD4`, `CD8`) covering 100% of all `503` distinct `assay_comments` strings (`0` unmapped): recovers all `6,819` single-subset rows (`4,730 MaDC`, `1,046 CD19`, `370 ImDC/ImmDC`, `363 CD14`, `197 CD4`, `113 CD8`) to exact `condition_id` and preserves all `27,787` shared multi-subset rows as `pmid_ambiguous` (`condition_id=""`) with exact `supported_condition_ids` (removing `2,954` false `cd8_t_cells_activated` attributions).
   - Add primary-source comparator links: `cd14_monocytes_resting -> immature_dcs` (`differentiation_cocktail_contrast`, `CSF2 + IL4` 6d) and `immature_dcs -> mature_dcs` (`multifactor_maturation_contrast_on_differentiated_background`, `IFNG + LPS` 24h on fixed `CSF2 + IL4` background). Keep `cd19_b_cells_activated` (`IL4 + CD40LG`) and `cd4`/`cd8` (`PMA + ionomycin`) unpaired (`constant_cocktail_context_only` / non-APC).
3. **`PMID 33392160` (Ghosh 2020 / `PMC7773052`, `9,900` rows)**:
   - Separate the three real primary-source arms (`THP-1 (untreated control)`, `THP-1 + biomaterial contact`, `THP-1 + LPS positive control (50 ng/mL, 24h)`) with `arm_resolution: no_row_discriminator` so the `9,900` unattributed IEDB rows land honestly at `pmid_ambiguous` instead of falsely asserting simultaneous `LPS;biomaterial_contact` on every row.
4. Add explicit `condition_control_for` links on verified matched same-system cytokine/stimulus pairs (`29242379`, `32488085`, `32938616`, `33043033`, `33936100`, `33968037` Class I only, `34087483`), keeping unpaired/constant-background studies (`28832583`, `30484775`, `31844290`, `33460454`, `34129938`, `35051231`, `38920720`) explicitly unlinked as cytokine contrasts.

### B. Shared contrast catalog & qualification recipe (`hitlist/apc_contrasts.py`)
- Ship `hitlist/data/evidence_revisions/apc_stimulus_protocol_context.json` recording measured cell population + CL ontology, `is_measured_apc`, partner/feeder cell roles (`PMID 28832583` `TIL1`/`TIL3` -> `expanded_t_lymphocyte`, `is_measured_apc=False`, `partner_cell_population="irradiated_feeder_cells"`), differentiation vs acute stimulus vs expansion protocols, factor classes (`cytokine` vs `non_cytokine_stimulus` vs `small_molecule_drug`), and literal doses/units/durations/order.
- Implement in `hitlist/apc_contrasts.py` (lazily exported from `hitlist/__init__.py`):
  - `dlbcl_treatment_conflict_contract()`
  - `apc_stimulus_protocol_contract()`
  - `build_stimulus_contrast_catalog(samples_df=None)`
  - `prepare_stimulus_contrast_data(source, *, target_factor=None, allow_multifactor_contrasts=False, include_context_only=False, exclude_class_pool=True, exclude_non_apc=False)` returning `TrainingTableSelection(accepted, excluded, summary)` with 100% row conservation and explicit exclusion reasons (`class_pool_attribution_excluded`, `unattributed_or_mixed_population_excluded`, `constant_cocktail_context_only`, `constant_cytokine_background_excluded`, `expansion_protocol_non_apc_context_only`, `unpaired_cytokine_context_only`, `multifactor_contrast_excluded`, `non_apc_population_excluded`, `target_factor_not_present`, `no_stimulus_context_in_study`).

### C. Verification
- Unit tests in `tests/test_apc_cytokine_contrasts.py` covering `#930` literal unit-conversion traps (`nmol/mL` != `nM`, `ug/ml` != `ng/mL`, transcription vs MS scope), `#904` 20-PMID contrast catalog, `32983136` single-arm vs multi-arm membership expansion, `33392160` non-attribution of unattributed rows, `38920720` constant cocktail exclusion, `28832583` non-APC TIL exclusion, `30833945` multifactor `IFNG;TNF` non-isolation, and class-pool row exclusion.
- `@pytest.mark.integration` test verifying against `ci-corpus-v4` (`32983136` `6,819` single-arm + `27,787` shared multi-arm; `38920720` `44` context-only; `33392160` `9,900` ambiguous; `30833945` `7,524` ambiguous; `28832583` `53,689`; `35561310` `7,421` with `#930` conflict flag).
