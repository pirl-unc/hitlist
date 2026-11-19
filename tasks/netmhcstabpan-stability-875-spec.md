# NetMHCstabpan Quantitative Stability Source Qualification — Specification (`#875`)

Refs: `pirl-unc/hitlist#875`, `pirl-unc/presto#138`, `pirl-unc/mhcgnomes#268`.

## Goal

Admit and qualify the published NetMHCstabpan 1.0 pMHC-I half-life archive (`Stability.txt`, `28,166` measurements across `75` constructs; Rasmussen et al. 2016, `PMID 27402703` / `PMC4976001`) through Hitlist's existing download, builder, provenance, binding/observations, and model-training interfaces without inventing censoring thresholds, stripping mutant MHC identities into wild-type alleles, treating unannotated synthetic sequences as mass-spectrometrically verified unmodified peptidoforms, or duplicating DTU's prior 10-allele IEDB stability submissions.

## Primary-Source & Corpus Evidence

1. **Archive identity & license posture**:
   - Upstream URL: `https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt`
   - Pinned digest: SHA-256 `5f2c6e82d198705a66974b4e42a0ed1412124c4b1e8f7033c7cd43bbdac07f67`, `799,880` bytes (`28,167` lines: tab-separated header `HLA\tPep\tThalf` on line 1 followed by `28,166` whitespace-separated data rows on lines `2..28167`).
   - `PMC4976001` carries a text-mining / fair-use notice rather than a Creative Commons or Apache-compatible redistribution license, and DTU's supplement page states no open redistribution license. Therefore `Stability.txt` is **not** bundled into the wheel/sdist; it is acquired on demand via `datacache` (`hitlist data fetch netmhcstabpan` / `fetch_netmhcstabpan_source`) with pinned SHA-256 and byte-size verification, while Hitlist ships the authored Apache-2.0 source contract (`hitlist/data/evidence_revisions/netmhcstabpan_stability.json`).
2. **Exact reconciliation with `PMC4976001` Table 2 & synthetic-negative separation**:
   - All `75` constructs and all `28,166` rows in `Stability.txt` match `Table 2` of `PMC4976001` construct-by-construct with `0` count differences (`22,487` positive values in `[0.05, 256.70]` hours + `5,679` literal `0.00`-hour rows).
   - `PMC4976001` (*Data sets*) separately describes enriching ANN training with `1,000` NetMHCpan-2.8-selected random natural 9-mers per allotype (`75,000` synthetic `0`-hour rows). Those predictor-generated negatives are absent from `Stability.txt` and are never synthesized or mixed with experimental rows.
3. **Assay protocol, zero-hour nondetection, and >48h extrapolation (`PMID 21044632` / `PMC4341823`)**:
   - Measurements use the homogeneous scintillation proximity assay (SPA) monitoring dissociation of $^{125}\text{I}$-labeled human $\beta_2\text{m}$ from recombinant biotinylated MHC-I heavy chains at $37^\circ\text{C}$ over a 48-hour reading window after overnight ($18^\circ\text{C}$) folding with $10\,\mu\text{M}$ synthetic 9-mer peptide (Harndahl et al. 2011, `PMID 21044632`). Reported $t_{1/2}$ values are geometric means of $\ge 2$ independent experiments fitted to one-phase exponential decay $Y = Y_0 e^{-kT}$ (`Plateau = 0`).
   - Literal `0.00`-hour rows (`5,679` rows) correspond in DTU's own IEDB submissions (`submission_id` `1000631`–`1000641`, `446` matched 9-mer zero rows) to `qualitative_measurement = "Negative"`, `quantitative_value = 0.0`. Physically $t_{1/2} = \ln(2)/k_d > 0$; `0.00` represents failure to form a quantifiable complex at $18^\circ\text{C}$ or complete dissociation prior to the first $37^\circ\text{C}$ TopCount read cycle, with per-well detection floor unreported in `Stability.txt`. Zero rows are never assigned an invented positive floor (`0.05` h / `< 0.1` h) and are excluded from exact positive half-life regression (`prepare_stability_training_data`).
   - Positive values exceeding the 48-hour reading window (`354` rows in `(48.0, 256.7]` h; `2` rows at `48.00` h) come from unconstrained exponential rate fits without a hard numeric ceiling in `Stability.txt`; they retain their reported values and are annotated `positive_extrapolated_beyond_48h_window` without inventing a `> 48h` right-censor.
4. **Mutant MHC constructs (`1,135` rows across 3 constructs; `pirl-unc/mhcgnomes#268`)**:
   - `HLA-B*14:01(C67S)` (`374` rows), `HLA-B*14:02(C67S)` (`382` rows), and `HLA-B*39:06(C67S)` (`379` rows) are engineered B-pocket `Cys67 -> Ser67` mature heavy-chain point mutants (`Table 2` of `PMC4976001`).
   - They map via the explicit contract to `mhcgnomes`'s canonical mutant designations (`HLA-B*14:01 C67S mutant`, `HLA-B*14:02 C67S mutant`, `HLA-B*39:06 C67S mutant`) while retaining literal `reported_mhc_construct`. They are never stripped to wild-type `HLA-B*14:01` / `B*14:02` / `B*39:06`. Unrecognized unparseable constructs remain `allele_resolution = "unresolved"`.
   - `prepare_stability_training_data` and `prepare_affinity_training_data` gain `allow_mutant_mhc: bool = False`, excluding mutant MHC constructs (`mutant_mhc_construct_excluded`) by default so wild-type sequence lookups never silently consume mutant stability/affinity labels, while admitting them as `mutant_class_i` / `mutant_class_ii_pair` when `allow_mutant_mhc=True`.
5. **IEDB assay-lineage deduplication vs cross-study overlap warnings**:
   - DTU deposited 10 HLA-I alleles (`6,297` rows: `5,963` 9-mers + `334` non-9-mers) to IEDB under `http://www.iedb.org/reference/1028282`..`1028294` (`submission_id` `1000631`–`1000641`, *"Large scale analysis of peptide - HLA-I stability"*, citing `PMID 21044632`). All `5,963` 9-mers (`5,517` positive + `446` zero) match `Stability.txt` within `0.015` h (`0` value conflicts).
   - Confirmed DTU submission matches (`confirmed_dtu_iedb_submission_duplicate`) deduplicate during `build_observations()` via `provenance.redirect(..., "netmhcstabpan_dtu_iedb_submission_overlap")`, enriching the retained IEDB rows' contributor graph with `PMID 27402703`.
   - Overlaps with other IEDB references (e.g., `PMID 23543059` `reference/1026371`, `PMID 32983097` `reference/1037619`, Beckman Coulter iTopia `reference/1014192`) are retained and flagged (`cross_reference_exact_value_overlap_warning` or `cross_reference_pmhc_different_value_overlap`) with semicolon-joined `iedb_overlap_assay_iris` and `iedb_overlap_reference_iris`.

## Architecture & Module Plan

1. **`hitlist/data/evidence_revisions/netmhcstabpan_stability.json`**:
   - Authored machine-readable source contract pinning URL, SHA-256, byte size, citation/assay protocol metadata, the 3 verified `C67S` construct mappings, the 10 DTU IEDB submission reference/submission IDs, and all 75 per-construct `Table 2` row counts.
2. **`hitlist/netmhcstabpan.py`** (exported lazily from `hitlist`):
   - `netmhcstabpan_source_contract() -> dict`
   - `fetch_netmhcstabpan_source(cache_directory=None, *, force=False, verbose=True) -> Path`
   - `scan_netmhcstabpan_stability(path=None, *, provenance=None, verify_pinned_digest=True, classify_source=True) -> pd.DataFrame`
   - `reconcile_netmhcstabpan_with_iedb(stab_df, binding_df, *, value_tolerance_hours=0.015) -> pd.DataFrame`
   - `deduplicate_netmhcstabpan_against_binding(stab_df, binding_df, *, provenance=None, value_tolerance_hours=0.015) -> tuple[pd.DataFrame, dict]`
   - `load_netmhcstabpan_stability(path=None, *, binding_df=None, deduplicate_confirmed_iedb=False, allow_download=False, verify_pinned_digest=True) -> pd.DataFrame`
3. **`hitlist/downloads.py` & `hitlist/builder.py`**:
   - Register `"netmhcstabpan"` in `FETCHABLE_DATASETS` with `expected_sha256` and `expected_size`, and enforce `expected_sha256` / `expected_size` in `fetch()` and `_cached_file_problem()`.
   - Include `"netmhcstabpan"` in `builder._source_paths()` and `builder._build_observations()` so when registered, `hitlist build observations` scans `Stability.txt`, deduplicates confirmed DTU IEDB lineage rows into `observation_contributors.parquet`, annotates cross-reference overlaps, validates MHC tokens, maps peptides, and writes qualified rows into `binding.parquet` without altering `ci-corpus-v4` cache validity when `"netmhcstabpan"` is unregistered.
4. **`hitlist/model_training.py`**:
   - Enforce `half_life > 0` in `_prepare_quantitative_binding_task` so literal `0.0`-hour / `0.0`-minute half-life nondetections route to `excluded` (`non_positive_measurement_value`) rather than entering `accepted` as exact zero half-lives.
   - Distinguish mutant MHC constructs (`is_mutant_mhc`, `allele_completeness in {"mutant_class_i", "mutant_class_ii_pair", "mutant_other"}`) and add `allow_mutant_mhc: bool = False` (`exclusion_reason = "mutant_mhc_construct_excluded"` by default) to `prepare_stability_training_data` and `prepare_affinity_training_data`.
   - Preserve NetMHCstabpan provenance/overlap columns (`reported_mhc_construct`, `source_line_number`, `stability_value_status`, `censoring_evidence_status`, `construct_resolution_status`, `peptide_chemistry_status`, `iedb_overlap_status`, `iedb_overlap_assay_iris`, `iedb_overlap_reference_iris`) when present on candidate frames.
5. **Tests, CI Qualification Workflow & Docs**:
   - `tests/test_netmhcstabpan.py`: synthetic + real-source unit/integration round-trip tests covering checksum/tamper rejection, line locators, zero-hour & >48h semantics, C67S vs unknown mutant construct handling, DTU IEDB lineage deduplication (`provenance.redirect`) vs cross-study overlap warnings, `build_observations` -> `load_binding` -> `generate_binding_table` -> `generate_training_table` -> `write_training_bundle` -> `prepare_stability_training_data` round-trip, and `downloads.fetch("netmhcstabpan")` checksum verification.
   - `.github/workflows/netmhcstabpan-stability.yml`: hosted CI qualification workflow fetching live checksum-pinned `Stability.txt` and running the real-source reconciliation suite.
   - Update `docs/model-training.md` with NetMHCstabpan stability qualification, zero-hour/mutant policies, and IEDB overlap handling.
