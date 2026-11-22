# Hitlist v2.11.0 Spec: Stopfer 2021 Iso18 Calibration (#849) & 7 Drug Studies Qualification (#903)

## Scope

1. **#849 — Stopfer 2021 (`PMID 34497125`, `PMC8449407`, `PXD024917`) Iso18 known-dose / raw-signal calibration:**
   - Pin all 5 PMC Open Data S3 workbooks/PDF (`sd01`, `sd03`, `sd06`, `sd07`, `sapp.pdf`) and 3 PRIDE `PXD024917` Excel Skyline exports (`Iso18_DirectSpike_TransitionResults.xlsx`, `Iso18_hipMHCspikein_TransitionResults.xlsx` noting the S7 double-`i` spelling `Iso18_hipMHCspiikein_TransitionResults.xlsx`, and `SKMEL5_Transitions_DMSO_MEKi.xlsx`) with verified SHA-256 hashes.
   - Parse and independently replay all 348 S3A/S3B calibration worksheet slots (`213` S3A positive cells across `71` dose rows + `135` S3B slots = `126` positive + `9` literal blanks), explicitly preserving the absent `GLFDQHFRL[+7]` `0.1 fmol` dose row without inventing a zero or fake cell coordinate.
   - Keep observation families strictly separated: `free_1h_direct_spike_summed_signal` (S3A), `embedded_2h_hipmhc_pre_ip_summed_signal` (S3B), `selected_transition_count` (S3C — never signal intensity), `predicted_affinity_nm` (S1 — never experimental binding), `analytical_4h_trigger_reagent`, and `skmel5_binimetinib_vs_dmso_transitions` (`SKMEL5_Transitions_DMSO_MEKi.xlsx`).
   - Cross-check S3A/S3B ordinal columns `C`/`D`/`E` against PRIDE transition exports without blindly assuming `C = run 1`, proving that all 9 S3B `100 fmol` Column E blanks correspond to missing PRIDE acquisition `100fmol_2` while Column D carries `100fmol_3`.
   - Audit S6A vs S6B isotope definitions (`72` isotopologue rows: `17` consistent, `54` inconsistent due to S6A Valine `13C3, 15N1 / 4.007 Da` vs S3/S6B `V[+6]`, `1` uncheckable Methionine `B99`/`E99` omitted from S6A) and emit row-level support/exclusion flags without altering original values.
   - Register `"stopfer-2021-iso18"` in `_dataset_catalog.py` and expose a peptide-grouped empirical calibration recipe explicitly blocking MS1 or APC-concentration transfer.

2. **#903 — Qualify remaining 7 annotated drug studies (+ #905 umbrella sync):**
   - Pin PMID, PMCID, DOI, repository accession, exact primary-source compounds/doses/times, genetic/background factors, vehicle/control edges, and typed eligibility status for all 7 studies (`34497125` binimetinib, `34129938` dTAG-13/doxycycline, `33981976` 5-azacytidine, `33912179` carbamazepine, `25576301` imatinib, `29632711` lenalidomide, `30524438` BET degraders ARV-825 / ARV-771 / JQ1-MDM2 + JQ1 control in BV173).
   - Provide verified primary-source ingestion for the two studies with directly recoverable per-arm quantitative tables:
     - `34497125` (Stopfer 2021 `PXD024917` `SKMEL5_Transitions_DMSO_MEKi.xlsx`: 100 nM binimetinib vs DMSO vehicle).
     - `34129938` (Pollock 2021 `sbpollock/NeoToma2021` commit `f7df59fd8a9e93e7fdb66b7ab3c18d017723614e`: `Fig4/Fig4C/toma1..3.csv` targeted MS3 TMT11-plex with proven channel map `MS3Quant2`=Control, `MS3Quant6`=Dox, `MS3Quant9`=Both, `MS3Quant10`=dTag, plus `Fig4/Fig4E/untargeted_MS2.csv` TMT6-plex `127`/`129`/`130`/`131` with blank QC channels `126`/`128` preserved separately and `engineered_construct_tool` mechanism class).
   - Register `"pollock-2021-neotoma"` in `_dataset_catalog.py`.
   - Record machine-readable explicit source gaps (with searched sources and exact missing relations, zero fabricated untreated arms) for the remaining 5 studies (`33981976`, `33912179`, `25576301`, `29632711`, `30524438`).
