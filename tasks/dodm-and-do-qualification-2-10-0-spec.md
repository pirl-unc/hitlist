# Spec: `hitlist==2.10.0` — DO:DM Ratio Peptidomics (`PMID 35085787` / `PXD024392`, #860) & Nanaware 2019 HLA-DO Source Qualification (`PMID 30573663` / `PXD010301`, #816)

## Goals

1. **DO:DM Ratio Peptidomics Adapter & Contract (`hitlist/dohm_ratio_peptidomics.py` + `hitlist/data/evidence_revisions/dohm_ratio_35085787.json`, #860):**
   - Pins CC-BY 4.0 `PMC10329146.1/mmc4.xlsx` (`7,838,853` bytes, SHA-256 `1b0b735fe1d5400651ae125b3da427d2d26004c42ecacda91029860546e5fb27`) plus `PXD024392` run/TMT channel mapping files.
   - Parses `Supp_data_MHC_peptides` (36,860 identification rows, 6,773 modified strings preserved verbatim with flanks, confidence fields `Xcorr`, `10lgP`, `ALC`, `Q_value`, `PEP_score`, and parent protein alternatives) and `Supp_data_MHC_quant` (10,587 rows × 36 runs = 381,132 cells: 60,072 positive, 0 explicit zeros, 321,060 missing — never converted to zeros or negatives).
   - Preserves exact `HLA-DRA*01:01/DRB1*04:01` pair, clone mapping (`G8` `(DO:DM)L`, `1C3` `(DO:DM)M`, `2D7` `(DO:DM)H`, `2D7_KO` `DOKO` derived from `2D7`), biological/technical replicate hierarchy, and strict namespace separation from `whole_cell_tmt_proteome`, `plateau_imputed_cores`, and computational `predicted_netmhc_affinity` (`Table S3 Affinity(nM)`).
   - Registers `"dodm-ratio-35085787"` in `hitlist/_dataset_catalog.py`.

2. **Nanaware 2019 HLA-DO Qualification & Title-Intensity Parser (`hitlist/nanaware_2019_do.py` + `hitlist/data/evidence_revisions/nanaware_2019_pxd010301.json`, #816):**
   - Pins official `MSV000082570` / `PXD010301` (`PMID 30573663`, `PMC6398211`) qualification contract separating human LG-2 control/DO-deficient clones from mouse splenic B cells, flagging legacy IEDB axis mismatch and outcome-selected S13 isotope-standard bias, and preserving explicit non-OA retrieval failure receipts for S10–S14 (`139110_1_supp_238824_pjk8j3.xlsx`–`139110_1_supp_238828_plk8l3.xls`).
   - Provides a source-faithful parser `extract_title_encoded_intensities()` preserving exact spectrum/identification IDs, duplicate-value detection, missingness, and `unresolved_unit_and_aggregation_semantics` without fabricating peptide quantities.
