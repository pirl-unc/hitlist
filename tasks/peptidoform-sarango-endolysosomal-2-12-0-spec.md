# Hitlist v2.12.0 Spec: Peptidoform Provenance (#896), Sarango 2022 PXD024417 (#858), and Öztemiz 2026 Endolysosomal Qualification (#812)

## Scope

1. **#896 — Source peptidoform and preparation-chemistry provenance for Stražar 2023 (`PMID 37301199`, `MSV000089225`) & Abelin/MAPTAC (`PMID 31495665`):**
   - Add `hitlist/data/evidence_revisions/peptidoform_preparation_contracts.json` and `hitlist/peptidoform_provenance.py`.
   - Record preparation chemistry & timing (Stražar 2023 `PMC10519123` iodoacetamide alkylation -> fixed cysteine carbamidomethylation + variable modifications; Abelin 2019 `PMID 31495665` MAPTAC DataS1B) separately from biological native modification status.
   - Preserve explicit `unmodified`, `modified`, `preparation_derivatized_only`, and `unresolved_form` dispositions; never blanket-fill `has_ptm=False` or collapse multi-form PSM matches into a single form.
   - Provide `join_psms_to_supplement_rows()` with exact `source_file`/`run_id`/`scan_number` provenance, separation of current-study rows (`internal_aff_clean1and2`) from reused IEDB/Abelin/decoys, and strict guard against silently aggregating modified pMHCs into unmodified affinity measurements.

2. **#858 — Sarango 2022 (`PMID 36215666`, `PXD024417`) run-level MHC-II qualification & note rewording:**
   - Add `hitlist/data/evidence_revisions/sarango_2022_pxd024417.json` and `hitlist/sarango_2022_pxd024417.py`.
   - Pin the verified 81-file PRIDE inventory (`40` `.raw`, `40` `.msf`, `1` `.sdrf.tsv` SHA-256 `793eeb55d34226b7bc42c692590d5fe79bbc469518ba423a09082931d2ea235a`, 40/40 raw filename match).
   - Separate biological material, condition (`Mock`, `siCTRL`, `siT6BP`), class/capture (`W6/32` vs `Tue39`), and 5 technical acquisitions per group; explicitly record that the second biological siRNA experiment mapping is unresolved in PRIDE.
   - Keep community SDRF MHC typing (`mhc typing method = not available`) strictly secondary — never promote to primary genotype, never restore withdrawn Class-I alleles, never manufacture a DRA allele.
   - Keep peptide MS intensities distinct from PLAtEAU core summaries and computational `DRB1*01:02` affinity estimates, and prevent expanding shared IEDB memberships or turning missing detections into negatives.
   - Reword the blanket reduction claim in `hitlist/data/pmid_overrides.yaml` for PMID `36215666` to repertoire alteration.
   - Register `"sarango-2022-pxd024417-sdrf"` in `hitlist/_dataset_catalog.py`.

3. **#812 — Öztemiz 2026 (`PMID 42846314`, `PMC13643861`, `10.3389/falgy.2026.1936335`) endolysosomal qualification & PeptideStudio synthetic exclusion guard:**
   - Add `hitlist/data/evidence_revisions/oztemiz_2026_endolysosomal.json` and `hitlist/endolysosomal_qualification.py`.
   - Inventory `PMC13643861` supplements (`Datasheet1..5.pdf`, no public PRIDE/MassIVE/Zenodo tabular peptide dataset), biological system (`Mus musculus` JAWSII isolated endolysosomal extract vs intact cells), substrates (`Bet v 1`, `Aes h 1`), preparation chemistry (`20 mM` iodoacetamide alkylation), and separate measurement families (`peptide_ms_signal`, `substrate_gel_depletion`, `cathepsin_zfr_amc_reporter_activity`).
   - Implement `guard_against_synthetic_peptidestudio_data()` rejecting `oztemiz-elif/peptide-studio` commit `18dd99da4a619abd8a28c7622e0809eb2c03e0ff`, `Example_degradome.xlsx`, `example.fasta`, and human myoglobin `P02144` synthetic demo tables, plus `endolysosomal_degradation_recipe()` rejecting unknown quantity units from concentration training while preserving dependent time points and short/nonbinding proteolytic fragments.
