# Spec: `hitlist==2.7.0` — Unified MHC Genotype Recovery, Non-Human Normalization & Cross-Species APM Orthologs (#943, #944)

## Goals

1. **Vectorized Condition-Level MHC Genotype Recovery (`hitlist/mhc_genotypes.py`, #943 & #944):**
   - Align `_allele_is_exact_molecule()` with `curation._names_one_molecule()` (`mhcgnomes` species haplotype roster + `get_known_allele`) so typos are rejected while all canonical mouse, rat, NHP, canine, bovine, porcine, equine, and avian alleles resolve cleanly.
   - Add `recover_mhc_genotypes_for_frame(df, *, overwrite_resolved=False)` and `summarize_mhc_genotype_recovery(df)` that enrich any MS, perturbed MS, binding, or T-cell DataFrame with unified `mhc_allele_set`, `mhc_allele_set_size`, `mhc_allele_cardinality` (`monoallelic` / `multi_allelic_bag` / `class_only_unknown`), `mhc_allele_set_provenance`, `mhc_species`, `host_species`, and `is_chimeric_mhc_host`.
   - Deduplicate unique row-context keys before resolution so multi-million-row corpus frames process in seconds.
   - Never overwrite already-resolved primary-source genotypes unless `overwrite_resolved=True`, and never conflate prediction-derived restrictions (`MHC binding prediction`) with experimental restriction.

2. **Cross-Species APM Ortholog Mapping (`hitlist/data/apm_orthologs.yaml` + `hitlist/apm_orthologs.py`, #944):**
   - Curated YAML mapping 37 core antigen processing & presentation genes across human, mouse, rat, macaque, dog, cow, pig, and chicken, including explicit `absent_in_species` annotations (e.g. `ERAP2` absent in `Mus musculus` and `Rattus norvegicus`).
   - Provide `load_apm_orthologs()`, `canonical_apm_ortholog_group()`, `species_ortholog_symbols()`, `is_apm_gene_absent_in_species()`, and `map_apm_genes_to_canonical()`.

3. **Multi-Species & Genotype-Recovery Support in `hitlist/model_training.py` (#943 & #944):**
   - Support `species="all"` (alias for `species=None`) uniformly across `prepare_ms_training_data`, `prepare_affinity_training_data`, and `prepare_stability_training_data`.
   - Add `recover_reference_genotypes: bool = False` to `prepare_ms_training_data()` to recover `mhc_allele_set` bags from `host_mhc_types`, curated cell lines, and inbred/transgenic animal strains when primary restriction is `class_only` / `unresolved`.
   - Expose `mhc_allele_cardinality`, `host_species`, and `is_chimeric_mhc_host` on `prepare_ms_training_data().accepted`.
