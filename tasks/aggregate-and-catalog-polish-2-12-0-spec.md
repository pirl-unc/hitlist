# Hitlist v2.12.0 Spec: Aggregate Allele Deduplication & Evidence Sort (#957) + Structured Catalog & Symlink Consistency (#958)

## Scope

1. **#957 — `hitlist.aggregate` multi-allelic deduplication and pMHC evidence ordering:**
   - In `aggregate_per_peptide`:
     - Parse semicolon-delimited multi-allelic donor-set strings before deduplicating so `ms_alleles` contains deduplicated individual alleles (never duplicating `"HLA-A*02:01;HLA-A*02:01;..."`) and `ms_allele_count` counts distinct individual alleles.
     - Add explicit `ms_exact_alleles` (`;`-joined distinct single-allele restrictions) and `ms_restriction_sets` (`|`-joined distinct raw restriction strings preserving donor-set boundaries).
   - In `aggregate_per_pmhc` and `aggregate_per_pmhc_with_refs`:
     - Sort within each peptide by evidence strength: exact single-allele restriction first, then monoallelic evidence (`ms_pmhc_has_mono_evidence` / `ms_pmhc_mono_hit_count > 0`), then PMID/reference count descending, then hit count descending, then `mhc_restriction` ascending.

2. **#958 — Downloads catalog structure, `hitlist data preflight`, and consistent symlink status:**
   - Add `available_dataset_catalog(category=None)` and `dataset_categories()` in `hitlist/downloads.py` (exported in `hitlist/__init__.py`) providing structured `category`, `group`, `species`, `kind`, and `description` metadata for every dataset without breaking `available_datasets() -> dict[str, str]`.
   - Enhance `hitlist data available` with `--category` and `--grouped` options and include structured metadata fields in `--json` output (alongside existing `hitlist data preflight` and `hitlist data fetch --dry-run`).
   - Align symlink inspection across `downloads.info()`, `downloads.preflight()`, `cache_inventory.list_cache_files()`, and `builder.build_observations()` (#954): report `is_symlink`, `resolved_path`, and `symlink_status` (`regular_file`, `valid_symlink`, `broken_symlink`, `non_regular_target`, `missing`), inspecting valid symlink targets cleanly while flagging broken/non-regular symlinks upfront.
