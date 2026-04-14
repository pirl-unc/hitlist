# Split binding data out of observations index

## Goal

`observations.parquet` currently holds only MS-eluted rows (binding rows are dropped in `builder.py:197`). Users have no way to query the binding subset. This change writes a second index, `binding.parquet`, and exposes it through a parallel API + CLI so consumers cannot accidentally mix the two.

## Non-goals

- No schema change to the MS observations table. Downstream consumers keep working unchanged.
- No new binding-specific columns (affinity/IC50 are not in IEDB's flat CSVs as separate fields — the `qualitative_measurement` tier is the only binding signal).
- No sample-metadata join on the binding side. "Sample" metadata (instrument, acquisition mode, tissue) is MS-specific and does not apply to binding assays.

## Plan

### 1. `builder.py` — write two parquets instead of one

- After scanning IEDB+CEDAR and classifying `is_binding_assay`, partition:
  - `is_binding_assay=False` → existing `observations.parquet`
  - `is_binding_assay=True` → new `binding.parquet`
- Apply supplementary data only to the MS side (`supplement.py:138` hardcodes `is_binding_assay=False`).
- Extend `observations_meta.json` with `n_binding_rows`; invalidate cache when either output is missing.
- Gene annotation: `build_peptide_mappings()` should cover the union of peptides from both parquets. `annotate_observations_with_genes()` runs on both outputs.

### 2. `observations.py` — add binding path + loader

- `binding_path() -> Path` — parallel to `observations_path()`.
- `load_binding(...)` — same filter signature as `load_observations` (mhc_class, mhc_species, source, peptide, mhc_allele, gene filters, serotype). Drop MS-only filters (`is_monoallelic` doesn't apply; binding has no cell-line-derived mono-allelic concept).
- Share the serotype post-filter helper.

### 3. `export.py` — add `generate_binding_table`

- Filters: class, species, source, gene/gene-name/gene-id, mhc-allele, serotype, min-allele-resolution.
- Skip: sample-metadata join (no `generate_ms_samples_table` equivalent), `--mono-allelic`, `--instrument-type`, `--acquisition-mode`.
- Return the binding rows as-is with the same gene annotation columns MS observations get.

### 4. `cli.py` — new `hitlist export binding` subcommand

- Mirror the `p_obs` parser but drop MS-only args.
- Dispatch in `_export()` with a new `elif args.export_command == "binding"`.

### 5. Tests

- `test_builder`: scanning a mixed fixture produces two parquets with correct row partition; no overlap; both get gene annotations.
- `test_observations`: `load_binding()` returns binding rows only; `load_observations()` returns MS rows only.
- `test_export`: `generate_binding_table()` applies gene + serotype filters.
- `test_cli`: `hitlist export binding --class I` writes CSV / parquet.

### 6. Docs

- README section on "Binding vs MS" with the separation guarantee.
- Note the mono-allelic curation model (PMID-level, not sample-level) and the `is_monoallelic=True` + empty `mhc_restriction` edge case for supplementary rows — flag it so consumers know to post-filter if they need a strict "mono-allelic AND has allele" subset.

### 7. Version

- Bump to 1.7.0 (new index + new public API + new CLI surface).

## Verification

- `./format.sh`, `./lint.sh`, `./test.sh` all pass.
- Manual: build a fresh index and confirm `binding.parquet` row count equals the previously-dropped count reported as "Excluded N binding assay rows" on earlier builds.

## Mono-allelic audit findings (reported to user, not fixed here)

- Mono-allelic is a PMID-level flag, not a per-sample property.
- No `data/samples/` directory exists. `ms_samples` entries in `pmid_overrides.yaml` hold study metadata (instrument, acquisition), not sample-level MHC genotypes.
- A row can be `is_monoallelic=True` with empty `mhc_restriction` when (a) the PMID has a `mono_allelic_host` override and (b) the specific row is a supplementary contaminant without a curated allele.
- No code enforces `is_monoallelic=True ⇒ mhc_restriction != ""`. Documenting this in README so consumers know. A strict filter helper can be added in a follow-up if the user wants.
