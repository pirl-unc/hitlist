# How curation works

hitlist turns raw IEDB/CEDAR ligand exports into two clean, ML-ready parquet
indexes. The value isn't the download — it's the curation layer in between: a
disciplined, per-study process that corrects mislabels, fills missing
provenance, and classifies every observation by biological source. This page
describes that pipeline end-to-end and the principles behind it.

## The pipeline

```
IEDB / CEDAR CSV ─┐
                  ├─► scan ─► provenance fill ─► classify ─► partition ─► observations.parquet  (MS-eluted)
paper supplements ┘                                              └──────► binding.parquet       (in-vitro binding)
                                                                          peptide_mappings.parquet (sidecar)
```

### 1. Scan (`scanner.scan`)

Raw IEDB/CEDAR CSVs have a two-row header and ~100 columns. The scanner resolves
the columns it needs by name (with hardcoded fallbacks), streams the file, and
for each row:

- **deduplicates** by IEDB assay IRI,
- **resolves MHC identity** once through
  `resolve_mhc_annotation(restriction, reported_class, species_context)`: molecule names
  are normalized, class is derived from actual alleles/genes/pairs when possible, and
  ambiguous species names use explicit per-study curation as parser context,
- **retains provenance** in `mhc_class_reported`, `mhc_class_source`,
  `mhc_class_corrected`, `mhc_species_source`, and
  `mhc_species_context_disagrees`; an explicit cross-species MHC is never erased merely
  because the study context differs, and
- **extracts post-translational modifications** into a separate column so the
  bare peptide sequence stays clean.

### 2. Provenance fill (per-PMID)

IEDB rows often carry `Other`, `unidentified`, `unknown`, or simply blank values
for source organism, tissue, disease, cell name, and culture condition. Where a
field is one of these **unresolved sentinels**, the scanner fills it from the
study's entry in [`pmid_overrides.yaml`](pmid-curation.md):

- `pmid_source_organism(pmid)` fills `source_organism` / `species`,
- `pmid_provenance(pmid)` fills `source_tissue`, `cell_name`, `disease`,
  `culture_condition`.

Fills fire **only on unresolved values** — curated data never overwrites real
IEDB annotation. The fill happens *before* classification, so a curated cell
line is classified correctly. This is the heart of why hitlist is more usable
than the raw export: the `Other`/`Unknown` long tail becomes real metadata.

### 3. Classify (`classify_ms_row`)

Each MS row is assigned exactly one biological-source category (`src_cancer`,
`src_healthy_tissue`, `src_ebv_lcl`, …) plus mono-allelic, allele-resolution,
and serotype flags. The full taxonomy and its rules are documented in
[Source classification](source-classification.md). Classification combines the
(now-filled) structured fields with any per-study override.

### 4. Partition: MS vs in-vitro binding

Every scanned row is tested by `is_binding_assay`. Qualitative binding tiers
(`Positive-High/Intermediate/Low`, `Negative`) and `Positive` rows whose assay
comments name a binding method (microarray, refolding, MEDi, competitive
IC50/β2m assays, …) are routed to **`binding.parquet`**; genuine MS-elution rows
go to **`observations.parquet`**. The two are written as separate files and are
**never silently mixed** — a downstream model can't accidentally train MS
presentation on binding-affinity rows.

### 5. Build (`builder.build_observations`)

The builder scans both sources, partitions them, concatenates, deduplicates by
assay IRI, folds in supplementary data (below), drops biologically implausible
rows (e.g. very short class-II peptides), and writes the two parquets atomically.
Before writing, the same token audit used by `hitlist qc mhc-tokens` checks MS,
binding, and curated sample metadata. Reviewed source defects and parser gaps are
reported separately; any new unrecognized token fails the build instead of being
silently treated as data.
Gene/protein annotations are **not** copied onto every row; they live in the
`peptide_mappings.parquet` sidecar and are joined on demand at query time.
Ensembl-backed mappings cover conventional protein-coding genes and the eight
coding IG/TR germline biotypes, preserve the source `gene_biotype`, and exclude
pseudogenes. This is germline attribution only: donor-specific recombined
receptors and peptides spanning V(D)J junctions are not represented by Ensembl.

## Supplementary ingestion

Some papers' peptides never reach IEDB. Those are ingested directly from their
PRIDE / jPOSTrepo supplementary tables (`supplement.scan_supplementary`). Each is
registered in `supplementary.yaml` with the IEDB-equivalent context it needs:

- the CSV must provide `peptide` and `mhc_class` (and optionally a per-peptide
  `mhc_restriction`),
- a `defaults:` block supplies `source_organism`, `species`, `disease`,
  `culture_condition`, `source_tissue`, `cell_name`, `host`.

`source_organism` / `species` are **curated per paper, never assumed** — a blank
surfaces loudly (the row is dropped by the source-species filter) rather than
silently defaulting to human. Supplementary rows are MS-only, share the scanner's
exact schema, and the builder drops any that duplicate an IEDB/CEDAR row.

## The reference registries

Classification draws on a few curated data files, all YAML, all editable without
touching Python:

| Registry | What it holds | Drives |
|---|---|---|
| [`pmid_overrides.yaml`](pmid-curation.md) | Per-study corrections, per-sample metadata, allele typings | Provenance fills, overrides, mono-allelic, attribution |
| `cell_lines.yaml` | Cellosaurus-enriched cell-line catalog (canonical name, lineage, `cell_type`, cancer/EBV-LCL status) | Cell-name normalization, `cell_type` |
| `monoallelic_lines.yaml` | HLA-null/low hosts (721.221, C1R, K562, …) with endogenous alleles + EBV-LCL flag | Mono-allelic detection, EBV-LCL auto-correction |
| `tissue_categories.yaml` | Reproductive / thymus / activated-APC tissue + cell-name sets | Healthy-subtype routing, APC classification |
| Species + viral proteome registry | Ensembl/UniProt reference proteomes per species; viral UPIDs | Peptide → source-protein mapping, flanking |

## Design principles

The pipeline encodes a handful of deliberate ideas. Understanding them explains
most of the curation decisions:

- **Never conflate MS and binding.** MS-elution evidence (a peptide *was*
  presented) and in-vitro binding (a peptide *can* bind) are different claims.
  They live in separate parquets by construction.
- **IEDB sentinels are fillable, not authoritative.** `Other` / `Unknown` /
  `unidentified` / blank are treated as missing data to be curated — but real
  IEDB values are never overwritten.
- **Trust mhcgnomes and Cellosaurus over free text and web snippets.** Allele
  and species identity come from mhcgnomes; cell-line identity and lineage come
  from the Cellosaurus-verified registry, which **wins over** IEDB's coarse
  cell-name suffix.
- **Per-study provenance is explicit, not heuristic.** Every fill and override
  is keyed to a reviewed `pmid_overrides.yaml` entry. Studies that genuinely
  have no source proteome (e.g. random-library refolding) are deliberately left
  blank rather than mislabeled.
- **Be conservative when unconfirmable.** An unknown cell-name synonym resolves
  to "unknown" rather than a guess; chimeric/engineered/xenograft default to
  `False` when the host axis is missing; mono-allelic is asserted only when the
  row's allele is actually resolved.
- **Model species on three independent axes.** `mhc_species` (from the allele),
  `source_species` (the proteome the peptide came from), and `host_organism`
  (where the cells lived at sampling) are tracked separately. A SARS-CoV-2
  epitope presented on human MHC in a human cell line differs on each axis — and
  filters can target any one of them. See
  [Source classification](source-classification.md#species-axes).
- **Keep sample-level and row-level claims distinct.** Mono-allelic status and
  per-donor allele attribution are sample-level facts; class-only rows are split
  per donor and tagged with explicit allele provenance
  (`exact` / `sample_allele_match` / `pmid_class_pool` / `unmatched`).
