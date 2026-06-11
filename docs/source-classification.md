# Source classification

Every IEDB/CEDAR mass-spectrometry observation is classified by biological source
context, so a downstream consumer can tell cancer evidence from healthy-tissue
evidence — the core distinction for target safety. Classification runs in
`curation.classify_ms_row` during the [build](curation-process.md), and the
resulting `src_*` flags are baked into `observations.parquet`.

## Categories

Categories are mutually exclusive and resolved in priority order:

| Priority | Category | Flag | Rule | Safety impact |
|---|---|---|---|---|
| 1 | **Cancer** | `src_cancer` | Process Type = "Occurrence of cancer", OR a non-EBV cell line | Positive targeting evidence |
| 2 | **Adjacent to tumor** | `src_adjacent_to_tumor` | Surgically-resected "normal" tissue (per-PMID override) | Ambiguous (may contain cancer) |
| 3 | **Activated APC** | `src_activated_apc` | Monocyte-derived DC/macrophage with pharmacological activation | Pharmacological artifact |
| 4 | **Healthy somatic** | `src_healthy_tissue` | Direct ex vivo + healthy donor + non-reproductive + non-thymic | **Safety signal** (off-target risk) |
| 5 | **Healthy thymus** | `src_healthy_thymus` | Direct ex vivo thymus | Expected for CTAs (AIRE-mediated) |
| 6 | **Healthy reproductive** | `src_healthy_reproductive` | Direct ex vivo testis, ovary, placenta, … | Expected for CTAs |
| 7 | **EBV-LCL** | `src_ebv_lcl` | EBV-transformed B-cell line | Not cancer, not healthy |
| 8 | **Cell line** | `src_cell_line` | Any "Cell Line / Clone" culture condition | Treated as cancer-derived (unless EBV-LCL) |

Healthy-reproductive is further split into `src_healthy_reproductive_female` and
`src_healthy_reproductive_male` where the tissue is sex-specific. Note that
immune-privileged sites like **testis** are grouped as *reproductive* on the
basis of immune privilege, not sex.

## Key rules

### All non-EBV cell lines are cancer-derived

Commercial cancer lines (HeLa, THP-1, A549, HCT 116) frequently appear in IEDB
marked "No immunization" with disease "healthy". That label describes the
*experiment*, not the *biology* — the cells are still cancer-derived. hitlist
classifies **every non-EBV cell line as cancer**, regardless of IEDB Process
Type. The one escape hatch is the `noncancer_cell_line` override, for the rare
genuinely non-malignant immortalized line (e.g. an engineered HEK or a
non-transformed line), which forces `src_cell_line` without `src_cancer`.

### EBV-LCL auto-correction

EBV-transformed B lymphoblastoid lines (B-LCLs) are a workhorse of
immunopeptidomics but are **not** cancer. IEDB tags them inconsistently — often
as a cell line, sometimes mislabeling the HLA-null host (721.221, C1R) as "HeLa
cells". hitlist corrects this automatically: when a row is mono-allelic on a host
flagged `ebv_lcl: true` in `monoallelic_lines.yaml`, it is forced to
`src_ebv_lcl` and `src_cancer = False`. This re-labels hundreds of thousands of
observations that IEDB calls cancer. Genuinely malignant mono-allelic hosts
(K562 = CML) are *not* flagged and keep `src_cancer`.

### Healthy requires Direct Ex Vivo

Only tissue taken directly from a healthy donor (not cultured, not passaged, no
disease) qualifies as genuinely healthy. This is the strictest definition and
produces the negative set used for off-target toxicity assessment. A per-study
`healthy` override can force this path when IEDB's structured fields are
incomplete.

### Thymus and reproductive tissue are separate from somatic

Cancer-testis-antigen expression in thymus (AIRE-mediated central tolerance) and
reproductive tissue (normal biology, immune-privileged) is *expected*. Finding a
CTA peptide there does not make it unsafe to target, so these get their own flags
and do not count as the somatic safety signal.

## Mono-allelic evidence

A peptide eluted from a mono-allelic sample (a single HLA allele, via an HLA-null
host or a MAPTAC-style tagged pulldown) can be attributed to that one allele with
confidence — the most valuable training signal. hitlist flags it two ways:

- **cell-name match** (`detect_monoallelic`): the cell name matches an alias in
  `monoallelic_lines.yaml`, *unless* the reported allele is one of the host's own
  endogenous alleles.
- **per-study override** (`mono_allelic_host` / `mono_allelic_method`): fires
  only when the row's allele is actually resolved (four-digit, two-digit, or
  serological), which de-flags class-only validation rows. It deliberately does
  **not** gate on cell name, because IEDB mislabels hosts.

## Cross-species systems

Three independent flags, computed at query/export time from the relationship
between the source, host, and MHC species axes:

| Flag | Meaning |
|---|---|
| `is_chimeric_system` | Source and MHC are different MHC-bearing genera (e.g. human cells displaying mouse MHC). Viral/bacterial sources and IEDB sentinels fall through to `False` — that's normal infection biology, not engineering. |
| `is_engineered_mhc` | A chimeric system where the host genus matches the *source* (native cells, transgenic MHC) — distinguishing HLA-transgenic / transfectant systems from heterologous-antigen studies. |
| `is_xenograft` | Cells of one genus grown in a host of another (e.g. a human tumor in an NSG mouse) — source ≠ host, host ≠ MHC. |

## Non-peptide ligands

Some MHC-like molecules present lipids, metabolites, or stress ligands rather
than peptides — CD1a–e, MR1, MICA/MICB, ULBP/RAET1, NKG2x, HFE. Their "peptide"
column carries chemical names, not sequences. `is_non_peptide_ligand` flags these
(by restriction-name regex) so they can be excluded from peptide-presentation
training by default. H2-M3, a genuine N-formyl-peptide presenter, is deliberately
*not* flagged.

## Species axes

hitlist tracks three independent species axes rather than a single "species"
field:

- **`mhc_species`** — derived from the allele (the MHC molecule's species).
- **`source_species`** — the proteome the peptide was sequenced from (the curated
  `source_organism`; the legacy `species` column is coalesced in and deprecated).
- **`host_organism`** — where the cells physically lived at sampling.

The cross-species flags above are defined precisely by the relationships between
these axes, and exports can filter on any one independently
(`--source-species`, `--host-species`, `--mhc-species`). The legacy
`human_only=` scan parameter is deprecated in favor of `mhc_species=`.

## Cancer-specific definition

`is_cancer_specific` (used by `aggregate.aggregate_per_peptide`) marks a peptide
that is targetable on safety grounds:

```
is_cancer_specific = found_in_cancer AND NOT found_in_healthy_somatic_tissue
```

The following do **not** disqualify a peptide:

- presence on thymus,
- presence on reproductive tissue,
- presence on tumor-adjacent tissue,
- presence on EBV-LCLs,
- presence on activated APCs.

## IEDB columns used

| Our column | IEDB field | Purpose |
|---|---|---|
| `process_type` | Process Type | Cancer vs healthy |
| `disease` | Disease | Specific disease |
| `culture_condition` | Culture Condition | Ex vivo / cell line / EBV-LCL |
| `source_tissue` | Source Tissue | Anatomical origin |
| `cell_name` | Cell Name | Named cell line or cell type |
| `host` | Host | Host organism (cross-species axis) |
| `mhc_restriction` | MHC Restriction Name | HLA allele |
| `mhc_class` | MHC Allele Class | Class I or II |

Column positions are resolved dynamically from the CSV header (with hardcoded
fallbacks), so the pipeline survives IEDB column reordering.
