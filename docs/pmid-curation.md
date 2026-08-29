# PMID curation overrides

hitlist applies expert per-study overrides to correct IEDB/CEDAR annotations that
don't reflect the true biological context of a sample, and to attach the
per-sample metadata (HLA genotype, perturbation, instrument) that MS pipelines
need. Overrides live in `hitlist/data/pmid_overrides.yaml` as data, loaded at
runtime by `curation.load_pmid_overrides()` — no code changes to add a study.

Each entry is keyed by `pmid` (or `submission_id` for unpublished IEDB
submissions) and curated against the paper's Methods. The file currently covers
**159 PMIDs**, accounting for ~89.5% of all observations.

## Entry schema

A representative entry:

```yaml
- pmid: 31495665
  study_label: "Author 2019 — short description"
  title: "Exact PubMed title"
  override: cell_line          # study-wide default classification (optional)
  note: "Why curation is needed — cite the Methods."
  source_organism: "Homo sapiens"   # provenance fills (only used where IEDB is blank)
  donors: 4
  hla_alleles:                 # alleles profiled in the study (allele pool)
    - "HLA-A*02:01"
    - "HLA-B*07:02"
  perturbations:
    - "HLA-DM editing (dm+/dm-)"
  ms_samples:
    - sample_label: "721.221-B*51:01 (WT)"
      n_samples: 3
      mhc: "HLA-B*51:01"
      mhc_class: "I"
      condition: "unperturbed"
    - sample_label: "721.221-B*51:01 ERAP1 KO"
      n_samples: 3
      mhc: "HLA-B*51:01"
      mhc_class: "I"
      condition: "ERAP1 CRISPR/Cas9 knockout"
  rules:
    - condition: { Source Tissue: "Blood" }
      override: healthy
      reason: "Blood-bank donors"
```

### Top-level keys

| Key | Purpose |
|---|---|
| `pmid` / `submission_id` | The entry key (int PMID, or string submission id). |
| `study_label`, `title`, `note` | Human-facing provenance. (`label:` is the **deprecated** name for `study_label`.) |
| `override` | Study-wide default classification — see vocabulary below. |
| `rules` | Conditional, per-row overrides (checked before `override`). |
| `source_organism`, `species`, `source_tissue`, `cell_name`, `disease`, `culture_condition` | **Provenance fills** — used only where the IEDB row is blank/`Other`/`unknown`; never overwrite real IEDB data. |
| `hla_alleles` | Alleles profiled in the study; the fallback pool for class-only allele expansion. |
| `mono_allelic_host` | HLA-null/low host name (must exist in `monoallelic_lines.yaml`); flags resolved-allele rows mono-allelic. |
| `mono_allelic_method` | Tagged-pulldown mono-allelic method (e.g. MAPTAC) — not a cell line. |
| `ms_samples` | Per-sample-type metadata (below). |
| `peptide_attributions` | Path to a CSV mapping `peptide` → `sample_label` for per-donor attribution. |
| `exclude_from_ms` | Exclude this study/sample from the MS index. |
| `donors`, `samples`, `tissues` | Counts. |
| `ip_antibody`, `acquisition_mode`, `instrument`, `fragmentation`, `labeling`, `search_engine`, `fdr`, `quantification_method` | MS-acquisition metadata (study-wide defaults; overridable per `ms_samples` entry). |
| `perturbations` | Non-standard processing (gene KO, cytokines, infection, …). |

### `override` vocabulary

| Override | Effect |
|---|---|
| `cell_line` | Force cell-line → cancer-derived (unless EBV-LCL). |
| `ebv_lcl` | Force EBV-LCL; not cancer. |
| `noncancer_cell_line` | Force cell line **without** `src_cancer` (rare non-malignant lines). |
| `cancer_patient` | Reclassify all rows as cancer-derived. |
| `adjacent` | Reclassify as tumor-adjacent normal tissue. |
| `healthy` | Confirmed healthy tissue (force the healthy path). |
| `activated_apc` | Reclassify as activated-APC artifact. |

### `ms_samples` fields

| Field | Meaning |
|---|---|
| `sample_label` | Sample description. (`type:` is the **deprecated** name.) |
| `n_samples` | Number of samples/replicates. (Use the `_samples` suffix — never a bare `n`.) |
| `mhc` | Donor genotype (`HLA-A*…` or a space-joined allele list). |
| `mhc_class` | `"I"`, `"II"`, `"I+II"`, or `"non-classical"`. Use `non-classical` for class Ib / MHC-Ib molecules — HLA-E, HLA-F, HLA-G, MR1, CD1, H2-Q — so `--class I` does not return them. A declared class that contradicts the sample's own alleles fails CI. |
| `condition` | Perturbation or `"unperturbed"`. |
| `classification`, `override`, `reason` | Per-sample classification override + rationale. |
| `source`, `species`, `reference_proteomes` | Per-sample provenance. |

## The `rules` mechanism

Many studies mix sample sources under one PMID (e.g. tumor tissue and adjacent
normal in the same submission). The `rules` list applies conditional overrides
**before** the study-wide `override`:

```yaml
rules:
  - condition:
      Source Tissue: [Blood, Bone Marrow, Cerebellum]
    override: healthy
    reason: "Blood-bank donors and autopsy CNS material"
  - condition:
      Source Tissue: [Colon, Kidney, Liver]
    override: adjacent
    reason: "Visceral Surgery / Urology — likely cancer resection margins"
```

Matching semantics (`_matches_condition`):

- **all** keys in a `condition` must match (AND); the first matching rule wins.
- A value may be a single string or a list (any-match).
- `Source Tissue`, `Cell Name`, `Culture Condition`, `Disease`, `Process Type`
  match by equality; `Assay Comments` matches by **case-insensitive substring**
  (IEDB concatenates per-arm provenance into one cell, e.g.
  `"eluted from CRC tissue. eluted from nonmalignant colon tissue."`), which lets
  one rule target a single arm of a multi-arm study.

If no rule matches, the study-wide `override` applies; if there's no override
either, the row falls through to structured-field classification.

## Per-donor attribution

The IEDB **Antigen Processing Comments** field sometimes carries per-sample
identifiers ("buffy coat 25", "colon 32"). For studies with a
`peptide_attributions` CSV, hitlist splits each class-only row into one row per
matched donor, each tagged with that donor's HLA typing and an explicit
`mhc_allele_provenance` (`exact` / `peptide_attribution` / `sample_allele_match`
/ `pmid_class_pool` / `unmatched`). This narrows a peptide's candidate alleles
from a disease-wide union down to the specific donors it was actually found in.

## Adding a new override

1. Read the paper's Methods — confirm tissue, disease, cell lines, HLA typing,
   and any perturbation. Don't trust the IEDB free-text fields blindly.
2. Add an entry to `pmid_overrides.yaml` with `study_label`, `title`, the
   appropriate `override`/`rules`, provenance fills for anything IEDB left blank,
   and `ms_samples` for per-sample structure.
3. If the study uses an HLA-null host or MAPTAC, set `mono_allelic_host` /
   `mono_allelic_method` (and add the host to `monoallelic_lines.yaml` if new).
4. Validate: `hitlist export alleles` parses every allele through mhcgnomes;
   `hitlist qc` flags normalization and cross-reference issues.

No code changes are needed — the YAML is loaded at runtime.

## Exporting curated metadata

```bash
hitlist export samples              # every ms_samples entry as CSV
hitlist export samples --class I    # MHC class I only
hitlist export summary              # species × class totals
hitlist export alleles              # validate alleles with mhcgnomes
```
