# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A carefully curated and harmonized source of truth for MHC ligand mass spectrometry data, built for pMHC target selection and model training.

hitlist ingests immunopeptidome data from [IEDB](https://www.iedb.org/), [CEDAR](https://cedar.iedb.org/), and supplementary sources (PRIDE), normalizes it into a unified schema, and annotates every observation with expert-curated sample metadata — biological source context, perturbation conditions, MHC class, species, cell line identity, and disease state. The goal is a single, auditable dataset that downstream tools (binding predictors, antigen prioritization pipelines, cleavage models) can consume without re-curating the same papers.

**34 studies curated** across 7 species, with per-sample perturbation conditions, MHC class I + II, and allele-level HLA typing. All MHC alleles validated with mhcgnomes.

From local IEDB + CEDAR data:
- **1.88M unique human class I peptides** across 1,449 studies
- **790K unique human class II peptides** across 805 studies
- **24 species** with MHC ligand MS data
- **790 unique MHC allele strings**, 789/790 valid in mhcgnomes

## Install

```bash
pip install hitlist
```

## Quick start

```bash
# Register your IEDB/CEDAR downloads
hitlist data register iedb /path/to/mhc_ligand_full.csv
hitlist data register cedar /path/to/cedar-mhc-ligand-full.csv

# Build the search index (one-time, ~90s per file, cached as parquet)
hitlist data index

# Export curated sample metadata
hitlist export samples --class I -o class_i_samples.csv
hitlist export counts --source merged -o peptide_counts.csv
hitlist export summary

# Generate a data quality report
hitlist report --class I --output report.txt
```

## What hitlist curates

Every IEDB/CEDAR mass spec observation is classified by:

**Biological source context** (mutually exclusive):

| Category | Flag | Rule |
|---|---|---|
| **Cancer** | `src_cancer` | Tumor tissue, cancer patient biofluids, or non-EBV cell lines |
| **Adjacent to tumor** | `src_adjacent_to_tumor` | Surgically resected "normal" tissue (per-PMID override) |
| **Activated APC** | `src_activated_apc` | Monocyte-derived DCs/macrophages with pharmacological activation |
| **Healthy somatic** | `src_healthy_tissue` | Direct ex vivo, healthy donor, non-reproductive, non-thymic |
| **Healthy thymus** | `src_healthy_thymus` | Direct ex vivo thymus (expected for CTAs, AIRE-mediated) |
| **Healthy reproductive** | `src_healthy_reproductive` | Direct ex vivo testis, ovary, etc. (expected for CTAs) |
| **EBV-LCL** | `src_ebv_lcl` | EBV-transformed B-cell lines |
| **Cell line** | `src_cell_line` | Any cultured cell line |

**Per-sample metadata** (from YAML overrides):

| Field | Examples |
|---|---|
| MHC class | I, II, I+II, non-classical |
| Species | Human, mouse, cattle, dog, chicken, rhesus, pig, fish |
| Perturbation | CRISPR KO (13 genes), decitabine, IFN-gamma, viral infection, HLA-DM editing, SILAC cross-presentation, TIL expansion |
| HLA alleles | Per-donor or per-cell-line 4-digit typing |
| Cell line identity | 721.221, HAP1, THP-1, etc. with mono-allelic detection |

## Key design rules

- **All non-EBV cell lines are cancer-derived**, even when IEDB labels them "No immunization"
- **Cancer-specific** = found in cancer AND NOT found in healthy somatic tissue
- Thymus, reproductive tissue, adjacent tissue, EBV-LCLs, and activated APCs do NOT disqualify a peptide from being cancer-specific
- **Perturbation conditions are tracked per sample** — gene KOs, drug treatments, cytokine stimulation, viral infection, and other modifications that alter the immunopeptidome
- **MHC class I and II are tracked separately** — filter with `--class I` or `--class II`

## PMID curation overrides

Expert per-study overrides in `hitlist/data/pmid_overrides.yaml`:

```yaml
- pmid: 29557506
  label: "Neidert 2018 — Tübingen/Zurich biobank"
  override: healthy
  rules:
    - condition:
        Source Tissue: [Blood, Bone Marrow, Cerebellum]
      override: healthy
      reason: "Blood bank donors and autopsy CNS material"
    - condition:
        Source Tissue: [Colon, Kidney, Liver, Lung]
      override: adjacent
      reason: "Visceral Surgery — likely cancer resection margins"
  ms_samples:
    - type: "blood (buffy coat)"
      condition: "unperturbed"
      mhc_class: "I"
      classification: healthy
```

See [docs/pmid-curation.md](docs/pmid-curation.md) for the full list of 34 curated studies, perturbation categories, and export commands.

## Proteome mapping

Map peptides to source proteins with flanking context:

```python
from hitlist.proteome import ProteomeIndex

# Human + viral proteomes, 10aa flanking
idx = ProteomeIndex.from_ensembl_plus_fastas(
    release=112,
    fasta_paths=["hpv16.fasta", "ebv.fasta", "influenza_a.fasta"],
)
df = idx.map_peptides(["SLLMWITQC", "GILGFVFTL"], flank=10)
```

## Data management and indexing

```bash
hitlist data available            # show all known datasets
hitlist data fetch hpv16          # auto-download viral proteome
hitlist data register iedb /path  # register manual download
hitlist data list                 # show datasets + index cache status
hitlist data index                # build/rebuild parquet index
hitlist data index --force        # force re-index
hitlist data info iedb            # detailed metadata
```

The index is cached as parquet in `~/.hitlist/index/` and reused when the source CSV hasn't changed. First index: ~90s for 7.7 GB. Subsequent reads: <1s.

## Export commands

```bash
hitlist export samples                     # per-sample conditions table
hitlist export samples --class I           # MHC class I only
hitlist export counts --source merged      # real peptide counts from IEDB+CEDAR
hitlist export counts --source all         # IEDB vs CEDAR side-by-side
hitlist export summary                     # species x class summary
hitlist export alleles                     # validate YAML alleles with mhcgnomes
hitlist export data-alleles                # validate all IEDB/CEDAR alleles
```

## Python API

```python
from hitlist.scanner import scan
from hitlist.curation import classify_ms_row, is_cancer_specific
from hitlist.aggregate import aggregate_per_peptide
from hitlist.indexer import get_index
from hitlist.export import generate_ms_samples_table, count_peptides_by_study

# Scan for specific peptides with source classification
hits = scan(peptides={"SLYNTVATL"}, iedb_path="mhc_ligand_full.csv", mhc_class="I")

# Per-peptide summary
summary = aggregate_per_peptide(hits)

# Cached index for fast counts (parquet)
study_df, allele_df = get_index("merged")

# Curated sample metadata from YAML
samples = generate_ms_samples_table(mhc_class="I")
```

## Development

```bash
./develop.sh    # install in dev mode
./format.sh     # ruff format
./lint.sh       # ruff check + format check
./test.sh       # pytest with coverage
./deploy.sh     # lint + test + build + upload to PyPI
```
