# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

Curated mass spectrometry evidence for MHC ligand data from [IEDB](https://www.iedb.org/) and [CEDAR](https://cedar.iedb.org/).

hitlist scans IEDB and CEDAR MHC ligand exports, classifies each observation by biological source context (cancer tissue, healthy tissue, cell line, tumor-adjacent, etc.), maps peptides to source proteins with flanking sequences, and produces data quality reports. PMID-level curation overrides and tissue classifications are stored as YAML data files, not hardcoded Python.

## Install

```bash
pip install hitlist
```

## Quick start

```bash
# Register your IEDB/CEDAR downloads
hitlist data register iedb /path/to/mhc_ligand_full.csv
hitlist data register cedar /path/to/cedar-mhc-ligand-full.csv

# Generate a data quality report
hitlist report
hitlist report --class I --output report.txt
```

```python
from hitlist.scanner import scan
from hitlist.curation import classify_ms_row, is_cancer_specific
from hitlist.aggregate import aggregate_per_peptide

# Scan for specific peptides
hits = scan(
    peptides={"SLYNTVATL", "GILGFVFTL"},
    iedb_path="mhc_ligand_full.csv",
    mhc_class="I",
)

# Or profile the entire dataset
full = scan(peptides=None, iedb_path="mhc_ligand_full.csv")

# Per-peptide summary with cancer-specific classification
summary = aggregate_per_peptide(hits)
```

## Source classification

Every IEDB/CEDAR mass spec observation is classified into one of these categories:

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

**Key rule**: all non-EBV cell lines are classified as cancer-derived, even when IEDB labels them "No immunization". This catches HeLa, THP-1, A549, and other cancer lines used in non-cancer studies.

**Cancer-specific** = found in cancer AND NOT found in healthy somatic tissue. Thymus, reproductive tissue, adjacent tissue, EBV-LCLs, and activated APCs do NOT disqualify.

## PMID curation overrides

Expert per-study overrides are stored in `hitlist/data/pmid_overrides.yaml`:

```yaml
- pmid: 29557506
  label: "Neidert 2018 — Tübingen/Zurich biobank"
  override: healthy
  tissue_overrides:
    Blood: healthy           # blood bank donors
    Bone Marrow: healthy     # hip arthroplasty
    Colon: adjacent          # Visceral Surgery dept → likely CRC margin
    Kidney: adjacent         # Urology → likely nephrectomy
    Liver: adjacent          # likely HCC/met adjacent
```

Tissue-level overrides take priority over study-level overrides.

## Proteome mapping

Map peptides to source proteins with flanking context:

```python
from hitlist.proteome import ProteomeIndex

# Human proteome (from pyensembl)
idx = ProteomeIndex.from_ensembl(release=112)

# Or combined human + viral
idx = ProteomeIndex.from_ensembl_plus_fastas(
    fasta_paths=["hpv16.fasta", "ebv.fasta"],
)

# Map peptides with 5-residue flanks
df = idx.map_peptides(["SLLMWITQC"], flank=5)
# → protein_id, gene_name, gene_id, position, n_flank, c_flank, n_sources, unique_n_flank, unique_c_flank
```

## Per-sample peptidome context

The full peptidome context for each sample is critical for interpreting whether a peptide's presence is meaningful:

```python
from hitlist.scanner import scan
from hitlist.samples import sample_peptidomes, overlay_targets

# Full scan (ALL peptides, not just targets)
full = scan(peptides=None, iedb_path="mhc_ligand_full.csv", mhc_class="I")

# Per-sample stats
samples = sample_peptidomes(full)

# Overlay CTA peptides for context fractions
# "1 CTA out of 762 peptides = 0.13% = stochastic noise"
context = overlay_targets(full, target_peptides=my_cta_set, label="cta")
```

## Data management

```bash
hitlist data available          # show all 14 known datasets
hitlist data fetch hpv16        # auto-download viral proteome from UniProt
hitlist data register iedb /path/to/file  # register manual download
hitlist data list               # show registered datasets with size/date
hitlist data info iedb          # detailed JSON metadata
hitlist data path iedb          # resolve to file path
hitlist data refresh hpv16      # re-download
hitlist data remove iedb        # unregister
```

Storage: `~/.hitlist/` (override with `HITLIST_DATA_DIR` env var).

## Development

```bash
./develop.sh    # install in dev mode
./format.sh     # ruff format
./lint.sh       # ruff check + format check
./test.sh       # pytest with coverage
./deploy.sh     # lint + test + build + upload to PyPI
```
