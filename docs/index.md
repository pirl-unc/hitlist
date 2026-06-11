# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A curated, harmonized, **ML-training-ready** MHC ligand mass-spectrometry dataset,
built from [IEDB](https://www.iedb.org/), [CEDAR](https://cedar.iedb.org/), and
paper supplementary tables.

hitlist scans the raw exports, **fills missing provenance** and **corrects
mislabels** from expert per-study curation, **classifies every observation by
biological source** (cancer tissue, healthy tissue, cell line, tumor-adjacent,
EBV-LCL, …), partitions MS-elution evidence from in-vitro binding into two
separate parquet files, maps peptides to source proteins with flanking context,
and ships everything as parquet plus a pandas-friendly Python API. The curation
overrides and classification rules are YAML data files, not hardcoded Python.

New here? Start with **[How curation works](curation-process.md)** for the
end-to-end pipeline and the ideas behind it.

## Install

```bash
pip install hitlist
```

## Quick start

```bash
# Register your IEDB/CEDAR downloads, then build the indexes
hitlist data register iedb /path/to/mhc_ligand_full.csv
hitlist data register cedar /path/to/cedar-mhc-ligand-full.csv
hitlist data build                 # writes observations.parquet + binding.parquet (+ sidecars)

# Data-quality report
hitlist report --class I --output report.txt
```

```python
from hitlist.scanner import scan
from hitlist.aggregate import aggregate_per_peptide

# Scan for specific peptides...
hits = scan(peptides={"SLYNTVATL", "GILGFVFTL"}, iedb_path="mhc_ligand_full.csv", mhc_class="I")

# ...or profile the entire dataset, then summarize per peptide with the
# is_cancer_specific flag.
full = scan(peptides=None, iedb_path="mhc_ligand_full.csv")
summary = aggregate_per_peptide(hits)
```

## The curation layer

The value of hitlist is the curation between download and parquet:

- **[How curation works](curation-process.md)** — the scan → fill → classify →
  partition → build pipeline, supplementary ingestion, and the design principles.
- **[Source classification](source-classification.md)** — the biological-source
  taxonomy (`src_cancer`, `src_healthy_tissue`, `src_ebv_lcl`, …), mono-allelic
  evidence, cross-species systems, and the cancer-specific definition.
- **[PMID curation overrides](pmid-curation.md)** — the `pmid_overrides.yaml`
  schema and how to add a study.

Two quick illustrations of why curation matters:

```yaml
# Reclassify a multi-arm study by IEDB field, in pmid_overrides.yaml.
# IEDB lumps every row under one Disease/Process Type; the only per-row
# signal is free-text Assay Comments, so the rules match on substring.
- pmid: 29789417
  study_label: "Loffler 2018 — CRC + matched normal colon"
  rules:
    - condition: { Assay Comments: "eluted from colorectal carcinoma (CRC) tissue." }
      override: cancer_patient
    - condition: { Assay Comments: "eluted from nonmalignant colon (NMC) tissue." }
      override: adjacent
```

All non-EBV cell lines are classified as cancer-derived even when IEDB labels
them "No immunization" (catching HeLa, THP-1, A549, …), while EBV-LCLs on
HLA-null hosts (721.221, C1R) are auto-corrected *out* of the cancer set. See
[Source classification](source-classification.md).

## Proteome mapping

Map peptides to source proteins with flanking context:

```python
from hitlist.proteome import ProteomeIndex

# Human proteome from pyensembl, or human + viral in one index
idx = ProteomeIndex.from_ensembl(release=112)
idx = ProteomeIndex.from_ensembl_plus_fastas(fasta_paths=["hpv16.fasta", "ebv.fasta"])

df = idx.map_peptides(["SLLMWITQC"], flank=5)
# → protein_id, gene_name, gene_id, position, n_flank, c_flank, ...
```

## Per-sample peptidome context

The full peptidome of each sample is essential context for judging whether a
peptide's presence is meaningful:

```python
from hitlist.scanner import scan
from hitlist.samples import sample_peptidomes, overlay_targets

full = scan(peptides=None, iedb_path="mhc_ligand_full.csv", mhc_class="I")
samples = sample_peptidomes(full)

# Overlay a target set for context fractions:
# "1 CTA out of 762 peptides = 0.13% = stochastic noise"
context = overlay_targets(full, target_peptides=my_cta_set, label="cta")
```

## Data management

```bash
hitlist data available          # list known datasets (IEDB/CEDAR + fetchable viral proteomes)
hitlist data fetch hpv16        # auto-download a viral proteome from UniProt
hitlist data register iedb /path/to/file  # register a manual download
hitlist data list               # registered datasets with size/date
hitlist data info iedb          # detailed JSON metadata
hitlist data path iedb          # resolve to file path
```

Storage: `~/.hitlist/` (override with the `HITLIST_DATA_DIR` env var).

## Development

```bash
./develop.sh    # install in dev mode
./format.sh     # ruff format
./lint.sh       # ruff check + format check
./test.sh       # pytest with coverage
./deploy.sh     # lint + test + build + upload to PyPI
```
