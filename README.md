# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A curated, harmonized, **ML-training-ready** MHC ligand mass-spectrometry dataset.

hitlist ingests immunopeptidome data from [IEDB](https://www.iedb.org/), [CEDAR](https://cedar.iedb.org/), and paper supplementary tables (PRIDE/jPOSTrepo); filters out binding-assay data; joins every observation to expert-curated sample metadata (HLA genotype, tissue, disease, perturbation, instrument); and ships it as a single parquet file + pandas-friendly Python API.

**As of v1.2.0** (human MHC-I, MS-eluted only):

| | |
|---|---|
| Unique peptides | **748,386** |
| Observations | **2,672,046** |
| Mono-allelic (exact allele) | 579,096 obs / 300K peptides / 119 alleles |
| Multi-allelic with allele match | 450K obs |
| Multi-allelic with class-pool (`N of M alleles`) | 784K obs |
| Curated PMIDs | **155** (89.5% of all observations) |
| Allele-resolved `sample_mhc` coverage | **74.8%** |

Across all species and classes: 4.05M observations, 1.29M peptides, 21 species.

## Install

```bash
pip install hitlist
```

## Quick start for ML training

```bash
# One-time: register IEDB + CEDAR downloads and build the observations table
hitlist data register iedb /path/to/mhc_ligand_full.csv
hitlist data register cedar /path/to/cedar-mhc-ligand-full.csv
hitlist data build                           # ~3 min, writes ~/.hitlist/observations.parquet

# Export training-ready CSVs
hitlist export observations --class I --species "Homo sapiens" --mono-allelic \
    --min-allele-resolution four_digit -o mono_allelic_classI.csv

hitlist export observations --class II --species "Homo sapiens" \
    -o multi_allelic_classII.csv
```

Binding-assay data (peptide microarrays, refolding, MEDi display) is **excluded by default** — the observations table contains only MS-eluted immunopeptidome data.

## Python API

```python
from hitlist.export import generate_observations_table

# Mono-allelic human class I: 579K observations with ground-truth allele
mono = generate_observations_table(
    mhc_class="I",
    species="Homo sapiens",
    is_mono_allelic=True,
    min_allele_resolution="four_digit",
)

# Multi-allelic with at least allele-pool info (74.8% of all rows)
multi = generate_observations_table(mhc_class="I", species="Homo sapiens")
multi_with_alleles = multi[multi["sample_mhc"].str.strip() != ""]
```

Species filters accept any variant — `"Homo sapiens"`, `"human"`, `"homo_sapiens"`, `"Homo sapiens (human)"` all work.

## Output schema

Each row of `generate_observations_table()` has (among others):

| Column | Meaning |
|---|---|
| `peptide` | Amino acid sequence |
| `mhc_restriction` | Allele from IEDB (may be `"HLA class I"` for multi-allelic studies) |
| `sample_mhc` | Allele(s) known for the sample the peptide came from — the **useful** field for training |
| `mhc_class` | `I`, `II`, or `non classical` |
| `mhc_species` | Canonical species (normalized via mhcgnomes) |
| `is_monoallelic` | True if sample has a single transfected allele (721.221, C1R, K562, MAPTAC…) |
| `has_peptide_level_allele` | True if `mhc_restriction` is a specific allele (not `"HLA class I"`) |
| `is_potential_contaminant` | True for MS-eluted peptides that failed NetMHCpan binding prediction (supplementary only) |
| `sample_match_type` | How `sample_mhc` was populated (see below) |
| `matched_sample_count` | Number of curated samples for this PMID |
| `src_cancer`, `src_healthy_tissue`, `src_ebv_lcl`, ... | Mutually-exclusive biological source categories |
| `source` | `iedb`, `cedar`, or `supplement` |
| `source_organism`, `reference_title`, `cell_name`, `source_tissue`, `disease` | IEDB sample context |
| `instrument`, `instrument_type`, `acquisition_mode`, `fragmentation`, `labeling`, `ip_antibody` | MS acquisition from ms_samples curation |

### `sample_match_type` — join provenance

| Value | Meaning | Training-grade? |
|---|---|---|
| `allele_match` | IEDB recorded a specific allele and it matched a curated sample genotype | **Yes** — high confidence |
| `single_sample_fallback` | IEDB class-only but study has exactly 1 sample, so `sample_mhc` = that sample's full genotype | Yes (for deconvolution) |
| `pmid_class_pool` | IEDB class-only and study has multiple samples — `sample_mhc` = union of all class-matching alleles across all samples | Yes (for deconvolution), lower precision |
| `unmatched` | No curated sample for this PMID, or all samples have `mhc: unknown` | No — `sample_mhc` empty |

## Curation layer

155 PMIDs curated in `hitlist/data/pmid_overrides.yaml` with per-sample HLA typing, tissue, perturbation, and instrument metadata. Supplementary data (PRIDE / paper tables) ingested via `hitlist/data/supplementary.yaml` — currently the full Gomez-Zepeda 2024 panel (JY, Raji, HeLa, SK-MEL-37, plasma).

Every observation is classified by mutually-exclusive biological source category:

| Category | Flag | Rule |
|---|---|---|
| Cancer | `src_cancer` | Tumor tissue, cancer patient biofluids, or non-EBV cell lines |
| Adjacent to tumor | `src_adjacent_to_tumor` | Surgically resected "normal" tissue (per-PMID override) |
| Activated APC | `src_activated_apc` | Monocyte-derived DCs/macrophages with pharmacological activation |
| Healthy somatic | `src_healthy_tissue` | Direct ex vivo, healthy donor, non-reproductive, non-thymic |
| Healthy thymus | `src_healthy_thymus` | Direct ex vivo thymus (expected for CTAs, AIRE-mediated) |
| Healthy reproductive | `src_healthy_reproductive` | Direct ex vivo testis, ovary (expected for CTAs) |
| EBV-LCL | `src_ebv_lcl` | EBV-transformed B-cell lines |
| Cell line | `src_cell_line` | Any cultured cell line |

**Cancer-specific** = `src_cancer AND NOT src_healthy_tissue`. Thymus, reproductive tissue, adjacent tissue, EBV-LCLs, and activated APCs do NOT disqualify a peptide from being cancer-specific.

## Flanking context (for cleavage models)

```python
from hitlist.proteome import ProteomeIndex

# Human proteome, 10aa flanks
idx = ProteomeIndex.from_ensembl(release=112)
flanking = idx.map_peptides(["SLLMWITQC", "GILGFVFTL"], flank=10)

# Or include viral/custom FASTAs
idx = ProteomeIndex.from_ensembl_plus_fastas(
    release=112,
    fasta_paths=["hpv16.fasta", "ebv.fasta", "influenza_a.fasta"],
)
```

Alternatively, build the observations table with flanking pre-computed:

```bash
hitlist data build --with-flanking --proteome-release 112
```

This adds `gene_name`, `gene_id`, `protein_id`, `position`, `n_flank`, `c_flank` columns. Current limitations: only Ensembl proteomes are auto-fetched; non-human and viral sources need user-supplied FASTAs (tracked in [#39](https://github.com/pirl-unc/hitlist/issues/39)).

## CLI reference

```bash
hitlist data register iedb /path/to/file.csv   # register source
hitlist data build [--force] [--with-flanking] # build observations.parquet
hitlist data list                              # inventory + index cache status
hitlist data available                         # known datasets
hitlist data fetch hpv16                       # auto-download viral proteome

hitlist export observations --class I --species human --mono-allelic \
    --min-allele-resolution four_digit -o train.csv
hitlist export observations -o all.parquet     # parquet output supported
hitlist export samples --class I               # per-sample conditions (YAML curation only)
hitlist export summary                         # species x class summary
hitlist export counts --source merged          # peptide counts per study
hitlist export alleles                         # validate YAML alleles with mhcgnomes
```

## Filter flags on `hitlist export observations`

| Flag | Values |
|---|---|
| `--class` | `I`, `II`, `non classical` |
| `--species` | Any species variant (normalized via mhcgnomes) |
| `--mono-allelic` / `--multi-allelic` | Filter on `is_monoallelic` |
| `--instrument-type` | `Orbitrap`, `timsTOF`, `TOF`, `QqQ`, ... |
| `--acquisition-mode` | `DDA`, `DIA`, `PRM` |
| `--min-allele-resolution` | `four_digit`, `two_digit`, `serological`, `class_only` |
| `--output` / `-o` | `.csv` or `.parquet` |

## Development

```bash
./develop.sh    # install in dev mode
./format.sh     # ruff format
./lint.sh       # ruff check + format check
./test.sh       # pytest with coverage
./deploy.sh     # lint + test + build + upload to PyPI
```

See [docs/pmid-curation.md](docs/pmid-curation.md) for the curation YAML format and per-study overrides.
