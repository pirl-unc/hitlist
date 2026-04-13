# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A curated, harmonized, **ML-training-ready** MHC ligand mass-spectrometry dataset.

hitlist ingests immunopeptidome data from [IEDB](https://www.iedb.org/), [CEDAR](https://cedar.iedb.org/), and paper supplementary tables (PRIDE/jPOSTrepo); filters out binding-assay data; joins every observation to expert-curated sample metadata (HLA genotype, tissue, disease, perturbation, instrument); and ships it as a single parquet file + pandas-friendly Python API.

## What's curated

| | Count |
|---|---|
| Curated PMIDs (`pmid_overrides.yaml`) | **155** — covers 89.5% of observations |
| `ms_samples` entries with per-sample metadata | 359 |
| `ms_samples` entries with 4-digit HLA typing | 237 |
| Supplementary CSVs ingested (PRIDE/jPOSTrepo) | 5 (JY, HeLa, SK-MEL-37, Raji, plasma — Gomez-Zepeda 2024) |
| Species reference proteomes (registry) | 19 (Ensembl: 4, UniProt: 15) |
| Viral reference proteomes (registry) | 30 distinct viruses, 54 name aliases |

## What's in the observations table

After `hitlist data build` (v1.4.4):

| | |
|---|---|
| **Total observations** (MS-eluted, all species) | **4,053,693** |
| **Unique peptides** | **1,285,987** |
| Unique MHC alleles | 691 |
| MHC species covered | 21 |
| IEDB rows | 3,986,991 |
| CEDAR rows | 595 |
| Supplementary rows | 66,107 |

### Human MHC-I breakdown

| | |
|---|---|
| Observations | 2,672,046 |
| Unique peptides | 748,386 |
| Mono-allelic (exact allele) | 579,096 obs / 300K peptides / 119 alleles |
| Multi-allelic with allele match | 450,399 obs |
| Multi-allelic with class-pool (`N of M alleles`) | 784,370 obs |
| Allele-resolved `sample_mhc` coverage | **74.8%** |

## Install

```bash
pip install hitlist
```

## Quick start for ML training

```bash
# One-time: register IEDB + CEDAR downloads and build
hitlist data register iedb /path/to/mhc_ligand_full.csv
hitlist data register cedar /path/to/cedar-mhc-ligand-full.csv
hitlist data build                           # ~90s, writes ~/.hitlist/observations.parquet

# Export training-ready CSVs
hitlist export observations --class I --species "Homo sapiens" --mono-allelic \
    --min-allele-resolution four_digit -o mono_allelic_classI.csv

hitlist export observations --class II --species "Homo sapiens" \
    -o multi_allelic_classII.csv
```

Binding-assay data (peptide microarrays, refolding, MEDi display) is **excluded at build time** — the observations table contains only MS-eluted immunopeptidome data.

## Python API

### Training-data export

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

### Raw observations loading

```python
from hitlist.observations import load_observations, is_built, observations_path

df = load_observations()                       # everything (MS-eluted only)
df = load_observations(mhc_class="I")          # class I only
df = load_observations(species="Homo sapiens") # human only
df = load_observations(source="iedb")          # filter by source
df = load_observations(columns=["peptide", "mhc_restriction", "src_cancer"])
```

### Building / curation

```python
from hitlist.builder import build_observations
from hitlist.curation import (
    classify_ms_row,
    normalize_species,
    normalize_allele,
    load_pmid_overrides,
)
from hitlist.supplement import scan_supplementary, load_supplementary_manifest

build_observations(with_flanking=True, use_uniprot_search=True, force=False)
normalize_species("human")           # → "Homo sapiens"
normalize_allele("H-2Kb")            # → "H2-K*b"
scan_supplementary()                 # DataFrame of curated paper-supplement peptides
```

### Flanking context (cleavage models)

```python
from hitlist.proteome import ProteomeIndex

# Human Ensembl (release 112), 10aa flanks, O(1) per-peptide lookup
idx = ProteomeIndex.from_ensembl(release=112, species="human")
flanking = idx.map_peptides(["SLLMWITQC", "GILGFVFTL"], flank=10)

# Mix human + custom FASTAs (e.g. viral)
idx = ProteomeIndex.from_ensembl_plus_fastas(
    release=112, fasta_paths=["hpv16.fasta", "ebv.fasta"]
)

# Non-Ensembl species from a FASTA
idx = ProteomeIndex.from_fasta("~/.hitlist/proteomes/sarcophilus_harrisii.fasta")
```

### Proteome registry / UniProt resolution

```python
from hitlist.downloads import (
    lookup_proteome,           # org string → registry entry (dict)
    fetch_species_proteome,    # download FASTA and cache to ~/.hitlist/proteomes/
    resolve_proteome_via_uniprot,  # direct UniProt REST lookup
    list_proteomes,            # manifest section
)

lookup_proteome("Mycobacterium tuberculosis", use_uniprot=True)
# → {'kind': 'uniprot', 'proteome_id': 'UP000001020', ...}
```

## Output schema — `generate_observations_table()`

| Column | Meaning |
|---|---|
| `peptide` | Amino acid sequence |
| `mhc_restriction` | Allele from IEDB (may be `"HLA class I"` for multi-allelic studies) |
| `sample_mhc` | Allele(s) known for the source sample — the **useful** field for training |
| `mhc_class` | `I`, `II`, or `non classical` |
| `mhc_species` | Canonical species (normalized via mhcgnomes) |
| `is_monoallelic` | True if sample has a single transfected allele (721.221, C1R, K562, MAPTAC…) |
| `has_peptide_level_allele` | True if `mhc_restriction` is a specific allele (not `"HLA class I"`) |
| `is_potential_contaminant` | True for MS-eluted peptides that failed NetMHCpan binding prediction |
| `sample_match_type` | How `sample_mhc` was populated (see below) |
| `matched_sample_count` | Number of curated samples for this PMID |
| `src_cancer`, `src_healthy_tissue`, `src_ebv_lcl`, ... | Mutually-exclusive biological source categories |
| `source` | `iedb`, `cedar`, or `supplement` |
| `source_organism`, `reference_title`, `cell_name`, `source_tissue`, `disease` | IEDB sample context |
| `instrument`, `instrument_type`, `acquisition_mode`, `fragmentation`, `labeling`, `ip_antibody` | MS acquisition from `ms_samples` curation |
| `gene_name`, `gene_id`, `protein_id`, `position`, `n_flank`, `c_flank`, `flanking_species` | Source-protein mapping (only with `--with-flanking`) |

### `sample_match_type` — join provenance

| Value | Meaning | Training-grade? |
|---|---|---|
| `allele_match` | IEDB recorded a specific allele and it matched a curated sample genotype | **Yes** — high confidence |
| `single_sample_fallback` | IEDB class-only but study has exactly 1 sample, so `sample_mhc` = that sample's full genotype | Yes (for deconvolution) |
| `pmid_class_pool` | IEDB class-only + multiple samples — `sample_mhc` = union of all class-matching alleles across samples | Yes (for deconvolution), lower precision |
| `unmatched` | No curated sample for this PMID, or all samples have `mhc: unknown` | No — `sample_mhc` empty |

## Biological source classification

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

## CLI reference

### Data management

```bash
hitlist data register <name> <path> [-d DESCRIPTION]    # register a local file
hitlist data fetch <name> [--force]                     # download a known dataset (IEDB/CEDAR/viral FASTAs)
hitlist data refresh <name>                             # re-download
hitlist data info <name>                                # detailed metadata (JSON)
hitlist data path <name>                                # print the registered path
hitlist data remove <name> [--delete]                   # unregister (optionally delete file)
hitlist data list                                       # show registered datasets
hitlist data available                                  # show all known datasets
```

### Build the observations table

```bash
hitlist data build [--force]                            # ~90s full scan with tqdm progress
hitlist data build --with-flanking                      # + source-protein mapping (auto-fetches proteomes)
hitlist data build --with-flanking --use-uniprot        # broader flanking coverage via UniProt REST
hitlist data build --with-flanking --no-fetch-proteomes # don't auto-download missing proteomes
hitlist data build --proteome-release 112               # Ensembl release for human/mouse/rat
```

### Proteome management

```bash
hitlist data fetch-proteomes [--min-observations N] [--use-uniprot] [--force]
hitlist data list-proteomes
```

### Index (for raw peptide counts)

```bash
hitlist data index [--source iedb|cedar|merged|all] [--force]
```

### Export

```bash
hitlist export observations [filters...] -o train.csv   # primary ML training export
hitlist export observations -o train.parquet            # parquet output supported
hitlist export samples [--class I|II]                   # per-sample conditions (YAML curation only)
hitlist export summary                                  # species x class summary
hitlist export counts [--source iedb|cedar|merged|all]  # peptide counts per PMID
hitlist export alleles                                  # validate YAML alleles with mhcgnomes
hitlist export data-alleles                             # validate all IEDB/CEDAR alleles
```

### Filters on `hitlist export observations`

| Flag | Values |
|---|---|
| `--class` | `I`, `II`, `non classical` |
| `--species` | Any species variant (normalized via mhcgnomes) |
| `--mono-allelic` / `--multi-allelic` | Filter on `is_monoallelic` |
| `--instrument-type` | `Orbitrap`, `timsTOF`, `TOF`, `QqQ`, ... |
| `--acquisition-mode` | `DDA`, `DIA`, `PRM` |
| `--min-allele-resolution` | `four_digit`, `two_digit`, `serological`, `class_only` |
| `--output` / `-o` | `.csv` or `.parquet` |

### Reports

```bash
hitlist report [--class I|II] [--output report.txt]
```

## Development

```bash
./develop.sh    # install in dev mode
./format.sh     # ruff format
./lint.sh       # ruff check + format check
./test.sh       # pytest with coverage (~3 min)
./deploy.sh     # lint + test + build + upload to PyPI
```

See [docs/pmid-curation.md](docs/pmid-curation.md) for the curation YAML format and per-study overrides.
