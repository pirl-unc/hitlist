# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A curated, harmonized, **ML-training-ready** MHC ligand mass-spectrometry dataset.

hitlist ingests immunopeptidome data from [IEDB](https://www.iedb.org/), [CEDAR](https://cedar.iedb.org/), and paper supplementary tables (PRIDE/jPOSTrepo); filters out binding-assay data; joins every observation to expert-curated sample metadata (HLA genotype, tissue, disease, perturbation, instrument); and ships it as a single parquet file + pandas-friendly Python API.

## What's curated

| | Count |
|---|---|
| Curated PMIDs (`pmid_overrides.yaml`) | **156** — covers ~90% of observations |
| Supplementary CSVs ingested (PRIDE/jPOSTrepo) | 6 (JY, HeLa, SK-MEL-37, Raji, plasma — Gomez-Zepeda 2024; Stražar 2023 HLA-II, 308K rows) |
| Species reference proteomes (registry) | 19 (Ensembl: 4, UniProt: 15) |
| Viral reference proteomes (registry) | 30 distinct viruses, 54 name aliases |

## What's in the observations table

After `hitlist data build` (v1.10.4):

| | |
|---|---|
| **Total observations** (MS-eluted, all species) | **4,421,724** |
| **Unique peptides** | **1,457,276** |
| Unique MHC alleles | 703 |
| MHC species covered | 21 |
| Unique PMIDs | 2,403 |
| IEDB rows | 3,986,991 |
| CEDAR rows | 595 |
| Supplementary rows | 434,138 |

### Human MHC-I breakdown

| | |
|---|---|
| Observations | 2,490,373 |
| Unique peptides | 650,652 |
| Mono-allelic rows | 532,494 |
| Four-digit allele resolution | 1,174,722 obs |

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

### Peptide → protein attribution and flanking context

`hitlist data build` always produces two parquet files (use `--no-mappings` to skip):

- `~/.hitlist/observations.parquet` — one row per assay observation
- `~/.hitlist/peptide_mappings.parquet` — one row per (peptide, protein, position)

The mappings sidecar **preserves multi-mapping** so a peptide shared by MAGEA1/A4/A10/A12 keeps every paralog. Observations additionally carry semicolon-joined identity columns:

| column | example |
|---|---|
| `gene_names` | `MAGEA4;MAGEA10` |
| `gene_ids` | `ENSG00000147381;ENSG00000124260` |
| `protein_ids` | `P43359;P43363` |
| `n_source_proteins` | `2` |

```python
from hitlist.observations import load_observations
from hitlist.mappings import load_peptide_mappings

# Central columns — fast for everyday filters (uses mappings sidecar for pushdown)
df = load_observations(gene_name="PRAME")

# Long form for paralog / position / flank analysis
mappings = load_peptide_mappings(gene_name="MAGEA4")
# columns: peptide, protein_id, gene_name, gene_id, position, n_flank, c_flank, proteome
```

For ad-hoc queries without building the full table:

```python
from hitlist.proteome import ProteomeIndex

idx = ProteomeIndex.from_ensembl(release=112, species="human")
flanking = idx.map_peptides(["SLLMWITQC", "GILGFVFTL"], flank=10)
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
| `gene_names`, `gene_ids`, `protein_ids`, `n_source_proteins` | Multi-mapping peptide → source-protein attribution (always populated; use `peptide_mappings.parquet` for long-form positions + flanks) |

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
hitlist data build                                      # always builds peptide_mappings.parquet
hitlist data build --use-uniprot                        # broader proteome coverage via UniProt REST
hitlist data build --no-mappings                        # skip mapping step (faster, no gene attribution)
hitlist data build --no-fetch-proteomes                 # don't auto-download missing proteomes
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
hitlist export observations [filters...] -o train.csv   # MS immunopeptidome + sample metadata
hitlist export observations -o train.parquet            # parquet output supported
hitlist export binding [filters...] -o binding.csv      # binding-assay index (separate from MS)
hitlist export samples [--class I|II]                   # per-sample conditions (YAML curation only)
hitlist export summary                                  # species x class summary
hitlist export counts [--source iedb|cedar|merged|all]  # peptide counts per PMID
hitlist export alleles                                  # validate YAML alleles with mhcgnomes
hitlist export data-alleles                             # validate all IEDB/CEDAR alleles
```

### MS vs binding: two separate indexes

Each `hitlist data build` writes **two** parquet files to `~/.hitlist/`:

- `observations.parquet` — MS-eluted immunopeptidome (IEDB + CEDAR + curated supplementary).
- `binding.parquet` — binding-assay rows (peptide microarray, refolding, MEDi, and
  quantitative-tier measurements like `Positive-High/Intermediate/Low`).

The two indexes are never mixed. Supplementary data is MS-only. Use
`hitlist export observations` for immunopeptidome training data and
`hitlist export binding` for affinity / binding-predictor work. They
share the same schema plus the mappings-sidecar gene annotations, so the
same filters apply to both.

### Filters on `hitlist export observations`

| Flag | Values |
|---|---|
| `--class` | `I`, `II`, `non classical` |
| `--species` | Any species variant (normalized via mhcgnomes) |
| `--mono-allelic` / `--multi-allelic` | Filter on `is_monoallelic` |
| `--instrument-type` | `Orbitrap`, `timsTOF`, `TOF`, `QqQ`, ... |
| `--acquisition-mode` | `DDA`, `DIA`, `PRM` |
| `--min-allele-resolution` | `four_digit`, `two_digit`, `serological`, `class_only` |
| `--mhc-allele` | Exact match on `mhc_restriction` after allele normalization. Repeatable / comma-separated. |
| `--gene` | Symbol, Ensembl ID, or old alias (HGNC synonym lookup). Repeatable / comma-separated. Requires the mappings sidecar (default-on at build). |
| `--gene-name` | Exact match on `gene_name` column (no HGNC lookup) |
| `--gene-id` | Exact match on `gene_id` column (ENSG) |
| `--serotype` | HLA serotype: locus-specific (`A24`, `B57`, `DR15`) or public epitope (`Bw4`, `Bw6`). Matches any serotype the allele belongs to, so `--serotype Bw4` returns A\*24:02, B\*27:05, B\*57:01, etc. Repeatable / comma-separated. |
| `--output` / `-o` | `.csv` or `.parquet` |

All filters are pushed down to the parquet reader (pyarrow), so `--gene PRAME` reads
only the matching row groups — typically milliseconds rather than a full table scan.
Examples:
- `hitlist export observations --gene PRAME --class I -o prame_classI.csv`
- `hitlist export observations --gene "MART-1"` (HGNC resolves to `MLANA`)
- `hitlist export observations --mhc-allele HLA-A*02:01 --mono-allelic`
- `hitlist export observations --serotype A24` (locus-specific)
- `hitlist export observations --serotype Bw4` (public epitope — A*23/24/25/32, B*13/27/44/51/52/53/57/58)

### Filters on `hitlist export binding`

Same shape as the observations filters minus the MS-specific ones
(`--mono-allelic`, `--instrument-type`, `--acquisition-mode`). `--source`
accepts only `iedb` or `cedar` — supplementary data is MS-only and never
appears in the binding index.

```bash
hitlist export binding --gene PRAME --class I -o prame_binding.csv
hitlist export binding --mhc-allele HLA-A*02:01 --serotype Bw4
```

### A note on mono-allelic curation

Mono-allelic is a **PMID-level** flag, not a per-sample property. Curation
lives in `pmid_overrides.yaml` (see `mono_allelic_host` and `ms_samples`).
Rows can legitimately have `is_monoallelic=True` with an empty
`mhc_restriction` — for example, supplementary contaminant peptides under
a mono-allelic PMID override carry the flag but not a per-row allele. If
your downstream pipeline needs a strict "mono-allelic AND has allele"
subset, post-filter on
`is_monoallelic & mhc_restriction.str.startswith("HLA-")`.

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
