# hitlist

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A curated, harmonized, **ML-training-ready** MHC ligand mass-spectrometry dataset.

hitlist ingests immunopeptidome data from [IEDB](https://www.iedb.org/), [CEDAR](https://cedar.iedb.org/), and paper supplementary tables (PRIDE/jPOSTrepo); partitions MS-eluted observations from in-vitro binding-assay measurements into two separate parquet files (so downstream consumers never silently conflate them); joins every MS observation to expert-curated sample metadata (HLA genotype, tissue, disease, perturbation, instrument); and ships both indexes as parquet + a pandas-friendly Python API.

## What's curated

| | Count |
|---|---|
| Curated PMIDs (`pmid_overrides.yaml`) | **155** — covers 89.5% of observations |
| `ms_samples` entries with per-sample metadata | 359 |
| `ms_samples` entries with 4-digit HLA typing | 237 |
| Supplementary CSVs ingested (PRIDE/jPOSTrepo) | 5 (JY, HeLa, SK-MEL-37, Raji, plasma — Gomez-Zepeda 2024) |
| Species reference proteomes (registry) | 19 (Ensembl: 4, UniProt: 15) |
| Viral reference proteomes (registry) | 30 distinct viruses, 54 name aliases |

## What's in the two indexes

After `hitlist data build` (snapshot of the shipping 1.10.x default build):

### `observations.parquet` — MS-eluted immunopeptidome

| | |
|---|---|
| **Total observations** (MS-eluted, all species) | **4,053,693** |
| **Unique peptides** | **1,285,987** |
| Unique MHC alleles | 691 |
| MHC species covered | 21 |
| IEDB rows | 3,986,991 |
| CEDAR rows | 595 |
| Supplementary rows | 66,107 |

### `binding.parquet` — in-vitro binding-assay measurements

| | |
|---|---|
| **Total binding rows** (peptide microarray, refolding, MEDi, qualitative-tier) | **895,785** |
| **Unique peptides** | **258,199** |

The two indexes share the schema (including gene annotations from the peptide-mappings sidecar), but supplementary curation is MS-only — binding is pure IEDB/CEDAR.

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
hitlist data build                           # a few minutes end-to-end;
                                             # writes observations.parquet +
                                             # binding.parquet + peptide_mappings.parquet

# Export training-ready CSVs
hitlist export training --include-evidence ms --class I --species "Homo sapiens" --mono-allelic \
    --min-allele-resolution four_digit -o mono_allelic_classI.csv

hitlist export training --include-evidence ms --class II --species "Homo sapiens" \
    -o classII_training.csv

# Presto-style flank-aware export: one row per (evidence row, peptide mapping)
hitlist export training --include-evidence both --class I --species "Homo sapiens" \
    --explode-mappings -o presto_training.parquet
```

`hitlist export training` does **not** create a new canonical store. It composes the existing `observations.parquet`, `binding.parquet`, and `peptide_mappings.parquet` indexes into one training-facing export surface. The low-level indexes keep their semantic boundaries; the training export gives downstream consumers one obvious API/CLI path when they want model-ready tables.

## Python API

### Training-data export

```python
from hitlist.export import generate_ms_observations_table
from hitlist.export import generate_training_table

# Mono-allelic human class I MS observations with ground-truth allele
mono_ms = generate_ms_observations_table(
    mhc_class="I",
    species="Homo sapiens",
    is_mono_allelic=True,
    min_allele_resolution="four_digit",
)

# Presto-style mapping-aware export: one row per (evidence row, peptide mapping)
presto = generate_training_table(
    include_evidence="both",
    mhc_class="I",
    species="Homo sapiens",
    explode_mappings=True,
)
# columns now include: evidence_kind, evidence_row_id, protein_id, position,
# n_flank, c_flank, proteome, proteome_source
```

`generate_observations_table()` remains available as a backward-compatible alias.

Species filters accept any variant — `"Homo sapiens"`, `"human"`, `"homo_sapiens"`, `"Homo sapiens (human)"` all work.

### Raw observations loading

```python
from hitlist.observations import (
    load_ms_observations,     # MS-eluted immunopeptidome
    load_binding,             # in-vitro binding-assay measurements
    load_all_evidence,        # union, tagged with an evidence_kind column
    is_built, is_binding_built,
    observations_path, binding_path,
)

# MS-elution (the default training-data path)
df = load_ms_observations()                       # everything (MS-eluted only)
df = load_ms_observations(mhc_class="I")          # class I only
df = load_ms_observations(species="Homo sapiens") # human only
df = load_ms_observations(source="iedb")          # filter by source
df = load_ms_observations(columns=["peptide", "mhc_restriction", "src_cancer"])

# Binding assays — same filter API, reads binding.parquet
bd = load_binding(mhc_class="I", mhc_restriction="HLA-A*02:01")

# Union — for affinity-predictor training, or UI flags that want both.
# Rows are tagged with evidence_kind ∈ {"ms", "binding"}.
both = load_all_evidence(gene_name="PRAME", mhc_class="I")
both["evidence_kind"].value_counts()
```

`load_observations()` remains available as a backward-compatible alias.

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

`hitlist data build` always produces three parquet files (use `--no-mappings` to skip `peptide_mappings.parquet`):

- `~/.hitlist/observations.parquet` — one row per assay observation
- `~/.hitlist/binding.parquet` — one row per binding-assay observation
- `~/.hitlist/peptide_mappings.parquet` — one row per (peptide, protein, position)

The mappings sidecar **preserves multi-mapping** so a peptide shared by MAGEA1/A4/A10/A12 keeps every paralog. Observations additionally carry semicolon-joined identity columns:

| column | example |
|---|---|
| `gene_names` | `MAGEA4;MAGEA10` |
| `gene_ids` | `ENSG00000147381;ENSG00000124260` |
| `protein_ids` | `P43359;P43363` |
| `n_source_proteins` | `2` |

```python
from hitlist.observations import load_ms_observations
from hitlist.mappings import load_peptide_mappings

# Central columns — fast for everyday filters (uses mappings sidecar for pushdown)
df = load_ms_observations(gene_name="PRAME")

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

## Output schema — `generate_ms_observations_table()`

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
hitlist export training [filters...] -o training.csv    # unified training export from canonical indexes
hitlist export samples [--class I|II]                   # per-sample conditions (YAML curation only)
hitlist export summary                                  # species x class summary
hitlist export counts [--source iedb|cedar|merged|all]  # peptide counts per PMID
hitlist export alleles                                  # validate YAML alleles with mhcgnomes
hitlist export data-alleles                             # validate all IEDB/CEDAR alleles
```

### Canonical indexes and the training export

Each `hitlist data build` writes **three** parquet files to `~/.hitlist/`:

- `observations.parquet` — MS-eluted immunopeptidome (IEDB + CEDAR + curated supplementary).
- `binding.parquet` — binding-assay rows (peptide microarray, refolding, MEDi, and
  quantitative-tier measurements like `Positive-High/Intermediate/Low`).
- `peptide_mappings.parquet` — long-form peptide → protein/position/flank mappings.

The canonical indexes are never silently mixed. Supplementary data is MS-only.
Use `hitlist export observations` and `hitlist export binding` when you want
the raw evidence families separately. Use `hitlist export training` or
`generate_training_table(...)` when you want a composed model-facing export
with `evidence_kind` tagging and optional mapping explosion for flank-aware
training pipelines.

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

### Filters on `hitlist export training`

`hitlist export training` exposes the shared pMHC filters plus two export-shape controls:

- `--include-evidence ms|binding|both` chooses which canonical evidence families to compose.
- `--explode-mappings` expands the output to one row per `(evidence row, peptide mapping)` with `protein_id`, `position`, `n_flank`, `c_flank`, `proteome`, and `proteome_source`.

MS-specific filters (`--mono-allelic`, `--instrument-type`, `--acquisition-mode`) apply only to the MS slice. Binding rows never gain fake sample context; they remain tagged as `evidence_kind="binding"` with `sample_match_type="not_applicable"`.

```bash
hitlist export training --include-evidence both --gene PRAME --class I -o prame_training.csv
hitlist export training --include-evidence ms --mono-allelic --class I -o mono_ms.csv
hitlist export training --include-evidence both --explode-mappings -o presto_training.parquet
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
