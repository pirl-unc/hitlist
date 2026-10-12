# Canine bulk-MS evidence — #666

## Reviewed sources and scope

Add explicit, offline-first bulk evidence imports for Pfizer's MSV000099744
protein table (2026-03-23 update) and the Labrador PeptideAtlas 2024-01 build 573.
Pin original assets by SHA256 and size, including the deposited searched FASTAs.
Keep these references distinct from complete isoform mapping backgrounds.
Source payloads do not enter package distributions. Source licenses remain
explicitly unresolved unless independently documented; article licenses do not
establish the terms of every deposited asset.

The Pfizer table has 313,263 protein-abundance rows and 38 tissue-pool identities.
BioReplicate identifies a pool, not an individual dog. The paper describes three
male and three female contributors; do not infer six contributors for every
sex-specific tissue or expand a pool into positive donors. Retain source-native
Abundance, IBAQ, rIBAQ and normalization uncertainty. ProteinGroup equals Protein
in this asset; that does not prove protein inference resolved all alternatives.
The deposit's peptide results reside in large Proteome Discoverer databases;
the chosen protein table cannot supply peptide detections or spectra.

Labrador's archive contains 18 experiment/sample records, peptide-to-experiment
detections, modified peptide observations, complete reported protein mappings
and protein-inference groups. Tissue experiments aggregate a stillborn female
and a six-month-old male with hip dysplasia. Preserve this context; archive sample
IDs are not resolved dog IDs. Plasma/urine experiments have separate adult scopes.
Library membership is not a tissue detection. Original IDs, row numbers, quality
fields and modifications remain available without inferring peptide-level q-values.

## Storage and API

Use the existing datacache verified downloader and shared bounded-cache checks
from managed-reference support. Downloads are explicit and budgeted; supplied
local assets are checked without copying into or modifying their source directory.
All source identities and limitations are recorded in a versioned YAML catalog.

`curate_canine_bulk(source_id, output_dir, local_assets=..., allow_download=False)`
publishes a new directory atomically after successful verification. Use streaming
CSV/ZIP reads and SQLite with bounded batches and disk-backed joins. Preserve
original tables/cells plus logical source rows; checked references are frozen.
Normalized SQL views expose samples, peptides, modified peptides, protein mappings,
protein groups and protein abundance with stable, source-namespaced identities.
Queries retain all alternatives and never duplicate observations per mapping.
Scope missing detection to the actual source experiment/search, never an entire
species or an unsearched isoform background.

Expose inspection and bounded table reads through `hitlist.bulk_proteomics` and
the package API. Existing cell-line APIs/defaults remain unchanged. New evidence
records explicitly declare bulk proteomics and incompatible ligand-response
semantics so they cannot enter positive DLA admission or the presentation
blacklist. No human Atlas fallback or implicit global observation build occurs.
Missing canine nonmalignant heart/brain/lung presentation coverage stays missing.

Bound input bytes, expanded ZIP bytes, logical records, working/output bytes and
query materialization. Reject bad hashes, duplicate identities, broken joins,
unsafe archive members, changed inputs and existing destinations. Failures remove
staging outputs. Store manifest hashes and row counts; verify them read-only.

## Acceptance and delivery

First add failing tiny offline fixtures for both source layouts. Exercise pool
identity, repeated acquisitions, donor ambiguity, developmental/health context,
all protein mappings, modified peptides, inference versus detection, original
record recovery, checksum failures, broken links, bounded reads and no DLA/MS
blacklist admission. Verify a real bounded source query against the original
assets and explain any remaining unavailable fields. Read available literature
for normal canine presentation coverage and record the audit's limits.

Run format, lint and full tests/CI without relaxing memory guards. Version this
as a separate PR, merge only after #667 is released and all final-head checks
pass, deploy from clean main and verify the exact published PyPI artifacts.
