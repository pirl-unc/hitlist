# Canine bulk proteomics

Hitlist 1.71.0 imports two checked canine bulk-MS sources explicitly. These
records support protein existence, source-scoped detectability and normal-tissue
counterevidence. They are not DLA-immunoprecipitated ligands and do not provide
presentation-blacklist coverage.

```python
from hitlist import curate_canine_bulk, load_bulk_evidence, verify_bulk_evidence

curate_canine_bulk(
    "labrador_2024_01", "dog-atlas",
    local_assets={"table": "atlas_build_573.tsv.zip", "reference": "Canis_familiaris.fasta"},
)
verify_bulk_evidence("dog-atlas")
heart = load_bulk_evidence("dog-atlas", tissue="Heart")
targets = heart[heart.is_bulk_detection.eq(1)]
alternatives = load_bulk_evidence("dog-atlas", table="mappings", peptide="ACDEFGHIK")
```

Alternatively, pass `allow_download=True` to acquire the catalog's pinned assets
through datacache. `fetch_canine_bulk_asset(source_id, "reference")` fetches just
one asset. Cache hits are checksum verified; corruption raises and nothing is
automatically evicted. Local assets are checked without modifying them. The
catalog is available through `canine_bulk_catalog()` and
`load_bulk_sources(include_canine=True)` in `hitlist.bulk_proteomics`.

The Pfizer summary can also be acquired through the shared orfdata registry:

```python
curate_canine_bulk(
    "pfizer_2026_03", "dog-pfizer",
    provider="orfdata", allow_download=True,
    orfdata_cache_root="/path/to/openvax-cache",  # Optional cache override.
)
```

This selects the reviewed `pfizer-atlas-dog-protein` release `2026-10-12.1`,
with pinned registry metadata and source/artifact checksums. Without
`allow_download=True`, both the dataset and reference must already be installed
in that orfdata cache. Use orfdata's `DatasetRegistry.download(...)` and
`download_reference(...)` for separate acquisition; `local_assets` belongs to
the original source provider and cannot be combined with `provider="orfdata"`.
The default provider remains `"source"`.

The adapter selects native iBAQ and rIBAQ separately, joins each pair by original
source identity, and verifies a byte-exact reconstruction of the original
protein table. It then uses the existing Hitlist writer and YAML sample contexts.
Protein rows, IDs, native strings, pool scope and reference sequences retain
their existing meaning. `manifest.json` additionally records the upstream
package version, dataset manifest, artifact digest, access terms and searched
reference specification. Readers and verification use that frozen bundle offline;
they do not require the current upstream catalog or reinterpret old generations.
Imports through the two providers describe the same study and must not be counted
as independent evidence. The peptide provider below adds a separately selected
generation from that same study.

The pinned quantitative download is 84.4 MB and expands to 1.40 GB; its matching
FASTA is 14.7 MB. The adapter uses disk staging for source-row ordering and needs
additional scratch/output space. `max_upstream_bytes` (default 2 GiB) bounds the
installed upstream artifact, `max_input_bytes` bounds the declared compressed
download and original assets, and `max_output_bytes` bounds each staging database
and the final bundle. These are artifact bounds, not a total cache/disk quota;
orfdata's existing cache entries are retained. No vendor database is acquired.

## Released peptide, PSM and protein-group evidence

On Python **3.11+**, select `provider="orfdata_peptides"` to consume ORFdata
**0.40+** and its pinned `pfizer-atlas-dog-peptide-evidence` generation
`2026-10-27.1`. All current Hitlist providers require Python 3.11+ and pandas 3+.

```python
from hitlist import open_canine_peptide_evidence

curate_canine_bulk(
    "pfizer_2026_03", "dog-pfizer-peptides",
    provider="orfdata_peptides", allow_download=True,
)
# Without allow_download, the pinned dataset must already be installed.
reader = open_canine_peptide_evidence("dog-pfizer-peptides")
value = next(reader.dataset.measurements(status="measured"))

# Common ORFdata APIs stream observations and preserve complete alternatives.
measurements = reader.dataset.measurements(target=value.target)
assignments = reader.dataset.mappings(source_identifier=value.target.identifier)
groups = reader.dataset.relations()

# A bounded dataframe retains native quantities, units, IDs and provenance.
peptide_rows = load_bulk_evidence(
    "dog-pfizer-peptides", peptide=value.target.identifier,
)
# Explicit native access includes confidence definitions, modifications and controls.
decoys = reader.native.records("DecoyPsms")
modifications = reader.native.records("FoundModifications")
```

The [upstream release](https://github.com/openvax/orfdata/releases/tag/data-pfizer-dog-peptide-evidence-2026-10-27.1)
downloads **6.90 GB** in four checksummed parts and installs **24.71 GB**.
Hitlist snapshots that complete artifact into a schema-2 bundle containing
`dataset.sqlite` and `manifest.json`. A separate searched FASTA or vendor
analysis database is not acquired. The snapshot uses an independent inode,
with a filesystem copy-on-write clone where available and a full-copy fallback.
Allow space for both acquisition and the snapshot on filesystems without clones.
The selected provider defaults to 8 GiB compressed input, 32 GiB installed input,
32 GiB output and 100 million selected native records. User-supplied `limits`
can tighten these bounds; undersized bounds fail before acquisition.

The artifact retains **70,280,798 native rows in 42 tables**, the complete
20,649-sequence searched reference, 38 tissue pools, four bridge channels,
two empty channels and 96 runs. The common quantitative view expands stored
vectors to 200,660,383 logical measurement cells; these are not independent
peptide detections or dogs. Use peptide/sample selectors or streaming queries.
`load_bulk_evidence` retains its 250,000-row bound and raises instead of truncating.

| Table | Returned records |
| --- | --- |
| `peptides` | Native measurement cells, with full provenance, sample context, modifications, value/status and unit/scale |
| `mappings` | Complete reported assignment sets and separate sequence-compatible candidates; nested `targets` retain every alternative |
| `groups` | Reported protein-group membership relations and the pinned same-study summary context |
| `samples`, `assays` | Source sample/channel and acquisition/aggregate identities |
| `raw_<native table>` | Exact native fields, including control/decoy tables and opaque numeric-vector bytes |

A peptide selector filters measurements and mappings without narrowing their
protein candidate sets. Tissue selectors apply to measurements, samples and
assays; mapping/group relations keep their native scopes. Use the reader's
`dataset` and documented `native` APIs for typed targets, native keys, missing
acquisition records and explicit relationship queries. `raw_` reads add no
Hitlist interpretation columns.

There is no confidence threshold or normalization in this adapter. Zero remains
zero; missing/nonfinite quantities retain their native state in provenance, and
native vectors remain byte-exact. A high native confidence label does not turn a
decoy into a detection. The normal measurement projection includes source target
rows; decoy records and definitions remain available through native tables.
Shared or unresolved assignments remain explicit. The same-study protein summary
is context, not another independent detection or abundance assigned to an
individual group member. Normalized rows have `evidence_kind="bulk"`, false
`is_ms_observation` and an empty MHC restriction. Pools do not identify individual
positive dogs and supply no DLA presentation-blacklist donor counts.

The frozen manifest retains source, artifact, reference and registry identities,
all native reconciliation metadata, and both instrument reports. Native run
metadata supports the deposit's Lumos report; the publication's Eclipse report
remains present. Old frozen metadata is not rewritten. `verify_bulk_evidence`
and `open_canine_peptide_evidence` verify the full artifact offline without a live
registry. After accepting an immutable bundle, the reader's explicit
`verify=False` avoids repeated hashing; bounded dataframe reads likewise do not
rehash the full artifact.

| Source ID | Available records | Scope and limitations |
|---|---|---|
| `pfizer_2026_03` | Original protein summary, plus explicitly selected peptide/PSM/group generation | Tissue pools; `BioReplicate` is not a dog. The original summary has no peptide detections and its reported representative does not resolve group alternatives. The companion generation retains native assignments and all compatible alternatives. |
| `labrador_2024_01` | Peptide/modified-peptide experiment detections, protein mappings/inference groups, metadata and full deposited reference | Tissue experiments aggregate a stillborn female and a six-month-old male with hip dysplasia. Adult plasma/urine have separate scopes. Archive experiment IDs do not resolve individual positive dogs. |

Original studies: [Pfizer atlas](https://www.nature.com/articles/s41597-026-07512-9)
and [Labrador PeptideAtlas](https://www.nature.com/articles/s41597-026-06647-z).
The [PeptideAtlas export](https://peptideatlas.org/builds/canine/) includes
contaminant and decoy mappings; they remain explicitly labeled. `is_bulk_detection`
requires a reported positive count and at least one source-target mapping.
Shared target/control mappings stay ambiguous; a target mapping does not identify
a unique source protein. No library membership is expanded into tissue detection.

The Pfizer import retains native Abundance, IBAQ, rIBAQ and sumIBAQ separately.
It does not convert them to TPM, pool them across quantifiers, or claim an
individually measured isoform. Nonfinite source values (including `-Inf`) produce
a missing numeric value with `reported_nonfinite` status; the original cell
remains in `raw_proteins`. Missing values are not zeros. Peptide-level Pfizer
information is available through the released shared peptide provider below.
The original summary provider retains its historical representative-only scope;
peptides are not inferred from protein abundance.

Since 1.71.1, the Pfizer instrument is explicitly unresolved: the
[publication](https://www.nature.com/articles/s41597-026-07512-9) reports Fusion
Eclipse acquisition (and mentions Lumos in its real-time-search description),
while the [deposit record](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD070290)
lists Fusion Lumos. The manifest preserves both cited `instrument_reports` and
`instrument_status: conflicting_reports`; normalized rows carry the unresolved
value. The peptide generation additionally retains native Lumos run reports without
erasing the conflicting publication claim. Existing frozen
bundles keep their original metadata snapshot; curate a new bundle to adopt
the corrected provenance.

Since 1.71.2, `canine_bulk.yaml` holds tissue groups and named sample contexts.
Each source selects a default context; `sample_context_by_id` assigns the five
Labrador adult-fluid experiments by their original sample IDs. Tissue names do
not infer age or health. These reviewed values are frozen into each new bundle's
manifest and sample rows. All 56 existing sample/pool annotations are preserved;
older bundles continue to replay their own frozen metadata.

Schema-1 directories contain `bulk.sqlite`, the byte-exact `reference.fasta`, and a
manifest with original asset hashes, source/version/protocol, limitations, table
counts, resource bounds and artifact hashes. Original table columns and logical
`_source_row` values survive; source `\N` null markers decode to SQL NULL. Tables
include `samples`, `peptides`, `proteins`, and for Labrador `mappings`,
`mapping_summary`, `modifications`, and `raw_<source table>` tables. Mapping
positions retain the source's one-based inclusive coordinates. Protein and
modification probabilities retain their original scope; no per-PSM q-values or
spectra are invented. Missing modification counts remain missing.

`verify_bulk_evidence` checks bytes, counts, schema-specific view semantics and
source relationships offline.
Small reads do not rehash the full database; verify once when accepting a bundle.
The query defaults to at most 250,000 rows and raises if exceeded. Filter by
`tissue` or `peptide`, choose another `table`, or explicitly raise `max_rows`.
An empty result describes this source's acquisition/search coverage, not absence
from a dog, a tissue generally, or every isoform in a complete mapping reference.

For the original providers, default import limits are 256 MiB per input, 512 MiB expanded ZIP data, five
million rows per input table/reference, one million residues per protein and
2 GiB published output. SQLite uses an 8 MiB page cache and disk-backed joins;
these limits are not a hard process-RSS or filesystem quota. The managed download
cache defaults to 1 GiB. Limits fail before publication and existing destinations
are never replaced. Source payloads and generated databases are not bundled in
the package; asset redistribution terms remain explicitly unresolved.

The canine presentation-coverage audit found these sources to be bulk proteomics,
while the reviewed DLA ligand studies concern tumors or engineered cell lines.
The literature search did not establish nonmalignant heart/brain/lung DLA ligand
coverage with two resolved dogs. This is a scoped availability finding, not proof
that no such data exist. `presentation_coverage` remains `missing`; none of these
bulk records can satisfy the presentation blacklist's two-dog rule.
