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
as independent evidence. Peptide/group enrichment remains a later source release.

The pinned quantitative download is 84.4 MB and expands to 1.40 GB; its matching
FASTA is 14.7 MB. The adapter uses disk staging for source-row ordering and needs
additional scratch/output space. `max_upstream_bytes` (default 2 GiB) bounds the
installed upstream artifact, `max_input_bytes` bounds the declared compressed
download and original assets, and `max_output_bytes` bounds each staging database
and the final bundle. These are artifact bounds, not a total cache/disk quota;
orfdata's existing cache entries are retained. No vendor database is acquired.

| Source ID | Available records | Scope and limitations |
|---|---|---|
| `pfizer_2026_03` | Protein-abundance table and deposited no-isoform search FASTA | Tissue pools; `BioReplicate` is not a dog. The reported protein representative does not resolve underlying protein-group alternatives. This table contains no peptide detections. |
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
information requires separate extraction from its deposited vendor databases;
they are not downloaded implicitly or inferred from protein abundance.
The outstanding extraction and protein-group reconciliation are tracked in
[#668](https://github.com/pirl-unc/hitlist/issues/668).

Since 1.71.1, the Pfizer instrument is explicitly unresolved: the
[publication](https://www.nature.com/articles/s41597-026-07512-9) reports Fusion
Eclipse acquisition (and mentions Lumos in its real-time-search description),
while the [deposit record](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD070290)
lists Fusion Lumos. The manifest preserves both cited `instrument_reports` and
`instrument_status: conflicting_reports`; normalized rows carry the unresolved
value. Raw/run metadata is needed to settle the conflict. Existing frozen
bundles keep their original metadata snapshot; curate a new bundle to adopt
the corrected provenance.

Since 1.71.2, `canine_bulk.yaml` holds tissue groups and named sample contexts.
Each source selects a default context; `sample_context_by_id` assigns the five
Labrador adult-fluid experiments by their original sample IDs. Tissue names do
not infer age or health. These reviewed values are frozen into each new bundle's
manifest and sample rows. All 56 existing sample/pool annotations are preserved;
older bundles continue to replay their own frozen metadata.

Every directory contains `bulk.sqlite`, the byte-exact `reference.fasta`, and a
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

Default import limits are 256 MiB per input, 512 MiB expanded ZIP data, five
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
