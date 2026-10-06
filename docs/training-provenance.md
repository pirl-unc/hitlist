# Training provenance and split audits

Hitlist keeps publication, source record, observation, curated experimental arm,
experimental origin, specimen, donor and acquisition identities separate.
Peptide/HLA overlap or a shared cell-line name does not establish shared material.

## Source contributors

Rebuild the indexes with Hitlist 1.64.0 or later to capture contributors:

```bash
hitlist build observations --force
```

`observation_contributors.parquet` records every contributing source row before
scanner, database and supplementary deduplication. The existing
`observations_meta.json` binds it to source snapshots and index content hashes.
Original mapped fields and complete CSV row values are JSON columns. Logical row
numbers start at 1 after the CSV headers; quoted multiline cells do not alter
this numbering. Supplementary row locators identify the ingested, sometimes
extracted CSV. They do not invent original publication sheet/row coordinates.
The supplementary manifest's upstream description is retained.

`provenance_id` links an observation to its contributors. It is a readable locator
scoped to the build's source snapshots, **not a specimen ID or a globally stable
identity across different source snapshots**. Use `evidence_row_id` to group
mapping alternatives and the manifest to interpret source locators. Projected
training exports always retain both evidence IDs, `evidence_kind`,
`provenance_id`, `provenance_status` and `lineage_context_id`.

Known assay copies retain each source record. Supplementary key matches retain
all candidate links with `overlap_unresolved`: a shared peptide, restriction and
PMID does not prove that two samples are the same. Contributor links add no
training rows or evidence weight. Rows with neither assay nor reference IDs
remain separate observations (#624).

```python
from hitlist import generate_training_table, load_contributors

training = generate_training_table(include_evidence="ms", columns=["peptide"])
contributors = load_contributors(training.provenance_id)
```

`provenance_status="indexed"` means the index supplies a contributor link.
Ordinary training exports check the recorded artifact sizes/timestamps; explicit
contributor reads and bundle creation additionally verify content hashes. A
missing or inconsistent claimed capture requires a rebuild. Older indexes stay
readable with `legacy_missing`; historical source contributors cannot be
recovered from an already deduplicated index.

### Build resources

Contributor capture compresses original source payloads losslessly. Export walks
256 retained observations at a time, resolving ancestry in indexed disk tables
before reading payloads. Multiple paths, relation flags, donor labels and every
original source row remain intact. The output uses Zstandard-compressed Parquet;
its schema and readable contributor IDs are unchanged.

The private SQLite scratch database has a **16 GiB default hard limit**, reduced
at startup when needed to leave the free-space reserve. Graph traversal uses the
same capped database; it does not perform a corpus-wide sort or create separate
recursive-query spill files. SQLite uses an 8 MiB page-cache target. Export buffers
up to 4 MiB of decoded payloads or 10,000 links, plus one potentially larger source
row and Arrow/Parquet encoding overhead.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `HITLIST_PROVENANCE_SCRATCH_DIR` | Python temporary directory (`TMPDIR` where set) | Filesystem for the private contributor database. |
| `HITLIST_PROVENANCE_MAX_GB` | `16` | Maximum database size in GiB, including graph work tables; fractional values are accepted. |
| `HITLIST_PROVENANCE_MIN_FREE_GB` | `1` | Free-space reserve checked before work and periodically on both scratch and output filesystems. This is a check, not a filesystem reservation. |

For example, to use a larger scratch volume with a 12 GiB database budget:

```bash
HITLIST_PROVENANCE_SCRATCH_DIR=/volumes/scratch \
HITLIST_PROVENANCE_MAX_GB=12 hitlist build observations --force
```

The database cap covers **temporary provenance storage**, not source downloads,
other build stages, existing artifacts, or the new output Parquet. Lossless final
output necessarily grows with the evidence, and the observation/binding DataFrames
still consume memory proportional to the corpus. Leave capacity for both the old
artifact and its replacement during publication. A given corpus is not guaranteed
to fit the default cap.

Capacity failures report the location, effective budget and recovery settings.
The collector removes its own scratch and partial contributor output and leaves
the previously published contributor artifact untouched. Scratch is disposable,
with journaling disabled; it is not a restart checkpoint. Retry the build after
freeing space or selecting a larger volume/budget. Fresh cached observations do
not allocate contributor scratch.

Publication of the complete multi-file index as a single generation is tracked
separately in [#645](https://github.com/pirl-unc/hitlist/issues/645).

## Reviewed lineage

`hitlist/data/specimen_lineage.yaml` is a versioned registry of typed entities,
namespaced reported IDs, evidence-backed aliases and study/arm contexts. Every
entity and alias needs a source reference and an evidence note. Alias targets
must have the same kind and point directly to a canonical entity. Each context
resolves the `experimental_origin`, `specimen`, `donor` and `acquisition` axes
independently as `resolved`, `unknown`, `pooled` or `ambiguous`. IDs are JSON
arrays in exports so pooled memberships remain representable in CSV and parquet.

Contexts attach through `(pmid, condition_id)` only when observation attribution
resolves an arm. The original `sample_attribution` explains how it did so;
ambiguous or unassigned observations remain unknown. Counting fallbacks in
`sample_identity.py` are never specimen IDs.

The initial registry links the 16 reused Abelin/Sarkizova monoallelic datasets
and the Bassani-Sternberg/Liepe fibroblast dataset at **experimental-origin**
precision. Physical specimen, donor and acquisition IDs remain unknown. Abelin's
comparison-only arms remain unprofiled. Registry coverage is deliberately partial.

## Versioned bundles

```bash
hitlist export training --include-evidence ms --class I \
    --columns peptide mhc_restriction --bundle train-bundle \
    --split-policy independent_experiments
```

```python
from hitlist import write_training_bundle, verify_training_bundle

write_training_bundle(
    "train-bundle", include_evidence="ms", map_source_proteins=True,
    columns=["peptide", "mhc_restriction", "protein_id"],
    split_policy="independent_experiments",
)
manifest = verify_training_bundle("train-bundle")
```

A new destination receives `training.parquet`, `identities.parquet`,
`contributors.parquet`, `lineage.json` and `manifest.json`. The identity table
retains audit fields even when the training projection omits them. The manifest
is published last; an interrupted directory without it is not a valid bundle.
Existing destinations are never overwritten.

Schema version 1 records output SHA-256 hashes and byte sizes, all effective
training options, Hitlist/dependency versions, source/build metadata, input-index
and current-curation hashes, the requested split policy, row/observation counts
and contributor/lineage coverage. No splitting randomness is used (`seed: null`).
Verification checks bytes and table relationships. Inputs are fingerprinted
before and after export; changed inputs abort publication. With peptide-origin
enrichment, the manifest also fingerprints the local Ensembl transcript cache;
if first use populates that cache, retry after it stabilizes. Custom backend
objects are rejected because their state cannot be reproduced from options.

Bundles preserve the requested rows; recording a split policy does not enforce
it or assign partitions. Legacy bundles explicitly report incomplete contributor
coverage. Preserve the bundle and manifest downstream through training, tuning
and model selection, rather than saving only peptide sequences.

## Audit proposed partitions

```bash
hitlist audit-splits --partition train=train-bundle --partition test=test-bundle \
    --policy independent_experiments --output split-audit.json
```

```python
from hitlist import audit_splits, audit_training_bundles

report = audit_splits({"train": train_frame, "test": test_frame},
                     policy="specimen_disjoint")
report = audit_training_bundles({"train": "train-bundle", "test": "test-bundle"},
                               policy="independent_experiments")
```

The report separates exact peptide, exact normalized peptide/restriction,
observation, reviewed experimental-origin, specimen, donor, acquisition and
whole-study overlap. Unresolved presenters do not become exact pMHC matches.
Repeated protein mappings count once per observation; extra observation rows
are reported separately. Direct frame callers must retain the audit columns and
use the same reviewed registry. Bundle audits recover them from the identities
table and require the same registry in both bundles.

| Policy | Required disjoint identities |
| --- | --- |
| `report_only` | Reports every axis without an independence claim |
| `peptide_disjoint` | Peptide sequences |
| `pmhc_disjoint` | Peptide and resolved restriction |
| `independent_experiments` | Observations and reviewed experimental origins |
| `specimen_disjoint` | Reviewed specimens |
| `donor_disjoint` | Reviewed donors |
| `whole_study` | PMIDs, as an explicitly conservative grouping |

Known forbidden overlap produces `fail`. No overlap with missing required
resolution produces `inconclusive`; only complete required coverage can produce
`pass`. Report-only audits return `reported`. The CLI exits 2 for failed or
inconclusive claims, 1 for invalid input, and 0 otherwise. Missing columns,
pooled/ambiguous material and incomplete registry coverage never prove
independence. The audit describes supplied partitions and makes no claim about
historical model weights or external predictors' training data.
