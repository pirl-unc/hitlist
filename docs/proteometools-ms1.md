# Nontryptic peptide MS1 response (PXD021013)

This source provides experimentally measured MS1 signals from chemically synthesized
peptide pools. It complements processing and MHC-elution data with non-MHC MS
observations. Crude synthesis, unknown individual purity and estimated 200/500 fmol
injections prevent interpreting these values as pure ionization efficiency or
intracellular concentration. HLA-I/HLA-II library labels describe sequence selection,
not measured MHC restriction.

The original study is [ProteomeTools nontryptic peptides](https://doi.org/10.1038/s41467-021-23713-9).
The source family is `proteometools_ms1`, package schema 23, revision
`pxd021013_ms1_v1`, catalog 28. Older package contracts remain unchanged.

## Source acquisition and package

Acquire the four core assets listed in
`hitlist/data/evidence_revisions/proteometools_ms1_core.json`, preserving their exact
reviewed bytes. The intended-product workbook is member
`41467_2021_23713_MOESM4_ESM.xlsx` of the original supplementary ZIP. Then acquire an
explicit subset of SEARCH ZIP files from the public deposited index. Supply their
original source names and the SHA256/byte counts recorded at acquisition:

```python
from hitlist.proteometools_ms1 import write_proteometools_ms1_package

write_proteometools_ms1_package(
    "/data/pxd021013-originals", "/data/pxd021013-generation",
    archives=[{
        "filename": "HLA1-first-DDA.zip",
        "source_name": "TUM_HLA_1_01_01_DDA-1h-R1-unspecific.zip",
        "sha256": "42b18e4ad3f689b84ad2b905757404fafd5f2f13ef2cd8bc2f4499610a52d8c6",
        "size_bytes": 21615641,
    }],
)
```

The writer does not download, overwrite destinations or modify the caller's files.
It takes independent immutable snapshots, validates deposited SEARCH->RAW identities,
replays all typed tables from original bytes and publishes atomically. SEARCH hashes
are caller-pinned acquisition receipts, not a remote signature. A new subset is a
new package generation; no package release is needed for each additional pool.

Only the reviewed single-run MaxQuant layout is projected. Experiment/overall
summary totals are preserved as source records, not mistaken for additional runs.
The remaining ZIP members stay in the original archived bytes. Unsupported layouts
fail explicitly rather than guessing aggregation or biological parent origins from
concatenated-library search coordinates.

## Queryable evidence and context

```python
from hitlist import EvidenceStore

store = EvidenceStore.open("/data/pxd021013-generation")
store.catalog()  # source registry, relations, fields and available recipes
rows = store.query(
    "proteometools_forms",
    select=["peptide_sequence", "quantitative_raw", "value_status",
            "peptide_modifications_raw", "native.pep", "native.run_id"],
    limit=100,
)
```

Seven relations expose original assets, positional source records, all intended
pool/sequence mappings, protocol contexts, deposited runs, sequence quantities and
molecular-form quantities. Forms link to sequence quantities and runs; runs link to
source records and protocol context. Raw strings accompany numeric values and
explicit missing/zero/invalid states. Every original source column is preserved in
positional row JSON, including repeated SDRF modification columns.

The primary synthetic-pool context is distinct from conflicting community SDRF
assertions (`cell`, generic Fusion, constant HCD/run1). No cell or specimen identity
is inferred from those assertions. Query native context fields for load alternatives,
unknown purity, unknown per-run load choice and conflict status. Run fields preserve
fixed/variable modifications, software, search enzyme, original acquisition label and
library origin. Search digestion is not laboratory digestion.

## Conditional response recipe

```python
store.materialize({
    "schema_version": 1,
    "name": "relative_MS1",
    "registered_recipe": "pxd021013_intended_unmodified_ms1_v1",
}, "/data/pxd021013-response")
```

Admission requires intended full-length product membership in that pool, canonical
sequence, positive finite sequence and form signal, PEP in [0,0.01) at both grains,
no reverse/contaminant flag and one unambiguous `Unmodified` variable-modification
form. Fixed modifications remain explicit: `Unmodified` does not imply no CAM.
No unknown molecular forms are summed, no absent products become negative examples,
and no held-out intensity/CV filter is applied.

Outputs include each original form's disposition, every planned product/run status,
source-linked examples, dependence groups, summary and frozen source tables.
Examples retain raw log10 signal and within-run centered log10 signal. Centering
defines a conditional composition endpoint; models must center their predictions
over the same candidate set. The reported target center is audit-only and must not
be used as an inference covariate. This recipe does not predict detection probability.

Pool groups connect all repeated intended sequences, including unobserved products,
before intensity or quality filtering. Off-planned background identifications cannot
enter this endpoint and do not connect target groups. A prospective experiment assigns whole components to splits;
the recipe does not promise that any chosen subset contains enough independent
components. Evaluate a separate observer before freezing it for processing/RL
comparisons. Source verification and synthetic software fixtures are not biological
predictive validation.
