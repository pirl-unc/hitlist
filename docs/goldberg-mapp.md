# Goldberg2025 original MAPP measurements

The `goldberg_mapp` family preserves original PXD052314 search results and
separate views from Goldberg2025 publisher Tables3/5. The pinned revision is
`goldberg2025_mapp_original_v1` (package schema25, catalog31).

For cross-source work, append `goldberg2025` together with the other
captured-fragment adapters and query the common [`mapp_*` API](mapp-canonical.md).
The canonical package adds lossless same-grain extensions for the source's
search, publisher and pooled-table layout; it does not replace this standalone,
independently replayable source receipt.

```python
from hitlist.goldberg_mapp import write_goldberg_mapp_package
from hitlist.evidence_store import EvidenceStore

write_goldberg_mapp_package("original-assets", "new-package")
store = EvidenceStore.open("new-package")
quantities = store.query(
    "goldberg_quantities",
    select=["measurement_id", "quantitative_value", "value_status",
            "native.quantity_kind", "native.identification_type"],
    limit=None,
).table
```

Six original assets are hash/size pinned in
`hitlist/data/evidence_revisions/goldberg_mapp.json`; the source-adapter receipt
used by the canonical package is separately pinned in
`hitlist/data/evidence_revisions/goldberg_mapp_assets.json`. The writer copies
complete assets to independent snapshots, materializes the typed tables, and
replays them before publishing a new directory. Reopening verifies both the
original bytes and the exact typed rows; editing a Parquet file and rehashing the
inventory is not sufficient to pass verification. No existing package is
overwritten.

The source grain remains explicit:

| Relations | What they preserve |
|---|---|
| `source_records`, `assets` | Original field order, literal values, row/file locators and complete original assets |
| `peptides`, `forms`, `identifications`, `groups` | Native identities, source contributor lists, all parent labels, confidence and reverse/contaminant flags |
| `quantities` | Separate peptide, molecular-form and individual-identification intensities, including total/run values, zeros, missing and invalid values |
| `references` | Source FASTA header assertions, including truncated taxonomy; full search reference sequences are absent |
| `acquisitions`, `assertions` | Original raw-file/experiment/sample labels and separately scoped community metadata |
| `publisher_cells`, `publisher_quantities`, `publisher_links` | Exact workbook coordinates, XML types/styles/formulas, distinct assay quantities, and original/publisher reconciliation claims |
| `pooled_slots` | Every ordered Table3 label/value slot, retaining repeated labels and unresolved sample/unit identity |
| `material_contexts`, `material_links` | Shared project/arm/acquisition graph with explicit linkage precision |

Original and publisher rows have distinct identities. The publisher-selected
subset omits unflagged original rows and includes nonhuman source annotations.
Original run labels disagree with two publisher replicate suffixes; these remain
unresolved exact-column links, not inferred aliases. Source human annotations
are not independently validated peptide origins. Cell-line species and peptide
origin must not be substituted for each other.

Whole-cell digested proteomics, proteasome-associated proteins, associated peptides
and extracellular secretome have different measured-pool labels. Matching repeat
labels do not establish aliquot-matched precursor measurements. The shared material
graph contains no invented physical specimens or aliquots. Unknown machinery
status is explicitly not assumed wild type. Community SDRF annotations retain
their separate provenance, including unmatched raw-file labels and conflicting
preparation/search descriptions.

Excel date-styled cells retain their original serial and style instead of guessed
gene names. Table3 repeated labels are never collapsed into a dictionary; numbers
in those slots do not establish a common unit, experiment, sample or KO target.
Other publisher tables remain in the original ZIP and are not projected as
activity-ranked or antimicrobial modeling data.

This is a source package, not a qualified training cohort. Baseline candidate
selection, confidence/matching policy, full-parent reference recovery, precursor
pairing, dependence groups and prospective evaluation require a separate declared
contract. A zero search intensity is not a measured biological absence. No output
is asserted to measure the complete free intracellular peptide pool or molarity.

## Baseline candidate review

The registered `goldberg2025_baseline_candidates_v1` recipe exports every original
peptide-intensity decision and a candidate-only view. It selects positive finite
canonical sequences with explicitly blank source flags and human-only source
header annotations in the original control acquisitions. Publisher inclusion is
an exposed axis, never a filter. Every peptide-level aggregate and other run
intensity retains its outside-scope disposition.

```python
from hitlist.evidence_recipe import materialize_recipe

materialize_recipe(store, {
    "schema_version": 1,
    "name": "baseline_candidate_review",
    "registered_recipe": "goldberg2025_baseline_candidates_v1",
}, "new-candidate-package")
```

`typed/decisions.parquet` contains all source peptide-intensity decisions;
`typed/candidates.parquet` contains the candidate subset. The export preserves
original linear quantities, positive-cell log10 values, confidence, identification
method, contributor IDs, parent alternatives and sample context. Its source tables
are snapshotted and its verifier recomputes the decisions, including after output
inventory rehashing. It creates no split or model.

Candidate status is **not** final training admission. Confidence/matching policy,
full search-reference support, physical precursor pairing and processing-model
eligibility remain separately labeled. A downstream experiment must declare its
endpoint and policies before fitting; these candidates are not certified free
intracellular concentrations.

## Conditional observed composition

Registered recipe `goldberg2025_observed_baseline_composition_v1` retains the
complete baseline decision universe and adds a stricter conditional signal cohort:
source peptide PEP in [0,.01), direct per-run MS/MS, one supplied accession, and
at least three distinct positive observed sequences per accession/acquisition.
Duplicate eligible sequences within a group are refused. Publisher selection is
not a filter. All original peptide/accession nodes, including excluded bridges,
define stable dependency components before the fixed 60/20/20 hash split.

The export includes decisions, component membership, group support, source-linked
examples, and a separate label-free covariate table. Targets are centered log10
intensity and intensity fractions within observed groups. Weights give equal mass
to components, then groups, then peptides. The declared support gate is reported,
including for insufficient cohorts; it never triggers a resplit. Verification
reconstructs all outputs from the bound source tables, even after rehashing.

These targets describe captured, observed signal in A549. They do not identify
free-peptide concentrations, unobserved mass, exact origins, or paired precursor
amounts. Sequence totals may aggregate molecular forms. Original source/confidence/
reference/material uncertainties remain separate fields.
