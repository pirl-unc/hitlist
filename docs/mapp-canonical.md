# Canonical captured-fragment MAPP package

Captured-fragment MAPP is one source-conserving package and one query API. The
public relations are not named after a deposit or a paper: every row instead
has `source_dataset` and `source_dataset_precision`.

| Admitted source dataset | Source adapter | Acquisition rows | Scope retained |
|---|---|---:|---|
| PXD009092 | `pxd009092` | 18 | HEK293 captured fragments, source arms/runs and publisher workbooks |
| PXD009508 | `wolflevy2018_controlled` | 18 | capture-control and epoxomicin panel |
| PXD007263 | `wolflevy2018_controlled` | 12 | cytokine-label time course |
| PXD009438 | `pxd009438` | 14 | CTRL/LUPUS source samples; PBMC/metadata conflict retained |
| PXD052314 | `goldberg2025` | 36 | A549 source acquisitions across MAPP, whole-cell and proteasome-composition views; 12 nonphysical source-label contexts, Table5 condition claims, specimen/aliquot mapping unresolved |

The five entries are 98 deposited acquisition rows. This is an assay/source
inventory, not 98 independent specimens, donors, biological replicates, or
training examples. PXD052314's 36 rows include distinct source assay pools;
only source-backed pool labels distinguish them. The original Goldberg source
package remains independently verifiable alongside its canonical projection.
Adjacent PXD009360/PXD009181/PXD009112 deposits are tryptic/conventional
proteomics, not captured-fragment MAPP; they remain separately typed evidence.
PXD009360 exposes the same source-assertion/material-context/material-link
field pattern in its `precursor_*` package, but it has no source-backed
condition, specimen, aliquot or acquisition-level MAPP pairing. PXD009181 and
PXD009112 remain audit-backed companion candidates, not MAPP rows or paired
precursor data.

PXD052314 is admitted at the complete pinned-asset/source-table boundary, not
as a hand-selected peptide list: six assets include its deposition workbook,
original MaxQuant archive, publisher supplement, project/file metadata and
community SDRF. The projection retains all three sets of 12 deposited raw-file
rows, Table5 MAPP search quantities and its distinct whole-cell/proteasome
views, publisher cells/quantities, and every ordered Table3 pooled-value slot.
This expands MAPP coverage without asserting that the parallel assay labels or
repeated Table3 labels identify the same specimen, aliquot, perturbation, or
free intracellular peptide pool. PXD052314 Table5 conditions are available as
source assertions only after a publisher header's literal raw-file run token
and deposited sample-label suffix identify one acquisition. The 12 shared
source-label contexts facilitate source-scoped condition queries but do not pair
the three assay views physically.

## Layout

The stable base relations are:

| Evidence type | Grain |
|---|---|
| `mapp_source_asset` | one pinned original source asset |
| `mapp_source_record` | one original source-table row |
| `mapp_peptide`, `mapp_peptide_form`, `mapp_group` | one deposited peptide, modification-specific form, or protein-group row |
| `mapp_acquisition` | one source acquisition |
| `mapp_identification` | one deposited peptide identification |
| `mapp_quantity_result`, `mapp_quantity_measurement` | one peptide/acquisition result and one of its representations |
| `mapp_derivation_input` | one replay/derivation edge |
| `mapp_source_assertion` | one source-scoped assertion |
| `mapp_material_context`, `mapp_material_link` | one qualified material context or context link |
| `mapp_label_alias` | one exact quantity-column/acquisition-label claim |

`mapp_acquisition_context` holds capture, treatment, time and source scope
fields; `mapp_quantity_evidence_summary` holds direct/matched evidence counts.
`mapp_publisher_cell` and `mapp_publisher_quantity` preserve workbook-specific
material without making it a property of every source. Reference mappings have
the corresponding `mapp_*` names. `mapp_source_table` is the mapping ledger:
it identifies the retained source-owned table, schema fields and row count for
every canonical projection.

PXD052314 adds same-grain extensions, rather than nullable Goldberg columns to
the other sources' base rows: `mapp_*_source_detail` preserves native search
rows, `mapp_source_reference` preserves FASTA-header assertions,
`mapp_search_quantity_detail` preserves source quantity fields,
`mapp_publisher_*_source_detail` and `mapp_publisher_reconciliation` preserve
publisher XML/reconciliation evidence, and `mapp_pooled_quantity_slot` retains
ordered Table3 slots. These extensions are data, not resolved donor, specimen,
aliquot, condition, perturbation or training labels.

The source-detail relations expose the same query contract as every other
evidence relation: their shared logical fields are selectable directly and
their named `canonical_*` links return the matching canonical relation. For
example, `mapp_search_quantity_detail.canonical_measurement` is an exact
one-to-one source-cell/measurement link; a publisher reconciliation link is
optional and remains an asserted match rather than a new identity. Pooled
slots link only to their original source record, never to an acquisition,
specimen or aliquot.

Every canonical table has the same first two columns:

```text
source_dataset             # exact PXD accession when the source row carries one
source_dataset_precision   # exact_project or source_adapter_scope_multiple_projects
```

The latter is important for shared controlled-panel reference tables: a
reference row shared by PXD009508 and PXD007263 is not silently duplicated or
assigned to one project. Original source tables/assets remain intact under their
adapter-owned `external/<source>/` paths. The canonical projections are written
to `mapp/typed/` and replayed row-for-row from those source tables during
verification.

## Query

Build a current external successor with all four source adapters together;
the controlled adapter contributes two PXD datasets.

```sh
hitlist evidence append-sources PREDECESSOR OUTPUT \
  --source-root pxd009092=PXD009092_ASSETS \
  --source-root wolflevy2018_controlled=CONTROLLED_ASSETS \
  --source-root pxd009438=PXD009438_ASSETS \
  --source-root goldberg2025=PXD052314_ASSETS
```

Then one query pattern spans all admitted sources:

```python
from hitlist.evidence_store import EvidenceStore

store = EvidenceStore.open("OUTPUT")
result = store.query(
    "mapp_quantity_measurement",
    select=[
        "source_dataset",
        "measurement_id",
        "result_id",
        "value_status",
        "quantitative_value",
    ],
    limit=None,
)
```

Source-specific evidence remains uniformly queryable when needed:

```python
search_cells = store.query(
    "mapp_search_quantity_detail",
    select=[
        "source_dataset",
        "measurement_id",
        "quantitative_raw",
        "value_status",
        "canonical_measurement.quantitative_value",
    ],
    joins=["canonical_measurement"],
    limit=None,
)
```

`mapp_acquisition` joins to `mapp_acquisition_context` and the material graph
when a source quantity has an acquisition identity. PXD052314 condition claims
use the same `mapp_source_assertion` relation as the rest of the API: filter
`source_scope=publisher_table5_readme_via_exact_header_and_deposition_raw` and the
`reported_condition_*` predicates, then join `subject_acquisition` and traverse
its qualified material links when a nonphysical source-label grouping is useful.
Assertions whose subject is not an acquisition retain a null optional link.
Aggregate or pooled source quantities retain a null acquisition link; they must
not be joined by matching their labels. No query turns a study arm, raw-file
label, source sample label, or repeat number into a donor, specimen, aliquot,
or cross-modal pairing. A source zero, missing intensity and unreported
intensity remain separate quantity states; no canonical projection turns them
into biological absence or a presentation label.

This is a deliberate breaking release. The old `processing_*`, `pbmc_mapp_*`,
`mapp_channel`, and `external_*` MAPP relation names are removed rather than
kept as aliases. Consumers should migrate to the `mapp_*` relations above and
filter `source_dataset` explicitly where an endpoint requires one source. The
expanded external envelope is package schema 27/catalog schema 17; canonical
MAPP v1 remains replay-verifiable for historical packages, while new packages
expose the v2 source-detail relations.
