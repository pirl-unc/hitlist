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

The four entries are 62 deposited acquisition rows. This is an assay/source
inventory, not 62 independent specimens, donors, biological replicates, or
training examples. The known Goldberg PXD052314 MAPP source remains preserved
in its own original-source package. Its distinct search/publisher layout is the
next candidate for an explicit canonical projection, tracked in
[#835](https://github.com/pirl-unc/hitlist/issues/835) after source recovery in
[#825](https://github.com/pirl-unc/hitlist/issues/825).
Adjacent PXD009360/PXD009181/PXD009112 deposits are tryptic/conventional
proteomics, not captured-fragment MAPP; they remain separately typed evidence.

## Layout

The stable base relations are:

| Evidence type | Grain |
|---|---|
| `mapp_source_record` | one original source-table row |
| `mapp_peptide`, `mapp_group` | one deposited peptide or protein-group row |
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

Build a current external successor with all three source adapters together;
the controlled adapter contributes two PXD datasets.

```sh
hitlist evidence append-sources PREDECESSOR OUTPUT \
  --source-root pxd009092=PXD009092_ASSETS \
  --source-root wolflevy2018_controlled=CONTROLLED_ASSETS \
  --source-root pxd009438=PXD009438_ASSETS
```

Then one query pattern spans all admitted sources:

```python
from hitlist.evidence_store import EvidenceStore

store = EvidenceStore.open("OUTPUT")
result = store.query(
    "mapp_quantity_result",
    select=[
        "source_dataset",
        "result_id",
        "acquisition.condition_text",
        "evidence_summary.n_direct_msms_evidence",
    ],
    joins=["acquisition", "evidence_summary"],
    limit=None,
)
```

`mapp_acquisition` joins to `mapp_acquisition_context` and the material graph.
No query turns a study arm, raw-file label, source sample label, or repeat
number into a donor, specimen, aliquot, or cross-modal pairing. A source zero,
missing intensity and unreported intensity remain separate quantity states; no
canonical projection turns them into biological absence or a presentation label.

This is a deliberate breaking release. The old `processing_*`, `pbmc_mapp_*`,
`mapp_channel`, and `external_*` MAPP relation names are removed rather than
kept as aliases. Consumers should migrate to the `mapp_*` relations above and
filter `source_dataset` explicitly where an endpoint requires one source.
