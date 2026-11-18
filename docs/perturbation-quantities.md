# Original perturbation quantities

Hitlist 2.1.17 exposes the original Stopfer 2020 melanoma treatment/control
workbooks through the shared evidence query and recipe APIs. Source: [PMID
32488085 / PXD017407](https://www.nature.com/articles/s41467-020-16588-9),
Supplementary Data 3 and 5. The twelve comparisons cover four cell lines at
1/10 uM palbociclib or 10 ng/mL IFNG for 72 hours, with three reported biological
replicates per arm.

```python
from hitlist import (
    EvidenceStore,
    fetch_perturbation_sources,
    write_perturbation_quantity_package,
    verify_recipe_package,
)

assets = fetch_perturbation_sources("/data/stopfer-originals")
write_perturbation_quantity_package(
    "/data/stopfer-originals", "/data/stopfer-evidence", assets=assets,
)
store = EvidenceStore.open("/data/stopfer-evidence")
quantities = store.query(
    "perturbation_measurements",
    select=["measurement_id", "quantity_kind", "quantitative_value",
            "source_locator", "form.peptide_sequence", "sample.arm_role"],
    joins=["form", "sample"],
    where={"quantity_kind": "hipmhc_normalized_reporter"},
    limit=None,
).table
store.materialize(
    {"schema_version": 1, "name": "stopfer-relative-response",
     "registered_recipe": "stopfer2020_unpaired_reporter_contrast_v1"},
    "/data/stopfer-relative-response",
)
verify_recipe_package("/data/stopfer-relative-response")
```

Acquisition uses Hitlist's existing datacache downloader, with pinned byte size
and SHA-256 checks. Source packages preserve exact workbook bytes, original
cells/formulas/cached values, row dispositions, conditions, channels, molecular
forms, file-map claims, quantities and reported statistics. No workbook is
recalculated. Destinations must be new directories; failures retain diagnostic
staging directories. Both source and recipe verification replay the original
workbooks, including when verification is offline.

The recipe writes `typed/contrasts.parquet` and `typed/decisions.parquet`.
Its target is `log2(mean treated / mean DMSO)` using the six positive finite
hipMHC-normalized channels from the same molecular form and plex. Counts and
sample standard deviations retain their units. Reported effects and p-values
remain separate fields. The optional `plex_ids` selection accepts the exact
IDs in `perturbation_plexes`: omitted/`None` means all, `[]` means none, and
unknown IDs raise an error. Every source candidate retains its decision,
including excluded aggregates, pooled precursor AUC rows and unselected plexes.

For Presto, retain `form_id`, `source_record_id`, `plex_id`, molecular modifications,
source project/reference, and contributing measurement IDs. Group this entire
source project together for split assignment, reconcile it with IEDB/CEDAR and
other copies of the same experiment, and apply protected-cohort/affinity overlap
checks before admitting rows to TRAIN. No row is automatically assigned to TRAIN.
Sample genotypes supply allele candidates, not experimentally assigned peptide
restrictions. Numbered channels do not establish cross-arm specimen pairing or
physical independence between different plexes.

Data 3/5 contain relative reporter signals. They do not provide peptide-specific
absolute copies/cell calibration. Keep raw, hipMHC-normalized and mean-adjusted
signals distinct. Methods report Percolator q <= 0.05 for labeled samples and
separate within-condition normalization for large shifts; neither strict 1%
identification filtering nor cancellation of condition-dependent MS/extraction
effects is established. Missing values and zeros are not nondetection negatives.

Four 10-uM sheets contain erroneous 1-uM channel headers; the qualified dose
uses the original title and file map while preserving those headers. The file
map's obsolete Table 2.x numbers are retained. The IPC298 aggregate is excluded,
and the eight phospho-[S] rows keep their molecular identity. These source
discrepancies are tracked in [#924](https://github.com/pirl-unc/hitlist/issues/924).
The independent source-qualification CI job reconciles original cells and all
reported effects using a separate workbook reader; its passing result is required
before release. Broader DLBCL and other quantitative acquisition remains in
[#901](https://github.com/pirl-unc/hitlist/issues/901).
