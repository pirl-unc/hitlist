# Declared training recipes

`hitlist.evidence_recipes` materializes registered endpoints from a verified
shared evidence generation. Discover the captured-fragment task through
`hitlist.evidence_catalog.describe_task("captured_fragment_conditional_composition")`;
its `recipe_ids` identify the executable contracts.

```python
from hitlist.evidence_recipes import (
    list_training_recipes, write_training_recipe, verify_training_recipe,
)

contracts = list_training_recipes()
manifest = write_training_recipe(
    external_package,
    new_destination,
    recipe_id="mapp_captured_composition_min2_v1",
)
verify_training_recipe(new_destination)
```

Unknown recipes and existing destinations fail explicitly. The writer verifies
the source package before reading typed relations and publishes into a new
directory after all files are complete. It emits the recipe contract, copied
source manifest, every peptide's admission/exclusion reasons, full parent
sequences, original accessions, all peptide positions, examples, measurement/run/
evidence contributors, coverage and file hashes. The output verifier checks
inventory, bytes and contract; independent source replay is a separate check.

Both initial recipes use human HEK293 MAPP PXD009092, untreated versus TNF+IFN,
with canonical peptides/parents, no reverse or contaminant flag, one source group,
two positive filtered runs and a direct-MS/MS-bearing run in each condition,
reported peptide PEP <=0.01, one full reference sequence and source-leading
coordinate/flank agreement. Peptide PEP selection does not resolve the reported
global-FDR discrepancy. Duplicate accessions do not create extra measurements.

`mapp_captured_composition_v1` requires three eligible peptides per parent.
`mapp_captured_composition_min2_v1` includes two-peptide sets, the minimum for a
nontrivial composition. Both use the same all-source connected components and
the original split salt. Extending admission never reseeds existing observations.
The graph includes excluded peptides and joins shared full sequences and source
protein groups. It constrains exact overlap, not homology/specimen independence.

Targets normalize published, nonimputed median-filtered intensities within each
parent and condition. Original log10 values remain, and all three run contributors
survive, including filter zeros. The normalization removes a common parent-wide
scale, but peptide-dependent capture/MS response remains. This endpoint predicts
relative captured signal conditional on an observed set; mass outside the set is
unidentified and unreported peptides are not negatives. It is not a calibrated
intracellular concentration or professional-APC validation endpoint.

These recipes are a scoped foundation for [the unified query/recipe work #739](https://github.com/pirl-unc/hitlist/issues/739),
not a completed multi-modality major release. They require the MAPP schema2 source
adapter in [PR758](https://github.com/pirl-unc/hitlist/pull/758). Independent precursor
estimates and exact parallel-proteome pairing remain in [#757](https://github.com/pirl-unc/hitlist/issues/757).
