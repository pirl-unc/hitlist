# Registered distribution recipes

`EvidenceStore.materialize` handles both row-wise recipes and registered distribution
tasks. Discover the latter through `store.catalog()["registered_recipes"]`; status
is computed from the relations actually present in the verified generation.

```python
from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package

store = EvidenceStore.open(current_package)
recipe = {
    "schema_version": 1,
    "name": "mapp-observed-composition",
    "registered_recipe": "mapp_captured_composition_min2_v1",
}
manifest = store.materialize(recipe, new_destination)
verify_recipe_package(new_destination, source_package=current_package)
```

The same JSON/YAML request works with `hitlist evidence materialize PACKAGE RECIPE OUTPUT`
and `hitlist evidence verify-recipe OUTPUT --source-package PACKAGE`. The canonical
manifest records `output_kind: registered_distribution`, the exact request, immutable
registered contract, source package hash/schema and source-peptide coverage. Distribution
examples retain their candidate sets; they are not independent scalar labels. The
JSONL relations preserve the prior experimental targets and source provenance.

Unknown recipes and existing destinations fail explicitly. The writer verifies
the source package before reading typed relations and publishes into a new
directory after all files are complete. It emits the recipe contract, copied
source manifest, every peptide's admission/exclusion reasons, full parent
sequences, original accessions, all peptide positions, examples, measurement/run/
evidence contributors, coverage and file hashes. The output verifier checks inventory, bytes, contract, coverage, source binding,
probability arithmetic, mappings and contributors; independent original-source replay
remains a separate check.

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
scale, including true changes in processing yield, while peptide-dependent capture/MS
response remains. This endpoint predicts
relative captured signal conditional on an observed set; mass outside the set is
unidentified and unreported peptides are not negatives. It is not a calibrated
intracellular concentration or professional-APC validation endpoint.

The MAPP schema2 adapter can be appended to a verified reference or adjudication
generation with `hitlist evidence append-sources PREDECESSOR_PACKAGE OUTPUT --source-root pxd009092=ASSETS`.
The schema-9 successor preserves every predecessor byte, source and modality; old
experimental packages and receipts are not relabeled. Independent precursor
estimates and exact parallel-proteome pairing remain in [#757](https://github.com/pirl-unc/hitlist/issues/757).

## Complete-run observation recipe

`mapp_complete_run_composition_v1` starts from the two-peptide recipe and requires
finite positive filtered quantities in all six UT/TNF+IFN runs, then at least two
complete peptides per parent. The source graph and partition salt are unchanged.
Publisher median targets are renormalized only over the remaining parent set;
they remain distinct from per-run measurements.

The additional `run_observations.jsonl` contains all six observations for every
base-recipe peptide, including ineligible peptides and filtered zeros. It preserves
the original channel and measurement records, acquisition/file and replicate labels,
source metadata, exact evidence contributors and explicit complete-peptide/parent
eligibility. Extra exclusion reasons remain in the all-source disposition ledger.
Prior target, contributor and disposition records are unchanged; the canonical
source/recipe envelope records the new generation separately.

This supports run-repeatability diagnostics without logging zeros or adding
pseudocounts. Complete-case selection changes the population. Acquisition labels
do not establish specimen pairing or a precise biological noise ceiling; no
filtered zero is promoted to biological absence.

The [PXD021013 relative MS1 recipe](proteometools-ms1.md) supplies conditional
within-run signals for intended synthesized products. It preserves synthesis/load
uncertainty and pool/sequence dependence; it is not a concentration calibration.
