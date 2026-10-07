# Assay modality and MS evidence

Hitlist 1.66.0 requires positive evidence of an MS assay before a record enters
`observations.parquet`. A false `is_binding_assay` flag alone is insufficient.
This corrects [#644](https://github.com/pirl-unc/hitlist/issues/644), including
PAGE4 fluorescence and MAGE crystallography/thermal-stability records.

| Partition | Admission | Reader |
|---|---|---|
| `observations.parquet` | Explicit mass-spectrometry method with a compatible ligand response, or a curated MS-only supplement without a method; no explicit negative outcome | `load_observations` / `load_ms_observations` |
| `binding.parquet` | Binding, affinity, kinetic or stability evidence | `load_binding` |
| `other_assays.parquet` | Structural, non-MS ligand, unknown/conflicting assays, and negative MS results | `load_other_assays` |

The shared `assay_annotations` function records `assay_modality`,
`assay_modality_source`, `is_binding_assay`, and `is_ms_observation`.
Modality and outcome are independent: a negative MS result still has modality
`ms`, but is not an observed positive ligand. Positive-High/Intermediate/Low
results and binding-related comments cannot override an explicit MS method.

Structured methods/responses take precedence. Fluorescence binding and thermal
stability remain binding evidence; crystallography and electron microscopy
remain structural evidence. Edman degradation, coelution and T-cell recognition
are non-MS ligand evidence. Conflicting MS methods and response endpoints are
unresolved. Missing or unrecognized methods never imply MS. Legacy binding
comments/tiers can establish a separately identified binding fallback.

This distinction follows IEDB's separation of binding assays, ligand elution,
and structure assays; see its [curation manual](https://curationwiki.iedb.org/wiki/index.php/Curation_Manual2.0)
and [assay workshop](https://help.iedb.org/hc/en-us/article_attachments/10126535621659).
These annotations describe the evidence modality, not vaccine efficacy or safety.

```python
from hitlist import assay_annotations, load_other_assays, load_all_evidence

record = assay_annotations(
    "Positive", "", "cellular MHC/direct/fluorescence", "qualitative binding"
)
assert record["assay_modality"] == "binding"
assert not record["is_ms_observation"]

other = load_other_assays(peptide="GVYDGREHTV")
all_evidence = load_all_evidence(peptide="GVYDGREHTV")
# evidence_kind is ms, binding, or other; assay_modality gives the detail.
```

All three partitions retain raw method/response/outcome fields, source
contributors and access to the shared peptide-mapping sidecar. Ordinary
MS/binding training exports retain their existing `ms`, `binding`, and `both`
options. CTA bundles include non-MS records in `excluded_observations.parquet`
with provenance and never count them as presentation. New CTA manifests record
`ms_policy_version=2`; standalone verification of older bundles uses their
original policy and does not reclassify them as current-policy exports.

## Upgrade and rebuild

Run `hitlist build observations` after upgrading. Observation artifact version 9
invalidates earlier builds, and the new partition participates in cache and
contributor-integrity checks. Rebuild affected downstream exports and designs.

Legacy MS/binding readers apply the new structured admission rules in Arrow,
before pandas materialization, including projected reads such as
`columns=["peptide"]`. They cannot recover rows discarded by an earlier build or
move old rows between partitions: a rebuild is required for complete corrected
MS, binding and other-assay indexes. `load_all_evidence` includes the other-assay
partition when present. Raw scans retain all modalities; consumers must use
`is_ms_observation` for positive MS admission. `hitlist report --from-csv` uses
the same admission rule as built-index reports.

Publication of a coherent generation of every build artifact is tracked
separately in [#645](https://github.com/pirl-unc/hitlist/issues/645).
