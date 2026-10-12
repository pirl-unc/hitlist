# Assay modality and MS evidence

Hitlist 1.66.0 requires positive evidence of an MS assay before a record enters
`observations.parquet`. A false `is_binding_assay` flag alone is insufficient.
This corrects [#644](https://github.com/pirl-unc/hitlist/issues/644), including
PAGE4 fluorescence and MAGE crystallography/thermal-stability records.

| Partition | Admission | Reader |
|---|---|---|
| `observations.parquet` | Established MS modality and a known positive outcome, or a blank outcome in a curated observed-MS supplement without a method | `load_observations` / `load_ms_observations` |
| `binding.parquet` | Binding, affinity, kinetic or stability evidence | `load_binding` |
| `other_assays.parquet` | Structural, non-MS ligand, unknown/conflicting assays, and negative or unestablished MS results | `load_other_assays` |

The shared `assay_annotations` function records `assay_modality`,
`assay_modality_source`, `is_binding_assay`, `is_ms_observation`, and `ms_outcome_status`.
Modality and outcome are independent: a negative MS result still has modality
`ms`, but is not an observed positive ligand. Positive-High/Intermediate/Low
results and binding-related comments cannot override an explicit MS method.

Starting in 1.70.1, a method alone does not establish detection (#665). Only
`Positive`, `Positive-High`, `Positive-Intermediate`, and `Positive-Low` qualify,
ignoring case and surrounding whitespace. A blank outcome qualifies only under
the curated observed-MS-supplement contract. Explicit unknown, inconclusive,
unrecognized and negative outcomes remain excluded, including in supplements.
`ms_outcome_status` records `positive_result`, `curated_ligand_table`, `negative`,
or `unestablished`, independently of modality. New CTA bundles retain specific
`negative_ms_outcome` and `unestablished_ms_outcome` exclusion reasons.

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
`ms_policy_version=3`; standalone verification of older bundles uses their
original policy and does not reclassify them as current-policy exports.
Policy 2 retains its nonnegative-outcome rule; policy 1 retains the original
modality heuristic. Successful historical replay does not mean an old bundle
passes current admission. Re-export evidence to apply the corrected policy.

## Upgrade and rebuild

Run `hitlist build observations` after upgrading. Observation artifact version 10
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
