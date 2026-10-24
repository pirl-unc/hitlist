# Original lymph and DC peptide quantities

The `nanaware_lymph` evidence family preserves original Nanaware2024 human native
lymph and mouse dendritic-cell I-Ab quantities. Its reviewed source revision is
`nanaware2024_lymph_quantities_v1` (package schema24, catalog29).

```python
from hitlist.nanaware_lymph import write_nanaware_lymph_package
from hitlist.evidence_store import EvidenceStore

write_nanaware_lymph_package("original-source-directory", "new-package")
store = EvidenceStore.open("new-package")
values = store.query(
    "lymph_quantities",
    select=["measurement_id", "peptide_sequence", "quantitative_value", "value_status"],
    limit=None,
).table
```

The fixed receipt in `data/evidence_revisions/nanaware_lymph.json` identifies exact
original assets, block boundaries, headers, sample channels and source-scoped
assertions. The writer snapshots verified original bytes; reopening independently
replays the canonical tables and detects modified rows even if file hashes were
updated. It never overwrites an existing destination.

Nine typed relations expose assets, original block rows, contexts, exported sample
labels, peptide-form observations, measurement cells, deposited reference sequences,
identification alternatives and contextual assertions. Logical same-grain fields
use the shared catalog; remaining original fields are queryable through `native.*`.
Explicit links connect each quantity to its original row, observation, sample and
context. No cross-modality specimen relationship or certain parent origin is inferred.

Only S6(a) A:Q and the two separate S4(e) search blocks are projected as quantities.
Other workbook contents remain original source assets. Areas and intensities remain
distinct; repeated charges/forms and search outputs are not merged. Missing, zero,
nonfinite, unsupported and formula cells retain explicit statuses. Source confidence
is retained without imposing a new threshold or treating nonobservations as negatives.

The human material is extracellular native lymph from IBD subjects. Its Sample1
donor/pooling identity is unresolved; the adjacent protein summary is derived from
the same peptides and does not provide independent precursor abundance. Primary
methods and community SDRF disagree about the instrument. Mouse DCs were expanded
with B16-FLT3-L treatment and must not be labeled untreated baseline APCs. The source
description and exported channel counts disagree about replicate/control coverage;
both remain recorded. Alternative search algorithms reuse the same channel IDs.
No gene-loss status or paired intracellular precursor concentrations are supplied.

Mouse S4(n) no-enzyme searches reuse acquisitions described as enzymatically digested
proteomics and are not admitted as native processing supervision. A search parameter
does not establish preparation chemistry. MHC-bound DC peptides are distinct from
free intracellular peptide quantities. Deposited DBSequence and PeptideEvidence
records preserve reference alternatives, not measured precursor origins.

This revision provides **source evidence only**. It advertises no training recipe,
concentration calibration, detection negatives, measured binding registers, or model
quality result. Endpoint selection, sequence-reference occurrence resolution,
confidence/aggregation policy, dependence groups and prospective splits require a
separate registered recipe before fitting.

## Conditional representative-feature composition

`nanaware2024_observed_composition_v1` is a registered recipe available from
Hitlist2.0.28. The older source package's source-only registry assertion describes
its original publication; the query API's registered-recipe inventory describes
current capabilities without rewriting that historical manifest.

The recipe retains positive original area measurements for chemistry-compatible
representative features, uniquely supported original parent sequences and groups
of at least three peptides. It selects source features by confidence before area,
retains every exclusion, connects all source dependencies before filtering, and
emits deterministic component splits plus centered log10 signals and fractions.
Original same-study identification alternatives enter origin ambiguity; original
reported length/start conflicts do not shorten sequences. Source context and
sample identifiers remain linked through snapshotted relations.

This conditional endpoint is neither total peptide molarity nor a free APC pool.
The DC database/library blocks share acquisitions; human lymph is extracellular.
An experiment must check independent-component support and model the observation
and selection limitations explicitly. See `tasks/lymph-composition-recipe-spec.md`
for the prospective experiment and exact contract; the registered verifier
recomputes all decisions from pinned source relations even after inventory rehash.
