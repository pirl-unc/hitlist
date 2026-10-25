# Native intracellular peptide evidence

`native_peptidomics` exposes the original PXD018722 Hi20S intracellular peptide
extract through the shared `EvidenceStore`. It uses package schema 27 and catalog
35. The pinned source is the control/PSMD2-knockdown HEK293T panel in
[Sahu et al.](https://doi.org/10.1038/s41467-021-26427-0), deposited in
[PRIDE](https://www.ebi.ac.uk/pride/archive/projects/PXD018722).

This is native extraction with MS detection. It is not MHC elution, proteasome
capture, a professional APC cohort, or calibrated intracellular concentration.
The adapter does not select a training cohort or infer negative examples from
undetected peptides. It complements the existing TMAB evidence family; it does
not silently migrate or reinterpret older packages.

## Build and query

Supply the original assets named in
`hitlist/data/evidence_revisions/native_peptidomics.json`. The contract pins their
exact hashes, source URLs, archive-member/range evidence and interpretation.
There is no implicit download. Source snapshots use independent filesystem clones
where supported. A complete archive checksum is not claimed for partially fetched
ZIP members. The writer refuses existing destinations, retains failed staging,
and verifies deterministic source replay before publishing a package.

```python
from hitlist.native_peptidomics import write_native_peptidomics_package
from hitlist.evidence_store import EvidenceStore

write_native_peptidomics_package("original-assets", "native-package")
store = EvidenceStore.open("native-package")

sequence_signal = store.query(
    "native_quantities",
    select=[
        "peptide.peptide_sequence",
        "acquisition.sample_label_raw",
        "quantitative_raw",
        "quantitative_value",
        "value_status",
        "quantity_form_scope",
    ],
    where={
        "observation_kind": "peptide_sequence",
        "quantity_kind": "source_intensity",
        "experiment_raw": "Cont1",
    },
    joins=["peptide", "acquisition"],
    limit=None,
).table
```

`iter_query` streams the same public relations. `store.catalog()` describes the
available fields, source registry and relationships. Opening a package verifies
the original assets and replays the typed tables; reserve space for verification.

## Evidence grains

| Relations (`native_` prefix) | Meaning |
|---|---|
| `assets`, `source_records`, `source_cells` | Original bytes, ordered text rows/XML nodes/reference records and literal publisher OOXML cells, including styles and cached formulas |
| `acquisitions`, `material_contexts`, `material_links` | Six reported runs, control/KD study arms and precision-qualified context membership |
| `peptides`, `forms`, `identifications` | Sequence identities, modification-specific identities and feature/matching records with original native IDs |
| `quantities` | Original signal at its sequence, form or feature grain; total and per-run values remain separate |
| `references`, `origins`, `declared_parents` | Complete deposited reference records, all literal/I-L-equivalent candidate spans and every source-reported parent alternative |
| `dependency_groups` | All-source connected components formed before supervised selection |
| `assertions`, `reconciliations` | Source-scoped metadata, explicit unresolved claims, identifier checks and retained discrepancies between source views |

Raw sequence intensity and LFQ intensity are different quantities. Sequence
intensity can aggregate modified forms; feature-level MSMS, MULTI-MSMS,
MULTI-MATCH and MULTI-SECPEP records remain distinguishable. Rounded feature/form
sums do not replace the original sequence-level values. Zero has an unknown
observation cause, and missing is a separate status. Units are arbitrary source
signal, never molarity.

The publisher workbook and original text are views of the same peptide entities,
not duplicate training measurements. Date-formatted gene cells are retained with
their original Excel representation and corresponding text annotations; gene names
are not reconstructed from dates.

## Context and interpretation

The article's triplicate-data and independent-experiment descriptions are preserved
at their figure scopes. Assignment of each deposited acquisition to a physical
specimen or independent experiment remains unresolved; labels such as `Cont1`
and `KD1` do not establish biological pairing. The source describes knockdown,
not a complete knockout or a numeric remaining PSMD2 activity. Its reported 20S
complex fraction is a separate cell-line characterization.

Every deposited reference and every origin alternative is retained. The documented
P22222/NS3CCNB1_HUMAN engineered construct remains explicitly marked even though
its header carries human taxonomy. Neither taxonomy nor a leading-parent label
proves an endogenous unique origin. Reporter possibilities, I/L alternatives,
unresolved contaminants and all declared parents participate in dependency groups;
the registered recipe below makes its eligibility decisions explicitly. Reporter sequences
are provenance/exclusion evidence, not predictive targets.

Matched precursor abundance, recovery calibration, run-specific treatment mapping
and APC transfer are not established by this package. A model-facing recipe and
biological validation remain separate work tracked in
[Hitlist #837](https://github.com/pirl-unc/hitlist/issues/837).

## Same-parent response recipe

`pxd018722_native_contrast_v1` materializes an explicitly conditional observed-signal
endpoint through the common recipe API:

```python
store.materialize(
    {
        "schema_version": 1,
        "name": "native-same-parent-response",
        "registered_recipe": "pxd018722_native_contrast_v1",
    },
    "native-contrasts",
)
```

An admitted peptide must have a standard 8–25-residue sequence, source PEP at most
0.01, no reverse/contaminant flag, only literal human nonengineered origins from
one complete parent sequence, and no unresolved or contradictory declared parent.
Its six original raw sequence intensities must all be finite and positive. A
parent needs at least two admitted peptides. The lowest fixed sequence/parent
hash selects its reference; signal magnitude does not select the reference.

For peptide i and reference j, the target is the KD mean of
`log10(I_i) - log10(I_j)` minus the control mean. The six individual run ratios
and twelve source-quantity contributions remain available. This yields one
contrast, without asserting paired specimens or six independent biological subjects.
Shared precursor abundance/run loading and fixed peptide response cancel only
under the stated multiplicative observation assumptions. Condition-dependent
extraction, ion suppression or modification composition can still affect the target.

The full source dependency component determines the split before eligibility.
Excluded peptides cannot break a component. Components, then parents within a
component, then contrasts within a parent receive equal weight in each split.
Training/validation/test fractions and minimum support counts are fixed in the
registered contract. `coverage.qualified_for_screen` reports whether those support
requirements pass; consumers must refuse a fit when they fail, rather than change
the split seed or silently loosen eligibility. Successful materialization alone
does not establish sufficient support or predictive accuracy.

The output retains source snapshots/manifest, the contract, all-peptide decisions,
admitted parents, examples, source contributors and all original component members.
The `typed/covariates.parquet` relation omits numeric targets and acquisition log
ratios; examples and contributors contain labels and must be accessed by split.
`verify_recipe_package(..., source_package=...)` replays every decision, target,
split and weight from pinned source tables and rejects changed derived values
even when their artifact hashes have been refreshed. The writer refuses overwrites
and retains failed staging. This recipe does not fit a model, convert zeros to
negative labels, or recover an unobserved complete intracellular distribution.
