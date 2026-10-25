# Intact-degradomics time courses

The standalone `intact_degradomics` package preserves PXD058121 original
processed workbooks and publisher supplements (DOI 10.1073/pnas.2419607122).
Revision `pxd058121_intact_degradomics_v1` uses package schema 26/catalog 32.
Its 14 original assets have pinned hashes, sizes, URLs and individual rights
scopes in `hitlist/data/evidence_revisions/intact_degradomics.json`.

```python
from hitlist.intact_degradomics import write_intact_degradomics_package
from hitlist.evidence_store import EvidenceStore

write_intact_degradomics_package("original-assets", "new-package")
store = EvidenceStore.open("new-package")
assigned = store.query(
    "intact_assignments",
    select=["peptide_sequence", "start_1based", "end_1based",
            "mass_link_status", "quantity.quantitative_value",
            "acquisition.time_minutes", "acquisition.organ_label"],
    joins=["quantity", "acquisition"],
    limit=None,
).table
```

The writer snapshots complete original assets, preserves literal OOXML cells,
materializes typed tables and verifies a deterministic source replay before
publishing the directory. Reopening repeats that verification. Source hashes
and typed rows must both agree; rehashing a changed Parquet table cannot certify
it. Existing outputs are never overwritten; failed staging/replay directories
remain available for diagnosis.

| Relations, all prefixed `intact_` | Meaning |
|---|---|
| `assets`, `source_records`, `source_cells` | Original bytes, rows, coordinates, XML types/styles, cached values and formula attributes; no recalculation |
| `acquisitions`, `source_views` | Deposited RAW file identities and separate workbook/figure views |
| `input_masses` | Measured mass observations, including unassigned masses and missing source IDs |
| `assignments` | Each proposed sequence/span interpretation, mass errors, modifications, ambiguity and qualified input-mass link |
| `supplement_rows`, `quantities` | Distinct envelope, fit-factor, assignment and publisher quantities; raw values, missing/zero/error states and source units |
| `reconciliations` | Repeated views and publisher/deposit comparisons with explicit conflicting fields |
| `assertions`, `material_contexts`, `material_links` | Source-scoped protocol/context assertions and shared material graph with unresolved physical identity |

These are human alpha-synuclein substrate and mouse-organ-derived purified 20S
preparations, not MHC pulldowns or intact APC measurements. Substrate species,
enzyme species and expression host are separate roles. Organs are reported from
one animal; R1–R3 labels do not establish independent specimens or reactions.
MHC class/allele and APC cell type are inapplicable. Numeric machinery subunit
amounts are unavailable; organ labels must not become invented PSMB quantities.

Input masses, candidate assignments and repeated publisher views are different
grains. Nine F1/F2 views reuse the same acquisitions. Multiple assignments may
refer to one mass; multiple masses may refer to one peptide. No implicit sum,
best-assignment selection or training eligibility is applied. Publisher identity
matching uses sequence/span/type and exact observed mass when necessary; measured
intensity never selects a match. Conflicting quantities remain separate.

The pinned interpretation contract handles the extra unlabeled identifier in
F2's 40-minute R1 assignment block, the absent input IDs in F3's brain-zero R2
block, the abbreviated multiplicity header, and lexical integer ID equivalence.
Missing IDs are never inferred from row order. Missing spleen 30-minute R3 stays
missing. Reconciliation does not grant new identification confidence or FDR.

Nominal zero already contains products. Source intensity, deconvoluted sums and
charge-normalized envelope signals are not molar concentrations or TIC. Unknown
peptide response and run scaling remain. Missing peptides are not zero/negative
labels. A supervised kinetic recipe and prospective split/model contract must
be qualified separately before training. One substrate and one animal cannot
establish independent new-protein or APC generalization.
