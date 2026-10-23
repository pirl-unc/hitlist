# hitlist

Hitlist 2.0 is a verified evidence store for immune and nonimmune training data.
The [evidence store guide](https://github.com/pirl-unc/hitlist/blob/main/docs/evidence-store.md) describes one catalog, query
and recipe API across MHC, T-cell, bulk MS, expression, source claims and lineage.
Assay results, numeric measurements and contextual claims keep their own grains.

[![Tests](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/hitlist/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hitlist.svg)](https://pypi.org/project/hitlist/)

A curated, source-preserving evidence system for model training.

```python
from hitlist import EvidenceStore

store = EvidenceStore.open("/path/to/verified-evidence-package")
print(store.catalog()["schema_version"])
print(store.query("arm_factor", select=["arm_id", "factor_type", "factor_value"],
                  where={"factor_type": "knockout_genes"}, limit=5).table)
print(store.query("bulk_numeric_measurement",
                  select=["measurement_id", "quantity_scale", "quantitative_value"],
                  where={"quantity_scale": "abundance_percentile"}, limit=5).table)
store.materialize("training-recipe.yaml", "/path/to/training-output")
```

The matching CLI is `hitlist evidence catalog`, `hitlist evidence query`,
`hitlist evidence materialize` and `hitlist evidence verify`. Each command takes
the same package path. Queries expose logical fields when source meaning and
grain match; every physical typed column is also accessible as `native.<column>`.
Related values require named joins, and many-valued relations remain separate
evidence rows. The catalog reports direct, related and unavailable fields,
source scope, row counts and ingest requirements. A recipe writes accepted,
excluded and ambiguous rows with counts, reasons, lineage, contributor keys
and a hash-bound manifest. It never rewrites source evidence.
The reference-augmented package adds source-backed mouse species roles and
exact peptide-to-parent alternatives while retaining every original assay and
numeric cell.

## Start here

- [Evidence store guide](https://github.com/pirl-unc/hitlist/blob/main/docs/evidence-store.md): catalog, grain-safe queries,
  named relationships, recipes and verification.
- [Pinned source and count audit](https://github.com/pirl-unc/hitlist/blob/main/tasks/evidence-store-results/README.md): 53
  materialized evidence types, 32 reconciled baselines and field disposition.
- [Migration history](https://github.com/pirl-unc/hitlist/blob/main/docs/evidence-catalog.md): source generations and
  scientific decisions behind the current package.
- [Legacy 1.x README](https://github.com/pirl-unc/hitlist/blob/main/docs/legacy-1x.md): historical indexes and commands.
  Its sample-level and training-readiness descriptions were superseded by
  the verified corpus audits and should not be used as current labels.

## Breaking release

2.0.0 removes several top-level loaders, exporters and static catalog helpers.
Open a verified package through `EvidenceStore` for current evidence requests;
use the catalog to see each field's actual readiness and required assay fields.
The original source records and typed relation files remain in the verified
evidence package; older module-level APIs remain in the Python distribution
for migration and audit. Training recipes
retain candidate decisions, source contributors and unresolved context, so a
new task does not silently turn evidence into a label.
