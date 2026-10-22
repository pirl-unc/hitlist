# PR 5 specification draft — external quantitative MS source generation

Depends on merged and published PR 4 / 1.82.21. Close #746; target 1.82.22.
No model-training admission is implied by source ingestion.

## Frozen inputs

Use the 1.82.21 verified supporting generation as the sole parent. Pin the
17 original PXD057588/PXD009092 assets in the Presto 2026-10-22 audit receipt
(manifest SHA-256 `14419b00aeff8caebdc67917522e329b63b07d1ea85bbb59e15ef0c07bc3a27c`),
the six-table 32-contrast transform contract, the source audit and the primary
paper's methods. The 17 original files currently stage in
`/private/tmp/hitlist-pr5-source-assets`; all have matched size and SHA-256.
Treat these paths only as ingestion inputs. Embed the exact original bytes,
contract, source receipt and predecessor files in one output generation.
Keep rights claims scoped: CC BY 4.0 supplements, CC0 PRIDE deposit, and any
unverified metadata access separate.

## Grains and identities

Register a source generation with source ID, revision/hash, adapter version,
pool, endpoint, access/license and disposition of every native field/panel.
Use stable IDs for source workbook, table/assay, source peptide row, channel
within that table, measured result at row×channel, 32 contrast definitions,
fold-change representation at row×contrast, derivation input links, source
material/condition claim, and ordered SDRF cell. Do not turn technical channels
or fractions into independent donors or specimens. Distinguish Table 11
digest peptides from the 10,662 MHC-bound rows and from protein quantities.
Parent protein strings remain literal labels, not verified mappings.

Material graph: measured macrophage versus measured tumor cell, coculture
partner, loss target on tumor partners only, untreated comparison arm, assay/plex
channel, and exact/unknown pairing precision. Same channel labels in Tables
9/11 are distinct IDs; same sample names do not prove shared aliquots. Preserve
Table 8 title/A2 conflict and Table 10 stated n=3 versus four G/C channels as
parallel source claims. The 72×45 SDRF maps to only six of 50 raw files; repeated
column headers retain positional keys and treatment `not available` stays
unreported. Do not broadcast SDRF context to other assays.

## Quantities and derivations

Materialize all 35,772 primary source peptide rows and 309,126 positive
raw-intensity cells; retain channel labels, native values and exact worksheet
coordinates. A generic `source_cells` relation inventories every populated
workbook coordinate, including independent horizontal significance panels and
method/header assertions, so their native information is queryable without
silently row-joining it to the primary peptide block. The audit inventories
62,911 sequence-block records across all panels (35,772 primary and 27,139
additional panel records); index each with its own table/panel key. Table 11's
second 25,110-row block aligns in source order, but other blocks do not, so
the general relation remains panel-scoped. Materialize all 200,724
published log2FC cells with exact source
coordinates and a derivation edge to one treated intensity and its contrast's
control intensities. Pin 32 contrast definitions, control membership,
normalization universe, operation order, source citation and tolerance.
Tables 7–10/12: log2(treated / arithmetic mean of peptide controls), unscaled.
Table 11: normalize each channel by its total over all retained rows before
the same ratio; a shared scalar multiplier cancels. Freeze the full-row
complete-case universe, no absent-as-zero or extra q-value filter. Keep the
reported q≤0.05 search policy as assay-level provenance; do not infer each
row's q-value. Preserve the horizontal significance panels with independent
sequence keys and no same-row join.

## Physical and API contract

Append generic external-source registry, source records, measured results,
quantitative representations, channel/material claims, contrast definitions,
derivation-input links, protocol/selection claims, conflict assertions and
ordered SDRF cells to the parent package. A source adapter must declare its
source field/row/cell inventory, required ingest fields and quantity rules;
new adapters enter through a registry rather than a new sidecar interface.
Catalog evidence types include MHC-bound peptide quantity and bulk-digest
peptide quantity separately, with source/pool and material precision filters.
Register Wolf-Levy, PXD017407, PXD013455 and PXD056426 acceptance fixtures as
explicit deferred source contracts until their own original assets and joins
are pinned; no fabricated measurements.

## Verification

Atomic write, full hash inventory, predecessor re-verification, all 17 source
asset hashes, schema/ID/foreign-key checks, exact table/channel/row/cell
reconciliation, conflicting claim retention, and source-only-panel recovery.
Independent scalar replay of all 200,724 published FCs within 1e-8 log2;
require the six per-table totals and opposite-policy mismatch diagnostic.
Offline fixtures: channel collision, parent-label ambiguity, Table 10 partner
KO, Table 8 contradiction, Table 10 replicate conflict, SDRF scope, independent
horizontal panels, no absent-as-zero, relative versus calibrated units,
fractions versus biological samples. Run format, lint, local/CI tests, wheel
audit, merge, clean-main release and PyPI exact-hash check before PR 6.
