# External quantitative source generation audit (1.82.22)

The [complete package manifest](manifest.json) identifies the 87-file external
generation. Its SHA-256 is
`daff7930e193c6dbde1fe84c065b024187232147fa006d9348e297766e95e69c`.
The generation embeds the verified 1.82.21 supporting predecessor, its manifest,
all 17 original PXD057588/PXD009092 source assets, source contracts and typed
relations. The source assets total 11,532,567 bytes and match the pinned
[Presto source audit](https://github.com/pirl-unc/presto/blob/51b895eb3925de87df7710c7bd6c78d131c25697/experiments/2026-10-22_1410_codex_availability-source-audit/README.md).
The original inputs and embedded source copies have distinct inodes; subsequent
edits to an ingestion input do not change this generation's source snapshot.

| Checked grain | Rows |
|---|---:|
| Populated original workbook cells | 851,830 |
| Independently keyed sequence-panel records | 62,911 |
| Primary peptide rows | 35,772 |
| Assay-scoped channels | 53 |
| Source-reported positive TMT intensity results | 309,126 |
| Source-reported log2 fold-change representations | 200,724 |
| All quantity representations | 509,850 |
| Contrast definitions | 32 |
| Treated/control derivation-input relations | 802,896 |
| Ordered community SDRF cells | 3,240 |

The 35,772 primary rows comprise 10,662 MHC-bound peptide rows in Tables 7–10
and 12 and 25,110 bulk-digest peptide rows in Table 11. A panel sequence is
not an extra assay row. The 72-row community SDRF annotates six of 50 deposited
RAW files; its labels cannot be propagated to unrelated assays. Parent protein
strings are literal source labels without verified sequence-to-protein mapping.
The project metadata reports an Orbitrap Exploris 480 at project scope, not
as a proven instrument for each acquisition.

## Independent verification

An independent streaming audit compared each original Excel cell, primary row,
panel record, channel, intensity result, reported fold change, contrast and
derivation input with the typed projection. It found **zero row, key, source-cell
or value mismatches**. All 3,240 SDRF cells matched their original positional
headers and values, including repeated headers. An independently coded scalar
replay of all 200,724 fold changes had maximum absolute error
`2.550737399076297e-14` log2, below the pinned `1e-8` tolerance. Table 11's
channel-total normalization was applied over all retained source rows before
the treated/control ratio; the five other tables use unscaled intensities.
Missing discarded rows were never filled with zero. The embedded package
verifier checks the parent, source hashes, typed schemas, row counts and exact
file inventory.

| Table | Primary rows | Source cells | Intensity results | Fold changes |
|---:|---:|---:|---:|---:|
| 7 | 2,824 | 47,408 | 16,944 | 8,472 |
| 8 | 3,554 | 54,446 | 21,324 | 10,662 |
| 9 | 1,309 | 31,108 | 11,781 | 7,854 |
| 10 | 2,613 | 80,149 | 28,743 | 20,904 |
| 11 | 25,110 | 627,790 | 225,990 | 150,660 |
| 12 | 362 | 10,929 | 4,344 | 2,172 |

The source-specific [quantity reconciliation](https://github.com/pirl-unc/presto/blob/30cc5368bfa58b936339c3e5600309625831d851/experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/README.md)
pins the six-table arithmetic contract. The package retains conflicting Table
8 polarization descriptions and Table 10 replicate descriptions as separate
source assertions. Table 10 measures macrophages while its MHC-I-loss claim
applies to tumor partners. Same-named TMT channels or source labels do not
establish shared specimens. All 17 assets remain in the output; workbook
supplements have publisher-reported CC BY 4.0, the PRIDE deposit reports CC0,
and rights for the API/SDRF snapshots remain unverified.

Wolf-Levy/PXD009092 quantities, PXD017407 calibration, PXD013455 technical
fractions and PXD056426/HAP1 control/loss evidence remain cataloged as
**contract-only candidates** with explicit defer reasons. Their unverified
measurements are not fabricated, and this source generation makes no training
eligibility claim.
