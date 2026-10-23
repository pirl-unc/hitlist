# Gomez-Zepeda original result lineage replay (2.0.11 candidate)

The independent script `scripts/verify_gomez_source_lineage.py` reads both
original publisher ZIP members directly and compares every original CSV cell
with the typed result table. It reconstructs every sequence × phase ×
condition × biological culture label × sample × acquisition file membership,
compares the complete membership table, checks the exact original-to-derived
peptide links, and checks every predecessor and successor artifact hash. It
does not import the Hitlist builder.

| Grain or reconciliation | Count |
|---|---:|
| Original S3 method-comparison rows | 135,688 |
| Original S5 spike-experiment rows | 459,228 |
| Original result rows total | 594,916 |
| Distinct source-run peptide memberships | 590,615 |
| Original rows outside derived 8–15-aa length | 51,491 |
| Exact linked derived peptide observations | 68,552 |
| Unchanged predecessor files | 256 |
| Successor files | 262 |

The final candidate manifest SHA-256 is
`ade958582dd2eb580f4e02187cd7445621e72fc07ec294a97624a09ba24a2659`.

The five source sequence unions are JY 20,983, Raji 24,325, HeLa 10,356,
SK-MEL-37 11,491 and plasma 1,397, exactly matching every packaged derived
peptide. The source rows outside 8–15 amino acids remain queryable with an
explicit null derived link; they are not silently removed. The source
`Area`/`Log2Area`, modification, charge, confidence, method, culture, sample
and acquisition fields remain as original strings. A source peak area is in
source arbitrary units. A non-detection is not encoded as a negative assay.

The original S5 rows explicitly separate JY_WT, JY_S, Raji_WT and Raji_S.
The released peptide lists merge some conditions, so a derived peptide can
link to several original memberships. This PR preserves those relationships;
issue #780 will use them for source-supported condition projection and repair
the known current cohort attribution errors. No donor or specimen ID is
inferred from culture labels or source file names.

The machine-readable independent replay is in `summary.json`. Source hashes,
row counts and archive URLs are pinned in
[`../gomez-source-lineage-211.md`](../gomez-source-lineage-211.md).
