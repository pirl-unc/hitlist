# Hitlist 2.0 evidence-store audit

The [machine-readable audit](audit.json) was generated from the verified
Hitlist 2.0 reference successor package. Its top-level manifest SHA-256 is
`d99f67e625c6c33a2054a56dea4f98fb8248ca5f717cc98af3662f228655fa6a`;
the audit JSON SHA-256 is
`02dd24ac68ebbd68067437cfd906e04f2f059221c588ce0765326475676aa731`.
The successor independently clones all 103 predecessor files, includes the
mouse UniProt 2026_03 FASTA and the [independently verified mapping ledgers](https://github.com/pirl-unc/presto/blob/24f1d2b/experiments/2026-10-22_1914_codex_pxd057588-shared-sequence-admission/README.md),
and records every predecessor and source digest. Each cloned file has a
different inode; APFS copy-on-write preserves prior bytes if either generation
changes. The pinned reference FASTA SHA-256 is
`41bc0fa6cb9118ba95f330a27b18c39559e9397b6d99d1e73cc094869c99787a`.

```python
import json
from pathlib import Path
from hitlist import EvidenceStore

store = EvidenceStore.open("/path/to/pinned-hitlist-2.0-package")
audit = store.audit()
assert all(item["status"] == "matched" for item in audit["baselines"])
Path("tasks/evidence-store-results/audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
```

All 32 pinned baselines match. The original three MHC assay partitions remain
4,486,991 / 814,268 / 26,537 rows, and T-cell availability remains 578,529
distinct IEDB assay IDs with 157,263 overlapping CEDAR contributors. The 807
reviewed arms and 216 studies each occur in two curation revisions. Bulk and
line-expression source counts and their separate numeric representations are
unchanged. PXD057588 retains 35,772 primary peptide source rows, 309,126
assay-channel results and 509,850 distinct numeric representations.

The appended reference evidence has 35,772 source-row mapping statuses,
30,587 distinct sequences, 52,277 exact parent positions, 25,758 UniProt
reference identities, 26,449 Table 9/11 parent-support sets, five
project-scoped mouse population-role claims and one reference-context lookup
per 309,126 original results. Of the source parent labels, 35,747 match their
sequence exactly, 24 IDs are missing and one resolved ID lacks the peptide.
These relations add candidate information; they are not extra assays,
independent proteins or specimen matches. All specimen links remain unresolved.
There are 5,374 matched reference components (largest: 862 distinct peptides)
and one unmatched sequence with no component key. Components constrain possible
splits; they do not identify donors, specimens or homology classes.

The catalog has 53 materialized and four contract-only evidence types. Of its
declared fields, 431 bind to physical columns, 21 to source-specific constants,
105 require named relationships, 68 are unavailable at the claimed grain and
43 are contract-only. All 779 physical column placements remain queryable
as `native.<column>`; 427 also have safe logical names and 352 remain
source-native. Shared files under distinct selectors do not add assay rows.

The report includes 66 source/endpoint/representation count groups, 33 T-cell
eligibility groups, 12 mouse-reference source-parent groups, field readiness,
native-field disposition and source rights/revisions. The bounded MHC replay
selected 20 candidates and retained 40 separate IEDB/CEDAR contributors; a
shared T-cell assay selected one candidate with two provider records. A bounded
mouse numeric replay selected 20 of 309,126 raw intensity cells, preserved the
other 309,106 as limit-omitted and all 200,724 fold-change representations as
query-filtered, and emitted 20 exact source peptide links. All three recipe
outputs verify their source manifest binding and output file hashes. They are
behavior checks, not full-corpus training exports or a model result.
