# Hitlist 2.0 evidence-store audit

The [machine-readable audit](audit.json) was generated from the verified
103-file PR 7 package with top-level manifest SHA-256
`560770376121758b3c18679c3bf4db78bc89f9e95cc72785a7314fe35167cb1e`.
The audit JSON itself has SHA-256
`4a27db1197a1efcb2ecf6d4b5a111810b6c500c06aa84b688efa9509a09a6fc9`.

```python
import json
from pathlib import Path
from hitlist import EvidenceStore

store = EvidenceStore.open("/path/to/pinned-103-file-package")
audit = store.audit()
assert all(item["status"] == "matched" for item in audit["baselines"])
Path("tasks/evidence-store-results/audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
```

All 25 pinned baselines match. The 807 reviewed arms and 216 studies are each
present in two curation revisions, giving 1,614 and 432 rows respectively.
The three MHC assay partitions remain 4,486,991 / 814,268 / 26,537 rows.
The T-cell availability table remains 578,529 distinct IEDB assay IDs;
157,263 overlapping CEDAR records are source contributors, not new assays.
Bulk peptide detections, protein numeric representations and line expression
keep their separate grains and exact source counts.

The 2.0 catalog has 46 materialized evidence types and four contract-only
types. Among declared fields, 367 bind directly to physical columns, 12 are
source-specific constants, 73 require a named relationship, 68 are unavailable
at the declared grain and 43 belong to contract-only types. Every one of 701
physical column placements remains queryable through `native.<column>`;
364 also have a safe logical alias, while 337 remain source-native. Shared
physical files intentionally occur under several evidence types with explicit
selectors; these placements are not distinct source bytes or extra assays.

The report includes 64 source/endpoint/representation count groups, 33
T-cell eligibility groups, per-field readiness, native-field disposition,
source revision/rights evidence and each baseline comparison. The bounded
real-package MHC recipe replay selected 20 positive MS candidates and retained
40 IEDB/CEDAR contributor records without duplicating labels. A shared T-cell
assay replay selected one functional peptide example with two provider records.
Both recipe outputs verified their source manifest binding and every output
file hash. These bounded replays validate query and recipe behavior; they are
not full-corpus training exports.
