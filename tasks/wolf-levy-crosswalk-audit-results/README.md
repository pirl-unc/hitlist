# Wolf–Levy companion-tryptic crosswalk audit

Audited October 25, 2026 for issue #823 / #757. This receipt covers the
original PRIDE project and file-inventory JSON for the three conventional
tryptic deposits adjacent to the Wolf–Levy captured-fragment MAPP studies.
It is intentionally a **negative-evidence** receipt: it records what the
reviewed source assets do and do not establish, without converting absence in
one asset class into a claim that no mapping exists anywhere.

| Deposit | Role relative to captured MAPP | Raw files | Raw-file records with `sampleProperties` | Result |
| --- | --- | ---: | ---: | --- |
| PXD009360 | HEK293 parallel tryptic candidate for PXD009092 | 9 | 0 | no condition, specimen, aliquot or exact MAPP crosswalk |
| PXD009181 | time-course tryptic companion candidate for PXD007263 | 24 | 0 | no condition, specimen, aliquot or exact MAPP crosswalk |
| PXD009112 | PBMC tryptic companion candidate for PXD009438 | 14 | 0 | no condition, specimen, aliquot or exact MAPP crosswalk |

The exact PRIDE endpoint URLs, retrieved byte hashes, sizes and count results
are frozen in
`hitlist/data/evidence_revisions/wolf_levy_companion_crosswalk_audit.json`.
PXD009360's current file inventory has the same SHA-256 as the separately
replayable `precursor_context` source receipt. The PXD009181 project describes
the cross-time-point study and PXD009112 describes the SLE PBMC study, but a
project description is not an acquisition-level material mapping.

The audit does not use acquisition order, embedded run numbers, a shared cell
line, publication-level statements, matching labels or quantitative
correlation to pair material. The source package now represents PXD009360's
project/acquisition membership and the unresolved claims through the same
source-assertion/material-context/material-link vocabulary as canonical MAPP.
PXD009181 and PXD009112 are retained in this receipt as unadmitted companion
source candidates; adding their search tables requires independently pinned
source assets and must preserve this unresolved status unless a narrower source
record is recovered.
