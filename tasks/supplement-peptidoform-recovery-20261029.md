# Supplementary peptidoform recovery (#854)

Implement lossless transport of supplied has_ptm, peptide_modifications and
peptide_extended fields through scan_supplementary. Parse explicit Boolean text
with true/false/unknown preserved; retain raw values and conflicting evidence.
Keep distinct peptidoforms and unknown versus explicitly-unmodified records
separate in deduplication and synthesized identity. File-level assertions require
an explicit cited policy rather than assuming all bare sequences are unmodified.

Review original source supplements for available peptidoform information; preserve
file hash, sheet/row or field locator and curation rationale. Unknown source
chemistry remains unknown. No blanket False fill. Downstream canonical exports
must retain the same fields so Presto can admit qualified observations.

Verification: fixtures for False/True/unknown/conflict, mixed-case strings,
modification-text preservation, same-sequence different-form survival, full
source/export round trips, and unchanged legacy unknown-source row counts.
Run format.sh, lint.sh and test.sh; use isolated environment/hosted execution if
local resource admission refuses. Version bump and PR references #854. Existing
checkouts and shared installed package remain unchanged during development.

Implementation review: source fields now survive scan → builder namespace → unified
training export. Both within-supplement deduplication and the cross-source overlap
join distinguish unknown and explicit chemistry. Synthetic fixtures cover all
three states and conflicts. Format and lint pass; nine focused tests pass. Local
full-suite preflight refused at 0.18 GiB available (2.5 GiB required); do not bypass
the memory guard. Run the repository's hosted release gate for the complete suite.
Original Stražar PSM recovery is being validated in Presto's registered curation
family; this transport PR does not label bare legacy source sequences unmodified.
