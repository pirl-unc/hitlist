# Recovered human-source baseline

The export from clean source commit `26d552fcbb38548e839cffb8441fe1fca4104120`
used released Hitlist 1.76.0 and the frozen Linux dependencies. All ten artifacts
were published and verified on the shared Modal volume at 2026-10-16T03:49:57Z.
The launcher then failed saving its receipt because it supplied an empty
source manifest to datacache (#695). The original log and immutable datacache
manifest are retained here. Original export runtime/RSS were lost and stay null.

The recovery source snapshot matches the SHA-256 in recovered-receipt.json.
It streamed every original artifact into shared datacache, verified sizes/hashes
and the complete native training bundle, then published an explicit recovery
receipt. Recovery took 244.14 seconds. Original artifact bytes remain unchanged.
The corrected receipt helper is tested against real datacache for both success
and failure, with result and log bytes as hash-pinned inputs.

Measured native coverage: 4,071,463 observations, 4,045,567 MS and 25,896 binding;
7,646,178 contributor links and 7,605,968 source records, no unresolved links.
Class I: 2,711,461; class II: 1,353,735; non-classical: 6,267. Source inspection
retains 1,081 nonempty PMIDs, 21 supplementary files, 133 cell-line names, 28 cell
types and 437 nonempty condition IDs, alongside missingness and original labels.

Donor/specimen/acquisition identity remains unknown for all observations.
Experimental origin is resolved for 57,188 observations across 17 identities;
4,014,275 remain unknown. Condition ID is missing for 2,090,691 observations and
cell type for 1,281,272. Units/censoring are not imputed or conflated across assays.
These are observation counts, not independent specimens or training examples.

This is the 1.76.0 baseline, before #694/#696 curation corrections. It must not
be relabeled as containing those fixes. It has no frozen partitions or model
qualification claim. It is published as [human-cohort-baseline-v1](https://github.com/pirl-unc/hitlist/releases/tag/human-cohort-baseline-v1) with explicit limits;
a corrected source generation and actual consumer audits remain under #680/#678.

Validation: format/lint; receipt/coverage regressions pass (four tests), then
./test.sh passes 3,039 tests. Final-head CI and release remain required.

All fourteen public files were independently re-downloaded and verified through
datacache. [public-artifacts.json](public-artifacts.json) pins every public URL,
SHA-256 and byte count. The source bundle manifest is
`039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d`.
