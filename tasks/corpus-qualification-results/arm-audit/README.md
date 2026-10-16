# Actual corpus v3 arm re-audit (#694, #696)

The original public corpus remains the immutable Hitlist 1.76.0 build. Its
manifest is 7f7f4035d8ec080b26385710d447633dba36ac5cce49008e48e14cf8894f20d5.
CI against that capture found missing/stale arm verdicts. These reports replay
all 90,455 affected deposited rows, preserving assay/provenance IDs and source
metadata. Derived sample assignments from the original build are excluded.
The exact local replay script is retained as a historical source snapshot;
its paths identify the datacache inputs listed in the inventories.

| Study | Captured rows | Resolved before | Resolved after | Residual ambiguity |
| --- | ---: | ---: | ---: | --- |
| van Balen, PMID 32350084 | 79,406 | 0 | 62,512 | 16,894 explicitly name several cell lines |
| Becker, PMID 33981976 | 11,049 | 0 | 0 | All lack exclusive vehicle/drug membership |

All 196 van Balen deposited statements explicitly name one or more of the 26
already curated source lines. The exact YAML mapping selects sole named sources
and retains multiple-source ambiguity. No donor identity or new cellular typing
is inferred. A pair-comparison defect falsely rejected beta/alpha versus
alpha/beta spellings of the same complete molecule (#696); both chain permutations
now compare by locus/species/allele depth/mutation, and different partners remain
incompatible. Existing cellular genotypes are unchanged.

The captured index records these motif-predicted restrictions as unknown.
New source-backed curation marks them predicted for the next index rebuild; the
replay deliberately preserves the captured field. The original frozen generation
is not retrospectively relabeled or declared current against changed curation.

Becker has 11,043 blank comments and six comparative treatment-effect comments.
Neither provides exclusive arm membership. The primary article [PMC8082087](https://pmc.ncbi.nlm.nih.gov/articles/PMC8082087/)
identifies DMSO solvent controls, so curated control metadata now says vehicle,
with known HCT 116 system and unknown observation-level vehicle/drug assignment.
[van Balen metadata](https://pubmed.ncbi.nlm.nih.gov/32350084/) and the exact
source acquisitions are pinned in literature.json. The publisher PDF returned
403 and was not used to assert a new typing audit. Exact deposited source-line
names plus existing curated samples support this attribution correction.

The 202 distinct deposited context patterns are a compact regression fixture;
its peptide is explicitly synthetic. Counts in these reports come from the full
actual-row replay, not the fixture. Downloadable before/after parquets retain
source IDs, and readable context counts show the assignment policy directly.
Full-suite and final-head corpus CI are still required before release.
