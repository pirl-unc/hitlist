# Primary HLA Ligand Atlas donor qualification

Prospective source audit by Codex / GPT-6, 2026-10-16. Issue #702, within #680/#678.
The clean executed commit, actual results, runtime and verification receipts will
be recorded after execution. No model, optimization or predictions are planned.

## Frozen contract

Use all release 2020.12 primary sample-hit observations and their donor/peptide
reference tables, pinned in `hitlist/data/hla_ligand_atlas.yaml`. Source:
https://hla-ligand-atlas.org/data; Marcu et al., PMID 33858848,
DOI https://doi.org/10.1136/jitc-2020-002071; PRIDE PXD019643; CC-BY-4.0.

Order the 21 namespaced donor IDs by SHA-256 of `atlas-donor-holdout-v1`, newline,
and donor ID. Assign the first three donors to test, next three to validation,
and remaining fifteen to training. Preserve every native observation, sample-hit
row pointer and donor typing. This rule uses neither measurements nor fitted
predictions. Donor groups and reported sample groups must be disjoint; peptide
overlap is measured and fully exported. No sequence-disjoint claim is planned.

The source defines sample groups as donor/tissue/HLA-class triplets. Physical
specimen, aliquot and acquisition identities are not recovered from those keys.
All tissues from one donor nevertheless stay together. Primary benign tissues
are not homogeneous cell types or proof of absence of nonmalignant disease.
All observations are positive eluted ligands; missing observations are not
negative labels. Donor HLA typing is not peptide-specific measured restriction.
Source quality control used binding predictions, a limitation for benchmarking
those predictors; their prediction scores are not new experimental labels.

The historical IEDB cohort is immutable. This source overlaps its Atlas study and
cannot simply be appended as independent evidence. A new prospective consumer
contract must exclude/group mirrors and audit other source contributors before
training. The published historical IC50 fit and failed quality decision remain.

## Reproduction

From the frozen clean checkout, Python 3.12.6, pandas 2.3.3, datacache 1.27.1:

```sh
python scripts/audit_atlas_lineage_v2.py --receipt tasks/atlas-lineage-results/audit-receipt.json
python scripts/verify_atlas_lineage.py tasks/atlas-lineage-results/audit-receipt.json --output tasks/atlas-lineage-results/independent-verification.json
```

The audit stores all primary/derived files in shared datacache generations and
emits a receipt. The independent standard-library verifier imports no Hitlist
reader or audit helpers and replays every source row, identity, typing, partition,
coverage count and overlap. Publication must expose those tables, policies,
source registry, code, hashes and verification, with public downloads checked.

Acquisition note: the first peptide-table request stalled and was interrupted;
a bounded retry succeeded. Both other primary tables were downloaded successfully.
The acquired peptide gzip also matches the independently existing 2025 cache bytes.
No partial source generation or failed audit is treated as complete evidence.

## Executed result and independent review

Successful clean recipe `fc5748b`, candidate Hitlist 1.82.0, retains all **967,449
observations**, **223,246 unique peptides**, **21 donors**, **29 tissues**, and
**418 reported sample groups**. The independent verifier checks all 17 generated
artifacts and reproduces every source-row join, identity, typing and split.

| Partition | Donors | Observations | Unique peptides | Reported sample groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 15 | 712,079 | 189,871 | 298 |
| Validation | 3 | 134,015 | 60,686 | 64 |
| Test | 3 | 121,355 | 50,697 | 56 |

Donors, observations and reported sample groups have **zero cross-partition
overlap**. Peptides overlap: 33,241 train/test, 43,232 train/validation, 18,447
validation/test. All 94,920 pair/peptide entries are downloadable. Thus the
**donor-disjoint policy passes; peptide-disjoint policy fails**. Physical specimen
and acquisition identities remain unknown. This source-only cohort is not a
qualified independent addition to IEDB or other Atlas reanalyses.

The original summary recipe `209c7a4` failed independent donor-typing verification
(#703): 19 donor-summary rows were misassociated by a categorical lookup. The
source observations and partitions were correct. The v2 recipe groups typing
within each donor and rejects inconsistent groups. Every artifact except the
summary, executed recipe and execution receipt is byte-identical across the two
runs. The original recipe, receipt and failed-verifier log are preserved, and no
public release or model used the rejected summary. Regressions cover categorical
ordering and conflicting donor typing.

The successful CPU builder took **64.865 seconds**, with process peak RSS
**1,752,547,328 bytes**; acquisition/imports excluded. Python 3.12.6, pandas 2.3.3,
datacache 1.27.1. No cloud/GPU/model/optimization/predictions. Full row tables and
primary bytes remain in shared datacache; source code holds compact receipts.

Public evidence: https://github.com/pirl-unc/hitlist/releases/tag/atlas-donor-lineage-v1
(publication and public verification remain required at this snapshot).


## Public verification

All **27 public assets** were independently downloaded through datacache and
hash-verified. The downloaded 17-artifact core then passed the independent full
967,449-row replay, donor typing, counts and every overlap entry. See
`publication.json` and `public-verification.json` for exact public URLs and hashes.
Source cohort publication is complete; software PR #704 still requires complete
local/CI checks, merge and a verified clean-main PyPI release.
