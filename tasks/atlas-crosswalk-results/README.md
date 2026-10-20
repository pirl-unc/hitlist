# Atlas community source crosswalk (2026-10-20)

This is a source-context audit of the frozen [Atlas donor cohort](../atlas-lineage-results/README.md),
not a new peptide-to-run assignment. The cohort remains positive-only, and its
train/validation/test donor allocation and historical Presto partitions are unchanged.

## Inputs and source drift

| Source | Origin | Exact SHA-256 | Role |
| --- | --- | --- | --- |
| Primary `sample_hits.tsv.gz` | [Atlas 2020.12 release](https://hla-ligand-atlas.org/data) | `0e8771113cae27c620b2b9619f035675ddcc87b8d7dfac50750e71ab60eb0167` | Published peptide, donor, tissue and HLA-class group; no run column |
| Community SDRF | [PRIDE PXD019643](https://ftp.pride.ebi.ac.uk/pride/data/archive/2021/04/PXD019643/PXD019643_community_annotated.sdrf.tsv) | `baa1907cc847abc382683492970eef9da6fbbb24ef4448dab6dbe76357c67fb2` | Community row, source name, reported donor/tissue/class, acquisition file |
| Complete file inventory | [PRIDE API](https://www.ebi.ac.uk/pride/ws/archive/v3/projects/PXD019643/files/all) | `3bac909670e737f9f71d8bb5aad1f5e26b33f68f2c3b8bace1a350f8d6b987b0` | Exact file names, accessions and categories |

The SDRF was changed in place on 2026-10-20. The 2026-10-16 snapshot in
[#705](https://github.com/pirl-unc/hitlist/issues/705) had 752 class-I rows,
SHA-256 `359f0a67ffef5ff5288ea35b8c5e44d6be236155219b3f15c1be474f0112993e`,
and 731 file names that did not exactly match the inventory. The new snapshot
has 1,468 rows for both classes and **all 1,468 peak-file names match exactly**.
The inventory changed only in its SDRF record's `fileSizeBytes` field, from
629,542 to 1,177,127; its other 3,425 records are unchanged. Both current
source bodies are frozen as deterministic gzip snapshots in this directory.
The catalog in `hitlist/data/atlas_community_crosswalk.yaml` also pins live
acquisition through datacache. PRIDE's SDRF inventory record has no checksum,
so the SHA-256 pins are local observations of the downloaded bytes.

## Join contract and measured coverage

1. Each SDRF `comment[data file]` joins to one unique PRIDE `PEAK` record by
   the complete, exact filename. The file URI basename must agree. This proves
   file inventory membership, not QC inclusion.
2. The same stem with `.raw` gives an exact inventory `RAW` match for 1,466
   rows. Two Duodenum `.mzML` rows have no matching `.raw` stem. They remain
   unresolved; the audit makes no guessed alias. Three deposited `PEAK` files
   are outside the SDRF, as is one extra raw file. All 483 `RESULT` records
   remain without a source-backed QC/result-to-observation join.
3. Primary and SDRF donor, tissue and class cells create a candidate context.
   Case and whitespace differences in tissue labels are normalized only after
   checking that no primary group collides. Every one of 418 primary groups
   has a context in the SDRF. Five primary groups have multiple SDRF source
   names, including time-series and Duodenum/Jejunum examples. Context does
   not identify a physical specimen, aliquot, included run or peptide origin.

| Frozen partition | HLA-I observations | HLA-II observations | Candidate context | QC / acquisition / specimen resolved |
| --- | ---: | ---: | ---: | ---: |
| Train | 305,072 | 407,007 | 712,079 | 0 |
| Validation | 56,187 | 77,828 | 134,015 | 0 |
| Test | 48,244 | 73,111 | 121,355 | 0 |
| **Total** | **409,503** | **557,946** | **967,449** | **0** |

The five multiple-source-name groups contain 28,350 observations. The 1,276
SDRF peak files in candidate primary contexts have no filename in more than
one donor partition. This is a candidate-file overlap result, not a pass on
actual acquisition independence. The physical specimen and included-run
verdicts are **inconclusive**. The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8054196/)
says that runs failing its QC are still deposited. The [Atlas processing
description](https://hla-ligand-atlas.org/docs/computational) gives the QC
rules, but neither the public `sample_hits` table nor these two PRIDE sources
provide an accepted run or result-file key for each released peptide.

## Artifacts and replay

`sdrf-file-crosswalk.tsv.gz` retains all original SDRF cells plus physical
row numbers and exact inventory matches. `inventory-crosswalk.tsv.gz` accounts
for every inventory record. `primary-group-crosswalk.tsv.gz` carries all 418
original group counts, frozen partitions, candidate source names/files and
explicit unknown lineage status. `summary.json` records content hashes for
inputs and outputs, per-class/per-partition observation coverage, and verdicts.
`independent-verification.json` is the separate replay result.

Run from the repository root with a pinned local Atlas primary source bundle:

```sh
python scripts/audit_atlas_crosswalk.py \
  --output-dir /tmp/atlas-crosswalk \
  --data-root /tmp/hitlist-data
python scripts/verify_atlas_crosswalk.py \
  /tmp/atlas-crosswalk \
  /tmp/hitlist-data/hla-ligand-atlas/2020.12/sources/bundles/<generation>/sample_hits.tsv.gz
```

The builder acquires and verifies both community sources through datacache.
For offline replay, pass `--community-dir` and `--primary-dir` to the builder
after extracting the two frozen source snapshots and acquiring the pinned
primary Atlas tables. A live PRIDE update that changes either body causes a
pin mismatch instead of silently changing the audit. The independent verifier
reads the frozen source snapshots and primary sample-hit table directly and
checks each crosswalk row, output hash, group count and split assignment.
