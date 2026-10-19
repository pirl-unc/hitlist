# Reported restrictions and inferred allele candidates

`mhc_restriction` retains a reported gene/locus statement such as HLA-DR, DQ,
DP or DRB1. `mhc_allele_set` separately intersects exact available typing with
that statement's MHC species, class and gene/locus. `mhc_allele_provenance`
labels that inference as `peptide_locus_match`, `sample_locus_match` or
`pmid_locus_pool`. A singleton is still an inferred candidate: it does not
change `allele_resolution`, establish exact restriction evidence or pass an
`exact` provenance filter.

This preserves the distinctions in [Ramarathinam et al. 2021, PMID 34357683](https://pubmed.ncbi.nlm.nih.gov/34357683/).
The paper describes the DR12/DQ7/DP4 repertoire of C1R cells. Narrowing an
HLA-DR observation to typed DR candidates preserves the deposited locus
evidence; replacing it with a measured allele assignment would overstate it.

Typing precedence is peptide-specific attribution, donor typing, then the
curated PMID pool. Matching chooses the strongest nonempty exact-typing tier
first. If its compatible intersection is empty, the row stays `unmatched`;
it never falls back to a broader pool. A PMID pool is a study-wide candidate
union, not a donor genotype. Incomplete donor alleles, serotypes and partial
pairs also block locus inference from a broader pool; missing fields or chains
are not filled from other samples. Explicit MHC species remains authoritative in
engineered or xenogeneic material; host and peptide-source species cannot
license incompatible MHC candidates. Curated context refines generic labels
and genus-level designations, while absent context stays unknown. The parser's
species ancestry determines compatibility, with sibling species excluded when
a specific species is evidenced.

Class-II candidate designations may be single chains or explicitly supplied
complete alpha/beta pairs. The code keeps either form as supplied and never
creates missing partners or pairs independent chains. A DR locus constraint
includes typed DRB3/4/5 as well as DRB1. Every chain of a supplied pair must
belong to a locus constraint; a gene constraint can match its chain inside an
intact pair. Candidate counts count designations, not proven complete pMHC
molecules. Low-resolution groups, partial pairs and free text are excluded
from exact typing.

Blank restrictions remain unmatched even when typing exists: they provide no
locus statement to intersect. Serotype, haplotype and catalog expansion remain
separate work. Historical class-only promotion is retained; new gene/locus
inference never uses that promotion path. Shared class-only checks now parse
non-human typing and use ontology classes instead of HLA prefixes.

Candidate sets are persisted at build time. **Rebuild existing observation
artifacts** to obtain this behavior; loading an old parquet does not recompute
its sets. Observation artifact version 8 invalidates cached build candidates.

## Reproducible coverage audit

Run from this checkout:

```sh
python scripts/allele_candidate_audit.py /path/to/observations.parquet /path/to/audit
```

The audit scans bounded Arrow batches and uses disk-backed distinct peptide
sets. It writes `manifest.json`, `summary.json`, `per_paper.json` and
`per_locus.json`. The manifest fingerprints input data, curation and matching
code and records package/parser versions. CI verifies the pinned `ci-corpus-v2`
download and uploads the report as `allele-candidate-audit`.

This is a replay of retained gene/locus and class-only statements against the
persisted candidate baseline, with blanks counted separately. Other rows keep
their baseline coverage: old promoted class-only source statements cannot be
recovered from parquet. It is not a full source rebuild or an allele-assignment
accuracy estimate. Distinct peptides mean distinct sequence strings; repeated
rows and sequences shared across papers count once in global totals.

On the verified raw `ci-corpus-v2` artifact (4,440,428 rows; SHA256
`0b34bd12471882eb28d4f9a1dad6cc48252fb93a3644f912f69a0a186228f847`):

| Population | Rows gaining candidates | Distinct peptides gaining candidates within population |
|---|---:|---:|
| Gene/locus statements | 134,052 | 98,810 |
| Retained class-only statements with non-human typing | 6,164 | 6,164 |
| Blank restrictions | 0 of 135,446 | 0 of 35,910 |

The gene/locus population has 147,422 rows; 13,370 stay unmatched. Of the
134,055 rows with available typing, three HLA-DP rows in PMID 34004174 have an
empty compatible intersection and stay unmatched. Across both replayed
populations, **66,134 sequences** gain their first candidate set anywhere in
the corpus (60,847 from the locus pass alone). This is smaller than the number
of affected sequences because many already have candidate evidence elsewhere.

Ramarathinam gains 76,855 rows and **62,467 unique deposited sequences** across
DR, DQ and DP. Its locus counts are 52,826 DR sequences, 6,872 DQ and 3,877 DP;
their sum exceeds the paper-level union because sequences can overlap loci.
The paper reports 71,350 unique peptides; that publication count is not the
same population as these deposited observations and is not used as the audit
denominator. No unreported chain or peptide is added to reconcile them.
The source-count reconciliation is tracked in [#632](https://github.com/pirl-unc/hitlist/issues/632).

All 29 papers with gene/locus gains:

| PMID | Rows gaining candidates | Unique sequences gaining candidates within gene/locus rows |
|---|---:|---:|
| 34357683 | 76,855 | 62,467 |
| 35154160 | 27,015 | 20,819 |
| 29314611 | 11,738 | 10,940 |
| 33043033 | 11,106 | 9,867 |
| 27726376 | 2,627 | 2,153 |
| 27090790 | 1,067 | 721 |
| 32162841 | 916 | 893 |
| 22299025 | 572 | 512 |
| 28560793 | 447 | 445 |
| 21081667 | 338 | 310 |
| 25135637 | 196 | 189 |
| 35494241 | 149 | 144 |
| 22348091 | 147 | 121 |
| 34004174 | 136 | 134 |
| 18566446 | 134 | 134 |
| 23783831 | 125 | 125 |
| 21467215 | 88 | 88 |
| 31020640 | 84 | 84 |
| 28489076 | 72 | 72 |
| 27550523 | 66 | 66 |
| 32244010 | 49 | 49 |
| 36828807 | 46 | 46 |
| 20132976 | 25 | 25 |
| 29567779 | 19 | 16 |
| 23372163 | 15 | 15 |
| 30626607 | 13 | 13 |
| 25911201 | 3 | 3 |
| 29974988 | 3 | 3 |
| 29025906 | 1 | 1 |

Per-paper counts must not be summed to estimate globally unique sequences.
