# Nine source-reviewed multimer assays, 2026-10-23

This review is first-class `assay_adjudication` evidence in the schema-8
successor package. Its versioned input is
[`hitlist/data/assay_adjudications.yaml`](../../hitlist/data/assay_adjudications.yaml),
SHA-256 `0cc78c8f956abccda03053e474019d6c8b1a762b45373c469891f9a933a81f11`.
Each row is a review of one IEDB source assay ID, not another measured assay or
one independent donor. Its claim links to the original outcome, antigen,
eligibility and IEDB/CEDAR contributor relations by `source_assay_id`.

| IEDB assay | Paper code / peptide | IEDB outcome, positive/tested | Primary-source finding | Review status |
| --- | --- | --- | --- | --- |
| [1646138](https://www.iedb.org/details_v3.php?id=1646138&type=assay) | HCgp39 / RSFTLASSETGVG | Negative, 3/20 | Overall gp39–DR4 staining was judged negative; three patients had weak individual signals. | Partially resolved; no binary peptide target |
| [12892501](https://www.iedb.org/details_v3.php?id=12892501&type=assay) | GRI-002 / VLIQDVPTL | Positive, 0/2 | Vaccine-induced responses in Table 2: 0/2. | Unresolved qualitative tier |
| [12892550](https://www.iedb.org/details_v3.php?id=12892550&type=assay) | SEZ6L-001 / LLLGSPAAA | Positive, 0/3 | Vaccine-induced responses: 0/3. | Unresolved qualitative tier |
| [12892559](https://www.iedb.org/details_v3.php?id=12892559&type=assay) | ASPM-002 / SYNPLWLRI | Positive, 0/1 | Vaccine-induced responses: 0/1. | Unresolved qualitative tier |
| [12892569](https://www.iedb.org/details_v3.php?id=12892569&type=assay) | FADS2-002 / PYNHQHEYF | Positive, 0/1 | Vaccine-induced responses: 0/1. | Unresolved qualitative tier |
| [12892606](https://www.iedb.org/details_v3.php?id=12892606&type=assay) | ST8SIA5-001 / VYYFHPQYL | Positive, 0/1 | Vaccine-induced responses: 0/1. | Unresolved qualitative tier |
| [12892609](https://www.iedb.org/details_v3.php?id=12892609&type=assay) | TMEM231-001 / TYIPPLLVAF | Positive, 0/1 | Vaccine-induced responses: 0/1. | Unresolved qualitative tier |
| [12892626](https://www.iedb.org/details_v3.php?id=12892626&type=assay) | BCA-005 / VKVNEAYRFRVALPAYPA | Positive, 0/1 | Vaccine-induced responses: 0/1; source restriction is pan-DR. | Unresolved qualitative tier and restricting allele |
| [2445272](https://www.iedb.org/details_v3.php?id=2445272&type=assay) | HNF4G-1M / MMYKDILLL | Positive, 0/6 | Supplementary Table III lists no HNF4G-1M positive frequency in donor lines. | Unresolved qualitative tier |

For 1646138, [Kotzin et al., Figures 3–4](https://pmc.ncbi.nlm.nih.gov/articles/PMC26656/)
report cohort and individual staining at different levels. The three weak
signals include broader DR4-subtype context; the IEDB HLA-DRB1*04:01 field is
preserved as reported, not promoted to an assured peptide restriction.

For the seven GAPVAC rows, the [corrected Nature Supplementary Information,
Table 2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0810-y/MediaObjects/41586_2018_810_MOESM1_ESM.pdf)
(PDF page 15, SHA-256
`e755519a29e08dc1378e09f1d95f94ac4bc893494df7330b99f6e76bba7a2631`)
matches every IEDB evaluated/positive subject count. Table 2 measures
vaccine-induced immune responses; Table 1's pre-immunogenicity counts concern a
different population and condition. Neither table explains why IEDB's
qualitative multimer tier says “Positive” for these zero-response rows.

For 2445272, [Hombrink et al., Supplementary Table
III](https://aacr.figshare.com/articles/journal_contribution/Supplementary_Table_3_from_Identification_of_Biological_Relevant_Minor_Histocompatibility_Antigens_within_the_B-lymphocyte_Derived_HLA-Ligandome_Using_a_Reverse_Immunology_Approach/22459571)
(publisher PDF SHA-256
`fa6758ca9d58cf4691dc9b6a0f17575ec88215fb023998a730cb7abe8b264058`)
reports measurements after one or two MHC-multimer pull-down and expansion
rounds. IEDB's `Direct Ex Vivo` identifies starting material, while the
source's measured lines have undergone preparation. The publication does not
resolve IEDB's Positive/0-of-6 discrepancy. The ledger's
[direct publisher file endpoint](https://api.figshare.com/v2/file/download/39910871)
was checked against the stated SHA-256. The publisher PDFs are linked and
hash-pinned in the review ledger, not redistributed in the package.

## Package and count reconciliation

The writer verified the schema-7 predecessor at
`/private/tmp/hitlist-v2-reference-200-final`, independently snapshotted all
123 listed files, added its exact predecessor manifest, the review input and
the nine-row Parquet relation, then verified the resulting 126-file schema-8
package at `/private/tmp/hitlist-v2-adjudication-204-reviewed`. The parent manifest
SHA-256 is `d99f67e625c6c33a2054a56dea4f98fb8248ca5f717cc98af3662f228655fa6a`;
the new root manifest SHA-256 is
`8f35dc5529b376f1b2e3b25f91dd2f2dc57d0a53c4d8a76c094294193708c9b6`.
The review Parquet SHA-256 is
`6f712666aede4ba12dd95cd9cb17f28dc721b344b28ca3a8b53f12907a36755a`.
All predecessor file hashes match, and predecessor and successor files have
distinct inodes.

| Grain or decision | Schema 7 | Schema 8 | Change |
| --- | ---: | ---: | ---: |
| IEDB T-cell assay IDs | 578,529 | 578,529 | 0 |
| T-cell source contributor rows | 735,792 | 735,792 | 0 |
| Direct-peptide multimer candidates | 8,432 | 8,432 | 0 |
| Accepted multimer training examples | 8,423 | 8,423 | 0 |
| All T-cell training examples | 55,844 | 55,844 | 0 |
| Discordant direct-peptide multimer exclusions | 9 | 9 | 0 |
| Assay-level review claims | 0 | 9 | +9 annotations |

`EvidenceStore.open` verified both packages. Schema 7 reports the review
relation as `not_in_generation`; schema 8 exposes all nine reviews and joins
each to its original outcome, eligibility and antigen claim. Querying assay
1646138 with the optional `adjudication` join returned one assay row. All
nine remain `training_status=excluded` with
`qualitative_subject_count_disagreement`; no source field was relabeled and no
new training example was admitted. The complete schema-8 audit reports 54
materialized evidence types, four declared contracts and all 33 count
baselines matched. The schema-6 numeric predecessor also remains readable and
reports the reference and review relations as `not_in_generation`.
