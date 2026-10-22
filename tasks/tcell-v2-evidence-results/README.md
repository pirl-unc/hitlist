# T-cell evidence generation audit (1.82.23)

The [complete 99-file package manifest](manifest.json) has SHA-256
`4c3275ca6da9a9e1de084fcc1a8d1243d4e67b1e9c3ed4d6307cbcce5c3738e3`.
It embeds the verified external predecessor, both original 2026-10-20 T-cell
source ZIPs, their receipt, a field-disposition inventory and typed Parquet
relations. The two ZIP snapshots have different inodes from their ingestion
inputs and match the pinned SHA-256s in `hitlist/data/tcell_audit_sources.yaml`.
IEDB and CEDAR source access is public; a redistributable license has not been
verified and the package makes no stronger rights claim.

| Source grain | Rows |
|---|---:|
| Distinct IEDB assay results | 578,529 |
| CEDAR contributor records that overlap IEDB | 157,263 |
| Combined source-contributor rows | 735,792 |
| CEDAR-only mutation source claims | 30,639 |
| Direct human peptide candidates | 56,025 |
| Bounded functional training examples | 47,421 |
| Bounded multimer binding examples | 8,423 |
| Direct candidates excluded for qualitative/subject-count disagreement | 181 |

The five availability families reconcile to 396,243 soluble factor release,
103,103 activation/proliferation, 24,962 cytotoxicity, 36,268 pMHC multimer
and 17,953 other assay IDs. A source method/readout family is an availability
classification, not a common binary target. CEDAR adds source evidence for
shared IDs; it does not add assays. The full 161-column IEDB and 162-column
CEDAR CSVs remain embedded, with every native header assigned a typed or
source-only disposition. All 157,263 shared IDs agree on their 158 common
fields. Contributor rows retain each provider's own assay, reference and
epitope IRIs. CEDAR's additional `Epitope | Mutation` field is populated on 30,639
shared assays, including 17 current training IDs (#750); its source claim is
retained without silently changing those examples.

## Eligibility and missingness

Every IEDB assay has an eligibility row containing its first failing gate,
all applicable failing gates, task and admission status. The deterministic
first-failure funnel is:

| Status / first failure | Assays |
|---|---:|
| Nonhuman epitope origin | 511,628 |
| Antigen not a linear peptide | 4,723 |
| Antigen not the direct epitope | 2,602 |
| Noncanonical or unknown peptide sequence | 2,258 |
| Other T-cell target family | 1,167 |
| Epitope not a linear peptide | 126 |
| Direct peptide candidates | 56,025 |

Of those candidates, 47,593 are functional source rows and 8,432 are multimer
source rows. The existing label-consistency rule excludes 172 functional and
nine multimer rows, retaining 55,844 examples in total. This reproduces the
bounded training contract; no availability-only assay becomes a peptide
training label. The 4,054 reported candidate single-HLA assignments retain
their conservative evidence status. Donor/specimen IDs are not supplied.
Numeric source text, units and inequalities remain separately typed; an
unparsable numeric string is retained rather than converted into a value.

## Independent verification

A second streaming audit compared all 578,529 typed assay rows to the existing
independently generated metadata export, and all 55,844 training examples to
the existing functional/multimer bundles. It re-read both original ZIPs to
check all 735,792 source-contributor locators, 30,639 mutation claims, and
every IEDB reported outcome, antigen and eligibility key. It found **zero
source-field, row-key, label or exclusion mismatches**. The 181 exclusions
exactly match the prior source-linked disagreement tables. The package
verifier checks every file hash, the exact parent manifest, source receipt,
schemas and pinned count reconciliation. Offline fixtures cover source
contributor disagreement, peptide pool exclusion, unknown HLA, negative
functional response versus positive multimer binding and CEDAR-only claims.
