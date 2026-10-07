# CTA expression evidence and tissue blacklists

Hitlist exports an auditable evidence bundle from one sample's **gene or
transcript TPM table**, without VCF/BAM. It combines OncoRef CTA definitions and
HPA evidence, indexed peptide presentation, complete reference mappings, and a
donor-resolved sequence blacklist. Tsarina and Vaxrank consume the evidence for
target selection and vaccine assembly; Hitlist does not assemble a vaccine or
predict presentation in this workflow.

## Export from expression

```bash
hitlist export cta-evidence \
  --expression patient.tsv \
  --atlas-dir hla_2020.12 --bundle patient-evidence

hitlist verify-evidence-bundle patient-evidence
```

Identifier and TPM columns and the gene/transcript level are detected from
conventional headers. With exactly one conventional file in the working
directory, `--expression` can also be omitted. The search accepts
`expression.tsv`, `expression.csv`, `expression.tab`, `expression.sf`,
`quant.sf`, `abundance.tsv`, `*.genes.results` and `*.isoforms.results`,
including their `.gz` variants. An explicit directory uses the same search.
It does not search subdirectories or choose between multiple candidates.
An explicit file can have any basename with one of these supported extensions.

| Input | Identifier and level | Expression default |
| --- | --- | --- |
| Gene table | `ensembl_gene_id`/`gene_id`, `gene`, `symbol`, `gene_symbol` or `gene_name`; stable ID preferred | One `TPM`, `TPM_<sample>` or `<sample>_TPM` column |
| Transcript table | `ensembl_transcript_id`/`transcript_id` or `transcript` | One TPM-labelled column |
| [Salmon `quant.sf`](https://salmon.readthedocs.io/en/latest/file_formats.html) | `Name`, transcript | `TPM` |
| [kallisto `abundance.tsv`](https://pachterlab.github.io/kallisto/manual) | `target_id`, transcript | `tpm` |
| [RSEM results](https://deweylab.github.io/RSEM/rsem-calculate-expression.html) | `gene_id` for genes; `transcript_id` for isoforms | `TPM`; auxiliary posterior estimates and confidence intervals do not count as samples |

Header matching ignores case, surrounding spaces, and differences between
spaces, hyphens and underscores. Neutral `Name`, `target_id` or `id` columns
containing Ensembl transcript IDs infer transcript level; Ensembl gene IDs
infer gene level, including gene-aggregated quantifier output. Other neutral
identifiers default to genes unless a recognized quantifier schema establishes
transcript level. Mixed gene/transcript identifiers require a level override.

Use `--id-column`, `--tpm-column` and `--expression-level gene|transcript` to
override detection. Explicit column names are exact. Generic tables containing
both gene and transcript axes require an identifier or level choice; tables
with several sample TPM columns require a TPM column choice. For example:

```bash
hitlist export cta-evidence --expression cohort.tsv \
  --id-column gene_id --tpm-column patient_123_TPM \
  --atlas-dir hla_2020.12 --bundle patient-evidence
```

The CLI reports the resolved path, columns and level. The manifest retains
these choices and `inferred_inputs` flags; the Python
`resolve_expression_table()` and `write_cta_evidence_bundle()` APIs use the
same defaults. Actual expression measurements must be supplied: no patient
values are synthesized. Values must be TPM; counts and FPKM are never inferred
as TPM. The default minimum is 2 TPM (`--min-tpm`).

The default definition is OncoRef's strict CTA set; `--cta-definition extended`
selects its extended set. Original cells, input row numbers, file hashes,
canonical identities, measured TPM and selection reasons are retained.
Missing TPM is unmeasured. Duplicate resolved identifiers fail rather than
silently sum aliases. A gene TPM does not establish an expressed isoform.

For transcript inputs, the detected or explicit transcript column supplies
versioned Ensembl transcript IDs, which are retained and resolved
against the installed `--ensembl-release` (default 112). Only mappings to the
named, eligible transcript are used to select peptides. All mappings for those
peptides are then exported, including unselected isoforms and non-CTA proteins.
The local reference database hash is captured; transcript resolution does not
download or build it implicitly.

By default, MAGE-family genes are excluded except MAGEA4. Repeat
`--exclude-gene-pattern 'MAGE*'` to replace the pattern list and repeat
`--allow-gene MAGEA4` to replace its exceptions. `--no-gene-exclusions` removes
the pattern filter. These choices affect input eligibility, never CTA membership
or the full human reference used to find shared sequences.

The exporter requires current observations, complete Ensembl mappings for the
chosen release, and complete raw contributor provenance. It fails on stale or
partial indexes; it does not start a full database build. Mapping queries select
only the required genes/peptides, and raw contributors are copied in batches.
The selected evidence and reference tables still need to fit in memory.

## Interpret the peptide evidence

`cta_specific` means every resolved human reference mapping is to a CTA in the
recorded definition, with no incomplete human mapping. A peptide can be shared
between several CTAs and still be CTA-specific. The bundle separately records
`shared_between_ctas`, `shared_with_other_proteins`, protein/gene counts, every
mapping coordinate and transcript, and `has_non_cta_match`.

`presentation.parquet` contains positive mass-spectrometry modality evidence:
an explicit MS assay method or a curated MS-only supplement without a conflicting
method. Legacy nonbinding rows with other modalities are retained in
`excluded_observations.parquet` and their raw contributors remain available
([modality issue #644](https://github.com/pirl-unc/hitlist/issues/644)). Sample HLA
typing is not an experimentally assigned peptide restriction. All observed HLA
classes are retained; the blacklist never depends on the patient's HLA.

`avoid_sequence` is true for a non-CTA reference match, unresolved human mapping,
or a known blacklisted sequence anywhere inside the peptide. `blacklisted`
marks exact observed sequences; `contains_blacklisted_sequence` and
`blacklist_matches` additionally identify known shorter exclusions inside a
longer candidate. This does not infer that unobserved nested epitopes were
measured. Nonmalignant essential-tissue observations with unresolved donors
remain visible as `tissue_review_required`; they do not prove independent people.
Absence from the blacklist is not evidence of safety or absence of presentation.

## Reusable, allele-independent blacklist

```bash
hitlist export tissue-blacklist \
  --atlas-dir hla_2020.12 --bundle tissue-blacklist
hitlist verify-evidence-bundle tissue-blacklist
```

The default rule is an exact MS-observed sequence in **nonmalignant heart,
brain, or lung in at least two distinct people across the union of those tissues**.
Brain includes cerebellum. Repeated assays, HLA classes and multiple tissues
from one donor count as one person. Tumors and cell lines do not qualify.
Tissue aliases and the threshold are recorded in the bundle's policy. The
reusable blacklist is independent of expression selection and MAGE exclusions.
Downstream consumers should exclude each forbidden contiguous sequence wherever
it occurs, including inside longer antigen segments.

Supply the primary [HLA Ligand Atlas 2020.12 tables](https://hla-ligand-atlas.org/data):
`peptides.tsv.gz`, `sample_hits.tsv.gz`, and `donors.tsv.gz`. Uncompressed TSVs
and the release archive's `HLA_*.tsv` names are also accepted. The
[Atlas publication](https://doi.org/10.1136/jitc-2020-002071) describes its
nonmalignant tissue resource; the data are CC-BY-4.0. The importer retains exact
sequences, including selenocysteine U, rather than substituting residues.

This first source adapter uses the **supplied Atlas snapshot**, not every MS
study in Hitlist. Its manifest records the release, source URLs, file hashes,
source table sizes and coverage. Atlas donor IDs survive in primary sample-hit
tables but are absent from the tissue-aggregated IEDB deposit; PMIDs and sample
counts cannot substitute for donors. The generic Python
`build_tissue_blacklist(observations, donor_aliases=...)` supports additional
reviewed donor-resolved rows; cross-study donor aliases must identify the same
person explicitly. Other sources are not silently merged into the Atlas bundle.

## Bundle contents and verification

| Artifact | Evidence |
| --- | --- |
| `expression.parquet` | Every input row, measured TPM, resolution and selection |
| `cta_reference.parquet` | OncoRef definition snapshot and HPA tissue-risk fields |
| `expression_mappings.parquet` | Links from selected input rows to peptide occurrences |
| `peptides.parquet`, `mappings.parquet` | Specificity, sharing, exclusions and complete indexed mappings |
| `presentation.parquet`, `excluded_observations.parquet` | Accepted MS and excluded assay rows |
| `identities.parquet`, `contributors.parquet`, `lineage.json` | Observation identities, complete original source records and reviewed lineage |
| `tissue_evidence.parquet`, `atlas_donors.parquet` | Original Atlas fields/row locations, donor typing and qualification |
| `tissue_risk.parquet`, `blacklist.parquet` | Distinct union/per-tissue donor counts, donor identities and exclusions |
| `forbidden_sequences.txt` | Sorted exact sequences for downstream exclusion |
| `manifest.json` | Hashes, source/index fingerprints, versions, parameters and coverage |

The standalone blacklist contains only the last four groups. Verification works
offline from the captured definition and tissue policy. It checks file hashes,
raw Atlas relationships, donor counts, mapping annotations, expression links and
contributor coverage. Hashes establish integrity relative to the manifest, not a
signature certifying its publisher. Original source files are fingerprinted;
rechecking them externally is separate from portable bundle verification.

Exports require a new destination and stage their artifacts before publication.
Input changes during export cause failure. The manifest is published last;
consumers must require it and verify the bundle before use. Gene/transcript
resolution and the peptide reference only establish specificity within the
recorded reference release, not all possible patient variants or unannotated ORFs.

Assay admission and migration are documented in [assay modality](assay-modality.md).
New bundles retain binding/other assays as excluded evidence with provenance and
record MS policy version 2; these records never contribute to presentation counts.
