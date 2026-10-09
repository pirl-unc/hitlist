# Species-scoped evidence bundles

`write_species_evidence_bundle` consumes explicit, local, frozen inputs. It does
not download data, rebuild the global index, import human CTA membership, or
substitute human normal-tissue observations for missing canine evidence. The
existing human CTA API and historical bundle verification remain unchanged.

```python
from hitlist import write_species_evidence_bundle, verify_evidence_bundle

# Reads ./species-evidence.json by default. The destination must not exist.
write_species_evidence_bundle("dog-evidence")
verify_evidence_bundle("dog-evidence")  # offline, including recomputation
```

This is an evidence interchange boundary for Canvax, Tsarina and other consumers.
It does not select a vaccine, generate theoretical epitope windows or assemble a
sequence. Candidate membership is a caller-supplied empirical discovery claim;
the bundle retains counterevidence and uncertainty independently of that claim.

## Input manifest

The manifest has `schema_version: 1` and these fields:

| Field | Contract |
|---|---|
| `reference` | `taxon` (NCBI integer), `assembly_accession`, `annotation_release`, `source_version`, and `asset_hashes` (asset name to SHA256). These are exactly the Canvax reference fields. |
| `reference_complete` | Must be `true`: the supplier declares that the FASTA contains the complete protein search background for the stated reference scope. Hitlist checks every supplied occurrence against it; it cannot independently prove an external reference's biological completeness. |
| `reference_scope` | A nonempty description of the supplied reference, including any source limitations. |
| `files` | Exactly `reference`, `expression`, `candidates`, `observations`, `contributors`, `lineage`, and `normal`. Each entry has a relative local `path`, SHA256 `sha256`, and integer `size_bytes`. Symlinks and paths escaping the manifest directory are rejected. |
| `policy` | Explicit versioned expression and tissue policy described below. |
| `normal_source` | Target `taxon`, `source`, `version`, `coverage` (`scoped` or `missing`), and `coverage_note`. The taxon must match the reference. |
| `peptides` | Optional additional exact, unmodified 5–50 amino-acid strings. All valid supplied observation strings are also mapped, including excluded non-MS observations. No nested sequence acquires MS evidence from a longer observation. |
| `include_il_equivalent` | Optional boolean, default `false`. I/L alternatives are separately labeled and never become exact-string MS observations. |

The protein FASTA can be plain or gzip-compressed; its **file-byte** SHA256 must
appear in `reference.asset_hashes`. The reference key is SHA256 of its UTF-8 JSON
with sorted keys, compact separators, and no nonfinite numbers. This matches
Canvax's `Reference.key`. Changing the reference requires matching occurrence
and sample keys, not just a new filename.

Create a file entry with:

```python
from pathlib import Path
from hitlist.provenance import file_digest

root = Path("inputs")
entry = {"path": "reference.fasta.gz", **file_digest(root / "reference.fasta.gz")}
```

## Expression, occurrences and candidate groups

`expression` is a JSON object containing `schema_version: 1`, the identical
`reference` object, and `occurrences`, `samples`, and `contributions` lists.
It accepts Canvax's schema-1 expression fields without a Canvax dependency.
Additional source fields are frozen unchanged.

Every occurrence supplies `occurrence_id`, `protein_id`, `gene_id`,
`transcript_id`, `sequence`, `reference_key`, `source`, and optionally
`complete` (default true). Protein IDs match FASTA accession tokens; UniProt
`sp|accession|name` and `tr|accession|name` headers use the accession. Complete
sequences normalize whitespace, case and one terminal stop, keeping I/L distinct.
An occurrence's normalized sequence must match its FASTA protein exactly.
Supply **all** known source occurrences, including noncandidate loci. Proteins
without occurrence annotations remain explicit unresolved mapping background.
Quarantined products must remain in the full FASTA even if they cannot enter the
validated occurrence list.

Samples supply `sample_id`, `study`, `donor` (namespaced identity or null),
`specimen`, `library`, `source`, `reference_key`, `assay`, `quantification`,
`unit`, `tissue`, `taxon`, and `health` (`healthy`, `tumor`, `disease`, `unknown`).
Other fields such as stage, sex, preparation, runs and QC remain in the frozen
input. Tissue strings must match the declared policy exactly.

Contributions supply `sample_id`, `allocation_id`, every compatible
`occurrence_ids`, `lower`, `upper`, and `scope` (`gene`, `promoter`, `transcript`).
They also retain `direct`, `coding_assignment`, original `provenance` and
inferential replicates. Bounds must both be missing or finite, ordered and
nonnegative. `direct` and `coding_assignment` default to true; scope defaults to
transcript, matching the interchange model. Set these explicitly when uncertain.

Allocation identities represent disjoint quantifier support. Exact duplicates
count once; conflicting duplicates and reused libraries across samples are
rejected. Gene/promoter and transcript estimates cannot be summed within a sample.
Only unambiguous transcript support establishes positive full-sequence abundance.
All alternatives remain visible; a missing source occurrence or measurement
leaves the combined upper bound unknown. An unannotated identical FASTA protein
also makes that upper bound unknown. Units and source/assay/quantifier namespaces
are never pooled. A counts or FPKM record cannot pass a TPM gate.

`candidates` is a JSON list, for example:

```json
[
  {
    "sequence_id": "<SHA256 of the normalized full protein sequence>",
    "admission_basis": "species_expression",
    "policy_id": "canine-testis-screen-v1",
    "support": [{"sample_id": "sample-1", "allocation_id": "measurement-1"}]
  }
]
```

Each support link must resolve to measured positive expression of an occurrence
in the sequence group. Human orthology or human CTA membership cannot supply
admission. The input candidate claim can survive gene-level uncertainty or
incompatible units, but it cannot pass the derived isoform/policy audit through
those measurements. Empty discovery/expression lists are permitted for a scoped
MS/reference replay and make no CTA discovery claim.

## Explicit policy and normal evidence

An example policy is below. These thresholds illustrate a research screen;
they are required inputs, not biological defaults or a safety certification.

```json
{
  "policy_id": "canine-testis-screen-v1",
  "version": 1,
  "allowed_tissues": ["testis"],
  "required_normal_tissues": ["heart", "brain", "lung"],
  "unit": "TPM",
  "allowed_min": 5,
  "normal_max": 0.5,
  "min_normal_donors": 2,
  "tissue_blacklist": {
    "version": 1,
    "tissue_status": "nonmalignant",
    "min_donors": 2,
    "tissue_groups": {"heart": ["heart"], "brain": ["brain"], "lung": ["lung"]}
  }
}
```

Expression audits report reproductive support, normal counterevidence and
missing normal coverage separately within each measurement namespace. An
identical protein expressed from an adult-heart locus remains counterevidence
at the shared sequence group. Coverage requires measured upper bounds and
distinct explicit donor identities. Missing samples, missing tissue names and
unknown abundance are not zeros.

`normal` is always an explicit Parquet table, even when empty. Its columns are
`peptide`, `donor_id`, `donor_status`, `source_tissue`, `tissue_status`,
`is_cell_line`, `source_record_id`, `source_taxon`, `source`, `assay_method`,
`response_measured`, and `qualitative_measurement`. Original fields, references
and typing can be additional columns. Donor IDs must be canonical across all
contributing studies, not publication-local aliases for the same animal.

The blacklist admits positive MS observations in nonmalignant primary heart,
brain or lung, with resolved donor IDs. It excludes a sequence after observations
in **at least two distinct donors across these tissues, on any allele**. Repeated
runs and different alleles in one donor do not create additional donors. Every
excluded row remains in the tissue audit with its reason. Tumors, cell lines,
unknown donors/tissues and non-MS assays do not qualify. Normal-source taxa
must match the reference; an empty snapshot must declare `coverage: "missing"`.
An empty forbidden-sequence file under missing coverage is not evidence of safety.

## Observations, contributors and lineage

`observations` is a scoped Hitlist Parquet export with original assay fields and
the source/host/presenting/MHC facets. Required fields are `peptide`,
`provenance_id`, `source`, `pmid`, `mhc_restriction`, `species`, `host`,
`presenting_species`, `mhc_species`, `condition_id`, `sample_attribution`,
`attributed_sample_label`, `assay_method`, `response_measured`, and
`qualitative_measurement`.

Add explicit `source_taxon`, `presenting_taxon`, and `mhc_taxon` integer columns
from reviewed source context; use zero for unknown. Preserve the reported species
names and the evidence supporting the taxon assignment. For PMID 42199926,
the three reviewed HCT116 transductant arms have human source/presenting cells
(9606) and canine MHC (9615). The five native tumor IP arms have canine
source/presenting cells; their peptide-level restrictions remain unassigned.
Do not derive source taxon from a DLA restriction or copy donor typing into that
restriction. Raw scanner output should first receive the reviewed arm context;
the bundle does not invent it from a peptide match.

Use explicit empty strings for unknown context; avoid nulls in required columns.
An existing unique `evidence_row_id` is retained, or Hitlist derives one from
the full required observation context. The complete `contributors` Parquet
must contain the existing Hitlist contributor schema, with exactly the set of
observation provenance IDs. Full original fields, row values and relationships
are retained. `lineage` is an explicit Hitlist schema-1 lineage JSON registry;
an empty registry leaves donor/specimen/acquisition identities unknown.
Lineage is recomputed using only this frozen registry.

Positive-MS admission is recomputed from original assay fields. Fluorescence,
structural and other non-MS assays remain excluded evidence. Native presentation
requires both source and presenting taxon to match the reference. Heterologous
MHC observations are separately labeled. Unknowns remain unknown. Exact support
means equality to the **reported peptide string**; I/L molecular discrimination
is not presumed. A longer observed ligand does not support its nested windows.

## Outputs, bounds and replay

The bundle freezes all inputs and emits:

- `mappings.parquet`: every full-reference match and source occurrence, with
  zero-based, half-open positions and separate exact/I/L-equivalent kinds.
- `presentation.parquet` and `excluded_observations.parquet`: original evidence,
  recomputed modality, context, lineage, and admission reasons.
- `expression_audit.json`, `group_expression.json`, `peptide_expression.json`:
  original-level bounds, allocation links, namespace-specific policy evidence
  and exact-mapping links. Bounds describe full sequence groups, not measured
  peptide abundance.
- `peptides.parquet`: reference mapping counts, exact reported MS counts split
  by context, expression counterevidence/uncertainty, normal coverage and blacklist.
- `tissue_evidence.parquet`, `tissue_risk.parquet`, `forbidden_sequences.txt`:
  the full normal-source audit and allele-independent donor exclusions.
- `input.json` and `manifest.json`: frozen input contract, reference key, limits,
  Hitlist version and SHA256/size for every artifact.

`candidate_specific_in_reference` means every exact match belongs to a supplied
candidate sequence group and has source annotation. It is **not** tissue
restriction: read the separate normal-expression and normal-MS fields. Mapping
completeness is scoped to the supplied full reference, including its limitations.

Default bounds are 256 MiB input files, 64 MiB per JSON input, 128 MiB uncompressed
Parquet metadata per table, 250,000 input rows per table/list, 512 MiB decompressed
FASTA, one million residues per protein, 100 million total reference residues,
50,000 peptide strings, 250,000 mapping/expression-link rows, and 512 MiB output.
The FASTA is streamed one protein at a time. A shared output budget checks bytes
before writes. Exceeding any bound removes staging files and publishes nothing;
an existing destination is never replaced. These are data/working-set controls,
not a claimed hard process-RSS limit.

Override named bounds through `limits={"max_mapping_rows": 500000}` when justified.
They are recorded in the manifest. Verification checks hashes and recomputes
mapping, expression, contributor, lineage and tissue relationships from frozen
inputs. It needs neither original source paths nor current curation registries.
