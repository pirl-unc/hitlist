# Search-scoped MS detectability datasets

Hitlist can join a theoretical digest to bulk peptide observations for Presto.
`build_detectability_training_set` returns a DataFrame;
`iter_detectability_training_set` streams batches; and
`export_detectability_training_set` writes Parquet plus a provenance manifest.
All three are available from `hitlist` and `hitlist.bulk_proteomics`.
`DetectabilitySearchSpace` is available from `hitlist` or `hitlist.detectability`.

An `observed=False` label means **not observed in the selected search scope**.
It does not establish intrinsic undetectability or complete digestion. The
same-scope parent-protein requirement controls one source of missingness; it
does not remove digestion, abundance, assignment, or acquisition biases.
Bulk shotgun evidence is separate from MHC presentation evidence.

## Search reference and settings

Supply the FASTA actually searched and an explicit search contract. A matching
checksum links the reference, observations, and search settings. A current
proteome cannot silently stand in for an unidentified historical reference.

```python
from hitlist import DetectabilitySearchSpace, export_detectability_training_set

# This JSON must describe the actual search: see the fields below.
search = DetectabilitySearchSpace.read("verified-search.json")
export_detectability_training_set(
    "presto-evidence",
    search_fasta="searched.fasta",
    search_space=search,
    cell_line="HeLa",
    digestion_enzyme="Trypsin/P",
    n_fractions_in_run=46,
    max_missed=2,
    length=(7, 30),
    require_protein_observed=True,
)
```

The contract requires `fasta_sha256`, `enzyme`, `max_missed_cleavages`,
`min_peptide_length`, `max_peptide_length`, `max_peptide_mass_da`,
`fixed_residue_modifications` (residue to monoisotopic mass delta), and
`provenance` (a reference to the verified search settings).
`DetectabilitySearchSpace.from_fasta(path, **settings)` computes the checksum.
`null` for either upper bound explicitly declares **no such bound**, not an
unknown bound. Do not fill missing historical settings with `null`.
The contract describes fully specific, unmodified-sequence candidates with
fixed residue modifications; the search must admit those forms. Variable
modifications may contribute to sequence-level observed labels. Protein
N-terminal processing and modification-specific detectability are not modeled.

Requested length/missed-cleavage limits must fit inside the recorded search
space. Candidate mass includes water and fixed residue modifications; candidates
above the search mass ceiling are excluded. Ambiguous/nonstandard residues are
excluded. FASTA duplicate identifiers, missing observed parents, or peptides
absent from their reported parent sequence fail before any labels are emitted.
FASTA headers may be UniProt `sp|accession|...` / `tr|accession|...`, or a plain
first-token identifier. The reference must contain the searched target proteins;
do not mix decoys into a target-only training reference without an explicit
search contract for that target subset.

## Historical Bekker-Jensen input

Defaults select HeLa, Trypsin/P, 46 fractions, no enrichment, pH 10, 7–30 residues,
at most two missed cleavages, and observed parents only. They resolve the
packaged observation scope, but **cannot supply the missing historical FASTA or
unrecorded search limits**. Calling without a verified search contract fails.

The original [PXD004452 deposit](https://www.ebi.ac.uk/pride/archive/projects/PXD004452)
records only a local `HUMAN.fasta` filename, no FASTA checksum or UniProt release.
The original ZIP contains no FASTA. Its README, ProteomeXchange metadata and
SDRFs do not resolve the identity. Linked reanalyses use other search references;
their reference must be paired with their own observations, not the original
labels. These unresolved source defects are tracked in
[#654](https://github.com/pirl-unc/hitlist/issues/654).

The packaged adapter reads compressed source CSVs in chunks, selecting one
curated scope before retaining rows. It avoids loading the complete built bulk
index. `data/bulk_proteomics/detectability.yaml` records original metadata
checksums, experiment names, acquisition-run counts, and these limitations:

- Correct publication: [PMID 28601559](https://pubmed.ncbi.nlm.nih.gov/28601559/).
  Legacy CSV/built-index readers correct the old unrelated PMID and retain
  `reported_reference`.
- Deposited parameters report MaxQuant **1.5.3.19**, minimum length **7**, fixed
  carbamidomethyl C, and **match-between-runs enabled**. The paper describes a
  different version and downstream MBR exclusion. Packaged peptide detections
  mean nonzero intensity, potentially including MBR; they are not asserted to
  be direct MS/MS identifications.
- Actual search allowances: Trypsin/P **3**, Chymotrypsin+ **4**, GluC;D.P **3**,
  LysC/P **2** missed cleavages. Contracts contradicting known limits fail.
- HeLa 46-fraction rows pool `Tryp-46fracs` and `HeLa-46fracs-IT-E1/E2`.
  The old ingest collapsed distinct replicate identities. The adapter retains
  the reported count in its hashed input table, returns a nullable detected
  count with `replicate_count_status="unresolved_aggregate"`, and records all
  three experiment IDs. The possible count is experiments, not proof of
  independent biological replicates.
- Protein abundance is the same-scope sum of intensities assigned to leading
  razor proteins. `protein_observed` records this assignment-based support;
  shared peptides do not prove each possible parent was independently observed.
  Percentiles use the existing 0–1 scale. Global modification annotations do
  not establish that a modification occurred in the selected arm.
- There is **no automatic depth comparison group**. The 14-fraction experiments
  were reused from an earlier study, and other protocols differ. First-seen
  depth is nullable for every packaged scope.

## Occurrences, labels, and provenance

Each candidate row preserves sequence, parent accession/gene, 1-based inclusive
positions, flanks, missed cleavages, fixed-modification mass, scoped `observed`,
replicate counts, parent observation/abundance, source, protocol, search-space
ID, enrichment, fractionation pH/depth and acquisition/assignment basis.
Repeated sequences at different positions and alternative parents remain
separate rows. Observation is joined **by sequence across the selected scope**:
a shared peptide assigned to another razor parent is never made negative.

Use `search_space.identifier` to attach the search contract to custom peptide
and protein tables. Supply both `peptide_observations` and
`protein_observations`, or neither. Both tables need `source`, `cell_line_name`,
`digestion_enzyme`, `n_fractions_in_run`, `enrichment`, `fractionation_ph`,
`protocol_id`, `search_space_id`, and `uniprot_acc`. Protein rows need positive
`n_peptides`, and optionally `gene_symbol` and `abundance_percentile`.
Peptide rows need uppercase unmodified `peptide`, `search_enzyme`,
`n_replicates_possible`, and one explicit value for each of `instrument`,
`fragmentation`, `acquisition_mode`, `labeling`, `search_engine`,
`detection_basis`, and `protein_observation_basis` in the selected scope.
`n_replicates_detected` may be nullable; complete `replicate_id` values can
establish counts directly. Conflicting aggregate counts are rejected.

DataFrame batches carry `attrs["detectability"]`. The export manifest preserves
reference and observation-table hashes, all query parameters, selected source
curation, search/acquisition controls, artifact checksum, and candidate counts.
Retain that manifest with the Parquet file; a CSV alone loses provenance.

## Optional comparable depth groups

For custom tables, `first_seen_at_n_fractions` is computed only when an explicit
`comparison_group` and `comparison_provenance` establish comparability.
Non-depth scope, search contract, acquisition basis, and replicate opportunity
must agree. `comparison_controls` is a JSON object recording identical
`lc_gradient_minutes`, `peptide_load_ug`, `sample_preparation` and
`acquisition_method`. Each depth needs one protocol with a JSON `experiment_ids`
list of source-qualified IDs. Reused experiment IDs, missing controls, or mixed
searches fail; they are never turned into a depth ladder. The manifest retains
the group, controls, provenance and protocol membership. This is a curated
assertion of comparability, not something Hitlist can infer from fraction count.

## Resource bounds

The iterator defaults to 25,000 candidates per batch, at most 2,000,000 candidate
occurrences, 1,000,000 observation rows per table, 256 MiB of compressed and
uncompressed FASTA, 1,000,000 residues per protein, and 500,000 FASTA identifiers.
All except the identifier ceiling are explicit keyword limits. Oversized input
or output raises an error; there is no silent truncation. Observation lookups
remain in memory within the row cap; the materializing builder also retains
all candidate batches, so use iteration/export for large references.

The exporter defaults to a 1 GiB total output cap (`max_output_bytes`), requires
a new destination, streams Parquet with Zstandard compression, and publishes
the directory only after complete validation. Failure removes temporary output.
A caller using the iterator directly must consume it fully and discard prior
batches on any error, including a late candidate limit or reference change.

## Digest compatibility

`digest_occurrences` and `digest` now share verified
[Cox Lab search definitions](https://github.com/cox-labs/PluginTutorial/blob/25bdb094d6b23c1f4c3e07aa4c766e43fa4bee5b/PluginTutorial/conf/enzymes.xml).
Trypsin excludes K/R–P cuts; Trypsin/P includes them. Chymotrypsin cuts F/W/Y;
Chymotrypsin+ also cuts L/M. GluC cuts E; GluC;D.P additionally cuts D–P pairs.
LysC excludes K–P; LysC/P includes it. The contradictory old long Trypsin/P
alias now raises an error. Choose an exact search name; legacy bulk biological
labels remain readable and the packaged adapter maps them from source evidence.
