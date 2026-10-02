# Literature-backed species contexts

`is_chimeric`, `is_engineered_mhc`, and `xenograft` are derived from species
labels in the observation corpus. They are useful for finding records to
review, but they cannot prove that those same labels describe a legitimate
cross-species experiment.

Hitlist therefore packages an independent primary-literature registry in
`hitlist/data/species_contexts.yaml`. Import `load_species_contexts` or
`generate_species_contexts_table` from `hitlist.species_contexts` to inspect the
manifest or its tabular records. The registry includes resolved findings,
partial findings, unresolved candidates, the frozen inventory counts, and the
remaining PMID review queue.

Run `python scripts/species_context_inventory.py OUTPUT_DIR` against a verified
observation corpus to reproduce the candidate-PMID and flagged-row inventory.
The frozen counts identify the 1.64.3 corpus used to select this review pass;
rerunning against a later corpus is expected to produce a new snapshot.

## Separate biological axes

Each record keeps these facts distinct:

- `presenting_species`: cells or tissue carrying the immunoprecipitated MHC.
- `in_vivo_host_species`: organism containing the material when harvested.
- `lineage_host_species`: an earlier host in the material's passage history.
- `introduced_mhc_species`: species origin of a transfected or transgenic MHC.
- `supported_foreign_species`: peptide-source species independently supported
  by a deliberate antigen exposure or culture supplement.
- `reviewed_candidate_species`: candidates examined during review, including
  candidates the paper did not support.

For example, PMID 32502341's B-ALL cells were human cells recovered from NSG
mouse spleens, so the record has a human presenting species and a mouse in-vivo
host. PMID 39111711 used a cultured human cell line whose earlier history
included mouse xenograft passage, so mouse is the lineage host and the in-vivo
host at sampling is blank. PMID 27893789 expressed canine DLA in human C1R and
K562 cells, so canine is the introduced-MHC species and human is the presenting
species.

## Evidence and uncertainty

`species_context_status` has three values:

- `resolved`: the primary source explains every candidate relation in this
  record's scope.
- `partial`: the experimental system is established, but one or more candidate
  source or host assignments remain unsupported.
- `unresolved`: the available primary source does not establish the candidate
  relation at the required scope.

An unsupported candidate stays in `reviewed_candidate_species`; it is never
copied into `supported_foreign_species`. Missing mention is not negative
evidence.

## Observation attachment

A literature record reaches `ms_samples` and observation/training exports only
when it explicitly names a curated `condition_id` from the same PMID. A record
with no `condition_ids` remains visible in the complete registry table and does
not annotate observations. This is how mixed papers can be reviewed before
their deposited rows can be assigned to the paper's individual experiments.

When an observation cannot be assigned to one arm, Hitlist may retain a species
axis shared by every candidate arm. It clears the literature record's ID,
status, scope, citation, reviewed candidates, and note because those identify a
specific review record.
