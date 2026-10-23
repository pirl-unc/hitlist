# Purified-proteasome digestion time courses

The optional `soh2024_digestion` source preserves the original processed
MSV000092631 / PXD044463 deposition from Soh et al. (2024),
[doi:10.1038/s41467-024-45339-3](https://doi.org/10.1038/s41467-024-45339-3).
It is purified-enzyme fragment MS evidence. It does not measure an APC's free
peptide pool or supply calibrated peptide concentrations.

Supply the 18 original, hash-pinned assets listed in
`hitlist/data/evidence_revisions/digestion_assets.json`: the nested processed ZIP,
identifier workbook, methods document and 15 literal substrate FASTAs.
Construction requires `Rscript` to decode the original serialized data frames.
It records R/exporter versions and original/decoded hashes. It never runs the
author's analysis pipeline. Published packages retain the original RData inside
the ZIP, typed Parquet values and original text fields. Package readers do not
require R. Original data are caller-supplied, not bundled with Hitlist.

```sh
hitlist evidence append-sources PREDECESSOR OUTPUT \
  --source-root soh2024_digestion=ORIGINAL_ASSETS
```

Use a verified reference or adjudication predecessor. To preserve previously
appended sources, include their source roots in the same command. An external
successor cannot itself be used as the predecessor. New generations use package
schema 9/catalog 14; prior catalog 11/12/13 external generations remain readable.

## Relation grains and context

| Evidence type | Grain |
| --- | --- |
| `digestion_metadata` | Original identifier-workbook row |
| `digestion_constructs` | Exact deposited substrate sequence |
| `digestion_runs` | Original main-search MS acquisition |
| `digestion_contexts` | Purified digestion condition and scoped protocol |
| `digestion_identifications` | Original assignment row |
| `digestion_ms1` | Original Skyline row, including fragment/charge/modification |
| `digestion_raw_quantities` | Original peptide/run quantity |
| `digestion_filtered_vectors` | Original filtered technical-replicate vector |
| `digestion_final_vectors` | Reported vector or annotation-only row |
| `digestion_final_values` | One original vector slot; repeated times retained |
| `digestion_conflicts` | Contradictory source assertions |

All use the same EvidenceStore discovery/query interface and expose original
columns through `native.<field>` and source-row JSON. Acquisition, construct,
condition and final-vector links are explicit. A query joining raw quantities
to acquisition metadata can inspect both supplied time and replicate labels.
Replicate labels do not create assured biological specimen pairs.

The 17 condition archives contain 15 substrates, including three Ffh ratios.
Context keeps human enzyme species, HeLa enzyme-preparation origin, substrate
species, input amounts, enzyme:substrate ratio, temperature and buffer separate.
Publication-wide protocol values are scoped claims, not measurements made for
every acquisition. Unknown substrate species stays unknown; human enzyme does
not imply a human substrate. This is not an APC cell-type or machinery-loss label.

## Missingness, conflicts and admission

Reported zero, missing/nonfinite source value, postprocessing-forced time-zero
value and unresolved missing contributor state remain distinct. Duplicate time
slots are not averaged again. Annotation-only rows produce no numeric samples.
Peptide modifications, contiguous/spliced assignments and alternative positions
remain original claims. Neither missingness nor a short product becomes a
negative binding or concentration label.

Two RF1 acquisitions named as time-zero controls have time one in the raw
quantity layer (2,278 rows). Both times survive, and the whole affected condition
is marked `unresolved_source_time_conflict`. Relabeling final averages would not
resolve their contributor history. Six contaminant-digestion enzyme labels also
contradict filename labels; those claims remain separate conflicts.

There is **no registered training recipe for this source yet**. Other conditions
are `pending_registered_recipe_and_independent_replay`. Typed source fidelity
does not establish correct biological observation likelihood or independently
replay all preprocessing arithmetic. Before fitting, verify the admitted
quantitative transformations, control joins, product mapping, temporal signal,
grouped split and treatment of MS response. Raw MS signal is not molar copies.

## Verification scope

Construction validates pinned assets, nested CRCs, run joins, source census,
literal constructs and vector alignment. Portable package verification checks
the complete artifact inventory and hashes, source contracts, schemas and row
counts. An independent original-source replay additionally compares every
original selected row and normalized value with all canonical query rows.
These checks do not claim that the original search, mapping or quantitative
analysis is scientifically correct. Quantitative arithmetic and training
admission remain separate gates tracked in Hitlist issue #770.
