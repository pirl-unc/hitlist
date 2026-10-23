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
| `digestion_members` | Original nested archive member, including retained-only assets |
| `digestion_metadata` | Original identifier-workbook row with column names and values |
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
The 391-member archive inventory identifies which original files have typed
rows and which remain original-only (for example logs, TICs and plots).

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
Numeric rows expose `measurement_unit=source_MS1_peak_area_arbitrary_units` and
different `quantity_scale` values for Skyline peaks, aSPIRE run quantities and
background-adjusted biological means. The paper reports label-free MS1 peak
area without further normalization; these scales are not molar units.

Two RF1 acquisitions named as time-zero controls have time one in the raw
quantity layer (2,278 rows). Both times survive, and the whole affected condition
is marked `unresolved_source_time_conflict`. Relabeling final averages would not
resolve their contributor history. Six contaminant-digestion enzyme labels also
contradict filename labels; those claims remain separate conflicts.
The 152 Skyline rows with blank source filenames keep their original rows and
carry `blank_original_source_name`; no main-search run is invented.

## Contiguous time-course recipe (2.0.9)

The registered `digestion_contiguous_timecourses_v1` recipe produces one curve
per source condition, I/L-equivalent peptide and reported biological label.
It retains all 53,413 final source vectors in 50,372 disposition groups. The
reviewed candidate admits 30,566 curves across 15 construct groups; the rest
have explicit, sometimes multiple exclusion reasons. Five conditions have no
admitted curve: RF1 retains its time conflict, CaM/IF2/UBE2S lack the required
source-backed zero-time coverage, and the Ffh 1:25 condition lacks sufficient
repeated common positive times after the source baseline rule.

The target is positive, background-adjusted biological-mean MS1 peak area at
reported nonzero digestion times, in `source_MS1_peak_area_arbitrary_units`.
It is conditional on the deposited final table. Whether that table exactly
matches the paper's additional positive 4 h (3 h for alpha-synuclein)
selection remains unresolved. Forced zero-time values,
missing/below-background slots and unobserved products are retained as source
states, never treated as negative or zero targets. Product eligibility requires
the source's contiguous PCP claim and exactly one I/L-equivalent span in its
literal deposited construct. Short and nonbinding fragments remain eligible.
Original aliases, spliced products and ambiguous positions survive in the
source package and disposition ledger.

An admitted curve needs actual zero-time raw quantity records matching the
count of filtered technical records, zero-time entries for all filtered
technical labels, three positive nonzero times, and a second reported
biological label with at least three common positive times. The source does not
prove that filtered technical labels identify paired physical acquisitions.
The same source parent and all Ffh ratios share a construct group. The frozen
five-fold split has 15 groups and guards exact source peptide overlap, but not
sequence homology or study independence. Source biological labels are not
independent donors. No curve is a molar fragment amount, APC concentration,
TAP observation or MHC label.

```json
{"schema_version": 1, "name": "digestion_contiguous_timecourses_v1", "registered_recipe": "digestion_contiguous_timecourses_v1"}
```

Pass this JSON to `EvidenceStore.materialize(recipe, output)` or
`hitlist evidence materialize PACKAGE recipe.json OUTPUT`. The export contains
source-linked curves and all candidate decisions, exact input Parquet tables,
the source manifest, declared contract and output hashes. Verification
recomputes every target and disposition from those frozen tables. The recipe
does not independently validate the original search, synthesis-error removal
or paper-level product-selection decisions, and it does not supply an MS response
likelihood. Biological fitting remains a separate step.

## Verification scope

Construction validates pinned assets, nested CRCs, run joins, source census,
literal constructs and vector alignment. Portable package verification checks
the complete artifact inventory and hashes, source contracts, schemas and row
counts. Run `python scripts/verify_digestion_roundtrip.py PACKAGE ORIGINAL_ASSETS
DECODED_RDATA` with the separately exported original RData tables from the
pinned source audit. It checks archive members, all typed original CSV/RData
rows, workbook cells, constructs and final vector slots.
Run `python scripts/verify_digestion_arithmetic.py PACKAGE REPORT_DIR` to
independently replay all 545,664 raw quantities and 280,664 final slots from
the original nested MS1 archive and source tables. Run
`python scripts/verify_digestion_timecourses.py PACKAGE RECIPE_OUTPUT` to
independently replay every curve decision, source slot, target and construct
split without importing the recipe producer. The [review report](https://github.com/pirl-unc/hitlist/blob/main/tasks/digestion-recipe-209-results/README.md)
records the exact manifests, counts and remaining scientific limits. These
checks do not establish the original search, mapping or synthesis-error
decisions as biologically correct; that scope remains open under #770.
