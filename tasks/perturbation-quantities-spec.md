# Original treatment/control quantities for Presto — #901

Implement a reusable perturbation-quantity evidence family using the existing
EvidenceStore catalog, typed relations, joins and registered recipe APIs. Start
with all twelve same-line comparisons in Stopfer 2020 (PMID 32488085,
PXD017407): four melanoma cell lines, 1/10 uM palbociclib or 10 ng/mL IFNG,
72 hours, each with its own reported DMSO channels. Source acquisition is the
original public Supplementary Data 3 and 5 workbooks plus article protocol.
Reserve 2.1.17 after dependent releases; verify before publishing.

## Source contract and qualification

- Preserve the exact immutable workbook bytes, hashes, every nonempty source
  row, positional headers, cached values, formulas and source-cell locators.
  No source workbook is edited or recalculated.
- Curate sheet/column mappings explicitly. Most sheets contain hipMHC-corrected
  reporter intensities; raw channels exist only for SKMEL5 palbociclib sheets.
  Preserve raw, hipMHC-normalized and mean-adjusted values as distinct kinds.
  Never describe these reporter quantities as copies/cell or absolute abundance.
- Titles and file maps identify 10 uM sheets whose channel headers erroneously
  say 1 uM. Preserve the original header and expose the qualified dose with an
  explicit source-conflict annotation. Data 3's file map uses obsolete Table 2
  numbering; reconcile only when cell-line/dose agrees with the unique sheet.
- An embedded Grand Total row in IPC298 10 uM is an aggregate, not a peptide.
  Preserve it with an excluded disposition. Retain phospho-[S] molecular forms;
  never silently pool them with unmodified sequences. Preserve pooled precursor
  AUC rows from Table 3.9 separately from treatment-specific reporter quantities.
- Source titles/Methods specify three biological replicates per arm and an
  unpaired test. Channel slots are source sample IDs scoped to their plex,
  never paired replicate matches, donors or proven physical specimens. Treat
  DMSO channels in different plexes as distinct source slots; cross-plex physical
  reuse remains unknown. Acquisition links describe the plex, not six independent
  runs or peptide-specific detections in every raw file.
- Preserve all reported protein/gene labels without assigning one protein of
  origin. The sample HLA genotype is a candidate set, not a peptide restriction.
  No nondetection, imputation, missing control or zero quantity is invented.
- Retain Methods' Percolator q <= 0.05 for labeled samples and the explicitly
  separate within-condition hipMHC normalization when fold changes exceed two.
  Independent all-sheet replay reconciles 17,763 comparisons / 106,578 positive
  finite normalized reporter cells; reported log2 effects agree within 1.01e-9.
  Preserve the original value and recomputation delta, including Data 3.8 row 1666.

## Public behavior

Expose assets, source records/dispositions, conditions, samples, plex/run links,
peptidoforms, measurement cells and reported summaries through common queries.
Use exact compound source identifiers and explicit unit suffixes on counts.
One registered contrast recipe uses positive finite hipMHC-normalized channels
for the same form and plex: mean(treated)/mean(control), log2 ratio, sample counts
and dispersion. No arbitrary replicate pairing or cross-plex pooling. Retain
reported effect/p-value separately and reconcile differences rather than replacing
source values. Complete and filtered outputs include exclusions and provenance.

Do not append these again as independent generic MS observations: original
PXD/PMID lineage and peptide/form identities must remain available for deduplication
against IEDB, CEDAR and supplementary contributors. Source admission is distinct
from Presto TRAIN eligibility; preserve frozen experiments and reserve checks.

## Verification and release

Test small representative source fixtures for positional headers, misleading
dose labels, aggregates, missing versus zero, forms, incomplete/unpaired contrasts,
identity conservation, tampering and common API joins/filters. Independently
replay the original workbooks and reconcile every source row/cell and each effect.
Run format/lint and unchanged guarded tests; final-head hosted gates and clean-main
release must pass. Publish source artifacts through the existing managed-asset
mechanism so consumers can reproduce them without a private scratch directory.

## Checklist

- [x] Retrieve original Data 3/5 and inspect all sheet headers and file maps.
- [ ] File source discrepancies and pin assets/explicit source contract.
- [ ] Implement normalized evidence package and common query bindings.
- [ ] Implement and independently verify registered contrast recipe.
- [ ] Qualify real-source outputs and train/evaluation boundaries.
- [ ] Pass required gates, review, merge and deploy.
