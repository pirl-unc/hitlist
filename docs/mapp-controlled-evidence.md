# Controlled MAPP observations

The `wolflevy2018_controlled` adapter preserves two Wolf-Levy 2018 PRIDE panels
using the same source, peptide, identification, run, quantity and context tables.

| Project | Run design | Observation scope |
|---|---|---|
| PXD009508 | alpha6/isotype/GAPDH capture × untreated/epoxomicin × three numbered repeats | Captured-fragment MS signal and capture controls |
| PXD007263 | source TNF labels at 0/1/2/6 hours × three numbered repeats | Proteasome-captured fragments during cytokine exposure |

These are acquisition repeats. Original specimen, aliquot and physical pairing
identities remain unknown. Human species comes from project metadata; HEK293
identity and treatment dose/duration have publication-level scopes. The literal
`alpha6` capture label remains visible and does not silently become a
source-supplied PSMA1 gene identifier. Epoxomicin exposure is pharmacological,
not a gene knockout or a measured residual activity fraction. The time-course
label says TNF while the paper describes TNF-alpha plus IFN-gamma; no assured
single-cytokine name is supplied by the normalized channel.

The adapter keeps every original selected table row, both literal acquisition
and intensity-column labels, all confidence/identification fields, reported
quantities, reconstructed evidence sums and contributor edges. Unique full-label
punctuation aliases are derived claims, not specimen links. Protein FDR 1 is
preserved separately from PSM FDR 0.01. Matching-between-runs evidence remains
distinct from direct MS/MS. Reported zeros and missing intensity are different
states, and neither becomes a biological negative.

Original `Human_UP_2016_02.fasta` bytes remain unavailable. The adapter exposes
all exact occurrences and original parent claims against two later human model
references, with version, sequence, accession and coordinate evidence. Mapping
alternatives never multiply measured quantities or establish unique precursor
abundance.

## Canonical queries and run recipe

Append this source with any other required sources to a verified reference or
adjudication predecessor. New packages use schema 9/catalog 13. Existing
catalog-12 packages remain readable; optional relations are exposed only when
their source is actually present.

```sh
hitlist evidence append-sources PREDECESSOR OUTPUT \
  --source-root pxd009092=MAPP_ASSETS \
  --source-root pxd004886=CALIBRATION_ASSETS \
  --source-root wolflevy2018_controlled=CONTROLLED_ASSETS
```

`processing_*` evidence types expose the shared relation grains through the
same query API as other sources. `processing_material_context` and
`processing_material_link` distinguish source project, study arm and acquisition
membership from a physical specimen/aliquot link. `processing_channel` includes
capture, treatment, dose/time, evidence scopes and nullable material identities.
`processing_quantity_result` reports direct-MS/MS and matched contributor counts.
Every original column is also available through `native.<field>`.

Materialize registered recipe `mapp_controlled_run_intensity_v1` for positive
response analysis. Each example is one reported positive peptide/run intensity
in log10 arbitrary units, with its source measurement ID, run context and
confidence. Admission requires a nonreverse/noncontaminant standard sequence,
finite peptide PEP ≤0.01, one source protein group and positive reported and
reconstructed values. It does not impose an MHC-length or binding filter and
does not require a single reference parent. All excluded measurements and their
reasons remain in the disposition ledger and source tables.

The recipe preserves direct-MS/MS coverage so a downstream contrast can declare
stricter identification requirements. It does not silently normalize each run
to sum to one: total captured signal and relative composition are different
endpoints. Missing measurements are not imputed into targets.

Train/validation/test groups use only sequences and reference alternatives.
Identical peptide sequences and peptides sharing an exact full protein sequence
in either later reference form connected components. The deterministic split
assigns components in expected 70/15/15 proportions; actual counts are reported.
This is exact sequence/parent separation within one study, not a guarantee of
homology separation or independent biological specimens.

No raw MS quantity here is a calibrated intracellular concentration. Precursor
abundance, capture recovery and peptide-specific MS response remain unresolved.
Cytokine exposure time is not time since substrate synthesis. Those distinctions
must remain explicit in any availability or cleavage-kinetics model.

## Complete-case inhibitor response

`mapp_controlled_complete_alpha6_response_v1` applies a declared response cohort
to the same run admissions: PXD009508 alpha6, all three positive untreated and
all three positive epoxomicin runs, with at least one direct-MS/MS identification
in each arm. It preserves the original component/split and all six measurement
IDs and values, reporting arithmetic mean intensity separately for each arm.
It does not create physical replicate pairs or normalize the source quantities.
The base run examples, all run exclusions, all peptide contrast decisions and
original source/context/contributor tables remain in the export. Verification
recomputes the means and membership from those source tables.

This selects on positive signal in both arms. It cannot measure detection of
unobserved peptides or total intracellular products; a conditional-response model
must state that it requires an observed untreated baseline. Use the base run
recipe for wider observation/missingness work. The strict complete-case cohort
and all zero/missing exclusions remain explicit, not silently converted into
absence or negative concentration labels.
