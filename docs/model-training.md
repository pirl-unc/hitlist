# Training Presto and MHCflurry

Hitlist's core exports (`generate_training_table`, `generate_binding_table`, and
`write_training_bundle`) preserve reported biology, assay phrasing, units,
inequalities, and alternative source-protein mappings losslessly. Training
predictors such as [MHCflurry](https://github.com/openvax/mhcflurry) and
[Presto](https://github.com/pirl-unc/presto) requires explicit task qualification
on top of those exports so downstream code never needs ad-hoc endpoint regexes,
unit guesses, silent inequality drops, arbitrary multi-mapping flank choices, or
conflation of study-wide allele unions with single-sample genotypes.

Every `prepare_*` function in `hitlist.model_training` (also re-exported from
top-level `hitlist`) accepts either an in-memory `DataFrame` or a verified
`training_bundle` directory path and returns a
`TrainingTableSelection(accepted, excluded, summary)` tuple conserving 100% of
candidate observations across `accepted` and `excluded`.

## Quantitative binding affinity (`nM`)

`qualify_binding_measurements` classifies binding rows without dropping any rows,
and `prepare_affinity_training_data` selects molar affinity measurements
standardized to `nM`:

- **Endpoint families (`MOLAR_AFFINITY_ENDPOINTS`)**: `ic50`, `kd_approx_ic50`,
  `kd_approx_ec50`, `kd`, and `ec50`. `IC50` is never relabeled `KD`; pass
  `endpoints=("ic50",)` when a model head requires literal IC50 supervision and
  `MOLAR_AFFINITY_ENDPOINTS` when training pan-allele affinity ensembles.
- **Linear-only unit conversion**: converts commensurate molar units (`fM`,
  `pM`, `nM`, `uM`/`µM`, `mM`, `M`) to `nM` while preserving
  `raw_measurement_value` and `raw_measurement_units`. Nonlinear or unitless
  scales (for example `log10(IC50)` or unitless `qualitative binding` scores)
  are excluded with `nonlinear_or_unsupported_unit` or
  `endpoint_not_selected:...`, never exponentiated.
- **Censoring and blank operators**: preserves `<`, `<=` (mapped to
  `measurement_inequality="<"` with `censoring_status` `left_censored_lt` /
  `left_censored_lte`) and `>`, `>=` (mapped to `">"` with
  `right_censored_gt` / `right_censored_gte`). IEDB numeric rows whose operator
  was left blank carry `censoring_status="unreported_operator_numeric"`;
  `blank_inequality_policy="treat_as_exact"` (default) sets
  `measurement_inequality="="` while keeping that provenance flag, and
  `blank_inequality_policy="exclude"` routes them to `excluded`. Rows with an
  inequality operator and no numeric value are excluded with
  `operator_without_value`.
- **Allele completeness**:
  - `scorable_class_i_only=True` restricts to two-field human classical class-Ia
    proteins accepted by MHCflurry's `Class1AffinityPredictor`.
  - With `scorable_class_i_only=False` (default for Presto), complete class-I
    alleles (`complete_class_i`), complete alpha/beta class-II pairs
    (`complete_class_ii_pair`), and human `HLA-DRB*` chains
    (`single_chain_class_ii_dr`, whose `HLA-DRA` groove is monomorphic) are
    accepted, while unpaired polymorphic `DP`/`DQ` single chains are excluded
    (`unpaired_class_ii_chain`) unless `allow_unpaired_class_ii_chains=True`.

### MHCflurry class-I affinity example

```python
from hitlist import prepare_affinity_training_data
from hitlist.export import generate_training_table

binding = generate_training_table(include_evidence="binding")
selection = prepare_affinity_training_data(
    binding,
    scorable_class_i_only=True,
    min_peptide_length=5,
    max_peptide_length=15,
)

df = selection.accepted
# Columns ready for Class1AffinityPredictor.fit(...) plus Hitlist provenance:
# allele, peptide, measurement_value, measurement_inequality, measurement_type,
# measurement_kind, measurement_source, censoring_status, evidence_row_id,
# provenance_id, assay_preparation, donor_applicability, ...
print(selection.summary)
```

### Presto multi-species / literal-IC50 affinity and stability example

```python
from hitlist import prepare_affinity_training_data, prepare_stability_training_data

# Strict literal IC50 across all cataloged species and MHC classes:
ic50 = prepare_affinity_training_data(
    "human-cohort-bundle",
    endpoints=("ic50",),
    species=None,
)

# pMHC kinetic half-life (min) and thermal dissociation temperature (celsius):
stability = prepare_stability_training_data(
    "human-cohort-bundle",
    species=None,
)
```

`prepare_stability_training_data` standardizes `half_life` to `min` (from
seconds, minutes, or hours) and `thermal_stability` (`50% dissociation
temperature`, `melting temperature`) to `celsius`. Keep `measurement_kind`
(`"half_life"` vs `"thermal_stability"`) separate when fitting distinct heads.
Literal `0.0` half-life rows (scintillation proximity assay nondetections or
subthreshold dissociation) are routed to `excluded` with
`non_positive_measurement_value` rather than treated as exact zero half-lives,
and engineered point-mutant MHC constructs (for example
`HLA-B*14:01 C67S mutant`) are routed to `excluded` with
`mutant_mhc_construct_excluded` unless `allow_mutant_mhc=True`.

### Source-qualified NetMHCstabpan 1.0 half-life ingestion (`#875`)

Hitlist can acquire, verify, and reconcile the published NetMHCstabpan 1.0
`Stability.txt` archive (`28,166` 9-mer measurements across `75` HLA-A/B
constructs; Rasmussen et al. 2016, `PMID 27402703` / `PMC4976001`) either as part
of a full `hitlist build observations` index rebuild (`hitlist data fetch
netmhcstabpan && hitlist build observations`) or directly in memory against an
existing `binding.parquet` table:

```python
from hitlist import (
    fetch_netmhcstabpan_source,
    load_netmhcstabpan_stability,
    prepare_stability_training_data,
)
from hitlist.observations import load_binding

stability_txt = fetch_netmhcstabpan_source()
iedb_binding = load_binding()

# Drops only confirmed DTU IEDB direct-submission duplicates (1028282..1028294)
# while retaining cross-study overlap warnings and full line/construct provenance:
stab_rows = load_netmhcstabpan_stability(
    stability_txt,
    binding_df=iedb_binding,
    deduplicate_confirmed_iedb=True,
)
selection = prepare_stability_training_data(stab_rows)
print(selection.summary)
```

## MS presentation and multi-instance learning (MIL)

When `generate_training_table(..., map_source_proteins=True)` expands a peptide
across multiple source proteins or isoforms, `prepare_ms_training_data`
collapses those rows back to **one row per `evidence_row_id`** so multi-mapped
ligands never receive inflated observation weight:

- `unique_mapping`: single mapped protein and flank pair.
- `concordant_flanks`: multiple proteins/genes share identical `(n_flank, c_flank)`;
  flanks are retained while ambiguous gene/protein scalars are blanked and full
  candidates remain in `candidate_protein_ids` / `candidate_gene_ids`.
- `expression_disambiguated`: conflicting flanks resolved to one flank pair via
  `with_peptide_origin=True` sample expression (`origin_transcript_id` /
  `origin_gene_id`) when `use_expression_disambiguation=True`.
- `ambiguous_flanks`: conflicting flanks with no unique expression winner;
  `n_flank` and `c_flank` are set to `""` rather than picking `.iloc[0]`
  arbitrarily, while all candidate pairs remain in `candidate_flank_pairs` JSON.
- `unmapped` / `not_mapped_in_input`: no source-protein hit or unmapped input.

Candidate presenting alleles are qualified into explicit `candidate_bag_status`
categories:

- `monoallelic_single`: single four-digit allele from a monoallelic line or
  construct.
- `experimental_single`: single experimentally evidenced four-digit allele on a
  multi-allelic background.
- `predicted_single_restriction`: excluded by default; retained only when
  `allow_predicted_restrictions=True`.
- `sample_candidate_bag`: sample-matched candidate allele bag from
  `mhc_allele_set` (for MIL deconvolution).
- `study_pool_candidates`: study-wide allele union (`pmid_class_pool`,
  `pmid_locus_pool`, `pmid_serotype_pool`), excluded by default so multi-donor
  study pools are never mistaken for single-sample genotypes.

### MHCflurry presentation / antigen-processing inputs

MHCflurry's presentation trainer expects at most six class-I alleles per sample
genotype. Pair `scorable_class_i_only=True` and `max_alleles_per_bag=6` with
`to_mhcflurry_presentation_inputs`:

```python
from hitlist import prepare_ms_training_data, to_mhcflurry_presentation_inputs
from hitlist.export import generate_training_table

ms = generate_training_table(
    include_evidence="ms",
    map_source_proteins=True,
    with_peptide_origin=True,
)
ms_selection = prepare_ms_training_data(
    ms,
    scorable_class_i_only=True,
    max_alleles_per_bag=6,
    min_peptide_length=8,
    max_peptide_length=15,
)
hits_df, sample_alleles = to_mhcflurry_presentation_inputs(ms_selection)
```

`hits_df` carries one row per accepted MS observation (`peptide`, `sample_id`,
`n_flank`, `c_flank`, `flank_status`, `evidence_row_id`, `provenance_id`, and
lineage columns), and `sample_alleles` maps each `sample_id` to `1..6` two-field
human class-I allele strings.

## Assay preparation and biological-independence audits

Every table returned by `generate_training_table`, `qualify_binding_measurements`,
and `prepare_*` includes `#706` lineage-applicability columns via
`attach_lineage_applicability`:

- `assay_preparation`: `purified_mhc`, `cellular_mhc`, `lysate_mhc`,
  `cell_associated_ms`, `generated_cellular_consensus`, or `unknown`.
- `donor_applicability` and `specimen_applicability`:
  - `not_applicable` on unresolved `purified_mhc` binding assays (recombinant
    cell-free protein preparations have no biological donor or tissue specimen),
  - `applicable_resolved` or `applicable_unresolved` on cellular binding,
    lysate assembly, and MS elution rows,
  - `applicability_unknown` when `assay_method` leaves preparation unspecified
    or conflicting.
- `experimental_origin_applicability` and `acquisition_applicability` remain
  applicable across all preparations because recombinant protein batches,
  peptide libraries, and assay runs can still be shared across partitions.

This classification separates cell-free recombinant IC50/KD/stability assays
from cellular assays with unresolved donor/specimen lineage in
`hitlist.audit_splits` and `hitlist.training_bundle.audit_training_bundles`
without altering historical `<axis>_status` values or falsely certifying an
unreviewed partition split as `pass`. See
[Training provenance and split audits](training-provenance.md) for full bundle
and split-audit contracts.
