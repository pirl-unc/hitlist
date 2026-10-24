# Hitlist 2.0 evidence store

`EvidenceStore` opens a frozen package and verifies its manifest chain, source
bytes, typed schemas and pinned reconciliation before answering a request.
The package contains the original source generations as well as typed relations;
the API makes each relation discoverable without searching for sidecars. The
package directory is the only location a reader supplies.

New external-source generations use package schema 9 and catalog 14. Their
predecessor can be a schema-7 reference package or a schema-8 adjudication
package. The latter retains all nine multimer reviews when MAPP and calibration
sources are appended. Both historical schema-8 families remain readable:
`review_revision` identifies adjudications and `appended_sources` identifies the
earlier source drafts. The released schema-9/catalog-11 MAPP, catalog-12
calibration and catalog-13 controlled-MAPP generations remain readable.
Ambiguous envelopes are rejected.
Actual source presence,
rather than the version number alone, determines available relations in the
verified generation.

A source-reviewed binding successor uses package schema 10 and catalog 15. It
embeds the complete verified schema-9 predecessor, the original article XML,
the review ledger, and both typed review relations. The same package path opens
all inherited and new evidence. Every predecessor file is retained and checked
against its original digest.

A Gomez-Zepeda source successor uses package schema 11 and catalog 16. It
inherits the entire verified schema-10 package and embeds the original
publisher S3 and S5 ZIPs, their source registry, original result rows, and
run-level peptide memberships. The old 68,552 peptide observations remain
byte-identical. Source rows carry both the original CSV columns and typed
common fields. The source archives are CC BY 4.0; all other sources retain
their own rights scope.

```python
from hitlist import EvidenceStore

store = EvidenceStore.open("/path/to/evidence-package")
catalog = store.catalog()
print(catalog["package_manifest_sha256"])
print([(item["name"], item.get("n_rows")) for item in catalog["evidence"]])
```

The audited schema-10 successor contains 116 materialized evidence types and
the schema-11 successor adds the two original Gomez-Zepeda relations. Both have
three declared contracts. The predecessor numeric and reference packages remain
readable and report later types as `not_in_generation`. The catalog
identifies grain, source population, endpoint, ingest requirements,
physical selector, native columns and relationships. Each logical field is
`physical`, `constant`, `related` or `unavailable`. A source-specific column
without a safe common alias remains available as `native.<column>`. A missing
field request raises `EvidenceFieldError`; unknown evidence types, predicates
and unsafe relationships fail explicitly.

Counts in the catalog are rows at their stated grain. Do not add rows from
result, measurement, contributor, context and mapping relations as if they were
independent assays. The v4 MHC partitions contain 4,486,991 positive MS,
814,268 binding and 26,537 other assay results. The 578,529 T-cell availability
rows are assay IDs; the bounded training projection contains 47,421 direct
functional and 8,423 multimer examples. The 2,269,288 bulk and 42,909
expression source rows include linked numeric representations without becoming
MHC or T-cell labels.

## Binding protocol review

`binding_assay_context` has one row for each of the 814,268 v4 binding assay
results, keyed by the original `observation_id`. Its original source assay ID,
PMID, method, endpoint, raw measurement, unit and inequality are immutable
anchors. `review_status` is `source_reviewed_study_scope` for the 938 results
from PMID 28481943 and `not_source_reviewed` for every other result. The 938
include 910 results in the corrected human-source-organism cohort and 28
influenza peptide controls outside that cohort. An unreviewed row has a null
`protocol_id`; it does not assert that protocol details were absent.

`binding_protocol_claim` has one row per source-scoped protocol assertion, not
one per assay or biological replicate. The current 21 claims cite the article
URL, original XML hash, Methods section and supporting excerpt. Statuses
distinguish a reported study-wide condition, a locus-specific condition, an
unresolved range or alternative, a reported interpretation, and an unreported
field. Source silence never means explicit absence. In this study, the exact
temperature, tracer concentration and peptide purity of an individual assay
are unresolved. The paper's conditional statement about IC50 approximating KD
is retained as interpretation; no KD measurement or label is created.

```python
reviewed = store.query(
    "mhc_binding",
    select=["observation_id", "reported_measurement.endpoint_reported",
            "binding_assay_context.review_status",
            "binding_assay_context.protocol_id"],
    joins=["reported_measurement", "binding_assay_context"],
    where={"binding_assay_context.review_status": "source_reviewed_study_scope"},
    limit=100,
)
temperature = store.query(
    "binding_protocol_claim",
    select=["protocol_id", "assertion_value", "claim_status",
            "claim_applicability", "source_locator"],
    where={"assertion_predicate": "incubation_temperature_alternatives"},
)
```

To construct this generation, call
`write_binding_assay_context_package(predecessor_package, source_xml, destination)`
with the verified schema-9/catalog-14 package and the pinned Europe PMC
`PMC5421785/fullTextXML` bytes. The writer checks the source hash, pinned v4
and cohort manifests, exact assay identities and every inherited file before
publishing the successor. See
[the source review and audit](https://github.com/pirl-unc/hitlist/blob/main/tasks/binding-assay-context-210-results/README.md).

## Gomez-Zepeda original result lineage

`gomez_source_result` is one original S3 or S5 CSV data row, including all
original columns under `native.<CSV column>`. The common fields include
`source_phase`, `condition_text`, `biological_replicate_label`,
`sample_label_raw`, `technical_replicate_label`, `acquisition_id`,
`peptide_sequence`, `peptide_modifications_raw`, `reported_peptidoform_raw`,
`charge_raw`, `identification_method_raw`, `identification_confidence_raw`,
`identification_q_value_raw`, `tims_ramp_raw`, `polygon_raw`,
`quantitative_raw` and `log2_area_raw`.
`quantitative_raw` is the original `Area` cell; its unit is source peak-area
arbitrary units. Blank and zero remain distinct. A CSV row does not identify
an independent donor, specimen or assay. `biological_replicate_label` is a
literal source cell and is not uniformly resolved to an independent culture;
in S3 it can simply name the cell line.

`gomez_peptide_membership` is one detected sequence in one reported phase,
condition, culture label, sample and acquisition file. It gives the number of
original source rows behind that membership. Each membership links to the
merged derived 8–15-amino-acid peptide observation where the source permits an
exact sequence and cell-source match. Original results outside that length
remain queryable with a null derived link and explicit
`derivation_link_status`. JY and Raji peptides often occur in both WT and
transfected conditions; the one merged peptide observation does not by itself
identify a unique arm. Missing sequence membership is not a negative result.

```python
source = store.query(
    "gomez_source_result",
    select=["source_record_id", "source_phase", "condition_text",
            "sample_label_raw", "acquisition_id", "peptide_sequence",
            "quantitative_raw", "native.mokapot_q_value"],
    where={"source_phase": "s5_spike_experiment", "cell_line_name": "JY"},
    limit=100,
)
memberships = store.query(
    "gomez_peptide_membership",
    select=["membership_id", "condition_text", "sample_label_raw",
            "peptide_sequence", "n_source_result_rows", "derived_observation_id"],
    where={"cell_line_name": "JY", "peptide_sequence": "AAAAAAAAA"},
)
```

To build this successor, call
`write_gomez_source_package(predecessor_package, archives_directory, destination)`.
The archive directory must contain the publisher's pinned `MOESM7.zip` and
`MOESM9.zip`. The writer checks the exact archive and member hashes, source
row counts, five derived peptide unions, complete derived coverage, and every
predecessor file before publication. Request the membership relation
explicitly when condition or run detail is needed; joining it onto merged
peptide observations would multiply training weights.

## Query one evidence grain

`query` returns an Arrow table and metadata. Its default limit is 1,000 rows;
pass `limit=None` only when materializing a manageable subset. `iter_query`
streams Arrow batches for large requests.

```python
binding = store.query(
    "mhc_binding",
    select=["observation_id", "peptide_sequence", "tested_mhc",
            "reported_measurement.endpoint",
            "reported_measurement.quantitative_value",
            "reported_measurement.measurement_unit",
            "reported_measurement.measurement_inequality"],
    joins=["reported_measurement"],
    where={"reported_measurement.endpoint": "qualitative binding"},
    limit=100,
)

knockouts = store.query(
    "arm_factor",
    select=["arm_id", "curation_revision", "factor_type", "factor_value",
            "claim_status", "evidence_basis"],
    where={"factor_type": "knockout_genes"},
)
cytokine_exposures = store.query(
    "arm_factor",
    select=["arm_id", "factor_type", "factor_value", "claim_status"],
    where={"factor_type": "cytokines"},
)
```

An arm factor is a biological condition claim. A T-cell soluble factor release
is an assay readout. Querying them uses the same grammar, while their source
roles and endpoints remain distinct. `assay_context` connects human MHC assay
rows to reviewed arms when the source attribution supports a link; unknown
donor and specimen identities remain unknown. `arm_factor` is one claim per
factor, so its many-valued relation is never broadcast onto an assay row.

```python
bulk = store.query(
    "bulk_ms_peptide",
    select=["observation_id", "peptide_sequence", "search_enzyme",
            "n_fractions_in_run", "protocol_scope.search_enzyme_status"],
    joins=["protocol_scope"],
    where={"search_enzyme": "Trypsin/P"},
)
protein_values = store.query(
    "bulk_numeric_measurement",
    select=["measurement_id", "observation_id", "source_dataset",
            "quantity_scale", "quantitative_value", "source_value_field"],
    where={"quantity_scale": "abundance_percentile"},
)
expression = store.query(
    "line_expression_measurement",
    select=["measurement_id", "observation_id", "source_dataset",
            "quantity_scale", "quantitative_value", "measurement_unit"],
    where={"quantity_scale": "tpm"},
)
```

Bulk peptide rows report detection, not a protein abundance value. Bulk protein
and gene-expression numeric cells are separate `measurement` evidence types
linked by `observation_id`; each source scale remains distinct. `n_peptides`
in a protein source row is a support count, not a quantitative abundance target.

```python
external = store.query(
    "external_mhc_peptide_quantity",
    select=["observation_id", "measured_pool",
            "peptide_source_row.peptide_sequence",
            "assay_channel.measured_population",
            "assay_channel.partner_population",
            "material_link_precision"],
    joins=["peptide_source_row", "assay_channel"],
)
instrument_claims = store.query(
    "external_context_assertion",
    select=["assertion_predicate", "assertion_value", "source_scope"],
    where={"assertion_predicate": "instrument"},
)
```

The PXD057588 instrument claim has project scope; it is not a measured-run
instrument field on each result. Its MHC-bound macrophage peptides and digest
peptides are separate result types. The tumor coculture partner is not the
measured macrophage. `external_quantity_measurement` holds source intensities
and reported/recomputed fold changes with their representations and derivation
links. The source peptide row and independently keyed extra worksheet panels
are distinct relations.

## Mouse reference and source roles

The [pinned PXD057588 reference audit](https://github.com/pirl-unc/presto/blob/24f1d2b/experiments/2026-10-22_1914_codex_pxd057588-shared-sequence-admission/README.md)
uses the reviewed mouse UniProt 2026_03 reference including isoforms. The
successor package keeps its FASTA and raw mapping ledgers under one verified
manifest. `external_reference_source_mapping` has one row for each of the
35,772 original peptide rows; `external_reference_occurrence` has 52,277
possible exact parent positions over 30,587 distinct sequences. Positions,
parent identities and connected components are alternatives and grouping
constraints, not new measurements or evidence of specimen pairing.

```python
species = store.query(
    "external_reference_species_role",
    select=["population_role", "population_label", "organism_taxon_id",
            "source_scope", "source_locator"],
    limit=None,
)
mouse_signal = store.query(
    "external_quantity_measurement",
    select=["measurement_id", "reported_value", "measurement_unit",
            "reference_context.source_record_id",
            "reference_context.source_parent_status",
            "reference_context.measured_population",
            "reference_context.partner_population",
            "reference_context.specimen_status",
            "reference_context.component"],
    joins=["reference_context"],
    where={"representation": "raw_tmt_intensity"},
    limit=20,
)
```

The organism claim is **Mus musculus (10090)** at PRIDE project scope. The
result context records that scope explicitly and retains
`assay_channel_unresolved_specimen`. Table 10 MHC-I loss belongs to the
`mhc_i_loss_tumor_cell` partner; its measured population is macrophage.
Missing source parent IDs (24) and the one resolved ID without the peptide
remain distinct from 35,747 exact source-parent matches. A reference match
does not establish the study's original search database or a measured protein.
Query parent support as its own `external_reference_parent_support` relation;
it holds source-row sets under literal, verified-source and exact-reference
definitions without multiplying either Table 9 ligand or Table 11 digest
quantities.

```python
tcell = store.query(
    "tcell_availability",
    select=["source_assay_id", "assay_family", "assay_method",
            "assay_readout", "qualitative_result"],
    where={"assay_family": "soluble_factor_release"},
)
direct_peptide = store.query(
    "tcell_functional_peptide",
    select=["source_assay_id", "peptide_sequence", "qualitative_result",
            "assay_family", "restriction_evidence"],
)
```

Availability metadata cannot supply an unverified peptide identity. Functional
response and multimer cell binding are different training projections.
CEDAR's 157,263 overlapping T-cell IDs remain contributors, not extra assays.

Nine source-discordant pMHC multimer assays have a reviewed interpretation at
`source_assay_review` grain. The review is a separate relation, not a replacement
outcome or an extra assay. It preserves IEDB's raw qualitative and subject-count
claims as validated anchors, links the primary paper or supplement, and states
what remains unresolved. All nine retain their existing exclusion from
peptide-level training. The [source-linked audit](https://github.com/pirl-unc/hitlist/blob/main/tasks/multimer-adjudication-results/README.md)
reconciles the full 578,529 assays and 55,844 training examples.

```python
review = store.query(
    "assay_adjudication",
    select=["source_assay_id", "review_status", "interpretation_code",
            "source_row_locator", "publication_url",
            "reported_outcome.qualitative_result",
            "eligibility.eligibility_status", "antigen_claim.reported_mhc"],
    joins=["reported_outcome", "eligibility", "antigen_claim"],
    limit=None,
)
```

The assay-to-review relationship is optional one-to-one. Source contributors
remain a separate many-row relation, so a query cannot silently multiply an
assay when one record appears in both IEDB and CEDAR. Schema-6 and schema-7
packages and historical schema-8 external-source packages remain readable;
their catalogs mark the review relation as absent.

## Training recipe

Hitlist 2.0 removes the `hitlist export training` command. Use
`hitlist evidence materialize` with a versioned recipe so every target has
an explicit grain, admission decision and source-manifest binding. The old
source export implementation remains available to reproduce historical
audits, but it is outside the 2.0 training interface.

The recipe is YAML or JSON. It names a result grain, a stable row key, exact
target interpretation, unit/scale and censoring policies where applicable,
context precision and grouping policy. `mapping_policy: no_expansion` keeps
multiple protein or candidate-MHC alternatives in their source relations.

```yaml
schema_version: 1
name: positive_mhc_ligand_example
mapping_policy: no_expansion
sources:
  - evidence_type: mhc_positive_ms
    row_id: observation_id
    select:
      - peptide_sequence
      - source_dataset
      - reported_measurement.endpoint
    joins: [reported_measurement]
    required_fields: [peptide_sequence]
    target:
      field: reported_measurement.qualitative_result
      mode: mapped
      dtype: boolean
      map: {Positive: true}
    grouping:
      fields: [source_dataset]
      on_missing: mark_uncertain
```

This grouping example keeps a source generation together; it does not claim
donor or specimen independence. A task that needs specimen-disjoint splits
must request source-backed specimen lineage and mark missing groups ambiguous
or excluded. The recipe outputs one row per queried candidate in exactly one
of `accepted`, `excluded` or `ambiguous` under `sources/<evidence_type>/`.
It also writes `contributors.parquet`, `lineage.parquet`, `coverage.json` and a
manifest. Coverage reconciles source rows, query exclusions, explicit limits
and every admission decision. Unknown qualitative tiers, unsupported units,
censored numeric values, unresolved context and grouping each have reason codes.
`contributors.parquet` has every linked IEDB/CEDAR source record for a selected
MHC or T-cell candidate. It can therefore have more rows than the accepted,
excluded and ambiguous candidate tables combined, without creating extra
training labels. Other evidence types emit their direct source key or an
explicit unresolved link status.
Numeric targets require explicit unit or scale, value-status and censoring
policies when the source records those axes. Accepted numeric targets are
stored as typed float values with `target_unit`; the source cell and all
alternatives remain in the evidence package. A recipe can declare a linear
`unit_conversion`, for example
`{output_unit: nM, factor_by_unit: {uM: 1000.0}}`. Every allowed source unit
must have a factor. Censored inequalities such as `>` remain ambiguous for
an exact numeric point target; they are never silently treated as equality.

For a numeric PXD057588 recipe, use `external_quantity_measurement` as the
candidate grain, join `reference_context`, select its `source_record_id`, and
group by `reference_context.component`. An explicit context policy can admit
condition-level observations while retaining unresolved specimen status.
Table 11 digest intensity is still a peptide measurement. Protein aggregation,
complete-case filtering and condition-surrogate matching need a separately
declared recipe; this interface never silently creates a precursor estimate or
interprets an unobserved peptide as a zero.

```bash
hitlist evidence catalog /path/to/evidence-package --type arm_factor
hitlist evidence query /path/to/evidence-package arm_factor \
  --select factor_type --select factor_value \
  --where '{"factor_type":"knockout_genes"}' --limit 5
hitlist evidence materialize /path/to/evidence-package recipe.yaml training-output
hitlist evidence verify-recipe training-output \
  --source-package /path/to/evidence-package
hitlist evidence build-reference /path/to/numeric-package \
  /path/to/pinned-reference-results /path/to/mouse-reference.fasta \
  /path/to/new-evidence-package
```

`hitlist evidence query --all --output data.parquet` streams a full relation.
The source package itself is immutable. Its construction writers and pinned
source receipts are documented in [the migration catalog](evidence-catalog.md)
and the [serial implementation plan](https://github.com/pirl-unc/hitlist/blob/main/tasks/pr8-evidence-store-spec.md).

# Evidence package snapshots

Each evidence-package generation snapshots its verified inputs and predecessor
files into new inodes before atomic publication. On filesystems that support
copy-on-write cloning, the snapshot shares storage blocks until either path is
modified; otherwise the writer makes a full copy. Editing an ingestion file or
an earlier package after publication cannot change a later package's bytes.
Each generation still records file hashes in its manifest, and verification
checks those hashes before the data is used.

# Gomez-Zepeda source-backed arm projection

Schema 12 adds `gomez_source_arm`, `gomez_source_membership_arm`,
`gomez_derived_arm_support`, `gomez_derived_arm_projection` and
`gomez_assay_context_projection`. These are additive relations over the
verified schema-11 source rows: the original S3/S5 result and membership
tables remain unchanged, and the predecessor context remains available at
`typed/assay_context.parquet`.

The arm table has nine biological arms: JY wild type, JY S1 and S2
transfectants, Raji wild type, Raji S1 and S2 transfectants, HeLa, SK-MEL-37,
and commercial human plasma. S5 mappings are derived from the native
`ConditionGenotype`, `Condition`, `Genotype`, and `BioReplicate` fields; S3 JY
rows are retained as wild-type method-comparison memberships. A derived peptide
is `unique_source_arm` only when all linked memberships support one arm. When
memberships support several arms, the projection keeps the complete arm set,
sets `arm_id` to null, and records `multi_source_arm`; this is evidence of
shared support, not a guessed donor or negative label.

The context projection contains exactly the Gomez-derived observations and is
therefore safe to join by `observation_id` without implying that unrelated
assays were rewritten. Every source row, source asset, membership, acquisition,
and historical context remains queryable through the predecessor bindings.

# Gomez-Zepeda prediction panels

Schema 13 adds `gomez_prediction_row` and `gomez_restriction_projection` over
the schema-12 arm projection. The two pinned NetMHCpan panels retain all
394,952 publisher rows, including alternative alleles, scores, ranks, Binder
tiers, raw source cells, and rows that do not join to a retained derived
peptide. The source prediction relation is a model measurement; it is not a
measured MHC-ligand or binding assay.

The projection is one row per Gomez-derived observation. It preserves the
native restriction value and evidence, records the preferred panel (S7 for JY,
S5 for the other materials, with S5 fallback where the preferred JY panel has
no Binder row), and stores selected and alternative prediction-row IDs. A
named native restriction is exposed as `predicted`; a blank native restriction
remains `unknown` with an explicit reason. No native assay row, source result,
or source membership is rewritten or removed, and a predicted restriction
does not establish a donor, specimen, or experimental assay label.

# Binding assay context availability

Schema 14 adds `binding_context_availability` as a one-row-per-binding-result
audit over the corrected human binding cohort. It preserves endpoint, method,
raw value, unit, inequality, source identifiers, and native assay comments,
then records whether each requested context axis is row-native, available only
as a source-scoped study claim, unreported, unresolved, or not reviewed.

The audit keeps literal IC50 and `KD (~IC50)` in separate endpoint families;
`kd_conversion_status` is always `not_performed` in this revision. The existing
binding protocol claims remain source-scoped and are not broadcast into assay
rows. Joining this relation to `mhc_binding` is one-to-one on `observation_id`,
so adding context cannot multiply assay or training-example weights.

# Native cellular peptide measurements

Schema 17 / catalog 22 exposes corrected revision `tmab_native_peptidomics_v3` of the
standalone `tmab_peptidomics` package family for the
Dasgupta 2014 and Gelman 2013 TMAB native-peptide supplements. It is not an
MHC-elution package and does not infer processing, knockout, absolute
concentration, donor identity, or specimen independence from a drug label.

The package has five homogeneous relations: hash-pinned source workbooks,
lossless source rows, experiment-level protocol context, one measurement per
reported ratio channel cell, and a lineage assessment per measurement. Raw
ratio text is retained alongside parsed values; bounds, missing signals,
overlapping peaks, and nonnumeric values have distinct statuses. Heat-map
republications remain source rows with an explicit deferred disposition rather
than being silently counted as independent measurements.

Build it from caller-supplied, hash-verified workbooks:

```python
from hitlist import write_tmab_peptidomics_package

write_tmab_peptidomics_package("/path/to/tmab-assets", "tmab-package")
```

Query `tmab_measurement` for native ratios and join `tmab_protocol_context`
for cell line, treatment, extraction, labeling and instrument details.
`tmab_measurement_lineage` flags possible cross-paper reuse but leaves
independence `unknown` until source reconciliation establishes it.

The superseded schema-15 and schema-16 revisions are rejected by this corrected reader; rebuild
from the same pinned original workbooks instead of relabeling old packages.
Source row/measurement IDs and raw quantities are preserved. Tentative or
unsupported identities, unknown sequences, nonpositive ratios, controls and
derived averages have explicit deferred source eligibility. Exact and censored
positive treated-ratio candidates still need an independence-qualified recipe.

`sequence_identity_status` is separate from `value_status`; supported acetylation
and methionine oxidation retain source tokens and position-bearing JSON.
Unsupported pyroglutamate/dehydration notation remains unresolved until a parent
mapping or source adjudication establishes its identity. Source-independent
fingerprints expose possible cross-paper reuse without asserting specimen identity.
Unmatched rows remain unresolved. Shared-control groups are explicitly source-label
proxies, not reconstructed physical denominators.

Protocol context distinguishes the publishing protocol's instrument from unresolved
original acquisition for reused measurements. It retains both source timing labels,
parsed incubation/harvest intervals, compound presence during harvesting and
primary Methods links. The 2014 protocol retains inhibitor during its harvest wash;
20 minutes incubation plus 15 minutes handling supports 35 minutes total exposure.

## Native response recipe

`tmab_native_response_v1` is registered in the common store catalog and materializes
grouped exact/censored treated-control responses. Supply a hash-pinned reviewed
human FASTA as an explicitly modern model reference:

```python
store.materialize({
    "schema_version": 1,
    "name": "native-response",
    "registered_recipe": "tmab_native_response_v1",
    "reference": {
        "path": "/path/to/human-reviewed.fasta",
        "sha256": "<sha256 of the exact input>",
        "source_url": "<original public reference URL>",
        "generation_label": "<recorded reference generation>",
        "license": "CC BY 4.0",
        "role": "model_reference_not_original_search",
    },
}, "native-response-recipe")
```

The immutable recipe snapshots the reference and native tables, retains every
exact sequence occurrence/parent alternative and all source dispositions, and
never treats a protein description as an accession or measured origin. Its
components connect all source sequences, reference alternatives, source parent
assertions and possible reuse before eligibility or split assignment. Charge
states and reported ratios contribute within a peptide-form/condition group;
matching source copies share one observation weight. Different-label candidate
reuse remains deferred. Exact and bounded observations remain separate, with
condition/component/form/observation weights normalized within each split.

These are peptide/parent component splits within the same studies. Shared-control
label proxies may cross them; physical denominator, donor and specimen identities
remain unknown. They do not establish homology or external biological independence.
Chemical exposure remains separate from genetic loss, and native extract ratios
are not absolute APC concentrations. Full-parent model support is reported
separately from sequence-response admission. Use `verify_recipe_package` with the
source package to replay every quantity, contributor, split and reference occurrence.

# T-cell and processing evidence

Schema 18/catalog 23 introduces the standalone `tcell_processing` package.
It copies the source-faithful T-cell v2 typed relations and the controlled
PXD009092 MAPP relations into one package/query envelope while keeping
functional responses, pMHC multimer binding, receptor evidence, direct
processing availability and captured products as separate families.

The family and availability ledgers distinguish materialized, unobserved and
unavailable source families. A missing receptor or direct cleavage/transport
row is not a measured negative. Query aliases include
`tcell_functional_response`, `pmhc_multimer_binding`,
`processing_measurement`, and `mapp_controlled_product`; the underlying
source-specific relations remain available for provenance joins. Donor or
specimen independence is not inferred, and MAPP captured products are not
intracellular concentration or direct cleavage-rate labels.

# Standalone proteasome processing evidence

Schema 19/catalog 24 introduces the standalone `digestion_processing` package
for the Soh et al. purified-human-20S proteasome time-course source. It retains
the original archive members, substrate constructs, metadata, runs and
instrument claims, identifications, MS1 rows, raw quantities, filtered and
final vectors, final value slots, and explicit source conflicts. The package
copies the pinned schema-9/10 source appendage without carrying unrelated MHC,
T-cell or bulk evidence files.

The package ledger separates raw MS1/search evidence, source raw quantities,
filtered/final time-course vectors, and conflict provenance. The registered
aliases `processing_ms1`, `processing_raw_quantity`, `processing_timecourse`,
`processing_run`, `processing_context`, and `processing_conflict` query those
relations through `EvidenceStore`. Source time conflicts, forced zeros,
missing/censored values, annotation-only rows, non-contiguous products and
splice/synthesis dispositions remain source statuses; they are not converted
to measured negatives. The ledger explicitly marks APC/TAP/clearance/MHC
processing as unavailable, because purified proteasome digestion measures
relative MS response rather than intracellular production or concentration.

# Standalone precursor context

Schema 20/catalog 25 begins the standalone `precursor_context` package for the
PXD009360 parallel tryptic proteome. The first package generation retains the
hash-pinned project and deposited-file metadata, the recovered MaxQuant search
archive, parameter and summary files, its 17 archive-member inventory, and nine
acquisition labels (448–456). The source archive is copied into the immutable
package; later generations will materialize its evidence, peptide, all-peptide
and protein-group rows.

The source contract deliberately exposes condition, specimen, aliquot and MAPP
pairing as unresolved claims. Run order, label similarity, abundance
correlation and same-cell-line assertions do not establish a biological link.
`precursor_context_family` and `precursor_context_availability` therefore show
zero precursor training examples in this first generation, while
`precursor_source_asset`, `precursor_acquisition`,
`precursor_table_inventory`, and `precursor_pairing_assertion` remain queryable
through `EvidenceStore`. MaxQuant LFQ and future precursor aggregates are not
absolute copies-per-cell or intracellular concentration labels.
