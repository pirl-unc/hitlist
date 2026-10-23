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

```python
from hitlist import EvidenceStore

store = EvidenceStore.open("/path/to/evidence-package")
catalog = store.catalog()
print(catalog["package_manifest_sha256"])
print([(item["name"], item.get("n_rows")) for item in catalog["evidence"]])
```

The audited schema-10 successor contains 116 materialized evidence types and
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
