# Evidence catalog (Hitlist 2.0 design)

The [Hitlist 2.0 plan](https://github.com/pirl-unc/hitlist/issues/733)
defines one evidence vocabulary without claiming that each indexed row is
ready to train a model. The MHC assay and biological-context migrations now
materialize typed results, measurements, arms, factors, protocol claims and
lineage. Later PRs add supporting modalities, T-cell assays and the common
query and training interface.

## Discover types and requirements

```python
from hitlist import describe_catalog, describe_evidence, describe_task, list_evidence

print(list_evidence(kind="observation"))
print(describe_evidence("tcell_availability")["grain"])
print(describe_task("mhc_quantitative_binding")["required"])
catalog = describe_catalog()  # JSON-serializable; no assay read or download
```

Unknown types and fields raise errors. A field listed here is a **logical v2
field**; its evidence status says whether it has a v2 materialization or is
still a contract. It is not a claim that every row has a value. Every task reports
`status="v2_contract_only"`. In particular, an indexable observation can lack
a task's peptide, endpoint, unit or cellular identity. The later training
recipe will report those eligibility failures per row.

| Logical type | Native materialization | Grain |
| --- | --- | --- |
| `mhc_positive_ms`, `mhc_binding`, `mhc_other` | Three Parquet partitions | One MHC assay result |
| `tcell_availability` | Pinned assay metadata export | One distinct IEDB assay ID |
| `tcell_functional_peptide`, `tcell_multimer_peptide` | Bounded training projections | One qualified source assay ID |
| `bulk_ms_peptide`, `bulk_ms_protein` | Bulk Parquet, selected by granularity | One material/protocol measurement |
| `line_expression_gene`, `line_expression_transcript` | Line-expression Parquet, selected by granularity | One gene/transcript profile measurement |
| `peptide_protein_occurrence` | Peptide mappings Parquet | One peptide/protein/position alternative |
| `source_contributor`, `reviewed_lineage`, `curated_material_arm` | Contributor Parquet and curated registries | Linked provenance/context |
| `assay_context`, `curated_study`, `arm_factor`, `arm_mhc_claim`, `arm_control_link`, `protocol_claim`, `curation_amendment` | Corrected human cohort and versioned curation | Assay context or one source claim |

## MHC evidence package

`write_mhc_evidence_package(source_dir, destination)` takes a complete portable
v4 corpus and writes one self-contained package. It keeps all native corpus
files, including contributor `original_fields` and `source_row_values`, then
adds `typed/mhc_observations.parquet` and `typed/mhc_measurements.parquet`.
The package manifest binds all files by hash, names the row/link keys and
assigns every one of the 71 native MHC fields to a typed, deferred or
source-only disposition. Its source registry retains the 23 raw source snapshot
hashes and states that original row payloads are available through the
contributor relation while upstream file access and license remain unverified.
It is published only after row and link reconciliation and
`verify_mhc_evidence_package` succeed.

```python
from hitlist import write_mhc_evidence_package, verify_mhc_evidence_package

manifest_path = write_mhc_evidence_package(pinned_v4_directory, output_directory)
manifest = verify_mhc_evidence_package(manifest_path.parent)
print(manifest["reconciliation"])
```

The pinned v4 conversion contains 5,327,796 distinct MHC results and the same
number of measurement rows across the three partitions. All 9,381,101 source
contributor links resolve, with no result lacking a link. The `mhc_other`
partition stays queryable: its 23,537 negative MS rows do not become positive
presentation labels, and structural or non-MS ligand results remain in their
own modalities. A source `positive_result` is interpreted within its assay
family. Tested MHC, raw reported MHC, a candidate set, and peptide restriction
remain distinct; this migration does not assign a peptide-level restricting
allele. A missing assay method on a curated MS supplement stays missing, with
the admission basis recorded separately. Peptide modifications, non-peptide
ligand and potential contaminant flags are typed inputs to future eligibility
rules.

The [v4 MHC reconciliation report](https://github.com/pirl-unc/hitlist/blob/main/tasks/mhc-v2-evidence-results/README.md)
records source, partition and link parity for the 1.82.19 package.

## Biological context package

`write_context_evidence_package(mhc_package, cohort_bundle, destination)`
augments a verified MHC package into one atomic generation. The output keeps
the original MHC native files, the complete corrected human-cohort training,
identity, contributor and lineage files, the cohort-time PMID curation, and
the current PMID curation. It binds every file by hash. A later catalog/query
reader opens only this generation; source directories are ingestion inputs,
not query-time sidecars.

```python
from hitlist import write_context_evidence_package, verify_context_evidence_package

path = write_context_evidence_package(mhc_package, cohort_bundle, output_directory)
manifest = verify_context_evidence_package(path.parent)
print(manifest["reconciliation"]["n_condition_link_rows"])
```

The corrected cohort covers 4,071,463 of the 5,327,796 MHC assay results.
All 71 fields it shares with the pinned v4 assay rows match by source ID,
including nulls; its 90 additional columns are context, protocol, lineage and
quality assertions at the same assay grain. The 2,042,689 cohort rows with a
`condition_id` point to 489 reviewed `(PMID, condition_id)` arms. The package
also retains 318 curated arms with no linked assay row in that bounded human
cohort. Each assay link retains `sample_attribution`; 48,975 linked rows use
`class_pool`, which is coarser than a direct sample statement. A missing arm
link does not mean an untreated arm.

Curated arms are study-arm keys, not physical specimens. The source revision
used for the cohort and the current curation revision retain the same 807 arm
keys. The later revision adds 35 source-backed cell changes on 27 arms,
including explicit control absences and soluble-MHC capture context. The
typed amendment relation shows both values rather than rewriting the 1.78.0
cohort table. `arm_factors` keeps concurrent interventions as distinct claims;
`none` means sourced explicit absence, while a blank field remains unreported.
When dose or duration appears only in prose, it remains in the raw condition
text; this migration does not parse or infer a numeric exposure from it.
Each typed claim retains whether its basis is curated text, a primary source,
independent cellular typing, or a broader study/arm curation statement.
Curated studies and documented control-to-comparison links have their own
typed relations; an untreated label alone is not a comparator link.
MHC candidates, selected restriction, independent cellular genotype and
capture/expression context have distinct roles. Study protocol defaults and
arm overrides remain separate; no instrument setting is promoted to a
per-run claim. Donor, specimen and acquisition identity remain unknown in this
cohort export.

The two T-cell projections share the availability source population. The
157,263 CEDAR T-cell IDs that overlap IEDB are contributor evidence, not extra
assays. A peptide/protein mapping is a relation, not another assay row.
Bulk protein and peptide views share one physical file but have different
measurement grains. Line expression is measured abundance on a stated scale;
a same-named cell line is not proof of the same specimen or intervention arm.

## Roles and null meanings

The catalog separates molecular inputs, biological context, measured targets,
protocol descriptors, provenance, lineage and relations. Applied cytokine
exposure is biological context; cytokine release is an assay endpoint.
Candidate MHC molecules for a material, a molecule tested in a binding assay,
peptide-level restriction and independent cellular genotype have distinct
fields. Instrument and readout describe measurement, not biological inputs.

`unreported` means no source claim; `explicit_absence` means a sourced
negative condition assertion; `not_applicable` means the field does not apply;
`unresolved` and `conflicting` preserve uncertainty. A missing condition is
never silently converted to an untreated control. Source values, censor
qualifiers and endpoint-specific units survive until a documented recipe
chooses a valid target transformation.

## Pinned baselines

`describe_catalog()["baselines"]` labels each count by generation and unit.
For the corrected 2026-10-16 v4 corpus, the baselines are 4,486,991 positive
MHC MS, 814,268 binding and 26,537 other MHC assay rows; 2,269,288 bulk
proteomics and 42,909 line-expression measurement rows. The pinned T-cell
availability audit has 578,529 distinct IEDB assay IDs. Its bounded training
projections have 47,421 functional and 8,423 multimer examples. These are
checks for later migration reports, not counts of independent donors,
specimens, or peptide–HLA assignments.

See [assay modality](assay-modality.md), [training provenance](training-provenance.md)
and the [T-cell availability audit](https://github.com/pirl-unc/hitlist/blob/main/tasks/tcell-availability-results/README.md)
for the present native contracts.
