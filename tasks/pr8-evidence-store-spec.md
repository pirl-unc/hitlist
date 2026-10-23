# PR 8 — one query and training recipe interface (#739)

## October 22 acceptance amendment: shared mouse reference evidence

Issue #739 now includes the independently verified PXD057588 mouse-reference
admission from `pirl-unc/presto` commit `24f1d2b`. Its exact 10 output hashes
and the UniProt 2026_03 isoform-inclusive FASTA SHA-256
`41bc0fa6cb9118ba95f330a27b18c39559e9397b6d99d1e73cc094869c99787a`
have been checked against local immutable inputs. The new generation must
preserve the current 103-file numeric package byte-for-byte, then append the
reference source and output ledgers under a new manifest schema. No source
quantity, assay result or peptide row is multiplied by a parent mapping.

The successor adds typed, keyed relations for 35,772 source-row mapping
statuses, 30,587 distinct source sequences, 52,277 exact position
alternatives, 25,758 reference parent identities (with matched sequences
where available), 26,449 Table 9/11 parent-support sets, and explicit
project-scoped mouse species roles. An assay-result context lookup links each
of the 309,126 unchanged quantitative results to its source peptide row,
mapping status, measured versus partner population and unresolved specimen
status. Numeric measurement recipes use this one-to-one lookup; the
one-to-many occurrence and parent-support evidence stays as separate rows.
The source parent's missing-ID, resolved-but-peptide-absent and exact-match
states remain distinct. Sequence components are candidate split constraints,
not donor, specimen, homology or independent-subject identities.

Implement the successor writer/verifier, catalog bindings, named links,
recipe example and source/endpoint audit before opening the PR. Verify every
new raw hash and parent manifest, all added row counts and join keys, the
reference spans against the pinned FASTA, and unchanged 4,486,991/814,268/
26,537 MHC, 578,529 T-cell availability, 309,126 external result and 509,850
external numeric counts. Species comes from the PRIDE project sample attribute
for these six panels with its project scope explicit; no human default or
per-run/specimen claim is created. Keep Table 10 tumor-partner MHC loss and
Table 8/10 source conflicts at their reported scope. A condition-surrogate
precursor adjustment requires a separate explicit aggregation recipe and is
not inferred from digest intensity.

Prerequisite: PR 7 (#751) merged; publish 1.82.24 before merging this PR. Branch
from merged commit `b124ecb` and bump to 2.0.0. This is the public breaking release:
remove superseded entry points where they obscure evidence identities or silently
collapse targets. Retain
original source generations and all physical relation files.

## Implementation sequence

1. Audit the 42 materialized registry entries against physical schemas and
   row counts. Record direct aliases, selector predicates, related fields and
   unavailable declarations. Add result-level MHC and external measurement
   entries. Every typed Parquet column must be reachable under a logical name
   or a namespaced source column so source detail remains queryable.
2. Build a schema-validated binding registry and `EvidenceStore`. Opening a
   package verifies the full manifest chain. Querying uses Arrow projection
   and predicates, preserves the declared grain and returns exact typed values.
   Permit joins only from a named relationship with verified keys/cardinality.
   Do not project many-valued claims onto one result implicitly.
3. Add a small versioned recipe language. Validate evidence type, endpoint,
   required fields, value/inequality/unit policy, context precision, alternative
   mappings and split identity before scanning. Emit one decision per candidate
   row, plus accepted, excluded, ambiguous, contributors and coverage files.
   The manifest binds recipe, source package and output hashes.
4. Expose catalog/query/recipe commands through one CLI group and document
   concrete MHC, T-cell, bulk MS, expression and context requests. Bump 2.0.0,
   remove misleading v1 top-level exports and the `hitlist export training`
   command, and document migration. Keep source export implementation available
   to historical source-audit code, but do not offer a competing public
   training command in the 2.0 CLI.
5. Verify synthetic adversarial fixtures and bounded real-package requests,
   reconcile all pinned baselines, run required gates, review and publish.

## Guardrails

The query registry is code plus physical package metadata, never a competing
data copy. Each selector is explicit (for example `evidence_type`, `grain`,
`task`, `table_number`). Each alias is one-to-one with a physical column at the
same grain. Any related value requires a named link and declares source and
target grain, key columns, and cardinality. Source-native columns without a
safe common semantic alias remain accessible under `native.<column>`.

Recipe exclusion is an observed decision, not deletion from the evidence
store. Each source row retains its own result identity across alternative
protein mappings, candidate MHC sets and contributors; a recipe either emits
separate alternatives or marks the row ambiguous. It may not multiply
independent labels. Ingest validity and task eligibility are separate checks.
Unknown context is never converted to explicit absence. A cohort study or
project-level instrument claim retains that scope in the result schema.

## User-facing contract

`EvidenceStore.open(package_directory)` verifies the top-level package manifest
and all predecessor/source hashes once, then exposes:

- `store.catalog()` — registered evidence types, source revisions, endpoint,
  grain, available fields, required ingest fields, relationships, rights and
  status (`materialized` versus `contract_only`).
- `store.query(evidence_type, select=..., where=..., joins=..., limit=...)` —
  Arrow batches with one declared grain. The same request grammar covers MHC,
  T-cell, bulk MS, expression and source/provenance relations. Unknown fields,
  unsupported predicates, unmaterialized types or unverified joins raise
  typed errors rather than returning empty/silent fallback data.
- `store.materialize(recipe, destination)` — a declarative, versioned task
  recipe with evidence types/endpoints, target interpretation, field and unit
  requirements, inequality handling, context precision, mapping policy and
  grouping/split policy. It produces accepted examples, excluded/ambiguous
  rows with reason codes, contributor/lineage relations, coverage and a
  hash-bound manifest. No package file or original source is rewritten.

Selection fields must be logical, not file-specific column names. Each adapter
maps logical fields to exact physical columns or source-backed relations with
an explicit level (result, assay, arm, profile, channel, study, project).
Preflight finds all 42 current `v2_materialized` catalog entries point to real
files, but 232 evidence-field declarations are not direct physical columns;
they live in related measurements/context tables or require a named alias.
The query mapping must account for every advertised field. If a source claim
exists only at project or study scope, expose that scope rather than
broadcasting it as a per-result fact. Declared fields without a supported
projection should be marked unavailable at that grain and raise on request.
Only documented joins can cross grains. Multiple mappings return alternatives
or require an explicit aggregation policy; they never duplicate labels by
default. The query result states the original grain, lineage precision and
source status. Null-state distinctions (`unreported`, `explicit_absence`,
`not_applicable`, `unresolved`, `conflicting`) survive projection.

The implementation must register a physical binding for each materialized
evidence type, including its exact selector and one-to-one logical aliases.
Register the existing MHC `typed/mhc_measurements.parquet` and external
`quantity_measurements.parquet` as named measurement evidence types; a result
row is never advertised as a numeric target merely because a related
measurement exists. PR 7 supplies the corresponding bulk protein and gene
expression measurement relations, each with a direct physical binding.
Catalog discovery distinguishes directly queryable fields, fields reachable
through a named relationship and contract-only fields. A declared field with
no physical binding must never be advertised as available. Validate bindings
against the frozen package's actual Parquet schemas at open time. Represent
source assets, source records/cells, assay context, protocol scopes, material
arms, factors and lineage as ordinary evidence types in the same request
surface. A named relation may expose a many-to-one lookup; one-to-many links
remain separate relation rows unless the recipe declares aggregation. This is
also the rule for provider overlaps, multiple peptide mappings and candidate
HLA sets.

Core identifiers and provenance are scoped: result, assay, material arm,
profile, channel, study and source record keys remain distinct. A source row
can support several typed records, while an assay can have several providers;
neither relation implies an extra assay or specimen. Ingest requirements are
per evidence type and target requirements are per task. The public catalog
reports both, plus the actual projection readiness of every logical field.

## Typed numeric predecessor (#751)

PR 7 materializes and verifies linked native bulk-protein and gene-expression
numeric representations. Query and recipe bindings must use those typed values
and their exact source-row lineage; peptide detections remain nonquantitative.

## Required examples

- MHC: select positive MS, binding or other assay evidence with exact
  endpoint/label semantics; filter reported peptide, tested molecule versus
  candidate set, and HLA assignment precision.
- Biological context: query sourced gene knockout and cytokine exposure
  claims on material arms, cell type, comparator and evidence basis; leave
  unknown specimens and absent conditions unfilled.
- Bulk and external MS: select enzyme/fraction/enrichment, instrument at
  supported protocol scope, measured pool, source, quantitative scale,
  derivation, exact versus surrogate material links and source raw cells.
- T-cell: request each of five availability families, direct functional
  response or multimer binding with antigen format, source outcome tier,
  numeric unit/inequality and restriction evidence; never use availability
  metadata alone as a peptide label.
- Expression: request measured gene/profile/scale, not source anchor rows as
  extra measurements; never assume a same-named line is the paired specimen.

## Verification

Reconcile all PR 1–6 baseline counts and source IDs on the final frozen
generation. Publish a table of native-to-logical field disposition and
readiness/eligibility funnels by source/endpoint. Add offline fixtures for
cross-family negative-label leakage, gene KO on a partner versus measured
cell, cytokine exposure versus release readout, mixed MS enzymes, project
instrument versus run instrument, mapping alternatives, and unknown specimen
relations. Exercise at least one recipe end to end on a bounded source
generation; verify source hashes, schema, keys, transformations, exclusions
and split-group uncertainty. Run format, lint, full tests and final-head CI;
merge, deploy clean main to PyPI and verify exact wheel/sdist hashes.
