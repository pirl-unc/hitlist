# Hitlist 2.0 evidence system — architecture and PR 1 specification

Parent: https://github.com/pirl-unc/hitlist/issues/733
PR 1 issue: https://github.com/pirl-unc/hitlist/issues/734

## Objective and boundaries

One semantic catalog must describe all current measured evidence and linked
context before data migration. This PR changes metadata and documentation only;
it must not rescan, relabel, count, deduplicate or project assay rows. Existing
assay exports and data bytes remain authoritative until the relevant migration
PR. The public v2 API flip and major version happen in PR 6.

A unified interface does not imply one wide file. Each relation keeps a declared
row grain so peptide mappings, source contributors, materials, assay runs and
expression values cannot silently multiply an assay count.

## Required model

- **Source dataset** is a pinned release or acquisition, with a source locator.
  Duplicate deposits remain contributors to an observation.
- **Observation** is one source-backed assay result or measurement at a declared
  grain. It has a family, endpoint and reported value/outcome. An assay ID alone
  does not establish independent donors, specimens or replicates.
- **Biological context** records cell/material identity, perturbations, exposures,
  measured or candidate MHC roles and source precision.
- **Measurement context** records protocol and apparatus: capture, digest, enzyme,
  assay method/readout, instrument and acquisition.
- **Relation** links grains, e.g. peptide to all compatible protein occurrences.
- **Training eligibility** is task-specific and strictly narrower than indexing.
  Required fields are declared separately. Unknown or missing information may
  leave a row indexable while masking a task target.

Each field declaration has one semantic role (identity, biological input,
measurement target, measurement protocol, provenance, relation or annotation),
a logical value kind and applicability. Null states must distinguish unknown,
explicit absence, not applicable, unresolved and conflicting where these are
reported. The catalog never substitutes one for another.

## PR 1 implementation

1. Add an immutable, versioned Python registry with dataclasses for evidence
   families, source/derived artifacts, row grains, field definitions and task
   requirement declarations. No heavy data import or implicit download.
2. Register current MHC positive-MS, binding and other-assay partitions; T-cell
   availability, functional and multimer projections; bulk peptide/protein MS;
   line expression; peptide-protein mappings; contributor and lineage relations.
   Derived T-cell projections must point to the same assay-ID source population.
3. Represent the roles of tested MHC molecule, candidate set, peptide-level
   restriction and independent cellular genotype separately. Keep biological
   cytokine exposure distinct from measured cytokine release.
4. Expose deterministic Python introspection and JSON-serializable descriptions
   for list/describe and strict field selection. Unknown artifact, task or field
   must raise a named error. No claim of training readiness merely because
   fields exist in the registry.
5. Record audited row-count baselines as dataset-generation-specific expected
   counts (not dynamic package-wide promises); document exact count units and
   CEDAR overlap. A future migration reconciliation report will compare to
   them, including source-linked row differences.
6. Add focused tests of registration completeness, roles, task requirements,
   null semantics, strict selection and deterministic serialization. Include
   packaging coverage for any new data resource. Update documentation.

## Later PR boundaries

PR 2 normalizes MHC observations/measurements/contributors. PR 3 normalizes
materials, interventions, MHC roles, lineage and protocols. PR 4 registers
supporting measurements and mappings with explicit join precision. PR 5
integrates T-cell source assays and qualifying projections. PR 6 supplies the
typed query, training recipes, frozen bundles and 2.0.0 breaking interface.
Every migration compares source/grain counts and retains an exclusion/reason
ledger. No consumer compatibility layer is required in v2.

## Verification

PR 1 must pass focused tests, format, lint and full test.sh. Inspect the wheel
or sdist when adding a packaged registry file. Run a no-corpus import/catalog
probe and verify no download/build is triggered. Review data/label invariants
against the v4 and T-cell audit summaries. Bump patch version for PR 1; merge
and deploy under AGENTS.md before starting PR 2.

## Risks

A field's existence in a current export is not proof that every row has it or
that its meaning is stable across sources. Registry declarations distinguish
available columns from required values and do not assert coverage. The verified
baselines are pinned to named artifacts; local or future generations may differ.
