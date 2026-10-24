# Binding assay context availability (#786)

Parent #785 (upstream #721). The predecessor is the verified binding-review
package available at schema 10/catalog 15. This change is additive: it keeps
all predecessor files and adds one audit row per retained binding observation.

## Contract

- schema 14/catalog 19, with a dynamically pinned schema-10 predecessor;
- 25,896 binding rows, including exactly 8,084 literal IC50 rows and 3,232
  separate `KD (~IC50)` rows;
- raw endpoint, method, units, qualifiers, source IDs and comments remain
  untouched; one audit row per `provenance_id` prevents join multiplication;
- availability statuses distinguish row-reported text, source-scoped study
  claims, unreported values, unresolved alternatives/ranges, and unreviewed
  studies;
- no study-level protocol claim is copied into an assay-level numeric value,
  no KD conversion is performed, and no training admission changes.

## Verification

The package verifier reconstructs the audit from the predecessor's corrected
cohort, checks the predecessor inventory and digest, validates endpoint counts,
checks unique observation identity, and compares the full materialized table.
Catalog links expose the audit to `mhc_binding` through `observation_id`.
