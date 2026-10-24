# Gomez-Zepeda prediction panels (#782)

Parent #778, after merged #779/#780. The pinned schema-12 release is Hitlist
2.0.12 at clean-main commit `d29e17f4faa248d6f59d2099d70d3e070c112845`.

## Scope

Preserve the publisher's MOESM8/S5 and MOESM10/S7 NetMHCpan prediction rows
alongside the source-result and arm-projection relations. The panels are model
outputs, not measured MHC presentation or experimental binding assays.

The prediction archive rows retain the source row, tested allele, EL/affinity
scores and ranks, Binder tier, cell-line scope, source phase, and exact linked
derived observation. Alternative alleles are retained. A separate projection
records the current derived CSV's selected allele, old `restriction_evidence`,
and projected `predicted` label; rows without a named selected allele remain
`unknown` with an explicit no-prediction reason.

## Frozen source facts

- MOESM8/S5 `S5_mhcpred_all_06154.csv`: 252,043 rows; publisher archive SHA-256
  `4e483930dc81c900fe5dcec3d0d89b56120060fd9e8e850713c04eefd2adedc`.
- MOESM10/S7 `S7_mhcpred_all_06133b.csv`: 142,909 rows; publisher archive
  SHA-256 `9f5d521156bd618052a47861202e6e1c3ba1ee44bfea460a22c65210ec3f489d`.
- The combined panels contain 394,952 original prediction rows and cover JY,
  Raji, HeLa, SK-MEL-37 and Plasma. S7 is the preferred panel for JY, matching
  the existing curation script; S5 supplies the other materials.
- Existing named derived restrictions: 50,761; blank restrictions: 17,791.
  The projection must preserve 68,552 derived observations and set exactly
  50,761 named rows to `predicted`, with 17,791 remaining `unknown`.

## Implementation contract

1. Add schema-13/catalog-18 source prediction and projection relations over a
   schema-12 predecessor. Do not rewrite native assay rows or delete source
   files. Preserve the predecessor manifest dynamically but require its schema,
   revision and Hitlist version.
2. Replay both CSVs byte-faithfully, normalize only the allele spelling needed
   for joins, and validate selected alleles against the source Binder/EL_Rank
   competition and existing derived CSVs.
3. Expose the projection through `EvidenceStore`, with explicit predicted vs
   unknown status and links to the source prediction rows and MHC observations.
4. Verify source-row counts, selected/alternative links, before/after labels,
   predecessor inventory and no assay loss. Run format, lint, focused/full
   tests, strict docs, hosted CI, clean-main release and PyPI hash checks.
