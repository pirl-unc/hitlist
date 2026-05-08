# Per-Donor Row Split for Attributed Peptides (issue #236)

## Goal

When the per-peptide attribution CSV maps a peptide to N matched donors, the scanner emits **one observation row per donor** (with that donor's typed alleles) instead of one row carrying the union of donor typings.

## Scope guarantee — other cohorts unaffected

This change ONLY fires when ALL three are true for a row:
1. `mhc_restriction` is class-only (e.g. `"HLA class I"`)
2. The PMID has a `peptide_attributions:` CSV registered in `pmid_overrides.yaml`
3. The peptide appears in that CSV

Today only PMID 31844290 (Sarkizova 2020) satisfies (2). Confirmed via:
```
$ grep -n "peptide_attributions:" hitlist/data/pmid_overrides.yaml
1135:    peptide_attributions: peptide_attributions/sarkizova_2020_patient_cohort.csv
```
Every other class-only row falls through unchanged to `host_mhc_types` (sample_allele_match) or `pmid_class_pool`.

## Design decisions

- **n_observations semantics: replicate.** Each per-donor row carries the same evidence as the original IEDB row. If 2 IEDB rows for SLLQHLIGL × 3 matched donors → 6 emitted rows. After aggregation each donor's typing shows n_obs=2. Preserves IEDB row count per donor; no information loss vs. today.
- **Donor identity: new column `attributed_sample_label`.** Empty for non-attributed rows. Carries the donor's `sample_label` (e.g. `"MEL3 (13240-006)"`) for attributed rows.
- **`mhc_restriction` promotion**: same logic as today (set_size > 0 + provenance != exact → promote), just operating on the per-donor set instead of the union set.
- **Provenance**: all per-donor rows continue to use `mhc_allele_provenance="peptide_attribution"`.

## Work plan

- [ ] Add `attribute_peptide_to_per_sample_typings(pmid, peptide) -> dict[str, frozenset[str]]` to `curation.py` (returns `{sample_label: alleles}`). Backed by a new lru_cached `_pmid_peptide_per_sample_typings` map. Keep the existing union-returning helper for backwards compatibility.
- [ ] In `scanner.py` per-IEDB-row class-only branch (around line 501-528): when the new helper returns N≥1 matched donors, loop over donors and emit one record per donor with that donor's typing. Non-attributed rows retain the current single-record path.
- [ ] Add `attributed_sample_label` to all scanner records (empty for non-attributed).
- [ ] Tests:
  - Unit: new curation helper returns per-sample mapping for SLLQHLIGL → 3 donors.
  - Scanner integration: attributed multi-donor row → N records, each with donor typing.
  - Regression: class-only row in non-attributed PMID → exactly 1 record, no `attributed_sample_label`.
- [ ] Run `./format.sh && ./lint.sh && ./test.sh`.
- [ ] Bump version to 1.30.43, commit, push, open PR referencing #236.

## Review section (to fill after merge)

_TBD_
