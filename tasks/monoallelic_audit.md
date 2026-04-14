# Mono-allelic curation audit (2026-04-14)

## Rule change — `is_monoallelic` is now a per-row/sample claim

`classify_ms_row()` in `hitlist/curation.py` previously treated the
PMID-level `mono_allelic_host` override as a blanket flag: every row
in an overridden PMID got `is_monoallelic=True`.  That is wrong in
two ways, and both occur in real curated data.

The fix tightens the PMID override so it only applies when:

1. **The row has a resolved allele** — `classify_allele_resolution` is
   `four_digit`, `two_digit`, or `serological`.  Empty / class-only /
   unresolved rows cannot claim mono-allelic status — we do not know
   which allele (if any) produced the peptide.
2. **The row's `cell_name` is consistent with the declared host** —
   either empty, an IEDB tissue-level placeholder (`"B cell"`,
   `"Other"`, `"unknown"`, etc.), or containing an alias of the host.
   A different specific cell line in the same paper (validation
   sample) is **not** overridden.

`detect_monoallelic()` on `cell_name` alone is unchanged; it still
catches the primary path where IEDB records the host cell name
explicitly.

## Papers currently marked mono-allelic (12)

| PMID | Paper | Host | Status | Action |
|------|-------|------|--------|--------|
| 25418920 | Schittenhelm 2015 — 8 HLA-B27 | C1R | clean | none |
| 25880248 | Giam 2015 — HLA-A\*01:01 | C1R | clean | none |
| 26783342 | Trolle 2016 — 721.221 length | 721.221 | clean | none |
| 27920218 | Alpizar 2017 — B phospho | C1R | clean | none |
| 28228285 | Abelin 2017 — 16 alleles | 721.221 | clean¹ | none |
| 28514659 | Hilton 2017 — HLA-B\*46:01 | 721.221 | clean | none |
| 28855257 | Mobbs 2017 — HLA-C\*06:02 | C1R | clean | none |
| 28904123 | Di Marco 2017 — HLA-C/E/G | C1R | clean | none |
| **30315122** | **Faridi 2018 — cis/trans-spliced** | **C1R** | **unresolved** | **fix below** |
| 31092671 | Guasp 2019 — ERAP1/2 | 721.221 | clean | none |
| **31844290** | **Sarkizova 2020 — 95 alleles + validation** | **721.221** | **multi-allelic mix** | **fix below** |
| 34561969 | Khan 2022 — HLA-A\*33:03 | 721.221 | clean | none |

¹ Abelin 2017 also contains multi-allelic validation samples
(HCC1937, HCT116, HeLa, fibroblasts + PBMCs) marked `mhc: unknown`
in YAML.  Under the new rule those rows resolve to
`is_monoallelic=False` automatically (rule 1 — unresolved allele).

### PMID 30315122 (Faridi 2018) — unresolved

`ms_samples[0].mhc` is the string `"unknown"`.  Under the old
code every row in this PMID was flagged mono-allelic even though
the curated metadata states the allele is unknown.  Under the new
rule these rows resolve to `is_monoallelic=False`.

If we can identify the actual host allele(s) for this paper (C1R
transfectants with a specific allele), the fix is to update the
`mhc` field on that sample in `pmid_overrides.yaml`.  Until then,
the new behavior is the correct one — we do not claim mono-allelic
without an allele.

### PMID 31844290 (Sarkizova 2020) — multi-allelic mix

The paper profiles **95 721.221 mono-allelic transfectants** plus
**12 multi-allelic patient-derived validation lines**:

- **Mono-allelic transfectants (95 samples)** — each expresses a
  single transfected class I allele.  `cell_name` contains
  "721.221" (various suffixes for the transfected allele).
  Behavior unchanged: these rows continue to flag
  `is_monoallelic=True`.

- **Validation cell lines (12 samples)** — HCC1937, A375, HCT116,
  HEK293T, SK-MEL-5, T47D, HeLa, etc.  Their native multi-allelic
  HLA haplotype is the restriction.  Under the old code these rows
  were incorrectly flagged mono-allelic by the PMID override.
  Under the new rule they resolve to `is_monoallelic=False`
  because their `cell_name` is a specific non-host line that
  is not in the ambiguous-placeholder set.

## Supplementary overlap

The only supplementary-contributing PMID is 38480730
(Gomez-Zepeda 2024), which is **not** mono-allelic.  No
supplementary rows are affected by this change.

## Verification

New unit tests in `tests/test_curation.py`:

- `test_pmid_mono_override_skipped_for_unresolved_allele` — empty
  and `HLA class I` both resolve to `is_monoallelic=False` under a
  mono-allelic PMID.
- `test_pmid_mono_override_skipped_for_different_cell_line` —
  `HCC1937` under PMID 31844290 is not mono-allelic; `721.221`
  under the same PMID still is.
- `test_pmid_mono_override_applies_for_ambiguous_cell_name` —
  empty / `B cell` / `Other` / `unknown` under a mono-allelic PMID
  still resolve to `is_monoallelic=True` (primary intended path).
