# Mono-allelic curation audit (2026-04-14)

## Rule change — `is_monoallelic` is now a per-row/sample claim

`classify_ms_row()` in `hitlist/curation.py` previously treated the
PMID-level `mono_allelic_host` override as a blanket flag: every row
in an overridden PMID got `is_monoallelic=True`.  That is wrong in
two ways, and both occur in real curated data.

The fix: the PMID override applies per-row only when **the row has a
resolved allele** — `classify_allele_resolution` returns `four_digit`,
`two_digit`, or `serological`.  Class-only / unresolved rows cannot
claim mono-allelic status because we do not know which allele (if
any) produced the peptide.

No cell_name check.  IEDB frequently mis-labels the host cell line
(Trolle 2016's 721.221 transfectants appear as
`"HeLa cells-Epithelial cell"`; Sarkizova 2020's transfectants
appear as `"B cell"`).  Gating on cell_name would silently block
these legitimate overrides.  The allele-resolution gate alone
handles mixed-cohort papers correctly because IEDB records their
validation rows with `mhc_restriction == "HLA class I"`
(class-only) — verified below against the actual built index.

`detect_monoallelic()` on `cell_name` alone is unchanged; it still
catches the primary path where IEDB records the host cell name
explicitly (e.g. `"C1R cells-B cell"` → C1R match).

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
| **31844290** | **Sarkizova 2020 — 95 alleles + validation** | **721.221** | **multi-allelic mix¹** | **fix below** |
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
**12 multi-allelic patient-derived validation samples** (patient-
derived primary tumors, all 12 enumerated in
`pmid_overrides.yaml::ms_samples`).

Verified against the built parquet (2026-04-14 snapshot), the IEDB
`cell_name` / `mhc_restriction` breakdown is:

| cell_name | rows | mhc_restriction | new `is_monoallelic` |
|---|---:|---|---|
| `"B cell"` | 219,893 | 95 distinct specific alleles | True (transfectants) |
| `""` (empty) | 16 | 2 specific alleles | True (transfectants) |
| `"Other"` | 16,642 | `"HLA class I"` only | False (class-only gate) |
| `"Glial cell"` | 12,730 | `"HLA class I"` only | False (class-only gate) |
| `"PBMC"` | 6,981 | `"HLA class I"` only | False (class-only gate) |

The key observation: IEDB records every validation row with
`mhc_restriction == "HLA class I"` (class-only).  The allele-
resolution gate alone correctly filters them out — no cell-name
reasoning required.

### PMID 26783342 (Trolle 2016) — would have been broken by cell-name gating

IEDB records Trolle's 721.221 transfectants under
`cell_name = "HeLa cells-Epithelial cell"` (15,638 rows) — wrong,
per the YAML note.  The paper is genuinely mono-allelic (5
transfected class I alleles, all specific).

- Under a cell-name consistency rule, the override would have been
  silently blocked and these legitimate rows would lose their
  mono-allelic flag.
- Under the final rule (allele-resolution alone), all 15,638 rows
  correctly stay `is_monoallelic=True`.

This case drove the decision to drop the cell-name check.

¹ An earlier draft of this audit cited "HCC1937, A375, HCT116,
HEK293T, SK-MEL-5, T47D, HeLa" as the Sarkizova validation cell
lines.  That was wrong — those are established cell lines that
do not appear in the paper.  The actual validation set is
patient-derived primary tumors (CLL, MEL, OV, GBM, ccRCC) with
curated 6-locus HLA for 11 of 12:

| Group | Samples | HLA-typed? |
|-------|---------|------------|
| CLL (B-cell leukemia) | DFCI-5341, DFCI-5328, DFCI-5283 | ✓ all 3 (6-locus) |
| Melanoma | MEL1, MEL2, MEL3, MEL15 | ✓ all 4 |
| Ovarian | OV1 | ✓ |
| Glioblastoma | GBM7, GBM9, GBM11 | ✓ all 3 |
| ccRCC | Pat9 | **✗ `mhc: unknown`** — paper did not HLA-type |

## Supplementary overlap

The only supplementary-contributing PMID is 38480730
(Gomez-Zepeda 2024), which is **not** mono-allelic.  No
supplementary rows are affected by this change.

## Verification

New unit tests in `tests/test_curation.py`:

- `test_pmid_mono_override_skipped_for_unresolved_allele` — empty
  and `"HLA class I"` both resolve to `is_monoallelic=False` under
  a mono-allelic PMID.
- `test_pmid_mono_override_skipped_for_validation_class_only` —
  Sarkizova's Glial-cell validation rows (class-only IEDB
  annotation) do not claim mono-allelic status.
- `test_pmid_mono_override_applies_across_cell_name_variants` —
  override fires for `"B cell"` (Sarkizova) AND
  `"HeLa cells-Epithelial cell"` (Trolle, wrong IEDB label) AND
  `"Splenocyte"` — cell_name is not a gate.
- `test_load_pmid_overrides_rejects_unknown_mono_host` — typo
  detection at YAML-load time.
- `test_load_pmid_overrides_warns_on_legacy_keys` — deprecation
  warning for old `type:` / `label:` keys.
