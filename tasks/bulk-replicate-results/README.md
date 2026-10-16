# Source-backed bulk experiment counts

Codex / GPT-6, 2026-10-16. Clean executed recipe
`832c95a10d98ab3e37535c7a4362014e4a6518df`, Hitlist 1.81.0 candidate.
[Download all evidence](https://github.com/pirl-unc/hitlist/releases/tag/bulk-experiment-counts-v1).

## Result

The complete 2,047,003-row original-analysis bulk peptide summary preserves every
row, row order, cell line, peptide/protein identity and non-count quantity. Existing
1.80.0 reviewed pH/reference annotations remain unchanged, with their original
values retained. These are peptide-arm rows, not distinct peptides or donors.

The HeLa 46-fraction pool contains three documented MaxQuant experiments:
`HeLa-46fracs-IT-E1`, `HeLa-46fracs-IT-E2`, and `Tryp-46fracs` (138 acquisition runs).
The old observed maximum produced a denominator of two. All 217,514 rows now show
the inventory total of three, but their per-peptide count and reproducibility
fraction are null because legacy aliases collapsed experiment identities. Their
original detected counts remain inspectable. The other 13 scopes retain their
validated counts. No observation is removed when the threshold is omitted.

| Minimum fraction | Legacy filter rows | Qualified filter rows | Rows excluded by correction |
| --- | ---: | ---: | ---: |
| Omitted | 2,047,003 | 2,047,003 | 0 |
| 0 | 2,047,003 | 1,829,489 | 217,514 |
| 0.5 | 2,030,253 | 1,812,739 | 217,514 |
| 1 | 1,643,350 | 1,533,246 | 110,104 |

These counts include every enrichment scope for inspection. The public API's
ordinary default additionally selects `enrichment="none"`; pass `enrichment=None`
to inspect all scopes. A threshold excludes unknown evidence even at zero.
`scope-counts.tsv` gives all 14 scope comparisons.

## Source and units

Primary source: [Bekker-Jensen et al., 2017, PMID 28601559](https://pubmed.ncbi.nlm.nih.gov/28601559/),
[DOI 10.1016/j.cels.2017.05.009](https://doi.org/10.1016/j.cels.2017.05.009),
[PRIDE PXD004452](https://www.ebi.ac.uk/pride/archive/projects/PXD004452).
The pinned `summary.txt` member was recovered directly from
[SearchResults.zip](https://ftp.pride.ebi.ac.uk/pride/data/archive/2017/06/PXD004452/SearchResults.zip)
using datacache HTTP range extraction. Its 587,345 bytes match SHA-256
`5eb5c41af2c7929be8a9354bf7c9fcb021ead7f47c07d0774a146b352e1a8860`.
The original archive is 21,699,898,516 bytes; only the selected compressed member
and bounded ZIP metadata were fetched, and the complete decoded member was verified.
Every reviewed experiment label, acquisition count and sorted raw-file ID hash
matches this primary summary. The nominal Tryp-39fracs scope contains 40 acquisition
runs, illustrating why fractions and acquisitions must remain separate.

- `n_experiments_possible` is the curated experiment-list cardinality;
  `n_experiments_detected` is a qualified per-peptide count where recoverable.
- `n_replicates_*` are documented compatibility names for those experiment counts.
- `reported_n_replicates_*` preserve original API/source annotations. The raw CSV
  has no denominator column, so its reported denominator remains empty.
- `n_acquisition_runs` is the deposited run inventory; all
  `n_biological_replicates` remain unknown. No donor or specimen independence follows.
- `replicate_count_status`, exact experiment IDs, source URL, summary hash and raw-file
  inventory hash retain the qualification basis. Invalid or missing counts stay null.
- The full audit table additionally includes explicitly named `legacy_inferred_*`
  diagnostics for comparison; those maxima are never the source of a new label.

The assay is bulk shotgun MS, not MHC ligand elution. An MHC allele set does not
apply. Human cell-line identities and enzyme/fractionation/enrichment context are
retained; this audit does not newly curate culture conditions, donor or specimen
identities. It does not turn protocols into a comparable depth ladder or establish
biological replication. Protein abundance values and original source files remain
unchanged. The previously published PXD013455 search-scoped dataset, partitions,
Presto fits and their failed/limited readiness conclusions are unchanged.

## Artifacts and reproduction

The 18-asset publication contains the complete annotated `peptides.csv.gz`, per-scope
counts and evidence, primary summary and recovery receipt, source registry, curation
YAML, frozen audit/projection/fractionation code, independent verifier and its receipt,
plan, publication code, README, manifest and SHA256SUMS. Large inputs and outputs
remain in shared datacache generations. Repository receipts record actual paths,
hashes and publication/verification results.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Original peptide CSV | 35,074,080 | 3ec65dd05e8ec58c3293e0fb5720570197a8b152cd14778130bae4224106e3d3 |
| Complete annotated peptide CSV | 44,147,940 | 760000129927a762c4de97f3e4c92f608a2dfe17e3d7201d0345adcee0703d4a |

From the pinned clean checkout, with dependencies installed:

```sh
python scripts/audit_bulk_replicates.py --receipt tasks/bulk-replicate-results/audit-receipt.json
python scripts/verify_bulk_replicates.py tasks/bulk-replicate-results/audit-receipt.json --output tasks/bulk-replicate-results/independent-verification.json
```

Actual execution used Python 3.12.6, pandas 2.3.3 and datacache 1.27.1, through isolated
`PYTHONPATH=/private/tmp/hitlist-replicate-provenance:/private/tmp/hitlist175-wheel-import`.
CPU only, no cloud/GPU/model/optimization. The builder streamed 25,000-row chunks in
two passes, taking 84.413 seconds with process peak RSS 403,210,240 bytes; acquisition
and imports are excluded. The independent standard-library row verifier uses no
count-projection helpers and verifies every original row/value, primary inventory,
qualified fraction, historical comparison and aggregate/per-scope result.

The local complete unit suite passed 3,074 tests; 60 focused cases also pass with
37 corpus-dependent cases explicitly excluded. Full CI/integration and exact-main
PyPI deployment remain release gates. This data-only audit introduces no train/test
membership or new predictive performance estimate.
