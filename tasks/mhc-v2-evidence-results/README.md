# MHC v2 evidence package reconciliation (#735)

The source is the [verified corrected v4 corpus](../corrected-corpus-results/README.md),
whose public manifest SHA-256 is
`2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7`.
The package was built with Hitlist 1.82.19 from those exact portable files. The
[package manifest](manifest.json) is a machine-readable inventory of all 12
content-hashed files, 71 native field dispositions, 23 source snapshot claims,
relation keys, licenses/access status and reconciliation counts. Its SHA-256 is
`efa5ea38cbed460a0a860d5ea2780314ead808a6b3eba064cff18228e2596674`.
The 1.82.19 package is a reproducible local result; the full Parquet directory
is not committed to Git.

| Native MHC result partition | v4 rows | Typed results | Change |
| --- | ---: | ---: | ---: |
| Positive ligand MS | 4,486,991 | 4,486,991 | 0 |
| Binding, affinity and stability | 814,268 | 814,268 | 0 |
| Other assays | 26,537 | 26,537 | 0 |
| Total | 5,327,796 | 5,327,796 | 0 |

The build emitted 5,327,796 typed measurements. All 9,381,101 original
contributor links remain in the package, with zero orphan links and zero
results without a link. An independent read of the pinned v4 Parquets compared
all 5,327,796 source rows in order to their typed result and measurement rows.
IDs, source/assay/reference IDs, peptide and modification fields, MHC text,
method/readout, modality/outcome, qualitative and quantitative values, units,
inequality, PTM and quality flags had zero mismatches. Source partition totals
and modality/outcome counts match the pinned audit; the complete counts are
in the manifest.

The `mhc_other` partition remains evidence. Its 23,537 negative MS rows are
not positive-presentation examples; 1,560 structural and 1,440 non-MS ligand
rows retain their own modality. `positive_result` is a source outcome within
a modality, so it never selects positive MS by itself. Binding endpoints retain
source readout text, numeric values, units and censoring without merging IC50,
KD, KD approximated from IC50/EC50, half-life and kinetic rates. The 535,026
curated supplementary MS rows retain blank reported method/readout fields and
an explicit curated admission basis. Reported MHC, experimentally tested MHC,
raw candidate sets and peptide-level restriction remain separate; this package
does not create a peptide-level restricting allele claim or donor/specimen ID.

To reproduce, install the pinned v4 bundle with `scripts/ci_corpus.py install`,
then call `hitlist.write_mhc_evidence_package(source_dir, destination)`. The
writer rejects unreviewed native columns, missing or changed source files,
duplicate result IDs, orphan links and results lacking contributors. It
publishes atomically only after package verification. Call
`hitlist.verify_mhc_evidence_package(destination)` to check the content hashes
and row inventory again. The package preserves original native files and raw
contributor cells for inspection; PRs 3–6 will normalize context and add the
cross-modality request and training API.
