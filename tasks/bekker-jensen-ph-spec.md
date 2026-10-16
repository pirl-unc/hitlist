# Source-backed Bekker-Jensen fractionation metadata — #688

## Evidence and scope

The packaged original-analysis CSVs and API documentation assign pH 10 to nearly
all Bekker-Jensen arms. The pinned primary paper (PMID 28601559, DOI
10.1016/j.cels.2017.05.009; XML SHA-256
`b29852c7016d078ce23c9ba0e3907456c11ba9944131c7f83d8d642bec03f3f6`)
reports buffer C at pH 8 in section sec4.4.4, with 39/46/70 fraction collections.
This denotes a reported buffer, not a measured gradient or eluate pH. Table S1
identifies the 14-fraction runs as reused Kelstrup 2014 data. The phosphopeptide
12-fraction experiment names explicitly distinguish pH8/pH10; retain their
label-derived basis rather than claiming the main methods establish both.
Resolve the 50-fraction scope only if an exact primary-source mapping is found;
otherwise remove its unsupported default and record the limitation.

The existing PXD013455 46-fraction reanalysis is already corrected and immutable.
The active Presto fit uses a separately frozen human ligand/binding export and
must remain unchanged. No new biological labels, protocol-comparability claim,
replicate independence or original searched-FASTA identity follows from this fix.

## Design

1. Audit every curated legacy scope against the pinned methods and supplements.
   Store evidence, exact section/sheet locations, original value, reviewed value
   and basis in YAML; no biological rules inferred from parser defaults.
2. Apply the same reviewed metadata at CSV fallback, old-parquet reads, current
   bulk builds and packaged detectability admission. Preserve the original
   legacy annotation separately. Keep unreviewed scope matches unknown, never
   assign a new default. Numeric pH filters exclude unknowns; unfiltered queries
   retain them. Preserve all peptide/protein quantities, identities and rows.
3. Make the original ingest consult the explicit experiment registry for pH.
   Preserve distinct pH-labelled arms and avoid null-key grouping loss. Do not
   download/reingest the original multi-gigabyte deposit for this metadata fix.
4. Update public documentation and source-level defaults, with explicit unknown
   and experiment-label cases. Bump Hitlist to 1.80.0 in its own PR.

## Evidence and verification

- Regression cases for ordinary, reused, named-pH, unknown, mixed-source and
  idempotent metadata application; both CSV/parquet routes and numeric filtering.
- Verify packaged detectability selection against legacy source axes before
  projecting reviewed metadata; no scope becomes qualified from this correction.
- Audit both complete legacy CSVs with datacache, publishing legible per-scope
  before/after counts, citable evidence, hashes and an all-other-fields/row-order
  preservation receipt. Keep full raw files shared and unchanged.
- Run format.sh, lint.sh, test.sh and integration/release checks; review, merge,
  deploy from clean main and independently verify public distributions.

Remaining original source recovery stays in #654; unsupported additional pH
protocols stay explicit rather than silently filled.
