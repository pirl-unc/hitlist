# Homogeneous native peptide measurements (#789)

## Specification

The Dasgupta 2014 and Gelman 2013 supplements are native cellular peptide
extract measurements, not MHC-elution or direct processing labels. The package
must preserve them at source grain while making study, material, treatment,
label/channel, acquisition, replicate, peptide and censored ratio semantics
queryable through the common EvidenceStore contract.

The first implementation uses the primary source workbooks (Table S1 for each
paper) for reported ratio cells and retains the heat-map workbooks (Table S2)
as source rows and explicitly deferred re-publications. A ratio cell is never
converted to an exact value when the source reports a bound, missing signal or
peak overlap. Article-level protocol assertions are stored separately from
row-level values; treatment names do not become genetic perturbations.

## Plan

- [ ] Add a source contract and parser for the four pinned workbooks, with
      immutable asset hashes, sheet dimensions, source row JSON and explicit
      row disposition.
- [ ] Materialize a generic `tmab_measurement` relation for every ratio slot,
      including missing and nonnumeric signals, with raw value, parsed value,
      inequality, censoring status and eligibility status kept separate.
- [ ] Materialize protocol context and source-row relations for cell line,
      organism, extraction scope, treatment, dose, exposure, TMAB labeling,
      instrument/acquisition and replicate/channel semantics.
- [ ] Add cross-paper lineage groups that expose possible reused measurements
      without treating matching labels as proven independent specimens.
- [ ] Register the package family, catalog fields, bindings, source links and
      EvidenceStore verification/query paths. Keep package construction
      caller-supplied and snapshot source files into the published package.
- [ ] Add focused tests for bounds, missingness, overlap, source preservation,
      lineage ambiguity, query fields and package tamper detection.
- [ ] Run `./format.sh`, `./lint.sh`, `./test.sh`, hosted checks, merge and
      publish 2.0.15; verify exact artifact hashes on PyPI.

## Review

Pending implementation and hosted release evidence.
