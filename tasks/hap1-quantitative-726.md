# HAP1 quantitative source admission (#726)

Base: e8faa7fd437d9f8b47d67a4c7836ad3df073f948, version 2.0.19. Presto's independently
registered source audit recovered PMC12090245/PXD056426 original supplements and
canonical search report. This implements a shared source package before an endpoint
recipe or new predictive fit. It does not replace the full availability objective.

## Source contract

- Pin the original article XML, supplement ZIP, canonical Search.zip, SDRF and
  complete PRIDE file inventory with SHA256, size and retrieval URL. Copy original
  bytes into the immutable package. Keep noncanonical search archive availability
  explicit: its supplement is available but its precursor provenance is not yet
  replayed. No synthetic fills, inferred limits or deletion of missing cells.
- Preserve each supplemental CSV row and every numeric/missing quantity cell.
  Canonical peptide intensity is normalized precursor sum / declared technical
  runs; noncanonical fields remain semantically deferred. Protein intensity is
  MaxLFQ; sampling density and coverage remain separate derived quantities.
- Project canonical DIA rows at precursor/run grain with original fields, raw and
  normalized values, peptidoform/charge, protein assignments and all Q/PEP fields.
  Retain unmapped runs. Preserve full protein-report rows and each raw/MaxLFQ
  sample cell. DDA search/log assets are retained and inventoried, not claimed to
  be a quantitative model-facing projection.
- Keep original source workbooks and SDRF rows as claims. Match raw filenames
  literally; unique punctuation aliases are candidate links only. Elution and
  proteomics sample IDs are modality-scoped. Equal biological replicate numbers
  across modalities do not establish the same aliquot or control specimen.
- Reuse the current HAP1 PMID override metadata through a frozen review contract,
  retaining source evidence and uncertainty. Context identifies human HAP1, MHC-I,
  supplied allele set, WT control and each single-gene KO. KO is an intervention
  assertion, not a measured residual activity or proof of zero pathway flux.
  Preserve instrument, acquisition, digest and measured-pool scopes separately.
- Parse every original FASTA entry including reverse/contaminant/standard records.
  Preserve all source protein assignments and enumerate every exact canonical
  peptide occurrence; reference candidates are not observed molecular origins.

## Shared API

Add standalone `hap1_quantitative` package schema 20 and catalog schema 25, exposed
through the existing EvidenceStore/query/link interfaces. Use streaming Parquet
relations: assets, source records, contexts, samples, runs, supplement quantities,
precursors, deposited protein quantities, references and peptide occurrences.
Common fields retain their existing semantic names; source-specific fields remain
queryable through `native.*`. Source package construction requires pinned assets;
tests may substitute an explicit tiny receipt at the private compile seam.

Use an atomic staging directory and a complete manifest. Verify hashes, schema,
file inventory and deterministic row content against a fresh streaming replay of
original sources. Avoid loading millions of expanded cells into Python lists.
Report counts by relation, source member, modality and value status. Missingness
disagreements remain separate source measurements; never overwrite either layer.

No model training eligibility is implied by a positive quantity. The subsequent
registered recipe must define canonical peptide identity, standards/background,
control aggregation, all-origin dependence components and a reserved evaluation
split. Source membership, elution quantity, protein quantity, inferred restriction,
processing and isolated MS response are different endpoints.

## Verification and delivery

1. Test raw/normalized distinctions, NA/zero/bound handling, row conservation,
   unmapped runs, ambiguous aliases, source/context links and standard/decoy origins.
2. Reject tampered sources, table content and manifest inventories. Exercise public
   EvidenceStore query/link operations on a tiny complete source fixture.
3. Run actual original assets through the writer and independently compare every
   supplemental quantity and sample/run attribution against the prior raw-format
   audit. Retain all discrepancies and source counts in the Presto registry.
4. Run format, lint and tests; version bump; publish a PR against current main.
   Merge/release remain separate tracked requirements, not inferred from tests.

Local execution only. No shared-environment modifications, cloud training, new
author messages or retuning against previously inspected biological tests.

## Original-source correction before retry

The first real build stopped before quantity projection: the community SDRF has
two `comment[sdrf template]` columns. A dictionary parser would lose claims.
Preserve this metadata table as ordered header/value arrays, including repeated
columns, while retaining strict unique-header checks for quantitative reports.
Extend the fixture with repeated SDRF headers before a new immutable attempt.
Attempt v1 source/exception receipts remain in Presto; no partial package survived.

## Nullable join defect exposed by the original package

The quantity replay passes, but querying an unmapped precursor with an optional
context join raises ArrowTypeError (`string vs null`). `_joined_values` passes an
all-null lookup set to Arrow. Add failing all-null and mixed-null tests, ignore null
keys for target lookup, preserve null rows and output type, and keep required-one
relationships strict about missing keys. This is a query fix; original source
package bytes and its completed arithmetic replay need not change.
