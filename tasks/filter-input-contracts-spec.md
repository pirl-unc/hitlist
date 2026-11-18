# Consistent peptide filter input contracts — #913

Build on #915's common empty-result finalization; candidate 2.1.14, recheck
before landing. Keep this work in its own branch/release.

## Public contract

- Every string-or-list membership filter in the peptide-index and mappings
  loaders accepts scalar comma-separated values or a list/tuple of values.
  Split commas in each element, trim surrounding whitespace, discard blank
  tokens, and preserve token meaning/case. A comma is a separator, not a
  biological equivalence rule. Reject non-string elements with an actionable
  TypeError naming the filter rather than stringifying missing/numeric values.
- None means no filter. Explicit empty/blank selections mean zero matches for
  peptide, gene, provenance, evidence, serotype and source/host species filters.
  Preserve the intentional existing ValueError for empty restriction and
  candidate-allele queries; validate those before reading rows. Never turn an
  empty selection into no filter, including when other filters have zero hits.
- Arrow membership values are typed strings even when empty. Gene selections
  use the same normalization in direct mappings and observation/binding APIs.
- Preserve explicit empty gene selection in high-level training exports too.
  Their gene/name/ID selectors are alternatives (union): an empty axis plus a
  nonempty other axis retains that other alternative; all explicitly supplied
  axes empty means zero rows, whereas all None means no gene filter.
- Preserve scalar-only source, species and class APIs and boolean predicates.
  Missing candidate-set columns must not silently disable requested membership
  filtering: raise an explanatory rebuild error, as serotype filtering does.
- CLI repeatable/free-text selectors use the same normalization. Vocabulary
  selectors retain validation and accept normalized comma/list equivalents;
  typos remain parser errors. Document the behavior and empty selection policy.

## Validation and review

Use realistic string/category indexes and all three public loaders. Compare
scalar, padded list, comma-containing list and repeated CLI forms; compare
projected and full reads; exercise empty values, invalid types, old schemas,
gene mappings, and None independently. Keep gene/provenance semantic labels
unchanged and assert exact returned rows, not merely a nonempty result.

Run format/lint and guarded test.sh; retain preflight refusals and rely on
hosted checks when local memory is insufficient. Refresh isolated editable
metadata, review the final diff and exact head, then merge and verify PyPI.
