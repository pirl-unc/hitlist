# Lessons

## 2026-04-23

- When adding a composed export on top of existing indexes, test the post-filter expansion path explicitly.
  Rule: if an export filters evidence rows first and then re-expands through a secondary index, add a regression test with a shared key (for example a shared peptide) to prove the secondary expansion still respects the original filter semantics.

- When introducing a stable row identity, test narrow projection mode as well as the default schema.
  Rule: if a new export documents a regrouping key like `evidence_row_id`, projected outputs must preserve it unless the API explicitly documents otherwise.

- Do not derive "allele-level" booleans from non-empty restriction strings when resolution metadata exists.
  Rule: prefer `allele_resolution` / equivalent schema fields over string-presence heuristics for any downstream flag that implies biological resolution.

- Tests for "index not built" paths should not depend on the user's global data directory state.
  Rule: when a test needs the unbuilt/empty-index branch, isolate `HITLIST_DATA_DIR` or monkeypatch the path helpers to a temp directory instead of conditionally skipping based on whatever exists in `~/.hitlist`.

- When a review points out non-elution validation rows leaking into an MS export, fix the assay classifier at the source instead of paper-specific sample metadata.
  Rule: if IEDB mixes competitive-binding validation rows into an otherwise elution-focused PMID, update `is_binding_assay()` and add an exact assay-comment regression so the rows move to `binding.parquet` for every downstream export.

- When a loader promises a packaged-data fallback, test the "corrupt built artifact" path explicitly.
  Rule: if a public API prefers a built parquet/index but documents a source-data fallback, add a regression with an unreadable fake artifact and assert the loader warns and still returns correct filtered rows.

## 2026-05-12

- Don't copy defensive try/except fallbacks from existing code without justifying that the failure mode is actually reachable.
  Rule: in #254 I copied a `try: EnsemblRelease(release, species=species) except TypeError: EnsemblRelease(release)` pattern from `proteome.py:from_ensembl` into a new helper. The fallback handles a pyensembl version from before 2017 — predates the project's `python>=3.9` floor and isn't reachable in any supported install. AGENTS.md explicitly bans this: "Don't add error handling, fallbacks, or validation for scenarios that can't happen." When tempted to copy a pattern, check whether the original is also dead before propagating it. The reviewer (and the user) shouldn't have to point this out twice.

- Don't paper over review-identified cruft by tagging it "minor, won't file" — confront it.
  Rule: in the v4 self-review I called out an uncovered TypeError-fallback branch and concluded "skip, version too old for it to matter." The right move was to delete the unreachable branch, not document the gap. If a branch can't be exercised by any in-support configuration, it's dead code; the test gap is a symptom, not the bug.

## 2026-06-08

- For cell-line IDENTITY in curation, trust IEDB's own `assay_comments` / the deposited PRIDE metadata over web-search summary snippets.
  Rule: in #36 batch 10 I labeled PMID 27503676 as the "JY" cell line (A*02:01/B*07:02/C*07:02, CVCL_0108) based on a WebSearch summary. IEDB's assay_comment for that PMID explicitly recorded a *different* full typing — "eluted from the HLA-A*01:01, -A*03:01, -B*07:02, -B*27:05, -C*02:02, and -C*07:02" line — which is GR (CVCL_C5VZ), not JY. The classification (ebv_lcl) was still right, but the line name, HLA typing, and Cellosaurus accession were all wrong. When curating a single-line PMID, read the per-row `assay_comments` and `mhc_restriction` FIRST; if a search snippet names a line whose HLA type contradicts IEDB's recorded type, the snippet is wrong. A fast cross-check: declared ms_sample alleles should be a superset-or-overlap of IEDB's recorded alleles for that PMID, never disjoint.

- When a single PMID has multiple `assay_comments` source descriptions, curate ALL of them — don't stop at the first/largest arm.
  Rule: #36 batch 9 PMID 28871256 has 875 rows: 697 from BLS DR transfectants AND 175 from the MGAR wild-type DR15 LCL. The original entry documented only the 697 BLS rows and cited "697 rows" as the total. Always `value_counts()` the `assay_comments` for a PMID before writing ms_samples, and reconcile the row-count claim against `len(df[df.pmid==...])`.
