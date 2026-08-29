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

## 2026-08-28

- "Tests pass" means CI passes, not that they passed on my machine.
  Rule: in #378 I reported "1025 tests pass" while CI was red on all four Python legs. Two new tests called `generate_observations_table()` directly, which needs the built `observations.parquet` — present locally, absent in CI. The repo already documents the fix in `tests/conftest.py`: an `is_built()` skip plus an explicit `@pytest.mark.integration`. Note the marker alone is insufficient — one CI job runs the whole suite without the `-m` filter, so the in-test skip is what keeps it green. Before claiming a PR is ready, check `gh pr checks`, not just `./test.sh`.

- Read the data table before inferring a mechanism from output shape.
  Rule: I characterised `parse("RT1-B") -> RT1-Bb` twice from the output alone — first as "invents a haplotype letter", then as "narrows a locus to one chain". Both wrong; it is a curated entry in `mhcgnomes.data.gene_aliases["RT1"]["B"] == "Bb"`, one call away. Same pattern on the species tree: I revised the model three times (taxonomy -> prefix scope -> taxonomy-with-nomenclature-nodes) because each version came from one or two examples instead of enumerating all 641 nodes. When the library ships the table, read the table.

- Don't generalise "verified equivalent" from a subset to the population.
  Rule: I told the user a `required_result_types` swap was "verified equivalent — 344 curated values, 0 differences", then found 1 difference across the full 1,174-string corpus vocabulary (`RT1-B`). State the population the check covered, and check the widest one available before saying "equivalent".

- Verify claims about our own code before asserting them in another repo's issue tracker.
  Rule: I commented on pirl-unc/mhcgnomes#102 that "that is what we switched to" about a change we had not made. Filing upstream is outward-facing; a maintainer acting on it is acting on our word. Read the call site, then write the comment.

- Prefer the dependency's own ontology/API over string-shape heuristics.
  Rule: comparing species by genus string was both too weak (accepted `Macaca mulatta` vs `Macaca fascicularis`) and too strong (rejected clade-level nodes like `Galliformes sp.`). `Species.is_ancestor_of` answers it directly. Corollary from the same fix: every species descends from `Gnathostomata sp.`, so "shares an ancestor" is trivially true and fails open — only a direct ancestor/descendant relation is meaningful.

- When an allow-list is the honest answer, add a staleness assertion with it.
  Rule: the reviewer asked for `assert not (_KNOWN_MISMATCHES - flagged)` alongside the allow-list. It immediately earned its keep: mhcgnomes 3.39.0 fixed Patr-AL to `Ib`, and the staleness check is what surfaced that the entry was now obsolete. An allow-list without one is a permanent blanket exemption for that key.
