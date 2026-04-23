# Lessons

## 2026-04-23

- When the user wants work landed incrementally, keep PRs sequential instead of piling up draft branches.
  Rule: merge or refresh one PR at a time and rebase the next branch on the newly merged `main` before opening more follow-on drafts.

- When adding a composed export on top of existing indexes, test the post-filter expansion path explicitly.
  Rule: if an export filters evidence rows first and then re-expands through a secondary index, add a regression test with a shared key (for example a shared peptide) to prove the secondary expansion still respects the original filter semantics.

- When introducing a stable row identity, test narrow projection mode as well as the default schema.
  Rule: if a new export documents a regrouping key like `evidence_row_id`, projected outputs must preserve it unless the API explicitly documents otherwise.

- Do not derive "allele-level" booleans from non-empty restriction strings when resolution metadata exists.
  Rule: prefer `allele_resolution` / equivalent schema fields over string-presence heuristics for any downstream flag that implies biological resolution.

- Tests for "index not built" paths should not depend on the user's global data directory state.
  Rule: when a test needs the unbuilt/empty-index branch, isolate `HITLIST_DATA_DIR` or monkeypatch the path helpers to a temp directory instead of conditionally skipping based on whatever exists in `~/.hitlist`.
