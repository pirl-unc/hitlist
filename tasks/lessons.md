# Lessons

## 2026-04-23

- When parallel PRs are in flight, start follow-on work from refreshed `main` in a fresh worktree and explain the path/branch choice up front.
  Rule: if a user asks to continue after another PR is merging, do not keep stacking changes on an older feature worktree; branch from updated `main`, and if the working directory path differs from the repo root, explain that it is a separate git worktree before deep implementation starts.

- When adding a composed export on top of existing indexes, test the post-filter expansion path explicitly.
  Rule: if an export filters evidence rows first and then re-expands through a secondary index, add a regression test with a shared key (for example a shared peptide) to prove the secondary expansion still respects the original filter semantics.

- When introducing a stable row identity, test narrow projection mode as well as the default schema.
  Rule: if a new export documents a regrouping key like `evidence_row_id`, projected outputs must preserve it unless the API explicitly documents otherwise.

- Do not derive "allele-level" booleans from non-empty restriction strings when resolution metadata exists.
  Rule: prefer `allele_resolution` / equivalent schema fields over string-presence heuristics for any downstream flag that implies biological resolution.

- Tests for "index not built" paths should not depend on the user's global data directory state.
  Rule: when a test needs the unbuilt/empty-index branch, isolate `HITLIST_DATA_DIR` or monkeypatch the path helpers to a temp directory instead of conditionally skipping based on whatever exists in `~/.hitlist`.
