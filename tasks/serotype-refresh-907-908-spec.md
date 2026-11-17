# Complete stored serotype refresh and dependency lock consistency

Issues: #907, #908 and #911. Base: main at ccfe326 (2.1.10). Work in an isolated
feature worktree so the independent APC curation PR remains untouched.

## Contract

An old observation or binding parquet must expose the same derived serotype,
serotypes and serotype_source as a fresh annotation under the installed,
supported mhcgnomes version. Preserve reported mhc_restriction, row identity,
categorical dtypes and caller column projections. Reconcile before applying
serotype membership and serotype_source filters, including when the filter
reads helper columns. Move serotype_source filtering out of parquet pushdown
so stale provenance cannot discard rows before their annotations are refreshed.
Do not infer an annotation for blank/null source restrictions.

The current refresh predicate infers historical changes from the current
retirement and split tables. It cannot detect removed aliases or unrelated
catalog additions. Replace that inference with annotation of each distinct
nonempty restriction; retain the existing comparison and changed-value-only
assignments. Remove obsolete predicate helpers if they have no remaining
callers. This keeps parsing proportional to the restriction vocabulary, not
the number of observation rows.

## Regression coverage

- Removed B*15:112 alias: stale B75 membership must disappear; B15 remains.
- A*02:01 IEF addition: A2.1 becomes queryable while A2 remains.
- Both observation and binding loaders, projected and full reads, categorical
  and string inputs, singular-only projection, and filter-only helper columns.
- Reported serological restrictions and donor-set provenance remain distinct;
  unchanged current annotations, blanks and null restrictions remain intact.
- Retain the existing B*15:11 split/broad and retired-name tests.
- Empty serotype results retain full/projected schemas and dtypes, including
  after parquet pushdown, source filtering, and on an empty stored table.
  The membership mask must remain boolean for both string and category storage.

## Bounded follow-up audit

Review filter normalization, empty selections, helper projections, and the
serotype/provenance curation contract. Include reproducible serotype-loader
defects in this PR; file unrelated API contracts and curation defects separately
with evidence and acceptance criteria, searching existing issues first. Do not
change scientific annotations without authoritative evidence. Inspect Presto's
consumer path to give concrete upgrade and cache-regeneration instructions.

## Dependency and version work

Refresh uv.lock against mhcgnomes>=3.70.2 with minimal necessary resolution
changes. Create a separate environment from that lock, run pip check and the
editable-install audit, and use it for verification. Preserve the shared venv.
Use that committed resolution in the existing Python 3.12 CI matrix leg; keep
freshly resolved dependencies in the other legs. This rejects stale locks and
tests the locked environment even when local memory prevents a suite run.
Use version 2.1.12 because open PRs already reserve 2.1.11; recheck before landing.

## Validation and handoff

First reproduce the new tests on the base, then verify the implementation.
Run ./format.sh, ./lint.sh and the guarded ./test.sh. Check uv lock --check,
package metadata, and the diff. Open a new PR linking both issues and describing
observed results and any remaining hosted checks. The user subsequently
authorized final review, merge and deployment: review the complete final diff,
require the exact-head hosted gates, merge, dispatch a clean-main release build,
publish its verified artifacts and verify both public PyPI downloads. Recheck
main/version before merging. Report included fixes and separate follow-up issues,
then give Presto upgrade and regression-check instructions.
