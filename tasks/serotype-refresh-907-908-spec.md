# Complete stored serotype refresh and dependency lock consistency

Issues: #907 and #908. Base: main at ccfe326 (2.1.10). Work in an isolated
feature worktree so the independent APC curation PR remains untouched.

## Contract

An old observation or binding parquet must expose the same derived serotype,
serotypes and serotype_source as a fresh annotation under the installed,
supported mhcgnomes version. Preserve reported mhc_restriction, row identity,
categorical dtypes and caller column projections. Reconcile before applying
serotype membership filters, including when the filter reads helper columns.
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

## Dependency and version work

Refresh uv.lock against mhcgnomes>=3.70.2 with minimal necessary resolution
changes. Create a separate environment from that lock, run pip check and the
editable-install audit, and use it for verification. Preserve the shared venv.
Use version 2.1.12 because open PRs already reserve 2.1.11; recheck before landing.

## Validation and handoff

First reproduce the new tests on the base, then verify the implementation.
Run ./format.sh, ./lint.sh and the guarded ./test.sh. Check uv lock --check,
package metadata, and the diff. Open a new PR linking both issues and describing
observed results and any remaining hosted checks. This request is to start a
new fixes PR; do not merge an unreviewed PR as part of that handoff.
