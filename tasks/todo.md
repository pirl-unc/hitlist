# September 22 backlog campaign

## #538 specification — release artifacts from the existing CI runner

The requested correctness fixes are merged, but local memory repeatedly
prevents the mandatory corpus tests from starting. Use the existing hosted
runner for the full test/build phase and retain local PyPI authentication.
Add `deploy.sh --build-only`, preserving every existing lint/test/build/license
gate and ordinary deployment behavior. Reject unknown options before doing
work, and never label a build-only result as a completed deployment.

Add a release-build workflow with a manual main-branch entry point and a
PR validation path limited to its own files. Require all five corpus files;
missing corpus must fail, not silently skip integration coverage. Install
the six locally developed dependencies from their current development heads
and record their resolved revisions. Run format, require a clean tree, then
the full build-only release with at most two workers and unchanged memory budgets.

Record the source commit, package version, installed dependencies and exact
wheel/sdist hashes in the artifact bundle. Before local upload, require a
successful manual workflow run on the current clean main commit, verify the
bundle against that commit/version, run the existing license checker and
twine metadata check, then upload those exact files and verify public PyPI
hashes. No PyPI credential leaves this machine. Bump to 1.62.39; subsequent
unmerged queue versions must move up before they land.

- [x] Implement build-only behavior and meaningful release-script regressions.
- [x] Add workflow, artifact provenance and rejection checks.
- [ ] Run format/lint, script regressions, workflow validation and CI review.
- [ ] Merge, run the full release build from clean main on CI, publish locally,
      and verify both PyPI artifact hashes before marking the fixes shipped.

Review: format/lint and actionlint 1.7.12 pass. All 22 deployment/artifact
checks pass on Python 3.9 and 3.12; the earlier combined run also passed all
17 unchanged memory/test-runner regressions. Build-only failures propagate
from lint, tests and license verification. Artifact validation checks the
successful manual workflow, exact clean-main commit, package version and
file hashes, then compares the manifest with the actual retained GitHub
artifact. A forged local manifest cannot borrow a real successful run ID.
PR-run builds are deliberately ineligible for publication. Full CI release
validation and clean-main publication remain required.

Artifact review found that `pip install --upgrade` resolved all six Git heads
but retained five same-version PyPI installations. The first workflow therefore
does not prove development-source validation and cannot authorize publication.
Remove those six installations in the disposable runner before installing their
Git refs. Validate VCS provenance immediately after installation and again when
writing the manifest; add a regression for same-version packages without a Git
revision. Repeat the full release build on the corrected head before merging.
All 24 release/deployment regressions now pass on Python 3.9 and 3.12;
format, lint and actionlint pass. Final-head CI and actual main publication
remain pending. The release workflow now reports individual skip reasons.
The first runner had 14.5 GiB available in both phases. Allow the existing
memory/CPU guard to select up to two workers, matching successful corpus CI;
it still falls back to one when memory permits only one, and refuses when
the unchanged per-worker budget cannot fit.

## #511 specification — numeric cell-line identifiers

The group identifier tokenizer drops one-digit numeric suffixes, conflating
SK-MEL-2 with SK-MEL-5 and making both match SK-MEL-28. Preserve numeric
tokens of any length while retaining the existing minimum for alphabetic
words and exact token boundaries. Do not alter the fuzzy arm scorer. Verify
single-line and explicitly multiple-line evidence and existing LM-MEL groups.
Ship 1.62.25 before #457's corrected sample roster depends on these identities.

- [x] Reproduce three single-line failures and preserve ambiguity controls.
- [ ] Fix numeric tokens; run format, lint, tests and review CI.
- [ ] Merge, deploy from clean main, verify PyPI.

Review: three SK-MEL cases fail before the fix; two LM-MEL and two explicit
mixed-line controls already pass. Numeric tokens retain exact boundaries;
the change cannot turn a mixed-line match into a single winner. Format/lint
and the focused non-integration group suite pass. Full tests and CI remain
required before merge.

## #457 specification — source-verified study identities

Re-read primary papers, supplementary sample maps, and every deposited row
description for PMIDs 28834231, 26375851, and 32488085 before editing their
YAML blocks. Correct Ritz's JY arm to MAVER-1 with the deposited six-allele
genotype; remove its unsupported EBV reference. Correct Schellens's pathogen
to measles Edmonston B, retaining honest arm ambiguity where IEDB merges
conditions. Replace Stopfer's unsupported A375 arms with the four melanoma
lines actually profiled, separating vehicle, two palbociclib doses, and IFN-gamma;
retain MDA-MB-231 only for the technical/absolute-quantification experiments.
Do not infer missing HLA typing from another cell line or from prediction.
Measure attribution before/after on all affected observations and pin source
facts and ambiguity behavior in regressions. Change only the three study
blocks plus necessary attribution support if a reproducible defect is found
and filed. Bump to 1.62.26, review, run format/lint/test and CI, merge/deploy.

- [x] Verify papers, supplement sample maps, genotypes and deposited descriptions.
- [x] Correct the three YAML records and update resolution notes from evidence.
- [x] Compare all affected rows and add meaningful regression coverage.
- [ ] Run required checks, review and merge PR, deploy and verify PyPI.

Review: all three source-fact regressions fail against the previous curation;
326 focused checks pass with corrected data. Semantically only the three
study blocks changed. All 7,646 MAVER-1 rows now resolve, the 5,093 HEK293 rows
keep their correct identity, and all 8,024 Schellens rows retain infection-arm
ambiguity. With #511, all 15,821 Stopfer rows identify their cell line; the
3,544 MDA-MB-231 rows reach quantification validation, while all 12,277 melanoma
rows retain unknown treatment pending #512. Found and filed a separate
disjoint-genotype text-attribution defect as #514; it does not affect these
stored rows because their source tissue is populated.

Primary sources: PMC5846733 / doi:10.1002/pmic.201700177 (explicit SSO/SSP
genotypes); PMC4574158 / doi:10.1371/journal.pone.0136417 (Methods, Table 1,
S1 Table); PMC7265461 / doi:10.1038/s41467-020-16588-9 (Methods, Figures 2–6,
Supplementary Data 3 and 5 file maps). UniProt identifies UP000100252 as
measles Edmonston B. Stopfer's main paper also corrects the old instrument,
search engine and invented A375 copy-number claim. Its supplementary raw
headers sometimes say 1uM on 10uM sheets; dose identity was checked against
sheet titles, normalized columns, file maps and the main paper, not inferred
from those inconsistent headers. No quantitative values were ingested here.

## #409 specification — modern license metadata

Replace the deprecated license table and license classifier with SPDX
`Apache-2.0` and explicit `license-files = ["LICENSE"]`. Raise the setuptools
build floor to its PEP 639-supporting release (77.0.0). Retain the same license
text. Add a release-time artifact assertion after `python -m build` and before
upload: both wheel and sdist must declare the expression and contain the exact
repository LICENSE bytes. Validate a real build and demonstrate a corrupted
license is rejected. Release 1.62.24 after #490.

- [x] Update metadata and add the pre-upload artifact verification.
- [x] Build wheel and sdist, check metadata and license bytes, test rejection.
- [ ] Run format/lint/test and CI; review, merge, deploy, verify publication.

Review: a real isolated PEP 517 build produced a wheel and sdist with
`License-Expression: Apache-2.0`, `License-File: LICENSE`, and unchanged license
bytes. The release checker rejects a wheel whose license content is corrupted.
`twine check`, format, lint, and explicit script lint/format checks pass.

Reference: setuptools' pyproject configuration guide documents that SPDX
expressions and `project.license-files` were introduced in 77.0.0:
https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

## #490 specification — strict prediction candidate boundary

PR #492 fixed the common case but still scores a row's entire reported set
when its intersection with the requested alleles is empty. It also compares
canonical row alleles with raw query spellings, making that fallback reachable
for aliases. Use the same normalized, serotype-expanded query set used for
observation filtering as the scoring allow-list. Normalize each candidate
token before intersection. An empty intersection stays unscored and retains
the original evidence restriction; it must never widen to outside alleles.
Each named sample continues to be queried independently. Unfiltered scans
retain their existing candidate behavior; the MHCflurry genotype limit is
unchanged. Release 1.62.23 after #504.

- [x] Reproduce alias, serotype, and no-overlap escapes with regression tests.
- [x] Enforce normalized intersection including the empty-set case.
- [x] Verify mixed scored/unscored rows and independent named genotypes.
- [ ] Run format/lint/test, review diff and CI, merge, deploy, verify PyPI.

Review: six new regression cases failed against the previous implementation;
all 86 pMHC tests now pass. Coverage includes canonical and shorthand allele
inputs, serotype expansion, two independently queried sample genotypes, and
unscored rows on both sides of a scored row. Format and lint pass. The change
does not raise MHCflurry's genotype limit or pool alleles across sample names.

Backlog review: #489 closed after a real eight-pair/two-allele prediction with
development MHCflurry; #482 retired after Table 1 confirmed five benign
patients already correctly curated; #67 closed as implemented by #83/#84/
#90/#103, retaining #95/#96/#361. Observed MHCflurry's 21-key warning is already
tracked upstream in openvax/mhcflurry#425 and #372.

## #504 specification — required allele parser

Core curation imports `mhcgnomes.Species` at module scope, so declare the
existing `mhcgnomes>=3.54.0` requirement in base dependencies. Preserve the
`alleles` extra as an empty compatibility alias for existing install commands.
Keep the installed-version floor check, correcting its optional-extra wording.
Refresh the lockfile without upgrading unrelated packages. Validate wheel
metadata and a clean base install's curation import and CLI, plus required
format/lint/test and CI gates. Release 1.62.22 after #501 ships.

- [x] Update dependency declaration, diagnostic wording, and lockfile.
- [x] Verify built metadata and isolated base installation.
- [x] Run format, lint, test; review PR and CI.
- [ ] Merge, deploy from clean main, verify PyPI publication.

Review: built wheel declares unconditional `Requires-Dist: mhcgnomes>=3.54.0`
and still provides the `alleles` extra. A fresh environment installed only the
wheel (no extras), imported curation from site-packages, and ran `hitlist --help`.
The 10 floor-check tests, format, lint, and `uv lock --check` pass. Retained all
existing locked package versions and unrelated resolution entries. All 1,659
non-integration tests pass; lint and all four Python CI jobs pass. Reviewed
the dependency graph and confirmed that the base requirement matches the
existing unconditional import. Rebased onto merged #505 without code changes;
the new PR head must also finish CI before merge.

## Scope and release contract

Review every open issue against current code and source evidence. Each issue
requiring changes gets its own feature branch, patch-or-greater version bump,
reviewed PR, passing CI, merge, and deployment from clean main. Retire obsolete
issues only with evidence. File newly discovered dependency defects upstream
and link them from affected PRs. Preserve unrelated local files and branches.

## Ordered plan

- [x] Establish an isolated environment using current development revisions of
      all locally developed libraries in the dependency graph; record revisions.
- [x] #501: remove obsolete dependency guards before relying on coverage.
- [x] #504: declare the existing mandatory mhcgnomes runtime dependency.
- [ ] #490, #489: preserve queried biological genotypes in scoring, then correct
      the MHCflurry presentation API call without weakening its genotype limit.
- [ ] #456: audit retired allele aliases and preserve reported provenance.
- [ ] #409, #291, #289, #306: packaging, cache, errors, and schema foundations.
- [ ] #482, #457, #452, #314, #230: primary-source-backed curation corrections.
- [ ] #357, #358, #140, #56: expression coverage and sample associations.
- [ ] #67, #95, #96, #361: bulk proteomics and detectability training data.
- [ ] #63, #176: profile and improve the remaining build bottlenecks.
- [ ] #46, #40, #39, #37: experimental-system, source, and attribution models.
- [ ] #18, #24, #41, #42: acquisition, quantitative and custom-database evidence.
- [ ] #33, #35, #36, #7, #8, #13, #14: source-verified remaining study curation;
      reconcile umbrella issues with their children before claiming completion.

## #501 specification

Verification changed this plan: metadata calls `mhcgnomes` optional, but
`curation.py` imports `Species` unconditionally at module scope. Remove the
obsolete try/except, all 15 early returns, and four remaining conditional test
branches. Assertions must execute unconditionally; an unavailable required
dependency must fail collection loudly. Track the metadata mismatch in #504
instead of adding another unsupported optional path. CI installs the allele
extra on every test job. Bump 1.62.20 to
1.62.21, run format/lint/test, review the diff, create PR, check CI, merge, deploy,
and verify the published version before moving to the next issue.

## Review and release log

- Initial state: main at 62ad259, version 1.62.20, 38 open issues, no open PRs.
- The existing deploy script publishes the current version; despite the older
  AGENTS wording it does not accept a version or perform bump/commit/push. Keep
  version bumps on feature branches and use the actual release behavior.
- Dependency environment: isolated `.venv`, development snapshots verified
  against upstream HEAD: datacache ee20b5a, pyensembl 4392376, gtfparse cb3788e,
  serializable 19c38ce, mhcgnomes 8d8f30a, mhcflurry 8b72541. The latter is
  upstream master, not the unrelated feature branch in the sibling checkout.
- Confirmed openvax/sercol#4 still blocks a combined development install with
  serializable 1.1.0 and added the resolver reproduction upstream. sercol and
  mhctools are not in hitlist's dependency graph and are not needed for its
  subprocess-based NetMHCpan integration; no constraints were bypassed.

# Issue #478 — ArrowTypeError merging serotype dictionary columns

## Objective

`build_observations()` crashes rebuilding the MS observations index inside
the cross-source concat step, reported from tsarina with a full
reproduction: `pa.concat_tables(ms_tables, promote_options="default")`
raises `ArrowTypeError` because two per-source partitions' `serotype`
column ended up dictionary-encoded at different index widths (`int8` vs
`int16`). Blocks any fresh build; does not affect an already-built parquet.

## Root cause

`_compress_categoricals` runs once per source partition (IEDB, CEDAR)
before the Arrow conversion, so pyarrow picks each partition's dictionary
index width from *that partition's own* distinct-value count independently
(<=127 categories -> int8, more -> int16). `promote_options="default"`
promotes null/missing columns but does not reconcile two dictionaries of
the same value type at different index widths.

Not scoped to `serotype` alone: all 28 columns in
`_CATEGORICAL_BUILD_COLUMNS` go through the identical per-partition
compression before the same two `pa.concat_tables` calls (MS and binding),
so any of them could hit this the moment one source's cardinality happens
to straddle the boundary and another's doesn't.

## Design

- `promote_options="permissive"` (pyarrow's own documented next tier up)
  widens mismatched-but-compatible types -- including dictionary index
  width -- to a common denominator, and still rejects a genuine mismatch
  like `int64` vs `string`. Verified both properties directly against
  pyarrow before trusting it.
- New `_CONCAT_PROMOTE_OPTIONS` module constant (both call sites now read
  from it) documents why, and gives tests one source of truth instead of
  a literal string that could silently drift from what production uses.

## Plan

- [x] Reproduce the exact `ArrowTypeError` with minimal synthetic tables
      before touching any code.
- [x] Empirically verify `promote_options="permissive"` fixes the dictionary-
      width mismatch AND still rejects a genuine `int64` vs `string` type
      mismatch (i.e. it widens compatible types, not relaxes validation).
- [x] Confirm via `_CATEGORICAL_BUILD_COLUMNS` that the same per-partition
      compression risk applies to all 28 categorical columns, not only
      `serotype` -- the fix is at the concat call, so it covers all of them.
- [x] Fix both `pa.concat_tables` call sites (MS, binding).
- [x] Regression tests, verified against the unfixed value before trusting
      them: the exact failure pinned, the fix confirmed end to end through
      the real `_compress_categoricals`, and a boundary test proving the
      fix doesn't also start accepting truly incompatible types.
- [x] Full `test_builder.py` + build-smoke tests (exercise `build_observations`
      end to end) pass.
- [x] Full combined-suite run twice; PR, CI, merge, deploy.

## Review

Reproduced the exact `ArrowTypeError` with two lines of synthetic pyarrow
tables before touching any code, then again through the real
`_compress_categoricals` helper to confirm the mechanism, not just the
symptom. Checked `promote_options="permissive"`'s actual boundary before
relying on it: it correctly merges `int8`/`int16` dictionary indices of the
same value type, and correctly still rejects `int64` vs `large_string` --
confirmed both directly against pyarrow, and pinned the second as its own
test so a future pyarrow upgrade that changed this behavior would be caught
rather than silently trusted.

Verified the new test suite is not merely trivially passing: with the
constant's value reverted to `"default"`, the fix test fails with the
issue's own literal error message; restored, it passes.

No artifact-version bump: this fixes a build-time crash, not any stored
column's values or meaning -- an existing artifact built before this
change needs no invalidation.

Full combined-suite run (`pytest -n 5 tests`, the same invocation
`test.sh --all` and `deploy.sh` use) plus the build-smoke tests that
exercise `build_observations` end to end: 1649 passed, 1 skipped, twice in
a row -- 1646 from before plus the 3 new tests, zero regressions.

# Issue #470 — CEDAR column misparsing

## Objective

CEDAR's export carries one more column than IEDB's (`Epitope | Mutation`
inserted at index 27), and three columns' name candidates never had a working
match on either source, so all three silently fell back to IEDB's positional
index -- correct by luck on IEDB, one column off on CEDAR. Two measured,
silent consequences: a binding assay (T2 stabilization) entering the
eluted-ligand output because the misread `assay_comments` comes back empty,
and mono-allelic cell-line detection never firing on CEDAR rows because the
misread `cell_name` returns an IRI instead of a name.

## Root cause (deeper than the issue itself identified)

`_resolve_columns` combines the two header rows as `f"{cat} | {fld}"` before
matching. `_COLUMN_NAMES["antigen_processing_comments"]`,
`["assay_comments"]`, and `["cell_name"]`'s candidates never included the
`" | "` separator, so they never matched EITHER source's real header (real
category is non-empty: "Antigen Processing" / "Assay" / "Antigen Presenting
Cell") -- confirmed empirically against both registered real files before
writing any fix. They "worked" on IEDB only because `_FALLBACK_INDICES` was
calibrated against IEDB's own column count. `assay_iri`/`ref_iri` have the
identical bug for a different reason: CEDAR literally names its first two
columns "CEDAR IRI" instead of "IEDB IRI".

## Design

- Add the missing pipe-separated candidates for all four broken keys.
- After the fix, verified all 27 `_COLUMN_NAMES` keys resolve by name alone
  on every known real/simulated layout (real IEDB, real CEDAR, and CEDAR with
  the exact reported extra column) -- the positional fallback is now a pure,
  never-triggered safety net for a genuinely unrecognized header.
- `_resolve_columns` now raises `ValueError` (naming the unresolved keys,
  the column count, and nearby header text) instead of silently using
  `_FALLBACK_INDICES` -- issue's suggestion #2, made safe by the point above.

## Plan

- [x] Verify the bug against real registered IEDB + CEDAR files (the local
      CEDAR predates the extra column, so also simulated the exact reported
      113-column layout by inserting it programmatically).
- [x] Fix the four broken `_COLUMN_NAMES` entries.
- [x] Fail-closed guard in `_resolve_columns`.
- [x] Fix 21 pre-existing scanner tests broken by the guard (three shared
      CSV-writing helpers relied on silent fallback for columns they didn't
      bother naming) by adding the missing names, not weakening the fix.
- [x] New regression tests: name-resolution position-independence, the
      exact three broken keys on the exact reported layout, the fail-closed
      guard, and two full `scan()` end-to-end tests reproducing both
      consequences verbatim from the issue (T2-stabilization binding assay,
      HMy2.C1R mono-allelic detection) -- verified each new test genuinely
      fails against the unfixed code with the exact reported symptom before
      trusting it.
- [x] Format, lint, full combined-suite run twice; PR, CI, merge, deploy.

## Review

Every new test was checked against the unfixed code before being trusted, not
just written and assumed correct: `git stash` on `hitlist/scanner.py` alone,
re-run, confirm each fails with the exact reported symptom
(`is_binding_assay` False instead of True, `is_monoallelic` False instead of
True, index 88 instead of 89, no raise where one is now expected), then
restore.

The first version of the new tests used blank category headers (matching the
existing `_write_tiny_iedb_csv` convention) and all five passed against the
UNFIXED code -- a false-negative regression suite. The real bug specifically
needs a non-empty category (`"Antigen Processing"`, `"Assay"`, `"Antigen
Presenting Cell"`) to expose the missing `" | "` separator; blank categories
let the old single-string candidates match trivially via the field-only
fallback pass. Rebuilt the header from real category/field text measured
directly off the registered IEDB file, keyed by real column index, with
CEDAR's extra column insertable at its exact reported index (27) -- this is
what makes the "before" run fail with the issue's own literal symptom
language rather than a synthetic proxy for it.

Fixing the fail-closed guard's blast radius took more than the four
`_COLUMN_NAMES` entries: 21 of 33 existing scanner tests relied on three
shared CSV-writing helpers that left most columns unnamed, silently accepting
whatever `_FALLBACK_INDICES` guessed. Extended each helper to name every key
(placed past the columns each helper's rows actually populate, so no existing
`row[N]` assignment needed to change) rather than softening the guard to
tolerate unnamed columns -- the guard's whole point is that an unnamed column
must not resolve silently.

Full combined-suite run (`pytest -n 5 tests`, the same invocation
`test.sh --all` and `deploy.sh` use): 1646 passed, 1 skipped, twice in a row
-- 1641 from before plus the 5 new tests, zero regressions.

# Issue #467 — mhcgnomes version-floor tripwire

## Objective

An installed `mhcgnomes` below the floor `pyproject.toml` declares fails the
suite with 200+ scattered `AttributeError`s across 15+ files (e.g.
`Species.compatible_with` missing below 3.39.0), with nothing connecting any
one traceback back to the actual cause. Reported from a release install of
3.33.4 shadowing a correctly-locked 3.64.2 in a shared environment's
site-packages. Add a tripwire that turns 200+ cryptic failures into one
message naming the declared floor, the installed version, and the resolved
import path.

Deliberately scoped to the tripwire the issue actually asks for, not to
modifying the shared environment itself: that environment is used by other
concurrent sessions and several sibling repos, and a version bump there is
someone else's call, not a fix this repo's code can make unilaterally. The
per-worktree `uv run` workaround already documented on the issue remains the
correct way to get a conforming environment; this tripwire is what makes the
*next* person who hits a shadowing install able to diagnose it in one line
instead of reverse-engineering it from a curation.py traceback.

## Design

- `tests/mhcgnomes_floor_check.py`: pure, independently-testable functions
  (`declared_floor`, `floor_violation_message`, `check`) mirroring the
  existing `tests/xdist_cache.py` pattern -- parsing/comparison logic lives
  outside conftest.py so it has a public surface to unit-test.
- `tests/conftest.py`'s new `pytest_configure` hook calls `check()` once,
  before collection, and `pytest.exit()`s the whole session with the
  message if it fails -- no per-test overhead, no chance of 200 confusing
  failures burying the one line that explains them.
- Version comparison via `packaging.version.Version` (already a transitive
  dependency), not string comparison -- `"3.9.0" > "3.54.0"` lexically,
  backwards, so a naive string floor check would have the wrong sense for
  any single-digit-vs-two-digit minor version pair.

## Plan

- [x] `tests/mhcgnomes_floor_check.py`: parse the floor, compare, message.
- [x] Wire into `tests/conftest.py`'s `pytest_configure`.
- [x] Unit tests: parsing (real file + synthetic + missing-declaration),
      message content, numeric-not-lexical comparison, the real
      environment's happy path, and the `pytest_configure` wiring itself.
- [x] End-to-end proof, not just a mock: a fake `mhcgnomes==3.33.4` package
      shadowed onto `PYTHONPATH`, reproducing the exact reported scenario,
      confirmed the whole session exits on one message instead of running
      into 215 failures.
- [x] Format, lint, targeted + full combined-suite run; PR, CI, merge,
      deploy from clean main.

## Review

End-to-end proof, not just a mock: built a fake `mhcgnomes` package declaring
`__version__ = "3.33.4"` and shadowed it onto `PYTHONPATH` ahead of the real
3.64.2 install -- an exact reproduction of the reported scenario. The whole
session now exits immediately with:

```
Exit: installed mhcgnomes 3.33.4 is older than the floor this project
declares in pyproject.toml's `alleles` extra (3.54.0).
  Resolved from: .../fake_mhcgnomes_shadow/mhcgnomes/__init__.py
  ...
```

instead of collecting and running into 215 scattered `AttributeError`s.

10 new unit tests cover the parsing (real pyproject.toml + a synthetic
string + a missing-declaration failure mode), the message content, numeric
vs. lexical version comparison, the real environment's happy path, and the
`pytest_configure` wiring itself (via `pytest.exit.Exception`).

Full combined-suite run (`pytest -n 5 tests`, the same invocation
`test.sh --all` and therefore `deploy.sh` use): 1641 passed, 1 skipped --
1631 from before plus the 10 new tests, zero regressions.

Not attempted: upgrading the shared virtualenv itself. That environment is
shared across concurrent sessions and several sibling repos; a version bump
there is a decision for whoever owns those other workloads, not something
this repo's code can safely make unilaterally. The tripwire is the part of
#467 that is actually this repo's to fix.

# Consolidate the isolated-curation test fixtures (#473 follow-up review)

## Objective

`/code-review` on #473 (the fixture-cache-leak fix) found no correctness bugs but
confirmed four real cleanup findings in the fix itself: an unguarded `finally`
that could mask a real test failure, redundant `try/finally` ceremony around a
`yield`-fixture pytest already tears down unconditionally, the same ~13-line
docstring and clear/yield/clear block duplicated across two files, and two
pre-existing tests now carrying manual cache-clear calls the fixture itself
already guarantees. Fix all four together.

## Design

- New shared `_isolated_curation_root` fixture in `tests/conftest.py`: copies the
  packaged curation YAML into an isolated temp tree, monkeypatches
  `curation._data_path` / `cell_name_parser._registry_path`, and clears every
  curation cache on setup and teardown via a plain `yield` (no `try/finally` --
  pytest already runs a fixture's post-`yield` code unconditionally, so wrapping
  it adds a masking risk with no added guarantee).
- `test_builder.py`'s `isolated_curation` and `test_cache_current.py`'s
  `curation_referencing_uncached_asset` become thin fixtures depending on the
  shared base, each owning only the fake `pmid_overrides.yaml` content and
  monkeypatches specific to their own tests.
- Removed the now-redundant manual `cache_clear()` calls in
  `test_cache_tracks_new_attribution_reference` and
  `test_curation_change_rebuilds_stored_evidence`.

## Scope note

Five more tests in `test_curation.py`/`test_exclude_from_ms.py` hand-roll a
narrower, single-file `_data_path` monkeypatch with their own correctly-scoped
`try/finally` cache_clear -- already safe, unlike the two fixtures this PR
touches. Left alone: a DRY improvement there would touch several already-correct,
unrelated tests for a purely stylistic gain, disproportionate to the risk.

## Plan

- [x] Shared `_isolated_curation_root` fixture in `tests/conftest.py`.
- [x] Both fixtures refactored to thin wrappers; unused imports removed.
- [x] Redundant manual cache-clear calls removed from the two dependent tests.
- [x] Verify the exact confirmed regression pair from #474 still passes.
- [x] Format, lint, targeted tests, full combined-suite run twice; PR, CI,
      merge, deploy from clean main.

## Review

`ruff check --fix` removed one now-unused `Path` import in `test_cache_current.py`
as a direct consequence of the refactor (the shared fixture owns that copy logic
now); no other changes needed to satisfy lint.

Verified in order: both fixtures' own tests (78 total, unchanged pass count) still
pass; the exact two-test pair that reproduced #474's regression still passes;
`pytest -n 5 tests` (the combined invocation that originally exposed the leak)
passed 1631/1631, twice in a row.

# Fixture cache leak found while deploying 1.62.4

## Objective

The 1.62.4 deploy (merge of #472) failed its own `./test.sh --all` gate with 23
failures, all in `test_curation.py`/`test_exclude_from_ms.py`, none touching code
`#472` changed. GitHub Actions CI on the same commit was green on all four Python
legs. Root-caused and fixed before retrying the deploy.

## Root cause

`deploy.sh` runs `test.sh --all`, which mixes integration and non-integration
tests in one `pytest -n 5` invocation with no `-m` filter. CI's Python 3.11 job
keeps them in two separate `pytest` invocations (`-m "not integration"` then
`-m integration`), so this bug had no way to surface there.

`tests/test_cache_current.py`'s `curation_referencing_uncached_asset` fixture (added
for #448) monkeypatches `curation._data_path` to an isolated single-PMID YAML tree
for the duration of one test, and never clears any curation cache. That was safe
when written: nothing in `observations_cache_is_current()`'s call graph touched
`curation.load_pmid_overrides()`'s `lru_cache`. #471 changed that —
`supplement.load_supplementary_manifest()` now calls `ms_excluded_pmids()`, which
calls `load_pmid_overrides()` — and `_source_fingerprints()` (part of the
predicate's call graph) reads the supplementary manifest. So the fixture's test
now populates the REAL, process-global `load_pmid_overrides` cache with its fake
single-PMID data, and nothing clears it afterward. Under `-n 5` with the full
~1630-item collection, whichever worker draws this test early in its queue then
serves every subsequent real-data curation test on that worker from the fake
cache for the rest of the run — reproduced deterministically (though the exact
failing set varies run to run, since xdist's dynamic scheduling isn't) down to a
minimal two-test repro:
`test_cache_current.py::test_observations_predicate_never_downloads` followed by
`test_curation.py::test_pmid_mono_allelic_override`.

`test_builder.py`'s pre-existing `isolated_curation` fixture has the identical
shape (patches `_data_path`, no fixture-level teardown) and has stayed safe only
because its three current tests each remember to clean up manually. Same bug
class, same fix.

## Plan

- [x] Bisect: confirmed via a minimal two-test repro, not a full-suite guess.
- [x] Harden both `_data_path`-patching fixtures to `yield` + unconditional
      `curation._clear_curation_caches()` teardown, so the next function that
      gains an indirect `load_pmid_overrides()` dependency can't reopen this.
- [x] Verify the exact confirmed pair passes; verify both fixtures' existing
      tests still pass; re-run the full `test.sh --all`-equivalent invocation.
- [x] Format, lint; PR, CI, merge, retry the deploy from clean main.

## Review

Root cause confirmed with a minimal two-test repro before touching anything:
`tests/test_cache_current.py::test_observations_predicate_never_downloads` followed
by `tests/test_curation.py::test_pmid_mono_allelic_override`, run together with no
xdist, fails identically to the deploy log. The fix (both fixtures now `yield` +
unconditionally clear curation caches in `finally`) turns that pair green.

Verified against the actual failure mode twice: `pytest -n 5 tests` (deploy.sh's
exact invocation, no `-m` filter) passed 1631/1631 both times, where main at
`c4fa4cc` failed 23 (first run) and 7 (second run, verbose) -- the varying failure
set across runs is itself evidence this was xdist scheduling exposing a real
process-global cache leak, not a fixed collection-order bug.

Not a production bug: `curation._clear_curation_caches()` already lists
`ms_excluded_pmids` (added correctly in #444/#466). The gap was purely in two test
fixtures that monkeypatch `_data_path` without a teardown, which stayed invisible
until #471 gave `load_supplementary_manifest()` an indirect path to
`load_pmid_overrides()` that didn't exist when the #448 fixture was written.
CI never caught it because the workflow keeps integration and non-integration
tests in two separate `pytest` invocations; `deploy.sh`'s `test.sh --all` runs
them together in one `-n 5` pass, which is exactly the condition needed to expose
a same-worker cache leak.

# Issue #471 — post-merge review findings on #466 (exclude_from_ms)

## Objective

`/code-review` on merged PR #466 surfaced 11 findings: a validation gap that lets
`exclude_from_ms` silently no-op, two QC signals that never learned about the new
filter, a supplementary-data contradiction with no cross-check, a second code path
that bypassed the filter entirely, two cleanup items, two test-coverage gaps, and
two doc corrections. Fix all of them in one release.

## Plan

- [x] `curation.ms_excluded_pmids()`: reject a non-boolean `exclude_from_ms` value
      loudly instead of silently treating it as "not excluded".
- [x] `qc.cross_reference()` / `curation_plan()`: skip excluded PMIDs so they stop
      reading as `yaml_only` gaps and inflating priority.
- [x] `qc._default_curated_mhc_samples()`: skip excluded PMIDs in the token audit.
- [x] `supplement.load_supplementary_manifest()`: raise if a manifest entry's PMID
      is also curated `exclude_from_ms: true` — the two curations disagree about
      whether the study is an elution experiment, and silently dropping the
      hand-vetted rows a few build steps later hid that.
- [x] `report._run_report_from_csv()`: apply the same MS-scoped exclusion the built
      corpus gets, so `--from-csv` stops diverging from `hitlist report`.
- [x] `builder._drop_masked_rows()`: shared skeleton for `_drop_short_mhc2_rows` and
      `_drop_excluded_from_ms`, closing the missing `.head(5)` cap on the latter.
- [x] Structural schema-guard test: a field not self-declared unread must be
      referenced somewhere outside its own `PMID_ENTRY_FIELDS` declaration, with a
      synthetic negative case proving the check isn't vacuous.
- [x] Restore the `"#444" in description` assertion dropped from the replacement test.
- [x] `tasks/todo.md`: fix the stale `(1.62.0)` heading to `(1.62.3)`, and note
      format/lint passed in #444's own review section, matching the prior entry.
- [x] Format, lint, full non-integration and integration tests; PR, CI, merge,
      deploy from clean main.

## Review

All 11 findings fixed with direct regression coverage for every new branch, not just
incidental coverage from existing tests.

**Validation (#1).** `ms_excluded_pmids()` now rejects a non-boolean `exclude_from_ms`
with a `ValueError` naming the PMID, instead of `is True` silently treating `1` or
`"true"` as "not excluded".

**QC signals (#2, #3).** `cross_reference()` (and therefore `curation_plan()`, which
consumes it) and `_default_curated_mhc_samples()` (the token audit's default sample
source) both skip `ms_excluded_pmids()` now. Verified against the real corpus: nothing
raises, and a direct unit test proves an excluded study's curated arm no longer reads
as a `yaml_only` gap while an unrelated study's genuine gap still does.

**Supplementary contradiction (#4).** `load_supplementary_manifest()` raises if any
entry's PMID is also curated `exclude_from_ms: true` — the two curations disagree
about whether the study is an elution experiment, and the alternative (silently
dropping the hand-vetted rows after `scan_supplementary` already logged them as
added) is exactly the kind of silent contradiction #436 and #444 both warn about.
Verified clean against the real packaged manifest (21 entries, 11 exclusions, zero
overlap) before adding the guard.

**Report path parity (#5).** `--from-csv` now applies the same MS-scoped exclusion the
built corpus gets via a new `_drop_excluded_ms_rows`, careful to only drop the
MS-classified subset — `df` here is the raw mixed scan, unlike the builder's
already-split `obs`, so a binding row for an excluded PMID stays either way.

**Cleanup (#6, #7).** `_drop_short_mhc2_rows` and `_drop_excluded_from_ms` now share
one `_drop_masked_rows` skeleton, which closes the missing `.head(5)` cap on the
exclusion loop as a side effect of not duplicating it.

**Test coverage (#8, #9).** Added the structural half of the schema guard: any
`PMID_ENTRY_FIELDS` entry not self-declared unread/informational must have its exact
name referenced somewhere outside its own declaration, with a synthetic negative case
(`_fields_claiming_a_reader_without_one`) proving the check actually catches an unwired
claim rather than passing vacuously. Restored the dropped `"#444" in description`
assertion.

**Documentation (#10, #11).** `#444`'s review section now notes format/lint passed,
matching the prior `#462` entry's convention; its heading cites the version it actually
shipped as (1.62.3, not 1.62.0 — three other PRs landed and bumped the version between
when that section was drafted and when it merged).

Full non-integration suite: 1580 passed (0 new failures). Targeted suite across every
changed module plus all new tests: 407 passed, 1 skipped (a real-corpus integration
test that predates artifact_version 5 locally — pre-existing, unrelated to this PR).
Coverage confirmed on every new branch: the validation raise, the contradiction raise,
`_drop_excluded_ms_rows`'s three return paths, and both new QC skip checks are all
exercised by a test, not just reached incidentally.

# Issue #454 — one duplicate-key-rejecting loader for every curation YAML

## Objective

`UniqueKeyLoader` (#450) guarded `pmid_overrides.yaml` and `condition_vocabulary.yaml`;
the other nine packaged YAML files still loaded through plain `safe_load`, where a
duplicated key silently discards the first value. All eleven parse clean today, so this
is a guard-rail change with no data edits and a byte-identical corpus.

## Design

- New leaf module `hitlist/curation_yaml.py` holding `UniqueKeyLoader` and
  `load_curation_yaml(path_or_traversable)`. It imports only PyYAML, so the light
  registry/download modules use it without pulling in the curation stack, and
  `conditions` drops its lazy import that dodged the `curation` cycle.
- `hitlist.curation.UniqueKeyLoader` stays importable under its original name.
- Every `yaml.safe_load` / `yaml.load` call in the package (12 sites, 9 modules) routes
  through the helper. A test greps the package for any direct PyYAML parse outside the
  leaf module, and a parametrized test loads every packaged YAML through the guard.

## Plan

- [x] Leaf module + re-export; route all twelve call sites.
- [x] Tests: rejection, path/traversable inputs, packaged-file sweep, bypass guard.
- [x] Document the guarantee in `docs/curation-process.md`.
- [ ] Format, lint, tests; bump the version, PR, CI, merge, deploy from clean main.
# Issue #468 — the warm-up deadline stops paying for process startup

## Objective

`_supervise_prefetch_tasks` started its wall-clock budget before building a **spawn**
pool, so the cost of starting a fresh interpreter was charged to a deadline that exists
to bound *network* stalls (#402, #407). On a loaded machine that could consume the whole
phase without a single fetch attempted, and it made
`test_prefetch_supervisor_terminates_blocked_inflight_call` depend on runner load — it
failed on the Python 3.11 leg of #465, a PR that touches nothing in `mappings.py`.

## Plan

- [x] Start the clock once the pool exists, so the budget measures warm-up work.
- [x] Regression that fails on the old code with the exact CI message: a context double
      whose `Pool` takes longer than the deadline, asserting the work was still attempted.
- [x] Prove the in-flight timeout test is deterministic again.
- [ ] Format, lint, full tests; PR, CI, merge, deploy from clean main.

## Review

The new test reproduces the CI failure verbatim against the unfixed supervisor
(`assert 'timed out' in '    prefetch deadline of 0s exhausted; skipping 1 proteome(s).'`)
and passes with the clock moved. Behaviour on either path was already correct — every
proteome is reported unavailable — so no corpus or artifact effect.

# Issue #448 — public, quiet cache-validity predicates

## Objective

Give library consumers (tsarina, presto) a side-effect-free way to ask whether the
built `observations.parquet` / `binding.parquet` set and the `peptide_mappings.parquet`
sidecar are current, so they can announce a rebuild before spending ten minutes on it
instead of buffering `build_observations()`'s status block and inferring afterwards.
`observations.is_built()` / `mappings.is_mappings_built()` stay existence-only.

## Design

- `hitlist.observations.observations_cache_is_current() -> bool | None`
  - `None` when no IEDB/CEDAR source is registered: validity is unknowable, and
    `build_observations()` would raise rather than answer.
  - Otherwise the same verdict `build_observations(force=False)` reaches before
    deciding to skip: artifact version, source + curation fingerprints, parquet
    fingerprints. Nothing printed, nothing written.
  - Must not download. `_curation_fingerprints` resolves `peptide_attributions`
    assets through `packaged_or_fetched`, which fetches a missing externalized CSV
    on a wheel install. The predicate resolves through a new
    `downloads.packaged_or_cached` instead and reports the cache stale when an
    asset is absent — which is what a build would conclude too, since fetching
    stamps a fresh mtime.
- `hitlist.mappings.mappings_cache_is_current(*, release=112, fetch_missing=True,
  use_uniprot=False, flank=DEFAULT_FLANK) -> bool`
  - Same contract check as `build_peptide_mappings(force=False)`; keyword defaults
    mirror the builder's, with a test that pins them together.
  - Returns `bool` only: the sidecar is either stamped against the observations on
    disk or it is not, so there is no unknowable case to encode as `None`.
- Both names join `_PUBLIC_API`; README documents them beside `is_built`.

## Plan

- [ ] `downloads.packaged_or_cached`: local-only twin of `packaged_or_fetched`.
- [ ] Thread `fetch_missing_assets` through `_curation_fingerprints`,
      `_source_fingerprints`, `_cache_is_valid`; default unchanged for the builder.
- [ ] Add the two predicates, export them, document them.
- [ ] Tests: `None` without sources; True/False on seeded caches; silent on stdout;
      never fetches; mapping defaults pinned to `build_peptide_mappings`.
- [ ] Smoke-test against the local `~/.hitlist` cache (artifact_version 3 vs code 4).
- [ ] Bump to 1.62.0, format, lint, test, PR, CI, merge, deploy from clean main.

# Flat experimental-condition columns

## Objective

Revise `tasks/condition-model-spec.md` around one consistent, flat condition schema that
curators can fill from papers and that reaches individual rows of the unified training table.
Keep source evidence, unknown values, and ambiguous peptide attribution explicit. Implement,
verify, and ship the resulting change through a versioned PR and PyPI release.

## Plan

- [x] Audit existing condition strings, metadata readers, and representative primary sources.
- [x] Write the flat schema, paper-extraction contract, migration, and acceptance criteria.
- [x] Implement validation, categorical annotations, and propagation through every export.
- [x] Curate the existing condition vocabulary and source-verified representative samples.
- [x] Verify combinations, controls, missing information, ambiguous assignments, and projection.
- [x] Run format, lint, tests, and a before/after corpus comparison; inspect the final diff.
- [ ] Bump the version, open the PR, verify CI, merge, and deploy from clean main.
- [ ] Record results and review the next relevant open issues.

## Review

Shipped as #450. 23 flat condition columns declared once in `hitlist/conditions.py`
and spliced into the loader schema, the samples row, the empty-frame schema, the
expression-anchor projection, the observation join and the training defaults.

**Curation.** All 761 arms annotated from the 167 distinct condition strings, written
into the YAML as reviewed data (`curated_text`), every pre-existing value byte-identical.
19 arms across four studies upgraded to `primary_source` with section-level locators.

**What the pilot caught.** Reading the papers disproved three curated facts that no
consistency check could see: Stopfer 2021's biopsies are snap-frozen not fresh, and
its `condition_control: untreated` asserted a therapy status the paper never states;
Javitt 2019's A549 genotype carried `HLA-B*07:02` where both the paper and IEDB's own
rows say `HLA-B*18:01`. Lorente 2019's arms are correct but incomplete (5 profiled,
2 curated) — filed as #452 rather than expanded here.

**Attribution.** The tie guard now compares curated `condition_id` instead of the
coarse category, so 184,811 rows stop being assigned an arm they were never entitled
to. `_candidates_disagree_on_arm` deliberately stays on the category — that gate asks
a different question, and identity there cost 7,629 correct discriminations.

**Filed, not worked around.** #451 (class-pool path leaves ungrouped rows blank
instead of `pmid_ambiguous`; the fix moves 1.23M rows and needs its own verdict pass),
#452 (Lorente arm split).

Corpus: 4,439,321 rows before and after. 1,532 tests pass including integration.

---

# Arm attribution cluster — #442 / #366 / #359 / #364 / #362

## Release split

Approved plan: three sequenced PRs, each version-bumped and deployed before the next branches.

1. **1.58.9 — #442 QC check, study-level schema guard, #362 close-out.** Code and docs only;
   the corpus must come out byte-identical.
2. **1.59.0 — #359 + #364.** `sample_group` curation vocabulary plus group-aware attribution,
   then the arm curation that depends on it.
3. **1.59.1 — #366.** `arm_resolution` accounting across all 28 ambiguous studies.

## What the investigation changed

Re-measuring on the current corpus before writing code contradicted two of the five issues.

- **#362 is already fully delivered**, not partly. Its two proposals (carry the matched sample
  onto observations; denormalize the APM block) are live, and its remaining "consumer caveat" —
  that a WT control inherits the study panel's gene flags — describes pre-#353 behavior.
  Verified on the Shapiro HAP1 panel: `HAP1 wildtype` reports `apm_genes_perturbed=""` and
  `apm_perturbed="false"` while `study_apm_perturbed` stays True. Documenting that caveat would
  have described a bug that no longer exists, so PR 1 pins the correct behavior with a test and
  closes the issue instead.
- **#359's premise is false.** A subagent ran the counterfactual end-to-end: making PMID
  29242379's arm labels symmetric yields `""`, not `pmid_ambiguous`, because the class-pool path
  has no `_consensus_meta` fallback; and the TIL/meningioma rows stay unattributed because
  `assay_comments` is blocked wholesale when candidate arms disagree. Labels alone make the
  output strictly worse, so PR 2 needs `sample_group` and a matcher change.
- **The asymmetry bug is a live mis-attribution, and wider than filed.** PMID 29242379's 3,919
  "attributed" rows land on the untreated arm because `ovarian` matches `source_tissue = Ovary`
  and appears on only that one of six candidates. PMID 30833945 has the identical pathology via
  `lung`. 8,595 rows are confidently attributed to a control arm on no evidence.
- **#366's biggest study is unresolvable by construction.** PMID 33858848 is 255,179 of the
  450,704 ambiguous rows; its arms are per donor and its evidence records tissue. Orthogonal
  axes, unrecoverable from IEDB.

## Steps

- [x] Re-measure all five issues on the current corpus; run the #359 counterfactual.
- [x] PR 1: `qc.sample_attribution_audit` + CLI, `PMID_ENTRY_FIELDS` guard, #362 close-out test.
- [x] PR 1: PR #445, CI green, merged, deployed 1.58.9, wheel verified; #442 and #362 closed.
- [x] PR 2: `sample_group` + group-aware attribution; curated all four studies.
- [x] PR 2: PR #446, CI green, merged, deployed 1.59.0; #359 and #364 closed.
- [x] PR 3: `arm_resolution` across all 31 ambiguous studies.
- [ ] PR 3: PR, CI, merge, deploy 1.59.1, close #366.

## Review

### PR 1 — #442 + #362 + the study-level guard (1.58.9)

`qc.sample_attribution_audit()` reports the 232 profiled arms that reach zero observation rows,
bucketed by whether the study attributes anything at all: 137 `label_mismatch_candidate` (the
join demonstrably works in that study, so the label is the suspect) and 95 `study_unattributed`
(a different failure, triaged per study). Deliberately not wired into `run_all` or
`curation_plan` — `sample_label` is synthesized by the export join, so this is the only check
needing the full enriched table, and either rollup would make every `hitlist qc` pay that build.

**#444 found while adding the guard.** `exclude_from_ms` is documented as excluding a study from
the MS index and is set on 11 studies, every one curated as *not* a mass-spec elution experiment
— yeast display, peptide microarray, refolding crystallography, computational tools. Nothing
reads it. 40,355 rows / 33,101 unique peptides from 6 of those studies are in the corpus and
reach the enriched export. Three more study-level keys are unread: `donors` (11 studies),
`samples` (2), `tissues` (1). `PMID_ENTRY_FIELDS` now declares all 37 permitted top-level keys
against their readers and the loader rejects the rest; the unread ones are declared as UNREAD
citing #444 rather than described as if they worked. `samples`/`tissues` renamed to
`n_samples`/`n_tissues` per the count-suffix rule.

Removing those 40,355 rows is a deliberate corpus change and ships separately.

### Review fixes folded into PR 1

`/code-review` on the branch returned 12 findings, 8 of them real defects in the #373 provenance
work already published as 1.58.8. Each was verified against the corpus before fixing.

- **`effective_override` skipped the `rules` level.** PMID 27846572's `primary fibroblasts`
  exported `cell_line` / `study` while the study's rule sends every Direct Ex Vivo fibroblast row
  to `healthy` — exported provenance contradicting the classification the build applied, on
  131,252 observation rows across 14 studies that carry both `rules` and `ms_samples`. Rules match
  per row, so a sample-level value cannot resolve them; the new origin `study_conditional` says
  the inherited value is a default a rule may supersede, instead of asserting finality.
- **`_consensus_meta` kept arm-specific claims on arm-less rows.** Arms of one study routinely
  agree on `override` (PMID 34129938 marks all six `cell_line`), so consensus preserved
  `origin="sample"` on rows whose `sample_label` it had just blanked. Latent only because those
  PMIDs have no rows; it would have broken `test_no_unattributed_row_carries_a_sample_level_override`
  on the next corpus refresh. Study-origin values still survive — those are deposit properties.
- **`_SAMPLE_PROVENANCE_COLUMNS` / `_TRAINING_DEFAULTS` not extended** — `--with-expression-anchors`
  silently dropped all six new columns, and binding rows got NaN where every other MS-only column
  gets `""`.
- Plus: a join docstring asserting the opposite of the join's behavior, an un-coerced legacy
  `notes` beside three coerced siblings, and a test using `.` as a regex stand-in for a literal `+`.

Three design findings taken: `note` → `sample_note` on observations (it collided with the
study-level `note` key *and* the `notes` column), `sample_override` dropped from the 4.4M-row join
as derivable from the other two, and a new test tying every declared `MS_SAMPLE_FIELDS` key to an
exported column — the old one only checked descriptions were non-empty, so "declared" could have
become a synonym for "accepted and ignored", the exact failure #373 exists to prevent.

Corpus effect: 13 sample rows move `study` → `study_conditional`; zero changes to observation row
identities. Gates: format, lint, 1,432 tests, build smoke.

### PR 2 — #359 + #364 (1.59.0)

`sample_group` names the sample *system* an arm belongs to. Attribution resolves the system
first — admitting IEDB's narrative fields, because identifying a system is what they do reliably
— then the arm within it, where those fields stay blocked. Opt-in per study and enforced
all-or-nothing at load, so a half-curated study cannot silently fall back.

The plan's counterfactual held: symmetric labels alone would have produced `""`. What makes the
study attributable is the group stage plus a `_consensus_meta` fallback on the class-pool path,
which the allele path already had.

| study | before | after |
|---|---|---|
| 29242379 Chong | 3,919 attributed, all to the wrong arm | 11,450 arm-resolved, 70,878 system-resolved |
| 30833945 Javitt | 4,676 attributed to the wrong arm | 7,524 system-resolved |
| 32938616 Faridi | 37,643 condition-only, panel-level | 37,643 line+condition, 54,526 line-resolved |
| 27371725 Nagarajan | 1,334 rows with no arm | all 1,970 reach a curated arm |

Corpus-wide: unattributed 1,630,156 -> 1,493,039; `discriminated` down exactly 8,595, the false
control-arm attributions and nothing else; `elution_conditions` unchanged at 74,304. Row
identities identical, and only the four curated PMIDs moved.

Two source findings changed the curation from what the issues assumed. **Nagarajan's classical
restrictions are NetMHC predictions**, not measurements: the study acid-eluted the whole cell
surface with no allele-specific pulldown, then assigned H2-Kb/H2-Db by NetMHC and Qa-2a by
Rankpep. Only Qa-1b was experimental, and no Qa-1b rows are in the corpus — so the entry sets
`restriction_evidence: predicted`, the vocabulary #415 exists for. The paper's real title covers
classical MHC, and IEDB was right that the cells are bone-marrow-derived dendritic cells, not the
curated "splenocytes". **Faridi ran three lines**, not a panel; LM-MEL-53's HLA type is stated
nowhere in the paper, so it stays class-only rather than inheriting LM-MEL-44's on the strength of
being the same patient.

Implementation note: the token scorer drops tokens under three characters, so `LM-MEL-44` and
`LM-MEL-33` both reduced to `mel` and tied — the digits that distinguish them were invisible. The
group selector now tries an alphanumerics-only containment test first (`lmmel44` inside
`lmmel44melanocyte`), per-row factual fields before narrative ones, falling back to token scoring.
That recovered all 37,643 elution-resolved rows, which the first attempt had cut to 14,617.

### PR 3 — #366 (1.59.1)

Every one of the 584,966 ambiguous rows now carries a recorded reason. Zero unexplained.

| verdict | rows | means |
|---|---|---|
| `axis_mismatch` | 255,179 | curated arms and recorded metadata on different axes (33858848: donor vs tissue) |
| `curation_gap` | 145,279 | a per-row discriminator exists; the only verdict marking real work |
| `no_row_discriminator` | 129,982 | measured — all four per-row fields take one distinct value across the study |
| `multi_arm_evidence` | 54,526 | the evidence positively places the peptide in more than one arm |

**75% of the ambiguity is settled**: a property of what was deposited, not of how carefully
anyone curated. That is the answer #366 wanted, and the reason a five-value vocabulary beat the
planned three — the data showed three genuinely different kinds of unresolvable, and calling
32938616's "peptide really was in both arms" case *unresolvable* would have been wrong.

Two guards stop a verdict becoming a way to stop looking. `arm_resolution_note` must carry the
measurement, and a note without a verdict is rejected at load. And a test re-measures every
`no_row_discriminator` study against the corpus: if a refresh gives one a varying per-row field,
the verdict is stale and the study is re-audited rather than silently trusted.

`hitlist qc sample-attribution --actionable-only` hides findings in settled studies, so the #442
audit distinguishes "unresolvable" from "unexamined".

Purely additive: one new column on the samples export, no pre-existing sample or observation
value changed.

---

# Sample-curation conservation and primary-source audit — #438 / #437 / #436 / #373

## Release split

The audit turned one issue into four with a hard dependency order, so it ships as three PRs
rather than one. Each bumps the version and deploys before the next branches from main.

1. **1.58.6 — #438 + #437, inventory conservation.** Every raw `ms_samples` record must survive
   loading and reach the sample export. This is foundational: #436 and #373 both reason about
   per-sample records, and today five of them do not exist as far as the loader is concerned.
2. **1.58.7 — #436, verified curation corrections.** Data-only fixes to the four studies whose
   sample curation source verification disproved.
3. **1.58.8 — #373, sample-level `override` / `note` semantics + schema guard.** Enabling a
   sample-level override before #436 would promote wrong curation into row classification, so
   this lands last.

## Specification

### PR 1 — #438 + #437 (1.58.6)

- `load_pmid_overrides()` rejects duplicate `pmid` identifiers before building its mapping,
  naming every duplicated key. Today a dict comprehension silently keeps the last entry.
- Consolidate the two duplicate pairs into one entry each, preserving both the study/sample
  metadata of the earlier block and the `source_organism` / `species` curation of the later one.
  Keep the exact `species` strings the scanner currently reads so no observation row changes.
  Correct the two wrong study labels the duplicates introduced (verified against PubMed:
  33460454 is Gastaldello 2021, not "Owen 2021"; 28188227 is Barnea 2017, not
  "Alvarez-Navarro 2018").
- `generate_ms_samples_table()` keeps `n_samples: 0` / `profiled: false` records instead of
  dropping them, exporting `profiled="false"` and a null `n_samples`.
- The observation metadata join excludes unprofiled samples explicitly, so restoring the
  metadata cannot manufacture a peptide observation or an arm match. (Binding needs no change:
  `generate_binding_table` never joins `ms_samples`.)
- Regressions: raw-YAML-to-loader and loader-to-export inventory conservation, duplicate
  rejection, unprofiled round-trip, join exclusion, and unchanged observation row identities.

### PR 2 — #436 (1.58.7)

- SKMEL5 +/- binimetinib replaces the A375 +/- trametinib curation for PMID 34497125.
- Add the MC38 idAdpgkG IFN-gamma / doxycycline / dTAG-13 arms for PMID 34129938.
- Add the WT/mock, TAP1-KO/mock, WT/H37Rv, TAP1-KO/H37Rv THP-1 conditions, primary human
  macrophages, and the Alg8-pulsed splenocytes for PMID 39438697.
- Correct the C1R note for PMID 27846572 (Caron 2015, not Bassani-Sternberg 2015) and add the
  missing T2 sample. Preserve each arm without inventing allele typing.
- Record source URLs, tables/figures, deposit file names, and affected corpus counts in an audit
  document; compare evidence identities before and after.

### PR 3 — #373 (1.58.8)

- Export original `note`, `classification`, and `reason` separately, retaining legacy `notes`.
  Carry the note and provenance override into MS/training evidence through the existing sample
  join. A sample override takes precedence only on a resolved sample attribution. Explicit null
  clears study-level override; absent sample override retains the normal PMID/rule behavior.
  Export the effective override value and its origin so null, inheritance, and ambiguity remain
  distinguishable. Preserve existing raw evidence fields and identities.
- Validate sample keys during YAML loading against an explicit mapping of consumed fields and a
  guard covering every current YAML key; no silently accepted typos.

## Steps

- [x] Enumerate 748 raw sample records: five lost to duplicate PMID replacement, then four
      explicit unprofiled records dropped from the 743 retained by the loader.
- [x] Verify the paper and deposit identities; identify and file additional curation defects.
- [x] PR 1: failing inventory/duplicate/unprofiled tests, then the loader, YAML, and export fixes.
- [x] PR 1: corpus identity comparison and gates.
- [x] PR 1: opened #439, CI green, merged, deployed 1.58.6, verified the published wheel.
- [x] PR 2: finished the source audit; applied and verified the curation corrections.
- [x] PR 2: corpus comparison, gates, PR #441, CI green, merged, deployed 1.58.7, wheel verified.
- [x] PR 3: override/null/inheritance/ambiguity tests, then the metadata and schema changes.
- [ ] PR 3: corpus comparison, gates, PR, CI, merge, deploy 1.58.8.

## Review

### PR 1 — #438 + #437 (1.58.6)

Eight new regressions in `tests/test_sample_inventory.py` failed first: duplicate PMIDs present
in the packaged YAML, no duplicate rejection in the loader, five sample records lost between the
file and the loaded mapping, two consolidated entries missing their samples or their provenance,
and four unprofiled records missing from the export.

Two existing tests pinned the behavior this PR reverses and were rewritten rather than deleted:
`test_ms_samples_no_zero_n` asserted every exported count was positive — true only because the
records it describes were being dropped — and `_KNOWN_CHIMERIC_SAMPLES` gained the two
HLA-B27-transgenic-rat arms, which are genuine chimeras (rat host and proteome, human HLA
transgene) that only became visible once 28188227's duplicate blocks were consolidated.

Verified against the full local corpus, before (ca5648c) versus after:

| | before | after |
|---|---|---|
| sample export rows | 739 | 748 |
| observation rows | 4,439,321 | 4,439,321 |

The nine added sample rows are exactly the five records the duplicate PMIDs discarded (three
Tasmanian devil, two transgenic rat) plus the four explicitly unprofiled records. No sample row
was removed, the column set is unchanged, and no field on any pre-existing row changed.

Observation `pmid`, `peptide`, `mhc_restriction`, `source_organism`, and `species` are identical
row-for-row across all 4,439,321 rows — the consolidation deliberately kept the exact `species`
strings the scanner already read, so no provenance fill moved. The only observation change is
metadata gained: 33,959 Tasmanian-devil rows that were previously unattributed now discriminate
into their three curated arms, including the IFN-gamma arm and its control-arm flags. PMID
28188227's 59,778 rows stay unattributed, correctly — both curated arms carry the same imprecise
`mhc: HLA-B*27`, so nothing distinguishes WT from ERAP1-KO and the join declines to guess.

The join guard is load-bearing rather than defensive bookkeeping: attribution path 3c matches
`attributed_sample_label` against `sample_label` with no allele involved and overrides every
heuristic above it, so a curated per-row label colliding with an unprofiled arm would attribute
real peptides to a sample the paper says was never profiled.

README corpus counts were stale independently of this change (159 PMIDs / 633 samples / 446
typed) and are now recomputed: 215 / 748 / 579, covering 96.0% of observations.

Gates: format, lint, 1,397 tests, and the packaged-build smoke tests all pass. CI, merge, and
publication are pending.

### PR 2 — #436 (1.58.7)

Source verification is complete for all four studies; the findings are recorded on #436 and
summarized in `docs/pmid-curation.md`. Three failure modes recurred: a plausible cell line
substituted for the real one (and its genotype carried along), a perturbation axis collapsed to a
single "unperturbed" arm, and a citation inverted.

Sample export 748 -> 755: seven removed (two phantom Liepe arms, two wrong-line Stopfer arms,
three replaced Leddy arms), fourteen added. Five surviving rows changed fields, all intended:
Liepe's GR-LCL and C1R genotypes, Pollock's two rows gaining study-level APM flags from the new
`perturbations:` block, and Stopfer's biopsy row gaining a `source`.

Observation rows stay at 4,439,321 with `pmid`, `peptide`, `mhc_restriction`, `source_organism`,
and `species` identical row-for-row. **Exactly 111 rows changed attribution**, and they are the
finding of this PR: T2's rows were being attributed to the phantom `JY (EBV-LCL)` sample by
`allele_exact` — JY's curated `HLA-A*02:01` matched them at the highest-confidence tier — so a
TAP-deficient hybridoma was labeled an EBV-LCL. A curated sample the paper never mentions is not
a harmless extra row; it competes for real evidence. They now attribute to T2, still
`allele_exact`.

Two existing tests were coupled to the corrected labels and were updated, not weakened:
`test_real_mixed_species_studies_keep_their_per_sample_species` (renamed arms; the per-sample
`species:` it guards is intact) and the stale docstring on
`test_resolver_skips_apc_when_cell_name_varies`, which explained the unassigned B-cell rows as
GR-LCL/JY ambiguity. Those 11,733 rows carry a six-allele `mhc_restriction` rather than a single
allele, so no allele path can fire; they were unassigned before and after, and JY was never a
candidate for them.

### PR 3 — #373 (1.58.8)

`override` (13 samples) and `note` (3) reached no consumer at sample level. Both are now
exported, along with `classification` and `reason` as their own columns beside the unchanged
legacy `notes`.

The resolution keeps four cases distinct rather than collapsing them to a value:
`sample` (the arm's own claim), `sample_null` (a curator considered this arm and decided
against one), `study` (inherited), `none` (nobody curated one). `sample_null` and `none`
produce the same empty value and mean different things; PMID 34497125 is the shape that
matters — two `cell_line` arms beside a patient-biopsy arm explicitly marked null.

`curation.MS_SAMPLE_FIELDS` declares all 22 permitted keys against what reads each, and the
loader rejects an undeclared key. `curation.OVERRIDE_VALUES` does the same for the override
vocabulary, which previously existed only in `classify_ms_row`'s branch chain and a YAML header
comment, so a misspelled override fell through to default classification. Study, rule, and
sample levels are all validated.

Verified before and after: sample export 755 -> 755 with six columns added, none removed, and
every pre-existing column byte-identical; observations 4,439,321 rows with `pmid`, `peptide`,
`mhc_restriction`, `source_organism`, `species`, `sample_label`, `sample_attribution`,
`perturbation`, and `is_control_arm` all identical row-for-row. Purely additive.

Origins across the 755 samples: study 513, none 229, sample 12, sample_null 1.

Two limits stated rather than papered over. The classification flags stay build-time and
PMID/rule-driven — `classify_ms_row` runs in the scanner, before any sample attribution exists,
so a sample-level override cannot feed them without recomputing flags in the export and letting
it disagree with the raw index. And `origin == "sample"` appears on no observation row today,
because all 13 override-bearing samples belong to PMIDs with zero rows in the current corpus;
the deterministic sample-export tests cover those cases, and the corpus test asserts the
negative — no unattributed row may carry a sample-origin value.

Filed #442 while verifying: 232 of 684 profiled curated samples are never attributed to any
observation row, including PMID 31844290's `ccRCC Pat9`, whose caveat is the one #373 exists to
surface.

### Audit carried into PR 3

The three PMIDs carrying sample overrides have zero rows in the current local indexes. The
two note-bearing studies contain 295,895 MS rows in total; Liepe 2016 also has 90 binding rows.
Source verification already disproved the SKMEL5/A375 and binimetinib/trametinib curation,
and found omitted MC38 and THP-1 perturbation arms.

Consolidating the duplicates also corrected two study labels that named the wrong first author:
33460454 is Gastaldello 2021 (Immunology, 10.1111/imm.13307), not "Owen 2021", and 28188227 is
Barnea 2017 (MCP, 10.1074/mcp.M116.066241), not "Alvarez-Navarro 2018". Both verified against
PubMed. 28188227's invented `title` was replaced with the published one.

---

# PR 3 specification — #426

- Preserve the full-load source-species selection in projected loads by reading both raw
  source-organism inputs before the derived filter is evaluated.
- Verify MS and binding with full output, peptide-only output, and explicit derived-column
  output. Include blank/null source organisms, populated fallback species, a conflicting fallback
  that must lose to the primary value, and unresolved rows.
- Ship as 1.58.3 after the gene-query PR has merged and deployed. Run the required gates on the
  final rebased branch; merge only after all CI checks pass, then deploy from clean main.

---

# PR 4 specification — #427

- Forward allele-set and provenance filters through the shared observations/binding loaders,
  preserving the export's comma-separated list input handling, and delete the duplicate filters.
- Verify canonical/bare/case-variant alleles, list and comma-separated inputs, combined provenance
  filters, and explicit empty inputs across raw loaders, MS/binding exports, and training modes.
- Ship as 1.58.4 after the preceding PRs have merged and deployed. Run format, lint, and the full
  default suite on the final branch, require every CI check, and deploy from clean main.

---

# Data consistency release series — issues #424–#427

## Priority and acceptance contract

1. **#424 / v1.58.1 — current curation in persisted data.** Include the scanner's curation YAML
   inputs and peptide-attribution CSVs in observations cache fingerprints. Cover cell-line
   metadata as well as study overrides, tissues, and monoallelic hosts. A curation-only edit must
   invalidate the cache and a normal rebuild must persist the edited value, including consecutive
   builds in the same interpreter. Existing metadata without these inputs must rebuild once.
2. **#425 / v1.58.2 — complete gene-query results.** Preserve OR semantics for mixed gene-symbol
   and Ensembl-ID queries across MS, binding, and training. Apply the same gene selection when
   expanding source-protein mappings; do not reintroduce unrelated mappings through a shared
   peptide. Keep explicitly supplied low-level name/ID filter semantics clear.
3. **#426 / v1.58.3 — projection-independent species filters.** A source-species filter must
   select the same evidence whether or not the output projects its derived column or fallback
   inputs. Exercise both indexes, missing source organism, and a nonmatching control row.
4. **#427 / v1.58.4 — one allele-set filter contract.** Route export filtering through the
   loader's normalization and validation so aliases, canonical inputs, and invalid empty queries
   behave consistently in MS, binding, and training exports.

## Execution and verification

- [x] Recheck the reported defects, current main, existing PRs, and release scripts.
- [x] PR 1: add failing cache/rebuild regressions; implement #424; review; format, lint, test;
      bump version; open PR; require all CI checks; merge; deploy from clean main; verify PyPI.
- [x] PR 2: add mixed-query and mapping-expansion regressions; implement #425; run all gates;
      bump version; open PR; require CI; merge; deploy from clean main; verify PyPI.
- [x] PR 3: add projected-filter regressions; implement #426; run all gates; bump version;
      open PR; require CI; merge; deploy from clean main; verify PyPI.
- [x] PR 4: add loader/export parity regressions; implement #427; run all gates; bump version;
      open PR; require CI; merge; deploy from clean main; verify PyPI.
- [x] Review remaining related Hitlist/ecosystem issues by dependency and data-quality impact.

Each PR has its own feature branch and patch release. The actual `deploy.sh` publishes the
version already in `hitlist/version.py`, so each bump is made explicitly in its PR. Reproduction
fixtures must be independent of the developer's data cache. Required release validation includes
`./test.sh --all` through the deployment script; never infer CI success from local tests.

## Review

- PR 2 design: keep the low-level mapping/observation filters conjunctive, and resolve the
  export's combined gene-query axis through one internal union loader. Use that same helper
  for evidence selection and source-protein expansion, deduplicating mappings that match both
  name and ID. Explicit peptide filters intersect the selected gene peptides, including empty
  intersections. Tests cover both evidence kinds, same-gene aliases, unknown queries, and shared
  peptides whose unrelated source-protein mappings must remain excluded.
- PR 1 implementation fingerprints the three curation YAMLs, cell-line registry, and every
  referenced peptide-attribution CSV with content hashes. Missing fingerprint entries invalidate
  old metadata automatically. Builds clear the file-backed curation caches and derived results.
- The two-build regression exposed #428: nested unbounded caches defeated bounded-cache eviction
  and retained stale classifications after `cache_clear()`. Removed all four redundant wrappers;
  source category and restriction evidence now both change on the next normal build.
- PR 1 validation: all seven new regressions pass; format and lint pass; `./test.sh` passes
  1,205 tests with one expected warning. The earlier focused builder/curation/smoke run passed
  290 tests. PR #429 passed every CI check, merged, and shipped as 1.58.1. The deployment's
  complete suite passed 1,229 tests, and both wheel and sdist hashes match the PyPI artifacts.
- PR 2's mixed-query regression also exposed #430: an untyped empty peptide IN predicate cannot
  bind to a large-string parquet column. The shared loader now types its peptide value set so
  unmatched queries and empty intersections return zero rows. Targeted export/loader tests pass
  201 tests. Format and lint pass; `./test.sh` passes 1,232 tests with one expected warning.
  PR #431 passed all CI jobs, merged, and shipped as 1.58.2. Its complete release suite passed
  1,256 tests. Both distribution hashes match the published PyPI artifacts.
- PR 3 adds the raw `species` fallback to the columns read for species-axis filters. All six
  new full/projected regressions pass, and the focused observations suite passes 45 tests.
  The branch is rebased on PR #431 and bumped to 1.58.3. Format and lint pass; the default suite
  passes 1,238 tests with one expected warning. PR #432 passed every CI job, merged, and
  shipped as 1.58.3. Its complete release suite passed 1,262 tests, and both distribution
  hashes match the published PyPI artifacts.
- PR 4 deletes the duplicate export membership filters and forwards both allele-set and
  provenance arguments through the loaders. All 42 new parity regressions pass; the combined
  export suite passes 200 tests. The branch incorporates PR #432's species fix and is bumped
  to 1.58.4. Format and lint pass; the default suite passes 1,280 tests with one expected
  warning. All seven original review reproductions pass on the combined series. PR #433 passed
  every CI job, merged, and shipped as 1.58.4; all 1,304 release tests passed and both PyPI artifact
  hashes matched the local builds. Final verification is also recorded in the PR description.

## Next priorities after this series

1. **#386 — export species-filter parity.** Expose the loader's source-species, host-species,
   and chimeric filters through exports. The projection fix in this series is its foundation.
2. **#373 — sample-curation schema validation.** Reject or implement ignored sample `override`
   and `note` fields, with explicit provenance semantics. Avoid accepting curation that has
   no effect on the output.
3. **#357 — expression coverage.** Make missing DepMap expression data visible and provide
   an explicit acquisition path; resolved sample identifiers alone do not establish coverage.
4. **Evidence-dependent curation.** Keep #359 and #366's ambiguous experimental arms unresolved
   until source evidence distinguishes them. Resolve pirl-unc/mhcgnomes#190 before extending
   the affected mouse curation in #364; track helper-gene parsing in mhcgnomes#191 separately.

The related mhcgnomes and openvax/pyensembl backlogs were reviewed. Parser correctness is upstream
of affected curation; unrelated parser display, packaging, and optional FASTA work do not block
the four releases above.

---

# Bug review — 2026-09-05

## Scope and approach

Review the current main snapshot (`8fd02b3`) for reproducible correctness defects, with emphasis
on public filtering/export contracts, sample attribution, and cached build artifacts. Use the
existing tests as a baseline and isolated temporary fixtures to verify suspected failures. This
is a findings review; production fixes and their release workflow are a separate follow-up.

## Steps

- [x] Read repository guidance, lessons, recent history, and package/test configuration.
- [x] Run formatting, lint, and the existing test suite to establish the baseline.
- [x] Trace public API and CLI paths through filtering, joins, and artifact reuse.
- [x] Reproduce actionable defects and check for existing GitHub issues.
- [x] File confirmed new bugs with exact reproduction and impact, as required by AGENTS.md.
- [x] Record verification results and report prioritized findings with source locations.

## Review

- Reviewed main `8fd02b3` / Hitlist 1.58.0. No production code changed; the local review branch
  contains only this task record. No PR/release is part of this findings-only review.
- `./format.sh` made no changes; `./lint.sh` passed; `./test.sh` passed all 1,198 default-suite
  tests with one expected backend-exception warning. Integration tests were excluded by the
  script's default selection. This is local baseline validation, not a new CI run.
- Seven isolated assertions reproduced four defects, independently covering both MS and binding
  paths where applicable. Reproductions use temporary fixtures and no network. The full local
  reproduction file is `/private/tmp/hitlist_bug_review_20260905.py`; each issue also contains a
  self-contained reproduction whose output was executed and verified before filing.
- **P1 — [#424](https://github.com/pirl-unc/hitlist/issues/424):** observations cache fingerprints
  omit the main curation YAML inputs. Changing `restriction_evidence` from experimental to
  predicted leaves the cache valid, permitting stale persisted scientific annotations.
- **P2 — [#425](https://github.com/pirl-unc/hitlist/issues/425):** mixed gene-symbol/Ensembl-ID
  export queries use AND instead of the documented OR semantics. Two individually matching
  genes yield zero rows when queried together; mapping expansion repeats the same conjunction.
- **P2 — [#426](https://github.com/pirl-unc/hitlist/issues/426):** projecting only `peptide`
  omits the raw `species` dependency of a `source_species` filter, dropping rows that need its
  documented fallback when `source_organism` is blank.
- **P2 — [#427](https://github.com/pirl-unc/hitlist/issues/427):** export allele-set filters omit
  normalization that the raw loaders perform. `A*02:01` matches the loader but silently returns
  zero exported rows where canonical `HLA-A*02:01` succeeds.

---

# Issues #418 and #419 — assay routing and provenance CLI parity

## Goal

Keep biochemical purified-MHC stability measurements out of the MS-elution index, and make every
MHC allele-set provenance value emitted by Hitlist valid at each CLI export entry point.

## Premise check

- The registered 2026-03-30 IEDB source has 222 rows for PMID 36423003: 146
  `cellular MHC/mass spectrometry` ligand-presentation rows and 76
  `purified MHC/direct/radioactivity` half-life rows.
- The existing qualitative-only classifier routes 66 non-`Positive` half-life rows correctly but
  leaks the 10 plain-`Positive` half-life rows into `observations.parquet`. They are eight
  BoLA-6*013:01, one BoLA-1*023:01, and one BoLA-2*012:01 row; the issue's three-DRB3/153-MS count
  does not match the current registered source or built corpus.
- The same mechanism affects 942 plain-`Positive`, purified-MHC half-life rows corpus-wide (646
  fluorescence and 296 radioactivity). Classification must be based on the structured method and
  response, not a PMID or allele allow-list.

## Design

- Extend `is_binding_assay` with optional `assay_method` and `response_measured` inputs, preserving
  source compatibility for existing two-argument callers.
- Treat a half-life response measured on purified MHC as binding evidence. Keep the existing
  qualitative/comment rules unchanged in this PR, and explicitly retain cellular-MHC mass
  spectrometry ligand-presentation rows.
- Pass the already-scanned structured fields into the classifier. Bump the observations artifact
  contract so existing cached parquets rebuild instead of retaining stale routing.
- Promote the complete MHC allele-set provenance tuple to one public constant, use it for the
  scanner contract and all three CLI `choices`, and export it through the lazy top-level API.
- Add scanner/classifier regressions for both assay types and end-to-end parser coverage proving
  every emitted provenance value is accepted by `export ms`, `export binding`, and
  `export training`.
- Bump to 1.58.0: this changes which public evidence index contains existing rows and adds a public
  vocabulary constant plus a new accepted CLI value.

## Steps

- [x] Reproduce both reports and audit #418 against the registered raw-source snapshot.
- [x] Correct #418's stale row-count/allele premise on the issue.
- [x] Add failing classifier, scanner, and CLI regressions.
- [x] Implement structured assay routing, cache invalidation, and shared provenance choices.
- [x] Verify targeted raw-source output and real rebuilt index counts.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [ ] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- The fix keys on a source-defined assay signature, not PMID or allele identity. An isolated full
  rebuild moved all 942 matching assay IRIs from MS to binding, left zero matching rows in MS, and
  confirmed every moved IRI in `binding.parquet`.
- PMID 36423003 now has 146 cellular mass-spectrometry rows and 76 purified-MHC half-life rows in
  their respective indexes. The current IEDB source already carries correct structured metadata,
  so no upstream issue is warranted.
- `MHC_ALLELE_PROVENANCE_VALUES` is now the public contract used by all three CLI parsers; the
  parser regression exercises every value for MS, binding, and training exports.
- Artifact contract version 2 forces existing cached parquets to rebuild with the corrected
  routing. Hitlist is bumped to 1.58.0 because existing evidence moves between public indexes and
  a public vocabulary/CLI value is added.
- `./format.sh` and `./lint.sh` passed; `./test.sh` passed 1,198 tests with one expected warning;
  `tests/test_build_smoke.py` passed 2/2.

---

# Issue #416 — preserve study-specific THP-1 HLA typing

## Goal

Stop offering HLA-A*24:02 and HLA-B*35:01 as candidate presenters for PMID 35051231 when that
study explicitly typed and analyzed its THP-1 sub-line as homozygous HLA-A*02:01,
HLA-B*15:11, and HLA-C*03:03. Preserve the heterozygous typing for PMID 33392160, whose authors
used DSMZ ACC-16 and reported that genotype.

## Source finding

- Nicholas 2022 S1 Table repeats A*02:01, B*15:11, and C*03:03 in both class-I haplotypes for
  THP-1. The Results explicitly call the three loci homozygous and assign 6,499 peptides to the
  A*02:01 and B*15:11 motifs.
- Ghosh 2020 reports A*02:01/A*24:02, B*15:11/B*35:01, and C*03:03 for its THP-1 culture and
  states that the line came from DSMZ ACC-16.
- Cellosaurus CVCL_0006 preserves both homozygous and heterozygous typing records and cites a
  paper specifically documenting the THP-1 HLA discrepancy. These are study/sub-line-specific
  facts, not two spellings of one canonical genotype.

## Design

- Replace only the three Nicholas THP-1 sample genotypes with the study's homozygous class-I
  typing while retaining its reported class-II typing unchanged.
- Keep Ghosh/DSMZ and other independently sourced THP-1 entries heterozygous unless their own
  source says otherwise. Do not create an allele alias or globally rewrite THP-1.
- Clarify the PMID notes and the shared `cell_lines.yaml` entry: the registry owns canonical line
  identity, while `ms_samples[].mhc` owns study-specific typing. Nicholas gives no catalogue
  number, so do not invent a second Cellosaurus accession or sub-line name.
- Add regressions at the curation and observation-join boundaries proving that PMID 35051231
  excludes A*24:02/B*35:01 and PMID 33392160 retains them.
- Bump the patch version to 1.57.2.

## Steps

- [x] Inspect issue #416 and verify Nicholas S1/Results, Ghosh Results, DSMZ, and Cellosaurus.
- [x] Add failing study-specific typing and observation-join regressions.
- [x] Correct the three Nicholas samples and document the source-specific registry contract.
- [x] Verify affected real-PMID outputs and the corpus-wide sample-ploidy audit.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [x] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- The correction is confined to the three Nicholas THP-1 arms. The Ghosh/DSMZ sample remains
  heterozygous, and the shared cell-line registry now explicitly directs consumers to the
  study-level genotype.
- A real-data join over PMID 35051231's B*15:11 observations contains neither A*24:02 nor B*35:01
  in exact or class-pool sample metadata; `sample_ploidy_audit()` remains empty corpus-wide.
- No upstream issue is warranted: Nicholas, Ghosh, DSMZ, and Cellosaurus faithfully expose the
  divergent source/sub-line typings; the defect was Hitlist's choice to substitute a global
  superset for Nicholas's reported genotype.
- `./format.sh` and `./lint.sh` passed; `./test.sh` passed 1,192 tests with one expected warning;
  `tests/test_build_smoke.py` passed 2/2.
- PR #422 merged as `bb11d78`; Hitlist 1.57.2 was uploaded to PyPI as both wheel and sdist.

---

# Issue #414 — separate the measured A19 genotype from predicted restrictions

## Goal

Represent both source facts from PMID 36423003 without converting one into the other: 2824TP's
measured A19 genotype contains `BoLA-6*014:02`, while the paper and supplement assign three retained
peptides to `BoLA-6*014:01` by prediction.

## Source finding

- Results 3.2 and the independent Vasoya et al. A19 definition both give the genotype as
  `BoLA-2*016:01 BoLA-6*014:02`.
- Table 3 and Supplementary Data 2.1 report predicted restrictions as `BoLA-6*014:01`; the
  supplement contains 23 such predictions, all from the two 2824TP runs, and no `*014:02`
  prediction. IEDB faithfully retains three final-table rows, so this is not an IEDB ingestion bug.
- Supplementary Data 2.1 also contains 16 `BoLA-2*008:01` predictions from 2123TP, but none survive
  into the final Table 3 / IEDB set. Keep it as measured sample typing without inventing an
  observation.

## Design

- Correct the 2824TP `ms_samples[].mhc` genotype to `BoLA-6*014:02`.
- Register the paper's three peptide-to-2824TP mappings through the existing
  `peptide_attributions` mechanism.
- Generalize scan-time attribution so a curated mapping can label an already allele-resolved row.
  Preserve its reported restriction, allele set, and `exact` allele-set provenance; only add the
  source-backed sample label. Emit one row per attributed sample if a future resolved peptide maps
  to more than one.
- Let the observation join use the existing `curated_sample_label` path. The public `sample_mhc`
  must show `*014:02`, while `mhc_restriction` remains the paper's predicted `*014:01` and
  `restriction_evidence` remains `predicted`.
- Do not create an allele alias or claim that `*014:01` and `*014:02` are equivalent molecules.
  Keep the unexplained source-level mismatch explicit in the study note and #414.
- Bump the patch version to 1.57.1, since this corrects existing sample typing and attribution
  without adding a new public schema field.

## Steps

- [x] Inspect the article, independent A19 definition, and all relevant supplementary workbooks.
- [x] Correct #414's issue record with the primary-source finding.
- [x] Add failing curation, scanner, and observation-join regressions.
- [x] Implement resolved-row sample attribution and correct the PMID curation/data asset.
- [x] Verify the real three-row output and the corpus-wide sample-ploidy audit.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [x] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- Primary-source checking showed that IEDB faithfully represents this paper's own prediction
  table, so no upstream IEDB issue is warranted. The unresolved mismatch is within the published
  source: its measured A19 genotype uses `*014:02`, while its retained predictions use `*014:01`.
- The three predictions now retain the exact resolved molecule and `predicted` evidence while a
  separate curated label selects 2824TP, whose public sample genotype contains `*014:02`.
- A real scan of the 7.77 GB IEDB export found exactly the three expected PMID rows with this
  separation. The corpus-wide sample-ploidy audit remains clean.
- Release gates are clean: formatting and lint passed, the full suite passed 1,190 tests with one
  expected warning, and the two build-smoke tests passed against regenerated artifacts.
- PR #421 merged as `c43937b`; Hitlist 1.57.1 was uploaded to PyPI as both wheel and sdist.

---

# Issue #415 — restriction evidence is separate from allele-set provenance

## Goal

Expose whether a named MHC restriction was experimentally isolated, implied by a monoallelic
system, computationally predicted, or not established. Keep this independent from
`mhc_allele_provenance`, which answers only where the candidate allele set came from.

## Design

- Add a categorical `restriction_evidence` column with four values: `experimental`,
  `monoallelic`, `predicted`, and `unknown`.
- Infer `experimental` only for resolved restrictions in binding assays and `monoallelic` only
  for resolved restrictions whose existing sample classifier proves a monoallelic system.
  Everything else remains `unknown` unless explicitly curated.
- Support PMID-level defaults plus condition-matched `restriction_evidence_rules`, so mixed
  studies can describe one evidence-generating method without relabeling unrelated rows.
- Curate PMID 36423003's resolved cellular-MHC/MS allele assignments as `predicted`; its
  class-only row and purified-MHC half-life rows must not inherit that claim.
- Carry the axis and an exact-value filter through the canonical observation, binding, export,
  training, and CLI paths. Do not change `mhc_allele_provenance` values or semantics.
- Treat the newly discovered purified-MHC index leak as separate issue #418 rather than hiding
  it in evidence curation.

## Steps

- [x] Add failing unit and scanner regressions for inferred, curated, mixed-study, and unresolved
      evidence states.
- [x] Implement and validate the evidence vocabulary and conditional curation API.
- [x] Persist `restriction_evidence` in scanner and supplementary rows; categorize it in builds.
- [x] Add load/export/training/CLI filters and document the new schema contract.
- [x] Curate PMID 36423003 and verify real-corpus counts by evidence and assay family.
- [x] Bump the minor version and update changelog/release-facing version surfaces.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [x] Open the PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- `mhc_allele_provenance=exact` remains a structural statement: the source row named a concrete
  allele. `restriction_evidence=predicted` can now accompany it without laundering a predictor's
  assignment into an experimental observation.
- The only inferred positive labels are mechanically safe: resolved binding-assay restrictions
  are `experimental`, and resolved rows already proven monoallelic are `monoallelic`. All other
  rows default to `unknown` unless a validated PMID rule says otherwise.
- PMID 36423003 uses a method-and-response rule. Raw-source verification separated resolved
  cellular-MS predictions from the class-only row and the purified-MHC half-life assays.
- That verification exposed two unrelated pre-existing defects, filed as #418 (binding-index
  leakage) and #419 (CLI rejects `peptide_attribution` provenance). Neither is hidden or folded
  into this PR.
- Focused checks passed 10 tests; the complete impacted curation/scanner/supplement/load/export
  suite passed 460 tests.
- Release gates are clean: formatting and lint passed, the full suite passed 1,185 tests, and the
  two build-smoke tests passed against regenerated artifacts.

---

# PR #417 review fixes — class-safe and heterodimer-safe attribution

## Goal

Resolve every review finding without weakening the new sample-MHC attribution contract:
peptide support must stay within the target MHC class, serotype matching must work for either
chain of a reported class-II heterodimer, typed nonmatching serotypes must not become unknown
support, and merged HLA-DM measurements must not be labeled as control samples.

## Design

- Infer the query MHC class from the requested allele or serotype and use it when the caller did
  not supply `mhc_class`; also gate every summary row against that target class so monkeypatched,
  legacy, or explicitly broad observation frames cannot contribute opposite-class evidence.
- Make the sample join symmetric for class-II pairs: retain exact full-restriction matching as
  first priority, then try each normalized observation-side heterodimer component. Carry the
  effective join key through ambiguous-candidate resolution and match-type provenance so a DQ8
  beta-chain expansion can match a full DQA1/DQB1 observation without rewriting the observation.
- Derive peptide-summary allele and serotype evidence from the full restriction plus all of its
  components. A target beta chain can therefore match a full pair exactly, and DQ8 can match the
  pair through its DQB1 component.
- Define `unknown_allele` only when the attributed sample provides neither exact-allele typing nor
  serotype typing. A known, nonmatching serotype is negative evidence for this query, not unknown.
- Replace each of the four MAPTAC samples whose deposited peptides merge `-DM` and `+DM` with two
  truthful experimental-arm samples. Their shared-allele observation join is intentionally
  ambiguous and therefore blanks arm/APM/control metadata via the existing consensus path.
- Treat all five defects as local: no upstream issue is warranted unless implementation exposes a
  dependency behavior that prevents component-aware matching rather than merely requiring it.

## Steps

- [x] Add focused regressions for cross-class summary leakage, DQ8 pair joining and summary
      support, nonmatching-serotype exclusion, and merged HLA-DM arm metadata.
- [x] Implement target-class and heterodimer-component matching with truthful provenance.
- [x] Split the four merged MAPTAC HLA-DM conditions into explicit `-DM` / `+DM` sample arms.
- [x] Run targeted tests and inspect the affected real-PMID outputs.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`.

### Second review: preserve precision through fallback paths

- [x] Reproduce noncanonical serotype parsing and multi-sample class-pool summary behavior.
- [x] Canonicalize every parsed serotype with its own `to_string()` representation before catalog
      lookup, for both whole fields and tokens inside mixed fields.
- [x] Represent class-pool exact alleles and reported serotypes separately; propagate both through
      the fallback without converting inferred serotype members into reported exact alleles.
- [x] Add focused unit and end-to-end regressions for spelling variants, allele joins, and
      `class_only_sample_serotype` summary provenance.
- [x] Re-run real-corpus checks, `./format.sh`, `./lint.sh`, `./test.sh`, and build smoke.
- [ ] Update the version/PR, wait for CI, merge, deploy from clean `main`, and verify PyPI.

## Review

- All five findings were local Hitlist defects; no upstream issue was warranted. The DQ8 catalog's
  beta-chain members are sufficient once Hitlist applies its own component-aware matching contract
  symmetrically to observation pairs.
- Peptide summaries now infer and push down the target class, retain a row-class backstop, filter
  mixed-sample genotypes to that class, derive serotypes from both chains of a pair, and reserve
  `unknown_allele` for samples with neither exact nor serotype typing.
- The observation join aliases a full class-II restriction only from a serotype member. A
  regression proves that two fully known pairs do not become a match merely because they share
  one chain.
- PMID 31495665 now has separate `dm-` and `dm+` samples for each of the four alleles whose peptide
  sets were merged during ingestion. Real merged observations resolve as `pmid_ambiguous` with
  blank `condition_category`, `apm_perturbed`, and `is_control_arm` rather than false controls.
- Real-corpus checks confirmed PMID 34433824's DQA1/DQB1 rows join to the DQ8 sample as
  `serotype_expansion`; its DQ8 peptide summary is nonempty; PMID 35051231 contributes no class-I
  row to a DRB1*11:01 query; and PMID 28467828 contributes no row to an unrelated DR4 query.
- Verification: 16 focused review regressions passed; `./format.sh` and `./lint.sh` passed;
  `./test.sh` passed 1,171 tests with one expected warning; `tests/test_build_smoke.py` passed 2/2.
- The second review's six focused cases now pass. Parsed serotypes use mhcgnomes' canonical
  representation for catalog lookup, while class pools serialize source-reported exact molecules
  and serotypes rather than expanded join candidates. The synthetic multi-sample regression now
  reports `class_only_sample_serotype`, never `class_only_sample_allele`.
- Final verification after both review rounds: the curation/export suite passed 362 tests;
  `./format.sh` and `./lint.sh` passed; `./test.sh` passed 1,176 tests with one expected warning;
  and `tests/test_build_smoke.py` passed 2/2.

---

# Issues #380, #381, and #374 — truthful sample-MHC attribution

## Goal

Fix the curated-sample MHC attribution defects in dependency order: one documented
sample-MHC candidate API (#380), correct per-sample genotypes for PMID 36423003 (#381),
and eliminate the declared-class/typed-allele contradictions (#374).

## Steps

- [x] Inspect the sample-join implementation, current YAML, corpus counts, and paper methods.
- [x] Implement and document the centralized sample-MHC attribution-candidate API.
- [x] Add a general audit for samples whose `mhc` pools several genotypes.
- [x] Curate PMID 36423003 and the remaining #374 samples from primary sources.
- [x] Add focused unit and invariant tests.
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`.
- [x] Bump the version and open a PR.

## Review

### What the verification changed

Three of the four premises in the issues were wrong, and checking first saved
implementing them:

- **#374 group 1 (HLA-G declared class I) was already fixed** — all three
  721.221-HLA-G transfectants declare `non-classical` today. No work needed.
- **#374 group 2 was real but mis-framed.** The eleven `I+II` samples were not
  contradictions: every study *did* profile both classes, and the class-II alleles
  were simply missing from the curation. The fix was to finish the typing from each
  paper's own table, not to weaken the declaration to `I`.
- **#381's allele table was incomplete and its acceptance criteria wrong.** The corpus
  holds 13 class-I BoLA alleles, not 6, and its class-II sample has three real DRB3
  genotypes. Curating all 13 onto one sample would pool eight animals and would also
  report a NetMHCpan prediction as an observation.

### The generalization

The #381 bug — an `mhc` field holding a union across samples rather than one
genotype — is a *class* of defect, not one entry. `qc.sample_ploidy_audit` detects it
without threshold tuning: a diploid donor carries at most two alleles per locus, so
three is proof of pooling. It found six samples; all six were wrong, and all six are
fixed here from primary sources. The audit now guards the corpus in CI.

Notably it also guards against doing #381 *wrong*: the pooled 13-allele curation the
issue asks for would fail it.

### Deliberately not done

- **Predicted-vs-observed restriction** (#415). `mhc_allele_provenance` has no value
  meaning "predicted", and 155 of PMID 36423003's rows resolve `exact` from a
  NetMHCpan <2%-rank assignment. This is not one study's problem — IEDB populates
  elution restrictions by inference routinely — so it needs a schema axis and a
  corpus-wide sweep, not a patch here.
- **THP-1 class-I typing conflict** (#416). Two primary sources disagree; the
  heterozygous DSMZ form is kept and the conflict filed rather than guessed.
- **BoLA-6*014:01 vs *014:02** (#414). IEDB and the paper disagree on one allele of
  one line. IEDB's value is curated so its rows still attribute, discrepancy recorded.

### Sample-count changes

| PMID | Before | After |
|---|---|---|
| 36423003 | 2 | 9 (8 per-line class-I + 1 locus-level class-II) |
| 32350084 | 2 | 26 (19 EBV-LCL + 7 K562) |
| 26768311 | 2 | 10 (5 allotypes x 2 conditions) |
| 31495665 class II | 2 | 14 (6 single arms + 4 alleles x 2 HLA-DM arms) |

---

# Comprehensive modality correctness — issues #382, #376, #396, #399

## Program goal

Make MHC identity and source-protein attribution explicit, correct, and auditable across MS
elution, binding assays, curated sample metadata, and peptide mappings. Ship the work in two
dependency-ordered PRs: the shared MHC identity contract first, then Ensembl IG/TR mapping
coverage and provenance.

## Phase 1 — contextual MHC identity and validation (#382, #376, #396)

### Design

- Introduce one cached, documented MHC annotation resolver that accepts the raw restriction,
  source-reported class, and optional curated species context. It returns the normalized
  restriction, resolved species and provenance, canonical class and provenance, plus explicit
  correction/conflict flags.
- Treat curated species as a parsing constraint when it can parse the designation; fall back to
  an explicit designation's unconstrained species for legitimate engineered-MHC systems. Cache
  keys include the species context. A compatible generic result such as `Bos sp.` may be refined
  to `Bos taurus`; an incompatible unconstrained guess is recorded as a context disagreement.
- Derive class only from actual molecules (`Allele`, `Gene`, `Pair`). Derive semicolon candidate
  sets component-wise when every resolved component agrees. Class-only, serotype-only, and
  unparseable restrictions retain the normalized source-reported class.
- Store `mhc_class_reported`, `mhc_class_source`, `mhc_class_corrected`,
  `mhc_species_source`, and `mhc_species_context_disagrees` on both MS and binding rows. Refresh
  these fields after donor-set promotion so the stored restriction and provenance cannot drift.
- Add a cross-modality MHC-token audit covering MS, binding, and curated sample MHC. Known source
  errors and parser gaps carry distinct statuses/reasons; any new unrecognized token fails the
  build. Expose the audit through the Python QC API, bare `hitlist qc`, and a dedicated CLI command.
- Version the observations artifact contract so existing parquets rebuild once instead of
  silently preserving the old schema and wrong classifications.
- Print build summaries for class corrections and incompatible contextual-species corrections.

### Verification

- [x] Unit-test contextual parsing, explicit-species fallback, class derivation/fallback,
      donor-set behavior, and correction flags.
- [x] Scanner-test the Bos contextual case, Caja/Mamu correction, class-only fallback, and
      post-promotion donor-set fields for both source classifications.
- [x] Unit-test known-invalid, parser-gap, sentinel, and unknown-token QC behavior across MS,
      binding, and curated sample inputs; add a real-corpus staleness/new-token guard.
- [x] Test artifact-version invalidation, schema columns, build summaries, CLI routing, and docs.
- [x] Bump the patch version; run targeted tests, format, lint, and the complete test suite.
- [x] Open a PR closing #382, #376, and #396; require all CI jobs, merge, deploy, and verify PyPI.

## Phase 2 — immunoglobulin/TCR mapping coverage (#399)

### Design

- Include Ensembl's coding IG/TR biotypes (`IG_V/D/J/C_gene`, `TR_V/D/J/C_gene`) alongside
  `protein_coding`; continue excluding pseudogenes and document the germline-only boundary.
- Carry source-gene biotype through `ProteomeIndex`, long-form peptide mappings, mapping schema,
  filters/exports, and artifact-version metadata so IG/TR attribution is distinguishable from a
  conventional protein-coding match.
- Keep `ProteomeIndex.from_ensembl(biotype="protein_coding")` as an explicit compatibility mode;
  make the new plural `gene_biotypes=` API and the mapping worker's task contract explicit.
- Test index construction and mapping with protein-coding, IG, TR, pseudogene, duplicate-sequence,
  cache round-trip, process-worker, and legacy-artifact cases. Quantify recovered current-corpus
  mappings before release.

### Verification

- [x] Implement and verify the expanded Ensembl index contract and mapping provenance.
- [x] Bump the patch version; run all required gates and corpus coverage comparisons.
- [ ] Open a PR closing #399; require all CI jobs, merge, deploy, and verify PyPI.

## Review section

- `resolve_mhc_annotation()` now owns normalization, contextual species resolution,
  molecule/donor-set class derivation, source fallback, and persisted provenance. Scanner and
  supplementary ingestion both use it before filtering and refresh it after set promotion.
- The registered 4.4M-row corpus has exactly five reviewed exceptional tokens: four
  `invalid_source` values (`HLA-B23`, `HLA-DR7A`, `HLA-DR3A`, `HLA-DR1B`) and one parser gap
  (`HLA-Cw16`). The audit finds no unrecognized token; its integration test pins both growth and
  stale allowlist entries.
- Observations artifact contract v1 forces a one-time rebuild for the new schema. Build output
  reports class corrections, species-context conflicts, and token-audit totals before writing.
- Verification: 338 affected non-integration tests passed; the dedicated corpus audit passed;
  the supplementary suite also passes under Python 3.9; format and lint passed; full
  `./test.sh --all -rs` passed 1,170 tests with zero skips and one expected warning.
- Phase 1 shipped in PR #412 as v1.55.7; every CI job passed and the wheel and sdist were
  verified on PyPI.
- Phase 2 centralizes the translated Ensembl policy as conventional `protein_coding` plus the
  eight coding IG/TR gene biotypes. Both gene and transcript records must satisfy the policy;
  pseudogenes remain excluded. `gene_biotype` now survives index metadata, worker normalization,
  sidecar filtering, and exploded training exports. Mapping artifact v2 forces a clean rebuild.
- The Ensembl 112 audit finds 420 translated IG/TR proteins. Against the current registered human
  corpus they produce 15,808 long-form mappings and recover 4,451 unique peptides with no prior
  human-proteome match (MS: 1,745 class I and 2,836 class II unique peptides).
- Phase 2 verification: targeted proteome/mapping/export tests passed 220 tests; format and lint
  passed; `./test.sh --all -rs` passed 1,175 tests with zero skips and one expected warning.

---

# Issue #410 — deterministic Alpizar resolver regression

## Goal

Remove the last full-suite skip without weakening the regression. The test must exercise the
public observations-export path against a small, version-controlled Alpizar-shaped fixture rather
than depending on whichever IEDB snapshot happens to be registered on the developer machine.

## Diagnosis and design

- PMID 27920218 is present in the current build (8,144 rows). The stale test selected zero rows
  because IEDB replaced the old 515 literal `HLA class I` restrictions with explicit
  semicolon-separated candidate-allele sets.
- Keep the biological behavior under test: ambiguous C1R rows must route to B*40:02, B*39:01, or
  the pooled sample from their antigen-processing text. Exercise both the historical class-only
  representation and the current allele-set representation.
- Use a temporary observations parquet plus a minimal synthetic PMID override and call
  `generate_observations_table()`. This covers the real class-pool orchestration and candidate
  scorer while remaining independent of the installed corpus.
- Correct the newly exposed provenance error: a class-pool candidate selected from row-level
  discriminator text must report `sample_attribution=discriminated`; `sample_match_type` remains
  `pmid_class_pool` because the restriction itself was not an exact allele match.
- Update the shipped Alpizar curation note to document the IEDB representation change.

## Steps

- [x] Replace the conditional full-corpus Alpizar test with the deterministic public-API fixture.
- [x] Correct and test class-pool discriminator provenance.
- [x] Update the Alpizar curation note and bump the patch version.
- [x] Run targeted tests, `./format.sh`, `./lint.sh`, and `./test.sh --all -rs`.
- [x] Review the diff, open a PR closing #410, merge, deploy, and verify PyPI.

## Review section

- The paper was never absent: the current corpus has 8,144 Alpizar rows. IEDB changed the 515
  ambiguous restrictions from `HLA class I` to candidate-allele sets, making the old filter stale.
- The replacement writes four small observation rows to a temporary parquet and exercises
  `generate_observations_table()` with an isolated Alpizar-shaped override. It covers the old
  class-only form, both current single-transfectant sets, and the current pooled set.
- Class-pool scoring now records `sample_attribution=discriminated` while correctly retaining
  `sample_match_type=pmid_class_pool`; the latter describes restriction-level evidence, whereas
  the former describes the sample-selection mechanism.
- Targeted export tests pass (117 passed, 17 integration tests deselected). Format and lint pass.
  The complete corpus suite passes 1,154 tests with zero skips and one expected warning.

---

# Issue #406 follow-up — isolate direct prefetch-worker tests

## Goal

Prevent direct unit calls to the child-only prefetch entry point from leaking its data-directory
override into later xdist tests. The full integration suite should retain only genuine
corpus-dependent skips.

## Steps

- [x] Scope `_prefetch_worker` test doubles and data-directory mutation to a monkeypatch context.
- [x] Add a regression assertion that the parent test process state is restored.
- [x] Bump to 1.55.5; run format, lint, targeted mixed-order tests, and `./test.sh --all -rs`.
- [x] Ship a follow-up PR, merge, deploy, and verify PyPI.

## Review section

- Direct `_prefetch_worker` tests now emulate the disposable child-process boundary with a nested
  monkeypatch context and assert that `_override_data_dir` is restored after each call.
- The mixed-order regression (`test_mappings.py` followed by `test_observations.py` in one worker)
  passes all 86 tests; the full suite passes 1,153 with only one legitimate corpus-dependent skip
  (`Alpizar 2017 not present in this build`). Format and lint pass. Version bumped to 1.55.5.

---

# Issues #402, #404, #405 — bounded/offline-safe mapping builds and artifact contract

## Goal

Make peptide-mapping builds terminate predictably, obey the documented no-fetch policy, and
rebuild sidecars whenever the code or parameters that define their contents change. Remove the
new timeout environment variables: safety deadlines are internal invariants, while legitimate
caller choices remain explicit function/CLI arguments.

## Design

- Replace the parent loop's before-call stopwatch with a supervised, killable child process.
  The parent submits one canonical at a time to a single-child process pool, records the in-flight
  name before dispatch, and waits only until one fixed absolute phase deadline. If the child does
  not answer, terminate it, report the named canonical, and skip it plus the unattempted tail.
- Return explicit prefetch outcomes. Workers may only receive UniProt/Ensembl tasks whose required
  local cache warm-up succeeded; a failed/timed-out fetch is not retried silently in a worker.
  This keeps the existing "failure is tolerated" contract without moving the same hang elsewhere.
- Replace `_per_canonical_mapping_worker`'s positional tuple protocol with a documented, picklable
  `MappingTask` value object. Keep index construction, peptide mapping, output normalization, and
  coverage accounting in one worker entry point so unit and real process-pool tests exercise the
  same API across cache/network policies and peptide lengths.
- Remove `HITLIST_PREFETCH_BUDGET`, `HITLIST_DOWNLOAD_TIMEOUT`, and their float parsers. Keep the
  socket timeout and prefetch deadline as finite positive internal constants. Tests may pass an
  internal deadline argument directly; users do not configure safety correctness through process
  state.
- Honor `fetch_missing=False` (#405): reuse existing cached artifacts, but do not launch the
  prefetch worker or any network download for an uncached proteome. Log and record skipped tasks.
- Add a monotonic peptide-mapping artifact version plus behavior-defining parameters (Ensembl
  release, UniProt search policy, flank width, seed length, and output schema) to the metadata.
  Legacy or mismatched metadata is stale.
- On an observations-cache hit with `build_mappings=True`, invoke the mappings builder so it can
  validate/rebuild only the sidecar. Do not rescan observations, bulk proteomics, or expression.
- Keep the pre-call progress line from #403 and make deadline/failure messages unconditional when
  they explain omitted output.

## Implementation and verification

- [x] Add failing regression tests for an actually blocked in-flight prefetch, explicit failed and
      unattempted outcomes, no worker retry, and `fetch_missing=False` network isolation.
- [x] Introduce and document `MappingTask`; expand direct and process-pool worker contract tests.
- [x] Implement supervised prefetch and remove the timeout environment-variable APIs/tests.
- [x] Add mapping artifact contract metadata and cache-validation tests, including legacy metadata
      and each behavior-defining parameter.
- [x] Add a builder early-return regression proving stale/missing mappings rebuild independently.
- [x] Identify all four default-suite skips; remove any state-dependent skip that masks a unit-test
      branch, or document why the integration/dependency skip is intentional.
- [x] Bump the patch version and update user-facing documentation/comments.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`; inspect the diff and test behavior.
- [x] Isolate the unrelated default-suite cache/multiprocessing flake found during final
      high-concurrency verification (#406), then rerun all required gates.
- [x] Push a PR linking #402, #404, and #405; check every CI job.
- [x] Merge, update clean `main`, run `./deploy.sh`, and verify the released version on PyPI.

## Review section

- Replaced the pre-call-only stopwatch with a single-child supervisor that names every in-flight
  request, enforces one absolute 900-second warm-up deadline, terminates a blocked call, and marks
  the current/unattempted proteomes unavailable. Mapping workers are structurally cache-only, so
  the same network operation cannot escape the deadline as an on-demand retry.
- Removed `HITLIST_PREFETCH_BUDGET` and `HITLIST_DOWNLOAD_TIMEOUT`. The finite socket timeout and
  warm-up deadline are internal safety constants; invalid/exhausted internal test deadlines fail
  closed. `fetch_missing=False` now reaches resolution, primary UPID fetches, and PMID overrides.
- `_per_canonical_mapping_worker` now accepts a documented `MappingTask` and returns a named
  `MappingResult`. One implementation builds one seed index, maps every peptide length, produces
  one normalized frame, distinguishes unavailable from zero matches, and preserves the full
  coverage denominator. Direct, pickle, corrupt-cache, and real process-pool cases cover the API.
- Mapping metadata now carries artifact version, Ensembl release, UniProt/fetch policy, flank,
  seed length, and schema. Legacy/mismatched/incomplete artifacts rebuild; observations cache hits
  still validate the independently cached mappings sidecar.
- The four default skips were two tests conditional on a developer's local observations cache and
  two tests conditional on the optional, non-PyPI `cancerdata` package. The first pair now uses an
  isolated empty data directory; the second injects a fake provider and separately tests the
  actionable missing-provider error. Latest default run: 1,130 passed, 0 skipped.
- `./format.sh`, `./lint.sh`, and `./test.sh -rs` pass. Version bumped from 1.55.3 to 1.55.4.
- Final high-concurrency review exposed #406: bulk/proteome tests depended on real user cache state,
  and one multiprocessing regression required a sandbox-forbidden Manager socket. The PR now
  isolates those caches per test and uses spawn-safe result files instead of a Manager service.

---

# Issue #46 — multi-axis species model (PR 1: schema + filters)

Scope (user-approved): **Schema + filters**, detection via **genus-aware heuristic + audit**.
Defer: effector_organism, mhc_donor_individual, build-time axis validation, per-PMID override curation.

## Design
New per-row columns derived purely from existing `host`, `source_organism`, `mhc_species`:
- `host_organism`  — normalize_species(host)            (clean binomial)
- `source_species` — normalize_species(source_organism) (clean binomial; disambiguates from data-`source`)
- `engineered_mhc` — bool: genus(mhc_species) != genus(host_organism), both animal genera
- `xenograft`      — bool: genus(source_species) != genus(host_organism), both animal genera
- `chimeric`       — bool: engineered_mhc | xenograft

Heuristic detail (kills false positives found in audit):
- Compare at **genus** level so `Sus scrofa`≈`Sus sp.`, `Mus musculus C57BL/6`≈`Mus musculus` don't flag.
- "Animal genera" = genera that appear in `mhc_species` (MHC-bearing => animal). This excludes
  virus/bacteria sources (SARS, Mtb, vaccinia → not xenograft) AND immunization models
  (chicken-OVA-in-mouse: Gallus not MHC-bearing in corpus → not xenograft). Principled, self-adapting.
- Unparseable host strings (e.g. "B6.ERAAP null") → genus not in set → conservatively non-chimeric.

## Steps
- [ ] curation.py: add `species_genus(s)` helper (genus token of normalize_species; "" if empty)
- [ ] curation.py: `compute_species_axes(df, animal_genera=None)` — adds the 5 columns; idempotent
      (skips columns already present); derives animal_genera from df.mhc_species if None.
- [ ] builder.py: call compute_species_axes(obs)/(binding) before _atomic_write_parquet.
- [ ] observations.py: add `source_species=`, `host_species=`, `exclude_chimeric=` to
      load_observations / load_ms_observations / load_binding / _load_peptide_index.
      Compute axes post-load when columns absent (old parquet) so filters work without a rebuild.
- [ ] qc.py: `audit_species_axes()` — counts chimeric/engineered/xenograft + lists suspicious rows.
- [ ] tests: unit-test species_genus + compute_species_axes; integration-test the three filters via a
      small temp parquet. Cover FP cases (Sus scrofa/Sus sp., pathogen-source, strain suffix) and
      TP cases (HLA-tg rat, dog-tumor-in-mouse).
- [ ] version bump; format/lint/test; PR.

## Review section

Discovered the heuristic was already half-built: `is_chimeric_system` + `is_engineered_mhc`
existed in curation.py (tested), and export.py already materializes `is_chimeric` /
`is_engineered_mhc` in the observations EXPORT. Gap filled by this PR:

- curation.py: added `is_xenograft(source, host, mhc)` — host-axis counterpart of
  is_engineered_mhc. 3-arg (needs mhc) to avoid flagging heterologous-antigen studies
  (foreign protein on native host cells, host genus == mhc genus) as xenografts.
- observations.py: `_attach_species_axes()` derives host_organism / source_species /
  is_chimeric / is_engineered_mhc / xenograft at LOAD time (no rebuild needed; same pattern
  as is_non_peptide_ligand). Added `source_species=` / `host_species=` / `exclude_chimeric=`
  to load_observations / load_ms_observations / load_binding / load_all_evidence + the shared
  _load_peptide_index. Registered the 5 derived columns in _DERIVED_COLUMN_DEPS.
- qc.py: `species_axis_audit()` — groups chimeric rows by (host, source, mhc) triple with
  severity (info = clean engineered/xeno; review = chimeric w/ populated host, neither).
- tests: is_xenograft unit tests; loader filter/column integration tests (FP cases: pathogen
  source, substrain, heterologous antigen; TP: HLA-tg rat, dog-tumor-in-mouse); qc audit tests.

Real-corpus numbers (load-time derived, no rebuild): is_chimeric 1.64%, engineered 1.58%,
xenograft 0.32%; exclude_chimeric drops 72,813 / 4.44M rows. host-human (4.12M) ≠ source-human
(3.49M) — the conflation #46 set out to fix is now expressible.

Deferred (noted in PR): effector_organism, mhc_donor_individual, build-time materialization +
axis validation, per-PMID chimeric override curation, adding `xenograft` to the export schema.
897 passed, 2 skipped. lint/format clean.

## v1.49.x — curated sample metadata self-consistency (#372/#374/#375/#379)

Shipped: per-sample `species` honored (2 mouse samples were exporting as human); HLA-G
transfectants moved to non-classical; 3 unparseable `mhc` tokens fixed; species-inference traps
pinned to explicit forms; class filters normalized at both boundaries so `non-classical` is
reachable end-to-end (was 18 samples / 0 observations); zero-match filters return an empty frame
instead of raising KeyError; `_mhc_class_matches` unified with `_sample_class_tokens`.

Review round: fixed a real bug in `species_compatible` (compared raw strings before resolving, so
`"Gallus gallus (chicken)"` vs `"Gallus gallus"` was False), removed a dead `try/except ImportError`
on a hard dependency, derived the parquet spelling set from the alias table, cached
`normalize_mhc_class_token`, split the typo guard from the allele-join guard, added staleness
assertions to both allow-lists, and updated the CLI help + curation doc for the non-classical
vocabulary.

Deferred (filed): #380 serotype/locus `mhc` values never reach the allele join; #381 PMID 36423003
has real BoLA alleles in IEDB but is curated class-only; #382 species inference is pinned only in
curated YAML, the ingest path still misclassifies; #374 remainder (11 `I+II` samples need their
class-II genotypes read out of the papers).

## v1.51.0 — adopt mhcgnomes' species API (#383)

Shipped: floored `mhcgnomes>=3.39.0` (CI installs latest, so an unpinned floor is what let a green
local run ship a red CI); deleted `curation.species_compatible` in favour of
`Species.compatible_with`; replaced the trap-pinning tests with the real invariant — no curated
`mhc` token may resolve with `species_source == "inferred"`. That guard found 4 chicken `BF2*`
tokens resolving by cross-species inference (PMIDs 18612635, 36695776), now pinned with `Gaga-`;
inferred tokens 4 → 0. Patr-AL is `Ib` upstream so its allow-list entry is gone (contradictions
12 → 11).

Review finding #4: the source-vs-MHC species invariant reached only the test suite. The samples
table now exports `mhc_species` and `species_axes_agreement`, and the guard test asserts on the
column rather than re-deriving it — they were briefly two implementations and disagreed on 19
serotype/locus rows. `_SAMPLE_PROVENANCE_COLUMNS` extended so the `--with-expression-anchors`
variant carries them too. Corpus: 651 agree, 35 undeterminable (12 of them `mhc: unknown`), 2
disagree — both engineered chimeras (#46), correct as curated and now visible.

Next: #380 (serotype/locus values never reach the allele join), #381 (PMID 36423003 has real BoLA
alleles in IEDB but is curated class-only), #382 (species inference pinned only in curated YAML;
the ingest path still misclassifies), #374 remainder (11 `I+II` samples need class-II genotypes
read out of the papers).

---

## Serotype provenance and the unreachable half of the serotype map (#455, #458)

Reported from tsarina, which moved its `--serotype` filter onto this column and
found that two different facts share one name, and that six specificities never
appear at all.

### The two problems

1. **#455 — 41,478 rows carry no serotype that should.**
   `mhcgnomes.data.serotypes["HLA"]` spells alleles two ways: 915 entries use
   the compact `C*0304` form, 11 use `C*15:02`. The 11 are the hand-curated
   rows its generator cannot reproduce (mhcgnomes#156). `_build_allele_to_serotypes_map`
   keyed by whatever the table held while `allele_to_all_serotypes` looked up a
   compact key it built itself, so those rows were unreachable and Cw12, Cw14,
   Cw15, Cw16, Cw17 and Cw18 were absent from every annotation. Worst of these
   is Cw16: curated in deliberately from WHO's `hla_nom.txt` (mhcgnomes#153),
   and discarded here by a key format.

2. **#458 — `serotypes` mixes primary data with a projection.**
   35,257 human rows are serologically typed studies where the serotype *is*
   the observation and no molecule was measured. 1,630,309 rows name a molecule
   and get a serotype computed from it. Both spell the result identically. The
   marker was `allele_resolution == "serological"`, which is an inference the
   consumer has to know to make, from a column named for resolution.

### Plan

- [x] Normalize both sides of the reverse map through one key helper, and
      assert every table entry stays reachable.
- [x] Delete `_build_allele_to_serotype_map`, unused since `allele_to_serotype`
      started delegating to the plural form.
- [x] Add `serotype_source` (`reported` / `derived` / `donor_set` / empty) to
      `MhcAnnotation`, so the distinction is a column rather than a deduction.
- [x] Record `mhcgnomes_version` in `observations_meta.json`: the derived
      columns are only as current as the library that computed them.
- [x] Bump the observations artifact version so every existing index rebuilds
      its derived columns.
- [ ] Thread `serotype_source` through the loader, exports and CLI the way
      `restriction_evidence` (#415) is threaded.
- [ ] Document both axes in the README.
- [ ] Tests: reachability of the whole table, the three source values, the
      artifact-version invalidation, and the new filter.
- [ ] Rebuild the local index and confirm the derived columns regenerate.
- [ ] Version bump, three gates, PR, deploy.

### Deliberately not in this PR

`#456` (retired designations such as `B*44:01` never resolving to their current
name, because `parse()` runs with `use_allele_aliases=False`) is a semantic
decision about whether `mhc_restriction` may stop being what the paper
reported. It stays open for a call rather than being bundled here.

# PR #462 review fixes and release

Specification: [pr-462-release-spec.md](pr-462-release-spec.md).

- [x] Confirm current PR, review findings, dependency data, and release scripts.
- [x] Preserve species in both serotype lookup directions; add regressions.
- [x] Correct worker sizing and interpreter selection; make continuation tests deterministic.
- [x] Measure corpus impact and validate the declared dependency floor.
- [ ] Run format, lint, and all tests; record review results.
- [ ] Update PR description, push, verify CI, and merge.
- [ ] Deploy from clean main and verify PyPI publication.
- [ ] Review the next relevant issue group.

## Review

Species-aware annotation, inverse lookup, and the shared public query normalizer
now agree across the catalog (#463, #449). The comparison in
`pr-462-serotype-impact.json` changes 27 observation and 464 binding rows;
human annotations are unchanged. Artifact version 4 forces regeneration on the
next build. The dependency floor remains mhcgnomes 3.54.0.

The test runner uses free/speculative memory, a single worker when probes fail,
and Python's own pytest module. The continuation test no longer depends on
process startup fitting into five seconds (#440); actual process termination
remains covered.

Format/lint pass with the locked Ruff version. The isolated environment passes
68 targeted query/runner checks and all 252 curation tests at the mhcgnomes floor.
Full local validation and CI are in progress before merge and deployment.

---

## #444 — honor `exclude_from_ms` (1.62.3)

- [x] Give the flag a reader: `curation.ms_excluded_pmids()`, registered in
      `_clear_curation_caches` so a rebuild sees YAML edits.
- [x] Drop the rows in `builder._drop_excluded_from_ms`, called on `obs` only.
- [x] Bump `_OBSERVATIONS_ARTIFACT_VERSION` 4 → 5 so stale caches rebuild.
- [x] Regression suite in `tests/test_exclude_from_ms.py`.
- [x] Re-describe the key in `PMID_ENTRY_FIELDS` and `docs/pmid-curation.md`.
- [ ] Merge, then republish the CI corpus so the integration assertion goes live.

### Review

The flag was documented, curated on 11 studies, and read by nothing. It now has
exactly one reader, and the drop runs next to `_drop_short_mhc2_rows` — after the
supplementary merge, so it covers IEDB, CEDAR and supplement in one place.

Measured by running the shipped helper over the real 4.4M-row corpus:

| | before | after |
|---|---:|---:|
| MS observation rows | 4,480,783 | 4,440,428 |
| MS peptides | 1,341,351 | 1,310,337 |

40,355 rows go (0.90% of the corpus), from 6 of the 11 studies; the other 5
contribute only binding rows. 31,014 peptides leave the corpus entirely — they
were never MS-observed, only yeast-displayed, microarrayed or predicted. The
remaining 2,087 are also seen in genuine elution studies and stay.

What the dropped rows actually were: 12,490 `purified MHC`, 12,154
`High throughput multiplexed assay`, 6,285 and 5,608 fluorescence-based
`purified MHC` variants, 2 `x-ray crystallography`. All 40,355 carried
`is_binding_assay = False`, which is why IEDB's own flag never caught them and
the MS/binding fork put them on the MS side. Chen 2019 is the sharp case: 3,816
of its rows are labelled `cellular MHC/mass spectrometry` by IEDB, but the paper
is the MARIA prediction tool. Only a curator reading the paper catches that, and
one did.

**Scoped to MS evidence.** `binding.parquet` keeps all 472,497 rows from these
studies, including Wendorff 2020's 418,890 microarray measurements. `binding` is
the complementary mask of the same scan and is never derived from `obs`, so the
filter cannot reach it by construction; a test pins it anyway.

The integration assertion at `generate_observations_table()` skips against a
corpus built before this change, because a build-time filter is a property of the
artifact and asserting it against the old artifact tests nothing. The CI corpus
(`ci-corpus-v1`) predates it, so that test skips in CI until the corpus is
republished; the unit tier covers the drop logic unconditionally.

**Local suite is not a clean signal right now.** 215 tests fail identically on
this branch and on clean main at 6ed972f: the shared virtualenv has mhcgnomes
3.33.4 in site-packages, below hitlist's own `>=3.54.0` floor, shadowing the
3.64.2 sibling checkout. `Species.compatible_with` does not exist at 3.33.4. The
failure sets diff clean, and this branch adds 6 passing tests. Filed separately
as #467.

Format/lint pass with the locked Ruff version — `env -u VIRTUAL_ENV uv run
./format.sh` / `./lint.sh` against the lockfile's mhcgnomes 3.64.2, unaffected
by the shared venv issue above (#467). CI's four Python legs are the clean
signal for the full suite; all passed on 8886b37.

## #483: split `test.sh --all` into two pytest processes (1.62.11)

`deploy.sh`'s combined `--all` pass had OOM-killed the machine three times in
one day, even at a single xdist worker. `test.sh` now runs non-integration and
integration tests as two separate `python -m pytest` invocations (CI has done
this since #272/#274) instead of one process carrying both. Each pass
re-probes available memory independently and gets its own per-worker budget —
`PER_WORKER_GB` (2.5, unchanged) for the light pass, a new
`INTEGRATION_PER_WORKER_GB` (5) for the integration pass.

**Real end-to-end run, not just `--collect-only`.** Ran the new script for
real: light pass 1620 passed (13m55s), integration pass 42 passed / 1 failed
(8m42s). No OOM kill, despite the machine being genuinely memory-starved by
two unrelated repos' test suites (`pirlygenes`, `vaxrank`) running the whole
time — system-wide available memory sat under 0.2GB for 56 of the sampled
10s ticks during the run. That contention is real evidence the split
survives non-ideal conditions, but it also means this isn't a clean
before/after: I don't have a same-methodology peak-RSS number for the old
single-process `--all` to compare against, only the earlier session's
differently-measured 4.6-8.7GB range from the OOM-killed attempts (killed
before reaching whatever their true peak would have been, so not a ceiling
either).

**Process RSS trace (own descendant PIDs, summed):** light pass peaked at
~3.6GB transiently, mostly running ~200-350MB; integration pass reset to
~80MB at the fresh-process boundary, then climbed to ~11.6GB peak while
building/holding the observation corpus fixture. Take the integration peak
as an upper bound, not a clean physical-memory number — summed RSS across a
coordinator + one xdist worker double-counts any pages both processes have
resident from the same mmapped fixture file (`tests/xdist_cache.py`, #262).
The one clean, methodology-independent result: it finished both passes
without being killed.

**Unrelated integration-test failure found, not fixed here**: the fresh
run's integration pass failed
`test_current_corpus_mhc_token_allowlist_is_complete_and_not_stale` — the
test's hardcoded allowlist still expects `HLA-DR3A`/`HLA-DR7A`/`HLA-DR1B` as
`invalid_source` exceptions, but the currently-registered corpus no longer
has them. Reproduces cleanly on `origin/main` HEAD with no local diff to
either file, so it's pre-existing data/test drift, unrelated to this PR.
Filed as #484 rather than fixed in-flight, since diagnosing which side (test
allowlist vs. corpus) is stale is its own task.

`tests/test_test_script.py` previously only asserted the old single-pass
`--all` behavior (and never covered the plain, non-`--all` path at all).
Rewrote it to assert two independent invocations for `--all` (own marker,
own worker count from its own per-worker budget, right cov flags each), plus
a new test for the previously-uncovered default path, plus a test that
extra CLI args reach both passes. Confirmed both new `--all` tests fail
against the pre-#483 script (only one invocation ever appears) and the
default-path test still passes against it, so the rewrite isn't vacuous in
either direction.

## #483 (continued): preflight memory guard, so a low-memory run aborts instead of getting killed later (1.62.13)

The two-pass split (1.62.11/1.62.12) mitigated one failure mode but left the
issue's other two asks open: no preflight guard at the low end, and no clear
failure signal when the OS kills the process. This PR adds the preflight
guard; the retry-once-after-delay idea from the issue is still not done.

`worker_count()`'s `mem_cap` used to hard-floor at 1 in the `awk` step
itself, so a critically-low reading could never actually reach 0 by the time
`workers < TEST_SH_MIN` ran — the "floor to TEST_SH_MIN and proceed anyway"
behavior the issue complained about was baked in twice over, not just once.

Removed that inner floor so `mem_cap` can be 0. After the full pipeline runs
(CPU cap, `TEST_SH_MIN` floor, `TEST_SH_MAX` ceiling), compare the *final*
worker count against `mem_cap` -- not `TEST_SH_MIN` directly, which the first
draft of this fix got wrong: a case with `TEST_SH_MIN=10, TEST_SH_MAX=2` and
enough memory for exactly 3-4 workers used to abort under that draft (`3 <
10`), even though the final, TEST_SH_MAX-clamped count of 2 was perfectly
safe. Comparing against the post-clamp `workers` instead fixes that -- it
only aborts when the number of workers that would *actually run* needs more
memory than is available, after every floor and ceiling has already applied.

Caught this by hand-deriving expected worker counts (page counts × page size
÷ per-worker GB, with `int()` truncation) for each existing parametrized
case in `tests/test_test_script.py` rather than trusting the numbers already
there -- one of the four existing cases (`10_000` speculative pages) turned
out to only work under the *old*, incorrect version of this guard, and
silently masked a real design bug once actually verified against a hand
computation instead of copied from a neighboring case.

`available_bytes` probe failing (the existing macOS/Linux fallback) still
bypasses the guard entirely and proceeds at `mem_cap=1` -- no evidence of
scarcity, nothing to abort on.

Verified: all new/changed cases in `tests/test_test_script.py` fail against
the pre-guard script and pass against the new one. Real (non-stub) dry run
against this machine's actual `vm_stat`/`sysctl` output at 6GB available
proceeds normally with 2 workers, confirming no false-positive abort under
ordinary conditions.

Still open on #483: `deploy.sh` retrying the test step once after a short
delay before giving up, since the underlying pressure (other apps, not
hitlist) can resolve on its own.

## #488: pmhc --predictor mhcflurry crash + best-allele narrowing not restricted to the queried genotype (1.62.14)

Found running a real user query (4 cancer-testis-antigen genes x a
6-allele patient typing, `--predictor mhcflurry`) -- two real bugs in the
same code path, both fixed here since both block the same real command.

**Crash on >6 unique (peptide, allele) pairs.** `_predict_mhcflurry` called
`Class1PresentationPredictor.predict(alleles=[[a] for a in ...])` -- a
list-of-single-element-lists. That's neither of the two shapes mhcflurry's
`alleles` argument actually accepts (a flat list of <=6 allele strings as
one shared genotype for every peptide, or a dict of sample_name -> alleles
paired with `sample_names`). Falls into the flat-list branch, and the
*number of scored rows* gets checked against the 6-allele-genotype limit --
crashes on essentially any real query. Fixed by keying the dict by allele
(one-allele "sample" per unique allele) and using each row's own allele as
its `sample_names` entry. Verified manually against the real mhcflurry
package with 8 distinct alleles before writing the regression test.

**One atypical-length peptide crashed the whole batch.** A real 16-mer
MS hit made `predict()` raise (`Class1PresentationPredictor`'s affinity
model only supports lengths 5-15) and abort scoring the other 32 peptides
in the same query. `_score_and_narrow_to_best_allele`'s own docstring
already promised "peptide-length mismatch" degrades to NaN gracefully --
`_predict_mhcflurry` just didn't implement that promise. Fixed by filtering
to the predictor's `supported_peptide_lengths` before calling `predict()`
and leaving out-of-range rows NaN.

**Best-allele narrowing wasn't restricted to the queried allele set.**
Separately, real output showed `best_predicted_allele` values (e.g.
HLA-B*07:02, HLA-B*08:01) that weren't among the 6 alleles the query asked
about at all. `_score_and_narrow_to_best_allele` scores a row's *entire*
recorded ambiguity set (every allele any matched study's genotype carried)
and picks the global best -- `query()`'s own `alleles` filter (the specific
individual's genotype) was never threaded into the narrowing step, only
into the initial corpus-level pushdown. Added an `allowed_alleles`
parameter: when the caller supplied a specific allele filter (`--mhc-allele`
or `--sample`), restrict each row's candidate set to the intersection with
it before scoring, falling back to the full set only if that intersection
is empty. Re-ran the original real query after the fix: every
`best_predicted_allele` in the output is now one of the 6 queried alleles.

Filed #488 with the root-cause writeup; both fixes + narrowing landed here.
Also filed #491 (separate, not fixed here): `--mhc-allele` should split a
single space/comma-joined token and gain a `--mhc-alleles` alias -- the
same real query first failed with 0 rows because all six alleles were
passed as one quoted shell argument.

## #493: surface whether a peptide also occurs in an un-queried gene (1.62.15)

User asked whether `pmhc` checks for a candidate peptide (from a queried
CTA gene panel) also occurring in some other, un-queried protein -- a real
TCR-T target-selection safety question, since a peptide shared with a
normal-tissue gene is a very different risk than one unique to the tumor
antigen.

Turned out the underlying data already existed and was already being
loaded on every `pmhc` query: `load_observations`'s auto-attach mechanism
(#238) joins `gene_names`/`gene_ids` from `peptide_mappings.parquet`
whenever the caller requests them, which `query()` always does (needed to
split multi-gene rows, one row per gene). But right after exploding into
one row per (peptide, gene), the "final precise gene filter" drops every
row whose gene isn't in the caller's query -- silently discarding the fact
that the SAME peptide also matched a gene the caller didn't ask about. The
code's own pre-existing comment already named this exact scenario ("e.g. a
KRAS-attributed peptide that also matches NRAS") without ever surfacing
it.

Added `other_genes` (semicolon-joined genes besides the ones queried whose
protein also contains this peptide, empty if unique to the query) and
`n_source_genes` (total distinct genes, including queried ones) by
capturing each peptide's full multi-gene mapping before the precise filter
runs. Deliberately doesn't try to classify "harmless same-family paralog"
vs. "unrelated gene" -- checked a real query's output by hand and every
flagged gene turned out to be a CT-antigen-family paralog (CTAG1A/CTAG1B
alongside CTAG2, the SSX family alongside SSX1/SSX2, XAGE1A alongside
XAGE1B), not a genuine off-target, but that judgment call belongs to the
reader, not something to hardcode against a possibly-stale gene-set
snapshot.

**A real correctness gotcha caught by testing, not review**: the
per-donor-row consolidation `_score_and_narrow_to_best_allele` does after
predictor narrowing (`_collapse_rows_sharing_narrowed_allele`) uses an
explicit `agg_spec` for its `.groupby().agg()` call -- any column not
named there is silently dropped, the same trap `mhc_species` and the
`_line_ids`/`_donor_ids`/`_donor_type_ids` columns already had (per their
own comments). Verified by writing the fix, then deliberately removing
only the `agg_spec` addition and confirming exactly one test failed (the
consolidation-path test) while the two `query()`-level tests still passed
-- proof the fix was targeting the right, specific bug rather than a vague
"add it everywhere and hope."

Also: caught mid-implementation that I'd been editing these files directly
in the shared main checkout instead of a worktree -- broke the project's
own golden rule #1. Recovered via `git stash` in main + `git worktree add`
+ `git stash pop` in the new worktree, without touching main's history;
nothing had been committed yet so nothing was actually at risk, but noting
it since it's exactly the mistake [[feedback_worktree_when_concurrent]]
warns about.

## #491: pmhc --mhc-allele accepts a pasted genotype string; --mhc-alleles alias (1.62.16)

`--mhc-allele "A*03:01 A*02:01 B*14:02 B*44:02 C*08:02 C*05:01"` -- six
alleles quoted as one shell argument, which is how anyone pasting a typing
report's genotype line would naturally write it -- silently matched nothing
and returned zero rows. `action="extend", nargs="+"` only separates what the
shell already split into distinct argv tokens; a single quoted string
arrives as ONE token and got matched literally against `mhc_restriction`.
No error, just a plausible-looking empty result.

Added `_split_allele_tokens`, applied where `_pmhc` reads the parsed value:
each raw token is split on commas and whitespace and flattened, so quoted,
comma-joined, and repeated-flag forms all work and compose. Added
`--mhc-alleles` as a plural alias (argparse keeps `dest="mhc_allele"` from
the first option string, same as the existing `--protein`/`--gene` pair on
this subparser -- verified rather than assumed).

Deliberately scoped to `pmhc`. The `data`/`report` subcommand's own
`--mhc-allele` help text already claims "Space-separated, comma-separated,
or repeated" without implementing it either, but that's a separate
pre-existing mismatch on a different code path and isn't what #491 asked
for.

Tests cover the helper directly (pass-through, space-joined, comma-joined,
mixed, empty) plus two CLI-level tests that drive real argparse through
`main()` with a stubbed `pmhc_query.query`, so the alias's dest resolution
and the plumbing in `_pmhc` are both exercised, not just the pure function.
All seven fail against the pre-fix CLI (the alias ones with argparse's own
"unrecognized arguments" error).
## #483 (final): deploy.sh retries the test gate once after a delay (1.62.17)

Last of #483's three asks. The two-pass split (1.62.11) and the preflight
memory guard (1.62.13) shipped earlier; this is the retry.

The memory pressure behind a failed or OS-killed `./test.sh --all` during a
deploy is usually ordinary desktop app usage on the shared machine, not
anything test.sh did, and it often clears within a couple of minutes --
observed repeatedly across this session's deploys. One retry after a delay
turns a class of deploy failure that currently needs a human to notice and
re-run into one that resolves itself.

Deliberately ONE retry, not a loop: a second consecutive failure propagates
for real rather than masking a genuine break behind indefinite retrying.
`DEPLOY_TEST_RETRY_DELAY_SECONDS` (default 120) is the knob, matching
test.sh's existing env-var tunable convention.

Implementation leans on `set -e`'s existing semantics rather than fighting
them: `if ! ./test.sh --all; then ... ./test.sh --all; fi`. A command in an
`if` condition is exempt from `set -e`, so the first failure is caught; the
retry is the last statement in the block and is NOT in a conditional, so
`set -e` aborts the script with the retry's own exit code if it also fails.
Verified both branches by simulation before writing the real tests.

First tests deploy.sh has ever had. It calls `./lint.sh` and `./test.sh` by
relative path, so the tests copy the real deploy.sh into a tmp dir beside
fake versions of those, with fake python/twine on PATH so the build and
upload steps can never touch anything real. Three cases: transient failure
then success (proceeds, exactly 2 calls), two failures (aborts, exactly 2
calls -- not a loop, and never reaches the build step), and first-try pass
(1 call, no retry noise). The two retry cases fail against the pre-fix
script; the pass-through case correctly passes either way.
## #497: other_genes correctness fixes from code review of #493 (1.62.18)

Code review of the just-shipped #493 work found several real defects. The
serious one, reproduced live against the repo's own fixture before fixing:

**Ensembl-ID queries false-flagged every co-queried sibling.**
`other_genes` subtracted `names`, the raw resolved query set.
`resolve_gene_query` fills only `ids` for an Ensembl ID and leaves `names`
EMPTY, so an ID-based query subtracted nothing:

    by SYMBOL  (["NRAS","KRAS"]):  other_genes = "" , ""      correct
    by ENS ID  (same two genes):   other_genes = "NRAS","KRAS"  false alarm

Exactly what an existing test pins as must-not-happen, just reached by the
other input shape. The same subtraction was also too WIDE: `names` is
HGNC-alias-expanded, so a gene whose approved symbol collides with one of
those aliases got silently erased from `other_genes` -- a false negative in
the one field whose whole job is to surface cross-reactivity, and strictly
worse than a false positive because the reader sees an empty cell and stops
looking.

Fixed by subtracting the symbols that actually survived the precise gene
filter instead of the raw query set. Covers both input shapes, drops nothing
to alias collisions, and on an unfiltered scan the set is empty by
construction so the field degrades to "every other gene", the only thing it
can mean when nothing was asked for.

Also from the same review: `other_genes` had no truncation while
`format_table` computes column widths once across the whole result, so one
peptide hitting a paralog family (PRAMEF, CT45A, GAGE) padded every other
row out to match -- now truncates like `pmids` does. `n_source_genes` was
documented and in `_empty_result` but never rendered in the default table.
My own comment calling `other_genes` a "peptide-level constant" was false
(it subtracts the row's own gene, so it's (peptide, gene)-level; `"first"`
aggregation is safe only because `gene_name` is in `group_cols`, and the
comment now says exactly that). Docstring covered only the filtered case.
Row-wise `apply(axis=1)` and a per-row lambda vectorized; `observed=True`
added to the one groupby in the file missing it.

**One review finding deliberately NOT taken.** It proposed fixing
`n_source_genes`'s undercount by counting Ensembl IDs instead of symbols.
Measured against the real 5.9M-row mapping table first: blank `gene_id` on
114,703 rows vs blank `gene_name` on 29,376 -- counting IDs would lose ~4x
MORE loci. Neither field alone is a complete key, and name/id pairing is
already unreliable by the time query() sees it (two independent
`_join_unique` calls upstream, then padded and exploded). A sound count has
to be derived at mapping time. Filed as #496 with the measurements; the
docstring now states the limitation ("at least this promiscuous, never at
most") rather than shipping a change that looks like a fix but isn't.

Tests added for the two branches that had none and where the semantics
actually differ: Ensembl-ID queries (fails against the shipped code) and
the unfiltered whole-corpus scan.

## #496: count source LOCI, not gene symbols, for n_source_genes (1.62.19)

The follow-up deliberately deferred out of #497. `n_source_genes` counted
distinct gene SYMBOLS off the exploded `gene_names` view, which silently
dropped every locus with no HGNC symbol -- and always in the reassuring
direction, so a reader concluded a candidate peptide was more gene-specific
(a safer TCR-T target) than it is.

**Measured before choosing a key**, against the real 5,875,577-row mapping
table, because the obvious fixes are both wrong:

| tier | rows |
|---|---|
| Ensembl ID present | 5,760,874 |
| no ID, symbol present | 106,470 |
| neither | 8,233 |
| blank `protein_id` | 0 |

So "just count gene_id" loses ~4x more loci than counting symbols does.
And a naive symbol-when-no-ID fallback double-counts, because **5,487
symbols appear BOTH with and without an ID** -- they'd land in one bucket
under the ENSG and another under the bare symbol.

New `mappings.locus_keys` therefore keys on the Ensembl ID, falls back to
the symbol *resolved to its ID first* (learned from the rows carrying both,
so the 5,487 collapse correctly), then the bare symbol when the corpus
never pairs it with an ID, and finally `protein_id`, which is never blank.
The last tier over-counts a nameless multi-isoform locus rather than
dropping it -- the safe direction for a promiscuity signal, and documented
as such.

Computed in `annotate_observations_with_genes` where the frame is still one
row per (peptide, protein) and gene_name/gene_id are genuinely row-aligned,
then wired through `_GENE_DERIVED` / `_DERIVED_COLUMN_DEPS` like the other
four derived columns, so every consumer of observations gets it rather than
just `pmhc_query`. That also removes the second, query-time code path that
was answering the same question differently.

**Corpus-wide impact:** of 1,282,910 peptides, 25,622 (2.00%) change count.
All 25,622 are increases; zero decrease, which is the evidence the
symbol->ID resolution didn't introduce double-counting. `AAAAAAPPPST` goes
from **0 to 2** -- the old count claimed it came from no gene at all.

`other_genes` still can't NAME an unnamed locus, so the docstring now points
out the tell: `n_source_genes` exceeding the `other_genes` entries plus the
row's own gene means there are unnamed loci behind it.

## #502: centralize "what is one sample"; --by-gene n_samples was ~always 1 (1.62.20)

`pmhc --by-gene` reported **1 sample for CTAG2 across 17 references** —
impossible on its face. `gene_distribution` counted
`nunique(attributed_sample_label)`, and that field is blank on 4,295,716 of
4,440,124 rows (**96.7%**), so every uncurated row collapsed into one
bucket: **13** distinct samples corpus-wide against **2,244** with the PMID
fallback. CTAG2 now reads 14, SSX2 1 -> 5.

The deeper problem wasn't the arithmetic, it was that `query()` had already
solved this properly — PMID fallback plus cell-line/donor decomposition —
so the repo published one column name under two definitions that had
drifted by two orders of magnitude. Same shape as #496's "two code paths
answering one question".

**Centralized rather than merely shared.** New `hitlist/sample_identity.py`
is the canonical home, with a fully public API (`SAMPLE_IDENTITY_COLUMNS`,
`SAMPLE_IDENTITY_ID_COLUMNS`, `sample_identity_ids`,
`add_sample_identity_columns`, `count_distinct_ids`, `count_samples`). The
attached columns are public too — `line_id` / `donor_id` / `donor_type_id`,
previously underscore-prefixed, which wrongly signalled callers shouldn't
rely on them when they are exactly the shared contract.

Also fixed the cell-line tier's missing last-resort fallback: a
`src_cell_line` row carrying neither a line name nor a mono-allelic host
got an empty ID and vanished from the count, where the donor tier had used
a PMID fallback all along (21 rows / 4 PMIDs / 19 peptides).

**Drift guard**, per the repo's no-private-interfaces rule: a test scans
every module for sample counts taken off the raw label and fails naming the
offending `file:line`. Verified it actually fires by reintroducing the bug
in `export.py` and watching it point at `export.py:4219`.

**Deliberately NOT unified:** `samples.py::sample_peptidomes` is a third
definition, grouping raw scanner output by
`(pmid, antigen_processing_comments)`. It runs one pipeline stage earlier,
before curation attaches `src_cell_line` / `cell_line_name` /
`monoallelic_host`, so it *cannot* use this definition. Documented in the
module docstring so nobody unifies it later and breaks it.

One test lesson worth recording: the cross-path agreement test initially
PASSED against the broken code, because the existing fixture has every
`attributed_sample_label` curated, so the old count agreed by accident.
Rewrote it on an unlabelled fixture where it fails pre-fix with
`assert 1 == 3` — the CTAG2 symptom in miniature. A green test that cannot
fail is worse than no test.
# Priority release prerequisite: phase-local memory retry (#526)

The pending clean-main 1.62.26 deployment reproduced #526: all 1,676
regular tests passed, the integration memory guard refused at 1.62 GiB,
and free memory recovered above 8 GiB during the retry delay. The current
deployment then reruns all regular tests instead of retrying the refused
integration preflight. Prepare the already-reviewed #527 patch directly
on main as a release prerequisite if this blocks publication again.

Keep the 2.5/5 GiB budgets and one retry. Only preflight refusals can retry;
actual pytest failures must stop immediately. Retain serial fallback guards,
test each phase independently, and preserve the reserved version 1.62.35.
Do not incorporate unrelated curation from the old stack.

- [x] Preserve the old #527 branch in a named backup.
- [x] Apply only its phase-retry implementation and script regression tests.
- [ ] Run format/lint/script tests, full tests, and supported-version CI.
- [ ] Review, merge and deploy before correctness PRs only if needed to
      resolve their release blocker; otherwise retain as the next foundation.

Restacked review: only deploy.sh, test.sh, their tests, version and planning
notes differ from main. All 20 script tests pass (51.48 s); format and lint
pass. Full local validation and final CI are still required.

# Retry memory preflight at the test phase boundary (#526)

Deployment currently repeats an already-passed 11-minute regular suite when
the separate integration process cannot start under its 5 GB memory guard.
Keep both phases and their budgets, but retry only the current phase's memory
preflight, once, inside a single invocation. Actual pytest failures must fail
immediately. A new deployment invocation must always run both phases again;
there is no persisted success token or cross-revision reuse.

Add an explicit `test.sh --retry-memory` option used by deploy.sh. Keep the
existing deploy delay environment setting, forwarding it to the test runner.
Probe again after the delay and cap retries per phase. The same guard must
apply to the actual single worker when pytest-xdist is absent; otherwise
serial fallback bypasses both memory protection and its retry path.

- [x] Reproduce phase replay and unguarded serial fallback with command stubs.
- [x] Implement bounded preflight-only retries, preserving the memory budgets.
- [x] Cover regular/integration refusal and recovery, exhausted retries, real
      test failures, new invocations, and serial/xdist worker counts.
- [ ] Run format.sh, lint.sh, test.sh, CI and self-review.
- [ ] Bump 1.62.35, open its own PR, merge and deploy from clean main.

Review: seven new resource-control regressions fail before the change; all 20
test/deployment script cases pass afterward (37.50 s). Format/lint pass.
Self-review confirms unchanged 2.5/5 GB budgets, one preflight retry per phase,
fresh probes after each wait, actual one-worker accounting without xdist,
immediate propagation of real test failures, no build after failed tests, and
no successful-phase state persisted between invocations. Existing deploy delay
configuration is preserved. Full tests/CI and release remain pending.
#358 RNA curation remains uncommitted and will use the next version when its
verified primary inputs and complete profile QC are ready.

# Retired allele identities without rewriting reported typing (#456)

## Priority release plan for genotype correctness (#528, #514)

The user elevated mutation-token fabrication and genotype-contradicting
attribution as urgent correctness defects. Their implementation exists in
PRs #530 and #531, but the old stack makes them wait for unrelated changes.
Retain #529's allele-identity handling as the correctness prerequisite.
The pending main release reproduced #526, so place its isolated phase-retry
repair first: #527 -> #529 -> #530 -> #531. Preserve the reserved
versions 1.62.35 through 1.62.38; patch versions may skip unshipped
numbers. Other PRs must be rebased and bumped beyond the priority releases
before they land. Preserve named backups of all original branch tips.

No scientific values should be changed by restacking. Re-run mutation
ownership and downstream join regressions, missing-tissue genotype tests,
and corpus audits on the new bases. Retain uncertainty for coarse typing,
missing loci, and predicted restrictions. Recheck latest development
dependencies without modifying an environment while tests are running.
Complete format, lint, full tests, final CI, review, merge and clean-main
PyPI deployment for each PR in dependency order. Never weaken memory guards
or report an unreleased draft as fixed for installed users.

- [x] Back up branch tips and restack the three priority PRs on main.
- [x] Verify regressions, current curation impact, and supported Python versions.
- [ ] Run format/lint/full tests and final-head CI on the restacked PRs.
- [ ] Publish pending main release, then merge/deploy #529, #530, and #531.
- [ ] Rebase and renumber the remaining queue after the priority releases.

Priority review: the new stack contains only allele identity, mutation
tokenization, and genotype-consistency changes. All 80 combined regressions
pass on Python 3.9 and 3.12. The mutation audit preserves every candidate
and precision category across the current main-based 772 sample records
and 376 distinct MHC fields, with unchanged YAML. No curated candidate
currently carries a mutation. The genotype audit again changes zero values
over 12,888 patterns representing 3,644,028 observations from 102 studies.
The identity audit still changes only B*44:01 on 7,127 rows across the full
5,333,255-row vocabulary. All six development dependency heads remain current.
Full local suites and final CI remain pending. With memory restored, the
pending clean-main 1.62.26 deployment has begun its serial full test gates.

IPD-IMGT/HLA 3.65.0 Deleted_alleles.txt (2026-07-14), HLA00317, confirms
B*4401 was a sequence error identical to B*44:02:01:01 (March 1994).
The audited 832-token vocabulary changes only B*44:01, on 7,127 rows.
Preserve the source's normalized restriction and field depth. A separate
cached identity resolver applies mhcgnomes aliases only when the gene or
first two allele fields change, truncating any added fields to the original
depth. Keep mutations on their original chains; require mhcgnomes 3.64.3,
which fixes upstream #193. Never enable aliases in the reported-value parser.

Use this resolver for derived serotypes, candidate sets, sample matching,
loader filters and pMHC aggregation/prediction candidates. Refresh stored
candidate identities before set filtering and refresh serotypes only for
restrictions containing a renamed allele. Preserve raw restrictions in
loaders and expose their distinct reported spellings in aggregate results.
Avoid deriving experimental certainty from an alias or expanding a genotype.

- [x] Read primary IPD retirement evidence and audit all restriction tokens.
- [x] Fix, review, merge and publish upstream mutation preservation (#193).
- [x] Add failing regressions for aliases, typing depth, mutations and chains,
      raw provenance, stored-index projection/filtering, aggregation and scoring.
- [x] Implement one derived identity helper and route relevant consumers to it.
- [x] Audit both complete indexes; assert only the documented identity changes.
- [ ] Run format, lint, focused checks, full tests and final CI; self-review.
- [ ] Bump 1.62.36, open its own PR, merge and deploy after the preceding releases.

Review: 37 initial regressions failed before the change. Format/lint and 463
focused tests pass on Python 3.12; 461 cases also passed on Python 3.9 before
adding the final two duplicate-set/projection regressions. All 929 catalog
members remain reachable. Current catalog memberships remain preferred over
memberships inherited from a retired designation. No genotype is expanded.

The 5,333,255-row audit covers both complete indexes and all 832 tokens.
Only B*44:01 changes identity (7,127 observation rows); all raw restrictions
and all parsed mutations retain their identity/chain. Canonicalizing a stale
catalog member B*15:112 -> B*15:11 additionally restores B15 membership on
63,990 observation rows. The primary HLA nomenclature broad/split table lists
B75 under B15: https://hla.alleles.org/pages/antigens/broads_and_splits/.
A direct old-index loader comparison verifies 71,117 serotype updates, 7,127
candidate updates, unchanged candidate counts and zero reported-name changes.
No binding rows change. Stored current names also refresh the affected
catalog membership, including under projection and serotype filters.

The latest six development dependency heads were checked; pyensembl was
updated to 2.10.17 at d4abbf26, and mhcgnomes 3.64.3 b276be94 is now the
required floor. The lockfile changes only that dependency and its requirement;
uv lock --check and pip check pass. The full feature suite was canceled to
reserve memory for the active clean-main release; it is not counted as a pass.
Full tests and final CI remain required.

Independent problems found and filed: mhcgnomes#196 (missing broad/split
memberships), #197 (missing documented retirement), and hitlist#528 (sample
field tokenization fabricates alleles from mutation labels). The resolver
preserves mutations; it cannot restore information already lost by that
separate tokenizer. Upstream #193 is fixed and published. RNA curation #358
remains independent and is reserved for 1.63.0.

# Parse mutant sample genotypes as molecules (#528)

Review follow-up, 2026-09-23: the same whitespace split remains in
`extract_allele_tokens` and `mhc_species_of`. The first invents HLA-E*76C
from B*08:01 E76C mutant; the second falsely assigns a mouse mutant both
human and mouse species. File the reproduced sibling defect and include
these consumers in the shared molecule segmentation fix. Move semicolon
handling into that shared helper. Preserve reported allele names in token
extraction, derived identities in sample matching, and coarse species
resolution. Audit all current sample fields for token/species changes as
well as candidate changes; reject malformed mutation tails consistently.

- [x] File the sibling-helper defect with exact reproductions (#537).
- [x] Add failing extraction/species regressions and share segmentation.
- [x] Repeat current-curation audit and focused tests on Python 3.9 and 3.12.
- [ ] Require final supported-version CI after this review follow-up.

Whitespace splitting can turn `E76C mutant` into an invented HLA-E allele,
while stripping the mutation from its actual B allele. Preserve complete
parseable molecules/pairs before splitting genotype lists. For mixed fields,
consume the longest parseable molecule/serotype/locus span, using mhcgnomes
instead of an HLA-only token regex. Keep whole-field coarse names and existing
list handling. Mutation tokens or a mutant marker left outside a successfully
parsed molecule must raise an explicit input error, never fall back to a
wild-type genotype. Do not mistake a compact genotype allele for a mutation;
use mhcgnomes' Mutation parser and test actual curated field vocabulary.

Require mhcgnomes 3.64.4 once upstream #198 passes CI and ships: whole-pair
alpha selectors must retain chain ownership before this tokenizer trusts
those parsed results. Test against the isolated upstream branch until then;
do not replace a dependency underneath an active release test process.

- [x] Add failing regressions for single mutants, mutated pairs, mixed lists,
      selectors, invalid mutation tails, and ordinary multi-allele genotypes.
- [x] Implement molecule-aware segmentation with explicit malformed-input errors.
- [x] Audit every curated sample MHC field before/after and preserve raw YAML.
- [x] Require the shipped upstream fix, update the lockfile and dependency envs.
- [ ] Run format, lint, full tests and final CI; review and fix findings.
- [ ] Bump 1.62.37, open a separate PR, merge and deploy in order.

Review: format/lint and 459 focused tests pass; all 15 new cases also pass on
Python 3.9. Tests exercise the export join for separate mutant and wild-type
arms, including full class-II pairs, in addition to segmentation and invalid
mutation rejection. Every candidate and precision category is unchanged over
all 775 curated sample records (376 distinct field values), with identical
raw YAML SHA-256. No mutation-bearing curated sample field currently exists;
the regressions use explicitly synthetic inputs.

Requires the published mhcgnomes 3.64.4, whose chain-selector fix passed
17,171 tests and CI on Python 3.9–3.12. Both isolated hitlist environments
use its latest merged development commit 5565acd. The lockfile only changes
mhcgnomes and its requirement; uv lock --check and pip check pass. The full
test entry point was refused before pytest by its unchanged memory guard
(0.22 GiB available; 2.5 GiB required), so full local validation remains
pending. Final CI, merge, and clean-main deployment are also required.

Priority-stack review: shared segmentation now also protects allele extraction
and sample species (#537), including class-II chain ownership, mouse mutants,
true mixed-species genotypes, haplotype/coarse species and malformed tails.
All 333 curation, mutation and identity checks pass on Python 3.9 and 3.12;
format and lint pass. The repeat audit covers the 772 samples on the direct
main-based priority stack, with 376 distinct fields: candidates, precision,
extracted molecules and species are all unchanged. Raw YAML SHA-256 is
7380766616315777df7593946a7637a30c7afe21cf5fabaa84ef4e08771e74d0.
All mutation examples remain synthetic. Final CI and the full local suite
are required again after this review change. Reserve 1.63.0 for RNA #358.

# Respect reported genotype evidence in class-pool attribution (#514)

The class-pool text scorer currently ignores the observation's restriction,
so a generic display-label word can select a sample with a disjoint reported
genotype. Re-read Ritz 2017 (PMID 28834231, PMC5846733): both cell lines were
SSO/SSP typed; MAVER-1's six alleles and homozygous HEK293's three alleles are
explicitly distinct. Reproduce the missing-tissue failure before editing.

Validate the proposed class-pool text/group winner without changing text
scoring weights: when observation and sample contain precise, reported
molecules at the same loci, reject a disjoint named winner. Unknown,
coarse, or serological typing must not be treated as complete exact evidence,
and derived/predicted candidate sets must not be used as reported typing.
Respect class-II chain versus complete-pair precision and mutation identity.
Missing loci are unknown, and an untyped candidate cannot win merely by
surviving exclusions. Genotype evidence may select a sample only when exactly
one candidate overlaps and every alternative is explicitly disjoint. Include
the restriction and its evidence category in row-discriminator
keys so rows with identical text but distinct typings cannot share a winner.
Leave curated per-row sample labels authoritative and keep raw data intact.

- [x] Reproduce the paper-grounded missing-tissue case and precision edge cases.
- [x] Implement the conservative winner check and use precise row keys.
- [x] Audit changed attributions against the stored corpus and primary sources.
- [ ] Run format/lint/full tests and final CI; review all behavioral changes.
- [ ] Bump 1.62.38, open a separate PR, merge and deploy after #530.

RNA #358 remains independently in progress and will use 1.62.39.

Review: three source-grounded export regressions fail on the base. Initial
testing exposed two unsafe generalizations, both corrected before review:
filtering candidates before text scoring changed relative token weights and
could promote an untyped cell line; treating absent loci as negative typing
would reject DRB3/DRB4 observations from samples typed only at DRB1. The final
check preserves scoring, handles gene coverage as uncertainty, and uses the
existing species-ancestry relation for Calu/DLA naming. Pair-vs-chain identity,
mutation identity, retired names, serotypes, partial alleles, unknown fields,
predicted restrictions and identical-text/different-genotype rows are tested.

Audit all 12,888 distinct attribution input patterns across 3,644,028 stored
observations in 102 multi-sample studies: zero existing attribution changes.
Raw source YAML and restrictions are unchanged. The Ritz paper explicitly
reports the two SSO/SSP-typed genotypes and HEK293 homozygosity; the missing-
tissue regressions reconstruct that source evidence. No full-corpus column
copy is added: restriction/evidence are part of the existing unique-row keys.
Full local tests and final CI remain required before merge, with clean-main
PyPI publication afterward.

Final focused validation: format/lint and 201 export/curation tests pass;
all 22 new cases also pass on Python 3.9. The audit remains unchanged after
using the existing restriction/evidence columns directly as row keys.

## PR #533 release-runner restack (1.62.40)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


# Attribute exact deposited elution conditions (#512)

Re-read Stopfer 2020 (PMID 32488085, PMC7265461), Figures 4/6 and the
Supplementary Data 3/5 file maps: separate DMSO, 1/10 uM palbociclib, and
10 ng/mL IFNG arms were profiled at 72 hours. IEDB contains seven exact
treatment-enumeration statements and seven independent MDA-MB-231
quantification statements. Of 12,277 melanoma observations, 8,390 name
one treatment (5,528 IFNG, 2,171 high-dose and 691 low-dose palbociclib);
3,887 name multiple arms, not simultaneous combination treatment.

Add a validated PMID-level YAML map from exact assay-comment strings to
the existing condition IDs supported by each unambiguous statement. Map the
three single-treatment statements; the four multiple-arm statements remain
unmapped and cannot select an arm, even if a caller supplies fewer candidates. Preserve
the unchanged comment text as source evidence; never use substring matching
for doses. Validate that keys are nonempty strings and values are nonempty
lists of distinct condition IDs declared by this study's profiled samples.
The existing elution-condition selector will use an exact curated match
when available, and return a winner only when exactly one candidate ID is
supported. Use it in both allele-match and class-pool tie-breaks. Keep raw
observations and the existing generic narrative admission rules unchanged.

- [x] Add failing source-comment/dose-boundary/multi-arm and schema tests.
- [x] Implement the validated mapping and both selection paths.
- [x] Review all seven statements and curate only the three unambiguous ones.
- [x] Audit all 15,821 observations and all 14 distinct comments before/after.
- [ ] Run format/lint/full tests and final CI; review, merge and deploy 1.62.40.

Review: 473 expanded curation/export tests pass and both the class-pool and
ambiguous exact-allele paths select source-supported condition IDs. The
audit covers 36,584 rows across Stopfer, Ritz, and Schellens: exactly 8,390
Stopfer rows acquire an arm, 3,887 multi-arm rows stay ambiguous, all 3,544
MDA quantification assignments are unchanged, and every value in the other
two studies is unchanged. Raw peptides, restrictions, comments, cell names,
and sample groups are identical. The categorical vocabulary expands only
to accommodate newly assigned metadata. Full local tests and final CI remain
required before merge, followed by clean-main PyPI publication.

Found and filed #532: filtering to one row can remove sample-group identity
because discriminator variability is computed within the selected query.
The unchanged base reproduces this with two Stopfer rows versus one; the
source-mapping tests exercise the complete multi-line study context. This
separate query-stability fix follows #512; no generic narrative guard is
relaxed here. RNA #358 remains independent and will use 1.63.0 once its
source data and validation are complete.

Final review: all 28 regression/schema cases pass on Python 3.9 and 3.12;
the 13 public-export regressions fail against the unchanged #531 base.
Format, lint, and whitespace checks pass. Full local tests and final CI
remain pending in the serial validation queue.
# Restack the next attribution fixes after the priority releases

Keep the reviewed #512 -> #534 -> #532 changes behind #528 and #514.
The old branches still include the unrelated queue that was removed from
the priority stack. Back up all three tips, then apply only their own
commits onto the final #531 head and preserve versions 1.62.40–1.62.41.
The independent RNA work remains reserved for 1.63.0.

This is a topology repair, not a new curation pass. Preserve the source-
verified dose statements, cohort restrictions, and query context rules.
Compare scientific source/test/data diffs against the old heads; investigate
any change beyond removal of unrelated ancestor commits. Repeat focused
checks and the recorded source audits on the new bases before release.
Keep memory-heavy audits and full suites serial with the active deployment.

- [x] Preserve old branch tips and restack only each issue's own commits.
- [ ] Review conflicts, confirm scoped diffs, and rerun focused validation.
- [ ] Repeat source/corpus audits and final supported-version CI.
- [ ] Run format/lint/full local tests, review, merge and publish after #531.

## PR #536 release-runner restack (1.62.41)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


# Keep peptide-to-patient attribution within its source cohort (#534)

Primary-source review confirms that Sarkizova 2020 (PMID 31844290,
PMC7008090) contains independent monoallelic 721.221 and patient-tumor
experiments. Supplementary Data 2 (publisher MOESM4, SHA256
46bf469653a4e252825f185e1d2946a717aace0d9e894cb0c0e81903100bbfca)
contains patient sample directories, including MEL_13240_005 and its IFNG
arm for NAPWAVTSL. Peptide overlap does not transfer that patient identity
to a separately recorded monoallelic assay. Reconcile the existing CSV
against the workbook without changing its valid patient assignments.

Declare `peptide_attribution_restrictions` in PMID YAML: an optional
validated nonempty list of exact reported restriction strings to which the
registered peptide map applies. For Sarkizova, allow only the deposited
`HLA class I` patient-cohort restriction. Match before donor-set promotion;
do not infer cohort eligibility from derived monoallelic flags or require
source classification to be enabled. Unscoped Connelley maps retain their
source-verified exact-allele behavior. Preserve the public peptide map API.

Repair derived labels in existing observation/binding artifacts at load:
for scoped studies, retain genuinely narrowed `peptide_attribution` rows,
but clear labels attached to out-of-scope exact restrictions. Remove only
the duplicate donor copies of those invalid rows, keyed by their original
PMID, assay IRI, peptide, and restriction. Retain raw evidence and valid
patient splits. Require a rebuild if a stale affected row has no source
assay identity; never guess which repeated rows are independent assays.
Projection must pull the repair inputs and then return exactly the requested
columns, including a one-column peptide query. Bump the artifact version
so a normal rebuild regenerates the corrected scan rather than skipping.

The current MS artifact has 89,723 mislabelled monoallelic rows representing
47,641 source assay/peptide/restriction records: 42,082 excess donor copies.
Its 54,682 valid patient-attribution rows must be unchanged. Audit the
complete affected study and both index modalities, then check scanner output
from original source rows and export metadata, including HLA-G. This fix
must precede the class-roster change in #535/#532.

- [x] Reconcile patient CSV pairs against original Supplementary Data 2.
- [x] Add failing mixed-cohort scan, old-index repair, projection, and schema tests.
- [x] Implement validated source scope, scanner guard, and derived-index repair.
- [x] Audit affected study and preserve Connelley exact-allele attribution.
- [ ] Run format/lint/full tests and CI; review, merge and deploy 1.62.41.

Review: the CSV exactly matches all 39,624 original workbook peptide–patient
pairs (zero missing or unsupported pairs) and remains unchanged. Across the
stored study, 89,723 false labels become 47,641 unlabelled independent source
records, removing only 42,082 excess donor copies. Their retained evidence
is identical; all 54,682 valid patient rows and all 146 Connelley MS rows
are identical before and after, including their exports. HLA-G rows now
export the corresponding 721.221 transfectants and HLA-G typing. Re-scanning
the original IEDB/CEDAR rows independently gives the exact same result as
repairing the old scan, with all 222 Connelley scan rows unchanged.

The initial regression run failed 14 cases on the unchanged base. Format,
lint, and 391 expanded tests pass; the additional public-export regression
also passes. All 64 final scanner/repair cases pass on Python 3.9. The local
full suite and final CI remain required; memory guards currently prevent
release validation. Latest development HEADs for mhcflurry, mhcgnomes,
pyensembl, datacache, gtfparse, and serializable were rechecked and match the
isolated test environment. This PR remains a draft until its gates pass.

## PR #535 release-runner restack (1.62.42)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


Restack review: stable patch IDs for each scientific source, data and test
delta are identical before/after: #512 4a8b5368, #534 68097bf1, and #532
4300627a. The sole conflict was between appended planning sections; both
were retained. Format and lint pass on all three new branches. Fresh CI and
repeat corpus/source audits remain pending, with full local suites kept
serial behind the active 1.62.35 deployment.


# Make sample attribution stable under output filtering (#532)

The same deposited observation must retain the same sample/group/condition
when a caller asks for one peptide, one restriction, or another narrower
output. The current variance guard measures the selected query, so removing
other rows can erase a known cell identity or admit narrative after a factual
field stops varying. Preserve the full-study boilerplate guard and the
existing unfiltered attribution behavior.

For queries with row filters, read a compact context containing only PMID,
restriction, class, and the four discriminator fields from the complete MS
artifact. Stream projected parquet batches and collapse duplicate patterns;
cache against path, nanosecond mtime, and size so rebuilds invalidate it.
Restrict the compact context to queried curated multi-sample studies. Do not
load peptides, expression, or other wide payloads for the context.

Use full-context restriction pairs when constructing sample join aliases.
Evaluate ambiguous-allele discriminator variation and winners against the
context; apply winners only to selected output rows. For the class-pool
variance gate, retain only context rows that the preceding allele/single-
sample paths leave without sample MHC, exactly as the current full-query
path does. Reuse the same candidate and winner dictionaries. Never add
context observations to the returned frame or change scoring weights.

Keep the complete curated sample roster when filtering observation class:
class selection must not turn a multi-sample study into a single-sample
study and change its fallback provenance or matched-sample count.

Tests must use a real temporary parquet and public filters, not simulate a
filtered loader that also hides the full study. Cover grouped cell identity,
exact-allele ties, ungrouped samples, treatment-dose attribution, unchanged
multi-arm ambiguity, and genuine constant narrative. Compare every relevant
metadata field between full and one-peptide exports; verify row counts,
raw values, and context-cache invalidation. Audit unfiltered output against
the base and a representative filtered query for each multi-sample study.

- [x] Add public-query regressions and demonstrate base failures.
- [x] Implement bounded full-study discriminator context without changing curation.
- [x] Finish memory review and all filtering paths after the curation prerequisite.
- [x] Audit full versus filtered attribution across the corpus.
- [ ] Run format/lint/full tests and CI; review, merge, deploy 1.62.42 after #534.

Review: 10 public-query regressions fail against the original base and two
class-roster cases fail before retaining the complete roster. Rebased onto
the source-verified cohort repair in #536; added explicit old-index HLA-G
checks for peptide, allele, and class queries. Format/lint and 275 expanded
export/repair tests pass after rebase. All 18 final query-context cases pass
on Python 3.12 and all 38 query-context/repair cases pass on Python 3.9.
Full local tests and final CI remain required before merge and clean-main
deployment.

The repeated audit uses freshly reconstructed patterns after correcting the
cohort data: 11,850 patterns represent 3,601,946 observations from 102 studies.
Every unfiltered value is identical to the corrected base. Of 132 narrow
queries (66 known and 66 ambiguous), the base's 24 known-sample discrepancies
disappear and all ambiguous cases stay unchanged. Every class-filtered
pattern matches the complete export. Before this change class filtering
changed matched-sample counts on 1,506,759 represented rows and attribution
provenance on 38,252, while all sample labels now agree after the separate
cohort repair. No scoring weights, source YAML, or stored artifacts change
in this PR. The compact context contains seven columns, streams 10,000-row
batches, and caches two file revisions without retaining peptide payloads.

## PR #516 release-runner restack (1.62.43)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #516 priority queue restack (1.62.43)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #289 specification — reusable manual-dataset guidance

IEDB/CEDAR now support automatic fetching, but HPA and DepMap still reach the
reported hard-coded CLI message. Make both `fetch` and missing `get_path`
manual-dataset errors name the download URL, expected file and registration
dataset key without choosing a downstream application's command prefix.
Keep exception types and behavior unchanged. Patch bump to 1.62.43; inspect
both real error paths in an isolated empty registry, run format/lint/test,
review CI, merge and deploy.

- [x] Update both manual-dataset error paths and verify the resulting messages.
- [ ] Run required gates, review CI, merge, deploy and verify PyPI.

Review: both actual error paths for absent `depmap_rna` retain the download
URL, exact expected filename and dataset key, with their original exception
types. Format/lint pass. No parser, registry or download behavior changed.

## PR #517 release-runner restack (1.62.44)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #517 priority queue restack (1.62.44)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #509 specification — class-scoped fixture lifetime

Two long-peptide class-scoped fixtures are instance methods despite not
using their instance. Pytest 9.1 deprecates this lifetime mismatch. Define
them as ordinary module functions with class scope and distinct names,
without changing their constructed indexes. Re-plan after Python 3.9 CI
showed that its classmethod descriptor lacks the metadata pytest expects;
module functions avoid that version-dependent descriptor behavior. Verify
the existing mapping tests with PytestRemovedIn10Warning promoted to an
error, then format/lint/test, review CI, merge and deploy 1.62.44.

- [x] Reproduce the warning as an error and correct fixture binding.
- [ ] Run required gates, review CI, merge, deploy and verify publication.

Review: the existing suite produced 12 setup errors with the deprecation
promoted to an error; all 30 tests now pass under that same warning policy.
Format/lint pass. Scope, construction and test assertions are unchanged.
The change follows pytest's documented classmethod fixture migration.

## PR #518 release-runner restack (1.62.45)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #518 priority queue restack (1.62.45)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #508 specification — explicit boolean mapping

Replace the two deprecated pandas no-silent-downcasting option contexts
with nullable-boolean conversion before filling missing mapped values,
then preserve the public plain-bool dtype. Check object, string and
categorical inputs, missing values, homogeneous categories and empty
frames. Use warnings-as-errors to verify the existing export tests, and
compare both flags over the actual restriction vocabulary. Run required
format/lint/test and CI; review, merge and deploy 1.62.45.

- [x] Reproduce warning/failure boundaries and fix both conversions.
- [x] Verify outputs/dtypes and existing export behavior.
- [ ] Complete full gates, merge, deploy and verify PyPI.

Review: all 15 dtype/missing-value cases fail before the fix with the
deprecated option warning treated as an error. The complete non-integration
export suite now passes under that policy (176 tests). Compared every output
column of the old/new training-default paths over all 1,304 distinct
restriction strings in observation and binding artifacts plus missing data,
for object, string and categorical inputs: zero differences. Plain boolean
dtypes and non-default row indexes are preserved. Format/lint pass.

## PR #519 release-runner restack (1.62.46)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #519 priority queue restack (1.62.46)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #306 specification — finish source-axis consistency

PRs #309/#312 already introduced canonical source_species and documented the
two raw inputs. Residual defect: an explicit `unidentified` source blocks a
valid legacy-species fallback in filters/flags, while pMHC's independent
warning coalesce treats it as resolved. Centralize missing-source handling
in the existing shared coalesce, preserving the original raw columns and
source_organism precedence for two valid inputs. Treat blank, unknown and
unidentified (case/whitespace insensitive) as missing, and keep unknown
source output blank. Use the coalesce for the pMHC warning. Ensure projected
chimeric/engineered/xenograft flags read the legacy fallback too. Verify
MS/binding filtering and narrow projections, categorical/null inputs, and
warning agreement; audit real raw-pair changes. Ship 1.62.46 after gates.

- [x] Reproduce source-sentinel and projection inconsistencies.
- [x] Share coalescing and complete derived-column dependencies.
- [ ] Audit affected corpus inputs, test, review CI, merge and deploy.

Review: seven regression cases fail before the fix. All 255 focused
observation, pMHC and export-species tests now pass; format/lint pass.
Audit covered 4,440,428 observation rows and 892,827 binding rows: 72,329
and 371 respectively change only canonical source_species from the literal
unidentified to blank. Raw fields are preserved, all three biological
flags are unchanged, and unresolved-source warning counts are unchanged.
The tests additionally cover valid legacy fallback, source precedence,
source-only query rows and narrow projections absent from this corpus.

## PR #521 release-runner restack (1.62.47)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #521 priority queue restack (1.62.47)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #452 specification — five ERAP2 experimental lines

Re-read Lorente 2019 (PMID 31530632 / PMC6823859), Experimental
Procedures and Results. Replace the two aggregated arms with parental
polyclonal WT, unedited process-matched WT1/WT2, and ERAP2 knockout
KO1/KO3. Record three independent biological preparations per line and
retain the shared B*40:02 transfectant / ERAP1 Hap8 context. Distinguish
parental untreated and process-matched control roles without claiming
the unedited controls are gene knockouts. Use unique condition IDs and
valid comparator references. Audit all 10,319 IEDB rows: three comments
distinguish WT-only, KO-only and both, but never identify a clone. Preserve
that ambiguity and state the actual clone-resolution limit; do not assign
WT-only rows to parental WT or any KO row to a guessed clone. Check all
other study entries are semantically unchanged. Release 1.62.47 after
source regressions, corpus audit, format/lint/full tests, CI and review.

- [x] Read primary methods and audit the deposited row descriptions.
- [x] Curate all five lines and honest attribution limits.
- [ ] Verify controls, replicates, observed-row impact and required gates.
- [ ] Review, merge, deploy and verify PyPI.

Review: all four source/attribution assertions fail against the old two-arm
curation; 86 focused curation, condition, group and arm-resolution tests
pass after the fix. Format/lint pass. All 10,319 observed rows retain their
raw B*40:02 restriction and blank clone/condition fields; they gain C1R
system attribution and the arm_not_recorded verdict. The scoped YAML
rewrite verifies that every other entry is semantically unchanged.
Primary methods expose a separate cellular-genotype versus selected-ligand
restriction ambiguity, filed as #520; this PR records that limitation
explicitly without silently changing the existing attribution contract.

Release ledger: #501 / PR #505 shipped 1.62.21. #504 / PR #506 shipped
1.62.22 after 1,659 regular and 43 integration tests. Both releases have
verified PyPI wheel and sdist artifacts. Remaining PRs still need their
full local gates, final review, merge and deployment.

## PR #523 release-runner restack (1.62.48)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #523 priority queue restack (1.62.48)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #357 specification — fetchable, usable DepMap expression

Use the pinned primary DepMap 24Q4 release (Figshare 27993248.v1), not
the changing portal landing page. Register the actual gene matrix,
transcript Profile matrix and their model/profile mapping companions.
Provide an explicit opt-in `data fetch depmap` bundle that downloads and
builds line_expression.parquet; normal package installation/build remains
independent of downloading multi-GB expression matrices. Keep each source's
normalization, citation and profile provenance accurate.

Primary-file audit disproves two assumptions in the current parser: the
transcript file is OmicsExpressionTranscriptsTPMLogp1Profile.csv (4.17 GB),
its rows are ProfileID, and its headers are SYMBOL (ENST...), not the
reverse. Stream rows and select registered systems before melting so the
whole release is never expanded in memory. Join profiles through the
release mappings, select source-appropriate profiles without summing
replicates, and reconstruct TPM from log2(TPM+1). Preserve compatibility
with already-registered supported files. Model metadata contains five of
the six listed lines; HEK293 is absent and must not be fabricated. HAP1
has a stranded RNA library, separately relevant to #358. The actual
checksum-verified gene matrix contains 19,193 HAP1 values: library
strandedness must not be used to infer matrix membership. Select the
default RNA profile and check whether its row is actually present.

The resolver must report exact/parent/family RNA only when its source has
rows for that key. Missing optional downloads retain lower-tier fallback.
Cache availability against the built artifact's identity so installing or
rebuilding data becomes visible in the same process. Test explicit source
availability, missing-data fallback, exact official header shapes, profile
selection, bounded parsing and a fake-download-to-peptide-origin end-to-end
path. Audit the real selected rows and source profile IDs. Run all required
checks, review CI, merge and deploy 1.62.48.

- [x] Inspect primary release metadata, README, model/profile maps and CSV prefixes.
- [x] Reproduce fetch/parser/availability failures and implement the complete path.
- [ ] Validate real data and focused/full test gates.
- [ ] Review, merge, deploy and verify PyPI.

Review: four regressions fail before the fix. All 293 expanded export, CLI,
dataset and expression tests pass; 98 expression/fetch tests pass after the
final profile-selection correction. Format/lint pass. Downloaded all five
primary files and verified their sizes and published MD5s. The real build
produces 1,502,002 rows, including 19,193 genes and 227,188 transcripts for
each of six lines (HAP1 included), with one release-default ProfileID per
line. Five requested lines now resolve to actual exact-line RNA; HEK293
retains tissue fallback because it is absent. HAP1 registry wiring belongs
to #358. The audit script completed in 51 seconds; the external timing
wrapper alone returned an error because sandbox policy disallows its
sysctl query. Its full JSON report and built index were verified separately.
Wrong legacy placeholder citations discovered during the audit are filed
as #522 and will be corrected separately from this download/parser change.
Full local tests, final CI, merge and deployment are still required.

## PR #524 release-runner restack (1.62.49)

Move the already reviewed scientific patch after the release-runner
foundation (#539, 1.62.39). Preserve its code, source data and tests exactly.
Retain the original branch; prove patch identity, bump the release version,
run format/lint and final-head CI, and repeat the relevant compact source
audit before merging. Run the full clean-main release workflow and verify
its tested artifacts before local PyPI publication. No memory gate is waived.

- [x] Prove the non-planning patch is unchanged.
- [ ] Run format/lint, focused checks, source audit and final-head CI.
- [ ] Review, merge, run the clean-main release and verify PyPI publication.


## PR #524 priority queue restack (1.62.49)

Move this previously reviewed change after #528/#514 and their source-verified
attribution follow-ups. Preserve the implementation, curation and tests exactly;
only the base, release version and planning records change. Keep the original
local branch, compare stable patch IDs, then run format/lint and fresh CI.
Repeat affected source audits and full local release gates before publication.

- [x] Preserve and compare the original non-planning patch.
- [ ] Run format/lint and fresh CI on this exact head.
- [ ] Review against the updated base, validate, merge and deploy in order.


## #522 specification — verified RNA-source provenance

Check each stored PMID against the original article and read the intended
papers' methods. Essletzbichler HAP1 is PMID 25373145, SRP044391,
TopHat alignment with mapping-weighted gene counts reported as FPKM;
remove the invented kallisto/TPM claim. Pearson is PMID 27841757;
its own cohort is 18 donor B-LCLs, while JY is an external validation
transcriptome reprocessed with kallisto 0.42.5. Retain only that supported
JY association and remove unsupported HHC coverage. Kaabinejadian
PMID 35154160 profiles HLA ligands, not host RNA; retire the falsely
attributed RNA source and its C1R/721.221 anchor references. These are
empty placeholders, so actual RNA values and fallback resolutions must
remain unchanged. Document source evidence and availability explicitly;
do not convert FPKM to TPM without a complete quantified feature set.
Add source-fact regression assertions and verify all registry links,
packaged data and existing fallback behavior. Run format/lint/full tests,
final CI, review, merge and deploy 1.62.49 in its own PR.

- [x] Re-read primary methods, provenance and data-access statements.
- [x] Correct or retire unsupported placeholder metadata.
- [ ] Verify unchanged values/fallbacks and all required gates.
- [ ] Review, merge, deploy and verify PyPI.

Review: all three primary-source regression checks fail before the change;
101 provenance/expression/fetch tests pass afterward, with format/lint.
Every registered line was resolved against an empty optional-data cache and
against the full 1,502,002-row expression index before/after: all anchors,
row counts and hashes are identical. Source IDs referenced by remaining
anchors all exist. No numerical expression file changed. Full tests and
final CI remain required before merge; deployment remains required after.
