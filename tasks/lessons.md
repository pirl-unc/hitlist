# Lessons

## 2026-10-07

- Standard input formats should work with defaults. Do not demand file and
  column flags when a unique conventional file/header identifies them. Share
  inference between CLI/API, preserve explicit overrides, record the resolved
  choices in provenance, and ask only when a real ambiguity remains. Never
  substitute invented patient expression or raw counts for missing TPM.

## 2026-10-06

- Dependency compatibility has three separate checks: declared bounds, the
  committed lockfile, and the environment actually running tests. Inspect and
  validate all three before calling an upgrade handled; permissive bounds and
  a manually upgraded environment do not prevent uv sync from restoring an
  obsolete transitive cap.

- Keep repository ownership explicit when a requested workflow spans packages.
  Hitlist accepts expression tables and exports auditable presentation evidence;
  it does not assemble vaccines. Tsarina and Vaxrank own downstream target
  selection and construct assembly. Mentions of their capabilities are context,
  not authorization to move those responsibilities into Hitlist. For this
  expression-table follow-up, deliver the evidence bundle only.

## 2026-10-05

- A named reference query that users expect in the normal install should not
  require discovering an extra without a concrete benefit. For CTA support,
  use the canonical packaged provider as a normal dependency while keeping
  its loading lazy; do not copy a static biological list to avoid the package.

- Optimize elapsed release time from measured phase costs. A unit run with
  an installed corpus is not automatically comparable to a no-corpus matrix
  job: coverage, concurrency, dependency revisions and runner variation
  differ. Audit actual reads before attributing the slowdown, and preserve
  full verification when reducing redundant work.

- Re-run every newly added categorical consumer case on both supported pandas
  majors before pushing. A pandas 2 pass does not establish pandas 3 behavior:
  replacing a blank categorical cell name with a display placeholder passed
  on 2.3.3 but failed on 3.0.5. Add the actual destination category before
  assigning, and test both null and existing blank inputs. A final test added
  after the broader verification needs the same cross-version check.

## 2026-10-01

- Generalize allele matching to species compatibility, not just acceptance of
  non-human allele syntax. Keep candidate identity/class validation separate
  from compatibility with the observation's evidenced presenting-MHC context.
  Source proteome, presenting cells, host and MHC species are distinct axes;
  documented xenografts or engineered MHC explain particular differences, not
  arbitrary cross-species assignments. Do not use a mismatch-derived xeno flag
  as its own evidence that the mismatch is valid. Preserve reported evidence,
  exclude incompatible inferred candidates with an auditable reason, and keep
  missing context explicitly uncertain rather than silently deleting rows.

## 2026-09-30

- Prefer readable compound observation IDs when the source and curated arm
  already have stable identifiers. A full cryptographic digest hides the useful
  provenance and makes compatibility tests hard to review. Use explicit
  namespaces and percent-escaped components to avoid delimiter collisions;
  reserve content hashes for integrity checks and artifact fingerprints.

## 2026-09-29

- On a shared workstation, a successful memory preflight is only a momentary
  capacity check. Other jobs can start during a long test run. When the user
  reports pressure, measure this task's processes, stop its heavy work and move
  release validation to CI. Resident memory alone excludes compressed/swapped
  pages; do not use a small RSS snapshot to deny contributing to the pressure.
  Keep the original failure/interruption in the record and never count a partial
  run as passing.

- Parameterized test data must be deterministic at collection time. gzip.compress
  embeds the current timestamp by default; putting its bytes in pytest parameters
  gave xdist workers different node IDs across a one-second boundary. Use mtime=0
  for fixed gzip fixtures and verify collection under multiple workers.

- Do not recommend a training identity or split key from its name or docstring.
  Reproduce its uniqueness across distinct donors, samples and alternate protein
  mappings first. #614 shows six donor observations sharing two evidence_row_id
  values; using that key alone can erase genotype-specific observations. #616
  shows that paper-level splitting also fails when experiments are reused across
  papers. Training readiness requires observation-identity and dataset-lineage
  audits in addition to peptide overlap checks. Preserve assay endpoint and unit
  when selecting binding labels; a numeric value alone can be a half-life in
  minutes, not an affinity (#615).

## 2026-09-28

- Keep closing keywords out of incidental cross-references. Writing "memory
  fix #610" in #609's description made GitHub close the still-unmerged memory
  PR when #609 merged. Use "memory PR #610" or "Refs #610" unless automatic
  closure is intentional, and inspect the closing event's `closer` before
  attributing an unexpected closure to a person.

- Closing a memory investigation does not close the memory problem. Report
  correctness, runtime, live allocation and Linux peak RSS separately; do not
  call a difference smaller than measured run-to-run variation a reduction.
  Keep the capacity issue open until comparable measurements establish useful
  headroom on the runner that is actually failing.

## 2026-09-27

- A rule is only as good as the remedy it prescribes. Check what it tells
  someone to do, not only what it catches.
  Rule: one PR shipped this trap six times. (1) A sibling-agreement check
  compared whole `condition_mhc_context` cells, so completing the `soluble_mhc`
  annotation the same PR had just filed as #588 would have made
  `load_pmid_overrides()` raise for the entire package. (2) Moved to the load
  path, it rejected ordinary designs -- an untreated wild-type arm beside a
  treated knockout -- because the engineering columns had left the material key,
  so nothing distinguished the two lines. (3) Its finding text told curators to
  write `none` into `condition_mhc_context`, the one engineering column
  `NONE_PERMITTED_CONDITION_COLUMNS` excludes. (4) It also said "say so in
  `sample_group`", which is all-or-none per study and rejects a one-to-one
  group/arm mapping, so a two-arm study fails either way. (5)+(6) #593 and #588
  kept prescribing superseded values in their *bodies* after the corrections
  landed only in comments. Every instance passed its own tests, because tests
  assert what a rule catches and say nothing about what it forbids. Two fixes:
  run each prescribed remedy through the real loader in a committed test -- that
  closed instances 1-4 as a class -- and re-ask the question of every issue and
  PR body after any behaviour change, which is the half no test reaches.
  A reader lands on the body, not on the correction below it.

- Show a new test failing before you trust it, and put the guard where a
  false positive costs the author rather than every consumer.
  Rule: a build-smoke assertion added to prove "the build creates its own data
  directory" passed with the entire fix reverted, because `register()` creates
  that directory two lines earlier; the mutation proof reported for it cannot
  have run. A partition test asserting `TREATMENT = INTERVENTION - ENGINEERING`
  is disjoint from `ENGINEERING` was true by construction and could never fail.
  A test that cannot be shown to fail is not evidence, and the check is cheap:
  break the thing, watch the named test go red, restore. Placement is the same
  judgement one level up -- a curation rule inside `load_pmid_overrides()` makes
  the installed package unimportable for everyone on a false positive, to catch
  an oversight worth one expression tier; the same rule as a `qc` audit plus a
  corpus test fails CI for whoever makes the edit, who can adjudicate it. Prefer
  the guard whose false positives land on the person able to judge them.

- A signal about the work decays on someone else's action. Re-check its
  subject before quoting it.
  Rule: four shapes of the same failure in one batch, all of which look like
  success. A CI monitor pinned to a commit kept reporting `unit (3.12): success`
  after the tip moved -- twice by its author's own push, once because *I* pushed
  a version bump onto its head, so "stop the monitors I superseded" is not
  enough; ask the API what the tip is now. A regenerated baseline that ate a
  newline was unparseable YAML, which is a file that looks like a ratchet. An
  issue body still prescribing a value the data had moved past. And a diff whose
  `-`/`+` pair was a block *moved* between jobs, which I read as a change and
  reported as a regression that did not exist -- settled only by reading the
  file at `origin/main`. Ask what a given green thing is asserting, and against
  which version of the subject.


- "Unreachable" is only as good as the condition it was measured under.
  Rule: a review said the class-pool `_select_by_elution_conditions` branch was
  unreachable after statement narrowing, and it was -- for a *mapped* statement.
  For an unmapped one it is IEDB's generic "untreated X; treated X" resolver,
  which needs no curated map at all, and deleting it cost PMID 32938616 and
  33968037 46,247 arms between them. "Unreachable after the change I am about
  to make" is not "unreachable". Before deleting a branch, name the inputs that
  reach it and check the ones the claim did not cover -- here, every study
  without a curated map, which is 2,313 of 2,319.

- A comparison that reports "everything changed" or "nothing changed" is
  broken until proven otherwise.
  Rule: the corpus replay was wrong two ways before it was right. First it
  compared rows positionally, and row order is not stable between exporter
  runs, so it silently compared unrelated rows -- fixed by joining on a
  7-column key and asserting the key is unique on both sides. Then it reported
  all 4,398,346 rows changed, because `NaN != NaN` and two pass-through columns
  are mostly null. Both produced a confident, plausible-looking number. The
  check that catches this class is cheap: a diff must be able to show one row
  it correctly calls unchanged *and* one it correctly calls changed before any
  of its totals are quoted. Same failure mode as a green test that asserts
  nothing. Twice before, the same measurement was wrong for reasons unrelated
  to the code under test (`sys.path[0]` importing the installed package, a
  concurrent agent overwriting the script) -- so the guard belongs in the
  script: state which build was loaded, assert it, and print a per-invocation
  OK line that gets checked.

- Widen the population before believing a "no impact elsewhere" claim.
  Rule: three review rounds of this PR measured 8 studies of 2,319 and each
  reported "only the three targets change". The fourth measured all 2,319 and
  found two more, because the change touched a branch every study runs through.
  The eight were the studies I had reasoned were relevant, which is exactly the
  set that cannot falsify the reasoning. If a change touches shared code, the
  population is everything that executes it, not the part the change was aimed
  at.

- Attribute a CI slowdown before shaving anything.
  Rule: `test (3.11)` cancelled twice at its 25-minute cap and the tempting fix
  was to trim the tests I had added. The measurements said otherwise: on the
  same pair of commits 3.9 went 5m22s to 11m52s while 3.12 got *faster*, the
  exporter took 3.2s on both heads over an identical frame, and the local suite
  moved +4% against CI's +46%. No code change can make one leg twice as slow
  and another faster; that shape is the runner. The guard added in that PR cost
  1.57s against a 63s shared fixture, so shaving it would have bought nothing
  and hidden a real capacity problem.

## 2026-09-23

- Give silent data-corruption fixes an independently shippable path. Keep
  only their real prerequisites in front of them; unrelated feature PRs
  should not delay genotype correctness. A tested draft is not a shipped
  fix, and status reports must distinguish those states explicitly.

## 2026-09-22

- Distinguish sequencing-library protocol from quantification mode and
  actual matrix coverage. DepMap 24Q4 marks HAP1's library as stranded but
  includes its values in the RSEM unstranded-mode gene matrix. Check the
  downloaded row IDs before claiming absence or filtering profiles by a
  metadata flag; retain the source's protocol and processing provenance.

- Respect biological units when diagnosing predictor input limits. MHCflurry's
  presentation predictor accepts a genotype of at most six class-I alleles;
  peptide/allele pairs and a pooled study allele union are not that genotype.
  Verify the caller's biological sample and the upstream API before deciding
  that a limit should be relaxed. Keep queried sample alleles separate from
  reported observational restrictions, and preserve both provenances.

## 2026-09-09

- When asked to fix all review findings and ship, resolve the known failing gate
  as well as the PR-specific defect. A timing test that passes only in isolation
  still needs a deterministic contract before relying on it for deployment.
- Verify claimed zero impact against the stored observation and binding vocabularies,
  not only curated sample tokens. #463 already changed 27 observation and 464 binding
  rows when checked across both local artifacts, despite its initial zero-impact claim.
- Use an isolated environment from the repository lockfile for release validation.
  The shared virtualenv changed from mhcgnomes 3.64.2 to unsupported 3.33.4 during
  this run, turning previously passing checks into misleading API/data failures.

- When the user needs condition categoricals on dataset rows, design the shared table columns
  before proposing an object hierarchy. Keep authoring flat, use the same column names for
  curation and exports, and add nesting only when a concrete source cannot be represented
  faithfully in the table. Verify paper-to-column extraction and row attribution separately.
- Two guards that look like the same question can need different answers. The arm *tie*
  guard and the narrative *admission* gate both asked "same arm?", so tightening both to
  `condition_id` looked consistent — and cost 7,629 correct discriminations, because the
  gate is really asking "do these differ by treatment?", where the coarse bucket is the
  right test. Before reusing a predicate, say out loud what each caller is asking.
- Measure a behavioural change against the base commit before believing it. A `git worktree`
  on the base plus one probe script turned "184,811 rows changed" from a worry into a
  reviewable table, and it is what surfaced that 204,668 rows were going *blank* rather
  than to `pmid_ambiguous` — a pre-existing asymmetry my change was quietly worsening.
- When a fix is right but its blast radius belongs to another change, measure the radius
  and put the number in the issue. #451 is a one-word diff (`elif _grouped:` -> `else:`)
  that moves 1.23M rows and obliges a per-study verdict pass; "1.23M rows" is what makes
  that obvious to the next reader, and guessing would have made it look like a drive-by.

## 2026-09-04

- If a release script is known to require network or unsandboxed build isolation, request that
  access on its first invocation. An approved command prefix does not make restricted-network DNS
  available; discovering that only after a long integration suite needlessly repeats the entire
  release gate.

- Never pass multiline Markdown containing backticks or apostrophes as an inline shell argument.
  Rule: create GitHub issue/PR/comment bodies with `apply_patch` in a temporary file and pass them
  with `--body-file`. Shell quoting is too easy to terminate early, after which Markdown backticks
  execute as commands and the outward-facing update either fails or becomes corrupted.

## 2026-09-03

- Candidate expansion and reported precision are different data and must not share one field.
  Rule: an expanded serotype member may be used as an internal join key, but any fallback or
  public result must still carry the source's serotype designation separately; never serialize
  inferred members into a field that downstream code reparses as exact reported typing.

- Once a domain parser accepts a spelling, serialize the parsed object instead of re-normalizing
  the raw token through a narrower helper.
  Rule: `normalize_allele()` intentionally canonicalizes molecules only, so a parsed Serotype must
  use `parsed.to_string()` before catalog lookup. Test case, prefix, and bare-name variants for
  every public input path.

- Precision-aware matching must be symmetric across both sides of a structured MHC restriction.
  Rule: when sample candidates can be single chains but observations can be full class-II pairs,
  regression-test the sample-to-observation join and the downstream summary separately. Expand a
  full observation to eligible single-chain sample typings, but never equate two fully known pairs
  merely because they share one chain.

- "Unknown" means no relevant typing exists, not that one representation-specific set is empty.
  Rule: before labeling support `unknown_allele`, check every known typing precision (exact allele,
  serotype, and any future typed designation). A nonmatching known serotype is exclusion evidence,
  not permission to include the row as unknown.

- A merged measurement is not an unperturbed sample just because one input arm was unperturbed.
  Rule: preserve experimentally distinct control/perturbation samples in curation even when the
  peptide artifact merges them; let the observation join emit unknown arm metadata unless the
  source provides a per-peptide discriminator.

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

- Edit source with a tool that fails on ambiguity, not with `str.replace(x, y, 1)`.
  Rule: four times in two days I did character-level surgery on 3000-line modules — `s = p.read_text()`, `s.index(anchor)`, `s.replace(old, new, 1)` — and four times the anchor was not unique, so the edit silently landed in the wrong function: the `pd.concat` fix went into `build_bulk_proteomics` instead of `build_line_expression` (with the wrong schema constant), an `output_cols` edit clobbered `discrepancies()` instead of `curation_plan()`, and two docstring inserts landed in unrelated functions. `replace(..., 1)` takes the *first* match and reports success either way. The Edit tool refuses a non-unique `old_string`, which turns every one of those into an error instead of a silent wrong edit. Use Edit for source changes; reserve scripted rewrites for genuinely mechanical, verified-unique substitutions. If a scripted edit is unavoidable, slice the target function out first and operate inside that span.

- Name the question, not the answers, when the same predicate is computed in several places.
  Rule: `qc.curation_plan` carried `has_borderline` / `has_implausible` — two booleans that actually meant "does the upstream frame contain this metric column?" — and threaded them through a string-keyed flag mapping and a `_metric_applies` helper with an unguarded `else`. Three call sites, three chances to drift, and a name that reads like a data verdict rather than a schema check. Replacing all of it with one `_available_optional_metrics(disc.columns) -> list[str]` removed the flags, the mapping, the helper and the failure mode together. When a boolean pair starts getting passed around, ask what question it answers and return that instead.

- The `str.replace` lesson applies to YAML data files too, and "scoped to a block" is the fix.
  Rule: I already had a lesson about `str.replace` landing in the wrong function, then repeated it on `pmid_overrides.yaml` — `s.replace("mhc: unknown", P4_genotype)` rewrote **12 samples across the whole file** when exactly one was in scope. The count was right there in the output and I only caught it because I printed replacement counts. Two things made the redo safe: bound the edit to the entry (`- pmid: N` .. next `- pmid:`) and assert the expected occurrence count inside that span. A data file has no compiler and no test that reads every entry, so a wrong edit here is quieter than a wrong edit in source. Print counts, assert them, scope the span.

- Verify an issue's premises before implementing its acceptance criteria.
  Rule: of four claims across #380/#381/#374, one was already fixed (HLA-G), one was mis-framed (the `I+II` samples were incomplete typing, not contradictions), and one had criteria that would have introduced a worse bug than the one it reported (#381's "curate the allele list observed in the corpus" pools eight animals and reports a NetMHCpan prediction as an observation). Issues are written from a snapshot and a partial read; they are evidence, not a spec. Check each claim against the code and the primary source first — it changed the scope of every one of the three.

- A helper is private only if it has no callers outside its own reasoning.
  Rule: the user asked why `_pmid_sample_alleles` was private and the honest answer was "no reason". It was already effectively public — all four exported `peptide_*_for_pmid` functions are thin wrappers over it — so callers got its output but could not ask for it directly, and the natural question ("what did this study type its samples to?") had no public answer. Before leaving an underscore on something, ask whether its result already escapes through a public function.

- When a curation defect has a mechanism, look for the invariant that detects the whole class.
  Rule: #381 reported one study whose `mhc` field pooled several animals. The mechanism — an `mhc` field holding a union across samples rather than one genotype — generalises, and the user asked for exactly that generalisation. The invariant turned out to be free of thresholds and of species: a diploid donor carries at most two alleles per locus, so three is proof of pooling. It found five more samples, all genuinely wrong, and it now also guards against "fixing" #381 the way the issue asked. One bug report plus a mechanism is often an audit waiting to be written.

- Never key a reverse map by an upstream table's own spelling. Assert that the
  whole vocabulary round-trips instead.
  Rule: #455 was a reverse map that used `mhcgnomes.data.serotypes`' allele
  strings as keys while the lookup built its own compact key from a parsed
  allele. 11 of 924 entries carry a different spelling — the hand-curated rows
  its generator cannot reproduce — so six serological specificities vanished
  from 41,478 rows with no error, no warning, and an empty tuple that reads
  exactly like "this allele has no serological equivalent". A `.get(key, ())`
  against data you do not control is a silent-wrong-answer machine. The test
  that belongs beside it is not "A*02:01 maps to A2" but "every entry in the
  source table is reachable", which fails on the next format change instead of
  discarding another locus.

- Two facts under one column name will be conflated by every consumer.
  Rule: `serotypes` held a measured serological typing (35,257 rows) and a
  computed membership (1,630,309 rows), spelled identically. The distinction
  was recoverable from `allele_resolution == "serological"`, so nothing was
  *lost* — but recoverable-by-inference is not the same as available, and
  tsarina duly built a filter that pooled the two. When a column's value can
  arrive either as primary data or as this library's projection, the provenance
  is part of the fact, not metadata about it: name it in its own column and put
  the library version that computed the projection in the artifact metadata.

- An "invariant" that only holds while the corpus is sparse is not an invariant.
  Rule: `test_species_summary_counts_are_coherent` asserted
  `n_peptides >= n_pmids` on the reasoning that each PMID contributes at least
  one peptide — true, and it implies no ordering, because two studies can report
  the same epitope. It passed for as long as no species happened to have a
  shared peptide, then a rebuild produced `Bos sp.` class I with 3 peptides
  across 4 PMIDs and the test failed with nothing wrong in the data. Since it is
  an `@pytest.mark.integration` test that skips when no index is built, CI never
  ran it: the failure surfaced only in `deploy.sh`, which gates on the local
  suite. Before asserting an ordering between two counts, name the operation
  that would violate it — here "two papers, one peptide" — and if it is ordinary
  science, the assertion is wrong rather than the data.

- When fixing test-framework deprecations, verify the oldest supported Python/pytest pair before propagating a stacked change. Class-scope fixtures that need no class state should be plain module functions; classmethod metadata differs before Python 3.10.

- In a change whose point is to separate two conflated facts, justify every
  blank field with the distinction the change draws — not the one it replaces.
  Rule: #520 splits "what the cell carries" from "what the experiment
  measured". Asked why two arms of PMID 35051231 carry no cellular typing, I
  answered "nothing was measured there" — which is the *old* conflation wearing
  the new field's name. The donors' genotypes are known and published; the
  reason blank is right is narrower and structural: `profiled: false` arms have
  no sample to type, and both donors' typing sits on their profiled arms in the
  same study, so no fact is lost. A rationale that would equally justify the
  bug you are fixing is not a rationale. Say which of the two facts is absent,
  and where the other one lives.

- Establish a benchmark's noise floor before you let it decide anything, and
  never size an object column with `memory_usage(deep=True)`.
  Rule: #566 cost me three retracted claims from two different measurement
  errors. (1) The integration suite's peak RSS measured the *unchanged*
  baseline at 16.25, 17.03, 20.38 and 17.57 GB — the xdist fixture mmaps an
  Arrow file and mmap'd pages count toward RSS — so a 2 GB "improvement" and a
  5 GB "regression" were both noise. Two runs of the baseline first would have
  cost eight minutes. (2) `memory_usage(deep=True)` reports 254 MB for two
  `object` columns holding `True`/`False`/`None`, and I quoted a 93% saving
  from narrowing them to nullable `boolean`. It charges ~28 bytes per element
  for what are *interned singletons*: 4.4M rows hold exactly 2 distinct object
  identities, the real cost is the 8-byte pointer array, and measured against
  `ps` the "fix" made steady-state RSS worse by 56 MB, because an `astype`
  adds the new array while the old array's pages are already resident. Size an
  object column by the RSS delta of a process that builds it, and remember
  that converting a column after the fact cannot return memory the allocator
  has already touched — only never materializing it can. Corollary from the
  review of that same PR: when you write down why an option was rejected,
  check the reason applies to the option. The read-time blocker I recorded
  says nothing about the build-time one, which is cheaper and untried.

- Measure the claim, or do not make it. Reasoning about a corpus is not
  evidence about it.
  Rule: across one session on #520/#565/#566/#564 every claim I measured
  survived review — 492 rows contradicting their deposited statement, 14,532
  wrongly excluded and 0 unions, 0 differing cells against main — and nearly
  every claim I reasoned to did not. A relayed "93% of rows" was 10%. A dtype
  "saving" of 254 MB was a 56 MB regression, because `memory_usage(deep=True)`
  charges per element for interned singletons. A 21→10 GB peak came from
  zero-copy buffers that broke 14 tests. A CI failure I attributed twice — once
  to a known flake, once to my own memory regression — was neither, and the
  cross-branch run history said so in one query. Two changes were justified by
  scenarios the same PR made unreachable. The pattern is specific: plausible
  mechanisms at corpus scale are worth exactly what an unmeasured mechanism is
  worth. Before writing a number in a PR body or a comment, run the thing.
