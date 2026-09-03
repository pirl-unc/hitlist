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
| 31495665 class II | 2 | 10 (one per allele) |

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
