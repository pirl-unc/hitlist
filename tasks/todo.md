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
