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
