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
