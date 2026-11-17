# Reconcile `mhcgnomes >= 3.70.2` Contract and `map_peptides` Empty-Result Flank Schema

Issues: `#897`, `#868`
Baseline: `43e226621434a483141764ed54faba725bbb3869` (`hitlist 2.1.9`, published `mhcgnomes 3.70.3`, `pandas 3.1.1`)

## 1. Problem Statement and Measured Baseline

On clean `main` (`43e2266`) in an isolated Python 3.12.10 environment with published
`mhcgnomes 3.70.3` and `pandas 3.1.1`, 89 unit tests fail across `test_curation.py`,
`test_allele_identity.py`, `test_allele_candidates.py`, `test_export.py`,
`test_predict.py`, `test_reassign_genotypes.py`, and `test_source_verification.py`:

1. **Deprecated `Allele.annotation_cystosolic` access (77 failures across `test_predict.py` and `test_reassign_genotypes.py`)**:
   - `hitlist/curation.py:_NON_SURFACE_ANNOTATION_ATTRS` accesses `annotation_cystosolic`.
   - `mhcgnomes >= 3.70.2` deprecates `annotation_cystosolic` in favor of `annotation_cytosolic` (`stacklevel=2`), which `pyproject.toml`'s `error::DeprecationWarning:(hitlist|tests)\..*` treats as an error.

2. **WHO split-versus-broad serotype ordering and catalog expansion (7 failures across `test_curation.py`, `test_allele_identity.py`, `test_export.py`)**:
   - `mhcgnomes >= 3.70.2` expands `direct_serotypes` (`serotypes.yaml`) through `serotype_splits.yaml` (`mhcgnomes.serotype_hierarchy.expand_serotype_memberships`), populating WHO broad parent antigens alongside their splits (`A9` for `A24`, `B17` for `B57`, `B12` for `B44`, `B15` for `B75`, `DR2` for `DR15`, `DQ3` for `DQ8`, `Cw3` for `Cw10`).
   - `_build_allele_to_serotypes_map()` ranked serotypes by `(_serotype_specificity_rank(n), not_current, len(n), n)`, causing shorter or alphabetically earlier broad parents (`HLA-A9`, `HLA-B17`, `HLA-B12`, `HLA-B15`, `HLA-DR2`) to precede their specific split antigens (`HLA-A24`, `HLA-B57`, `HLA-B44`, `HLA-B75`, `HLA-DR15`), and ranking `Cw*`/`DPw*` serotypes alongside public epitopes (`Bw4`, `Bw6`).

3. **Colon-significant IMGT/HLA nomenclature for `HLA-B*15:112` versus `HLA-B*15:11` and stored serotype refresh (2 failures in `test_allele_identity.py`)**:
   - In `mhcgnomes`'s `serotypes.yaml`, 2-field IMGT/HLA assignments omit colons (`A*02105` -> `HLA-A*02:105`, `B*15102` -> `HLA-B*15:102`, `B*15112` -> `HLA-B*15:112`, `B*15114` -> `HLA-B*15:114`). Modern `HLA-B*15:112` (formerly `B*9512`) is the 112th `HLA-B*15` allele, assigned to broad `HLA-B15` alone (not `HLA-B75`), whereas `HLA-B*15:11` belongs to `HLA-B75` and broad `HLA-B15`.
   - `mhcgnomes >= 3.70.2` preserves field boundaries in `get_allele_alias`, so `HLA-B*15:112` is no longer erroneously conflated with pre-2010 compact `B*15112` (`B*15:11:02` / `HLA-B*15:11`).
   - `observations.py` refreshed stored `serotypes` only when `serotype_requires_alias_refresh` fired, leaving pre-3.70 parquets stale when `mhcgnomes` expanded WHO broad memberships (`HLA-B*15:11` stored as `"HLA-B75"` instead of `"HLA-B75;HLA-B15"`).

4. **Mixed/descriptive suffixes in `Host | MHC Types Present` slots (1 failure in `test_allele_candidates.py`)**:
   - After `pirl-unc/mhcgnomes#200`, `parse("HLA-DRB1*12:01 class II")` raises `ParseError` instead of swallowing `"class II"`. `_mhc_field_scan` then split the slot on whitespace into `[("HLA-DRB1*12:01", Allele), ("class II", MhcClass)]`, and `_parse_host_mhc_types` -> `_exact_typing_tokens` dropped the `MhcClass` span and treated `"HLA-DRB1*12:01"` as an exact candidate. Across all 881 distinct non-empty `host_mhc_types` values in `ci-corpus-v4`, legitimate `;`-delimited slots contain only exact allele spans with zero leftover or imprecise spans.

5. **Observation-to-sample `(pmid, allele)` join symmetry for newly catalogued aliases (`Gaga-BF2*13:01` -> `Gaga-BF2*004:01`, 1 failure in `test_source_verification.py`)**:
   - `mhcgnomes >= 3.70.2` resolves chicken `Gaga-BF2*13:01` to IPD-MHC `Gaga-BF2*004:01`. `sample_mhc_candidates` indexed curated arms under `resolve_allele_identity` (`Gaga-BF2*004:01`), while `export.generate_observations_table` indexed observations by their preserved reported `mhc_restriction` (`Gaga-BF2*13:01`), causing Stage 2 `allele_exact` lookup to miss and fall through to Stage 3b `discriminated`.

6. **Inconsistent `ProteomeIndex.map_peptides` flank columns on empty/unmatched queries (`#868`)**:
   - Populated `map_peptides` frames emit `unique_n_flank` and `unique_c_flank`; empty/unmatched returns and the docstring used `unique_nflank` and `unique_cflank` (and the docstring omitted `transcript_id` and `is_canonical_transcript`).

## 2. Design & Invariants

1. **Cytosolic annotation & dependency floor**:
   - Use `"annotation_cytosolic" if hasattr(Allele, "annotation_cytosolic") else "annotation_cystosolic"` on the `Allele` class in `_NON_SURFACE_ANNOTATION_ATTRS`.
   - Raise `pyproject.toml` dependency floor to `mhcgnomes>=3.70.2` so local environments and CI share the identical WMDA broad/split serotype graph and alias table.
2. **Serotype specificity ordering (`_build_allele_to_serotypes_map`)**:
   - Recognize `Cw`, `Aw`, and `DPw` in `_LOCUS_SEROTYPE_RE` (`^(A|Aw|B|C|Cw|DR|DQ|DP|DPw|DM|DO)\d`).
   - Use `mhcgnomes.serotype_hierarchy.serotype_descendants(serotype_splits.get(prefix, {}))` so any broad antigen whose split descendant is also present on the allele ranks after its split (`HLA-A24` before `HLA-A9`, `HLA-B57` before `HLA-B17`, `HLA-B44` before `HLA-B12`, `HLA-B75` before `HLA-B15`, `HLA-DR15` before `HLA-DR2`, `HLA-Cw10` before `HLA-Cw3`), while retaining valid broad memberships in `allele_to_all_serotypes`, keeping `HLA-A2` before IEF sub-specificity `HLA-A2.1` (`"." in name`), keeping `HLA-A24` before 4-digit `HLA-A2403`, and keeping public epitopes (`Bw4`, `Bw6`) last.
3. **Colon-significant `HLA-B*15:112` vs `HLA-B*15:11` and true retired aliases (`HLA-B*44:01` -> `HLA-B*44:02`)**:
   - Keep modern colon-separated `HLA-B*15:112` distinct from `HLA-B*15:11` (`allele_to_all_serotypes("HLA-B*15:112") == ("HLA-B15",)`, `allele_to_all_serotypes("HLA-B*15:11") == ("HLA-B75", "HLA-B15")`), while genuine retired catalog designations (`HLA-B*44:01` -> `HLA-B*44:02`) share full split+broad serotypes (`("HLA-B44", "HLA-B12", "HLA-Bw4")`).
4. **Stored serotype refresh (`observations.py`)**:
   - Refresh `serotype`, `serotypes`, and `serotype_source` from `mhc_restriction` whenever `resolve_mhc_annotation` differs from the stored values, preserving `category` dtype when incoming columns are categorical and leaving already-current columns untouched.
5. **Conservative `host_mhc_types` slot parsing**:
   - Require each `;`-delimited slot of `host_mhc_types` to parse cleanly into exact allele/pair span(s) with no unparsed tokens and no accompanying class/imprecise sentinel spans (`"HLA-DRB1*12:01 class II"` -> rejected as exact candidate).
6. **Symmetric `(pmid, allele)` join in `export.generate_observations_table`**:
   - Register observed restriction spellings whose `resolve_allele_identity` matches a sample candidate component so both reported (`Gaga-BF2*13:01`) and canonical (`Gaga-BF2*004:01`) observation restrictions match via `allele_exact` without mutating `mhc_restriction` or `sample_mhc`.
7. **`ProteomeIndex.map_peptides` schema consistency (`#868`)**:
   - Define `_MAP_PEPTIDES_COLUMNS` once for both empty and populated returns (`unique_n_flank`, `unique_c_flank`), update docstring, and add regressions for `[]` and unmatched `["ZZZZZ"]` inputs.
