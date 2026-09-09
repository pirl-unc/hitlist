# PMID curation overrides

hitlist applies expert per-study overrides to correct IEDB/CEDAR annotations that
don't reflect the true biological context of a sample, and to attach the
per-sample metadata (HLA genotype, perturbation, instrument) that MS pipelines
need. Overrides live in `hitlist/data/pmid_overrides.yaml` as data, loaded at
runtime by `curation.load_pmid_overrides()` — no code changes to add a study.

Each entry is keyed by `pmid` (or `submission_id` for unpublished IEDB
submissions) and curated against the paper's Methods. The file currently covers
**159 PMIDs**, accounting for ~89.5% of all observations.

## Entry schema

A representative entry:

```yaml
- pmid: 31495665
  study_label: "Author 2019 — short description"
  title: "Exact PubMed title"
  override: cell_line          # study-wide default classification (optional)
  note: "Why curation is needed — cite the Methods."
  source_organism: "Homo sapiens"   # provenance fills (only used where IEDB is blank)
  donors: 4
  hla_alleles:                 # alleles profiled in the study (allele pool)
    - "HLA-A*02:01"
    - "HLA-B*07:02"
  perturbations:
    - "HLA-DM editing (dm+/dm-)"
  ms_samples:
    - sample_label: "721.221-B*51:01 (WT)"
      n_samples: 3
      mhc: "HLA-B*51:01"
      mhc_class: "I"
      condition: "unperturbed"
    - sample_label: "721.221-B*51:01 ERAP1 KO"
      n_samples: 3
      mhc: "HLA-B*51:01"
      mhc_class: "I"
      condition: "ERAP1 CRISPR/Cas9 knockout"
  rules:
    - condition: { Source Tissue: "Blood" }
      override: healthy
      reason: "Blood-bank donors"
```

### Top-level keys

| Key | Purpose |
|---|---|
| `pmid` / `submission_id` | The entry key (int PMID, or string submission id). |
| `study_label`, `title`, `note` | Human-facing provenance. (`label:` is the **deprecated** name for `study_label`.) |
| `override` | Study-wide default classification — see vocabulary below. |
| `rules` | Conditional, per-row overrides (checked before `override`). |
| `source_organism`, `species`, `source_tissue`, `cell_name`, `disease`, `culture_condition` | **Provenance fills** — used only where the IEDB row is blank/`Other`/`unknown`; never overwrite real IEDB data. |
| `hla_alleles` | Alleles profiled in the study; the fallback pool for class-only allele expansion. |
| `mono_allelic_host` | HLA-null/low host name (must exist in `monoallelic_lines.yaml`); flags resolved-allele rows mono-allelic. |
| `mono_allelic_method` | Tagged-pulldown mono-allelic method (e.g. MAPTAC) — not a cell line. |
| `ms_samples` | Per-sample-type metadata (below). |
| `peptide_attributions` | Path to a CSV mapping `peptide` → `sample_label` for per-donor attribution. |
| `exclude_from_ms` | **Not honored — see [#444](https://github.com/pirl-unc/hitlist/issues/444).** Intended to exclude a study from the MS index, and set on 11 studies curated as non-MS (yeast display, microarray, refolding, computational). No code reads it, so their 40,355 rows are in the corpus. |
| `donors` | **Not honored.** Curated on 11 studies, read by nothing (#444). |
| `n_samples`, `n_tissues` | Informational counts. Read by nothing, but named per the count-suffix rule; they were bare `samples:` / `tissues:` until the study-level guard went in. |
| `ip_antibody`, `acquisition_mode`, `instrument`, `fragmentation`, `labeling`, `search_engine`, `fdr`, `quantification_method` | MS-acquisition metadata (study-wide defaults; overridable per `ms_samples` entry). |
| `perturbations` | Non-standard processing (gene KO, cytokines, infection, …). |

Every top-level key is declared in `curation.PMID_ENTRY_FIELDS`, mapped to what
reads it, and **loading rejects an undeclared key** — the study-level twin of the
`MS_SAMPLE_FIELDS` guard. Its absence is why four keys above ended up curated
with no reader. A key that is accepted but unread says so in its description
rather than describing behavior it does not have.

### `override` vocabulary

| Override | Effect |
|---|---|
| `cell_line` | Force cell-line → cancer-derived (unless EBV-LCL). |
| `ebv_lcl` | Force EBV-LCL; not cancer. |
| `noncancer_cell_line` | Force cell line **without** `src_cancer` (rare non-malignant lines). |
| `cancer_patient` | Reclassify all rows as cancer-derived. |
| `adjacent` | Reclassify as tumor-adjacent normal tissue. |
| `healthy` | Confirmed healthy tissue (force the healthy path). |
| `activated_apc` | Reclassify as activated-APC artifact. |

### `ms_samples` fields

| Field | Meaning |
|---|---|
| `sample_label` | Sample description. (`type:` is the **deprecated** name.) |
| `n_samples` | Number of samples/replicates. (Use the `_samples` suffix — never a bare `n`.) |
| `mhc` | Donor genotype (`HLA-A*…` or a space-joined allele list). |
| `mhc_class` | `"I"`, `"II"`, `"I+II"`, or `"non-classical"`. Use `non-classical` for class Ib / MHC-Ib molecules — HLA-E, HLA-F, HLA-G, MR1, CD1, H2-Q — so `--class I` does not return them. A declared class that contradicts the sample's own alleles fails CI. |
| `condition` | Perturbation or `"unperturbed"`. |
| `classification`, `reason` | Per-sample classification note + rationale. Exported as their own columns, and jointly as the legacy `notes` (classification if present, else reason). |
| `override` | Per-sample provenance override. Present with a value = this sample's claim; present but **null** = "deliberately none here", which is *not* the same as omitting the key and inheriting the study's. See below. |
| `note` | Free-text analytic caveat about this arm, exported and carried to attributed observations. |
| `source`, `species`, `reference_proteomes` | Per-sample provenance. |
| `sample_group` | The sample **system** this arm belongs to — a cell line, tissue, or donor cohort (#359). Attribution resolves the system first, then the arm within it. **Opt-in per study and all-or-none**: curating it on a subset raises at load, as does a one-to-one group/arm mapping. See below. |
| `profiled` | `false` (or `n_samples: 0`) for an arm that exists in the paper but was never run on the instrument. It is exported as a metadata row and excluded from observation attribution, so it can never be matched to a peptide. |

Every key an `ms_samples` entry may carry is declared in
`curation.MS_SAMPLE_FIELDS`, mapped to what reads it. **Loading rejects an
undeclared key.** Adding a field means adding it there together with its
reader — otherwise it looks exactly like a field that works while reaching no
consumer, which is how `override`, `note`, and `species` sat unread (#373).

### `sample_group` — system before arm

The arm scorer reads `sample_label` and `perturbation` as one bag of tokens,
so it cannot tell a *system* descriptor from a *condition* one. Two things
followed from that, both observed in the corpus:

- Extra identifying words on one arm of a pair decided the pair. PMID
  29242379's untreated UWB arm alone carried `(ovarian carcinoma)`, and
  `ovarian` matching `source_tissue = "Ovary"` took all 3,919 of its rows —
  with nothing about treatment in evidence.
- A system whose arm is unambiguous was not attributable either, because the
  narrative fields are withheld wholesale whenever candidate arms disagree.

`sample_group` splits the question. The **system stage** admits IEDB's
narrative fields — naming a system is what they do reliably, and they say
nothing about treatment. The **arm stage** keeps blocking them. When arms of
a known system tie, the row reports `sample_attribution = "group_ambiguous"`:
system known, arm withheld.

Three rules the loader enforces, because each failure is silent otherwise:

| rule | why |
|---|---|
| all arms of a study carry it, or none | a partially grouped study falls back to the ungrouped path, losing the grouping with no error |
| a group must not hold exactly one arm when there are several groups | a 1:1 group/arm mapping makes the system stage select an *arm* from narrative text, which is what #354 forbids |
| every arm of a multi-arm group carries its group name | otherwise an arm with extra identifying words wins on them alone — the original bug |

Do not group a study with only one system: naming it says no more than naming
the PMID, and the stage declines in that case rather than relabelling rows.

### How `override` resolves

Three columns keep the levels distinguishable rather than collapsing them to
one value:

| sample YAML | `sample_override` | `effective_override` | `effective_override_origin` |
|---|---|---|---|
| `override: cell_line` | `cell_line` | `cell_line` | `sample` |
| `override:` (null) | `""` | `""` | `sample_null` |
| key omitted, study has one, study has **no** `rules` | `""` | the study's value | `study` |
| key omitted, study has one **and** has `rules` | `""` | the study's value | `study_conditional` |
| key omitted, study has none | `""` | `""` | `none` |

`sample_null` and `none` produce the same value and mean different things: the
first records that a curator considered this arm and decided against an
override, the second that nobody did. PMID 34497125 is the shape this exists
for — two `cell_line` cell-line arms beside a patient-biopsy arm explicitly
marked null.

`study_conditional` exists because `rules` match **per observation row**, not
per sample, so whether one supersedes the study default is not knowable from
the sample. PMID 27846572 inherits `cell_line` while its rule sends every
Direct Ex Vivo fibroblast row to `healthy`; reporting plain `study` there would
assert a value the build contradicts on 3,614 rows.

These columns describe **the curation, and only the curation**. The
classification flags (`src_cancer`, `src_cell_line`, …) are computed at build
time by `classify_ms_row`, which never sees `ms_samples` — it applies `rules`
then the PMID-level `override`. So a sample-level `override: null` clears the
exported metadata value and changes no flag. If you need the override the build
actually applied to a row, read the `src_*` flags, not these columns.

On the observations export the free-text note is renamed `sample_note`, the way
`mhc` becomes `sample_mhc` — `note` is also a study-level YAML key and `notes`
is an adjacent column meaning classification-or-reason, so the bare name was
ambiguous between three things. `sample_override` is not carried there at all:
it is derivable from the other two and a redundant object column is expensive
across 4.4M rows.

## The `rules` mechanism

Many studies mix sample sources under one PMID (e.g. tumor tissue and adjacent
normal in the same submission). The `rules` list applies conditional overrides
**before** the study-wide `override`:

```yaml
rules:
  - condition:
      Source Tissue: [Blood, Bone Marrow, Cerebellum]
    override: healthy
    reason: "Blood-bank donors and autopsy CNS material"
  - condition:
      Source Tissue: [Colon, Kidney, Liver]
    override: adjacent
    reason: "Visceral Surgery / Urology — likely cancer resection margins"
```

Matching semantics (`_matches_condition`):

- **all** keys in a `condition` must match (AND); the first matching rule wins.
- A value may be a single string or a list (any-match).
- `Source Tissue`, `Cell Name`, `Culture Condition`, `Disease`, `Process Type`
  match by equality; `Assay Comments` matches by **case-insensitive substring**
  (IEDB concatenates per-arm provenance into one cell, e.g.
  `"eluted from CRC tissue. eluted from nonmalignant colon tissue."`), which lets
  one rule target a single arm of a multi-arm study.

If no rule matches, the study-wide `override` applies; if there's no override
either, the row falls through to structured-field classification.

## Per-donor attribution

Supplementary data sometimes provide per-peptide sample identifiers that IEDB
does not carry as structured fields. For studies with a `peptide_attributions`
CSV, hitlist emits one row per matched sample. A class-only source row narrows to
that sample's typing and records the corresponding `mhc_allele_provenance`
(`exact` / `peptide_attribution` / `sample_allele_match` / `pmid_class_pool` /
`unmatched`). An already allele-resolved source row keeps its reported
restriction and provenance; the mapping adds only `attributed_sample_label`.
This preserves independent facts such as a predicted restriction that differs
from the sample's measured genotype.

## Adding a new override

**One entry per PMID.** The loader keys its mapping by PMID and rejects a file
that declares one twice. To add a field to an already-curated study — a
provenance fill, an acquisition field, another sample — edit that study's
existing entry. A second block for the same PMID used to replace the first
silently, which is how two studies and five sample records disappeared (#438).

1. Read the paper's Methods — confirm tissue, disease, cell lines, HLA typing,
   and any perturbation. Don't trust the IEDB free-text fields blindly.
2. Add an entry to `pmid_overrides.yaml` with `study_label`, `title`, the
   appropriate `override`/`rules`, provenance fills for anything IEDB left blank,
   and `ms_samples` for per-sample structure.
3. If the study uses an HLA-null host or MAPTAC, set `mono_allelic_host` /
   `mono_allelic_method` (and add the host to `monoallelic_lines.yaml` if new).
4. Validate: `hitlist export alleles` parses every allele through mhcgnomes;
   `hitlist qc` flags normalization and cross-reference issues.

No code changes are needed — the YAML is loaded at runtime.

## The two APM levels (#353, #362)

Both reach the per-observation export, and they answer different questions:

| column | scope | question |
|---|---|---|
| `apm_genes_perturbed`, `apm_perturbed`, `condition_category` | **this sample's own arm** | was *this* peptide's arm perturbed? |
| `study_apm_genes`, `study_apm_perturbed` | **the parent deposit's panel** | did the study run a perturbation at all? |

The per-sample flags are derived only from the arm's own `condition`. Folding a
study's `perturbations` panel into them is what #353 fixed: the Shapiro HAP1
CRISPR panel made all 12 arms claim the same 11 genes, so `HAP1 wildtype`
reported `apm_erap1_perturbed=True` and per-gene filtering selected whole
studies instead of perturbed samples.

That matters most for the control arm. A model that featurizes the study-level
column as if it were per-sample inverts its own control, which cancels the
KO-vs-WT contrast rather than merely adding noise. `HAP1 wildtype` rows today
carry `apm_genes_perturbed=""` with `study_apm_perturbed=True`, and
`tests/test_sample_attribution_audit.py` pins that through the join.

## Arm resolution — why a row has no arm (#366)

584,966 observation rows sit at `pmid_ambiguous` or `group_ambiguous`. The
useful question is not how many, but which of them more curation could fix.
`arm_resolution` records that per study, once, so a study is not
re-investigated every time someone notices the number.

| verdict | rows | means |
|---|---|---|
| `axis_mismatch` | 255,179 | Curated arms and recorded metadata are on different axes. PMID 33858848 curates per **donor**; IEDB records per **tissue** (29 values) and never records donor. No matcher can bridge them. |
| `curation_gap` | 145,279 | A per-row discriminator is present and the arms are not yet curated to use it. **The only verdict marking real work.** |
| `no_row_discriminator` | 129,982 | Measured: `cell_name`, `source_tissue`, `antigen_processing_comments` and `assay_comments` each take exactly one distinct value across the study. |
| `multi_arm_evidence` | 54,526 | The evidence positively places the peptide in more than one arm — eluted from both the treated and untreated sample. Nothing is missing. |

Three quarters of the ambiguity is settled: it is a property of what was
deposited, not of how carefully anyone curated. `arm_resolution_note` carries
the measurement behind each verdict, and loading rejects a note without a
verdict — the note explains a judgement, it is not one.

`hitlist qc sample-attribution --actionable-only` hides findings in settled
studies. A `no_row_discriminator` verdict is a measurement, so a test
re-measures it: if a corpus refresh gives one of those studies a varying
per-row field, the verdict is stale and the study gets looked at again rather
than being silently trusted.

## Source-verified corrections (#436)

Four studies had curation that was internally consistent and wrong. Each
correction below is grounded in the primary source; the pattern is worth
reading before adding curation of your own, because every one of these
would have passed a plausibility check.

| PMID | Was | Is | Source |
|---|---|---|---|
| 34497125 | A375 ± trametinib | SKMEL5 ± binimetinib (100 nM, 72 h) vs DMSO | [10.1073/pnas.2111173118](https://doi.org/10.1073/pnas.2111173118); PXD024917 file names |
| 34129938 | one unperturbed MC38 arm | four idAdpgkG MC38 arms, all on 20 ng/ml IFN-γ | [10.1016/j.mcpro.2021.100108](https://doi.org/10.1016/j.mcpro.2021.100108) Methods; MSV000086582 |
| 39438697 | two THP-1 arms, unperturbed splenocytes | WT/TAP1-KO × mock/H37Rv, plus hMDMs and Alg8-pulsed splenocytes | [10.1038/s41596-024-01076-x](https://doi.org/10.1038/s41596-024-01076-x); the authors' `conditions_table_TAP.csv` |
| 27846572 | GR-LCL, JY, C1R, HeLa, fibroblasts | GR-LCL, C1R, T2, fibroblasts | [10.1126/science.aaf4384](https://doi.org/10.1126/science.aaf4384) |

Three failure modes recur:

**A plausible line substituted for the real one.** PMID 34497125 carried A375
and trametinib. The paper says SKMEL5 and binimetinib and mentions neither of
the others anywhere. The damage was not the label: the per-sample `mhc` was
A375's genotype, attached to a study that never used A375. When the correct
line's typing is not recorded — not in the paper, not in `cell_lines.yaml` —
the honest `mhc` is `HLA class I`, not a substitute genotype.

**A perturbation axis collapsed.** PMID 34129938's arms are all IFN-γ-treated
and differ by doxycycline induction and dTAG-13 degradation; one "unperturbed
MC38" entry asserted the opposite for all of them. PMID 39438697's four THP-1
conditions are a 2×2 of TAP1 knockout by Mtb infection; two arms cannot
express it. Both now reach `apm_genes_perturbed` (`ifn_gamma`, `tap1`), which
is the point — a collapsed axis is invisible to every downstream filter.

**A citation inverted.** PMID 27846572's C1R note sourced the data to
Bassani-Sternberg 2015. The paper cites C1R to reference 5 (Caron 2015) and
the *fibroblasts* to reference 6 (Bassani-Sternberg 2015) — exactly
backwards. The same study also carried JY and HeLa samples the paper never
mentions and for which the corpus holds no row, while T2, its TAP-deficient
control with 111 rows, was absent.

Two lessons for new curation:

1. **Check the evidence, not only the paper.** Liepe 2016's 18,664 rows fall
   into exactly four `cell_name` groups. That is what disproved JY and HeLa,
   and it is where the four-digit GR-LCL typing and C1R's missing `HLA-B*40:02`
   came from. A curated genotype that no row carries is a claim about nothing.
2. **State what the source states.** "All 10 biopsies were HLA-A*02:01-positive"
   is a selection criterion; putting that allele in `mhc` would read as a
   mono-allelic sample. It belongs in `source`.

`tests/test_curated_study_sources.py` pins each of these facts to the sentence
in the source that establishes it.

## Exporting curated metadata

```bash
hitlist export samples              # every ms_samples entry as CSV
hitlist export samples --class I    # MHC class I only
hitlist export summary              # species × class totals
hitlist export alleles              # validate alleles with mhcgnomes
```
