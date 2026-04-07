# PMID Curation Overrides

hitlist applies expert per-study overrides to correct IEDB annotations that don't reflect the true biological context of each sample. Overrides are stored in `hitlist/data/pmid_overrides.yaml` as data, not hardcoded Python.

**34 studies curated** across 7 species, with per-sample MHC class, perturbation conditions, and estimated peptide counts.

## Species x MHC class summary

| Species | Class | Studies | Sample Types | Peptides |
|---------|-------|---------|-------------|----------|
| Bos taurus (cattle) | I | 1 | 1 | ~5,000 |
| Bos taurus (cattle) | II | 2 | 2 | ~5,000 |
| Canis lupus familiaris (dog) | I | 2 | 4 | ~7,742 |
| Carassius gibelio (Prussian carp) | II | 1 | 1 | ~276 |
| Gallus gallus (chicken) | I | 2 | 4 | ~1,064 |
| Gallus gallus (chicken) | II | 1 | 2 | ~1,000 |
| **Homo sapiens (human)** | **I** | **23** | **112** | **~745,000** |
| **Homo sapiens (human)** | **II** | **4** | **22** | **~261,000** |
| Macaca mulatta (rhesus macaque) | I | 1 | 6 | ~65 |
| Sus scrofa (pig) | I | 1 | 1 | ~200 |
| Sus scrofa (pig) | II | 1 | 3 | ~1,000 |

Regenerate this table: `hitlist export summary`

## Exporting data

```bash
hitlist export samples                    # all ms_samples as CSV
hitlist export samples --class I          # MHC class I only
hitlist export samples --class II -o c2.csv
hitlist export summary                    # species x class totals
hitlist export alleles                    # validate alleles with mhcgnomes
```

Each ms_sample entry includes: species, sample type, perturbation condition, PMID, study label, MHC class, estimated peptide count, and whether the count is from the paper or estimated.

## Override types

| Override | Meaning |
|---|---|
| `healthy` | Confirmed healthy tissue (default classification is correct) |
| `cancer_patient` | Reclassify all rows as cancer-derived |
| `adjacent` | Reclassify as tumor-adjacent normal tissue |
| `activated_apc` | Reclassify as activated APC artifact |
| `cell_line` | Reclassify as cell line (when IEDB annotation is wrong) |

## Per-tissue refinement

Some studies (e.g. Neidert biobank) have mixed sample sources. The `rules` field allows conditional classification within a single study:

```yaml
- pmid: 29557506
  label: "Neidert 2018 — Tübingen/Zurich biobank"
  override: healthy
  rules:
    - condition:
        Source Tissue:
          - Blood
          - Bone Marrow
          - Cerebellum
      override: healthy
      reason: "Blood bank donors and autopsy CNS material"
    - condition:
        Source Tissue:
          - Colon
          - Kidney
          - Liver
      override: adjacent
      reason: "Visceral Surgery — likely cancer resection margins"
```

Conditional rules are checked first (in order); the PMID-level override is the fallback.

## Per-sample metadata

Each study has an `ms_samples` list with structured per-sample-type entries:

```yaml
ms_samples:
  - type: "721.221-B*51:01 (WT)"
    n: 3
    condition: "unperturbed"
    mhc_class: "I"
    peptides: 3000
    peptides_estimated: true
  - type: "721.221-B*51:01 ERAP1 KO"
    n: 3
    condition: "ERAP1 CRISPR/Cas9 knockout"
    mhc_class: "I"
    peptides: 2500
    peptides_estimated: true
```

Fields: `type` (sample description), `n` (sample count), `source`, `condition` (perturbation or "unperturbed"), `mhc_class` ("I", "II", or "I+II"), `peptides` (unique peptide count), `peptides_estimated` (true if estimated rather than from paper).

## Perturbation categories

| Category | Studies |
|----------|---------|
| CRISPR gene KO (ERAP1, ERAP2, B2M, TAP1/2, TAPBP, IRF2, PDIA3, GANAB, SPPL3, CANX, CALR) | 31092671, 40113210 |
| shRNA knockdown (ERAP1) | 31092671 |
| DNMT inhibitor (decitabine 1 uM 72h) | 27412690 |
| TKI resistance (imatinib 1 uM) | 25576301 |
| IFN-gamma stimulation | 31844290 |
| DC differentiation + maturation (GM-CSF + IL-4 → LPS + IFN-gamma) | 32983136, 38920720 |
| T cell activation (PMA + Ionomycin) | 32983136 |
| B cell activation (IL-4 + CD40L) | 32983136 |
| Influenza A/H3N2 infection | 35051231 |
| Canine distemper virus (CDV) | 29475511 |
| Marek's disease virus (MDV) | 33901176 |
| PRRSV (porcine) | 36146698, 32796065 |
| *Theileria parva* (bovine) | 36423003 |
| CyHV-2 (fish) | 41459947 |
| Synthetic peptide pulsing | 38920720 |
| HLA-DM editing (dm+/dm-) | 31495665 |
| SILAC cross-presentation | 31495665 |
| CMV pp65 transgene | 23481700 |
| TIL expansion (IL-2 + OKT3) | 28832583 |
| Macrophage differentiation (PMA) | 35051231 |

## Peptide-to-protein mapping

Use `ProteomeIndex.from_ensembl_plus_fastas()` to map peptides back to source proteins with flanking context:

```python
from hitlist.proteome import ProteomeIndex

idx = ProteomeIndex.from_ensembl_plus_fastas(
    release=112,
    fastas=["influenza_a.fasta", "cmv.fasta"],  # viral proteomes
    lengths=(8, 25),  # peptide length range
)
hits = idx.map_peptides(peptides, flank=10)  # 10aa flanking on each side
```

This handles human + viral source proteins in a single index, returning position, N-terminal flank, C-terminal flank, gene name, and protein ID for each peptide.

## Adding new overrides

Edit `hitlist/data/pmid_overrides.yaml`:

```yaml
- pmid: 12345678
  label: "Author Year — short description"
  title: "Exact PubMed title"
  override: adjacent  # or healthy, cancer_patient, activated_apc, cell_line, ~
  note: "Why this override is needed."
  donors: 10
  hla_alleles:
    profiled: ["HLA-A*02:01"]
  perturbations:
    - "IFN-gamma stimulation (100 IU/mL, 48h)"
  ms_samples:
    - type: "sample description"
      n: 3
      condition: "unperturbed"
      mhc_class: "I"
      peptides: 3000
      peptides_estimated: true
  rules:
    - condition:
        Culture Condition: "Direct Ex Vivo"
      override: healthy
      reason: "Healthy donor controls"
```

No code changes needed. The YAML is loaded at runtime by `hitlist.curation.load_pmid_overrides()`.

## MHC allele validation

All 233 MHC alleles in the YAML parse correctly with mhcgnomes 3.20.0. Validate with:

```bash
hitlist export alleles
```

## Per-donor analysis

The IEDB **Antigen Processing Comments** field (column 88) contains sample identifiers in some studies (e.g. "buffy coat 25", "colon 32"). hitlist preserves this field and the `hitlist.samples` module uses it for per-donor peptidome analysis.

Key finding from Neidert (PMID 29557506):
- 20/22 samples with CTA peptides have exactly 1 CTA peptide
- Zero MAGE co-expression in any sample
- Pattern rules out occult cancer → consistent with stochastic low-level expression
