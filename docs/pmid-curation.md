# PMID Curation Overrides

hitlist applies expert per-study overrides to correct IEDB annotations that don't reflect the true biological context of each sample. Overrides are stored in `hitlist/data/pmid_overrides.yaml` as data, not hardcoded Python.

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

## Optional metadata fields

The YAML supports optional informational fields that do not affect classification:

| Field | Purpose |
|---|---|
| `hla_alleles` | HLA alleles profiled in the study |
| `perturbations` | Non-standard antigen processing conditions (gene KO, cytokines, infection, etc.) |

## Curated studies

### Gold-standard healthy tissue

| PMID | Study | Donors | Notes |
|---|---|---|---|
| 33858848 | Marcu 2021 (HLA Ligand Atlas) | 21 | 14 autopsy + 5 living thymus + 2 ovary, 29 tissues, 227 samples |
| 29557506 | Neidert 2018 (Tübingen biobank) | ~160 | Per-tissue overrides for surgical vs autopsy samples |
| 27862975 | Ritz 2017 (soluble HLA) | 3 | Soluble HLA from serum and plasma (weak evidence, sHLA bias) |

### Mixed studies (conditional rules)

| PMID | Study | Rules | Notes |
|---|---|---|---|
| 36589698 | de Rooij 2022 (CTA TCR library) | Ex vivo → healthy | Cell lines (U266, RPMI8226, UM9, C4-2B4) + ovarian tumors + healthy PBMCs |
| 38920720 | Hesnard 2024 (ovarian antigen) | Ex vivo → healthy | moDCs pulsed with synthetic peptides, healthy donor leukapheresis |
| 29786170 | Ternette 2018 (TNBC) | Cancer → cancer, ex vivo → adjacent | Paired tumor + adjacent normal breast, HLA-A*02:01 only |
| 26992070 | Ritz 2016 (HLA peptidome) | Healthy sera → healthy, disease sera → cancer | Cell lines + 8 melanoma sera + 4 healthy sera |
| 31154438 | Shraibman 2019 (GBM) | Healthy → healthy, AS → neutral | 10 GBM tumors, 106 GBM plasma, 6 healthy + 30 AS controls |

### Reclassified studies

| PMID | Study | Override | Reason |
|---|---|---|---|
| 32983136 | Marino 2020 | `activated_apc` | Monocyte-derived DCs (LPS + IFN-gamma), T cells (PMA + Ionomycin), B cells (IL-4 + CD40L) |
| 35051231 | Nicholas 2022 | `adjacent` | Lung from surgery patients + influenza infection perturbation |
| 28514659 | Hilton 2017 | `cell_line` | 721.221 transfectants wrongly annotated as spleen |

### Mono-allelic 721.221 studies

| PMID | Study | Alleles | Notes |
|---|---|---|---|
| 28228285 | Abelin 2017 | 16 HLA-I | 721.221 transfectants, IEDB cell_name = "B cell" |
| 31844290 | Sarkizova 2020 | 95 HLA-I | 721.221 transfectants (79 new + 16 from Abelin), IEDB cell_name = "B cell" |
| 31092671 | Guasp 2019 | HLA-B*51:01 | 721.221 + ERAP1/ERAP2 CRISPR KOs, antigen processing perturbation |

## Adding new overrides

Edit `hitlist/data/pmid_overrides.yaml`:

```yaml
- pmid: 12345678
  label: "Author Year — short description"
  title: "Exact PubMed title"
  override: adjacent  # or healthy, cancer_patient, activated_apc, cell_line, ~
  note: "Why this override is needed."
  donors: 10  # optional
  hla_alleles:  # optional
    profiled: ["HLA-A*02:01"]
  perturbations:  # optional
    - "IFN-gamma stimulation (100 IU/mL, 48h)"
  rules:  # optional per-condition refinement
    - condition:
        Culture Condition: "Direct Ex Vivo"
      override: healthy
      reason: "Healthy donor controls"
```

No code changes needed. The YAML is loaded at runtime by `hitlist.curation.load_pmid_overrides()`.

## Per-donor analysis

The IEDB **Antigen Processing Comments** field (column 88) contains sample identifiers in some studies (e.g. "buffy coat 25", "colon 32"). hitlist preserves this field and the `hitlist.samples` module uses it for per-donor peptidome analysis.

Key finding from Neidert (PMID 29557506):
- 20/22 samples with CTA peptides have exactly 1 CTA peptide
- Zero MAGE co-expression in any sample
- Pattern rules out occult cancer → consistent with stochastic low-level expression
