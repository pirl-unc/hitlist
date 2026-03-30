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

Some studies (e.g. Neidert biobank) have mixed sample sources. The `tissue_overrides` field allows per-tissue classification within a single study:

```yaml
- pmid: 29557506
  label: "Neidert 2018 — Tübingen/Zurich biobank"
  override: healthy
  tissue_overrides:
    Blood: healthy           # blood bank donors
    Bone Marrow: healthy     # hip arthroplasty
    Cerebellum: healthy      # autopsy
    Colon: adjacent          # Visceral Surgery → likely CRC margin
    Kidney: adjacent         # Urology → likely nephrectomy
    Liver: adjacent          # likely HCC/met adjacent
```

Tissue-level overrides take priority over the study-level override.

## Curated studies

### Gold-standard healthy tissue

| PMID | Study | Donors | Notes |
|---|---|---|---|
| 33858848 | Marcu 2021 (HLA Ligand Atlas) | 21 | 16 autopsy + 5 living thymus + 2 ovary, 29 tissues, 227 samples |
| 29557506 | Neidert 2018 (Tübingen biobank) | ~160 | Per-tissue overrides for surgical vs autopsy samples |
| 27862975 | Bassani-Sternberg 2017 | 3 | Soluble HLA from serum (weak evidence, sHLA bias) |
| 36589698 | CTA-specific TCR library | — | Healthy donor blood |
| 38920720 | Ovarian cancer antigen study | — | Healthy donor blood control |

### Reclassified studies

| PMID | Study | Override | Reason |
|---|---|---|---|
| 32983136 | Marino 2020 | `activated_apc` | Monocyte-derived DCs matured with LPS + IFN-gamma (~765 proteins upregulated) |
| 35051231 | Pyke 2022 | `adjacent` | Lung from surgery patients ("clinical reasons" = cancer resection) |
| 29786170 | TNBC immunopeptidomics | `adjacent` | Paired tumor + adjacent normal breast |
| 26992070 | Caron 2015 | `cancer_patient` | Melanoma patient serum + cell lines |
| 31154438 | GBM study | `cancer_patient` | GBM patient plasma + tumor tissue |
| 28514659 | HLA-B*46:01 study | `cell_line` | 721.221 transfectants wrongly annotated as spleen |

## Adding new overrides

Edit `hitlist/data/pmid_overrides.yaml`:

```yaml
- pmid: 12345678
  label: "Author Year — short description"
  override: adjacent  # or healthy, cancer_patient, activated_apc, cell_line
  note: "Why this override is needed."
  donors: 10  # optional
  tissue_overrides:  # optional per-tissue refinement
    Blood: healthy
    Liver: adjacent
```

No code changes needed. The YAML is loaded at runtime by `hitlist.curation.load_pmid_overrides()`.

## Per-donor analysis

The IEDB **Antigen Processing Comments** field (column 88) contains sample identifiers in some studies (e.g. "buffy coat 25", "colon 32"). hitlist preserves this field and the `hitlist.samples` module uses it for per-donor peptidome analysis.

Key finding from Neidert (PMID 29557506):
- 20/22 samples with CTA peptides have exactly 1 CTA peptide
- Zero MAGE co-expression in any sample
- Pattern rules out occult cancer → consistent with stochastic low-level expression
