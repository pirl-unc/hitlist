# Source Classification

Every IEDB/CEDAR mass spectrometry observation is classified by biological source context to distinguish cancer evidence from healthy tissue evidence.

## Categories

| Priority | Category | Flag | Rule | Safety impact |
|---|---|---|---|---|
| 1 | **Cancer** | `src_cancer` | Process Type = "Occurrence of cancer" OR non-EBV cell line | Positive targeting evidence |
| 2 | **Adjacent to tumor** | `src_adjacent_to_tumor` | Per-PMID override for resection margins | Ambiguous (may contain cancer) |
| 3 | **Activated APC** | `src_activated_apc` | Per-PMID override OR DC/macrophage from blood | Pharmacological artifact |
| 4 | **Healthy somatic** | `src_healthy_tissue` | Direct Ex Vivo + healthy + non-reproductive + non-thymic | **SAFETY SIGNAL** (off-target risk) |
| 5 | **Healthy thymus** | `src_healthy_thymus` | Direct Ex Vivo + healthy + thymus | Expected for CTAs (AIRE) |
| 6 | **Healthy reproductive** | `src_healthy_reproductive` | Direct Ex Vivo + healthy + reproductive tissue | Expected for CTAs |
| 7 | **EBV-LCL** | `src_ebv_lcl` | EBV-transformed B-cell line | Not cancer, not healthy |
| 8 | **Cell line** | `src_cell_line` | Any "Cell Line / Clone" culture condition | Treated as cancer-derived |

## Key rules

### All non-EBV cell lines are cancer-derived

Many commercial cancer cell lines (HeLa, THP-1, A549, HCT 116) appear in IEDB studies marked "No immunization" with disease "healthy". This is misleading -- they are cancer-derived lines. hitlist classifies **all non-EBV cell lines as cancer-derived** regardless of IEDB Process Type.

### Healthy requires Direct Ex Vivo

Only tissue taken directly from a donor (not cultured, not passaged) from a healthy individual with no disease qualifies as genuinely healthy. This is the strictest definition and produces the negative set used for off-target toxicity assessment.

### Host filtering, not epitope source filtering

The `human_only` parameter filters on the **Host** organism (the antigen-presenting cell), not the epitope source. This correctly retains viral peptides (HPV E7, EBV LMP1, etc.) presented on human MHC molecules.

### Thymus and reproductive tissue are separate

CTA expression in thymus (AIRE-mediated) and reproductive tissue (normal biological function) is expected. Finding a CTA peptide in these tissues does NOT make it unsafe to target.

## Cancer-specific definition

```
is_cancer_specific = found_in_cancer AND NOT found_in_healthy_tissue
```

The following do NOT disqualify a peptide from being cancer-specific:
- Presence on thymus
- Presence on reproductive tissue
- Presence on tumor-adjacent tissue
- Presence on EBV-LCLs
- Presence on activated APCs

## IEDB columns used

| Our column | IEDB field | Index | Purpose |
|---|---|---|---|
| `process_type` | Process Type | 50 | Cancer vs healthy |
| `disease` | Disease | 51 | Specific disease |
| `culture_condition` | Culture Condition | 106 | Ex vivo / cell line / EBV-LCL |
| `source_tissue` | Source Tissue | 102 | Anatomical origin |
| `cell_name` | Cell Name | 104 | Named cell line or cell type |
| `host` | Host | 43 | APC organism (for human_only filter) |
| `mhc_restriction` | MHC Restriction Name | 107 | HLA allele |
| `mhc_class` | MHC Allele Class | 111 | Class I or II |

Column indices are resolved dynamically from CSV headers with hardcoded fallbacks.
