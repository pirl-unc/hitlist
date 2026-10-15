# PXD013455 recovery record — 2026-10-15

## Primary sources

- Original experiments: Bekker-Jensen et al., *An Optimized Shotgun Strategy for
  the Rapid Generation of Comprehensive Human Proteomes* (2017),
  [PMID 28601559](https://pubmed.ncbi.nlm.nih.gov/28601559/),
  [DOI 10.1016/j.cels.2017.05.009](https://doi.org/10.1016/j.cels.2017.05.009),
  PXD004452. Europe PMC full text: PMC5493283.
- Reanalysis: Jarnuczak et al., *An integrated landscape of protein expression
  in human cancer* (2021), [PMID 33893311](https://pubmed.ncbi.nlm.nih.gov/33893311/),
  [DOI 10.1038/s41597-021-00890-2](https://www.nature.com/articles/s41597-021-00890-2),
  [PXD013455](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD013455).
- Deposited inputs: `https://ftp.pride.ebi.ac.uk/pride/data/archive/2019/07/PXD013455/`.
- Paper supplements recovered from Europe PMC's public `supplementaryFiles` API,
  after the publisher CDN could not resolve. Original bytes are retained via
  datacache; selected workbook members are datacache materializations.

## Recovered bytes

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `Human-ReferenceProteome-Canonical-Isoform-71591.fasta` | 46,989,754 | `88e76d6479481fe18f4420c1d5f3082b6d7bfbf4c091316cf42506a64471b9cc` |
| `mqpar-celllines.xml` | 609,198 | `df650867a840d0bd834a8130c00196c0454a244a00b1f5be3dac53d575d6f367` |
| `experimentalDesignTemplate-celllines.txt` | 254,201 | `7c3f80c1afa9e868f91e780edd5aaa72b6f43951a16c8976baa99e74b2e4a590` |
| `txt/parameters.txt` expanded | 3,557 | `4b00211bcabda41c079f4a901aeea4e3e88496e494df31b7bb69b0f5d65a8779` |
| `txt/parameters.txt` compressed | 1,163 | `9afdfcc399cc918ff7fa364fa321b30d397f596c0b570a30e04d45571d365167` |
| `txt/summary.txt` expanded | 1,524,343 | `48cc1cf6551ef58dd0e34e8efea5d39205083a330f9d58c19b234a9594e9f71e` |
| `txt/summary.txt` compressed | 397,662 | `3bd17800387aaa2ca30cdb10c892926754df379aefbd9059c4073d77818e1066` |
| `txt/peptides.txt` expanded | 3,105,083,523 | `4b19b9248aa2fa3a498fa3a16320afbfcc1d8a9a81b33cadbe15357f65435a20` |
| `txt/peptides.txt` compressed | 417,911,174 | `be1a03a2a7d942196b762552e8669af9965dcc0b3fbb981050ea79f8cd6d632d` |
| Reanalysis Supplementary Table 1 XLSX | 35,614 | `f6ddcb22edd10b0bd5883eded155b98a901cf22ace4d8570ddd641291bf7eec5` |
| Original Table S1 (`mmc2.xlsx`) | 14,847 | `a97f88e188206cad8cbe1fd222638ec104d07c5e8d4b4d105e675704cd7ee55c` |
| Original Table S8 (`mmc9.xlsx`) | 14,406 | `f00a32ec18278d41b91a1b33307a35da00e84aab128971f4629e5c411e4c7dc6` |

The ZIP is 35,402,923,317 bytes. Member retrieval used exact HTTP ranges with
ETag `"83e2dbd35-5d1063d7a7dc0"`; CRC32 values are respectively 2547b4ce,
b9d4aac1 and c25e63aa. The FASTA matches #654's earlier trusted digest. Other
hashes above were observed independently during this recovery and now provide
pins for reproduction; they do not establish recovery of the missing portable
archive with SHA256 dc8cae02….

## Biological and assay metadata

Reanalysis Supplementary Table 1, sheet `Supplementary-Table-1-samples-l`,
row 118 identifies `ppb.iBAQ.Hela_1_Bekker.Jensen_Tryp_46Fractions` as HELA,
CellLine, Cervix, study/batch Bekker.Jensen. Tumour individual and breast subtype
are marked #N/A. The normalized display spelling does not itself establish a
biological specimen or replicate identity. The exact MaxQuant experiment key is
`Hela_1_Bekker-Jensen_Tryp_46Fractions`.

The original paper's Cells section describes epithelial cervical carcinoma
HeLa, female, with DMEM culture and serum supplementation, and explicitly says
cell-line authentication was not performed. Keep the reported culture conditions
with their paper-level attribution; no perturbation assignment or authenticated
specimen identity is inferred from this description. These are shotgun
proteomics assays; MHC allele sets are not applicable and are not inferred.

Both the standalone XML and effective result summary assign exactly 46 raw
files to the selected experiment, with 46 distinct fraction identifiers,
Trypsin/P and two permitted missed cleavages. Effective parameters report
MaxQuant 1.6.0.16, minimum peptide length 7, maximum peptide mass 4,600 Da,
fixed carbamidomethyl-C, and match-between-runs enabled. The standalone XML
reports version 1.6.0.1; keep both source values rather than replacing one.

The searched FASTA contains 93,588 unique accessions: 71,591 without an isoform
suffix and 21,997 isoforms. Its filename and the publication's sequence-count
description must not be substituted for this measured inventory.

Primary-method review found an error in the earlier reconstruction: its pH 10
value was an unsupported default. The original paper's Offline High pH
Reversed-Phase HPLC Fractionation section reports buffer C at pH 8. The new
registry records pH 8 explicitly as reported buffer pH, not measured gradient or
eluate pH. Broader legacy metadata review remains tracked in #688. This deliberate
correction, source receipts and new lineage columns require new artifact hashes.

The implemented source recipe independently reproduced all retained counts,
including 150,855 MS/MS and 21,178 matching identifications. Effective summary,
standalone XML and design assignments agree on every selected raw file/fraction.

Original Table S1 reports 166,620 peptides for the 46-fraction trypsin experiment.
That original-analysis summary is not the reanalysis label table. Keep original
and reanalysis counts separately, including MS/MS versus matching evidence.

A fresh complete scan counted 713,496 source peptide rows. The selected scope
has 175,134 positive-intensity rows before excluding 3,101 reverse/contaminant
rows, leaving exactly 172,033 distinct accepted sequences and 12,287 observed
leading-razor parents. All those parents occur in the pinned FASTA. The parser
must explicitly allow bounded large fields: MaxQuant's full-table identifier
lists exceed Python CSV's default 131,072-character field limit.

## Additional scopes and limitations

The XML contains additional HeLa Trypsin/P 39- and 70-fraction scopes and HeLa
Chymotrypsin+, GluC and LysC scopes, plus A549, HCT116, MCF7 and SHSY5Y scopes.
These share the deposited search reference but still require independent
scope/result/settings admission. No transfer claim is established by inventory.

The HeLa 39-fraction trypsin scope contains **40 raw files and 39 distinct
fraction identifiers**. Two files share fraction 439, one with a QC-labelled
basename. Preserve these as different counts; do not convert run count into
fraction depth or assume an additional biological replicate. The 14-fraction
experiments in original Table S1 explicitly reuse Kelstrup et al. (2014).

Prospective detectability, biological replicate rates and comparable first-seen
depth remain unsupported. Candidate labels must refer to one verified search
scope and condition on observed parents. Shared observed sequences must never
make alternate parent occurrences negative. The actual rebuilt candidate export,
independent label/coordinate validation, public artifacts and Presto consumer
are still required by #681/#70.
