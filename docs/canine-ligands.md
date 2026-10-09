# Offline canine ligand curation

Hitlist curates the original MS observation worksheets from
[Kaabinejadian et al., PMID 42199926](https://pmc.ncbi.nlm.nih.gov/articles/PMC13200046/)
through an explicit offline API. Supply the six original `mmc2.xlsx` through
`mmc7.xlsx` files. Their published MD5s, SHA256s, sizes, source URLs and selected
worksheets are pinned in `hitlist/data/canine_ligands.yaml`.

```bash
pip install 'hitlist[curation]'
```

```python
import json
from pathlib import Path

from hitlist import curate_canine_ligands
from hitlist.provenance import ContributorCollector
from hitlist.supplement import scan_supplementary

output = Path("canine-curated")  # must not already exist
manifest = curate_canine_ligands("local-original-workbooks", output)
with ContributorCollector(scratch_dir=output, max_scratch_bytes=256 * 1024**2) as collector:
    observations = scan_supplementary(
        entries=json.loads(manifest.read_text()),
        directory=output,
        allow_download=False,
        provenance=collector,
    )
    observations.to_parquet(output / "observations.parquet", index=False)
    provenance = collector.write([observations], output / "contributors.parquet")
(output / "provenance.json").write_text(json.dumps(provenance, indent=2))
```

The curator never downloads. It checks all original files before parsing, reads
the selected worksheets incrementally, verifies every sequence/length and row
count, and publishes a deterministic manifest only after the complete export
passes. Missing, modified or malformed inputs fail without publishing a partial
result. The installed package contains the profile and curation code; it does
not contain or automatically fetch the original workbooks or derived peptides.

The curated scope is all reported **8–30-residue** observations, broader than the
paper's 8–14-residue summaries:

| Experimental arm | Observation rows |
|---|---:|
| HCT116 DLA-88*003:02 | 811 |
| HCT116 DLA-88*012:01 | 912 |
| HCT116 DLA-88*501:01 | 1,857 |
| Lola H58A | 207 |
| Lola BB7.6 | 330 |
| 163828A BB7.6 | 1,671 |
| Lily H58A | 60 |
| Bogey BB7.6 | 10,511 |

This yields **16,359 observations and original contributors**: 3,580 from
human-host monoallelic DLA experiments and 12,779 from four canine tumors
(12,181 distinct reported canine peptide strings). The two Lola IPs share one
dog and one osteosarcoma specimen. Raw acquisition IDs remain unknown.

Human HCT116 source proteins and presenting cells remain human; the introduced
MHC is canine. Ordinary HCT116 is not classified as an HLA-null host. The curated
engineered variant is `HCT116 HLA-I KO`. The DLA-88*501:01 construct's reported
`AA21 P > L` change is retained without inventing its numbering convention.
Independent species-context records document the transduction mechanism.

Tumor peptides retain class-I, unassigned restrictions. RNA-derived DLA typing
is retained as reported context, including Bogey's DLA-88L qualification; it
does not assign each peptide to every allele. Human-TAA comparison worksheets,
binding predictions, flow cytometry and Western blots do not enter MS evidence
or establish canine CTA membership.

Every source row retains its workbook URL/hash, sheet, one-based row number,
reported protein accessions, sample, IP antibody, search-database description
and raw deposit. Reported protein accessions are not verified unique mappings.
The 1% study FDR is not an individual peptide q-value. Missing q-values, spectra,
exact searched FASTA hashes, modification details and I/L discrimination remain
explicitly unresolved. Matching a reported string does not establish I/L
discrimination, and no contained peptide window inherits its MS observation.

The output is a scoped supplementary source, not a replacement for a complete
Hitlist index. Exact `attributed_sample_label` values connect observations to the
packaged experimental arms and reviewed donor/specimen lineage during export.
Species-scoped CTA evidence bundles with explicit canine references and normal
tissue policies are tracked in [#661](https://github.com/pirl-unc/hitlist/issues/661).
Hitlist supplies evidence; downstream consumers assemble vaccine sequences.

## Other reviewed local CSVs

`scan_supplementary(entries=..., directory=..., allow_download=False)` also
accepts caller-reviewed MS manifest entries. Both local and packaged entries
undergo the same excluded-from-MS conflict check. Explicit entries require local
files and never fall back to an unrelated download-registry filename, even when
`allow_download=True`. A supplied `sha256` or `size_bytes` is checked. A missing
`peptide` column fails visibly. An explicit row label overrides the entry's
default `attributed_sample_label`, and distinct labels preserve separate
observations. Original fields and the complete manifest survive in contributors.
Calls without explicit inputs preserve the existing packaged scanner behavior.
