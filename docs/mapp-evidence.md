# MAPP proteasome-associated fragment evidence

`MAPPAdapter` imports the pinned PXD009092 MaxQuant tables and original publisher
workbooks through `write_external_evidence_package`. It preserves 10,673 peptide
rows, 149,959 evidence rows, 18 source runs and every original TSV field.
It does not assign intracellular concentration or training eligibility.

```python
from hitlist.external_evidence import write_external_evidence_package

write_external_evidence_package(
    supporting_package,
    destination,
    source_roots={"pxd009092": original_mapp_assets},
    source_ids=("pxd009092",),
)
```

Supply the exact relative paths and bytes in
`hitlist/data/evidence_revisions/mapp_assets.json`. Include `pxd057588` and its
source root when that source should also appear in the new generation. Source
selection is explicit. This builds an external-source generation from its
supporting predecessor; it does not automatically append or rebuild later
T-cell, numeric or query generations.

The typed relations separate source records, peptide identities, group rows,
channels, individual evidence observations, peptide/run results, quantity
representations, derivation inputs, source assertions and publisher cells.
All 36 group rows lacking numeric IDs remain quarantined records with unique
source-row identities. Neither reverse nor contaminant rows are silently removed.

Evidence intensity sums, the median of the three BEADS sums and published
bead-filtered intensities remain distinct. The reconstructed rule retains a run
sum when it is at least twice the BEADS median. Reported rounding is checked at
relative tolerance 1e-4 or absolute tolerance 1 intensity unit. Filtered zeros
with positive evidence, no evidence, and missing intensities have different
statuses; none is silently made a biological negative. PEP confidence remains
separate from quantity and FDR. Literal `NaN` PEP values retain their raw text
and an explicit missing-confidence status.

Publisher UT/TNF+IFN values retain their log10 median representation and links to
all three filtered source runs, including zeros. Imputed columns retain conflicting
headers and no assigned condition. The selected protein panel remains original
cells with unresolved specimen/acquisition identity, not a precursor denominator.
Source parent lists, leading coordinates and flanks remain source claims; the
versioned reference occurrence join is still pending. The unavailable original
2016 FASTA cannot be silently replaced by a later reference.

Run context preserves human HEK293 measured cells, actual UT/TNF+IFN/CHX conditions,
BEADS background and NO-DSP/NO-INH procedural controls. The shared `T+I` filename
prefix does not assign treatment. Procedural preservation controls are not gene
knockouts. Source specimen identities remain unresolved. Search scope is 6–40
residues; unreported shorter products are not negatives. The project/paper FDR
conflict and unresolved PXD009360 parallel-tryptic pairing remain assertions.

Original inputs are supplied locally and copied into the caller's package, not
bundled in the Python distribution. PXD009092 declares EBI terms of use, not CC0;
publisher workbook redistribution terms remain unverified. Preserve source
attribution: Wolf-Levy et al., *Nature Biotechnology* 36, 1110–1116 (2018),
[doi:10.1038/nbt.4279](https://www.nature.com/articles/nbt.4279), PRIDE PXD009092.

Tracked work: [source/reference integration #752](https://github.com/pirl-unc/hitlist/issues/752)
and [shared query/recipe interface #739](https://github.com/pirl-unc/hitlist/issues/739).
This importer alone does not complete either issue or establish a biological fit.
