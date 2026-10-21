# IEDB/CEDAR T-cell assay availability, 2026-10-20

This is a source availability audit for [#723](https://github.com/pirl-unc/hitlist/issues/723),
not a frozen training export. Hitlist also exposes a row-level assay metadata
export through `python -m hitlist.tcell_assays OUTPUT_DIR` and the
`iter_tcell_assay_metadata` / `write_tcell_assay_metadata` Python APIs. That
export excludes antigen and allele identities and remains separate from the
MS/binding training table. The audit reads the public T-cell CSV metric exports from
the [IEDB database export](https://www.iedb.org/database_export_v3.php) and
[CEDAR database export](https://cedar.iedb.org/database_export_v3.php).
IEDB's [query API](https://query-api.iedb.org/docs/swagger/) also identifies
T-cell assay, MHC-ligand and receptor records as separate entities. The exact
mutable ZIP bytes inspected are pinned in
[`tcell_audit_sources.yaml`](../../hitlist/data/tcell_audit_sources.yaml);
[`summary.json`](summary.json) is the complete machine-readable aggregate.
No epitope sequences or raw assay records are committed.

Run `python scripts/audit_tcell_availability.py --output
tasks/tcell-availability-results/summary.json` to acquire the pinned files
through datacache and regenerate the summary. Alternatively pass both
`--iedb` and `--cedar` to use already downloaded ZIPs; either way the script
checks byte size and SHA-256 before parsing. The downloaded IEDB ZIP member is
dated 2026-10-13 and the CEDAR member 2026-10-15. This audit counts distinct
source assay IDs, not epitopes, donors, experiments or independent specimens.

## Coverage

| Source | Distinct assay IDs | Distinct references | Negative | Positive tiers combined |
| --- | ---: | ---: | ---: | ---: |
| IEDB | 578,529 | 12,357 | 363,584 | 214,945 |
| CEDAR | 157,263 | 4,069 | 87,981 | 69,282 |

All 157,263 CEDAR assay IDs occur in IEDB. The deduplicated union is therefore
578,529 assay IDs; summing the source rows would double count CEDAR. All rows
in these two pinned exports have one of five *reported* qualitative values:
`Negative`, `Positive`, `Positive-Low`, `Positive-Intermediate` or
`Positive-High`. These are assay-level outcomes, not measured negative MS
presentation labels. The tiers remain distinct in `summary.json`.

| IEDB method/readout family (conservative metadata routing) | Assays |
| --- | ---: |
| Soluble factor release | 396,243 |
| Activation, degranulation or proliferation | 103,103 |
| Cytotoxicity | 24,962 |
| pMHC multimer/tetramer | 36,268 |
| Other T-cell assay | 17,953 |

The family labels only organize an availability report. They are not new
training targets. The JSON retains exact source method and readout counts and
the positive/negative tiers per family. In particular, a multimer signal is
binding evidence, not proof of functional activation. The broad `other` group
includes distinct methods and readouts, including receptor-binding kinetics;
it must be reviewed before any target routing.

## Restriction and experimental context

IEDB reports a restriction *name* on 445,614 assays. Only 135,751 names have
the simple syntax of one two-field HLA allele; that syntax alone says nothing
about how the restriction was established. Another 309,863 names are broader,
nonhuman or otherwise outside that narrow pattern, and 132,915 are blank.
The audit cross-tabulates restriction-name resolution with the source's
restriction evidence code. It separately records `Host | MHC Present` on
25,679 assays; host genotype is not a measured restriction.

The original export includes assay method/readout, antigen object type, host,
effector, APC, stimulation and culture fields. Availability is uneven:
563,993 IEDB assays name an effector cell; 521,761 name an APC; 183,816
report an in-vitro process; and 26,263 have a TCR name. A TCR name alone does
not establish a paired, receptor-specific sequence measurement. Quantitative
measurement appears on 5,814 assays, units on 4,169 and an inequality on
1,101. Some quantitative rows are kinetic/binding assays, so a number must
never be routed to a cellular-response target without checking method,
readout, units and qualifier.

## Processing and next gate

The T-cell export's method and readout fields have no direct-processing
keyword candidates for cleavage, transport, proteasome or processing. This
does not establish that such measurements are absent from IEDB's other
entities or from study supplements. The MHC-ligand source already used by
Hitlist describes antigen-processing *context* for some MS assays; this is not
a direct cleavage/transport measurement. Direct processing needs a separate
source-specific inventory before it can become a target family.

The bounded human-origin peptide functional export is available via
`python -m hitlist.tcell_functional OUTPUT_DIR` and the corresponding public
iterator/writer APIs. It retains the peptide-to-assay-to-outcome link for
direct peptide cellular assays while keeping MHC evidence and specimen
uncertainty explicit. It does not turn the full availability count into a
functional training set; other antigen origins, antigen formats, multimers,
receptor-specific and direct-processing targets remain outside that export.
The remaining #723 work includes separate target-family exports, study and
specimen independence audits where source identities permit them, and
downstream training integration. This availability report alone does not
qualify the data for a Presto training run.

The follow-up `hitlist.tcell_training` bundle standardizes the bounded
human-origin direct-peptide cellular subset into a typed training table while
preserving the full source-faithful assay table and explicit exclusions. Its
manifest reports per-family positive/negative coverage and missing lineage;
it does not assert independent specimens or a certified model holdout.

For the separate Atlas MS cohort, every published row already has a donor
label. The [Atlas crosswalk audit](../atlas-crosswalk-results/README.md) still
marks peptide-to-acquisition and physical-specimen identities unknown. That
uncertainty remains recorded; it does not erase directly reported T-cell assay
outcomes in these separate sources.
