# Corrected human cohort: verified native export

[Download all 21 assets](https://github.com/pirl-unc/hitlist/releases/tag/human-cohort-corrected-v1).
Every public download was independently hash/size-verified through datacache, and
the native training bundle passed its consumer verifier. The [verification receipt](public-verification.json)
contains durable URLs and hashes. Native manifest SHA-256:
`b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`.

Producer: released **Hitlist 1.78.0**, corrected complete corpus v4. Export recipes
match commit `8fc7f86dcf840feb2c945e9da472198514e83d50`; its already completed
snapshot receipt was uncommitted at launch. This was not a clean Git launch; no
producer or recipe source changed during execution. `reproduce.tar.gz` preserves
exact recipe bytes, dependencies, parent receipt, inputs and original export log.
Runtime: 1,473.964 seconds; peak runner RSS: 12,472,029,184 bytes, excluding children.
CPU: eight; RAM: 64 GiB; scratch: 512 GiB; GPU: none.

## Corpus and lineage

There are **4,071,463 observations**: 4,045,567 MHC-MS and 25,896 binding, with
2,711,461 class I, 1,353,735 class II and 6,267 nonclassical observations. The
7,646,178 contributor links cover every observation with zero unresolved links
and 7,605,968 distinct source records. Contributor and lineage bytes are identical
to the original human baseline; missing biological identity remains missing.

Known experimental origins cover 57,188 observations across 17 identities;
4,014,275 observations have unknown origins. All 4,071,463 have unknown donor,
specimen and acquisition identity. These reports cannot establish independent
experiments, donors or specimens. There are 1,081 nonempty PMIDs, 21 supplement
files, 133 cell lines and 28 cell types. Cell type is missing for 1,281,272 rows;
489,384 supplementary rows still lack assay method/response/qualitative fields.

## Measured changes from the original human baseline

The complete comparison covers **161 fields in all 4,071,463 rows**, verifying
unchanged native observation IDs, source/assay/provenance keys and row order.
Exactly **89,567 observations / 662,897 cells / 11 metadata fields** changed.
Peptide sequences, measurements, endpoints, units and censoring are unchanged.

| Change | Actual evidence |
| --- | --- |
| Predicted restriction and source condition attribution | PMID 32350084: 78,744 observations; 61,917 gain resolved condition attribution and 16,827 remain ambiguous across cell lines. |
| Treatment uncertainty represented explicitly | PMID 33981976: 10,823 observations retain HCT 116 group attribution but the treatment arm remains unknown. |
| Condition coverage | 463 distinct condition IDs versus 437; missing condition IDs fall from 2,090,691 to 2,028,774. |

Use [field counts](comparison-field-counts.tsv), [study counts](comparison-study-counts.tsv),
[78 value transitions](comparison-value-changes.tsv) and the [39-cell cited preview](comparison-preview.tsv)
for inspection. The public compressed changed-cell table contains every change
with observation ID, source/assay keys and citations. The original generic
1,378-row inspection preview is byte-identical and misses these corrections; it
is not representative and should not substitute for the dedicated change audit.

`arm_resolution` is a study-level reason applicable to ambiguous rows (see
[the curation contract](../../docs/pmid-curation.md)); its study-level value also
appears on resolved rows. Interpret it alongside row-level `condition_id` and
`sample_attribution`, not as proof that every row in that study is unresolved.
Sample MHC sets remain distinct from measured or predicted restriction evidence.

## Consumer handoff and remaining limits

The original baseline and its earlier Presto memberships/results remain immutable.
A new Presto preflight uses this manifest, released Presto 0.1.13, reservoir caps
and relative_deficit_v2 whole-component allocation. Its assignments, coverage and
leakage reports must be frozen separately; prior results cannot qualify these
inputs. Positive elution is not detectability supervision, and unknown lineage
cannot be counted as a successful independence audit. #678/#680 remain open.

The complete corpus, native cohort and field audits are public and verified.
Format/lint and 21 focused qualification/comparison/recovery tests pass. Final-head
CI, review, merge and the 1.79.0 PyPI distribution remain required.
