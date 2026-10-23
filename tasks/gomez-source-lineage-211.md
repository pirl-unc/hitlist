# Gomez-Zepeda source-result and run-lineage successor (#779)

## Objective and source pins

Restore the original per-condition, culture, injection and measured peak-area
evidence behind PMID 38480730 without changing the released 68,552 derived
peptide observations or their current training labels. The verified 2.0.10
schema-10/catalog-15 package is the predecessor; its manifest SHA-256 is
`58ac75009b6c6452cf19b317908f6043f1be414c81dc8ae9e9423a185467721d`.
Every predecessor artifact remains byte identical in the successor.

Primary open-access article: Gomez-Zepeda et al., Nature Communications 2024,
PMID 38480730, DOI `10.1038/s41467-024-46380-y`, PMCID `PMC10937930`.
Europe PMC full-text XML SHA-256
`3311abf23d895b7f37f18cd8dbf5ea3a273d785d6c34db24b2aaa0b5a1d97260`.
Its license is CC BY 4.0. Public publisher archive pins:

| Supplement | ZIP bytes | ZIP SHA-256 | CSV member | CSV rows | Member SHA-256 |
|---|---:|---|---|---:|---|
| Data 3, MOESM7 | 11,399,069 | `ab6db7e77d04881b8ee191b4d6b2025e117870a995089836d34be02ef2ca04af` | `S4_pep_all_06154.csv` | 135,688 | `2baa1a205db58a5a054dbd53dcc4635b8929b1fe18e3f40c98cf1b4e93d5bb17` |
| Data 5, MOESM9 | 16,117,475 | `00ceccb43126c8e10ec97a540df1bdf3b7738dc76fed83256b19d27ebe2a047c` | `S6_pep_all_06133b.csv` | 459,228 | `4c5baa1ed8f915a895521ec183a1cb0d45091efef5a9cfd9024c93be95854c71` |

The original S4 table has `Condition`, `TIMS_ramp`, `Cell.Line`, `Polygon`,
`BioReplicate`, `Sample`, `TechReplicate`, `Source_File_ID`, `Found_By`, `Area`
and `Log2Area`. Original S6 adds `ConditionGenotype`, `Genotype`,
`mokapot_q_value` and related confidence fields. The source paper describes
JY_WT and Raji_WT (three cultures each), JY_S1/S2 and Raji_S1/S2 (two
cultures each), and three injection replicates per sample. The source file
identity is an acquisition locator, not proof of a donor or independent
physical specimen beyond the explicitly reported culture/replicate slots.

For source `Sequence` values of length 8–15, the exact union by cell line
matches the five packaged derived CSVs with no missing or extra sequences:
JY 20,983, Raji 24,325, HeLa 10,356, SK-MEL-37 11,491 and plasma 1,397.
The S6 JY 8–15-aa WT and S sets share 15,509 sequences; Raji WT and S share
18,015. A merged peptide row therefore cannot be assigned one arm by sequence
alone. S3 method-comparison rows and S5 spike-experiment rows remain distinct.

## Schema and package contract

1. Add a schema-11/catalog-16 source-lineage successor over the verified
   schema-10 package. Snapshot every predecessor file under an independent
   inode; embed its manifest, the two original ZIPs and a revisioned source
   registry. Check exact inventory and recursive parent/source hashes before
   publication or read. Historical readers stay available.
2. Materialize `gomez_source_result` at one original CSV data row. Stable
   source row ID = archive ID + member + one-based data-row index. Preserve
   every original source column as a raw string in a queryable `native.*`
   field, along with typed common identifiers: cell line, experimental
   condition, biological-replicate label, sample/injection label, technical
   replicate, source-file ID/name, peptide sequence, molecular form, charge,
   reported identification/confidence and original area/log2-area values.
   Numeric projections must be losslessly traceable to their raw strings;
   source peak-area units remain arbitrary/unreported, never concentration.
   Preserve blank versus numeric zero and do not treat one source row as an
   independent donor, specimen or assay result.
3. Materialize `gomez_peptide_membership` at one distinct sequence × source
   phase × condition × reported sample/source file. It is detected-source
   membership, not an unobserved-negative table. Record explicit
   `n_source_result_rows`, source-result link and original sample/culture IDs.
   Multiple memberships may point to the same derived observation. Rows with
   out-of-scope sequence lengths remain in source results and membership,
   with a null derived observation link and an explicit reason.
4. Reconcile every 8–15-aa source sequence to the exact packaged derived
   observation ID using its cell-line source file and peptide row. Assert the
   five exact unions above, all 68,552 original derived IDs, and no changed
   original assay/contributor payloads. A source peptide occurring in both
   arms has multiple memberships. Do not promote a source condition to the
   existing single-arm training label in this PR; #780 handles the projection.
5. Register the two relations, fields, grains and links in `EvidenceStore`.
   Expose direct source-result and membership queries; a many-cardinality
   relation may require explicit separate queries to avoid multiplying
   observation weights. Document source-specific and common required fields.

## Verification and release

- Build a real successor from the pinned schema-10 package and exact original
  archives; independently stream/replay both ZIP members and compare every
  original row, every typed value, all sequence/condition/run memberships,
  derived links and predecessor file hashes. Publish machine-readable counts,
  source-linked examples, and remaining null/unresolved identities.
- Fixture tests reject changed ZIP/member hashes, duplicate/missing source
  rows, wrong derived links, lost result fields, fabricated conditions,
  unowned artifacts and predecessor byte changes.
- Run `./format.sh`, `./lint.sh`, `./test.sh`, strict docs and distribution
  checks. Review exact-head CI, merge a version-bumped PR, run the clean-main
  release gate and verify both PyPI artifacts. Keep #778/#780 open for the
  condition-label repair and source projection after this source release.
