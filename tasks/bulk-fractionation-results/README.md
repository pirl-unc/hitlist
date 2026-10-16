# Reviewed Bekker-Jensen fractionation metadata

The complete legacy original-analysis summaries retain all **2,047,003 peptide-arm rows** and **156,211 protein-arm rows**. Every original field except the known incorrect citation and the reviewed pH annotation is unchanged, in the same order. `reported_reference` and `legacy_fractionation_ph` retain both prior annotations. Row counts are not counts of independent experiments, unique peptides, donors or biological replicates.

| Scope category | Peptide-arm rows | Protein-arm rows | Reviewed pH and basis |
| --- | ---: | ---: | --- |
| Ordinary 39/46/70-fraction arms | 1,740,267 | 124,706 | 8; paper-reported buffer C |
| Reused 14-fraction experiments | 207,570 | 11,557 | Unknown; original protocol not established |
| `Phospho-50fracs` | 39,448 | 7,232 | Unknown; exact methods-to-scope linkage unresolved |
| Named `Tryp-Phos-pH8` / `Tryp-Phos-pH10` arms | 59,718 | 12,716 | 8 / 10 from deposited experiment labels |

The [primary paper](https://doi.org/10.1016/j.cels.2017.05.009) reports buffer C pH 8 in “Offline High pH Reversed-Phase HPLC Fractionation” (XML section `sec4.4.4`). This is a buffer annotation, not a measurement of the gradient or eluate. Table S1 identifies the reused 14-fraction experiments; separately named phosphopeptide arms do not inherit a blanket default. `scope-evidence.tsv` gives the exact section/supplement/deposit attribution for all 14 curated scopes. Source URLs, byte sizes and SHA-256 hashes are in `source-registry.json`; the same YAML registry is packaged with Hitlist.

The tables retain HeLa, A549, HCT116, HEK293 and MCF7 cell-line identities, digestion/enrichment/fraction-count axes, original source labels and assay context. They are **shotgun bulk proteomics**, not MHC-ligand observations; MHC allele sets are not applicable. Cell-type and culture-condition annotations beyond the retained source/line/protocol fields were not requalified in this pH audit. Donor, specimen and biological-replicate identities remain unresolved. General bulk replicate-denominator inference remains tracked in [#699](https://github.com/pirl-unc/hitlist/issues/699).

## Downloadable evidence

The [public artifact release](https://github.com/pirl-unc/hitlist/releases/tag/bulk-fractionation-review-v1) contains 15 independently re-downloaded and hash-verified assets:

- `peptides.csv.gz` and `proteins.csv.gz`: complete corrected tables, with original annotations retained.
- `scope-counts.tsv`: complete before/after counts by granularity and exact curated scope.
- `scope-evidence.tsv`, `registry.yaml` and `source-registry.json`: human-readable citable evidence and immutable source pins.
- `audit.json`, `independent-verification.json`, `publication.json` and `SHA256SUMS`: measured preservation results and integrity receipts.
- `audit.py`, `projection.py`, `loader.py` and `verify.py`: exact executed producer and independent verification code.

All acquisition and materialization use datacache. Original public CSVs and previously published PXD013455 and training bundles remain immutable. The corrected outputs do not establish the original searched FASTA, protocol comparability, biological independence or a newly qualified detectability scope.

## Reproduction and validation

Actual audit recipe: clean `dad214358c640ece1afc4dafd283211df766bbce`, candidate Hitlist 1.80.0, Python 3.12.6, pandas 2.3.3. Runtime **64.873 seconds**, process peak RSS **273,481,728 bytes**. From that checkout:

```sh
python scripts/audit_bulk_fractionation.py --receipt /tmp/bulk-ph-audit.json
```

The verifier was added after the original audit and compares every serialized field independently. Run it from the final PR checkout, or use the published `verify.py` with the receipt produced above:

```sh
python scripts/verify_bulk_fractionation.py /tmp/bulk-ph-audit.json --output /tmp/bulk-ph-verification.json
```

Both complete serialized tables independently verify against their pinned originals. Focused tests pass **98** on both pandas 2.3.3 and 3.0.6, including old/new parquet and CSV readers, exact experiment-name lookup, unknown-pH row retention and within-arm abundance ranking. Formatting/lint pass. The local full-suite memory guard refused at 0.27 GB available; this is not a passed suite. Remote full/integration/release validation and final software publication remain pending in [#700](https://github.com/pirl-unc/hitlist/pull/700).
