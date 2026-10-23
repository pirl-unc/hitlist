# Binding assay context: source review and independent replay

This audit replays the schema-10/catalog-15 binding successor directly from its
embedded original v4 and corrected-cohort Parquet files, the source article XML,
and the review ledger. The replay script imports PyArrow but no Hitlist code:

```bash
python scripts/verify_binding_assay_context.py \
  /path/to/binding-review-package \
  --output tasks/binding-assay-context-210-results/replay.json
```

The replay checked the SHA-256 and size of every one of the successor's 256
files, exact retention of all 251 predecessor file digests, the pinned v4 and
cohort manifests, all 814,268 binding observation IDs and original measurement
anchors, all 21 claim values/statuses/excerpts against the article Methods
section, and the corrected human cohort's exact source rows. The successor
manifest SHA-256 is
`58ac75009b6c6452cf19b317908f6043f1be414c81dc8ae9e9423a185467721d`.
The original Europe PMC [article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5421785/fullTextXML)
has SHA-256 `948816f8584a2779082048ad161bf12d8c8f0ebae2c2a70315d9d22efcaf65aa`;
its source and review ledger are embedded in the package. Machine-readable
endpoint, method and PMID inventories are in [replay.json](replay.json).

| Population | Assay rows |
|---|---:|
| All v4 binding | 814,268 |
| Corrected human-source-organism binding cohort | 25,896 |
| Literal IC50 in that cohort | 8,084 |
| PMID 28481943 binding review in full v4 | 938 |
| Of those, in corrected cohort | 910 |
| Of those, influenza-peptide controls outside cohort | 28 |

The cohort's literal IC50 rows are 8,080 IEDB and four CEDAR, from 198
distinct populated PMIDs plus 543 rows without PMID. Exactly 560 have a
nonempty assay comment. The source-review coverage is 910 of 8,084 cohort IC50
assays, not 910 independent donors or replicates. The largest other IC50 study
is PMID 15749848 with 920 cohort assay rows; it remains unreviewed here.
The exact corrected-cohort binding-row comparison found zero changed original
measurement anchors. The 813,330 other v4 binding results, including 24,986
corrected-cohort binding results, remain explicitly `not_source_reviewed`.

The 938 reviewed rows all report the same IEDB method,
`purified MHC/competitive/radioactivity`, endpoint `half maximal inhibitory
concentration (IC50)`, and unit `nM`. The [source paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177140)
supplies protocol context beyond those existing IEDB/CEDAR assay fields. It
reports a study-wide competitive radioligand protocol, a 0.1–1 nM tracer
concentration range, room-temperature or 37°C incubation, a two-day incubation,
six competitor concentrations across a 100,000-fold range, and at least three
independent experiments. The exact tracer concentration, temperature, peptide
purity, pH, replicate IDs and measurement uncertainty for an individual assay
are unresolved or unreported. Three capture antibodies apply by HLA-II locus,
not to every assay indiscriminately. The paper's conditional IC50/KD statement
is an interpretation claim; no KD result is added.

Among the 21 claims, 10 are reported at study scope, three at HLA-II locus
scope, one is an unresolved per-assay range, two are unresolved per-assay
alternatives, one is an interpretation rather than a measurement, and four
fields are unreported. An [example IEDB assay](http://www.iedb.org/assay/3318674)
corresponds to `observation:record:iedb:row:677282`; its source-linked
review row points to PMID 28481943 and `PMC5421785/fullTextXML#sec003`.
Two more source-linked assay examples are included in `replay.json`.

The successor retains all original source partitions, contributor payloads,
prior curated claims, nonimmune evidence, and training projections. The new
`binding_assay_context` relation has one review-state row per original binding
result, including unreviewed results with a null protocol. The 21
`binding_protocol_claim` rows remain source scoped. These rows are context, not
additional binding assays or training labels.
