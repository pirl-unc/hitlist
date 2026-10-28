# Released-format canine consumer fixture

`dataset.sqlite.gz` is a deterministic-gzip encoding of an ORFdata storage-v6
consumer database. `manifest.json` is its embedded dataset manifest and
`expected.json` pins its uncompressed bytes, common counts and the canonical
JSON digest of all 429 original measurements.

Generated with ORFdata commit `ff59b2cf34f4327788bec6670d96bc47688d1737`
(the implementation released in 0.40.0), using its
`tests/test_pfizer_evidence.py::source` fixture with `synthetic-memberships`,
`tests/test_pfizer_native.py::inputs`, and `orfdata.sources.pfizer_native.import_native`.
The source contains bounded native slices, three explicitly synthetic acquisition
rows (including a nonfinite scalar and opaque bytes), shared/dangling synthetic
memberships, and one synthetic high-confidence decoy copied from the first target
PSM. That decoy is a software control, not a claim about a deposited identification.

The native slices originate in MSV000099744 / PXD070290, Oziolor et al., Pfizer,
DOI https://doi.org/10.1038/s41597-026-07512-9. Deposited analysis facts are CC0-1.0;
searched sequences retain UniProt Consortium attribution under CC-BY-4.0. The
fixture's ten-sequence FASTA and partial native selection do not claim complete
study coverage. See the upstream `tests/data/pfizer/vendor/NOTICE.md` at the pinned
commit for source selection and provenance. No original publication workbook is
redistributed. Local input URLs in the manifest are fixture-construction locators;
consumer tests deliberately have no access to those inputs or a separate FASTA.

The fixed database tests the consumer APIs without copying a vendor reader into
Hitlist or requiring extraction dependencies at test time. It has 44 sample/channel
records, 1,100 assay contexts, 429 measurements, 15 mappings and three group
relations. Rebuilding the fixture requires the upstream extraction/test environment;
ordinary Hitlist tests only install its compressed artifact through DatasetRegistry.
