# Frozen MHC schema-1 package

Synthetic five-assay fixture generated with `hitlist/mhc_evidence.py` from
Hitlist 2.0.42 (`13b91424295e01286db46e8e9818fcc2fe816834`), using the
`_source` helper in `tests/test_mhc_evidence.py`. The recorded Hitlist version
was fixed to 2.0.42; all artifact bytes and digests come from that writer.
The source includes positive-MS, binding and other-assay rows, with six
contributor links. No experimental patient or donor data are included.

This freezes the pre-core-contract manifest (package schema 1 / catalog 2)
for historical replay regression tests. It must not be regenerated with the
new schema-2 writer. README.md is documentation, outside the hashed inventory.
