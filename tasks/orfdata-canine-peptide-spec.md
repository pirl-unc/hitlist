# #668 — released canine peptide and protein-group evidence

## Scope

Consume ORFdata's released `pfizer-atlas-dog-peptide-evidence` generation
2026-10-27.1 through its public quantitative, mapping, relation and documented
CanineEvidence APIs. Keep the original Pfizer summary provider unchanged.
The companion datasets describe the same study (MSV000099744 / PXD070290),
not independent measurements or presentation-positive dogs. No vendor reader,
confidence threshold, inferred individual abundance or new normalization belongs
in Hitlist.

## Frozen bundle and explicit acquisition

Add an explicit `orfdata_peptides` provider to curate_canine_bulk. Its new
schema-2 bundle contains an independent snapshot of the released dataset.sqlite
and a manifest carrying the frozen source context, ORFdata manifest/distribution,
registry identity, exact installed artifact digest, source-reference pin and
original summary identity. Use the existing copy-on-write snapshot helper;
never hardlink mutable ingestion/cache inputs. Writes stage and verify before
publication, and refuse to replace an existing output. Schema-1 summary and PeptideAtlas bundles retain unchanged
verification and loader behavior. Historical schema-2 verification uses frozen
metadata, with no live registry or acquisition.

Pin the released registry manifest plus download metadata and validate the
complete searched reference and original protein-summary SHA before acquisition.
The installed artifact is 24,710,258,688 bytes, SHA-256
9014c5201e21d76d50d286bdc56abc6d2eaafdd7b936d927255bda72c7e28a57;
compressed transfer is 6,896,030,151 bytes in four hash-pinned parts. Retain
separate transfer, installed and output bounds, reject undersized limits before
acquisition, and require explicit allow_download. Ordinary reads remain offline.
The complete searched reference is embedded; do not acquire a separate FASTA.
ORFdata >=0.40.0 requires Python >=3.11. Preserve Hitlist's existing
Python-3.10 support and legacy providers: use conditional dependency bounds
(orfdata>=0.40 on Python>=3.11, existing baseline on 3.10), a clear provider
version guard, and Python-3.10 tests for the unavailable-provider error. The
removed `data` extra is no longer needed. Do not silently raise Hitlist's
overall Python floor in this integration.

## Reads and evidence semantics

Provide bounded reads through load_bulk_evidence and a streaming public reader
for source-native detail. Forward common measurements, mappings and group
relations through ORFdata; do not interpret its private SQLite schema. Preserve
native IDs, sequence/modifications, confidence and definitions, target/decoy
status, all source memberships and all exact/I-L candidates separately, unit,
scale, zero/missing/nonfinite state, and acquisition/sample/plex/channel lineage.
Expose native records and links through the documented adapter. A destination
or peptide selection must retain every alternative in each selected mapping.

Bulk rows carry bulk semantics, empty MHC restrictions and no positive-DLA flag.
Tissue-pool identity remains distinct from individual dogs; bridges and empty
channels remain controls. Preserve the publication Eclipse/deposit Lumos conflict
and native instrument records. The old summary remains representative-only;
the new generation's membership context does not reassign summary abundance.

## Verification and publication

Build a small offline released-format fixture with real ORFdata APIs, covering
shared assignments, reported versus compatible alternatives, source/reference
mismatch, confidence/decoy, modifications, missing/nonfinite quantities, corrupt
artifacts, explicit acquisition bounds, interrupted writes and historical replay.
Verify frozen bundles after disabling network and live catalogs. Preserve output
bytes and read-only source files. Reconcile the full real released artifact hash,
all source/public counts, complete reference identity, summary lineage and
selected source/query records. Existing verified managed source:
/private/tmp/orfdata-followups-20261026/released-040-cache (read-only).
Record full-scan versus sample verification honestly; do not claim a new vendor
source audit where the evidence is upstream's pinned acceptance receipt.

Run format, lint, focused tests, test.sh and final-head CI. Bump from the actual
released main version, merge a separate PR, run clean-main deploy.sh via the
established release workflow if local memory is insufficient, verify PyPI hashes,
and update #668 with exact completed acceptance and any remaining limitations.
