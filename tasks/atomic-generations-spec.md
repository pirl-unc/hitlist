# Complete dataset generations — Hitlist #645

## Scope and sequence

The authorized series covers Hitlist and Presto. MHCflurry is deferred and its
repository, issues, adapters and training runs are outside this series.
1. Publish a coherent Hitlist generation (#645).
2. Consume frozen Hitlist inputs in Presto with durable lineage (#71).
3. Publish/consume the verified search-scoped detectability dataset (#681/#70).
4. Build the pinned corpus and prospectively frozen partitions (#680).
5. Qualify declared Presto outputs with bounded real-data fitting and held-out
   comparisons (#48/#53). Software checks alone do not qualify a dataset.

Every stage lands through a reviewed, versioned PR and the established release
channel. Preserve user worktrees, registered sources, and historical caches.

## First PR contract

A full observations rebuild stages observations, binding, other assays,
contributors, optional mappings, bulk proteomics, line expression and their
metadata together. Use the public datacache materialize API under the existing
resolved Hitlist cache root. Datacache owns input copies, locking, hashing,
receipts, staging cleanup and the atomic directory rename that makes a complete
generation current. Do not implement a competing cache or publication protocol.
Retain prior generations and legacy flat artifacts; never overwrite a published
generation. Explicitly account for input copies, new outputs and retained old
outputs in disk requirements. This guarantees atomic visibility, not survival
of arbitrary filesystem damage or power loss.

Use original registered paths for freshness identity while scanning datacache's
private copies. Reject a changed input/curation identity during the build.
Keep artifact-local metadata independent of temporary absolute output paths so
it remains valid after publication. Preserve all raw assay, source and biological
fields; no new biological inference or changes to label meaning in this PR.

Readers resolve a complete generation once for a logical multi-file operation.
A public dataset_snapshot context supports callers making several API calls;
Hitlist loaders, joins, exports and bundle writers pin their own outer operation.
Nested calls inherit the pin. Reads are quiet, offline and do not create files.
An absent generation falls back to the legacy flat layout; a corrupt generation
fails explicitly rather than silently mixing it with old flat files.

Standalone mapping/bulk/expression rebuilds must also publish a replacement
complete generation when a generation is installed. Preserve unchanged output
bytes, refresh the metadata they affect, and reject publication if their parent
generation changed while preparing the update. Internal full-build calls write
only within the active staging context. No mutable aliases into a generation.

Include a human-readable README identifying every artifact, observed hashes in
the datacache receipt, source locations, policy limits, and how to inspect/use the
snapshot. Source literature, cell types, allele sets, conditions and assays remain
explicit columns in data; corpus publication later adds source-linked inventory
and coverage/unknown counts for these dimensions. Unknown is a documented state,
never an inferred donor, specimen, assay or allele assignment.

## Validation and delivery

Use tiny real-format inputs with the actual builder and contributor collector.
Inject failures after every output write, metadata creation, validation, and at
datacache's final publication rename. Assert the prior generation is still
readable and byte-identical, and that fresh readers never see a mixed set. Test
an old reader spanning successful publication, first migration from flat files,
optional mappings, standalone updates, repeated builds/cache hits, corrupt-store
rejection, and path/metadata validity after staging moves. No full-corpus rebuild
on an inadequately provisioned machine and no bypassing the memory guard.

Run format.sh, lint.sh and test.sh; use CI for full suites when local capacity
refuses. Review the complete diff, exact tested commit and built distributions,
merge, build/deploy from clean main, verify public hashes, and record the receipt
in #645/#678 before moving to Presto #71.

## Final series artifacts

Publish downloadable tables and sidecars via datacache with pinned hashes,
source citations/accessions, human-readable study/sample/assay inventory, explicit
label policies and units/censoring, exact partition assignments, observed coverage
reports, and leakage audits over the selected data. Keep peptide overlap,
observation identity, reused experiments, donors and specimens separate. Report
unresolved lineage and restrict unsupported independence or prediction claims.
