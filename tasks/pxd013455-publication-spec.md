# PXD013455 source recovery and scoped publication — Hitlist #681

## Scope and dependency order

Recover the previously validated reanalysis from its public proteomics deposit
and paper supplements, then publish a durable, datacache-managed evidence bundle.
Do not require the user to locate an archive that earlier curation produced.
Hitlist atomic generations (#686) precede this; Presto frozen-input/partition PRs
remain the next consumer dependency. MHCflurry is excluded from the active scope.
Original PXD004452 reference recovery (#654) remains distinct from PXD013455.

## Source reconstruction

Search primary PRIDE/ProteomeXchange records and the original/reanalysis papers.
Retain URL, accession, PMID/DOI, retrieval identity, original bytes and checksums
for deposited search reference, effective results parameters, standalone XML,
experiment design and scoped observed peptide results. Read the paper supplements
for biological/technical metadata; do not use a protein-expression summary as a
substitute for the search-scoped peptide evidence. Reconcile reported reference
sequence counts and MaxQuant versions against deposited bytes explicitly.

The prior #654 receipt is a comparison target, not a substitute for revalidation:
searched FASTA SHA256 88e76d6479481fe18f4420c1d5f3082b6d7bfbf4c091316cf42506a64471b9cc,
46,989,754 bytes; 172,033 selected nonzero-intensity observations for
Hela_1_Bekker-Jensen_Tryp_46Fractions (150,855 MS/MS, 21,178 matching), 46 raw files;
1,656,863 candidate occurrences / 163,994 distinct observed candidate sequences.
Prior candidate parquet SHA256 4ebefacbf8460ba0abf1f9cfefe0423b06ab4b4016a05a3587b0cdbece2f8fca.
If exact bytes cannot be reproduced, explain dependency/formatting or substantive
changes and validate semantic equality; never relabel different bytes as the old
artifact. The original portable archive has not been found locally or in releases.

## Storage and publication

All source acquisitions and reusable outputs use public datacache APIs below the
shared Hitlist data root. Keep a dedicated immutable namespace for this reanalysis.
Use bounded ZIP member extraction for the large result archive; preserve member
metadata, compressed/uncompressed hashes and CRC validation. Do not download the
entire 33 GiB archive merely to obtain one result member. If datacache lacks a
needed transport feature, file it in the correct repository and implement/release
the reusable primitive there rather than creating a parallel cache in Hitlist.

Publish downloadable artifacts, a complete SHA256 inventory, source/citation
registry, explicit label policy, human-readable README and schemas, reproducible
import/rebuild commands, scope/support report and an independent validation receipt.
Keep cell line, cell type, condition, assay, exact search enzyme, acquisition and
source lineage separate. MHC allele sets are not applicable to non-MHC shotgun
assays; do not invent them from the cell line.

## Label and qualification contract

Preserve direct MS/MS versus matching evidence and positive detection opportunity.
Absence means not observed in this specified search scope, conditional on an
observed leading-razor parent; it is not intrinsic undetectability. All occurrence
coordinates/flanks remain linked; an observed shared sequence never makes an
alternative parent a negative. Graded replicate/depth labels stay null unless
actual identities and comparability are evidenced. Inventory additional scopes
only with their matched reference/results/settings and state which transfer
claims remain unsupported.

## Verification and release

Test importer corruption, bounded extraction, source/schema admission and protocol
identity with small fixtures. Independently validate the real selected observations,
reference membership, occurrence labels, coordinates and scope counts. Run format,
lint and full tests, bump version, land via PR and deploy from clean main. Publish
and download-verify public evidence artifacts; link the receipt from #681/#678 and
the Presto consumer PR. Software tests alone do not qualify this dataset.
