# Versioned UniProt references

## Request and decisions

Package human UniProt 2015_10 as a Hitlist-managed data asset and track multiple
UniProt releases. The user selected managed download/cache rather than adding
the sequences to every wheel. Keep releases reproducible and storage bounded.
Preserve the existing general/species download APIs and their caches.

The original Bekker-Jensen paper names all human Swiss-Prot and TrEMBL entries
plus isoforms. A comparison of 661,142 original packaged peptides with the
PXD013455 reanalysis reference explains 661,025 exactly; all 117 residuals are
explained by historical canonical/isoform sequences. BRCA2 sequence version 2
points to a pre-2015_11 snapshot. This supports 2015_10 as a candidate but does
not prove that it was the exact original searched FASTA. Reference-release
identity and a study's search-reference identity remain separate facts.

## Source acquisition

Recover the actual human subset of the official archived release, including
canonical entries and reviewed isoforms. The archive metadata identifies
knowledgebase2015_10.tar.gz (31,820,941,423 bytes, MD5
ccf2882074ba5d9c9ffa6f98f4c5d6af) and the Swiss-Prot-only archive
(1,473,521,091 bytes, MD5 5ec54827ce179421278826c09f9bc2e6).
Inspect the archive layout and stream extraction; never expand/store the full
database. Bound output and per-record memory, preserve complete source hashes,
release metadata, taxonomy selection, isoform reconstruction/source, counts and
license attribution. Validate completeness against official release statistics.
If acquisition exposes an obstacle, record it and revise this section before
changing the proposed reference identity. Never ship the mixed-version audit
reference under an exact UniProt release name.

Publish a human-only immutable asset with its build receipt to Hitlist's data
release, then pin its exact URL, size and SHA256 in the package. Test download
and reuse against actual published bytes. Add a second real release when it
can be acquired directly from the official source with verified release
metadata; multiple-version behavior must also be covered by independent tests.

## Data contract and API

Use a bundled YAML catalog of explicit collection/release definitions. Each
entry names upstream release, taxonomy, selection (all entries versus reference
proteome; reviewed/unreviewed; isoforms), source/build provenance, immutable
download URL, size, SHA256, sequence counts and license. No mutable `latest`
alias. The requested human collection has a pinned default release.

Reuse datacache's atomic/resumable transfer and integrity checks directly.
Its general fixed-path registry has a last-version-only root receipt; the
UniProt catalog and per-file receipts avoid that ambiguity without introducing
a second writable manifest. Add a UniProt-specific API that resolves/fetches a release, reports its
metadata and lists all available/cached releases. Files coexist under the
Hitlist data directory with collection and release in their paths. On cache
reuse, verify the trusted catalog hash; corruption raises and explicit force
repairs. A root receipt's last-download entry must never be mistaken for the
identity of another cached release. Unknown releases fail explicitly.

Expose the capability in `hitlist data` with fetch, path, info, list and remove
operations and JSON output. Existing `data dirs` and inventory include this
managed location and trusted per-version metadata. Reading metadata/listing
must neither download nor create directories. Removal is explicit and scoped
to managed files, with unrelated/manual FASTAs untouched.

Enforce a per-asset limit and a total UniProt-cache budget before transfer,
including partial downloads and replacement headroom; never silently evict
historical versions. Serialize writers so concurrent fetches cannot bypass the
budget. Defaults are 256 MiB per asset and 1 GiB total logical file bytes, with
2 MiB control/transfer headroom per new download. New transfers use datacache's
bounded POSIX resumable downloader. Inspection and cached reads are portable;
new non-POSIX downloads fail explicitly because the general downloader checks
size only after writing. These are file-byte limits, not filesystem quotas.
Use a plain installed FASTA so existing FASTA consumers can use it.
Keep large reference payloads outside wheel/sdist and Git history.

Reference descriptors/provenance must be consumable by detectability exports
without asserting that a release was searched in a particular study. Preserve
the exact FASTA hash and reference identity in any supported integration.

## Verification

- Two releases coexist; defaults are fixed; unknown releases and invalid paths
  fail before network access; metadata is correct after fetching either order.
- Interrupted, oversized and corrupt transfers preserve complete installed
  files. Cache reads verify expected hash. Cache budget includes replacement
  and partial bytes; no automatic eviction or unrelated file deletion.
- CLI/API parity, relocated data directory, read-only inspection and cache
  inventory hashes. New files appear in built wheel/sdist metadata while
  reference payloads do not.
- Actual historical extraction has independent sequence/count/peptide coverage
  checks, complete source receipts, and reproducible deterministic output.
- Run format.sh, lint.sh and test.sh. Use CI for full corpus checks if the
  unchanged local memory guard refuses; preserve the refusal in the record.
- Version bump, PR, exact-head checks, merge, clean-main deploy.sh release
  build, PyPI upload and downloaded-byte verification.

## Review

The 2015_10 source is currently blocked by official HTTPS TLS timeouts from
both the workstation and GitHub Actions (run 37811660769); the FTP release
symlink points to an inaccessible directory. Keep historical recovery open.
Prepare the independently verified REST release 2026_03 as the first available
managed asset; never substitute it for a request for 2015_10. The optional
user explicitly confirmed shipping the cache with 2026_03 and keeping 2015_10
recovery open in #658. A later retry of the historical extraction remains
independent of this release's acceptance criteria.

Focused client/extractor/CLI/inventory/detectability/public-API tests passed
(180). The first full CI run found one shared-loader integration failure:
the new catalog used plain safe YAML parsing rather than the existing
duplicate-key-rejecting curation loader. All other unit tests passed (2774 on
Python 3.12). Use the shared loader, add a duplicate-catalog regression and
repeat the complete final-head checks before merge. Full release still pending.
