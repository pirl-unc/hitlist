# #589: datacache consolidation specification

The released cache-root and proteome-root fixes stay intact. Move transfer,
retry, decompression, integrity, and provenance mechanics into datacache >=1.14.0.
Hitlist continues to own dataset names, biological metadata, terms notices,
cache-root selection, and its public Path-returning download API.

## First release: downloads and inventory

- Preserve download_to_file's default raw bytes, explicit decompression heuristic,
  cache messages, force refresh, Path result, socket timeout, and RuntimeError
  context with original cause. Delegate atomic publication and bounded retries.
- Expose optional expected_size, expected_sha256 and resume arguments. Turn on
  resume for mirrored assets whose registry supplies both size and hash; do not
  infer trustworthy integrity metadata from HTTP Content-Length. Raw downloads
  without hashes may opt into datacache's size + strong-ETag protocol.
- Record datacache provenance for newly acquired files. Legacy cache hits remain
  read-only and have unknown provenance unless a pre-existing manifest says more.
- Keep list_datasets() compatible. Make `data list` include mirrored assets,
  and add `data list --all` / list_cache_files() for one inventory across built data,
  mirrored assets, registered external paths, proteomes, and proteome indexes.
  Deduplicate overlapping locations, retain missing registered paths, distinguish
  availability from hash verification, and omit staging/metadata files. Listing
  must never download, relocate, create directories, or write receipts. Optional
  --verify checks trusted asset hashes; ordinary listing checks sizes only.
- Use local HTTP fixtures to prove raw gzip/HTML behavior, ZIP selection,
  retryable vs permanent responses, failed refresh preservation, interruption
  and range resumption, and receipt-based inspection. Preserve existing public
  tests while replacing tests of the removed urllib implementation.

## Subsequent registry migration

Audit downstream tsarina callers before selecting an adapter. Existing paths
`<root>/<name>/<version>/<filename>`, pre-download local_path(), error_cls,
single-Path return values, status keys and legacy manifest/cache reuse are
compatibility requirements. Datacache's managed generation store cannot take
over a populated legacy directory. Use a separate managed namespace and an
explicit compatibility layer if adopting that store; never silently move or
delete old files. Require offline legacy-cache and failed-refresh tests. Re-plan
if that adapter adds more generic cache machinery than it removes; file any
upstream API gap before choosing a workaround.

## Release verification

Each PR bumps the patch version and runs format.sh, lint.sh, test.sh (unit plus
integration with the verified CI corpus). Review the diff and current-head CI,
merge, deploy from clean main, and verify published wheel/sdist hashes on PyPI.
Only then mark the release complete. Finish the independent curation/training
assessment during long validation runs.

## Registry implementation decision

The bundle adapter would add a second namespace, symlink publication and parallel
receipts. Instead, datacache #83 / PR #84 now supplies VersionedFileRegistry,
which preserves the established fixed paths and manifest schema directly. It
also fixes the reproduced concurrent manifest lost-update race (hitlist #618).
Hitlist will subclass that shared implementation solely for its public error
default, human messages, 300-second timeout and literal-URL transform policy.
The shared base owns version resolution, paths, cache reuse, bounded-memory
receipt hashing, locking and atomic manifest writes. Bump hitlist to 1.63.16
and require the published datacache 1.15.0. Compare old/new public outputs on
legacy fixtures and run normal format/lint/full unit/integration/release gates.

Registry review: datacache 1.15.0 shipped from clean master with 764 tests
and both full CI matrices passing. Published wheel/sdist hashes match. The
hitlist adapter's offline legacy results were compared directly with released
1.63.14; paths, Path returns and status dictionaries match, and no cache metadata
is created on reuse. Focused tests pass against the published package. Full
hitlist gates and PyPI publication remain pending.
