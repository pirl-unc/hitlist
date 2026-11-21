# Specification: `fetch()` Size/Space Preflight, Early Terms Notice, Stale Registration Reporting & Typed Dataset/Space Errors (`#884`, `v2.3.0`)

## Goal

Make `hitlist.downloads` and `hitlist data` safe, transparent, and machine-readable before multi-GB dataset acquisitions (e.g., IEDB 7.8 GB, mirrored data assets, and future PRIDE/MassIVE raw archive imports like `#858`, `#849`, `#837`):

1. **Read-only fetch plan (`downloads.preflight(name, *, force=False)`) & CLI (`hitlist data fetch <name> --dry-run`)**:
   - Returns JSON-serializable metadata without touching the network or creating target directories:
     - `name`: dataset name
     - `path`: string path where the file will land
     - `registered`: whether `name` is present in `manifest.json`
     - `exists`: whether `path` currently exists on disk
     - `stale`: whether `name` is registered in `manifest.json` but its registered file path no longer exists
     - `would_download`: `True` when `force=True` or `path` does not exist on disk
     - `expected_size_bytes`: declared `size_bytes` / `expected_size` (`int`) or `None` when unknown
     - `free_bytes`: free space on the filesystem hosting the nearest existing ancestor of `path`
     - `enough_space`: `None` when `expected_size_bytes` is `None`, else `free_bytes >= expected_size_bytes`
     - `terms`: governance/terms-of-use URL (`str`) or `None`
   - Covers `FETCHABLE_DATASETS` (`size_bytes` or `expected_size`, including `iedb` at `7_800_000_000` bytes and `netmhcstabpan` at `799_880` bytes) and mirrored `data_assets()` (`size_bytes`).
   - Refuses `depmap` (directing callers to its individual `_DEPMAP_FILES`), `MANUAL_DATASETS`, and unknown dataset names with `DatasetError`.

2. **Space preflight enforcement & typed errors (`hitlist/errors.py`)**:
   - Add `DatasetError(HitlistError, ValueError)` for unknown/manual/bundle-only dataset names so existing `except ValueError` callers keep working and `hitlist.cli.main` prints a single `Error: ...` line with exit code 1.
   - Add `InsufficientSpaceError(DownloadError)` carrying `.path`, `.required_bytes`, and `.available_bytes` attributes (subclassing `DownloadError` -> `HitlistError` -> `RuntimeError`), aligning with upstream `openvax/datacache#136` while `openvax/datacache#135` and `#136` remain open. Also translate any future upstream `datacache` `InsufficientSpace`/`InsufficientSpaceError` inside `_typed_download_errors()`.
   - Both `fetch()` and `fetch_data_asset()` check space before starting a download (when downloading or `force=True`) and raise `InsufficientSpaceError` when `enough_space is False`, never touching the network or writing partial files.
   - Upgrade short-body `min_size_bytes` rejection in `fetch()` from bare `RuntimeError` to `IntegrityError` (a `HitlistError(RuntimeError)` subclass) so CLI prints one clean `Error: ...` line.

3. **Early terms-of-use notice & stale-registration reporting**:
   - In `fetch()`, if `entry` is registered and `Path(entry["path"])` is missing, print a stale-registration notice to `stderr` before starting the download.
   - Print the terms-of-use notice (`ds["terms"]`) to `stderr` **before** calling `download_to_file()`, not after transfer completion.

4. **Existence visibility in `list_datasets()` and `info()`**:
   - Add `"exists": bool` to every entry returned by `list_datasets()` and to every branch of `info()` (registered, mirrored asset, `depmap`, `FETCHABLE_DATASETS`, `MANUAL_DATASETS`).

5. **Public API exports, offline test hygiene, docs & release wiring**:
   - Export `preflight`, `list_cache_files`, `fetch_all_data_assets`, `list_proteomes`, `DatasetError`, and `InsufficientSpaceError` lazily in `hitlist/__init__.py`.
   - Keep `tests/test_export.py`, `tests/test_pmhc_query.py`, and `tests/test_versioned_datasets.py` hermetically offline and independent of host free-disk state.
   - Update `docs/downloads.md`, `docs/legacy-1x.md`, `tests/data/cli_help_snapshot.txt`, `.github/workflows/release-build.yml`, and bump `hitlist/version.py` to `2.3.0`.
