# Specification: Uniform Data Download Registry, CLI & Python API (`v2.4.0`)

## Goal

Ensure every auto-downloadable dataset and managed bundle in `hitlist` is discoverable, preflightable, fetchable, inspectable, path-resolvable, and removable through ONE uniform Python API (`hitlist.downloads` / `hitlist.*`) and ONE uniform CLI (`hitlist data ...`), with consistent typed `HitlistError` handling and zero split-cache bugs.

## Design

1. **Unified Managed Dataset Catalog (`hitlist/_dataset_catalog.py`)**:
   - Exposes a clean, lazy-loading registry of managed external datasets & bundles that previously had bespoke downloaders outside `hitlist.downloads`:
     - `uniprot/human` (default release), `uniprot/human/2026_03`, `uniprot/human/2015_10` -> backed by `hitlist.uniprot`
     - `hla-ligand-atlas` -> backed by `hitlist.atlas.atlas_sources()`
     - `hla-ligand-atlas-crosswalk` -> backed by `hitlist.atlas.atlas_crosswalk_sources()` (from `hitlist/data/atlas_community_crosswalk.yaml`)
     - `tcell-assays` -> backed by `hitlist.tcell_assays.tcell_assay_sources()`
     - `perturbation-quantities` -> backed by `hitlist.perturbation_quantities.fetch_perturbation_sources()`
     - `canine-bulk/pfizer_2026_03`, `canine-bulk/pfizer_2026_03/table`, `canine-bulk/pfizer_2026_03/reference`, `canine-bulk/labrador_2024_01`, `canine-bulk/labrador_2024_01/table`, `canine-bulk/labrador_2024_01/reference` -> backed by `hitlist.canine_bulk`
     - `pxd013455` (portable archive) -> backed by `hitlist.pxd013455_download.load_pxd013455_bundle()`
     - `pxd013455-sources` (raw recovery) -> backed by `hitlist.pxd013455.recover_pxd013455_sources()`
     - `dlbcl-35561310-supplements` -> extracts checksum-pinned `advancesADV2021006069-suppl3.xlsx` and `advancesADV2021006069-suppl5.xlsx` from PMC9327544's Europe PMC `supplementaryFiles` ZIP into `data_dir() / "dlbcl_35561310_supplements"`; `load_dlbcl_supplement_rows(source_directory=None)` resolves automatically via `downloads.get_path("dlbcl-35561310-supplements")`.
   - Every entry implements uniform `available`, `preflight(force=...)`, `fetch(force=...)`, `info()`, `get_path()`, `remove(delete_file=...)`, and `installed_cache_entries()` (for `cache_inventory.py`).
   - All fetches run free-space preflight (`InsufficientSpaceError`), print `terms`/license notices before downloading, and wrap `datacache`/`OSError`/`ValueError` failures in `HitlistError` (`DownloadError`, `IntegrityError`, `InsufficientSpaceError`, `DatasetError`).

2. **Fix `netmhcstabpan` default-cache split**:
   - `hitlist/netmhcstabpan.py::fetch_netmhcstabpan_source(cache_directory=None, force=False, verbose=True)` delegates to `downloads.fetch("netmhcstabpan", force=force)` when `cache_directory is None`, so it writes to `data_dir()/netmhcstabpan-1.0-Stability.txt`, registers `"netmhcstabpan"` in `manifest.json`, and makes `load_netmhcstabpan_stability()` / `scan_netmhcstabpan_stability()` work immediately. Explicit `cache_directory` callers keep existing behavior.

3. **Composite Bundle `preflight()` + Uniform `remove()` in `hitlist/downloads.py`**:
   - `preflight("depmap", force=False)` now returns a composite bundle plan (`kind: "bundle"`, aggregated `expected_size_bytes`, `would_download`, `enough_space`, `terms`, `files: [...]`) instead of raising `DatasetError`.
   - `remove(name, delete_file=True)` supports mirrored `data_assets()` and managed catalog datasets/bundles in addition to `manifest.json["datasets"]`.
   - `list_cache_files()` surfaces installed managed bundles with their canonical names (skipping uninstalled ones so `include_unregistered=False` stays clean).

4. **Uniform CLI (`hitlist/cli_data.py`) & Top-Level Python API (`hitlist/__init__.py`)**:
   - Add `hitlist data preflight <name> [--force]` (same JSON as `hitlist data fetch <name> --dry-run [--force]`).
   - Add `--json` and dynamic column width to `hitlist data available`.
   - Export `data_dir`, `data_asset_dir`, `set_data_dir`, `data_assets`, `fetch_data_asset`, `atlas_crosswalk_sources`, `fetch_dlbcl_supplements` lazily in `hitlist/__init__.py`.
   - Regenerate `tests/data/cli_help_snapshot.txt`, update `docs/downloads.md`, `.github/workflows/release-build.yml`, and bump `hitlist/version.py` to `2.4.0`.
