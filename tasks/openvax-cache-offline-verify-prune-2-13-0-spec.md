# Hitlist v2.13.0 Spec: OPENVAX_DATA_CACHE Fallback, Offline Mode, and `hitlist data verify|prune` (#960)

## Scope

1. **Honor `OPENVAX_DATA_CACHE` fallback (`hitlist/downloads.py`, `hitlist/cli_data.py`):**
   - Add `OPENVAX_DATA_CACHE_ENV_VAR = "OPENVAX_DATA_CACHE"` and origin `"openvax_env"` in `DATA_DIR_ORIGINS` and `_DATA_DIR_ORIGIN_LABELS`.
   - `resolve_data_dir()` checks `set_data_dir()` override -> `$HITLIST_DATA_DIR` -> `$OPENVAX_DATA_CACHE/hitlist` -> populated legacy `~/.hitlist` -> `datacache.get_data_dir("hitlist")`.
   - `data_asset_dir()` also honors `$OPENVAX_DATA_CACHE/hitlist` when set (with `datacache.resolve_cache_dir` compatibility when present) so both built indexes and mirrored assets respect HPC shared cache roots while preserving existing `~/.hitlist` and `~/Library/Caches/hitlist` installs untouched.

2. **Global offline kill-switch (`OPENVAX_OFFLINE` / `DATACACHE_OFFLINE` / `--offline`):**
   - Add `OfflineError(DownloadError)` in `hitlist/errors.py` and `is_offline()` / `set_offline()` in `hitlist/downloads.py`.
   - Guard `download_to_file()`, `fetch()`, `fetch_data_asset()`, `fetch_all_data_assets()`, and `refresh()` so any missing or `force=True` network fetch raises `OfflineError` before touching the network while valid cached files continue to resolve offline.
   - Wire `--offline` onto `hitlist data fetch`, `hitlist data fetch-all`, and `hitlist data refresh`.

3. **Add `hitlist data verify` and `hitlist data prune` subcommands:**
   - Add `verify_cache_files(include_unregistered=False)` wrapping `list_cache_files(verify=True)` and CLI `hitlist data verify [--all] [--json]` (exits non-zero when any checked file fails integrity/validity).
   - Add `prune_cache_bundles(keep=1, apply=False, root=None)` wrapping `datacache.list_bundles` / `datacache.prune_bundles` and CLI `hitlist data prune [--keep N] [--yes] [--json]` (dry-run by default; deletes only with `--yes`).
