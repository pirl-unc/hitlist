"""Read-only inventory of hitlist's existing cache locations (#589)."""

from __future__ import annotations

import os
from pathlib import Path

import datacache


def list_cache_files(*, verify: bool = False, include_unregistered: bool = True) -> list[dict]:
    """List cached files and registered/mirrored paths, without creating anything.

    ``verify`` checks trusted mirrored-asset hashes. Otherwise their sizes are
    checked, but ``verified`` remains false. Recorded hashes are historical
    receipts, never proof that the current file was hashed by this inspection.
    Missing registered paths and assets remain visible. Set include_unregistered
    to false to limit the view to named datasets and mirrored assets. Directory enumeration
    failures appear as inaccessible rows. Hidden transfer state and cache
    metadata are omitted; symlink directories are not traversed.
    """
    from .downloads import data_asset_dir, data_assets, data_dir, list_datasets
    from .proteome import proteome_index_cache_dir

    entries = {}

    def add(path, kind, name=None, metadata=None):
        path = Path(path)
        # Resolve aliases for deduplication, retaining the first displayed path.
        key = os.path.realpath(path)
        entry = entries.setdefault(key, {"path": path, "kinds": set(), "names": set()})
        entry["kinds"].add(kind)
        if name:
            entry["names"].add(name)
        if metadata:
            entry["metadata"] = metadata

    for name, record in list_datasets().items():
        add(record["path"], "registered dataset", name)
    assets_root = data_asset_dir()
    for name, metadata in data_assets().items():
        add(assets_root / name, "mirrored asset", name, metadata)

    roots = [(data_dir(), "built data"), (assets_root, "data asset cache")]
    legacy = Path.home() / ".hitlist"
    if legacy.exists():
        roots.append((legacy, "legacy cache"))
    roots.append((proteome_index_cache_dir(), "proteome index"))
    errors = []
    visited = set()

    def scan_error(error):
        errors.append(
            {
                "path": str(error.filename),
                "status": "inaccessible",
                "verified": False,
                "size_bytes": None,
                "kinds": ["cache directory"],
                "names": [],
                "source_url": None,
                "fetched_at": None,
                "recorded_sha256": None,
                "error": str(error),
            }
        )

    for root, kind in roots if include_unregistered else ():
        # Missing cache roots are normal and must not be created by listing.
        try:
            if not root.exists():
                continue
        except OSError as error:
            scan_error(error)
            continue
        for directory, dirs, files in os.walk(root, onerror=scan_error):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            key = os.path.realpath(directory)
            if key in visited:
                dirs[:] = []
                continue
            visited.add(key)
            for name in sorted(files):
                if (
                    name.startswith(".")
                    or name.endswith((".tmp", "_meta.json"))
                    or name == "manifest.json"
                ):
                    continue
                path = Path(directory) / name
                file_kind = "proteome index" if path.parent == proteome_index_cache_dir() else kind
                add(path, file_kind)

    rows = []
    for entry in entries.values():
        meta = entry.get("metadata", {})
        state = datacache.inspect_file(
            entry["path"],
            expected_size=meta.get("size_bytes"),
            expected_sha256=meta.get("sha256") if verify else None,
        )
        rows.append(
            {
                "path": state.path,
                "status": state.status,
                "verified": state.verified,
                "size_bytes": state.size,
                "kinds": sorted(entry["kinds"]),
                "names": sorted(entry["names"]),
                "source_url": state.source_url,
                "fetched_at": state.fetched_at,
                "recorded_sha256": state.recorded_sha256,
                "error": str(state.error) if state.error else None,
            }
        )
    return sorted(rows + errors, key=lambda row: row["path"])
