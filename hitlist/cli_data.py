# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""``hitlist data``: dataset registry, cache inventory and legacy build aliases."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from .cli_build import (
    _add_build_observations_args,
    _add_build_proteomes_args,
    _data_build,
    _data_fetch_proteomes,
)
from .cli_common import print_subgroup_help
from .downloads import (
    DATA_DIR_ENV_VAR,
    available_datasets,
    data_asset_dir,
    fetch,
    get_path,
    info,
    list_cache_files,
    refresh,
    remove,
    resolve_data_dir,
)
from .downloads import (
    register as register_dataset,
)


def _fmt_size(size: int) -> str:
    if size > 1_000_000_000:
        return f"{size / 1e9:.1f} GB"
    if size > 1_000_000:
        return f"{size / 1e6:.1f} MB"
    if size > 1_000:
        return f"{size / 1e3:.1f} KB"
    return f"{size} B"


# Upper bound on how many top-level files ``_dir_summary`` will ``stat``.  The
# proteome index cache routinely holds thousands, and ``hitlist data list``
# should not spend a visible fraction of a second measuring them.
_DIR_SUMMARY_STAT_LIMIT = 256


def _dir_summary(path: Path) -> str:
    """Cheap one-line "what's in here" for a cache directory.

    One ``os.scandir`` of the top level and nothing deeper.  Classification
    comes from the dirent, so counting costs no ``stat`` calls; only the size
    pass stats, and only when there are at most ``_DIR_SUMMARY_STAT_LIMIT``
    files.  (``Path.iterdir`` + ``Path.is_file`` would stat every entry *before*
    that guard could apply, which measured ~90 ms on a 7.5k-file index cache.)

    Sizes cover top-level regular files only, so a directory with
    ``proteomes/`` or an index cache under it holds more than the number shown
    — hence the explicit subdirectory count.
    """
    try:
        with os.scandir(path) as it:
            entries = list(it)
    except FileNotFoundError:
        return "not created yet"
    except OSError as err:
        return f"unreadable ({err.strerror or err})"
    # Emptiness is decided by the entry count, not by the classification below:
    # a dangling symlink, a socket or a FIFO is neither a file nor a directory,
    # and reporting a directory that holds one as "empty" would be a lie.
    if not entries:
        return "empty"
    files = [e for e in entries if e.is_file()]
    n_subdirs = sum(1 for e in entries if e.is_dir())
    n_other = len(entries) - len(files) - n_subdirs
    parts = []
    if len(files) > _DIR_SUMMARY_STAT_LIMIT:
        parts.append(f"{len(files):,} files (size not measured)")
    elif files:
        total = 0
        for f in files:
            # A concurrent build renames ``.partial``/``.tmp`` siblings into
            # place in this very directory, so a name listed a moment ago can
            # be gone by the time we stat it. A summary line is not worth a
            # traceback.
            with contextlib.suppress(OSError):
                total += f.stat().st_size
        parts.append(f"{_fmt_size(total)} in {len(files):,} files")
    if n_subdirs:
        parts.append(f"{n_subdirs:,} subdirectories")
    if n_other:
        parts.append(f"{n_other:,} other entries")
    return ", ".join(parts)


#: How each :data:`hitlist.downloads.DATA_DIR_ORIGINS` value reads to a user.
#: ``test_cli_data_dirs`` asserts this covers the whole tuple, so a new
#: resolution rule cannot quietly print a bare token.
_DATA_DIR_ORIGIN_LABELS = {
    "override": "set_data_dir()",
    "env": f"${DATA_DIR_ENV_VAR}",
    "legacy": "legacy ~/.hitlist, still supported",
    "default": "datacache default",
}


def _data_location_lines() -> list[str]:
    """Every directory hitlist reads or writes, and why it resolved there (#291).

    ``data_dir()`` (built indexes) and ``data_asset_dir()`` (mirrored CSVs
    fetched through datacache) are the same directory on a fresh install and
    diverge once the first is overridden or still on the legacy location — so
    they are printed as one line or two accordingly.
    """
    from .proteome import default_proteome_index_cache_dir, proteome_index_cache_dir

    # One resolution, not two: `data_dir()` + `data_dir_origin()` would run it
    # twice and could print a path that disagrees with its own explanation.
    built, origin = resolve_data_dir()
    assets = data_asset_dir()
    why = _DATA_DIR_ORIGIN_LABELS.get(origin, origin)
    index_cache = proteome_index_cache_dir()
    index_why = (
        "data directory"
        if index_cache == default_proteome_index_cache_dir()
        else "set_disk_cache_dir()"
    )
    rows = (
        [("built indexes + assets", built, why)]
        if built == assets
        else [("built indexes", built, why), ("data assets", assets, "datacache")]
    )
    from .dataset import dataset_snapshot

    with dataset_snapshot(root=built) as generation:
        if generation != built:
            rows.append(("current dataset", generation, "datacache generation"))
    rows.append(("proteome index cache", index_cache, index_why))
    rows.append(("UniProt references", built / "uniprot", "data directory"))
    width = max(len(label) for label, _, _ in rows)
    return [
        f"  {label:<{width}}  {path}  ({why})  — {_dir_summary(path)}" for label, path, why in rows
    ]


def _print_data_locations() -> None:
    print("Data directories:")
    for line in _data_location_lines():
        print(line)


def _data_dirs(args: argparse.Namespace) -> None:
    _print_data_locations()


def _data_list(args: argparse.Namespace) -> None:
    rows = list_cache_files(
        verify=getattr(args, "verify", False),
        include_unregistered=getattr(args, "all", False) or getattr(args, "json", False),
    )
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return
    print(f"{'Status':<14} {'Size':>12}  {'Integrity':<12} Path")
    for row in rows:
        size = _fmt_size(row["size_bytes"]) if row["size_bytes"] is not None else "—"
        integrity = "verified" if row["verified"] else "unchecked"
        print(f"{row['status']:<14} {size:>12}  {integrity:<12} {row['path']}")
        if row["names"]:
            print(f"  name: {', '.join(row['names'])}")
        if row["source_url"]:
            print(f"  source: {row['source_url']}  fetched: {row['fetched_at']}")
        if row["error"] and row["status"] != "missing":
            print(f"  {row['error']}")
    _print_data_locations()


def _data_available(args: argparse.Namespace) -> None:
    datasets = available_datasets()
    registered = {
        name
        for row in list_cache_files(include_unregistered=False)
        if row["status"] == "available"
        for name in row["names"]
    }
    print(f"{'Name':<12} {'Status':<12} Description")
    print("-" * 75)
    for name, desc in sorted(datasets.items()):
        status = "installed" if name in registered else ""
        print(f"{name:<12} {status:<12} {desc}")
    print()
    _print_data_locations()


def _data_register(args: argparse.Namespace) -> None:
    try:
        p = register_dataset(args.name, args.path, description=args.description)
        print(f"Registered '{args.name}' -> {p}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_fetch(args: argparse.Namespace) -> None:
    try:
        p = fetch(args.name, force=args.force)
        print(f"Ready: {p}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_fetch_all(args: argparse.Namespace) -> None:
    from .downloads import fetch_all_data_assets

    fetch_all_data_assets(force=args.force)


def _data_refresh(args: argparse.Namespace) -> None:
    try:
        p = refresh(args.name)
        print(f"Refreshed: {p}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_info(args: argparse.Namespace) -> None:
    try:
        d = info(args.name)
        print(json.dumps(d, indent=2, default=str))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_path(args: argparse.Namespace) -> None:
    try:
        print(str(get_path(args.name)))
    except (KeyError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _data_remove(args: argparse.Namespace) -> None:
    if not remove(args.name, delete_file=args.delete):
        print(f"No dataset '{args.name}' was registered (nothing to do).", file=sys.stderr)
        sys.exit(1)
    if args.delete:
        print(f"Unregistered and deleted '{args.name}'.")
    else:
        print(f"Unregistered '{args.name}' (file kept on disk).")


def _data_list_proteomes(args: argparse.Namespace) -> None:
    from .downloads import list_proteomes

    proteomes = list_proteomes()
    if not proteomes:
        print("No proteomes registered.  Run: hitlist build proteomes")
        return
    print(f"{len(proteomes)} proteomes registered:\n")
    for species, meta in sorted(proteomes.items()):
        kind = meta.get("kind", "?")
        if kind == "ensembl":
            print(
                f"  {species:35s}  ensembl (species={meta.get('species')}, "
                f"release={meta.get('release')})"
            )
        else:
            size = meta.get("size_bytes", 0)
            print(f"  {species:35s}  uniprot:{meta.get('proteome_id')}  ({size:,} bytes)")


def _build_data_parser(sub: argparse._SubParsersAction) -> None:
    dp = sub.add_parser("data", help="Manage external datasets")
    dp.set_defaults(_subgroup_parser=dp, func=_handle_data)
    ds = dp.add_subparsers(dest="data_command")
    from .cli_uniprot import add_uniprot_parser

    add_uniprot_parser(ds)

    p = ds.add_parser("list", help="Show registered datasets or the complete cache inventory")
    p.add_argument("--all", action="store_true", help="Inspect files across all cache locations")
    p.add_argument("--verify", action="store_true", help="Verify trusted mirrored-asset hashes")
    p.add_argument("--json", action="store_true", help="Print the complete inventory as JSON")
    ds.add_parser("available", help="Show all known datasets")
    ds.add_parser(
        "dirs",
        help="Show every directory hitlist reads/writes, and why it resolved there",
    )

    p = ds.add_parser("register", help="Register a local file")
    p.add_argument("name", help="Dataset name")
    p.add_argument("path", help="Path to the file")
    p.add_argument("--description", "-d", help="Optional description")

    p = ds.add_parser("fetch", help="Download a fetchable dataset")
    p.add_argument("name", help="Dataset name")
    p.add_argument("--force", "-f", action="store_true", help="Re-download")

    p = ds.add_parser(
        "fetch-all",
        help="Fetch ALL mirrored data-asset CSVs (paper-derived) into the cache dir",
    )
    p.add_argument("--force", "-f", action="store_true", help="Re-download even if cached")

    p = ds.add_parser("refresh", help="Re-download a fetchable dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("info", help="Detailed metadata for a dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("path", help="Print path to a dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("remove", help="Unregister a dataset")
    p.add_argument("name", help="Dataset name")
    p.add_argument("--delete", action="store_true", help="Also delete the file")

    p = ds.add_parser(
        "build",
        help="DEPRECATED — use `hitlist build observations`. Builds observations.parquet.",
    )
    _add_build_observations_args(p)

    p = ds.add_parser(
        "fetch-proteomes",
        help="DEPRECATED — use `hitlist build proteomes`. Auto-fetch reference proteomes.",
    )
    _add_build_proteomes_args(p)

    ds.add_parser("list-proteomes", help="List downloaded reference proteomes")


_DEPRECATED_DATA_SUBCOMMANDS = {
    "build": "hitlist build observations",
    "fetch-proteomes": "hitlist build proteomes",
}


def _handle_data(args: argparse.Namespace) -> None:
    from .cli_uniprot import handle_uniprot

    handlers = {
        "uniprot": handle_uniprot,
        "list": _data_list,
        "available": _data_available,
        "dirs": _data_dirs,
        "register": _data_register,
        "fetch": _data_fetch,
        "fetch-all": _data_fetch_all,
        "refresh": _data_refresh,
        "info": _data_info,
        "path": _data_path,
        "remove": _data_remove,
        "build": _data_build,
        "fetch-proteomes": _data_fetch_proteomes,
        "list-proteomes": _data_list_proteomes,
    }
    if args.data_command is None:
        print_subgroup_help(args._subgroup_parser)
        sys.exit(1)
    if args.data_command in _DEPRECATED_DATA_SUBCOMMANDS:
        new_path = _DEPRECATED_DATA_SUBCOMMANDS[args.data_command]
        print(
            f"Note: `hitlist data {args.data_command}` is deprecated; use `{new_path}` instead.",
            file=sys.stderr,
        )
    handlers[args.data_command](args)


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist data``."""
    _build_data_parser(sub)
