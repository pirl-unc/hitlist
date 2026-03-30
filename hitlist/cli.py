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
# limitations under the License.

"""CLI for hitlist data management.

Usage::

    hitlist data list                         # show registered datasets
    hitlist data available                    # show all known datasets
    hitlist data register iedb /path/to/file  # register a manual download
    hitlist data fetch hpv16                  # auto-download
    hitlist data refresh hpv16                # re-download
    hitlist data info iedb                    # detailed metadata
    hitlist data path iedb                    # print file path
    hitlist data remove iedb                  # unregister
    hitlist data remove iedb --delete         # unregister + delete file
"""

from __future__ import annotations

import argparse
import json
import sys

from .downloads import (
    available_datasets,
    data_dir,
    fetch,
    get_path,
    info,
    list_datasets,
    refresh,
    register,
    remove,
)


def _fmt_size(size: int) -> str:
    if size > 1_000_000_000:
        return f"{size / 1e9:.1f} GB"
    if size > 1_000_000:
        return f"{size / 1e6:.1f} MB"
    if size > 1_000:
        return f"{size / 1e3:.1f} KB"
    return f"{size} B"


def _data_list(args: argparse.Namespace) -> None:
    datasets = list_datasets()
    if not datasets:
        print("No datasets registered.")
        print(f"Data directory: {data_dir()}")
        print("Run 'hitlist data available' to see known datasets.")
        return
    print(f"{'Name':<12} {'Size':>12}  {'Date':<12} Description")
    print("-" * 75)
    for name, ds in sorted(datasets.items()):
        size_str = _fmt_size(ds.get("size_bytes", 0))
        date = ds.get("registered", "")[:10]
        desc = ds.get("description", "")
        print(f"{name:<12} {size_str:>12}  {date:<12} {desc}")
    print(f"\nData directory: {data_dir()}")


def _data_available(args: argparse.Namespace) -> None:
    datasets = available_datasets()
    registered = set(list_datasets().keys())
    print(f"{'Name':<12} {'Status':<12} Description")
    print("-" * 75)
    for name, desc in sorted(datasets.items()):
        status = "installed" if name in registered else ""
        print(f"{name:<12} {status:<12} {desc}")


def _data_register(args: argparse.Namespace) -> None:
    try:
        p = register(args.name, args.path, description=args.description)
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
    remove(args.name, delete_file=args.delete)
    if args.delete:
        print(f"Unregistered and deleted '{args.name}'.")
    else:
        print(f"Unregistered '{args.name}' (file kept on disk).")


def _build_data_parser(sub: argparse._SubParsersAction) -> None:
    dp = sub.add_parser("data", help="Manage external datasets")
    ds = dp.add_subparsers(dest="data_command")

    ds.add_parser("list", help="Show registered datasets")
    ds.add_parser("available", help="Show all known datasets")

    p = ds.add_parser("register", help="Register a local file")
    p.add_argument("name", help="Dataset name")
    p.add_argument("path", help="Path to the file")
    p.add_argument("--description", "-d", help="Optional description")

    p = ds.add_parser("fetch", help="Download a fetchable dataset")
    p.add_argument("name", help="Dataset name")
    p.add_argument("--force", "-f", action="store_true", help="Re-download")

    p = ds.add_parser("refresh", help="Re-download a fetchable dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("info", help="Detailed metadata for a dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("path", help="Print path to a dataset")
    p.add_argument("name", help="Dataset name")

    p = ds.add_parser("remove", help="Unregister a dataset")
    p.add_argument("name", help="Dataset name")
    p.add_argument("--delete", action="store_true", help="Also delete the file")


def _handle_data(args: argparse.Namespace) -> None:
    handlers = {
        "list": _data_list,
        "available": _data_available,
        "register": _data_register,
        "fetch": _data_fetch,
        "refresh": _data_refresh,
        "info": _data_info,
        "path": _data_path,
        "remove": _data_remove,
    }
    if args.data_command is None:
        print("Usage: hitlist data {list,available,register,fetch,refresh,info,path,remove}")
        sys.exit(1)
    handlers[args.data_command](args)


def _report(args: argparse.Namespace) -> None:
    from .report import run_report

    text = run_report(mhc_class=args.mhc_class, output=args.output)
    if not args.output:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hitlist",
        description="hitlist: curated mass spectrometry evidence for MHC ligand data",
    )
    sub = parser.add_subparsers(dest="command")
    _build_data_parser(sub)

    p_report = sub.add_parser(
        "report", help="Generate data quality report from registered IEDB/CEDAR"
    )
    p_report.add_argument("--class", dest="mhc_class", help="MHC class filter (I or II)")
    p_report.add_argument("--output", "-o", help="Save report to file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "data":
        _handle_data(args)
    elif args.command == "report":
        _report(args)


if __name__ == "__main__":
    main()
