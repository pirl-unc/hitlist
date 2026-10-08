"""Management commands for pinned UniProt reference releases."""

import json

from . import uniprot


def add_uniprot_parser(sub):
    parser = sub.add_parser("uniprot", help="Manage release-pinned UniProt references")
    actions = parser.add_subparsers(dest="uniprot_command", required=True)
    for action in ("list", "fetch", "path", "info", "remove"):
        command = actions.add_parser(action)
        command.add_argument("--collection", default=uniprot.DEFAULT_COLLECTION)
        if action != "list":
            command.add_argument("--release", required=action == "remove")
        if action in ("list", "info"):
            command.add_argument("--verify", action="store_true", help="Check cached SHA256 hashes")
        command.add_argument("--json", action="store_true", help="Print JSON")
        if action == "fetch":
            command.add_argument("--force", action="store_true", help="Replace/repair cached bytes")
            command.add_argument(
                "--max-asset-bytes", type=int, default=uniprot.DEFAULT_MAX_ASSET_BYTES
            )
            command.add_argument(
                "--max-cache-bytes", type=int, default=uniprot.DEFAULT_MAX_CACHE_BYTES
            )


def handle_uniprot(args):
    options = {"collection": args.collection}
    if args.uniprot_command != "list":
        options["release"] = args.release
    try:
        if args.uniprot_command == "list":
            result = uniprot.list_uniprot_references(**options, verify=args.verify)
        elif args.uniprot_command == "info":
            result = uniprot.uniprot_info(**options, verify=args.verify)
        elif args.uniprot_command == "fetch":
            result = str(
                uniprot.fetch_uniprot_reference(
                    **options,
                    force=args.force,
                    max_asset_bytes=args.max_asset_bytes,
                    max_cache_bytes=args.max_cache_bytes,
                )
            )
        elif args.uniprot_command == "path":
            result = str(uniprot.uniprot_path(**options))
        else:
            result = {"removed": uniprot.remove_uniprot_reference(**options), **options}
    except (OSError, ValueError, NotImplementedError) as error:
        raise SystemExit(f"UniProt: {error}") from error
    if args.json or isinstance(result, dict):
        print(json.dumps(result, indent=2))
    elif isinstance(result, list):
        for row in result:
            print(
                f"{row['collection']}\t{row['release']}\t{row['status']}\t{row['size_bytes']} bytes\t{row['path']}"
            )
    else:
        print(result)
