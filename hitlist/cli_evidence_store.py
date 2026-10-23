"""One CLI surface for verified evidence discovery, query and recipes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_recipe import verify_recipe_package
from .evidence_store import EvidenceQueryError, EvidenceStore


def register_evidence_cli(sub):
    parser = sub.add_parser("evidence", help="Query a verified multi-modality evidence package")
    parser.set_defaults(_subgroup_parser=parser)
    commands = parser.add_subparsers(dest="evidence_command")

    catalog = commands.add_parser("catalog", help="Describe evidence types and field readiness")
    catalog.add_argument("package", type=Path)
    catalog.add_argument("--type", dest="evidence_type")
    catalog.add_argument("--output", type=Path)

    audit = commands.add_parser("audit", help="Reconcile source counts and field readiness")
    audit.add_argument("package", type=Path)
    audit.add_argument("--output", type=Path)

    query = commands.add_parser("query", help="Query one evidence grain with logical field names")
    query.add_argument("package", type=Path)
    query.add_argument("evidence_type")
    query.add_argument("--select", action="append", default=[])
    query.add_argument("--where", default="{}", help="JSON object of field predicates")
    query.add_argument("--join", action="append", default=[])
    query.add_argument("--limit", type=int, default=1000)
    query.add_argument("--all", action="store_true", help="Stream every matching row")
    query.add_argument("--output", type=Path, help=".parquet or .jsonl; required with --all")

    recipe = commands.add_parser("materialize", help="Apply a versioned training recipe")
    recipe.add_argument("package", type=Path)
    recipe.add_argument("recipe", type=Path)
    recipe.add_argument("output", type=Path)

    verify = commands.add_parser("verify", help="Verify the package manifest chain")
    verify.add_argument("package", type=Path)

    verify_recipe = commands.add_parser("verify-recipe", help="Verify a training output")
    verify_recipe.add_argument("output", type=Path)
    verify_recipe.add_argument("--source-package", type=Path)


def _output_json(value, destination=None):
    content = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if destination:
        destination.write_text(content)
    else:
        sys.stdout.write(content)


def _query(store, args):
    if args.all and args.output is None:
        raise EvidenceQueryError("--all requires --output")
    if args.output and args.output.suffix not in {".parquet", ".jsonl"}:
        raise EvidenceQueryError("Query output must end in .parquet or .jsonl")
    try:
        where = json.loads(args.where)
    except json.JSONDecodeError as error:
        raise EvidenceQueryError(f"Invalid --where JSON: {error}") from error
    select = args.select or None
    limit = None if args.all else args.limit
    request = {
        "select": select,
        "where": where,
        "joins": args.join,
        "limit": limit,
    }
    if not args.all:
        result = store.query(args.evidence_type, **request)
        if args.output is None:
            for row in result.table.to_pylist():
                sys.stdout.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        elif args.output.suffix == ".parquet":
            pq.write_table(result.table, args.output)
        else:
            with args.output.open("w") as stream:
                for row in result.table.to_pylist():
                    stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return
    if args.output.suffix == ".parquet":
        writer = None
        try:
            for batch in store.iter_query(args.evidence_type, **request):
                if writer is None:
                    writer = pq.ParquetWriter(args.output, batch.table.schema, compression="zstd")
                writer.write_table(batch.table)
            if writer is None:
                empty = store.query(args.evidence_type, **{**request, "limit": 0})
                writer = pq.ParquetWriter(args.output, empty.table.schema, compression="zstd")
        finally:
            if writer is not None:
                writer.close()
    else:
        with args.output.open("w") as stream:
            for batch in store.iter_query(args.evidence_type, **request):
                for row in batch.table.to_pylist():
                    stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def handle_evidence_cli(args):
    try:
        if args.evidence_command == "verify-recipe":
            manifest = verify_recipe_package(args.output, source_package=args.source_package)
            _output_json({"recipe": manifest["recipe"]["name"], "coverage": manifest["coverage"]})
            return
        if args.evidence_command is None:
            args._subgroup_parser.print_help(sys.stderr)
            raise SystemExit(1)
        store = EvidenceStore.open(args.package)
        if args.evidence_command == "catalog":
            catalog = store.catalog()
            if args.evidence_type:
                entries = [
                    item for item in catalog["evidence"] if item["name"] == args.evidence_type
                ]
                if not entries:
                    raise EvidenceQueryError(f"Unknown evidence type: {args.evidence_type}")
                catalog = entries[0]
            _output_json(catalog, args.output)
        elif args.evidence_command == "audit":
            _output_json(store.audit(), args.output)
        elif args.evidence_command == "query":
            _query(store, args)
        elif args.evidence_command == "materialize":
            print(store.materialize(args.recipe, args.output))
        elif args.evidence_command == "verify":
            _output_json(
                {
                    "package_manifest_sha256": store._manifest_hash(),
                    "package_schema_version": store.manifest["schema_version"],
                    "n_materialized_evidence_types": len(store.bindings),
                }
            )
    except (EvidenceQueryError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Evidence request failed: {error}") from error
