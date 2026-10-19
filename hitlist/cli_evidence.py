"""Expression-table evidence and independently reusable tissue-blacklist CLI."""

import json
import sys


def add_evidence_parsers(sub, export_sub):
    cta = export_sub.add_parser(
        "cta-evidence",
        help="Export expression-selected CTA evidence, mappings and tissue exclusions",
    )
    cta.add_argument(
        "--expression",
        help="TPM table or directory; default: one conventional expression/quantifier file in the current directory",
    )
    cta.add_argument("--id-column", help="Override automatic identifier-column detection")
    cta.add_argument(
        "--tpm-column", help="Override automatic TPM-column detection or choose a sample"
    )
    cta.add_argument(
        "--expression-level",
        choices=["auto", "gene", "transcript"],
        default="auto",
        help="Default: infer from identifier headers, values or quantifier format",
    )
    cta.add_argument("--min-tpm", type=float, default=2.0)
    cta.add_argument("--cta-definition", choices=["strict", "extended"], default="strict")
    cta.add_argument("--ensembl-release", type=int, default=112)
    exclusions = cta.add_mutually_exclusive_group()
    exclusions.add_argument(
        "--exclude-gene-pattern",
        action="append",
        help="Replace default MAGE* exclusions with repeatable gene-symbol globs",
    )
    exclusions.add_argument("--no-gene-exclusions", action="store_true")
    cta.add_argument(
        "--allow-gene",
        action="append",
        help="Replace default MAGEA4 exception with repeatable exact symbols",
    )
    blacklist = export_sub.add_parser(
        "tissue-blacklist",
        help="Export sequences observed in nonmalignant heart/brain/lung in at least two donors",
    )
    for parser in (cta, blacklist):
        parser.add_argument(
            "--atlas-dir",
            required=True,
            help="HLA Ligand Atlas 2020.12 peptides, sample_hits and donors TSV/GZ tables",
        )
        parser.add_argument(
            "--bundle",
            required=True,
            help="New evidence bundle directory; existing paths are never overwritten",
        )
    verify = sub.add_parser(
        "verify-evidence-bundle",
        help="Verify hashes and scientific relationships in an evidence bundle",
    )
    verify.add_argument("directory")


def handle_evidence(args):
    from .evidence_bundle import (
        verify_evidence_bundle,
        write_cta_evidence_bundle,
        write_tissue_blacklist_bundle,
    )

    try:
        if args.command == "verify-evidence-bundle":
            result = verify_evidence_bundle(args.directory)
            print(f"Verified {result['kind']} bundle: {args.directory}")
            return
        print(
            "Reading donor-resolved tissue evidence and verifying source provenance...",
            file=sys.stderr,
        )
        if args.export_command == "tissue-blacklist":
            path = write_tissue_blacklist_bundle(args.bundle, atlas_dir=args.atlas_dir)
        else:
            path = write_cta_evidence_bundle(
                args.bundle,
                args.expression,
                atlas_dir=args.atlas_dir,
                id_column=args.id_column,
                tpm_column=args.tpm_column,
                level=args.expression_level,
                min_tpm=args.min_tpm,
                definition=args.cta_definition,
                ensembl_release=args.ensembl_release,
                exclude_gene_patterns=()
                if args.no_gene_exclusions
                else tuple(args.exclude_gene_pattern or ["MAGE*"]),
                allow_genes=tuple(args.allow_gene or ["MAGEA4"]),
            )
            expression = json.loads(path.read_text())["expression"]
            print(
                f"Expression: {expression['input']['path']} ({expression['level']}; ID={expression['id_column']}; TPM={expression['tpm_column']})"
            )
        print(f"Wrote verified evidence bundle: {path}")
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
