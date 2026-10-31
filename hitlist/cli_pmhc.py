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

"""``hitlist pmhc``: per-protein x allele pMHC evidence."""

from __future__ import annotations

import argparse
import sys


def _parse_pmhc_samples(inline_specs: list[str], tsv_path: str | None) -> dict[str, list[str]]:
    """Parse ``--sample`` / ``--samples`` into ``{name: [allele, ...]}``.

    Inline form: ``"patient1:HLA-A*02:01,HLA-A*24:02"`` — colon splits the
    name from the allele list, commas split alleles. Whitespace around the
    name and alleles is stripped.

    TSV form: two columns, ``name`` and ``alleles`` (alleles comma-separated
    in the second column). Header row optional — if the first row's first
    cell is exactly ``name`` we skip it. Lines starting with ``#`` are
    ignored as comments.

    Raises ``ValueError`` on malformed input — the CLI catches and prints.
    """
    out: dict[str, list[str]] = {}

    for spec in inline_specs:
        if ":" not in spec:
            raise ValueError(
                f"--sample {spec!r} is missing the ':' separator; expected NAME:ALLELE[,ALLELE,...]"
            )
        name, _, allele_csv = spec.partition(":")
        name = name.strip()
        if not name:
            raise ValueError(f"--sample {spec!r} has an empty name")
        alleles = [a.strip() for a in allele_csv.split(",") if a.strip()]
        if not alleles:
            raise ValueError(f"--sample {name!r} has no alleles after the ':'")
        if name in out:
            raise ValueError(f"--sample {name!r} given more than once")
        out[name] = alleles

    if tsv_path:
        from pathlib import Path

        path = Path(tsv_path)
        if not path.exists():
            raise ValueError(f"--samples file not found: {tsv_path}")
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"{tsv_path}:{lineno} expected two tab-separated columns "
                    f"(name, alleles); got {len(parts)} column(s)"
                )
            name = parts[0].strip()
            allele_field = parts[1].strip()
            # Skip the optional header row.
            if lineno == 1 and name.lower() == "name":
                continue
            if not name:
                raise ValueError(f"{tsv_path}:{lineno} has an empty name")
            alleles = [a.strip() for a in allele_field.split(",") if a.strip()]
            if not alleles:
                raise ValueError(f"{tsv_path}:{lineno} sample {name!r} has no alleles")
            if name in out:
                raise ValueError(
                    f"sample {name!r} given more than once (--sample inline + {tsv_path}:{lineno})"
                )
            out[name] = alleles

    if not out:
        raise ValueError("no samples parsed from --sample / --samples input")
    return out


def _split_allele_tokens(raw: list[str]) -> list[str]:
    """Flatten --mhc-allele's raw tokens, splitting each on commas/whitespace.

    ``action="extend", nargs="+"`` alone only separates alleles the shell
    already split into distinct argv tokens -- a single quoted, pasted
    genotype string ("HLA-A*02:01 HLA-B*07:02" or "HLA-A*02:01,HLA-B*07:02")
    arrives as ONE token and would otherwise be matched against literally
    (#491), silently returning zero rows instead of erroring or working.
    """
    out: list[str] = []
    for token in raw:
        out.extend(token.replace(",", " ").split())
    return out


def _warn_unrecognized_genes(proteins: list[str]) -> None:
    """Print a stderr hint when a ``--gene`` query matched nothing because the
    symbol isn't in the corpus (likely a typo), vs. a real gene with no evidence.

    Kept distinct from the empty-result body so CSV/JSON output stays clean — the
    note goes to stderr only.  Silent when the symbol IS recognized (the empty
    result then genuinely means "no MS evidence under these filters").
    """
    from . import pmhc_query

    unknown = pmhc_query.unrecognized_genes(proteins)
    if not unknown:
        return
    pretty = ", ".join(repr(g) for g in unknown)
    print(
        f"Note: {pretty} matched no gene in the hitlist corpus. If unexpected, "
        f"check the spelling — current HGNC symbols, aliases, and Ensembl gene "
        f"IDs are accepted (e.g. 'XAGE1A', not 'XAGE12').",
        file=sys.stderr,
    )


def _pmhc(args: argparse.Namespace) -> None:
    """Run the pmhc evidence query.

    ``--protein`` / ``--mhc-allele`` / ``--sample`` / ``--samples`` are
    all optional. ``--sample``/``--samples`` (paired sample-to-allele
    input) are mutually exclusive with ``--mhc-allele`` (cross-product
    input).
    """
    from . import pmhc_query

    # --list-source-contexts: print the catalog and exit (no query).
    if getattr(args, "list_source_contexts", False):
        print("Available --source-context values:\n")
        width = max(len(k) for k in pmhc_query.SOURCE_CONTEXTS)
        for name, cols in pmhc_query.SOURCE_CONTEXTS.items():
            desc = pmhc_query.SOURCE_CONTEXT_DESCRIPTIONS.get(name, "")
            print(f"  {name:<{width}}  {desc}")
            print(f"  {'':<{width}}  flags: {', '.join(cols)}")
        return

    # --list-gene-sets: print the named gene sets and exit.
    if getattr(args, "list_gene_sets", False):
        from .genes import list_gene_sets

        sets = list_gene_sets()
        if not sets:
            print("No gene sets defined.")
            return
        print("Available --gene-set values:\n")
        for name, desc in sets:
            print(f"  {name}  {desc}")
        return

    proteins = getattr(args, "protein", []) or []
    # --gene-set NAME expands into the gene filter (composes with --gene).
    gene_set = getattr(args, "gene_set", None)
    if gene_set:
        from .genes import load_gene_set

        try:
            proteins = list(proteins) + load_gene_set(gene_set)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)
    alleles = _split_allele_tokens(getattr(args, "mhc_allele", []) or [])
    inline_samples = getattr(args, "sample", None) or []
    samples_path = getattr(args, "samples", None)
    species = getattr(args, "species", None)
    source_species = getattr(args, "source_species", None)
    host_species = getattr(args, "host_species", None)
    source_context = getattr(args, "source_context", None)

    # --by-gene: per-gene rollup (one row per gene with evidence totals).
    if getattr(args, "by_gene", False):
        try:
            dist = pmhc_query.gene_distribution(
                proteins=proteins,
                species=species,
                source_species=source_species,
                host_species=host_species,
                source_context=source_context,
                verbose=True,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if dist.empty and proteins:
            _warn_unrecognized_genes(proteins)
        fmt = getattr(args, "format", "table")
        if fmt == "csv":
            text = dist.to_csv(index=False)
        elif fmt == "json":
            text = dist.to_json(orient="records", indent=2)
        else:
            text = pmhc_query.format_gene_table(dist)
        out = getattr(args, "output", None)
        if out:
            from pathlib import Path

            Path(out).write_text(text + "\n")
            print(f"Wrote {out}")
        else:
            print(text)
        return

    # --by-tissue: where (tissues / cell types) are the gene's peptides from?
    if getattr(args, "by_tissue", False):
        expand_lines = getattr(args, "expand_lines", False)
        try:
            dist = pmhc_query.tissue_distribution(
                proteins=proteins,
                species=species,
                source_species=source_species,
                host_species=host_species,
                source_context=source_context,
                show_empty=getattr(args, "show_zeros", False),
                expand_lines=expand_lines,
                expand_tissues=getattr(args, "expand_tissues", False),
                verbose=True,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if dist.empty and proteins:
            _warn_unrecognized_genes(proteins)
        out = getattr(args, "output", None)
        fmt = getattr(args, "format", "table")
        if fmt == "csv":
            text = dist.to_csv(index=False)
        elif fmt == "json":
            text = dist.to_json(orient="records", indent=2)
        else:
            text = pmhc_query.format_tissue_table(dist)
        if out:
            from pathlib import Path

            Path(out).write_text(text + "\n")
            print(f"Wrote {out}")
        else:
            print(text)
        return
    predictor = getattr(args, "predictor", None)
    min_binder_class = getattr(args, "min_binder_class", None)
    min_references = getattr(args, "min_references", 1)
    min_samples = getattr(args, "min_samples", 1)
    cell_type = getattr(args, "cell_type", None)
    fmt = getattr(args, "format", "table")
    output = getattr(args, "output", None)

    has_sample_input = bool(inline_samples) or bool(samples_path)
    if has_sample_input and alleles:
        print(
            "Error: --mhc-allele cannot be combined with --sample / --samples; "
            "drop one. Use --mhc-allele for a cross-product (every protein x "
            "every allele); use --sample/--samples when each sample has its "
            "own allele set.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        if has_sample_input:
            samples_to_alleles = _parse_pmhc_samples(inline_samples, samples_path)
            df = pmhc_query.query_by_samples(
                samples_to_alleles=samples_to_alleles,
                proteins=proteins,
                species=species,
                source_context=source_context,
                predictor=predictor,
                min_binder_class=min_binder_class,
                min_references=min_references,
                min_samples=min_samples,
                cell_type=cell_type,
                verbose=True,
            )
        else:
            df = pmhc_query.query(
                proteins=proteins,
                alleles=alleles,
                species=species,
                source_species=source_species,
                host_species=host_species,
                source_context=source_context,
                predictor=predictor,
                min_binder_class=min_binder_class,
                min_references=min_references,
                min_samples=min_samples,
                cell_type=cell_type,
                verbose=True,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty and proteins:
        _warn_unrecognized_genes(proteins)

    if fmt == "csv":
        text = df.to_csv(index=False)
    elif fmt == "json":
        text = df.to_json(orient="records", indent=2)
    else:  # table
        text = pmhc_query.format_table(df)

    if output:
        from pathlib import Path

        Path(output).write_text(text)
        print(f"Wrote {output}", file=sys.stderr)
    else:
        print(text)


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist pmhc``."""
    p_pmhc = sub.add_parser(
        "pmhc",
        help=(
            "Per-protein x allele pMHC evidence: MS-attested peptides grouped "
            "by (gene, allele), with optional NetMHCpan/MHCflurry scoring."
        ),
        description=(
            "Per-protein x allele pMHC evidence query.\n\n"
            "Both --protein and --mhc-allele are optional; pass neither to "
            "scan the whole corpus, just one to fix that axis. Pass "
            "--predictor {mhcflurry,netmhcpan} to add binding-affinity scoring."
        ),
    )
    p_pmhc.add_argument(
        "--protein",
        "--gene",
        action="extend",
        nargs="+",
        help=("One or more proteins: HGNC symbol, Ensembl ENSG, or alias. Omit to scan all genes."),
    )
    p_pmhc.add_argument(
        "--mhc-allele",
        "--mhc-alleles",
        action="extend",
        nargs="+",
        help=(
            "One or more 4-digit MHC alleles (HLA-A*02:01, HLA-B*07:02, ...). "
            "Space-separated, comma-separated, or repeated -- a whole pasted "
            "genotype string works as one argument, e.g. "
            '--mhc-allele "HLA-A*02:01 HLA-B*07:02" or '
            "--mhc-allele HLA-A*02:01,HLA-B*07:02. "
            "Omit to scan all alleles. Mutually exclusive with --sample / --samples — "
            "use those when each sample has its own allele set instead of a "
            "cross-product across all of them."
        ),
    )
    p_pmhc.add_argument(
        "--sample",
        action="append",
        metavar="NAME:ALLELE[,ALLELE,...]",
        help=(
            "Per-sample paired (name, allele set). Repeatable; "
            "alleles are comma-separated after the colon. "
            "Example: --sample patient1:HLA-A*02:01,HLA-A*24:02 "
            "--sample patient2:HLA-A*01:01,HLA-A*24:02. "
            "Each sample is queried independently against the protein "
            "list — results are sectioned by sample in the output."
        ),
    )
    p_pmhc.add_argument(
        "--samples",
        metavar="PATH.tsv",
        help=(
            "Batch form of --sample: path to a TSV with one row per sample, "
            "two columns 'name' and 'alleles' (alleles comma-separated). "
            "Header row optional. Mutually exclusive with --mhc-allele."
        ),
    )
    p_pmhc.add_argument(
        "--species",
        "--mhc-species",
        dest="species",
        help=(
            "Filter by MHC species (the species of the HLA/MHC molecule, e.g. "
            "'human' / 'Homo sapiens'). Without it, results include every species "
            "the peptides were observed on, sectioned with human first. The three "
            "species axes are independent — see --source-species / --host-species."
        ),
    )
    p_pmhc.add_argument(
        "--source-species",
        dest="source_species",
        help=(
            "Filter by SOURCE species — the organism whose proteome the peptide "
            "comes from (e.g. 'human' drops a SARS-CoV-2 viral epitope even though "
            "its MHC and host cell are human)."
        ),
    )
    p_pmhc.add_argument(
        "--host-species",
        dest="host_species",
        help=(
            "Filter by HOST species — the organism of the cell the peptide was "
            "expressed/eluted in (e.g. a human gene presented in a mouse line)."
        ),
    )
    from .pmhc_query import SOURCE_CONTEXTS

    p_pmhc.add_argument(
        "--source-context",
        choices=sorted(SOURCE_CONTEXTS),
        help=(
            "Keep only observations from this curated source context: "
            "healthy, cancer, cell_line, ebv_lcl, adjacent, activated_apc, "
            "ex_vivo. Run --list-source-contexts to see what each means."
        ),
    )
    p_pmhc.add_argument(
        "--list-source-contexts",
        action="store_true",
        help="Print the available --source-context values (with descriptions) and exit.",
    )
    p_pmhc.add_argument(
        "--by-tissue",
        action="store_true",
        help=(
            "Summarize WHERE the gene's peptides come from, in four sections: "
            "healthy tissues / cancer tissues (by source tissue) and cancer / "
            "non-cancer cell lines (by cell line). Each section shows "
            "observations, unique peptides, references; sorted by observations. "
            "Combine with --source-context and --species. Ignores "
            "--mhc-allele / --sample / --predictor."
        ),
    )
    p_pmhc.add_argument(
        "--show-zeros",
        action="store_true",
        help="With --by-tissue, show empty sections (as '(none)') instead of eliding them.",
    )
    p_pmhc.add_argument(
        "--by-gene",
        action="store_true",
        help=(
            "Per-GENE rollup: one row per gene with n_observations / "
            "n_unique_peptides / n_references / n_samples, sorted by observations. "
            "Pair with --gene-set CTA (or --gene) to summarize a panel. Honors "
            "--source-context and --species; ignores --mhc-allele / --predictor."
        ),
    )
    p_pmhc.add_argument(
        "--gene-set",
        dest="gene_set",
        help=(
            "Expand a named gene set into the --gene filter (e.g. 'CTA' for the "
            "cancer-testis-antigen panel). See --list-gene-sets."
        ),
    )
    p_pmhc.add_argument(
        "--list-gene-sets",
        action="store_true",
        help="Print the available --gene-set names (with descriptions) and exit.",
    )
    p_pmhc.add_argument(
        "--expand-lines",
        action="store_true",
        help=(
            "With --by-tissue, group the cell-line sections by individual line "
            "name (THP-1, HeLa, ...) instead of the default cell TYPE of origin "
            "(Monocyte, Melanocyte, B cell, ..., EBV-LCL)."
        ),
    )
    p_pmhc.add_argument(
        "--expand-tissues",
        action="store_true",
        help=(
            "With --by-tissue, split anatomical roll-up groups into their specific "
            "organs (GI tract -> Esophagus/Stomach/Colon/...; CNS -> Brain/"
            "Cerebellum/Spinal Cord). Rows a study labelled only with the umbrella "
            "term show as '<group> (region unspecified)'. Default rolls them up."
        ),
    )
    p_pmhc.add_argument(
        "--predictor",
        choices=["mhcflurry", "netmhcpan"],
        help=(
            "Optional binding-affinity predictor.  If omitted, only MS evidence "
            "is reported (no affinity column)."
        ),
    )
    p_pmhc.add_argument(
        "--min-binder-class",
        choices=["strong", "medium", "weak"],
        default=None,
        help=(
            "Drop rows whose predicted binder tier is below this threshold.  "
            "'strong' keeps only strong; 'medium' keeps strong+medium; 'weak' "
            "drops only non-binders.  Requires --predictor."
        ),
    )
    p_pmhc.add_argument(
        "--min-references",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Drop rows backed by fewer than N distinct PMIDs.  Default 1 "
            "(no filter).  Use 2+ to require independently re-observed peptides."
        ),
    )
    p_pmhc.add_argument(
        "--min-samples",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Drop rows backed by fewer than N distinct samples (donors / "
            "cell lines / runs).  Default 1 (no filter).  A single PMID often "
            "contributes many samples — --min-samples is usually a stronger "
            "evidence signal than --min-references."
        ),
    )
    p_pmhc.add_argument(
        "--cell-type",
        action="append",
        metavar="TYPE",
        help=(
            "Keep only observations from this cell type (the tissue/cell-type "
            'half of IEDB\'s Cell Name, e.g. "Melanocyte", "B cell"). '
            "Case-insensitive; repeat the flag to allow several types. "
            "Requires a corpus built with hitlist >= 1.30.57."
        ),
    )
    p_pmhc.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help=(
            "Output shape (default: table — protein > allele section "
            "headers with peptide rows in an aligned table beneath each allele)."
        ),
    )
    p_pmhc.add_argument("--output", "-o", help="Write to file (.csv / .json / .txt)")
    p_pmhc.set_defaults(func=_pmhc)
