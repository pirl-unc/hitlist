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

"""``hitlist build``: observations, proteomes and scan-index builds."""

from __future__ import annotations

import argparse
import sys

from .cli_common import print_subgroup_help


def _data_build(args: argparse.Namespace) -> None:
    from .builder import build_observations

    build_observations(
        with_flanking=args.with_flanking,
        proteome_release=args.proteome_release,
        force=args.force,
        fetch_missing_proteomes=not args.no_fetch_proteomes,
        use_uniprot_search=args.use_uniprot,
        build_mappings=not args.no_mappings,
    )


def _data_fetch_proteomes(args: argparse.Namespace) -> None:
    """Auto-fetch reference proteomes for species present in observations."""
    import pandas as pd

    from .downloads import fetch_species_proteome, lookup_proteome
    from .observations import is_built, load_observations

    if not is_built():
        print(
            "Observations table not built. Run: hitlist build observations",
            file=sys.stderr,
        )
        sys.exit(1)

    obs = load_observations(columns=["source_organism", "mhc_species"])
    counts = pd.Series(dtype=int)
    if "source_organism" in obs.columns:
        counts = obs["source_organism"].value_counts()

    # Also include mhc_species entries that aren't already covered by
    # source_organism — this captures species where source_organism is
    # missing but the MHC allele reveals the host species.
    mhc_counts = obs["mhc_species"].value_counts()
    for sp, n in mhc_counts.items():
        if sp and sp not in counts.index:
            counts[sp] = n

    if counts.empty:
        print("No species information found in observations.")
        sys.exit(0)

    use_uniprot = getattr(args, "use_uniprot", False)

    # Plan the work
    plan: list[tuple[str, int, dict]] = []
    skipped: list[tuple[str, int]] = []
    for organism, n in counts.items():
        if n < args.min_observations:
            continue
        entry = lookup_proteome(str(organism), use_uniprot=use_uniprot)
        if entry is None:
            skipped.append((str(organism), int(n)))
            continue
        plan.append((str(organism), int(n), entry))

    if not plan:
        print(
            f"No registered proteomes for any species with >= {args.min_observations} observations."
        )
        if skipped:
            print("\nUnmatched organisms (no proteome registered):")
            for organism, n in skipped[:10]:
                print(f"  {organism!r}: {n:,} obs")
        sys.exit(0)

    print(f"Fetching reference proteomes for {len(plan)} species/organism(s):\n")
    for organism, n, entry in plan:
        canonical = entry.get("canonical_species", organism)
        kind = entry["kind"]
        proteome_id = entry.get("proteome_id", f"ensembl-{entry.get('release')}")
        print(f"  {organism!r}  ({n:,} obs)  → {canonical}  [{kind}:{proteome_id}]")
    print()

    fetched: list[str] = []
    cached: list[str] = []
    for organism, _n, _entry in plan:
        path = fetch_species_proteome(
            organism, force=args.force, verbose=True, use_uniprot=use_uniprot
        )
        if path is None:
            cached.append(organism)  # ensembl — no local file to track
        else:
            if path.exists():
                fetched.append(organism)

    print(
        f"\nDone. Fetched: {len(fetched)}, Ensembl (no download): {len(cached)}, "
        f"Skipped unknown: {len(skipped)}"
    )
    if skipped:
        print("\nUnmatched organisms (no proteome registered):")
        for organism, n in skipped[:10]:
            print(f"  {organism!r}: {n:,} obs")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


def _add_build_observations_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for the `hitlist build observations` (== legacy `data build`)
    command.  Shared between the canonical and deprecated entry points so the
    flag set never drifts between them."""
    p.add_argument(
        "--no-mappings",
        action="store_true",
        default=False,
        help=(
            "Skip building the peptide_mappings sidecar.  By default, every "
            "build produces multi-mapping peptide→protein attribution and "
            "annotates observations with semicolon-joined gene_names/gene_ids/"
            "protein_ids columns."
        ),
    )
    p.add_argument(
        "--with-flanking",
        action="store_true",
        help="Deprecated — flanks are always stored in peptide_mappings.parquet.",
    )
    p.add_argument(
        "--proteome-release", type=int, default=112, help="Ensembl release (default 112)"
    )
    p.add_argument(
        "--no-fetch-proteomes",
        action="store_true",
        default=False,
        help="Do not auto-fetch missing proteomes when --with-flanking is set",
    )
    p.add_argument(
        "--use-uniprot",
        action="store_true",
        default=False,
        help=(
            "Query UniProt REST for organisms not in the curated registry. "
            "Required to map peptides from rare/pathogen source proteomes."
        ),
    )
    p.add_argument("--force", "-f", action="store_true", help="Rebuild even if cached")


def _add_build_proteomes_args(p: argparse.ArgumentParser) -> None:
    """Argparse setup for `hitlist build proteomes` (== legacy `data fetch-proteomes`)."""
    p.add_argument(
        "--min-observations",
        type=int,
        default=100,
        help="Only fetch proteomes for species with >= N observations (default 100)",
    )
    p.add_argument(
        "--use-uniprot",
        action="store_true",
        default=False,
        help=(
            "Query UniProt REST for organisms not in the curated registry. "
            "Resolved mappings are cached in the manifest."
        ),
    )
    p.add_argument("--force", "-f", action="store_true", help="Re-download cached proteomes")


def _build_top_level_build_parser(sub: argparse._SubParsersAction) -> None:
    """Register the canonical ``hitlist build`` group."""
    p_build = sub.add_parser(
        "build",
        help=(
            "Build canonical artifacts: observations.parquet (from IEDB/CEDAR), "
            "cached scan index, reference proteomes."
        ),
    )
    p_build.set_defaults(_subgroup_parser=p_build, func=_handle_build)
    bs = p_build.add_subparsers(dest="build_command")

    p = bs.add_parser("observations", help="Build unified observations table from IEDB/CEDAR")
    _add_build_observations_args(p)

    p = bs.add_parser(
        "proteomes",
        help="Auto-fetch reference proteomes for species present in observations",
    )
    _add_build_proteomes_args(p)


def _handle_build(args: argparse.Namespace) -> None:
    """Dispatch ``hitlist build <subcommand>`` to the same handlers as the
    legacy ``hitlist data {build,fetch-proteomes}`` entry points."""
    cmd = getattr(args, "build_command", None)
    if cmd is None:
        print_subgroup_help(args._subgroup_parser)
        sys.exit(1)
    handlers = {
        "observations": _data_build,
        "proteomes": _data_fetch_proteomes,
    }
    handlers[cmd](args)


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist build``."""
    _build_top_level_build_parser(sub)
