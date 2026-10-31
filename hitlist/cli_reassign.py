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

"""``hitlist reassign-alleles``: reassign class-only peptides by predicted binding."""

from __future__ import annotations

import argparse
import sys


def _reassign(args: argparse.Namespace) -> None:
    from .predict import reassign_class_only_alleles

    try:
        df = reassign_class_only_alleles(
            method=args.method,
            mhc_class=args.mhc_class,
            max_alleles_per_sample=args.max_alleles_per_sample,
        )
    except (RuntimeError, NotImplementedError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Scored {len(df):,} peptide/sample contexts.  "
        f"Strong binders: {df['is_strong_binder'].sum():,}  "
        f"Weak binders: {df['is_weak_binder'].sum():,}",
        file=sys.stderr,
    )

    if args.output:
        if args.output.endswith(".parquet"):
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Wrote {len(df):,} rows to {args.output}", file=sys.stderr)
    else:
        print(df.to_csv(index=False), end="")


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist reassign-alleles``."""
    p_reassign = sub.add_parser(
        "reassign-alleles",
        help=(
            "Reassign class-only peptides (mhc_restriction='HLA class I') "
            "to their best-scoring allele via MHCflurry / NetMHCpan"
        ),
    )
    p_reassign.add_argument(
        "--method",
        choices=["mhcflurry", "netmhcpan"],
        default="mhcflurry",
        help="Binding predictor (default: mhcflurry)",
    )
    p_reassign.add_argument(
        "--class",
        dest="mhc_class",
        choices=["I"],
        default="I",
        help="MHC class (only I supported in v1.8.0)",
    )
    p_reassign.add_argument(
        "--max-alleles-per-sample",
        type=int,
        default=6,
        help=(
            "Skip samples with more alleles than this — likely pooled-donor "
            "curation artifacts whose 'best allele' is meaningless (default: 6)"
        ),
    )
    p_reassign.add_argument("--output", "-o", help="Write CSV to file")
    p_reassign.set_defaults(func=_reassign)
