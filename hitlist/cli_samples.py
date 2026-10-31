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

"""``hitlist samples`` / ``proteomics`` / ``expression``: promoted export aliases."""

from __future__ import annotations

import argparse

from .cli_export import (
    _add_export_bulk_proteomics_args,
    _add_line_expression_args,
    _add_samples_args,
    run_export,
)


def _handle_samples(args: argparse.Namespace) -> None:
    """Run ``hitlist samples`` — same backing logic as the legacy
    ``hitlist export samples``."""
    run_export(args, "samples")


def _handle_proteomics(args: argparse.Namespace) -> None:
    """Run ``hitlist proteomics`` — same backing logic as ``hitlist export bulk-proteomics``."""
    run_export(args, "bulk-proteomics")


def _handle_expression(args: argparse.Namespace) -> None:
    """Run ``hitlist expression`` — same backing logic as ``hitlist export line-expression``."""
    run_export(args, "line-expression")


def register(sub: argparse._SubParsersAction) -> None:
    """Register the promoted aliases."""
    # ── samples subcommand (promoted from `export samples`) ────────────
    p_samples = sub.add_parser(
        "samples",
        help=(
            "Per-sample MS conditions table (cell line, donor, instrument, "
            "perturbation, etc.).  Replaces `export samples`."
        ),
    )
    _add_samples_args(p_samples)
    p_samples.set_defaults(func=_handle_samples)

    # ── proteomics (promoted from `export bulk-proteomics`) ────────────
    p_prot = sub.add_parser(
        "proteomics",
        help=(
            "Bulk (non-MHC) proteomics index: shotgun MS peptides + protein "
            "abundances from CCLE and Bekker-Jensen 2017.  Replaces "
            "`export bulk-proteomics`."
        ),
    )
    _add_export_bulk_proteomics_args(p_prot)
    p_prot.set_defaults(func=_handle_proteomics)

    # ── expression (promoted from `export line-expression`) ────────────
    p_expr = sub.add_parser(
        "expression",
        help=("Per-line RNA / transcript TPM with provenance.  Replaces `export line-expression`."),
    )
    _add_line_expression_args(p_expr)
    p_expr.set_defaults(func=_handle_expression)
