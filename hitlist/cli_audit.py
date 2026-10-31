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

"""``hitlist audit-splits``: audit partitions supplied as training bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _handle_split_audit(args):
    from .training_bundle import audit_training_bundles

    try:
        partitions = {}
        for value in args.partition:
            name, separator, directory = value.partition("=")
            if not separator or not name or not directory or name in partitions:
                raise ValueError("Each partition must be a unique NAME=DIRECTORY")
            partitions[name] = directory
        report = audit_training_bundles(partitions, policy=args.policy)
        content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(content)
        else:
            print(content, end="")
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if report["verdict"] in {"fail", "inconclusive"}:
        sys.exit(2)


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist audit-splits``."""
    from .split_audit import POLICIES

    p_audit = sub.add_parser(
        "audit-splits", help="Audit partitions supplied as verified training bundles"
    )
    p_audit.add_argument("--partition", action="append", required=True, metavar="NAME=DIRECTORY")
    p_audit.add_argument("--policy", choices=sorted(POLICIES), default="report_only")
    p_audit.add_argument("--output", "-o", help="Write the JSON audit report")
    p_audit.set_defaults(func=_handle_split_audit)
