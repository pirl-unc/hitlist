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

"""``hitlist path``: resolve the active verified dataset bundle or a table inside it (#989)."""

from __future__ import annotations

import argparse
import sys

from .dataset import BUNDLE_TABLE_ALIASES, bundle_path, stale_root_files
from .errors import DatasetError


def _cmd_path(args: argparse.Namespace) -> None:
    stale = stale_root_files(root=args.root)
    if stale:
        names = ", ".join(p.name for p in stale[:4])
        if len(stale) > 4:
            names += f", ... ({len(stale)} total)"
        print(
            f"Warning: stale root-level cache files ({names}) are superseded by the active bundle.",
            file=sys.stderr,
        )
    try:
        resolved = bundle_path(
            args.table,
            root=args.root,
            allow_legacy=args.legacy,
            verify=args.check,
            must_exist=args.check,
            warn_stale=False,
        )
    except FileNotFoundError as exc:
        raise DatasetError(str(exc)) from exc
    print(resolved)


def register(sub: argparse._SubParsersAction) -> None:
    aliases = ", ".join(
        sorted(
            {
                "observations",
                "binding",
                "other_assays",
                "mappings",
                "samples",
                "alleles",
                "proteomes",
                "contributors",
            }
            & set(BUNDLE_TABLE_ALIASES)
        )
    )
    p = sub.add_parser(
        "path",
        help="Print the active dataset bundle directory (or a table inside it)",
        description=(
            "Resolve the active verified Hitlist bundle directory or a named table "
            f"inside it (aliases include: {aliases}; or any exact filename)."
        ),
    )
    p.add_argument(
        "table",
        nargs="?",
        default=None,
        help="Optional table alias or filename inside the bundle (omit for the bundle directory).",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Override the Hitlist cache root directory.",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Fall back to a legacy flat root cache when no published bundle exists.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify the bundle hashes and require the target path to exist.",
    )
    p.set_defaults(func=_cmd_path)
