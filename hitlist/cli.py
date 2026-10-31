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

import sys

from . import (
    cli_build,
    cli_data,
    cli_export,
    cli_pmhc,
    cli_qc,
    cli_reassign,
    cli_report,
    cli_samples,
)
from .cli_common import print_subgroup_help
from .cli_help import ColorArgumentParser
from .downloads import legacy_data_dir_notice
from .errors import HitlistError


def main() -> None:
    """Console entry point: expected failures (``HitlistError``) exit 1 without a traceback."""
    try:
        _main()
    except HitlistError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def _main() -> None:
    from .version import __version__

    parser = ColorArgumentParser(
        prog="hitlist",
        description="hitlist: verified immune and nonimmune evidence for model training",
    )
    parser.add_argument("--version", action="version", version=f"hitlist {__version__}")
    sub = parser.add_subparsers(dest="command")
    # Registration order is the order commands appear in ``--help`` (pinned by
    # tests/test_cli_help_snapshot.py).  Every command group binds its handler with
    # ``set_defaults(func=...)``, so dispatch below is just ``args.func(args)``.
    cli_data.register(sub)
    cli_build.register(sub)
    from .cli_evidence_store import register_evidence_cli

    register_evidence_cli(sub)
    cli_report.register(sub)
    cli_export.register(sub)
    cli_reassign.register(sub)
    cli_qc.register(sub)
    cli_pmhc.register(sub)
    cli_samples.register(sub)

    args = parser.parse_args()
    if args.command is None:
        print_subgroup_help(parser)
        sys.exit(1)
    # The library resolves the data directory silently (`data_dir()` backs
    # `observations_cache_is_current()`, which promises no output); the CLI is
    # where a user is told they are still on the legacy location (#291).
    if args.command != "evidence":
        notice = legacy_data_dir_notice()
        if notice:
            print(notice, file=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
