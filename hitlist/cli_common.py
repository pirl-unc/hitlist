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

"""Helpers shared by the ``hitlist`` command-group modules."""

from __future__ import annotations

import argparse
import sys


def print_banner(stream=sys.stderr) -> None:
    """One-line version + observations-build status banner.

    Shown above subgroup help so users get quick orientation: which
    version they're on, and whether the canonical artifact downstream
    commands depend on has been built yet.
    """
    from .version import __version__

    print(f"hitlist v{__version__}", file=stream)
    try:
        from .observations import is_built

        built = is_built()
    except Exception:
        built = False
    state = "built" if built else "NOT BUILT — run `hitlist build observations`"
    print(f"observations.parquet: {state}", file=stream)
    print(file=stream)


def print_subgroup_help(parser: argparse.ArgumentParser) -> None:
    """Banner + full help for a subgroup parser.

    Used when a user runs e.g. ``hitlist pmhc`` with no subcommand —
    instead of a one-line "Usage: hitlist pmhc {query}" string, show
    what each subcommand actually does and the args it takes.
    """
    print_banner()
    parser.print_help(sys.stderr)
