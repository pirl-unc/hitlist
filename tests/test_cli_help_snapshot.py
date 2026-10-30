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

"""Snapshot of ``hitlist --help`` and every subcommand's ``--help`` (recursively).

Guards the CLI surface (command tree, argument names, help text, ordering) against
accidental change during structural refactors of ``hitlist.cli`` (#887).  To accept an
intentional change, regenerate with ``python tests/test_cli_help_snapshot.py``.
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent / "data" / "cli_help_snapshot.txt"


class _ParserCaptured(Exception):
    pass


def _capture_root_parser():
    """Run ``main`` just far enough to build the parser, and return it."""
    from hitlist import cli

    captured = []
    original = argparse.ArgumentParser.parse_args

    def grab(self, *args, **kwargs):
        captured.append(self)
        raise _ParserCaptured

    argparse.ArgumentParser.parse_args = grab
    try:
        with contextlib.suppress(_ParserCaptured):
            cli.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured[0]


def _command_paths(parser, prefix=()):
    yield prefix
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from _command_paths(child, (*prefix, name))


def _help_text(path):
    from hitlist import cli

    out = io.StringIO()
    argv = ["hitlist", *path, "--help"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
            cli.main()
    finally:
        sys.argv = old_argv
    return out.getvalue()


def render_snapshot() -> str:
    os.environ["NO_COLOR"] = "1"
    os.environ["COLUMNS"] = "100"
    parser = _capture_root_parser()
    sections = []
    for path in _command_paths(parser):
        title = " ".join(("hitlist", *path))
        sections.append(f"===== {title} --help =====\n{_help_text(path)}")
    return "\n".join(sections)


def test_cli_help_matches_snapshot(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "100")
    assert render_snapshot() == SNAPSHOT_PATH.read_text()


if __name__ == "__main__":
    SNAPSHOT_PATH.write_text(render_snapshot())
    print(f"wrote {SNAPSHOT_PATH}")
