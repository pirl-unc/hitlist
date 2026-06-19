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

"""Subtle help coloring: env gating + the alignment-safety invariant."""

import argparse
import re

from hitlist.cli_help import ColorArgumentParser, SubtleHelpFormatter, color_enabled

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _parser() -> ColorArgumentParser:
    p = ColorArgumentParser(prog="demo", description="demo tool")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("data", help="MS evidence datasets + observations index")
    sub.add_parser("panel", help="Build a CTA x HLA pMHC matrix")
    return p


def test_color_enabled_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled() is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert color_enabled() is True


def test_no_color_means_no_escapes(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert "\033[" not in _parser().format_help()


def test_forced_color_colors_subcommands_and_headings(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    out = _parser().format_help()
    assert "\033[36mdata\033[0m" in out  # subcommand name in cyan
    assert "\033[1m" in out  # bold section heading
    # The plain option name is NOT colored (only subcommands).
    assert "\033[36m-h" not in out


def test_coloring_is_alignment_safe(monkeypatch):
    # Stripping the ANSI from colored help must reproduce the plain help exactly:
    # coloring may only ADD escape sequences, never change layout/text.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    colored = _parser().format_help()
    monkeypatch.delenv("FORCE_COLOR")
    monkeypatch.setenv("NO_COLOR", "1")
    plain = _parser().format_help()
    assert _ANSI.sub("", colored) == plain


def test_subparser_tree_inherits_formatter():
    p = _parser()
    sub_action = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    for child in sub_action.choices.values():
        assert child.formatter_class is SubtleHelpFormatter
