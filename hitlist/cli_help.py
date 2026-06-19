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

"""Subtle ANSI coloring for argparse ``--help`` output.

Shared by hitlist and tsarina so their CLIs look identical. The treatment is
deliberately minimal: subcommand names in cyan, section headings in bold. Color
is emitted only when writing to a TTY, and honors the ``NO_COLOR`` and
``FORCE_COLOR`` conventions.

Use :class:`ColorArgumentParser` as the root parser; ``add_subparsers`` uses
``type(self)`` as its parser class, so the whole subcommand tree inherits the
formatter automatically -- one line of wiring per CLI.
"""

from __future__ import annotations

import argparse
import os
import sys

_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def color_enabled(stream=None) -> bool:
    """True if help output to *stream* (default stdout) should be colorized.

    ``NO_COLOR`` (https://no-color.org/) disables color unconditionally;
    ``FORCE_COLOR`` forces it on; otherwise color is used only on a TTY.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream if stream is not None else sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


class SubtleHelpFormatter(argparse.HelpFormatter):
    """Colors subcommand names (cyan) and section headings (bold) on a TTY.

    Coloring is applied in :meth:`_format_action` *after* argparse has measured
    the plain text and laid out the help columns -- it only wraps the
    already-positioned token in ANSI escapes, so visible widths (and thus
    alignment) are unchanged. Verified by the round-trip test: stripping the
    ANSI from colored help reproduces the plain help byte-for-byte.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._color = color_enabled()

    def _format_action(self, action: argparse.Action) -> str:
        text = super()._format_action(action)
        # Only the subparser-choice pseudo-actions are subcommands.
        if self._color and type(action).__name__ == "_ChoicesPseudoAction":
            invocation = self._format_action_invocation(action)
            if invocation:
                text = text.replace(invocation, f"{_CYAN}{invocation}{_RESET}", 1)
        return text

    def start_section(self, heading: str | None) -> None:
        if self._color and heading:
            heading = f"{_BOLD}{heading}{_RESET}"
        super().start_section(heading)


class ColorArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` defaulting to :class:`SubtleHelpFormatter`.

    Because ``add_subparsers`` uses ``type(self)`` as its ``parser_class``, every
    subparser created from a root ``ColorArgumentParser`` inherits the formatter,
    so the whole command tree is colored from one root declaration.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", SubtleHelpFormatter)
        super().__init__(*args, **kwargs)
