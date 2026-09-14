"""Version-floor tripwire for ``mhcgnomes`` (#467).

An environment whose installed ``mhcgnomes`` sits below the floor this
project declares in ``pyproject.toml`` fails the suite with 200+ scattered
``AttributeError``\\ s (e.g. ``Species.compatible_with`` missing below
3.39.0, spread across 15+ test files) that read like a broad regression
rather than one stale dependency. Nothing in any one traceback connects it
back to the actual cause: a release install of an old ``mhcgnomes`` in
site-packages shadowing a newer, correctly-locked one.

This module turns that into one message naming the declared floor, the
installed version, and the resolved import path -- exactly what a
shadowing install needs to be diagnosed at a glance. It is wired into
``conftest.py``'s ``pytest_configure`` hook, which runs once before any
test is collected.

Kept separate from ``conftest.py`` so the parsing/comparison logic has a
public surface that can be unit-tested independently of an actual pytest
session, mirroring ``tests.xdist_cache``.
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.version import Version

#: Matches the exact declaration in pyproject.toml's ``alleles`` extra,
#: e.g. ``"mhcgnomes>=3.54.0"``. Deliberately narrow: a format drift here
#: should fail loudly (see `declared_floor`) rather than silently stop
#: enforcing the floor.
_FLOOR_PATTERN = re.compile(r'"mhcgnomes>=([\d.]+)"')


def declared_floor(pyproject_text: str) -> str:
    """The mhcgnomes version floor declared in ``pyproject.toml``.

    Raises
    ------
    ValueError
        If no ``mhcgnomes>=X.Y.Z`` declaration is found. Silently skipping
        the tripwire because the declaration's format changed would be
        worse than a loud failure naming what to update.
    """
    match = _FLOOR_PATTERN.search(pyproject_text)
    if not match:
        raise ValueError(
            "could not find a `mhcgnomes>=X.Y.Z` floor declaration in pyproject.toml; "
            "the mhcgnomes version tripwire (#467, tests/mhcgnomes_floor_check.py) "
            "needs updating to match its new form."
        )
    return match.group(1)


def floor_violation_message(*, floor: str, installed: str, resolved_from: str) -> str | None:
    """``None`` if ``installed`` satisfies ``floor``, else the tripwire message."""
    if Version(installed) >= Version(floor):
        return None
    return (
        f"installed mhcgnomes {installed} is older than the floor this project "
        f"declares in pyproject.toml's `alleles` extra ({floor}).\n"
        f"  Resolved from: {resolved_from}\n"
        f"  This is almost always a stale or shadowing install -- e.g. a release "
        f"version in a shared environment's site-packages taking precedence over "
        f"a newer sibling checkout or a locked dependency.\n"
        f"  Fix: run `uv sync --locked --extra alleles --extra dev` (or "
        f"`pip install -e '.[alleles,dev]'`) in the environment actually running "
        f"pytest, then re-run."
    )


def check() -> str | None:
    """The tripwire itself: parse the declared floor, compare it against
    the installed ``mhcgnomes``, and return a diagnostic message if it
    falls short -- ``None`` if the environment is fine.
    """
    import mhcgnomes

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    floor = declared_floor(pyproject.read_text())
    return floor_violation_message(
        floor=floor,
        installed=mhcgnomes.__version__,
        resolved_from=mhcgnomes.__file__,
    )
