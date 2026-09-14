"""Tests for the ``tests.mhcgnomes_floor_check`` version tripwire (#467).

Exercises the parsing and comparison logic directly -- pure functions, no
need to actually install a stale mhcgnomes to prove the message is right --
plus one direct check that `conftest.pytest_configure` wires the tripwire
up: calls `pytest.exit` with the check's message when it fails, and does
nothing when it passes.
"""

from __future__ import annotations

import pytest

from tests import conftest
from tests.mhcgnomes_floor_check import check, declared_floor, floor_violation_message

# ── declared_floor ───────────────────────────────────────────────────────


def test_declared_floor_parses_the_real_pyproject():
    """The real file, not a synthetic one -- proves the regex still matches
    pyproject.toml's actual current form, not just a string shaped for it.
    """
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    floor = declared_floor(text)
    assert floor.count(".") == 2
    assert all(part.isdigit() for part in floor.split("."))


def test_declared_floor_parses_a_synthetic_declaration():
    text = '[project.optional-dependencies]\nalleles = [\n    "mhcgnomes>=3.54.0",\n]\n'
    assert declared_floor(text) == "3.54.0"


def test_declared_floor_raises_when_the_declaration_is_missing():
    with pytest.raises(ValueError, match="could not find"):
        declared_floor("[project]\nname = 'hitlist'\n")


# ── floor_violation_message ─────────────────────────────────────────────


def test_floor_violation_message_is_none_when_installed_matches_the_floor():
    assert floor_violation_message(floor="3.54.0", installed="3.54.0", resolved_from="/x") is None


def test_floor_violation_message_is_none_when_installed_exceeds_the_floor():
    assert floor_violation_message(floor="3.54.0", installed="3.64.2", resolved_from="/x") is None


def test_floor_violation_message_names_both_versions_and_the_resolved_path():
    """The exact regression this issue reports: mhcgnomes 3.33.4 shadowing
    a 3.64.2 sibling checkout, from `/path/to/shared-virtual-env`.
    """
    message = floor_violation_message(
        floor="3.54.0",
        installed="3.33.4",
        resolved_from="/path/to/shared-virtual-env/lib/python3.12/site-packages/mhcgnomes/__init__.py",
    )
    assert message is not None
    assert "3.33.4" in message
    assert "3.54.0" in message
    assert "shared-virtual-env" in message


def test_floor_violation_message_compares_numerically_not_lexically():
    """3.9.0 must satisfy a 3.54.0 floor as "older", not as "newer" by a
    naive string comparison ("3.9.0" > "3.54.0" lexically, backwards).
    """
    message = floor_violation_message(floor="3.54.0", installed="3.9.0", resolved_from="/x")
    assert message is not None
    message_ok = floor_violation_message(floor="3.9.0", installed="3.54.0", resolved_from="/x")
    assert message_ok is None


# ── check() against the real environment ────────────────────────────────


def test_check_passes_in_this_environment():
    """This suite is running at all, so the installed mhcgnomes must already
    satisfy the floor -- a real end-to-end run of the tripwire's happy path.
    """
    assert check() is None


# ── pytest_configure wiring ──────────────────────────────────────────────


def test_pytest_configure_exits_when_the_check_fails(monkeypatch):
    monkeypatch.setattr(
        conftest, "_check_mhcgnomes_floor", lambda: "installed mhcgnomes 3.33.4 is older..."
    )
    with pytest.raises(pytest.exit.Exception, match=r"3\.33\.4"):
        conftest.pytest_configure(config=None)


def test_pytest_configure_does_nothing_when_the_check_passes(monkeypatch):
    monkeypatch.setattr(conftest, "_check_mhcgnomes_floor", lambda: None)
    conftest.pytest_configure(config=None)  # must not raise
