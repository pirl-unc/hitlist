"""Test the ``hitlist --version`` flag added in v1.29.2."""

from __future__ import annotations

import pytest


def test_version_flag_prints_version_and_exits(monkeypatch, capsys):
    from hitlist.cli import main
    from hitlist.version import __version__

    monkeypatch.setattr("sys.argv", ["hitlist", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
