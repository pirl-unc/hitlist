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


@pytest.mark.parametrize("export_command", ["ms", "binding", "training"])
def test_export_cli_accepts_every_mhc_allele_provenance(export_command, monkeypatch):
    """#419: the parser vocabulary cannot lag values emitted by the scanner."""
    from hitlist.cli import main
    from hitlist.curation import MHC_ALLELE_PROVENANCE_VALUES

    parsed = []
    monkeypatch.setattr("hitlist.cli._export", lambda args: parsed.append(args))

    for provenance in MHC_ALLELE_PROVENANCE_VALUES:
        monkeypatch.setattr(
            "sys.argv",
            [
                "hitlist",
                "export",
                export_command,
                "--mhc-allele-provenance",
                provenance,
            ],
        )
        main()

    assert [args.mhc_allele_provenance for args in parsed] == [
        [provenance] for provenance in MHC_ALLELE_PROVENANCE_VALUES
    ]
