"""The new optional provider does not raise Hitlist's overall Python floor."""

import pytest

from hitlist import canine_peptide_evidence


def test_python_310_gets_explicit_provider_requirement(monkeypatch):
    monkeypatch.setattr(canine_peptide_evidence.sys, "version_info", (3, 10, 0))
    with pytest.raises(RuntimeError, match=r"Python 3\.11"):
        canine_peptide_evidence.open_canine_peptide_evidence("unused")


def test_older_orfdata_gets_explicit_provider_requirement(monkeypatch):
    monkeypatch.setattr(canine_peptide_evidence.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(canine_peptide_evidence, "version", lambda name: "0.39.0")
    with pytest.raises(RuntimeError, match=r"ORFdata 0\.40"):
        canine_peptide_evidence.open_canine_peptide_evidence("unused")
