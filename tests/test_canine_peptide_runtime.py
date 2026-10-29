"""Native peptide readers fail clearly on unsupported ORFdata installations."""

import pytest

from hitlist import canine_peptide_evidence


def test_older_orfdata_gets_explicit_provider_requirement(monkeypatch):
    monkeypatch.setattr(canine_peptide_evidence, "version", lambda name: "0.39.0")
    with pytest.raises(RuntimeError, match=r"ORFdata 0\.40"):
        canine_peptide_evidence.open_canine_peptide_evidence("unused")
