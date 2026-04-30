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

"""Tests for hitlist.peptide_modifications — IEDB PTM annotation parser."""

from __future__ import annotations

from hitlist.peptide_modifications import parse_peptide_modifications


def test_no_ptm_passes_through():
    """Bare AA sequence: bare equals input, no modifications, has_ptm False."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("LQPFPQPQLPY")
    assert bare == "LQPFPQPQLPY"
    assert mods == ""
    assert has_ptm is False
    assert ext == "LQPFPQPQLPY"


def test_single_position_deamidation():
    """The most common pattern: ``BARE + DEAM(Q8)``."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("LQPFPQPQLPY + DEAM(Q8)")
    assert bare == "LQPFPQPQLPY"
    assert mods == "DEAM:Q8"
    assert has_ptm is True
    # ProForma maps DEAM → Deamidated. Position 8 is Q8 (1-indexed).
    assert ext == "LQPFPQPQ[Deamidated]LPY"


def test_multi_position_methylation():
    """``MCM(G3,G4)`` — multiple positions, same modification type."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("CLGGLLTMV + MCM(G3,G4)")
    assert bare == "CLGGLLTMV"
    # Canonical form lists each (mod, residue, position) tuple.
    assert "MCM:G3" in mods
    assert "MCM:G4" in mods
    assert has_ptm is True
    # G at positions 3 and 4 both get tagged.
    assert ext == "CLG[MCM]G[MCM]LLTMV"


def test_positions_with_whitespace():
    """IEDB sometimes embeds whitespace inside the position list."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("EVLYLKPLAGVYRSLKKQLE + MCM(K6, P7)")
    assert bare == "EVLYLKPLAGVYRSLKKQLE"
    assert "MCM:K6" in mods and "MCM:P7" in mods
    assert has_ptm is True


def test_trailing_space_inside_parens():
    """``MCM(F7, A8 )`` — trailing whitespace inside the position group."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("GILGFVFTL + MCM(F7, A8 )")
    assert bare == "GILGFVFTL"
    assert "MCM:F7" in mods and "MCM:A8" in mods
    assert has_ptm is True


def test_oxidation_proforma_tag():
    """``OX`` should map to the Unimod / ProForma standard ``Oxidation``."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("YLAGAATMV + OX(M8)")
    assert bare == "YLAGAATMV"
    assert mods == "OX:M8"
    assert has_ptm is True
    assert ext == "YLAGAATM[Oxidation]V"


def test_phospho_proforma_tag():
    """``PHOS`` should map to ``Phospho`` (canonical ProForma tag)."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("RLPGLAS + PHOS(S7)")
    assert bare == "RLPGLAS"
    assert mods == "PHOS:S7"
    assert has_ptm is True
    assert ext == "RLPGLAS[Phospho]"


def test_unknown_short_code_preserved_in_extended():
    """A modification short code we haven't mapped is still emitted in
    the extended string using its raw IEDB code, so no information is
    lost."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("PEPTIDE + ZZZ(P1)")
    # ZZZ isn't in our table — falls through to the literal short code.
    assert bare == "PEPTIDE"
    assert mods == "ZZZ:P1"
    assert has_ptm is True
    assert ext == "P[ZZZ]EPTIDE"


def test_unparseable_annotation_preserves_input():
    """Garbage that vaguely looks like a PTM but doesn't match the
    grammar — preserve the original string in both bare and extended,
    flag has_ptm False so consumers can grep these out."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("WEIRD STRING + xyz")
    assert bare == "WEIRD STRING + xyz"
    assert mods == ""
    assert has_ptm is False
    assert ext == "WEIRD STRING + xyz"


def test_residue_mismatch_skipped_in_extended():
    """If IEDB claims ``DEAM(Q8)`` but position 8 isn't a Q, we still
    parse the bare peptide and emit the canonical mod, but we DON'T
    inject a misleading inline tag at the wrong residue."""
    # Position 8 of "LQPFPQPALPY" is A, not Q.
    bare, mods, has_ptm, ext = parse_peptide_modifications("LQPFPQPALPY + DEAM(Q8)")
    assert bare == "LQPFPQPALPY"
    assert mods == "DEAM:Q8"
    assert has_ptm is True
    # Extended should NOT inject [Deamidated] at the wrong residue.
    assert "[Deamidated]" not in ext


def test_empty_string():
    """Empty input survives without crashing."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("")
    assert bare == ""
    assert mods == ""
    assert has_ptm is False
    assert ext == ""


def test_none_input_treated_as_empty():
    """Defensive: ``None`` → empty (avoids AttributeError on str()
    chain)."""
    bare, mods, has_ptm, ext = parse_peptide_modifications(None)  # type: ignore[arg-type]
    assert bare == ""
    assert mods == ""
    assert has_ptm is False
    assert ext == ""


def test_acetylation_proforma_tag():
    """``ACET`` → ``Acetyl``."""
    bare, mods, has_ptm, ext = parse_peptide_modifications("KAVYNFATM + ACET(K1)")
    assert bare == "KAVYNFATM"
    assert mods == "ACET:K1"
    assert has_ptm is True
    assert ext == "K[Acetyl]AVYNFATM"


def test_citrullination_proforma_tag():
    """``CITR`` → ``Citrullination``."""
    # VRPSGRYV: V=1 R=2 P=3 S=4 G=5 R=6 Y=7 V=8 — the R6 is the
    # second R, so the tag goes at index 5 (0-indexed).
    bare, mods, has_ptm, ext = parse_peptide_modifications("VRPSGRYV + CITR(R6)")
    assert bare == "VRPSGRYV"
    assert mods == "CITR:R6"
    assert has_ptm is True
    assert ext == "VRPSGR[Citrullination]YV"
