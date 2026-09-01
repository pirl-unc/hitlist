"""Peptides longer than the indexed k-mer length still map (hitlist#394).

The packed index encodes each k-mer into one 63-bit integer, so it can only be
built at lengths where ``bits * k <= 63`` -- roughly 12 residues at 5 bits.
The mapping pass therefore indexed 8..11 and dropped everything above, which
silently excluded **every** class II peptide: they run 12-25 residues, so none
of the 569,670 unique class II peptides (1,395,872 observation rows) ever
received flanks, position, gene or protein annotation.

Widening the indexed lengths is not the fix -- above the bit budget the build
falls back to the legacy dict index, the ~10 GB-per-length path that #109
removed. Instead a long peptide is resolved against the longest existing index
by prefix-and-verify.
"""

import pytest

from hitlist.proteome import ProteomeIndex

#: 20 residues -- a typical class II ligand length, and well past the 63-bit
#: encoding ceiling of ~12.
LONG_PEPTIDE = "MKVLAAGIVGWQTYRSDEFH"
#: LONG_PEPTIDE sits at offset 5, with 5 residues of context on each side.
HOST = "AAAAA" + LONG_PEPTIDE + "CCCCCGGGGG"
#: Shares LONG_PEPTIDE's first 11 residues and then diverges. A prefix lookup
#: alone would return this; verification must reject it.
DECOY = "TTTTT" + LONG_PEPTIDE[:11] + "PPPPPPPPPPPP"

INDEXED_LENGTHS = (8, 9, 10, 11)


@pytest.fixture(scope="module")
def index():
    proteins = {"P1": HOST, "P2": DECOY}
    meta = {"P1": {"gene_name": "G1"}, "P2": {"gene_name": "G2"}}
    return ProteomeIndex._build(proteins, meta, lengths=INDEXED_LENGTHS, verbose=False)


class TestLongPeptideLookup:
    def test_long_peptide_is_found(self):
        proteins = {"P1": HOST}
        idx = ProteomeIndex._build(proteins, {"P1": {}}, lengths=INDEXED_LENGTHS, verbose=False)
        assert idx.lookup(LONG_PEPTIDE) == [("P1", 5)]

    def test_prefix_match_alone_is_not_enough(self, index):
        """The decoy shares 11 residues and must still be rejected.

        Without the verification step this returns a spurious hit, and the
        peptide would be annotated to the wrong protein with wrong flanks --
        worse than the missing data it replaces.
        """
        assert index.lookup(LONG_PEPTIDE) == [("P1", 5)]

    def test_absent_long_peptide_returns_nothing(self, index):
        assert index.lookup("W" * 20) == []

    def test_flanks_are_correct_for_a_long_peptide(self, index):
        frame = index.map_peptides([LONG_PEPTIDE], flank=5, verbose=False)
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["n_flank"] == "AAAAA"
        assert row["c_flank"] == "CCCCC"
        assert row["position"] == 5

    @pytest.mark.parametrize("length", [12, 15, 20, 25])
    def test_the_whole_class_ii_range_maps(self, length):
        """12-25 covers 96.1% of class II observation rows."""
        peptide = HOST[5 : 5 + length]
        idx = ProteomeIndex._build({"P1": HOST}, {"P1": {}}, lengths=INDEXED_LENGTHS, verbose=False)
        assert idx.lookup(peptide) == [("P1", 5)]


class TestShortAndExactLengthsAreUnchanged:
    """The fix must not disturb the MHC-I path it borrows the index from."""

    @pytest.mark.parametrize("length", INDEXED_LENGTHS)
    def test_indexed_lengths_still_exact_match(self, index, length):
        peptide = HOST[5 : 5 + length]
        hits = index.lookup(peptide)
        assert ("P1", 5) in hits

    def test_peptide_shorter_than_every_index_finds_nothing(self, index):
        """Below the shortest index there is nothing to prefix against."""
        assert index.lookup(HOST[5:11]) == []
