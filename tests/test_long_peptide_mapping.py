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


# ── One seed index, any peptide length (#398) ────────────────────────────────

SEED_ONLY = (7,)


class TestSingleSeedIndex:
    """The index carries ONE length; peptide length is a separate concern.

    Indexing 8/9/10/11 separately cost ~4x the memory and build time to buy
    nothing: seed selectivity is flat in k on a real proteome because
    multiplicity comes from isoforms, not sequence repetition.
    """

    @pytest.fixture(scope="class")
    def seed_index(self):
        proteins = {"P1": HOST, "P2": DECOY}
        meta = {"P1": {"gene_name": "G1"}, "P2": {"gene_name": "G2"}}
        return ProteomeIndex._build(proteins, meta, lengths=SEED_ONLY, verbose=False)

    def test_index_stores_a_single_length(self, seed_index):
        assert seed_index.lengths == SEED_ONLY

    def test_long_peptide_maps_through_the_seed(self, seed_index):
        """20 residues located by a 7-residue seed -- 13 past the seed and
        8 past the 63-bit encoding ceiling."""
        assert seed_index.lookup(LONG_PEPTIDE) == [("P1", 5)]

    @pytest.mark.parametrize("length", [7, 8, 9, 11, 12, 15, 20])
    def test_every_length_at_or_above_the_seed_resolves(self, seed_index, length):
        """The class I lengths and the class II lengths take the same path."""
        pep = HOST[5 : 5 + length]
        assert ("P1", 5) in seed_index.lookup(pep)

    def test_verification_rejects_a_seed_match_that_diverges(self, seed_index):
        """DECOY shares LONG_PEPTIDE's first 11 residues, so it matches the
        seed. Only checking the full span against the source sequence keeps
        the answer exact -- delete that check and this returns P2 too."""
        hits = dict(seed_index.lookup(LONG_PEPTIDE))
        assert "P2" not in hits
        # ...and the shared prefix really does hit both, so the test is not
        # vacuous: the seed alone cannot tell them apart.
        assert len(seed_index.lookup(LONG_PEPTIDE[:7])) == 2

    def test_below_the_seed_returns_nothing_rather_than_guessing(self, seed_index):
        """A 6-mer cannot be seeded at k=7. Returning [] is honest; the
        alternative is lowering the seed, which costs far more than the
        2,826 corpus rows at length 2-6 are worth (k=5 mean 16.6 hits/seed,
        worst case 10,022)."""
        assert seed_index.lookup(LONG_PEPTIDE[:6]) == []
        assert seed_index.lookup("") == []


class TestFlankWidth:
    """Flanks are sliced after the position is known, so width is an output
    decision independent of the index."""

    @pytest.fixture(scope="class")
    def seed_index(self):
        proteins = {"P1": HOST}
        return ProteomeIndex._build(
            proteins, {"P1": {"gene_name": "G1"}}, lengths=SEED_ONLY, verbose=False
        )

    def test_default_flank_is_fifteen(self):
        from hitlist.proteome import DEFAULT_FLANK

        assert DEFAULT_FLANK == 15

    def test_flanks_truncate_at_termini_rather_than_padding(self, seed_index):
        """HOST puts LONG_PEPTIDE at offset 5, so a 15-residue N-flank cannot
        be filled. The row is still a real mapping -- a short flank means
        'near a terminus', never 'context missing'. Consumers that read a
        short flank as absent data invert exactly the distinction #392 was
        about, one layer down.
        """
        df = seed_index.map_peptides([LONG_PEPTIDE], flank=15, verbose=False)
        row = df.iloc[0]
        assert row["position"] == 5
        assert row["n_flank"] == "AAAAA"  # all 5 residues that exist, not padded
        assert len(row["n_flank"]) < 15
        assert row["c_flank"] == "CCCCCGGGGG"  # 10 available, also truncated

    def test_full_width_flank_when_the_sequence_allows(self):
        pep = "MKVLAAGIVG"
        seq = "W" * 20 + pep + "Y" * 20
        idx = ProteomeIndex._build(
            {"P1": seq}, {"P1": {"gene_name": "G1"}}, lengths=SEED_ONLY, verbose=False
        )
        row = idx.map_peptides([pep], flank=15, verbose=False).iloc[0]
        assert row["n_flank"] == "W" * 15
        assert row["c_flank"] == "Y" * 15
