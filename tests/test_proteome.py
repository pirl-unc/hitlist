import pandas as pd

from hitlist.proteome import ProteomeIndex


def _make_index():
    """Build a small test index from 2 proteins."""
    proteins = {
        "PROT1": "ACDEFGHIKLMNPQRSTVWY",  # 20 aa
        "PROT2": "XXXXXGHIKLMYYYY",  # 15 aa, shares GHIKLM with PROT1
    }
    meta = {
        "PROT1": {"gene_name": "GENE1", "gene_id": "ENSG1"},
        "PROT2": {"gene_name": "GENE2", "gene_id": "ENSG2"},
    }
    return ProteomeIndex._build(proteins, meta, lengths=(5, 6), verbose=False)


def test_build_index():
    idx = _make_index()
    assert len(idx.proteins) == 2
    assert len(idx.index) > 0


def test_lookup_unique():
    idx = _make_index()
    # ACDEF only in PROT1
    hits = idx.lookup("ACDEF")
    assert len(hits) == 1
    assert hits[0][0] == "PROT1"
    assert hits[0][1] == 0  # position 0


def test_lookup_shared():
    idx = _make_index()
    # GHIKL is in both proteins
    hits = idx.lookup("GHIKL")
    assert len(hits) == 2
    prot_ids = {h[0] for h in hits}
    assert prot_ids == {"PROT1", "PROT2"}


def test_lookup_missing():
    idx = _make_index()
    assert idx.lookup("ZZZZZ") == []


def test_map_peptides_flanks():
    idx = _make_index()
    df = idx.map_peptides(["ACDEF"], flank=3, verbose=False)
    assert len(df) == 1
    assert df.iloc[0]["n_flank"] == ""  # at start, no N-flank
    assert df.iloc[0]["c_flank"] == "GHI"  # 3 residues after ACDEF


def test_map_peptides_shared_different_flanks():
    idx = _make_index()
    df = idx.map_peptides(["GHIKL"], flank=2, verbose=False)
    assert len(df) == 2
    # unique_n_flank should be empty (different flanks in different proteins)
    assert df.iloc[0]["unique_n_flank"] == ""


def test_map_peptides_unique_flank():
    idx = _make_index()
    df = idx.map_peptides(["ACDEF"], flank=2, verbose=False)
    assert len(df) == 1
    assert df.iloc[0]["unique_n_flank"] == ""  # only 1 source, but nflank is empty string
    assert df.iloc[0]["unique_c_flank"] == "GH"  # unique since only 1 source
    assert df.iloc[0]["n_sources"] == 1


def test_map_peptides_empty():
    idx = _make_index()
    df = idx.map_peptides([], verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "peptide" in df.columns
    # Empty df has base columns only
    assert "peptide" in df.columns


def test_merge():
    proteins_a = {"PROT_A": "ACDEFGHIKLMN"}
    meta_a = {"PROT_A": {"gene_name": "GENE_A", "gene_id": "A1"}}
    proteins_b = {"PROT_B": "PQRSTVWYACDE"}
    meta_b = {"PROT_B": {"gene_name": "GENE_B", "gene_id": "B1"}}

    idx_a = ProteomeIndex._build(proteins_a, meta_a, lengths=(5,), verbose=False)
    idx_b = ProteomeIndex._build(proteins_b, meta_b, lengths=(5,), verbose=False)

    merged = idx_a.merge(idx_b)
    assert len(merged.proteins) == 2
    # ACDEF in both (shared between PROT_A pos 0 and PROT_B pos 7)
    # but "ACDE" is only 4 chars, we indexed 5-mers
    # ACDEF: PROT_A at 0, not in PROT_B (PROT_B has VYACDE, YACDE at pos 7)
    hits = merged.lookup("ACDEF")
    assert len(hits) == 1  # only in PROT_A
    # PQRST only in PROT_B
    hits_b = merged.lookup("PQRST")
    assert len(hits_b) == 1
    assert hits_b[0][0] == "PROT_B"


def test_merge_map_peptides():
    idx_human = ProteomeIndex._build(
        {"HUMAN1": "ACDEFGHIKLMN"},
        {"HUMAN1": {"gene_name": "MAGEA4", "gene_id": "ENSG1"}},
        lengths=(5,),
        verbose=False,
    )
    idx_viral = ProteomeIndex._build(
        {"VIRAL1": "PQRSTVWYACDE"},
        {"VIRAL1": {"gene_name": "E7", "gene_id": "HPV16"}},
        lengths=(5,),
        verbose=False,
    )
    merged = idx_human.merge(idx_viral)
    df = merged.map_peptides(["PQRST", "ACDEF"], flank=3, verbose=False)
    assert len(df) == 2
    genes = set(df["gene_name"])
    assert genes == {"MAGEA4", "E7"}


def test_from_fasta(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">sp|P12345|TEST_HUMAN GN=TESTGENE\nACDEFGHIKL\n>sp|Q99999|OTH\nMNPQRSTVWY\n")
    idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert len(idx.proteins) == 2
    # GN= parsing
    assert idx.protein_meta["sp|P12345|TEST_HUMAN"]["gene_name"] == "TESTGENE"
    df = idx.map_peptides(["ACDEF"], flank=3, verbose=False)
    assert len(df) == 1
    assert df.iloc[0]["gene_name"] == "TESTGENE"


def test_from_fasta_caches_by_path(tmp_path):
    """Repeated from_fasta with the same inputs returns the same instance.

    This matters because hitlist.builder._add_flanking iterates canonical
    species and many viral strain variants resolve to a single shared
    cached FASTA — without memoization we'd rebuild the same index N
    times per build. See issue pirl-unc/hitlist#86.
    """
    from hitlist.proteome import clear_fasta_index_cache

    clear_fasta_index_cache()
    fasta = tmp_path / "shared.fasta"
    fasta.write_text(">sp|P12345|TEST_HUMAN GN=TESTGENE\nACDEFGHIKL\n")
    idx1 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    idx2 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    # Same instance — memoized by (path, size, mtime, lengths, ...).
    assert idx1 is idx2


def test_from_fasta_cache_invalidates_on_mtime(tmp_path):
    """Touching the FASTA (size or mtime change) rebuilds the index."""
    from hitlist.proteome import clear_fasta_index_cache

    clear_fasta_index_cache()
    fasta = tmp_path / "drifting.fasta"
    fasta.write_text(">sp|P12345|A\nACDEFGHIKL\n")
    idx1 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)

    # Replace with different content + bump mtime.
    import os
    import time

    time.sleep(0.01)
    fasta.write_text(">sp|Q99999|B\nMNPQRSTVWY\n")
    os.utime(fasta, None)
    idx2 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert idx1 is not idx2
    assert "sp|P12345|A" in idx1.proteins
    assert "sp|Q99999|B" in idx2.proteins


def test_from_fasta_cache_keyed_on_lengths(tmp_path):
    """Different `lengths` kwarg → different cache entry."""
    from hitlist.proteome import clear_fasta_index_cache

    clear_fasta_index_cache()
    fasta = tmp_path / "multi_len.fasta"
    fasta.write_text(">sp|P1|A\nACDEFGHIKLMNPQ\n")
    idx_5 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    idx_8 = ProteomeIndex.from_fasta(fasta, lengths=(8,), verbose=False)
    assert idx_5 is not idx_8
    assert idx_5.lengths == (5,)
    assert idx_8.lengths == (8,)


# ---------------------------------------------------------------------------
# K-mer set primitive (#99) — shared with tsarina, perseus, topiary.
# ---------------------------------------------------------------------------


def _small_index_with_genes():
    """Build a small ProteomeIndex with explicit gene_id metadata."""
    proteins = {
        "P1": "ACDEFGHIKL",  # 10 aa
        "P2": "MNPQRSTVWY",  # 10 aa
        "P3": "AAAAAACDEF",  # 10 aa — shares ACDEF with P1
    }
    meta = {
        "P1": {"gene_name": "GENE_A", "gene_id": "ENSG001"},
        "P2": {"gene_name": "GENE_B", "gene_id": "ENSG002"},
        "P3": {"gene_name": "GENE_A", "gene_id": "ENSG001"},
    }
    return ProteomeIndex._build(proteins, meta, lengths=(5,), verbose=False)


def test_all_kmers_returns_frozenset():
    idx = _small_index_with_genes()
    kmers = idx.all_kmers
    assert isinstance(kmers, frozenset)
    # Well-defined: every 5-mer in the three 10-aa proteins.
    expected = set()
    for seq in idx.proteins.values():
        for i in range(len(seq) - 4):
            expected.add(seq[i : i + 5])
    assert kmers == frozenset(expected)


def test_all_kmers_cached_same_instance():
    idx = _small_index_with_genes()
    a = idx.all_kmers
    b = idx.all_kmers
    # cached_property returns the same frozenset object.
    assert a is b


def test_kmers_for_genes_subset():
    idx = _small_index_with_genes()
    kmers_a = idx.kmers_for_genes(frozenset({"ENSG001"}))
    kmers_b = idx.kmers_for_genes(frozenset({"ENSG002"}))
    # P2's unique k-mers must not be in the A-only set.
    assert "MNPQR" not in kmers_a
    assert "MNPQR" in kmers_b
    # ACDEF appears in P1 (ENSG001) and at end of P3 (ENSG001) but NOT P2.
    assert "ACDEF" in kmers_a
    assert "ACDEF" not in kmers_b
    # Full-union sanity: every protein is covered by exactly one gene ID
    # in this fixture, so the two subsets union to the full k-mer set.
    assert (kmers_a | kmers_b) == idx.all_kmers


def test_kmers_for_genes_empty_input():
    idx = _small_index_with_genes()
    assert idx.kmers_for_genes(frozenset()) == frozenset()


def test_kmers_for_genes_unknown_gene():
    idx = _small_index_with_genes()
    # Gene IDs absent from the index → empty result, no crash.
    assert idx.kmers_for_genes(frozenset({"ENSG_NOSUCH"})) == frozenset()


# ---------------------------------------------------------------------------
# In-silico protease digest (#104).
# ---------------------------------------------------------------------------


def test_digest_trypsin_basic():
    """Trypsin/P cleaves K/R, not before P."""
    from hitlist.proteome import digest

    # Layout: M(0) E(1) R(2) K(3) P(4) K(5) L(6) A(7) S(8) R(9) P(10) E(11) K(12)
    # Cuts:
    #   R@2 → next K, not P → cleave (cut at 3)
    #   K@3 → next P → NO
    #   K@5 → next L → cleave (cut at 6)
    #   R@9 → next P → NO
    #   K@12 → end of seq → NO
    # Result: [0:3]=MER, [3:6]=KPK, [6:13]=LASRPEK
    seq = "MERKPKLASRPEK"
    peps = digest(seq, enzyme="Trypsin/P (cleaves K/R except before P)", min_len=2, max_missed=0)
    assert peps == {"MER", "KPK", "LASRPEK"}


def test_digest_trypsin_aliases():
    """Short aliases resolve to the canonical form."""
    from hitlist.proteome import digest

    seq = "MERKASRLEK"
    canonical = digest(seq, enzyme="Trypsin/P (cleaves K/R except before P)", min_len=2)
    for alias in ("Trypsin", "Trypsin/P", "trypsin"):
        assert digest(seq, enzyme=alias, min_len=2) == canonical


def test_digest_chymotrypsin_plus_includes_m():
    """MaxQuant Chymotrypsin+ cleaves F/W/Y/L/M, not before P."""
    from hitlist.proteome import digest

    seq = "AFAWAYALAMAP"
    peps = digest(seq, enzyme="Chymotrypsin", min_len=2, max_missed=0)
    # Interior cuts produce peptides ending in F/W/Y/L/M; the trailing
    # fragment "AP" is the protein tail after the last M@9 cut and is
    # allowed to end in whatever residue is at the C-terminus.
    c_terms = {p[-1] for p in peps}
    assert c_terms.issubset(set("FWYLMAP"))


def test_digest_gluc_cleaves_both_e_and_d():
    """GluC in hitlist is the bicarbonate variant ``GluC;D.P`` — cleaves E AND D."""
    from hitlist.proteome import digest

    seq = "AAEAADAAAEP"
    peps = digest(seq, enzyme="GluC", min_len=2, max_missed=0)
    # E@2 → cut ; D@5 → cut ; E@9 before P → NO cut → segments AAE, AAD, AAAEP
    assert peps == {"AAE", "AAD", "AAAEP"}
    # D-ending peptide proves bicarbonate rule (phosphate GluC would be E-only).
    assert any(p.endswith("D") for p in peps)


def test_digest_lysc_cleaves_k_p():
    """LysC/P allows K-P cleavage (unlike Trypsin)."""
    from hitlist.proteome import digest

    seq = "AAAKPAAAK"
    tryp = digest(seq, enzyme="Trypsin/P", min_len=2, max_missed=0)
    lysc = digest(seq, enzyme="LysC", min_len=2, max_missed=0)
    # Trypsin: K@3 before P → NO cut → one long peptide.
    assert "AAAKPAAAK" in tryp
    # LysC: K@3 before P → cut → AAAK + PAAAK.
    assert lysc == {"AAAK", "PAAAK"}


def test_digest_missed_cleavages():
    """max_missed=2 emits peptides spanning up to 3 fully-cleaved segments."""
    from hitlist.proteome import digest

    seq = "AAKBBKCCKDDKEE"
    zero = digest(seq, enzyme="Trypsin/P", min_len=2, max_missed=0)
    two = digest(seq, enzyme="Trypsin/P", min_len=2, max_missed=2)
    assert zero < two
    assert "AAKBBK" in two  # 1-missed
    assert "AAKBBK" not in zero
    assert "AAKBBKCCK" in two  # 2-missed


def test_digest_length_bounds_inclusive():
    from hitlist.proteome import digest

    # Cuts after K@1, K@3, K@5, K@7 → fully-cleaved segments
    # AK(2), BK(2), CK(2), DK(2), EE(2). All are 2-mers.
    seq = "AKBKCKDKEE"
    peps = digest(seq, enzyme="Trypsin/P", min_len=2, max_len=2, max_missed=0)
    assert peps == {"AK", "BK", "CK", "DK", "EE"}
    # min_len=3 drops all of the above; only multi-segment peptides from
    # missed cleavages survive.
    peps_3 = digest(seq, enzyme="Trypsin/P", min_len=3, max_missed=2)
    assert all(len(p) >= 3 for p in peps_3)
    assert "AKBK" in peps_3  # 1-missed, 4 chars


def test_digest_unknown_enzyme_raises():
    import pytest

    from hitlist.proteome import digest

    with pytest.raises(ValueError, match="Unknown enzyme"):
        digest("MKRPE", enzyme="ProteinaseK")
