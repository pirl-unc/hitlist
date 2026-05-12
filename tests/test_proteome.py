import pandas as pd
import pytest

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


# ── Bounded LRU cache (issue #109 regression guard) ────────────────────────


def test_from_fasta_cache_is_bounded_after_many_distinct_fastas(tmp_path):
    """Indexing N > maxsize distinct FASTAs in sequence must NOT accumulate
    every ProteomeIndex in memory.  Pre-#109 the cache was unbounded and
    the human proteome alone (~10 GB resident) would persist for the rest
    of the build.  The bounded LRU caps live entries at ``maxsize``.
    """
    from hitlist.proteome import (
        _FASTA_INDEX_CACHE,
        clear_fasta_index_cache,
        set_fasta_index_cache_maxsize,
    )

    clear_fasta_index_cache()
    set_fasta_index_cache_maxsize(3)
    try:
        for i in range(10):
            fasta = tmp_path / f"distinct_{i}.fasta"
            fasta.write_text(f">sp|P{i:05d}|A\nACDEFGHIKLMN\n")
            ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
        assert len(_FASTA_INDEX_CACHE) == 3, (
            f"expected ≤3 cached entries with maxsize=3, got "
            f"{len(_FASTA_INDEX_CACHE)} — LRU eviction is broken"
        )
    finally:
        # Restore the production default so subsequent tests aren't affected.
        set_fasta_index_cache_maxsize(4)
        clear_fasta_index_cache()


def test_from_fasta_cache_evicts_least_recently_used(tmp_path):
    """The LRU eviction order must be true-LRU: the oldest UNTOUCHED entry
    is evicted, while a recent re-fetch of an old entry keeps it warm.
    """
    from hitlist.proteome import (
        _FASTA_INDEX_CACHE,
        clear_fasta_index_cache,
        set_fasta_index_cache_maxsize,
    )

    clear_fasta_index_cache()
    set_fasta_index_cache_maxsize(2)
    try:
        a = tmp_path / "a.fasta"
        b = tmp_path / "b.fasta"
        c = tmp_path / "c.fasta"
        for f, sym in ((a, "A"), (b, "B"), (c, "C")):
            f.write_text(f">sp|P|{sym}\nACDEFGHIKLMN\n")

        idx_a = ProteomeIndex.from_fasta(a, lengths=(5,), verbose=False)
        idx_b = ProteomeIndex.from_fasta(b, lengths=(5,), verbose=False)
        # Touch A so it becomes most-recently-used; B should now be the
        # eviction target when C arrives.
        idx_a_again = ProteomeIndex.from_fasta(a, lengths=(5,), verbose=False)
        assert idx_a_again is idx_a

        ProteomeIndex.from_fasta(c, lengths=(5,), verbose=False)
        assert len(_FASTA_INDEX_CACHE) == 2

        # A and C remain; B was evicted, so re-fetching B builds a fresh
        # instance (different identity from idx_b).
        idx_b_after = ProteomeIndex.from_fasta(b, lengths=(5,), verbose=False)
        assert idx_b_after is not idx_b
    finally:
        set_fasta_index_cache_maxsize(4)
        clear_fasta_index_cache()


def test_from_fasta_cache_maxsize_zero_disables_caching(tmp_path):
    """``set_fasta_index_cache_maxsize(0)`` disables the cache entirely.

    Useful as an escape hatch for pipelines that explicitly want every
    FASTA re-indexed (e.g. when an external process is rewriting them
    with stable mtimes).
    """
    from hitlist.proteome import (
        _FASTA_INDEX_CACHE,
        clear_fasta_index_cache,
        set_fasta_index_cache_maxsize,
    )

    clear_fasta_index_cache()
    set_fasta_index_cache_maxsize(0)
    try:
        fasta = tmp_path / "no_cache.fasta"
        fasta.write_text(">sp|P|A\nACDEFGHIKLMN\n")
        idx1 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
        idx2 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
        assert idx1 is not idx2
        assert len(_FASTA_INDEX_CACHE) == 0
    finally:
        set_fasta_index_cache_maxsize(4)
        clear_fasta_index_cache()


def test_from_fasta_cache_maxsize_negative_rejected():
    from hitlist.proteome import set_fasta_index_cache_maxsize

    with pytest.raises(ValueError, match="non-negative"):
        set_fasta_index_cache_maxsize(-1)


# ── Transcript-aware mapping output (issue #141) ───────────────────────────


def test_from_fasta_meta_includes_transcript_columns(tmp_path):
    """FASTA-backed indexes carry empty transcript_id + canonical=False so
    the schema is uniform with the Ensembl path (issue #141).
    """
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">sp|P1|A GN=GENA\nACDEFGHIKLMN\n")
    idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    meta = idx.protein_meta["sp|P1|A"]
    assert meta["transcript_id"] == ""
    assert meta["is_canonical_transcript"] is False
    assert meta["gene_name"] == "GENA"


def test_map_peptides_emits_transcript_columns(tmp_path):
    """``map_peptides`` output now carries ``transcript_id`` and
    ``is_canonical_transcript`` (issue #141).  Both default to empty /
    False on FASTA-backed indexes; the columns are always present.
    """
    fasta = tmp_path / "tx.fasta"
    fasta.write_text(">sp|P1|A GN=GENA\nACDEFGHIKL\n")
    idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    df = idx.map_peptides(["ACDEF"], flank=2, verbose=False)
    assert "transcript_id" in df.columns
    assert "is_canonical_transcript" in df.columns
    assert (df["transcript_id"] == "").all()
    assert (df["is_canonical_transcript"] == False).all()  # noqa: E712


def test_map_peptides_empty_emits_transcript_columns(tmp_path):
    """The empty-result schema also carries the new columns (issue #141)."""
    fasta = tmp_path / "tx.fasta"
    fasta.write_text(">sp|P1|A\nACDEFGHIKL\n")
    idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    df = idx.map_peptides(["ZZZZZ"], verbose=False)
    assert "transcript_id" in df.columns
    assert "is_canonical_transcript" in df.columns


def test_from_ensembl_keeps_all_protein_coding_transcripts():
    """Issue #141: ``from_ensembl`` must NOT collapse each gene to its
    longest transcript.  We verify the new behavior by stubbing
    pyensembl with a synthetic gene that has 3 protein-coding transcripts
    with distinct sequences, then asserting the index carries all three
    proteins and tags exactly the longest as canonical.
    """
    import sys
    import types

    # Build a fake pyensembl module + EnsemblRelease that yields one gene
    # with three protein-coding transcripts.
    fake_module = types.ModuleType("pyensembl")

    class _FakeTranscript:
        def __init__(self, tid, protein_id, seq):
            self.id = tid
            self.protein_id = protein_id
            self.protein_sequence = seq
            self.biotype = "protein_coding"

    class _FakeGene:
        def __init__(self):
            self.name = "FAKEGENE"
            self.id = "ENSG00000FAKE"
            self.biotype = "protein_coding"
            self.transcripts = [
                _FakeTranscript("ENST_T1", "ENSP_T1", "ACDEFGHIKL"),  # 10aa
                _FakeTranscript("ENST_T2", "ENSP_T2", "MNPQRSTVWYACD"),  # 13aa (longest)
                _FakeTranscript("ENST_T3", "ENSP_T3", "MMMMMMM"),  # 7aa
            ]

    class _FakeEnsembl:
        def __init__(self, *args, **kwargs):
            pass

        def genes(self):
            return [_FakeGene()]

    fake_module.EnsemblRelease = _FakeEnsembl
    sys.modules["pyensembl"] = fake_module
    try:
        idx = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)

        assert set(idx.proteins.keys()) == {"ENSP_T1", "ENSP_T2", "ENSP_T3"}, (
            "expected one entry per protein-coding transcript, not just the longest"
        )
        # The longest (T2 / 13aa) is canonical; the others are not.
        assert idx.protein_meta["ENSP_T2"]["is_canonical_transcript"] is True
        assert idx.protein_meta["ENSP_T1"]["is_canonical_transcript"] is False
        assert idx.protein_meta["ENSP_T3"]["is_canonical_transcript"] is False
        # transcript_id flows through as the ENST.
        assert idx.protein_meta["ENSP_T1"]["transcript_id"] == "ENST_T1"
        assert idx.protein_meta["ENSP_T2"]["transcript_id"] == "ENST_T2"
        # gene_name / gene_id propagate from the gene record.
        assert idx.protein_meta["ENSP_T1"]["gene_name"] == "FAKEGENE"
        assert idx.protein_meta["ENSP_T1"]["gene_id"] == "ENSG00000FAKE"
    finally:
        del sys.modules["pyensembl"]


def test_from_ensembl_falls_back_to_transcript_id_when_protein_id_missing():
    """Older pyensembl releases or some species don't expose ``t.protein_id``.
    The index must still build by falling back to the transcript ID, with
    transcript_id set correctly so downstream code can still distinguish
    the transcript even when protein_id collides with it.
    """
    import sys
    import types

    fake_module = types.ModuleType("pyensembl")

    class _FakeTranscriptNoProteinId:
        def __init__(self, tid, seq):
            self.id = tid
            # protein_id intentionally absent
            self.protein_sequence = seq
            self.biotype = "protein_coding"

    class _FakeGene:
        def __init__(self):
            self.name = "GENB"
            self.id = "ENSG00000B"
            self.biotype = "protein_coding"
            self.transcripts = [_FakeTranscriptNoProteinId("ENST_B1", "AAAAAAAA")]

    class _FakeEnsembl:
        def __init__(self, *args, **kwargs):
            pass

        def genes(self):
            return [_FakeGene()]

    fake_module.EnsemblRelease = _FakeEnsembl
    sys.modules["pyensembl"] = fake_module
    try:
        idx = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
        assert "ENST_B1" in idx.proteins
        assert idx.protein_meta["ENST_B1"]["transcript_id"] == "ENST_B1"
        assert idx.protein_meta["ENST_B1"]["is_canonical_transcript"] is True
    finally:
        del sys.modules["pyensembl"]


def test_set_fasta_index_cache_maxsize_shrinks_existing_cache(tmp_path):
    """Calling set_fasta_index_cache_maxsize with a smaller bound evicts
    existing entries down to the new bound (so tests / scripts that
    discover memory pressure mid-run can shed cached proteomes without
    wiping the whole cache).
    """
    from hitlist.proteome import (
        _FASTA_INDEX_CACHE,
        clear_fasta_index_cache,
        set_fasta_index_cache_maxsize,
    )

    clear_fasta_index_cache()
    set_fasta_index_cache_maxsize(4)
    try:
        for i in range(4):
            f = tmp_path / f"shrink_{i}.fasta"
            f.write_text(f">sp|P{i}|A\nACDEFGHIKLMN\n")
            ProteomeIndex.from_fasta(f, lengths=(5,), verbose=False)
        assert len(_FASTA_INDEX_CACHE) == 4

        set_fasta_index_cache_maxsize(2)
        assert len(_FASTA_INDEX_CACHE) == 2
    finally:
        set_fasta_index_cache_maxsize(4)
        clear_fasta_index_cache()


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


# ── On-disk persistent cache (#246) ───────────────────────────────────────


@pytest.fixture
def isolated_disk_cache(tmp_path, monkeypatch):
    """Point the proteome-index disk cache at tmp_path for the duration
    of the test, and reset to default afterward.

    Also clears the in-memory cache so tests assert disk-cache behavior
    rather than getting in-memory hits.
    """
    from hitlist.proteome import (
        clear_disk_cache,
        clear_fasta_index_cache,
        set_disk_cache_dir,
    )

    set_disk_cache_dir(tmp_path / "proteome_idx_cache")
    clear_fasta_index_cache()
    # Make sure the default cap is sane for tests that don't set their own.
    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", "1")
    yield
    clear_disk_cache()
    set_disk_cache_dir(None)
    clear_fasta_index_cache()


def test_disk_cache_persists_index_across_in_memory_eviction(tmp_path, isolated_disk_cache):
    """Build the index once, drop the in-memory cache, build again — the
    second build must hit the disk cache instead of re-running _build.
    """
    from unittest.mock import patch

    from hitlist.proteome import clear_fasta_index_cache

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQRSTVWY\n")

    # First call: cold build, writes to disk cache.
    idx1 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert len(idx1.proteins) == 1

    # Drop the in-memory cache so the next call can't hit it.
    clear_fasta_index_cache()

    # Second call: must NOT call _build — disk cache should serve it.
    real_build = ProteomeIndex._build
    with patch.object(ProteomeIndex, "_build", side_effect=AssertionError("_build called")):
        idx2 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)

    # Same content, different instance (pickle round-trip).
    assert idx1.proteins == idx2.proteins
    assert idx1 is not idx2

    # Sanity: confirm we restored _build.
    assert ProteomeIndex._build is real_build or callable(ProteomeIndex._build)


def test_disk_cache_invalidates_on_fasta_mtime_change(tmp_path, isolated_disk_cache):
    """Editing the FASTA bumps mtime → cache key changes → fresh build.

    The stale on-disk pickle remains until cap-eviction collects it,
    which is fine — it just won't be loaded for any future key.
    """
    import time

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")
    idx1 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert "sp|P|A" in idx1.proteins

    time.sleep(0.01)
    fasta.write_text(">sp|Q|B\nMNPQRSTVWY\n")
    import os

    os.utime(fasta, None)

    idx2 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    # Different FASTA → different content; the new build should reflect
    # the new content, not return the stale cached pickle.
    assert "sp|Q|B" in idx2.proteins
    assert "sp|P|A" not in idx2.proteins


def test_disk_cache_keyed_on_lengths(tmp_path, isolated_disk_cache):
    """Different `lengths` → different cache filename → independent entries."""
    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")

    idx_5 = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    idx_8 = ProteomeIndex.from_fasta(fasta, lengths=(8,), verbose=False)
    assert idx_5.lengths == (5,)
    assert idx_8.lengths == (8,)

    files = sorted(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    # Two distinct cache files — one per length tuple.
    assert len(files) == 2


def test_disk_cache_format_version_in_filename(tmp_path, isolated_disk_cache):
    """Cache filenames carry the format-version prefix so a future
    schema bump doesn't load stale-format pickles silently.
    """
    from hitlist.proteome import _INDEX_FORMAT_VERSION, _PROTEOME_INDEX_DISK_CACHE_DIR

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")
    ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)

    files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    assert len(files) == 1
    assert files[0].name.startswith(f"v{_INDEX_FORMAT_VERSION}_")


def test_disk_cache_corrupt_pickle_falls_back_to_rebuild(tmp_path, isolated_disk_cache):
    """A truncated / garbage cache file must NOT crash — the loader
    drops it and the rebuild path takes over (then writes a fresh
    pickle).
    """
    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR, clear_fasta_index_cache

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")

    # Cold build to populate the cache.
    ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    cache_files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"\x00\x01\x02not-a-valid-pickle")

    clear_fasta_index_cache()

    # Should rebuild without raising.
    idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert "sp|P|A" in idx.proteins

    # Corrupt file was unlinked + replaced with a fresh, valid pickle.
    cache_files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    assert len(cache_files) == 1
    # And it loads cleanly now.
    import pickle as _pkl

    with open(cache_files[0], "rb") as f:
        loaded = _pkl.load(f)
    assert "sp|P|A" in loaded.proteins


def test_disk_cache_eviction_under_cap(tmp_path, isolated_disk_cache, monkeypatch):
    """Total cache size > cap → oldest-mtime files are evicted on next write."""
    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR, clear_fasta_index_cache

    # Tiny cap so the second write triggers eviction.
    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", str(1 / 1024 / 1024))  # 1 MB cap

    # Write a series of distinct FASTAs so each produces its own cache file.
    fastas = []
    for i in range(4):
        f = tmp_path / f"f_{i}.fasta"
        f.write_text(f">sp|P{i:05d}|A\n{'ACDEFGHIKLMNPQRSTVWY' * 200}\n")
        fastas.append(f)
        clear_fasta_index_cache()
        ProteomeIndex.from_fasta(f, lengths=(5,), verbose=False)

    cache_files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    total = sum(p.stat().st_size for p in cache_files)
    cap_bytes = int((1 / 1024 / 1024) * 1024**3)
    # Eviction may leave us slightly over cap if a single index exceeds
    # the cap on its own (we never delete the file we just wrote since
    # the eviction loop touches oldest-first and stops once under cap).
    # The strict invariant: total ≤ cap + size of newest entry.
    if cache_files:
        newest = max(cache_files, key=lambda p: p.stat().st_mtime)
        assert total <= cap_bytes + newest.stat().st_size


def test_disk_cache_disabled_when_cap_zero(tmp_path, isolated_disk_cache, monkeypatch):
    """Setting the cap to 0 GB skips both reads and writes."""
    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR

    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", "0")

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")
    ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)

    # No cache files written.
    assert (
        not list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
        or not _PROTEOME_INDEX_DISK_CACHE_DIR.exists()
    )


def test_disk_cache_promotes_into_in_memory_cache(tmp_path, isolated_disk_cache):
    """A disk-cache hit should populate the in-memory cache so subsequent
    same-process calls hit the fast path."""
    from hitlist.proteome import _FASTA_INDEX_CACHE, clear_fasta_index_cache

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")

    # Cold build + in-memory eviction.
    ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    clear_fasta_index_cache()
    assert len(_FASTA_INDEX_CACHE) == 0

    # Disk hit re-populates in-memory cache.
    idx_disk = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert len(_FASTA_INDEX_CACHE) == 1

    # And subsequent call returns the same instance.
    idx_mem = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert idx_mem is idx_disk


def test_disk_cache_write_failure_does_not_crash_build(tmp_path, isolated_disk_cache, monkeypatch):
    """If the disk write fails (e.g. pickle.dump raises), the build
    must complete successfully and the failure is surfaced via
    ``warnings.warn`` — never crash the caller.

    Catches the ``tmp_path`` NameError that previously hid behind a
    `# pragma: no cover` in the except clause.
    """
    import pickle as _pkl
    import warnings as _w

    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")

    # Force the pickle.dump inside _write_index_to_disk to raise.
    def _boom(*a, **kw):
        raise OSError("simulated disk-full")

    monkeypatch.setattr("hitlist.proteome.pickle.dump", _boom)

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        # Build must succeed despite the cache-write failure.
        idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert "sp|P|A" in idx.proteins
    assert any("Failed to write proteome index cache" in str(w.message) for w in caught)

    # No half-written .tmp left behind in the cache dir.
    if _PROTEOME_INDEX_DISK_CACHE_DIR.exists():
        leftovers = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.tmp"))
        assert not leftovers, f"expected no .tmp leftovers, got {leftovers}"

    # Sanity: the genuine pickle path still works after we restore it.
    monkeypatch.undo()
    # Subsequent build should write a real cache file.
    from hitlist.proteome import clear_fasta_index_cache

    clear_fasta_index_cache()
    ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    assert len(files) == 1
    with open(files[0], "rb") as f:
        loaded = _pkl.load(f)
    assert "sp|P|A" in loaded.proteins


def test_disk_cache_write_failure_before_tmp_open_does_not_crash(tmp_path, monkeypatch):
    """The failure path must also handle the case where the exception
    fires BEFORE ``tempfile.NamedTemporaryFile`` ever opens — i.e.
    ``tmp_path`` was never assigned.  Previously this NameError'd in
    the except clause.
    """
    import warnings as _w

    from hitlist.proteome import (
        clear_disk_cache,
        clear_fasta_index_cache,
        set_disk_cache_dir,
    )

    set_disk_cache_dir(tmp_path / "idx_cache")
    clear_fasta_index_cache()
    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", "1")
    try:
        # Make _PROTEOME_INDEX_DISK_CACHE_DIR.mkdir() raise — fires
        # before tmp_path is ever bound in _write_index_to_disk.
        from pathlib import Path as _Path

        original_mkdir = _Path.mkdir

        def _mkdir_boom(self, *a, **kw):
            if "idx_cache" in str(self):
                raise PermissionError("simulated denied mkdir")
            return original_mkdir(self, *a, **kw)

        monkeypatch.setattr(_Path, "mkdir", _mkdir_boom)

        fasta = tmp_path / "p.fasta"
        fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQ\n")

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            # Must not raise NameError or PermissionError.
            idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
        assert "sp|P|A" in idx.proteins
        assert any("Failed to write proteome index cache" in str(w.message) for w in caught)
    finally:
        clear_disk_cache()
        set_disk_cache_dir(None)
        clear_fasta_index_cache()


def test_disk_cache_concurrent_writes_are_safe(tmp_path, isolated_disk_cache):
    """Two processes that race to build the same FASTA must not corrupt
    each other's cache file.  The atomic ``.tmp + os.replace`` pattern
    guarantees the final ``*.pkl`` is always a complete, valid pickle.

    Uses multiprocessing (forked workers each rebuild + write) and
    asserts post-hoc that the cache file loads cleanly and matches
    in-process content.
    """
    import multiprocessing as mp

    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR, set_disk_cache_dir

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P|A\nACDEFGHIKLMNPQRSTVWY\n")

    # The fork target needs its own setup since process state isn't
    # inherited cleanly under spawn (the default on macOS).
    cache_dir = str(_PROTEOME_INDEX_DISK_CACHE_DIR)

    def _worker(fasta_str, cache_dir_str, return_dict, idx):
        from hitlist.proteome import (
            ProteomeIndex,
            clear_fasta_index_cache,
            set_disk_cache_dir,
        )

        set_disk_cache_dir(cache_dir_str)
        clear_fasta_index_cache()
        idx_obj = ProteomeIndex.from_fasta(fasta_str, lengths=(5,), verbose=False)
        return_dict[idx] = len(idx_obj.proteins)

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    manager = ctx.Manager()
    return_dict = manager.dict()
    procs = [
        ctx.Process(target=_worker, args=(str(fasta), cache_dir, return_dict, i)) for i in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker {p.pid} exited {p.exitcode}"

    # All workers built the same content.
    assert dict(return_dict) == {0: 1, 1: 1, 2: 1, 3: 1}

    # Exactly one cache file (atomic rename means the final entry is
    # whichever winner finished last; previous writes' .tmp files don't
    # survive past their rename).
    cache_files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.pkl"))
    assert len(cache_files) == 1, f"expected 1 cache file, got {cache_files}"

    # And it loads cleanly (no half-written corruption).
    set_disk_cache_dir(cache_dir)
    from hitlist.proteome import clear_fasta_index_cache

    clear_fasta_index_cache()
    idx = ProteomeIndex.from_fasta(fasta, lengths=(5,), verbose=False)
    assert "sp|P|A" in idx.proteins

    # No leftover .tmp files (each worker's tempfile was atomically renamed).
    leftovers = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("*.tmp"))
    assert not leftovers, f"expected no .tmp leftovers, got {leftovers}"


def test_clear_disk_cache_removes_tmp_files_too(tmp_path, isolated_disk_cache):
    """``clear_disk_cache`` should reap leftover ``.tmp`` partial-write
    files alongside the ``.pkl`` cache files.
    """
    from hitlist.proteome import (
        _PROTEOME_INDEX_DISK_CACHE_DIR,
        clear_disk_cache,
    )

    _PROTEOME_INDEX_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl = _PROTEOME_INDEX_DISK_CACHE_DIR / "v1_x_L5_deadbeef.pkl"
    tmp = _PROTEOME_INDEX_DISK_CACHE_DIR / "v1_x_L5_deadbeef.pkl.tmpABC.tmp"
    pkl.write_bytes(b"not actually a pickle")
    tmp.write_bytes(b"half-written garbage")
    assert pkl.exists() and tmp.exists()

    clear_disk_cache()
    assert not pkl.exists()
    assert not tmp.exists()


# ── from_ensembl on-disk cache (#251) ─────────────────────────────────────


def _install_fake_ensembl(tmp_path, monkeypatch, gtf_size_marker: bytes = b"v1"):
    """Install a synthetic ``pyensembl`` module + matching GTF file.

    The fake ``EnsemblRelease.gtf_path`` returns a real on-disk file in
    ``tmp_path`` so the cache invalidator (size + mtime) works.  Tweak
    ``gtf_size_marker`` between calls to simulate a GTF re-download.
    """
    import sys
    import types

    gtf = tmp_path / "fake.gtf"
    gtf.write_bytes(gtf_size_marker)

    fake_module = types.ModuleType("pyensembl")

    class _FakeTranscript:
        def __init__(self, tid, protein_id, seq):
            self.id = tid
            self.protein_id = protein_id
            self.protein_sequence = seq
            self.biotype = "protein_coding"

    class _FakeGene:
        def __init__(self):
            self.name = "FAKEGENE"
            self.id = "ENSG00000FAKE"
            self.biotype = "protein_coding"
            self.transcripts = [
                _FakeTranscript("ENST_T1", "ENSP_T1", "ACDEFGHIKLMNPQ"),
            ]

    class _FakeEnsembl:
        def __init__(self, *args, **kwargs):
            self.gtf_path = str(gtf)

        def genes(self):
            return [_FakeGene()]

    fake_module.EnsemblRelease = _FakeEnsembl
    monkeypatch.setitem(sys.modules, "pyensembl", fake_module)
    return gtf


def test_from_ensembl_disk_cache_persists_across_in_memory_eviction(
    tmp_path, isolated_disk_cache, monkeypatch
):
    """Build via from_ensembl, drop the in-memory cache, build again —
    the second build must hit the on-disk cache instead of re-running
    ``_build`` (or even re-iterating the pyensembl genes list).
    """
    from unittest.mock import patch

    from hitlist.proteome import clear_fasta_index_cache

    _install_fake_ensembl(tmp_path, monkeypatch)

    idx1 = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    assert "ENSP_T1" in idx1.proteins

    clear_fasta_index_cache()

    # Second call must NOT call _build.  If pyensembl's genes() is called
    # again that's also wasted work but tolerable; the critical assertion
    # is that the expensive k-mer pass is skipped.
    with patch.object(ProteomeIndex, "_build", side_effect=AssertionError("_build called")):
        idx2 = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)

    assert idx1.proteins == idx2.proteins
    assert idx1.protein_meta == idx2.protein_meta
    # Pickle round-trip → different instance.
    assert idx1 is not idx2


def test_from_ensembl_cache_invalidates_on_gtf_change(tmp_path, isolated_disk_cache, monkeypatch):
    """Editing the local GTF (size or mtime change) must produce a
    different cache key → fresh build with the new gene set.
    """
    import time

    from hitlist.proteome import clear_fasta_index_cache

    gtf = _install_fake_ensembl(tmp_path, monkeypatch, gtf_size_marker=b"v1")
    idx1 = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    assert "ENSP_T1" in idx1.proteins
    n_before = len(idx1.proteins)

    # Simulate a GTF re-download: bump size + mtime.
    time.sleep(0.01)
    gtf.write_bytes(b"v2_with_more_bytes" * 8)
    import os as _os

    _os.utime(gtf, None)

    clear_fasta_index_cache()
    # Same fake genes list, but different cache key → fresh _build.
    idx2 = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    assert "ENSP_T1" in idx2.proteins
    assert len(idx2.proteins) == n_before
    # Distinct cache files on disk (one per (release, gtf-fingerprint)).
    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR

    files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("v*ensembl*.pkl"))
    assert len(files) == 2, f"expected 2 cache files (one per GTF version), got {files}"


def test_from_ensembl_cache_keyed_on_release_and_species(
    tmp_path, isolated_disk_cache, monkeypatch
):
    """Different (release, species, biotype, lengths) → independent cache
    files."""
    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR, clear_fasta_index_cache

    _install_fake_ensembl(tmp_path, monkeypatch)

    ProteomeIndex.from_ensembl(release=111, species="human", lengths=(5,), verbose=False)
    clear_fasta_index_cache()
    ProteomeIndex.from_ensembl(release=112, species="human", lengths=(5,), verbose=False)
    clear_fasta_index_cache()
    ProteomeIndex.from_ensembl(release=112, species="mouse", lengths=(5,), verbose=False)
    clear_fasta_index_cache()
    ProteomeIndex.from_ensembl(release=112, species="human", lengths=(8,), verbose=False)

    files = sorted(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("v*ensembl*.pkl"))
    # 4 distinct keys → 4 distinct files.
    assert len(files) == 4, f"expected 4 cache files, got {[f.name for f in files]}"


def test_from_ensembl_cache_disabled_when_gtf_unresolvable(
    tmp_path, isolated_disk_cache, monkeypatch
):
    """When pyensembl's release can't expose a usable GTF path (synthetic
    test fakes, missing local data), caching is silently skipped and the
    cold path runs as before.  Backwards-compatible: the existing
    ``test_from_ensembl_*`` tests with no gtf_path attribute keep working.
    """
    import sys
    import types

    from hitlist.proteome import _PROTEOME_INDEX_DISK_CACHE_DIR

    fake_module = types.ModuleType("pyensembl")

    class _FakeTranscriptNoProteinId:
        def __init__(self, tid, seq):
            self.id = tid
            self.protein_sequence = seq
            self.biotype = "protein_coding"

    class _FakeGene:
        def __init__(self):
            self.name = "G"
            self.id = "ENSG00000X"
            self.biotype = "protein_coding"
            self.transcripts = [_FakeTranscriptNoProteinId("ENST_X1", "AAAAAAAA")]

    class _FakeEnsembl:
        # No gtf_path attribute at all.
        def __init__(self, *args, **kwargs):
            pass

        def genes(self):
            return [_FakeGene()]

    fake_module.EnsemblRelease = _FakeEnsembl
    monkeypatch.setitem(sys.modules, "pyensembl", fake_module)

    idx = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    assert "ENST_X1" in idx.proteins
    # No cache files written when caching is disabled.
    files = list(_PROTEOME_INDEX_DISK_CACHE_DIR.glob("v*ensembl*.pkl"))
    assert files == [], f"expected no cache files when gtf_path is unresolvable, got {files}"


def test_from_ensembl_disk_hit_promotes_into_in_memory_cache(
    tmp_path, isolated_disk_cache, monkeypatch
):
    """Same promotion semantics as from_fasta: on a disk-cache hit, the
    in-memory cache gets populated so subsequent calls hit the fast
    path."""
    from hitlist.proteome import _FASTA_INDEX_CACHE, clear_fasta_index_cache

    _install_fake_ensembl(tmp_path, monkeypatch)

    ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    clear_fasta_index_cache()
    assert len(_FASTA_INDEX_CACHE) == 0

    # Disk-cache hit populates in-memory cache.
    idx_disk = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    assert len(_FASTA_INDEX_CACHE) == 1

    # Subsequent call returns the same instance.
    idx_mem = ProteomeIndex.from_ensembl(release=999, lengths=(5,), verbose=False)
    assert idx_mem is idx_disk
