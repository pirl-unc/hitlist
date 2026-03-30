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
