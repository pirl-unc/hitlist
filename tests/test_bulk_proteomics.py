from hitlist.bulk_proteomics import (
    available_cell_lines,
    available_peptide_cell_lines,
    available_protein_cell_lines,
    load_bulk_peptides,
    load_bulk_proteomics,
)


def test_available_cell_lines():
    cells = available_cell_lines()
    # Union across both indices
    assert "MDA-MB-231" in cells
    assert "HCT116" in cells
    assert "A549" in cells
    assert "K562" in cells
    assert "THP-1" in cells
    assert "MCF7" in cells
    assert "Jurkat" in cells
    # Peptide-only lines show up in the union
    assert "HeLa" in cells
    assert "HEK293" in cells


def test_available_protein_cell_lines():
    cells = available_protein_cell_lines()
    assert "MDA-MB-231" in cells
    assert "HCT116" in cells
    # HeLa and HEK293 are NOT in CCLE
    assert "HeLa" not in cells
    assert "HEK293" not in cells


def test_available_peptide_cell_lines():
    cells = available_peptide_cell_lines()
    assert set(cells) == {"A549", "HCT116", "HEK293", "HeLa", "MCF7"}


def test_load_bulk_proteomics_full():
    df = load_bulk_proteomics()
    assert len(df) > 50_000, f"expected >50K rows across 7 cell lines, got {len(df)}"
    expected_cols = {
        "cell_line",
        "gene_symbol",
        "uniprot_acc",
        "protein_id",
        "abundance_log2_normalized",
        "source",
        "reference",
    }
    assert expected_cols.issubset(df.columns)
    assert set(df["source"]) == {"CCLE_Nusinow_2020"}


def test_load_bulk_proteomics_filter_cell_line():
    df = load_bulk_proteomics(cell_line="MDA-MB-231")
    assert len(df) > 5_000
    assert set(df["cell_line"]) == {"MDA-MB-231"}
    # Case-insensitive match
    df2 = load_bulk_proteomics(cell_line="mda-mb-231")
    assert len(df2) == len(df)


def test_load_bulk_proteomics_filter_gene():
    df = load_bulk_proteomics(gene_name="TP53")
    assert len(df) >= 5, f"TP53 should be detected in most cell lines, got {len(df)}"
    assert set(df["gene_symbol"]) == {"TP53"}


def test_load_bulk_proteomics_combined_filter():
    df = load_bulk_proteomics(cell_line="HCT116", gene_name=["KRAS", "TP53"])
    assert set(df["cell_line"]) == {"HCT116"}
    assert set(df["gene_symbol"]).issubset({"KRAS", "TP53"})


def test_load_bulk_peptides_full():
    df = load_bulk_peptides()
    assert len(df) > 500_000, f"expected >500K peptide rows across 5 cell lines, got {len(df)}"
    expected_cols = {
        "peptide",
        "cell_line",
        "uniprot_acc",
        "gene_symbol",
        "length",
        "start_position",
        "end_position",
        "source",
        "reference",
    }
    assert expected_cols.issubset(df.columns)
    assert set(df["source"]) == {"Bekker-Jensen_2017"}


def test_load_bulk_peptides_filter_cell_line():
    df = load_bulk_peptides(cell_line="HeLa")
    assert len(df) > 100_000
    assert set(df["cell_line"]) == {"HeLa"}
    # Case-insensitive
    assert len(load_bulk_peptides(cell_line="hela")) == len(df)


def test_load_bulk_peptides_filter_gene():
    df = load_bulk_peptides(gene_name="TP53")
    assert len(df) > 10, f"TP53 should have peptides across ≥3 cell lines, got {len(df)}"
    assert set(df["gene_symbol"]) == {"TP53"}


def test_load_bulk_peptides_intra_protein_bias():
    """Core use case: within-protein peptide detectability differs by cell line."""
    df = load_bulk_peptides(gene_name="TP53")
    per_line = df.groupby("cell_line").size()
    # Different cell lines detect different numbers of TP53 peptides
    assert per_line.max() > per_line.min()
    # Positions are populated — needed for intra-protein coverage analysis
    assert df["start_position"].notna().all()
    assert df["end_position"].notna().all()
    assert (df["end_position"] >= df["start_position"]).all()


def test_load_bulk_peptides_filter_uniprot():
    # TP53 → P04637
    df = load_bulk_peptides(uniprot_acc="P04637")
    assert len(df) > 0
    assert set(df["uniprot_acc"]) == {"P04637"}
