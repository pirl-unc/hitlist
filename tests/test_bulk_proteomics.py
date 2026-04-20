from hitlist.bulk_proteomics import (
    available_cell_lines,
    load_bulk_proteomics,
)


def test_available_cell_lines():
    cells = available_cell_lines()
    assert "MDA-MB-231" in cells
    assert "HCT116" in cells
    assert "A549" in cells
    assert "K562" in cells
    assert "THP-1" in cells
    assert "MCF7" in cells
    assert "Jurkat" in cells


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
