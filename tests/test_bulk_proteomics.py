from hitlist.bulk_proteomics import (
    available_cell_lines,
    available_peptide_cell_lines,
    available_protein_cell_lines,
    is_bulk_proteomics_built,
    load_bulk_peptides,
    load_bulk_proteomics,
    load_bulk_sources,
)

# Harmonized acquisition metadata columns — same names appear in the
# ms_samples schema emitted by hitlist.export for observations.parquet,
# so downstream MS-bias code can read both indexes with one column list.
_HARMONIZED_ACQUISITION_COLS = {
    "pmid",
    "reference",
    "study_label",
    "species",
    "cell_line_name",
    "sample_label",
    "instrument",
    "instrument_type",
    "fragmentation",
    "acquisition_mode",
    "labeling",
    "search_engine",
    "fdr",
}

_BULK_PREP_COLS = {
    "digestion",
    "digestion_enzyme",
    "fractionation",
    "n_fractions",
    "quantification",
}


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
    # HeLa and HEK293 are now covered via Bekker-Jensen-derived abundance
    assert "HeLa" in cells
    assert "HEK293" in cells


def test_available_peptide_cell_lines():
    cells = available_peptide_cell_lines()
    assert set(cells) == {"A549", "HCT116", "HEK293", "HeLa", "MCF7"}


def test_load_bulk_proteomics_full():
    df = load_bulk_proteomics()
    assert len(df) > 50_000, f"expected >50K rows across 7 cell lines, got {len(df)}"
    expected_cols = {
        "cell_line_name",
        "gene_symbol",
        "uniprot_acc",
        "source",
        "reference",
        "abundance_percentile",
    }
    assert expected_cols.issubset(df.columns)
    # Default load is the UNION of CCLE + Bekker-Jensen protein-level
    assert set(df["source"]) == {"CCLE_Nusinow_2020", "Bekker-Jensen_2017"}


def test_load_bulk_proteomics_filter_cell_line():
    df = load_bulk_proteomics(cell_line="MDA-MB-231")
    assert len(df) > 5_000
    assert set(df["cell_line_name"]) == {"MDA-MB-231"}
    # Case-insensitive match
    df2 = load_bulk_proteomics(cell_line="mda-mb-231")
    assert len(df2) == len(df)


def test_load_bulk_proteomics_filter_gene():
    df = load_bulk_proteomics(gene_name="TP53")
    assert len(df) >= 5, f"TP53 should be detected in most cell lines, got {len(df)}"
    assert set(df["gene_symbol"]) == {"TP53"}


def test_load_bulk_proteomics_combined_filter():
    df = load_bulk_proteomics(cell_line="HCT116", gene_name=["KRAS", "TP53"])
    assert set(df["cell_line_name"]) == {"HCT116"}
    assert set(df["gene_symbol"]).issubset({"KRAS", "TP53"})


def test_load_bulk_peptides_full():
    df = load_bulk_peptides()
    assert len(df) > 500_000, f"expected >500K peptide rows across 5 cell lines, got {len(df)}"
    expected_cols = {
        "peptide",
        "cell_line_name",
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
    assert set(df["cell_line_name"]) == {"HeLa"}
    # Case-insensitive
    assert len(load_bulk_peptides(cell_line="hela")) == len(df)


def test_load_bulk_peptides_filter_gene():
    df = load_bulk_peptides(gene_name="TP53")
    assert len(df) > 10, f"TP53 should have peptides across ≥3 cell lines, got {len(df)}"
    assert set(df["gene_symbol"]) == {"TP53"}


def test_load_bulk_peptides_intra_protein_bias():
    """Core use case: within-protein peptide detectability differs by cell line."""
    df = load_bulk_peptides(gene_name="TP53")
    per_line = df.groupby("cell_line_name").size()
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


def test_load_bulk_proteomics_hela_via_bekker_jensen():
    """HeLa abundance is derived from Bekker-Jensen peptide intensities."""
    df = load_bulk_proteomics(cell_line="HeLa")
    assert len(df) > 10_000
    assert set(df["source"]) == {"Bekker-Jensen_2017"}
    # abundance_percentile is populated
    assert df["abundance_percentile"].between(0, 1).all()


def test_load_bulk_proteomics_source_filter():
    ccle = load_bulk_proteomics(cell_line="A549", source="CCLE_Nusinow_2020")
    bj = load_bulk_proteomics(cell_line="A549", source="Bekker-Jensen_2017")
    assert set(ccle["source"]) == {"CCLE_Nusinow_2020"}
    assert set(bj["source"]) == {"Bekker-Jensen_2017"}
    # Both sources independently cover A549
    assert len(ccle) > 5_000
    assert len(bj) > 10_000


def test_load_bulk_sources_shape():
    srcs = load_bulk_sources()
    assert len(srcs) == 2
    ids = {s["source_id"] for s in srcs}
    assert ids == {"CCLE_Nusinow_2020", "Bekker-Jensen_2017"}
    # Every source must declare digestion + instrument + granularity
    for s in srcs:
        assert s["digestion"]
        assert s["instrument"]
        assert s["granularity"] in {"peptide", "protein", "peptide_and_protein"}
        assert isinstance(s["cell_lines_covered"], list)
        assert len(s["cell_lines_covered"]) > 0


def test_load_bulk_sources_digest_is_tryptic():
    """Both current sources use tryptic digest — sanity check for downstream assumptions."""
    srcs = load_bulk_sources()
    for s in srcs:
        assert s["digestion"] == "tryptic"


def test_load_bulk_sources_harmonized_fields():
    """Sources expose the harmonized acquisition fields (matching ms_samples schema)."""
    srcs = load_bulk_sources()
    for s in srcs:
        assert s.get("pmid") and isinstance(s["pmid"], int)
        assert s.get("study_label")
        assert s.get("species")
        assert s.get("fragmentation")
        assert s.get("acquisition_mode")
        assert s.get("labeling")
        assert isinstance(s.get("n_fractions"), int)


def test_build_bulk_proteomics_parquet():
    """build_bulk_proteomics writes a unified long-form parquet."""
    import pandas as pd

    from hitlist.builder import build_bulk_proteomics

    df = build_bulk_proteomics(verbose=False)
    assert is_bulk_proteomics_built()
    assert len(df) > 1_000_000
    # Both granularities present
    assert set(df["granularity"]) == {"protein", "peptide"}
    # Harmonized acquisition columns populated on every row
    missing = _HARMONIZED_ACQUISITION_COLS - set(df.columns)
    assert not missing, f"missing harmonized columns: {missing}"
    bulk_missing = _BULK_PREP_COLS - set(df.columns)
    assert not bulk_missing, f"missing bulk-prep columns: {bulk_missing}"
    # Sanity: all rows have populated instrument + digestion
    assert (df["instrument"] != "").all()
    assert (df["digestion"] == "tryptic").all()
    # CCLE rows carry TMT label, BJ rows are label-free
    ccle_rows = df[df["source"] == "CCLE_Nusinow_2020"]
    bj_rows = df[df["source"] == "Bekker-Jensen_2017"]
    assert (ccle_rows["labeling"].str.contains("TMT")).all()
    assert (bj_rows["labeling"] == "label-free").all()
    # pmid is an integer with real PMIDs
    assert set(df["pmid"].dropna().unique()) == {31978347, 28591648}
    # Evidence kind stamped so downstream can filter a unified index
    assert set(df["evidence_kind"]) == {"bulk_proteomics"}
    # Parquet round-trips to the same shape
    from hitlist.bulk_proteomics import bulk_proteomics_path

    roundtrip = pd.read_parquet(bulk_proteomics_path())
    assert roundtrip.shape == df.shape


def test_loaders_read_from_parquet_when_built():
    """Once built, loaders return rows with the harmonized metadata columns."""
    from hitlist.builder import build_bulk_proteomics

    build_bulk_proteomics(verbose=False)
    proteins = load_bulk_proteomics()
    peptides = load_bulk_peptides()
    for df in (proteins, peptides):
        # Acquisition metadata present on every row
        missing = _HARMONIZED_ACQUISITION_COLS - set(df.columns)
        assert not missing, f"missing harmonized columns: {missing}"
        assert (df["instrument"] != "").all()
        assert (df["fragmentation"] != "").all()
