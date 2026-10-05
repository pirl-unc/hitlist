"""Small, test-owned CSVs exercise bulk queries without loading the reference corpus."""

import argparse

import pandas as pd
import pytest

from hitlist import bulk_proteomics as bp
from hitlist import downloads
from hitlist.builder import build_bulk_proteomics
from hitlist.cli import _export_bulk


@pytest.fixture
def bulk_data(tmp_path, monkeypatch):
    """Use the real CSV readers and builder with six deliberately distinct arms."""
    peptides = pd.DataFrame(
        {
            "peptide": ["A", "B", "C", "D", "E", "F"],
            "cell_line": ["HeLa", "HeLa", "HeLa", "HeLa", "HeLa", "A549"],
            "gene_symbol": ["G1", "G2", "G1", "G1", "G1", "G1"],
            "uniprot_acc": ["P1", "P2", "P1", "P1", "P1", "P1"],
            "length": [8, 9, 11, 15, 12, 9],
            "start_position": [1] * 6,
            "end_position": [8, 9, 11, 15, 12, 9],
            "digestion_enzyme": ["Trypsin/P"] * 3 + ["LysC", "Trypsin/P", "Trypsin/P"],
            "n_fractions_in_run": [14, 14, 46, 39, 12, 46],
            "enrichment": ["none"] * 4 + ["TiO2", "none"],
            "fractionation_ph": [10.0] * 4 + [8.0, 10.0],
            "n_replicates_detected": [1, 3, 1, 1, 1, 1],
            "modifications": [""] * 6,
            "source": ["Bekker-Jensen_2017"] * 6,
            "reference": ["PMID:28591648"] * 6,
        }
    )
    proteins = peptides.drop_duplicates(["cell_line", "gene_symbol"]).drop(
        columns=["peptide", "length", "start_position", "end_position", "n_replicates_detected"]
    )
    proteins["abundance_percentile"] = [0.95, 0.5, 0.7]
    proteins["n_peptides"] = 2
    proteins["log2_intensity"] = 1.0
    ccle = pd.DataFrame(
        {
            "cell_line": ["A549", "A549"],
            "gene_symbol": ["G1", "G2"],
            "uniprot_acc": ["P1", "P2"],
            "abundance_log2_normalized": [1.0, 2.0],
        }
    )
    for filename, frame in (
        ("bekker_jensen_2017_peptides.csv.gz", peptides),
        ("bekker_jensen_2017_protein_abundance.csv.gz", proteins),
        ("ccle_nusinow_2020.csv.gz", ccle),
    ):
        frame.to_csv(tmp_path / filename, index=False, compression="gzip")
    monkeypatch.setattr(bp, "_bulk_data_path", lambda filename: str(tmp_path / filename))
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "built")
    caches = (bp._load_bj, bp._load_bj_protein, bp._load_ccle, bp._read_parquet_cached)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.fixture(params=["csv", "parquet"])
def query_source(bulk_data, request):
    if request.param == "parquet":
        build_bulk_proteomics()


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, ["A", "B", "C", "D", "F"]),
        ({"cell_line": "hela", "gene_name": ["G1"], "uniprot_acc": "P1"}, ["A", "C", "D"]),
        ({"digestion_enzyme": ["LysC"]}, ["D"]),
        ({"n_fractions_in_run": [14]}, ["A", "B"]),
        ({"enrichment": "TiO2", "fractionation_ph": [8]}, ["E"]),
        ({"enrichment": None}, ["A", "B", "C", "D", "E", "F"]),
        ({"length_min": 9, "length_max": 11}, ["B", "C", "F"]),
        ({"length_min": 15}, ["D"]),
        ({"length_max": 8}, ["A"]),
        # G2 supplies the arm's 3-replicate denominator even after the gene filter.
        ({"gene_name": "G1", "min_reproducibility": 1}, ["C", "D", "F"]),
        ({"cell_line": [], "enrichment": None}, []),
    ],
)
def test_peptide_filters_on_both_sources(query_source, filters, expected):
    assert bp.load_bulk_peptides(**filters)["peptide"].tolist() == expected


def test_cell_line_lists_use_test_owned_sources(bulk_data):
    assert bp.available_cell_lines() == ["A549", "HeLa"]
    assert bp.available_peptide_cell_lines() == ["A549", "HeLa"]
    assert bp.available_protein_cell_lines() == ["A549", "HeLa"]


def test_protein_percentiles_and_source_filters(query_source):
    result = bp.load_bulk_proteomics(
        cell_line="a549", source="CCLE_Nusinow_2020", abundance_percentile_min=0.75
    )
    assert result["gene_symbol"].tolist() == ["G2"]
    window = bp.load_bulk_proteomics(
        cell_line="HeLa", abundance_percentile_min=0.4, abundance_percentile_max=0.6
    )
    assert window["gene_symbol"].tolist() == ["G2"]


def test_query_results_do_not_mutate_cached_sources(query_source):
    expected = bp.load_bulk_peptides()
    changed = bp.load_bulk_peptides()
    changed.loc[0, "gene_symbol"] = "CHANGED"
    changed.loc[0, "n_replicates_possible"] = 99
    pd.testing.assert_frame_equal(bp.load_bulk_peptides(), expected)


@pytest.mark.parametrize("granularity", ["peptide", "protein", "both"])
def test_cli_filters_and_granularity(query_source, granularity):
    args = argparse.Namespace(
        granularity=granularity,
        cell_line=["HeLa"],
        gene_name=["G1"],
        uniprot_acc=None,
        source=None,
        digestion_enzyme=None,
        n_fractions=None,
        enrichment="both",
        fractionation_ph=None,
        length_min=9,
        length_max=12,
        abundance_percentile_min=0.9,
        abundance_percentile_max=None,
    )
    result = _export_bulk(args)
    assert set(result["cell_line_name"]) == {"HeLa"}
    assert set(result["gene_symbol"]) == {"G1"}
    expected = {"peptide", "protein"} if granularity == "both" else {granularity}
    assert set(result["granularity"]) == expected
    peptide = result[result["granularity"] == "peptide"]
    if len(peptide):
        assert peptide["peptide"].tolist() == ["C", "E"]
