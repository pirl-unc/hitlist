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

"""Tests for hitlist.pmhc_query — protein x allele MS evidence lookup."""

from __future__ import annotations

import pandas as pd


def _write_obs_fixture(tmp_path):
    """Tiny observations.parquet fixture covering:
    - one peptide that multi-maps to NRAS+KRAS (semicolon-joined gene_names)
    - one peptide unique to NRAS
    - one peptide on a different allele
    - one peptide on a different gene (BRAF)
    """
    df = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV", "KLVVVGAGGV", "ILDTAGREEY", "ALAVLGFFV", "FLPNKQRTV"],
            "pmid": [9263005, 36099883, 33858848, 38480730, 33298915],
            "mhc_class": ["I"] * 5,
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA-A*02:01",
            ],
            "gene_names": ["NRAS;KRAS", "NRAS;KRAS", "NRAS", "NRAS", "BRAF"],
            "gene_ids": [
                "ENSG00000213281;ENSG00000133703",
                "ENSG00000213281;ENSG00000133703",
                "ENSG00000213281",
                "ENSG00000213281",
                "ENSG00000157764",
            ],
            "species": ["Homo sapiens"] * 5,
            "mhc_species": ["Homo sapiens"] * 5,
            "source": ["iedb"] * 5,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    return path


def test_pmhc_query_groups_by_gene_and_allele(tmp_path, monkeypatch):
    """Single protein, two alleles: rows group by (gene, allele) and
    aggregate observation counts + PMID lists per peptide."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A*02:01", "HLA-B*07:02"], use_hgnc=False)
    # NRAS rows only: KLVVVGAGGV (x2 obs), ILDTAGREEY, and ALAVLGFFV (the
    # one on B*07:02).  KLVVVGAGGV's two obs come from different PMIDs.
    assert set(df["gene_name"]) == {"NRAS"}
    klvv = df[df["peptide"] == "KLVVVGAGGV"].iloc[0]
    assert klvv["n_observations"] == 2
    assert "9263005" in klvv["pmids"] and "36099883" in klvv["pmids"]
    # Allele grouping preserved.
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}
    # Per-allele evidence sort: most-attested peptide first within an allele.
    a02 = df[df["mhc_allele"] == "HLA-A*02:01"].reset_index(drop=True)
    assert a02.iloc[0]["peptide"] == "KLVVVGAGGV"  # n=2 wins over n=1


def test_pmhc_query_multi_mapping_peptide_appears_under_each_gene(tmp_path, monkeypatch):
    """KLVVVGAGGV multi-maps to NRAS+KRAS.  Asking for both genes returns
    the peptide once per gene, not just once."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(proteins=["NRAS", "KRAS"], alleles=["HLA-A*02:01"], use_hgnc=False)
    # KLVVVGAGGV should appear under both NRAS and KRAS rows.
    klvv = df[df["peptide"] == "KLVVVGAGGV"]
    assert set(klvv["gene_name"]) == {"NRAS", "KRAS"}


def test_pmhc_query_filters_to_requested_proteins(tmp_path, monkeypatch):
    """Asking for only NRAS must NOT return KRAS rows even though
    KLVVVGAGGV multi-maps to both."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A*02:01"], use_hgnc=False)
    assert set(df["gene_name"]) == {"NRAS"}
    assert "KRAS" not in df["gene_name"].values


def test_pmhc_query_empty_when_no_match(tmp_path, monkeypatch):
    """A protein with no MS evidence returns an empty (but well-shaped)
    DataFrame."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(proteins=["TP53"], alleles=["HLA-A*02:01"], use_hgnc=False)
    assert df.empty
    # Schema is preserved so consumers can still read columns.
    for col in ("gene_name", "gene_id", "mhc_allele", "peptide", "n_observations", "pmids"):
        assert col in df.columns


def test_pmhc_query_no_proteins_returns_all_genes(tmp_path, monkeypatch):
    """v1.29.6: ``proteins=None`` (or empty) is no longer an error — it
    means "all genes". Filtering by allele alone returns every gene that
    presents on that allele."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(alleles=["HLA-A*02:01"], use_hgnc=False)
    # The fixture has KLVVVGAGGV (NRAS+KRAS), ILDTAGREEY (NRAS), FLPNKQRTV (BRAF) on A*02:01.
    assert set(df["gene_name"]) == {"NRAS", "KRAS", "BRAF"}
    assert set(df["mhc_allele"]) == {"HLA-A*02:01"}


def test_pmhc_query_no_alleles_returns_all_alleles(tmp_path, monkeypatch):
    """v1.29.6: ``alleles=None`` (or empty) is no longer an error — it
    means "all alleles". Filtering by protein alone returns every allele
    that presents the requested gene's peptides."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    assert set(df["gene_name"]) == {"NRAS"}
    # NRAS peptides span both A*02:01 and B*07:02 in the fixture.
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}


def test_pmhc_query_no_filters_returns_everything(tmp_path, monkeypatch):
    """v1.29.6: ``query()`` with no filters returns the full corpus
    aggregated to (gene, allele, peptide). Useful as a "show me all
    pMHC evidence" command."""
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(use_hgnc=False)
    # All 3 genes from the fixture, both alleles.
    assert set(df["gene_name"]) == {"NRAS", "KRAS", "BRAF"}
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}


def test_pmhc_query_format_table_renders_grouped_table(tmp_path, monkeypatch):
    """The table formatter renders protein > allele as section headers with
    peptide rows in a column-aligned table beneath each allele.  The shape
    is "table grouped under section headers", not a flat CSV nor a
    tree-drawing-character tree.
    """
    from hitlist import pmhc_query

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A*02:01", "HLA-B*07:02"], use_hgnc=False)
    text = pmhc_query.format_table(df)

    # Section headers present (gene with id, then nested allele).
    assert "NRAS (ENSG00000213281)" in text
    assert "HLA-A*02:01" in text
    assert "HLA-B*07:02" in text
    # Column header row + rule + data rows under each allele.
    assert "peptide" in text
    assert "n_obs" in text
    assert "pmids" in text
    assert "----" in text  # rule line under each header
    assert "KLVVVGAGGV" in text
    assert "9263005" in text

    # Order: gene → allele → peptide (top-down).
    nras_idx = text.find("NRAS")
    a02_idx = text.find("HLA-A*02:01")
    pep_idx = text.find("KLVVVGAGGV")
    assert nras_idx < a02_idx < pep_idx
