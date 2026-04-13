"""Tests for gene name/ID resolution with HGNC synonym support."""

from __future__ import annotations

from hitlist.genes import _is_ensembl_gene_id, resolve_gene_query


def test_ensembl_id_detection():
    assert _is_ensembl_gene_id("ENSG00000120337")
    assert _is_ensembl_gene_id("ENSG00000120337.15")  # versioned
    assert not _is_ensembl_gene_id("PRAME")
    assert not _is_ensembl_gene_id("")
    assert not _is_ensembl_gene_id("ENSG")


def test_resolve_gene_query_symbol():
    """A current symbol should land in the 'names' set."""
    spec = resolve_gene_query("PRAME", use_hgnc=False)
    assert "PRAME" in spec["names"]
    assert not spec["ids"]


def test_resolve_gene_query_ensembl_id():
    """Ensembl IDs go to 'ids', not 'names'."""
    spec = resolve_gene_query("ENSG00000185686", use_hgnc=False)
    assert spec["ids"] == {"ENSG00000185686"}
    assert not spec["names"]


def test_resolve_gene_query_comma_list():
    """Comma-separated input yields multiple resolutions."""
    spec = resolve_gene_query("PRAME,MAGEA1,ENSG00000120337", use_hgnc=False)
    assert "PRAME" in spec["names"]
    assert "MAGEA1" in spec["names"]
    assert "ENSG00000120337" in spec["ids"]


def test_resolve_gene_query_uses_hgnc_synonyms(monkeypatch):
    """HGNC fallback should resolve aliases like 'MART-1' → 'MLANA'."""
    from hitlist import genes

    # Clear the lru_cache from prior tests and force a fresh cache on disk
    genes.resolve_hgnc_symbol.cache_clear()

    def fake_fetch(query, timeout=10):
        return [{"symbol": "MLANA"}] if query == "MART-1" else []

    monkeypatch.setattr(genes, "_fetch_hgnc", fake_fetch)
    monkeypatch.setattr(genes, "_load_cache", lambda: {})
    monkeypatch.setattr(genes, "_save_cache", lambda c: None)

    spec = resolve_gene_query("MART-1")
    assert "MLANA" in spec["names"]


def test_resolve_gene_query_empty():
    assert resolve_gene_query("") == {"names": set(), "ids": set()}
    assert resolve_gene_query("  ") == {"names": set(), "ids": set()}
