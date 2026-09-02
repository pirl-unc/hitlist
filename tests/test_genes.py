"""Tests for gene name/ID resolution with HGNC synonym support."""

from __future__ import annotations

import sys
import types

import pytest

from hitlist.genes import (
    _is_ensembl_gene_id,
    list_gene_sets,
    load_gene_set,
    resolve_gene_query,
)


@pytest.fixture
def fake_cancerdata(monkeypatch):
    expected = ["PRAME", "MAGEA4", "CAGE1", "BRDT", *[f"CTA{i}" for i in range(201)]]
    package = types.ModuleType("cancerdata")
    cta = types.ModuleType("cancerdata.cta")
    cta.CTA_gene_names = lambda: list(expected)
    package.cta = cta
    monkeypatch.setitem(sys.modules, "cancerdata", package)
    monkeypatch.setitem(sys.modules, "cancerdata.cta", cta)
    return expected


def test_load_gene_set_cta_sourced_from_cancerdata(fake_cancerdata):
    """The CTA set delegates to the cancerdata package (its CTpedia/daSilva2017
    candidates filtered by HPA reproductive/thymus restriction) — a single
    source of truth, not a hand-maintained duplicate."""
    genes = load_gene_set("CTA")
    assert set(genes) == set(fake_cancerdata)
    assert len(genes) > 200  # the full restriction-filtered panel, not a subset
    for expected in ("PRAME", "MAGEA4", "CAGE1", "BRDT"):
        assert expected in genes
    assert len(genes) == len(set(genes))


def test_load_gene_set_is_case_insensitive(fake_cancerdata):
    assert load_gene_set("cta") == load_gene_set("CTA")


def test_load_gene_set_missing_optional_provider_is_actionable(monkeypatch):
    from hitlist.genes import _genes_from_provider

    monkeypatch.delitem(sys.modules, "cancerdata", raising=False)
    monkeypatch.delitem(sys.modules, "cancerdata.cta", raising=False)
    real_import = __import__

    def import_without_cancerdata(name, *args, **kwargs):
        if name == "cancerdata":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_cancerdata)

    with pytest.raises(
        RuntimeError,
        match=r"pip install git\+https://github.com/pirl-unc/cancerdata",
    ):
        _genes_from_provider("cancerdata", set_name="CTA")


def test_load_gene_set_unknown_raises():
    with pytest.raises(KeyError, match="unknown gene set"):
        load_gene_set("NOPE")


def test_unknown_provider_raises():
    from hitlist.genes import _genes_from_provider

    with pytest.raises(RuntimeError, match="unknown provider"):
        _genes_from_provider("not_a_real_provider", set_name="X")


def test_list_gene_sets_includes_cta():
    sets = dict(list_gene_sets())
    assert "CTA" in sets
    assert "cancer-testis" in sets["CTA"].lower()


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
