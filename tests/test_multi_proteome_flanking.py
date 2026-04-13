"""Tests for multi-proteome flanking (species registry + _add_flanking routing)."""

from __future__ import annotations

import pandas as pd

from hitlist.downloads import SPECIES_PROTEOMES, VIRAL_PROTEOMES, lookup_proteome
from hitlist.proteome import ProteomeIndex


def test_species_registry_contains_expected_entries():
    assert "Homo sapiens" in SPECIES_PROTEOMES
    assert "Mus musculus" in SPECIES_PROTEOMES
    assert "Sarcophilus harrisii" in SPECIES_PROTEOMES
    # At least one viral proteome
    assert "severe acute respiratory syndrome coronavirus 2" in VIRAL_PROTEOMES


def test_lookup_proteome_ensembl_species():
    for variant in ("Homo sapiens", "human", "homo_sapiens", "Homo sapiens (human)"):
        entry = lookup_proteome(variant)
        assert entry is not None, f"failed for {variant!r}"
        assert entry["kind"] == "ensembl"
        assert entry["canonical_species"] == "Homo sapiens"


def test_lookup_proteome_uniprot_species():
    entry = lookup_proteome("Sarcophilus harrisii")
    assert entry is not None
    assert entry["kind"] == "uniprot"
    assert entry["proteome_id"] == "UP000007648"


def test_lookup_proteome_dog_normalizes():
    # normalize_species("Canis lupus familiaris") → "Canis lupus"
    entry = lookup_proteome("Canis lupus familiaris")
    assert entry is not None
    assert entry["canonical_species"] == "Canis lupus"


def test_lookup_proteome_viral_substring():
    # IEDB often includes strain info in source_organism
    entry = lookup_proteome("Epstein-Barr virus (strain B95-8)")
    assert entry is not None
    assert entry["kind"] == "uniprot"
    assert entry.get("key") == "ebv"


def test_lookup_proteome_unknown():
    assert lookup_proteome("Pteropus alecto") is None
    assert lookup_proteome("unidentified") is None
    assert lookup_proteome("") is None


def test_lookup_proteome_sars_disambiguation():
    """SARS-CoV-1 and SARS-CoV-2 must resolve to different proteomes."""
    cov1 = lookup_proteome("SARS-CoV1")
    cov2 = lookup_proteome("SARS-CoV2 Omicron")
    assert cov1 is not None
    assert cov2 is not None
    assert cov1["proteome_id"] == "UP000000354"
    assert cov2["proteome_id"] == "UP000464024"

    # Canonical IEDB strings should also disambiguate
    cov2_full = lookup_proteome("Severe acute respiratory syndrome coronavirus 2")
    cov1_full = lookup_proteome("Severe acute respiratory syndrome coronavirus")
    assert cov2_full["proteome_id"] == "UP000464024"
    assert cov1_full["proteome_id"] == "UP000000354"


def test_lookup_proteome_herpesvirus_aliases():
    """IEDB alphaherpesvirus/betaherpesvirus names resolve to correct HSV/HCMV."""
    assert lookup_proteome("Human betaherpesvirus 5")["proteome_id"] == "UP000000938"  # HCMV
    assert lookup_proteome("Human cytomegalovirus")["proteome_id"] == "UP000000938"
    assert lookup_proteome("Human alphaherpesvirus 1")["proteome_id"] == "UP000009294"  # HSV-1
    assert lookup_proteome("Human alphaherpesvirus 2")["proteome_id"] == "UP000001874"  # HSV-2
    assert lookup_proteome("Human gammaherpesvirus 8")["proteome_id"] == "UP000009113"  # KSHV


def test_lookup_proteome_strain_variant_substring():
    """Strain suffixes on known species names should still resolve."""
    assert lookup_proteome("Theileria parva strain Muguga") is not None
    assert lookup_proteome("Theileria parva strain Muguga")["proteome_id"] == "UP000001949"
    # Murid betaherpesvirus 1 (strain Smith) → MCMV
    mcmv = lookup_proteome("Murine cytomegalovirus (strain Smith)")
    assert mcmv is not None
    assert mcmv["proteome_id"] == "UP000008774"


def test_lookup_proteome_genus_abbreviated():
    """IEDB's 'Sus sp.' / 'Canis sp.' should resolve to type species."""
    sus = lookup_proteome("Sus sp.")
    assert sus is not None
    assert sus["proteome_id"] == "UP000008227"  # Sus scrofa
    canis = lookup_proteome("Canis sp.")
    assert canis is not None
    assert canis["proteome_id"] == "UP000002254"  # Canis lupus


def test_lookup_proteome_uniprot_fallback_uses_cache(tmp_path, monkeypatch):
    """UniProt fallback should hit the network once, then use manifest cache."""
    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    calls = {"n": 0}

    def fake_resolve(organism, timeout=15):
        calls["n"] += 1
        if organism == "Mycobacterium tuberculosis":
            return {
                "proteome_id": "UP000001584",
                "scientific_name": "Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)",
                "taxon_id": 83332,
                "proteome_type": "Reference and representative proteome",
                "protein_count": 3997,
            }
        return None

    monkeypatch.setattr(downloads, "resolve_proteome_via_uniprot", fake_resolve)

    # First call hits the mock
    r1 = downloads.lookup_proteome("Mycobacterium tuberculosis", use_uniprot=True)
    assert r1 is not None
    assert r1["proteome_id"] == "UP000001584"
    assert calls["n"] == 1

    # Second call should use the manifest cache, not the mock
    r2 = downloads.lookup_proteome("Mycobacterium tuberculosis", use_uniprot=True)
    assert r2 == r1
    assert calls["n"] == 1  # no new network call

    # Negative results cached too
    neg1 = downloads.lookup_proteome("Nonexistent organism", use_uniprot=True)
    assert neg1 is None
    assert calls["n"] == 2
    neg2 = downloads.lookup_proteome("Nonexistent organism", use_uniprot=True)
    assert neg2 is None
    assert calls["n"] == 2  # negative also cached


def test_lookup_proteome_no_uniprot_by_default(tmp_path, monkeypatch):
    """Without use_uniprot=True, unknown organisms should return None without calling UniProt."""
    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    def fake_resolve(organism, timeout=15):
        raise AssertionError("Should not call UniProt without use_uniprot=True")

    monkeypatch.setattr(downloads, "resolve_proteome_via_uniprot", fake_resolve)

    assert downloads.lookup_proteome("Mycobacterium tuberculosis") is None


def test_fetch_species_proteome_ensembl_no_download(tmp_path, monkeypatch):
    """Ensembl species should return None (no FASTA to download) and update manifest."""
    from hitlist import downloads
    from hitlist.downloads import fetch_species_proteome

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    result = fetch_species_proteome("Homo sapiens", verbose=False)
    assert result is None

    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    import json

    data = json.loads(manifest.read_text())
    assert "Homo sapiens" in data.get("proteomes", {})
    assert data["proteomes"]["Homo sapiens"]["kind"] == "ensembl"


def test_add_flanking_per_species_routing(tmp_path, monkeypatch):
    """_add_flanking should route each observation to its species' proteome."""
    from hitlist import builder, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    # Synthetic obs with 2 species
    obs = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "ZZZZZZZZZ"],
            "source_organism": ["Homo sapiens", "Mus musculus", "unidentified"],
            "mhc_species": ["Homo sapiens", "Mus musculus", ""],
            "mhc_class": ["I", "I", "I"],
        }
    )

    # Build tiny in-memory proteomes
    human_idx = ProteomeIndex._build(
        {"HUMAN1": "XXXAAAAAAAAAYYY"},
        {"HUMAN1": {"gene_name": "HGENE", "gene_id": "ENSG1"}},
        lengths=(9,),
        verbose=False,
    )
    mouse_idx = ProteomeIndex._build(
        {"MOUSE1": "QQQBBBBBBBBBPPP"},
        {"MOUSE1": {"gene_name": "MGENE", "gene_id": "ENSMUSG1"}},
        lengths=(9,),
        verbose=False,
    )

    # Monkeypatch _load_species_index to return our in-memory indices
    def fake_load(organism, release, verbose, use_uniprot=False):
        if organism == "Homo sapiens":
            return human_idx, "Homo sapiens"
        if organism == "Mus musculus":
            return mouse_idx, "Mus musculus"
        return None, None

    monkeypatch.setattr(builder, "_load_species_index", fake_load)
    # Skip auto-fetch
    from hitlist import downloads

    monkeypatch.setattr(downloads, "fetch_species_proteome", lambda *a, **kw: None)

    result = builder._add_flanking(obs.copy(), release=112, fetch_missing=False)

    # Human peptide mapped to human gene
    human_row = result[result["peptide"] == "AAAAAAAAA"].iloc[0]
    assert human_row["gene_name"] == "HGENE"
    assert human_row["flanking_species"] == "Homo sapiens"

    # Mouse peptide mapped to mouse gene
    mouse_row = result[result["peptide"] == "BBBBBBBBB"].iloc[0]
    assert mouse_row["gene_name"] == "MGENE"
    assert mouse_row["flanking_species"] == "Mus musculus"

    # Unidentified organism: no mapping
    unk_row = result[result["peptide"] == "ZZZZZZZZZ"].iloc[0]
    assert pd.isna(unk_row["gene_name"])
