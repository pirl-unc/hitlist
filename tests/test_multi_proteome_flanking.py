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


def test_fetch_species_proteome_ensembl_no_download(tmp_path, monkeypatch):
    """Ensembl species should return None (no FASTA to download) and update manifest."""
    from hitlist.downloads import fetch_species_proteome, set_data_dir

    set_data_dir(tmp_path)
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
    from hitlist import builder
    from hitlist.downloads import set_data_dir

    set_data_dir(tmp_path)

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
    def fake_load(organism, release, verbose):
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
