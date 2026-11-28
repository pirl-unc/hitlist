"""Tests for host-proteome registry, deterministic primary flanks, and host-matched flank loading (#991)."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from hitlist import builder, bundle_tables, downloads, host_flanks, mappings
from hitlist.parquet_io import atomic_write_parquet
from tests.test_build_smoke import _write_synthetic_iedb


def test_build_primary_flanks_deterministic_canonical_tiebreaking():
    mappings_df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "GILGFVFTL", "GILGFVFTL", "GILGFVFTL", "SLYNTVATL"],
            "proteome": [
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Bos taurus",
                "Homo sapiens",
            ],
            "protein_id": [
                "ENSP_NONCANON",
                "ENSP_CANON_B",
                "ENSP_CANON_A",
                "ENSBTAP1",
                "ENSP_TERM",
            ],
            "gene_name": ["M1", "M1", "M1", "M1_BOV", "G1"],
            "gene_id": ["ENSG1", "ENSG1", "ENSG1", "ENSBTAG1", "ENSG2"],
            "gene_biotype": [
                "protein_coding",
                "protein_coding",
                "protein_coding",
                "protein_coding",
                "protein_coding",
            ],
            "transcript_id": ["ENST1", "ENST2", "ENST3", "", "ENST4"],
            "is_canonical_transcript": [False, True, True, False, True],
            "position": [10, 50, 20, 15, 0],
            "n_flank": ["ZZZZZ", "BBBBB", "AAAAA", "CCCCC", ""],
            "c_flank": ["ZZZZZ", "BBBBB", "TTTTT", "GGGGG", "DDDDD"],
        }
    )

    pf = host_flanks.build_primary_flanks_table(mappings_df)
    assert len(pf) == 3

    # Canonical + lexicographically smallest protein_id wins for (GILGFVFTL, Homo sapiens).
    human_gil = pf[(pf["peptide"] == "GILGFVFTL") & (pf["proteome_id"] == "Homo sapiens")].iloc[0]
    assert human_gil["protein_id"] == "ENSP_CANON_A"
    assert human_gil["n_flank"] == "AAAAA"
    assert human_gil["c_flank"] == "TTTTT"
    assert human_gil["n_mappings"] == 3

    # Bos taurus retains its own separate 1:1 host flank row.
    bovine_gil = pf[(pf["peptide"] == "GILGFVFTL") & (pf["proteome_id"] == "Bos taurus")].iloc[0]
    assert bovine_gil["protein_id"] == "ENSBTAP1"
    assert bovine_gil["n_flank"] == "CCCCC"
    assert bovine_gil["n_mappings"] == 1


def test_build_proteomes_registry_hashes_fasta_and_includes_zero_hit_proteomes(tmp_path):
    fasta = tmp_path / "proteomes" / "bos_taurus.fasta"
    fasta.parent.mkdir(parents=True)
    fasta_bytes = b">P1 bovine\nMAAAAAGILGFVFTLTTTTT\n>P2 bovine\nMCCCCCCCC\n"
    fasta.write_bytes(fasta_bytes)
    expected_sha = hashlib.sha256(fasta_bytes).hexdigest()

    mappings_df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL"],
            "protein_id": ["P1"],
            "proteome": ["Bos taurus"],
            "proteome_source": ["uniprot:UP000009136"],
        }
    )
    meta = {
        "per_proteome": {
            "Bos taurus": {"peptides_searched": 10, "peptides_matched": 1},
            "Canis lupus familiaris": {"peptides_searched": 5, "peptides_matched": 0},
        }
    }
    manifest = {
        "Bos taurus": {
            "path": str(fasta),
            "taxon_id": 9913,
            "source": "uniprot:UP000009136",
        }
    }
    reg = host_flanks.build_proteomes_registry(
        mappings_df,
        mappings_meta=meta,
        manifest_proteomes=manifest,
        root=tmp_path,
    )
    assert set(reg["proteome_id"]) == {"Bos taurus", "Canis lupus familiaris"}

    bos = reg[reg["proteome_id"] == "Bos taurus"].iloc[0]
    assert int(bos["taxon_id"]) == 9913
    assert bos["sha256"] == expected_sha
    assert int(bos["n_proteins"]) == 2
    assert int(bos["n_peptides_mapped"]) == 1
    assert bos["fasta_path"] == "proteomes/bos_taurus.fasta"

    dog = reg[reg["proteome_id"] == "Canis lupus familiaris"].iloc[0]
    assert int(dog["n_peptides_mapped"]) == 0


def test_attach_host_flanks_prevents_cross_species_leak_and_load_flanked_ligands(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    obs = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV", "GILGFVFTL"],
            "mhc_class": ["I", "I", "I"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Bos taurus"],
            "source_organism": ["Homo sapiens", "Homo sapiens", "Bos taurus"],
            "mhc_allele_set": ["HLA-A*02:01", "HLA-A*02:01", "BoLA-1*023:01"],
            "pmid": [28228285, 28228285, 36423003],
            "source": ["iedb", "iedb", "iedb"],
            "has_ptm": [False, False, False],
            "is_non_peptide_ligand": [False, False, False],
            "is_potential_contaminant": [False, False, False],
        }
    )
    # Only Homo sapiens has GILGFVFTL mapped — Bos taurus cohort has 0% mapped flanks!
    pf = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL"],
            "proteome_id": ["Homo sapiens"],
            "n_flank": ["AAAAA"],
            "c_flank": ["TTTTT"],
            "protein_id": ["ENSP1"],
            "gene_name": ["M1"],
            "gene_id": ["ENSG1"],
            "transcript_id": ["ENST1"],
            "is_canonical_transcript": [True],
            "position": [10],
            "n_mappings": [1],
        }
    )
    atomic_write_parquet(obs, bundle / "observations.parquet")
    atomic_write_parquet(pf, bundle / "primary_flanks.parquet")

    # Bos taurus must NOT inherit Homo sapiens flanks even though GILGFVFTL matches by peptide alone!
    cattle = host_flanks.load_flanked_ligands(species="Bos taurus", mhc_class="I", bundle=bundle)
    assert len(cattle) == 1
    assert cattle.iloc[0]["n_flank"] == ""
    assert cattle.iloc[0]["c_flank"] == ""
    assert bool(cattle.iloc[0]["flank_mapped"]) is False
    assert cattle.iloc[0]["cohort_flank_coverage"] == pytest.approx(0.0)

    # Filtering with min_flank_coverage=0.4 drops the 0%-flank Cattle cohort and keeps the 50%-flank Human cohort.
    filtered = host_flanks.load_flanked_ligands(
        species=None, mhc_class="I", bundle=bundle, min_flank_coverage=0.4
    )
    assert set(filtered["mhc_species"]) == {"Homo sapiens"}
    assert filtered.iloc[0]["cohort_flank_coverage"] == pytest.approx(0.5)

    flanked_only = host_flanks.load_flanked_ligands(
        species="Homo sapiens", mhc_class="I", bundle=bundle, require_flanks=True
    )
    assert flanked_only["peptide"].tolist() == ["GILGFVFTL"]
    assert flanked_only.iloc[0]["n_flank"] == "AAAAA"


def test_attach_host_flanks_maps_subspecies_to_canonical_proteome_id():
    """Canis lupus familiaris rows must match Canis lupus proteome_id (#991)."""
    obs = pd.DataFrame(
        {
            "peptide": ["KLQCVDLHV"],
            "mhc_species": ["Canis lupus familiaris"],
            "source_organism": ["Canis lupus familiaris"],
        }
    )
    pf = pd.DataFrame(
        {
            "peptide": ["KLQCVDLHV"],
            "proteome_id": ["Canis lupus"],
            "n_flank": ["DDDDD"],
            "c_flank": ["EEEEE"],
            "protein_id": ["ENSCAFP1"],
            "gene_name": ["DOG1"],
            "gene_id": ["ENSCAFG1"],
            "transcript_id": ["ENSCAFT1"],
            "is_canonical_transcript": [True],
            "position": [12],
            "n_mappings": [1],
        }
    )
    attached = host_flanks.attach_host_flanks(obs, primary_flanks=pf)
    assert attached.iloc[0]["primary_proteome_id"] == "Canis lupus"
    assert attached.iloc[0]["n_flank"] == "DDDDD"
    assert attached.iloc[0]["c_flank"] == "EEEEE"
    assert bool(attached.iloc[0]["has_flank"]) is True


def test_build_observations_writes_proteomes_and_primary_flanks(
    tmp_path, monkeypatch, _isolated_curation_root
):
    from hitlist import supplement

    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    source = tmp_path / "iedb.csv"
    _write_synthetic_iedb(source)
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())

    def empty_index(path):
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder, "build_bulk_proteomics", lambda **kw: empty_index(builder._bulk_proteomics_path())
    )
    monkeypatch.setattr(
        builder, "build_line_expression", lambda **kw: empty_index(builder._line_expression_path())
    )

    def tiny_mappings(**kwargs):
        path = mappings.mappings_path()
        frame = kwargs.get("obs_override")
        if frame is None:
            frame = pd.read_parquet(builder._observations_path())
        peps = frame["peptide"].drop_duplicates().tolist()
        m = pd.DataFrame(
            {
                "peptide": peps,
                "protein_id": [f"ENSP{i}" for i in range(len(peps))],
                "gene_name": [f"GENE{i}" for i in range(len(peps))],
                "gene_id": [f"ENSG{i}" for i in range(len(peps))],
                "gene_biotype": ["protein_coding"] * len(peps),
                "transcript_id": [f"ENST{i}" for i in range(len(peps))],
                "is_canonical_transcript": [True] * len(peps),
                "position": [10] * len(peps),
                "n_flank": ["AAAAA"] * len(peps),
                "c_flank": ["TTTTT"] * len(peps),
                "proteome": ["Homo sapiens"] * len(peps),
                "proteome_source": ["ensembl"] * len(peps),
            }
        )
        atomic_write_parquet(m, path)
        mappings.mappings_meta_path().write_text(
            json.dumps(
                {
                    "observations": mappings._obs_fingerprint(),
                    "per_proteome": {
                        "Homo sapiens": {
                            "peptides_searched": len(peps),
                            "peptides_matched": len(peps),
                        }
                    },
                }
            )
        )
        return path

    monkeypatch.setattr(mappings, "build_peptide_mappings", tiny_mappings)

    obs_path = builder.build_observations()
    bundle = obs_path.parent
    assert (bundle / "proteomes.parquet").is_file()
    assert (bundle / "primary_flanks.parquet").is_file()

    reg = host_flanks.load_proteomes()
    assert "Homo sapiens" in set(reg["proteome_id"])
    flanks = host_flanks.load_primary_flanks(proteome_id="Homo sapiens")
    assert not flanks.empty
    assert (flanks["n_flank"] == "AAAAA").all()

    # samples.parquet automatically receives real flank_coverage from primary_flanks!
    samples = bundle_tables.load_samples_index()
    assert (samples["n_flanked_peptides"] > 0).all()
    assert (samples["flank_coverage"] > 0.0).all()
