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

"""Tests for non-model vertebrate proteomes, BoLA-I PXD008151 supplement, and Cat/Bat curation (#980)."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.version import Version

from hitlist import curation, downloads, supplement
from hitlist.curation_yaml import load_curation_yaml


def test_mhcgnomes_and_mhcseqs_minimum_versions():
    import mhcgnomes
    import mhcseqs

    assert Version(mhcgnomes.__version__) >= Version("3.91.0")
    assert Version(mhcseqs.__version__) >= Version("2.14.0")
    assert mhcseqs.normalize_allele_name("FLA-E*00301") == "FLA-E*003:01"
    assert mhcseqs.normalize_allele_name("BoLA-6*01301") == "BoLA-6*013:01"
    assert mhcseqs.get_ncbi_taxid("Ptal") == 9402


def test_nonmodel_vertebrate_reference_proteomes_registered():
    expected = {
        "Felis catus": ("UP000823872", 9685),
        "Felis sp.": ("UP000823872", 9685),
        "Pteropus alecto": ("UP000010552", 9402),
        "Myotis lucifugus": ("UP000001074", 59463),
        "Ovis aries": ("UP000809102", 9940),
        "Ovis sp.": ("UP000809102", 9940),
        "Capra hircus": ("UP000291000", 9925),
        "Anas platyrhynchos": ("UP000016666", 8840),
        "Xenopus laevis": ("UP000186698", 8355),
        "Ailuropoda melanoleuca": ("UP000008912", 9646),
        "Oryctolagus cuniculus": ("UP000001811", 9986),
        "Oryctolagus sp.": ("UP000001811", 9986),
        "Callithrix jacchus": ("UP000008225", 9483),
    }
    for species, (upid, taxon_id) in expected.items():
        entry = downloads.lookup_proteome(species)
        assert entry is not None, f"Missing proteome entry for {species!r}"
        assert entry["kind"] == "uniprot"
        assert entry["proteome_id"] == upid
        assert entry["expected_taxon_id"] == taxon_id


def test_cat_bat_and_cattle_pmid_curations():
    overrides = curation.load_pmid_overrides()

    # Cat: Qiao 2025 cellular/elution vs Liang 2018 crystallography/binding
    qiao = overrides[40073263]
    assert qiao["species"] == "Felis catus"
    assert not qiao.get("exclude_from_ms")
    assert {s["mhc"] for s in qiao["ms_samples"]} == {"FLA-E*003:01", "FLA-K*007:01"}
    assert overrides[29263258]["exclude_from_ms"] is True

    # Bat: Wynne 2016 cellular PaKiT03 elution vs Qu 2019 RPLD-MS and Lu 2019 X-ray
    wynne = overrides[27183594]
    assert wynne["species"] == "Pteropus alecto"
    assert not wynne.get("exclude_from_ms")
    assert wynne["ms_samples"][0]["mhc"] == "Ptal-N*01:01"
    assert overrides[31076531]["exclude_from_ms"] is True
    assert overrides[31498797]["exclude_from_ms"] is True

    # Cattle: Nielsen 2018 BoLA-I PXD008151
    nielsen = overrides[29115832]
    assert nielsen["species"] == "Bos taurus"
    assert len(nielsen["ms_samples"]) == 5


def test_supplement_per_row_host_mhc_types_fixture(tmp_path):
    """Unit test: per-row host_mhc_types resolves each bovine haplotype without PMID pooling."""
    csv_path = tmp_path / "bola_fixture.csv"
    csv_path.write_text(
        "peptide,mhc_class,mhc_restriction,host_mhc_types,attributed_sample_label\n"
        "SYFPEITHI,I,BoLA-6*013:01,BoLA-6*013:01,641_A18_Tpm (BoLA-I A18)\n"
        "SIINFEKLA,I,BoLA class I,BoLA-2*012:01;BoLA-3*002:01,1011_A10_Tpm (BoLA-I A10)\n"
        "GILGFVFTL,I,BoLA class I,BoLA-1*023:01;BoLA-4*024:01;BoLA-2*025:01,104003_A14 (BoLA-I A14)\n",
        encoding="utf-8",
    )
    entry = {
        "pmid": 29115832,
        "file": csv_path.name,
        "study_label": "BoLA fixture",
        "defaults": {
            "host": "Bos taurus (cattle)",
            "species": "Bos taurus",
            "source_organism": "Bos taurus",
            "culture_condition": "Cell Line / Clone",
            "cell_name": "Lymphoblast",
            "source_tissue": "Blood",
            "process_type": "No immunization",
            "disease": "healthy",
        },
    }
    df = supplement.scan_supplementary(entries=[entry], directory=tmp_path, allow_download=False)
    by_pep = df.set_index("peptide")
    assert (df["mhc_species"] == "Bos taurus").all()
    assert by_pep.loc["SYFPEITHI", "mhc_allele_provenance"] == "exact"
    assert by_pep.loc["SYFPEITHI", "mhc_allele_set_size"] == 1
    assert by_pep.loc["SIINFEKLA", "mhc_allele_provenance"] == "sample_allele_match"
    assert by_pep.loc["SIINFEKLA", "mhc_allele_set_size"] == 2
    assert by_pep.loc["GILGFVFTL", "mhc_allele_provenance"] == "sample_allele_match"
    assert by_pep.loc["GILGFVFTL", "mhc_allele_set_size"] == 3


@pytest.mark.integration
def test_nielsen_2018_bola_class1_supplement_scans_with_per_sample_haplotypes():
    entries = [e for e in supplement.load_supplementary_manifest() if e["pmid"] == 29115832]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["file"] == "nielsen_2018_bola_class1.csv.gz"

    df = supplement.scan_supplementary(entries=[entry], allow_download=False)
    assert len(df) == 36_943
    assert df["peptide"].nunique() == 27_314
    assert (df["mhc_species"] == "Bos taurus").all()
    assert (df["mhc_class"] == "I").all()
    assert df["attributed_sample_label"].nunique() == 5

    # Monoallelic A18 sample resolves as exact single allele (BoLA-6*013:01 / Bota-6*013:01)
    a18 = df[df["attributed_sample_label"] == "641_A18_Tpm (BoLA-I A18)"]
    assert len(a18) == 6_911
    assert (a18["mhc_allele_provenance"] == "exact").all()
    assert (a18["mhc_allele_set_size"] == 1).all()

    # Multi-allelic A10 and A14 samples use per-row host_mhc_types (2 and 3 alleles, never pooled)
    a10 = df[df["attributed_sample_label"].str.contains("A10")]
    assert len(a10) == 7_696 + 8_460
    assert (a10["mhc_allele_provenance"] == "sample_allele_match").all()
    assert (a10["mhc_allele_set_size"] == 2).all()

    a14 = df[df["attributed_sample_label"].str.contains("A14")]
    assert len(a14) == 6_821 + 7_055
    assert (a14["mhc_allele_provenance"] == "sample_allele_match").all()
    assert (a14["mhc_allele_set_size"] == 3).all()


def test_apm_orthologs_psmb11_marks_chicken_absent():
    apm = load_curation_yaml(
        Path(__file__).resolve().parent.parent / "hitlist" / "data" / "apm_orthologs.yaml"
    )
    assert "Gallus gallus" in apm["ortholog_groups"]["PSMB11"]["absent_in_species"]
