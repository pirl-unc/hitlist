"""Reviewed offline inputs keep source identity and cannot fetch unrelated data."""

import hashlib
import json

import pandas as pd
import pytest

from hitlist import supplement
from hitlist.provenance import ContributorCollector


def entry(filename="dog.csv"):
    return {
        "pmid": 42199926,
        "file": filename,
        "source": "fixture:original-workbook#All-Peptides",
        "defaults": {
            "host": "Canis lupus familiaris",
            "species": "Canis lupus familiaris",
            "source_organism": "Canis lupus familiaris",
            "culture_condition": "Direct Ex Vivo",
            "source_tissue": "bone",
            "disease": "cancer",
            "attributed_sample_label": "Lola H58A",
        },
    }


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline input attempted a network download")

    monkeypatch.setattr("hitlist.downloads.fetch_data_asset", forbidden)
    monkeypatch.setattr("requests.get", forbidden)


def test_explicit_source_retains_sample_and_duplicate_contributors(tmp_path):
    path = tmp_path / "dog.csv"
    path.write_text(
        "peptide,mhc_class,mhc_restriction,source_row,source_protein_mappings\n"
        "ACDEFGHIK,I,MHC class I,3,protein-a:protein-b\n"
        "ACDEFGHIK,I,MHC class I,4,protein-a:protein-b\n"
    )
    source = {**entry(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with ContributorCollector(scratch_dir=tmp_path) as collector:
        frame = supplement.scan_supplementary(
            entries=[source], directory=tmp_path, allow_download=False, provenance=collector
        )
        collector.write([frame], tmp_path / "contributors.parquet")
    assert len(frame) == 1
    assert frame.attributed_sample_label.tolist() == ["Lola H58A"]
    assert frame.mhc_allele_set.tolist() == [""]
    assert frame.restriction_evidence.tolist() == ["unknown"]
    assert frame.mhc_species.iloc[0].startswith("Canis")
    assert frame.is_ms_observation.all()
    contributors = pd.read_parquet(tmp_path / "contributors.parquet")
    assert len(contributors) == 2
    assert contributors.attributed_sample_label.eq("Lola H58A").all()
    fields = [json.loads(value) for value in contributors.original_fields]
    assert {row["source_row"] for row in fields} == {"3", "4"}
    assert all(row["manifest"] == source for row in fields)
    assert all(row["source_protein_mappings"] == "protein-a:protein-b" for row in fields)


@pytest.mark.parametrize("allow_download", [False, True])
def test_explicit_missing_file_cannot_fetch_global_asset(tmp_path, monkeypatch, allow_download):
    monkeypatch.setattr("hitlist.downloads.EXTERNAL_DATA_ASSETS", {"dog.csv": {}})
    with pytest.raises(FileNotFoundError):
        supplement.scan_supplementary(
            entries=[entry()], directory=tmp_path, allow_download=allow_download
        )


def test_packaged_manifest_can_be_used_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(supplement, "load_supplementary_manifest", lambda: [entry()])
    with pytest.raises(FileNotFoundError):
        supplement.scan_supplementary(directory=tmp_path, allow_download=False)


def test_explicit_inputs_cannot_bypass_ms_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(supplement, "ms_excluded_pmids", lambda: {42199926})
    with pytest.raises(ValueError, match="exclude_from_ms"):
        supplement.scan_supplementary(entries=[entry()], directory=tmp_path, allow_download=False)


def test_row_labels_override_defaults_without_collapsing_distinct_samples(tmp_path):
    (tmp_path / "dog.csv").write_text(
        "peptide,mhc_class,mhc_restriction,attributed_sample_label\n"
        "ACDEFGHIK,I,MHC class I,Lola H58A\n"
        "ACDEFGHIK,I,MHC class I,Lola BB7.6\n"
    )
    result = supplement.scan_supplementary(
        entries=[entry()], directory=tmp_path, allow_download=False
    )
    assert result.attributed_sample_label.tolist() == ["Lola H58A", "Lola BB7.6"]


def test_explicit_source_hash_and_schema_are_checked(tmp_path):
    path = tmp_path / "dog.csv"
    path.write_text("peptide\nACDEFGHIK\n")
    with pytest.raises(ValueError, match="checksum"):
        supplement.scan_supplementary(
            entries=[{**entry(), "sha256": "0" * 64}], directory=tmp_path, allow_download=False
        )
    path.write_text("wrong_column\nACDEFGHIK\n")
    with pytest.raises(ValueError, match="peptide"):
        supplement.scan_supplementary(entries=[entry()], directory=tmp_path, allow_download=False)


@pytest.mark.parametrize("filename", ["../dog.csv", "/dog.csv"])
def test_explicit_filenames_are_local_components(tmp_path, filename):
    with pytest.raises(ValueError, match="filename"):
        supplement.scan_supplementary(
            entries=[entry(filename)], directory=tmp_path, allow_download=False
        )
