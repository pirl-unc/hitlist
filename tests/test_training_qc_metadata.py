"""Curated MS QC follows resolved arms and consensus, never binding rows (#18)."""

import pandas as pd
import pytest

from hitlist import downloads, export


@pytest.fixture
def qc_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    def sample(label, mhc, engine, fdr):
        return {
            "sample_label": label,
            "mhc": mhc,
            "mhc_class": "I",
            "n_samples": 1,
            "condition": "unperturbed",
            "search_engine": engine,
            "fdr": fdr,
        }

    overrides = {
        99999001: {
            "study_label": "different QC",
            "species": "Homo sapiens",
            "ms_samples": [
                sample("alpha", "HLA-A*02:01", "MaxQuant", "PSM 1%; protein 5%"),
                sample("beta", "HLA-B*07:02", "PEAKS", "peptide 0.5%"),
            ],
        },
        99999002: {
            "study_label": "shared QC",
            "species": "Homo sapiens",
            "ms_samples": [
                sample("gamma", "HLA-A*02:01", "Mascot", "PSM 1%"),
                sample("delta", "HLA-B*07:02", "Mascot", "PSM 1%"),
            ],
        },
    }
    monkeypatch.setattr(export, "load_pmid_overrides", lambda: overrides)
    rows = pd.DataFrame(
        {
            "peptide": [letter * 9 for letter in "ACDEF"],
            "pmid": pd.array([99999001, 99999001, 99999001, 99999002, 99999003], dtype="Int64"),
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA class I",
                "HLA class I",
                "HLA-A*02:01",
            ],
            "mhc_class": ["I"] * 5,
            "mhc_species": ["Homo sapiens"] * 5,
            "source": ["iedb"] * 5,
            "assay_iri": [f"ms:{i}" for i in range(5)],
        }
    )
    rows.assign(assay_method="mass spectrometry").to_parquet(
        tmp_path / "observations.parquet", index=False
    )
    rows.iloc[[0]].assign(assay_iri="binding:1").assign(assay_method="binding assay").to_parquet(
        tmp_path / "binding.parquet", index=False
    )


@pytest.mark.parametrize("mode", ["observations", "ms", "both", "binding"])
@pytest.mark.parametrize("projected", [False, True])
def test_qc_metadata_uses_exact_arm_or_consensus(qc_indexes, mode, projected):
    fields = ["assay_iri", "search_engine", "fdr"]
    options = {"columns": fields} if projected else {}
    result = (
        export.generate_observations_table(**options)
        if mode == "observations"
        else export.generate_training_table(include_evidence=mode, **options)
    )
    expected = {
        "ms:0": ("MaxQuant", "PSM 1%; protein 5%"),
        "ms:1": ("PEAKS", "peptide 0.5%"),
        "ms:2": ("", ""),
        "ms:3": ("Mascot", "PSM 1%"),
        "ms:4": ("", ""),
        "binding:1": ("", ""),
    }
    assert {row.assay_iri: (row.search_engine, row.fdr) for row in result.itertuples()} == {
        key: value
        for key, value in expected.items()
        if (key.startswith("ms:") and mode != "binding")
        or (key.startswith("binding:") and mode in {"both", "binding"})
    }
