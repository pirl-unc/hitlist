"""Full-length join intermediates must not retain replaced metadata (#566)."""

import weakref

import pandas as pd
import pytest

from hitlist import export


@pytest.mark.parametrize("labels", [None, "blank", "hit", "no_hit"])
def test_reindexed_metadata_is_released_before_final_annotations(tmp_path, monkeypatch, labels):
    overrides = {
        99999001: {
            "study_label": "synthetic",
            "species": "Homo sapiens (human)",
            "ms_samples": [
                {
                    "sample_label": "sample",
                    "n_samples": 1,
                    "condition": "unperturbed",
                    "mhc_class": "I",
                    "mhc": "HLA-A*02:01",
                }
            ],
        }
    }
    monkeypatch.setattr(export, "load_pmid_overrides", lambda: overrides)
    observations = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["HLA-A*02:01", "", ""],
            "mhc_class": ["I"] * 3,
            "reference_iri": ["iri:1"] * 3,
            "pmid": pd.array([99999001, 99999001, 99999002], dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "is_monoallelic": [False] * 3,
            "is_binding_assay": [False] * 3,
            "qualitative_measurement": ["Positive"] * 3,
        }
    )
    if labels is not None:
        observations["attributed_sample_label"] = {
            "blank": ["", "", ""],
            "hit": ["sample", "unknown", "sample"],
            "no_hit": ["unknown", "unknown", "sample"],
        }[labels]
    path = tmp_path / "observations.parquet"
    observations.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    joined = {}
    reindex = pd.DataFrame.reindex

    def track_join(self, *args, **kwargs):
        result = reindex(self, *args, **kwargs)
        if "_pmid_int" in self.index.names:
            if "_label" in self.index.names:
                joined["curated_label"] = weakref.ref(result)
            else:
                for column in ("sample_label", "sample_label_fb"):
                    if column in result.columns and len(result) == len(observations):
                        joined[column] = weakref.ref(result)
        return result

    monkeypatch.setattr(pd.DataFrame, "reindex", track_join)
    annotate = export._compute_has_peptide_level_allele
    checked = []

    def check_lifetime(*args, **kwargs):
        expected = {"sample_label", "sample_label_fb"}
        if labels in ("hit", "no_hit"):
            expected.add("curated_label")
        assert set(joined) == expected
        retained = [name for name, reference in joined.items() if reference() is not None]
        assert not retained, f"Completed joins still retain full-length metadata: {retained}"
        checked.append(True)
        return annotate(*args, **kwargs)

    monkeypatch.setattr(export, "_compute_has_peptide_level_allele", check_lifetime)
    result = export.generate_observations_table()
    assert checked == [True]
    assert result["sample_label"].tolist() == ["sample", "sample", ""]
    assert result["sample_attribution"].tolist() == [
        "curated_sample_label" if labels == "hit" else "allele_exact",
        "single_sample_pmid",
        "",
    ]
