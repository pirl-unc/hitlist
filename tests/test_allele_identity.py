"""Retired designations are identities, never new reported typing (#456)."""

import pandas as pd
import pytest

from hitlist import curation
from hitlist.observations import load_binding, load_observations


@pytest.mark.parametrize(
    "reported, expected",
    [
        ("B*44:01", "HLA-B*44:02"),
        ("HLA-B*44:01:01", "HLA-B*44:01:01"),
        ("HLA-B*35:43:02", "HLA-B*35:185"),
        ("HLA-A*02:17:01", "HLA-A*02:17:01"),
        ("HLA-B*44:01;HLA-B*44:02", "HLA-B*44:02"),
        ("HLA-A*02:01;HLA-B*44:01", "HLA-A*02:01;HLA-B*44:02"),
        ("HLA-B*41:01", "HLA-B*41:01"),
        ("HLA-B*44", "HLA-B*44"),
        ("HLA-B44", "HLA-B44"),
        ("H-2Kb", "H2-K*b"),
        ("HLA-B*44:01 E76C mutant", "HLA-B*44:02 E76C mutant"),
        ("HLA-B*08:01 E76C mutant", "HLA-B*08:01 E76C mutant"),
        ("HLA-DRA*01:01 F54C mutant/DRB1*01:01", "HLA-DRA*01:01 F54C mutant/DRB1*01:01"),
        ("HLA-DRA*01:01/DRB1*01:01 C30S mutant", "HLA-DRA*01:01/DRB1*01:01 C30S mutant"),
    ],
)
def test_identity_preserves_precision_and_mutations(reported, expected):
    assert curation.resolve_allele_identity(reported) == expected
    assert curation.resolve_allele_identity(expected) == expected


def test_reported_annotation_survives_derived_rename():
    annotation = curation.resolve_mhc_annotation("HLA-B*44:01")
    assert annotation.restriction == "HLA-B*44:01"
    assert curation.normalize_allele("B*44:01") == "HLA-B*44:01"
    assert curation.allele_to_all_serotypes("HLA-B*44:01") == ("HLA-B44", "HLA-B12", "HLA-Bw4")
    assert curation.expand_allele_set("HLA-B*44:01") == ("HLA-B*44:02", "exact", 1)
    assert curation.sample_mhc_candidates("B*44:01 B*44:02").exact == {"HLA-B*44:02"}
    assert curation.expand_allele_components("HLA-B*44:01") == ["HLA-B*44:02"]


@pytest.mark.parametrize("loader", [load_observations, load_binding])
@pytest.mark.parametrize("query_name", ["HLA-B*44:01", "HLA-B*44:02"])
@pytest.mark.parametrize("filter_name", ["mhc_restriction", "mhc_allele_in_set", "serotype"])
@pytest.mark.parametrize("project", [False, True])
def test_stored_aliases_filter_symmetrically_without_rewriting_source(
    tmp_path, monkeypatch, loader, query_name, filter_name, project
):
    path = tmp_path / "index.parquet"
    reported = ["HLA-B*44:01", "HLA-B*44:02", "HLA-A*02:01"]
    pd.DataFrame(
        {
            "peptide": ["OLDPEPTID", "NEWPEPTID", "UNRELATED"],
            "mhc_restriction": reported,
            "mhc_allele_set": reported,
            "mhc_allele_set_size": [1, 1, 1],
            "serotypes": ["", "HLA-B44;HLA-Bw4", "HLA-A2"],
        }
    ).assign(
        assay_method="binding assay" if loader is load_binding else "mass spectrometry",
        qualitative_measurement="Positive",
    ).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: path)
    columns = ["peptide", "serotypes"] if project else None
    query_value = "HLA-B44" if filter_name == "serotype" else query_name
    df = loader(columns=columns, **{filter_name: query_value})
    assert list(df.peptide) == ["OLDPEPTID", "NEWPEPTID"]
    assert all("HLA-B44" in value for value in df.serotypes)
    if project:
        assert set(df.columns) == set(columns)
    else:
        assert list(df.mhc_restriction) == reported[:2]
        assert list(df.mhc_allele_set) == ["HLA-B*44:02"] * 2
        assert list(df.mhc_allele_set_size) == [1, 1]


def test_retired_catalog_entry_and_current_name_share_serotypes():
    # Colon boundaries are significant (#897): modern HLA-B*15:112 (112th B*15
    # allele, formerly B*9512) has broad HLA-B15 only, whereas HLA-B*15:11 has
    # split HLA-B75 followed by its WHO broad parent HLA-B15.
    assert curation.allele_to_all_serotypes("HLA-B*15:112") == ("HLA-B15",)
    assert curation.allele_to_all_serotypes("HLA-B*15:11") == ("HLA-B75", "HLA-B15")
    assert curation.allele_to_serotype("HLA-B*15:11") == "HLA-B75"
    assert "HLA-B*15:11" in curation.serotype_to_alleles("HLA-B15")
    assert "HLA-B*15:112" in curation.serotype_to_alleles("HLA-B15")


def test_stored_current_name_refreshes_retired_catalog_membership(tmp_path, monkeypatch):
    path = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK"],
            "mhc_restriction": pd.Series(["HLA-B*15:11"], dtype="category"),
            "serotype": pd.Series(["HLA-B15"], dtype="category"),
            "serotypes": ["HLA-B75"],
            "serotype_source": pd.Series(["derived"], dtype="category"),
        }
    ).assign(assay_method="mass spectrometry", qualitative_measurement="Positive").to_parquet(
        path, index=False
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    result = load_observations(serotype="HLA-B15", columns=["peptide", "serotype", "serotypes"])
    assert list(result.peptide) == ["ACDEFGHIK"]
    assert list(result.serotype) == ["HLA-B75"]
    assert isinstance(result.serotype.dtype, pd.CategoricalDtype)
    assert list(result.serotypes) == ["HLA-B75;HLA-B15"]
    assert set(result.columns) == {"peptide", "serotype", "serotypes"}


@pytest.mark.parametrize("loader", [load_observations, load_binding])
@pytest.mark.parametrize("dtype", ["string", "category"])
@pytest.mark.parametrize("columns", [None, ["peptide", "serotype"], ["peptide"]])
def test_stored_serotypes_follow_current_catalog(tmp_path, monkeypatch, loader, dtype, columns):
    """Refresh removals and additions without losing reported or donor-set provenance (#907)."""
    path = tmp_path / "index.parquet"
    restrictions = [
        "HLA-B*15:112",
        "HLA-A*02:01",
        "HLA-A2",
        "HLA-B*15:112;HLA-A*02:01",
    ]
    frame = pd.DataFrame(
        {
            "peptide": ["REMOVED", "ADDED", "REPORTED", "DONOR"],
            "mhc_restriction": restrictions,
            "serotype": ["HLA-B75", "HLA-A2", "HLA-A2", "HLA-B75"],
            "serotypes": ["HLA-B75;HLA-B15", "HLA-A2", "HLA-A2", "HLA-B75;HLA-B15;HLA-A2"],
            "serotype_source": ["", "", "reported", "donor_set"],
        }
    )
    annotation_columns = ["serotype", "serotypes", "serotype_source"]
    for column in ["mhc_restriction", *annotation_columns]:
        frame[column] = frame[column].astype(dtype)
    frame.assign(
        assay_method="binding assay" if loader is load_binding else "mass spectrometry",
        qualitative_measurement="Positive",
    ).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: path)

    result = loader(columns=columns)
    expected = [curation.resolve_mhc_annotation(value).as_record_fields() for value in restrictions]
    for column in annotation_columns:
        if column in result:
            assert result[column].tolist() == [fields[column] for fields in expected]
            if dtype == "category":
                assert isinstance(result[column].dtype, pd.CategoricalDtype)
    if columns is None:
        assert result.mhc_restriction.tolist() == restrictions
        assert result.serotype_source.tolist() == ["derived", "derived", "reported", "donor_set"]
    else:
        assert result.columns.tolist() == columns

    # Membership is refreshed before filtering, even if its inputs are not projected.
    assert loader(serotype="HLA-B75", columns=columns).empty
    assert loader(serotype="HLA-B15", columns=columns).peptide.tolist() == ["REMOVED", "DONOR"]
    assert loader(serotype="HLA-A2.1", columns=columns).peptide.tolist() == ["ADDED", "DONOR"]
    assert loader(serotype="HLA-A2", columns=columns).peptide.tolist() == [
        "ADDED",
        "REPORTED",
        "DONOR",
    ]
    assert loader(serotype_source="derived", columns=columns).peptide.tolist() == [
        "REMOVED",
        "ADDED",
    ]
    assert loader(serotype_source="reported", columns=columns).peptide.tolist() == ["REPORTED"]
    assert loader(
        serotype="HLA-A2.1", serotype_source=["derived", "donor_set"], columns=columns
    ).peptide.tolist() == ["ADDED", "DONOR"]


def test_serotype_refresh_preserves_missing_restrictions_and_current_categories(
    tmp_path, monkeypatch
):
    path = tmp_path / "index.parquet"
    frame = pd.DataFrame(
        {
            "peptide": ["CURRENT", "BLANK", "MISSING"],
            "mhc_restriction": pd.Categorical(["HLA-A*02:01", "", None]),
            "serotype": pd.Categorical(["HLA-A2", "", None]),
            "serotypes": pd.Categorical(["HLA-A2;HLA-A2.1", "", None]),
            "serotype_source": pd.Categorical(["derived", "", None]),
        }
    )
    frame.assign(assay_method="mass spectrometry", qualitative_measurement="Positive").to_parquet(
        path, index=False
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    result = load_observations(columns=frame.columns.tolist())
    annotation_columns = ["peptide", "serotype", "serotypes", "serotype_source"]
    pd.testing.assert_frame_equal(result[annotation_columns], frame[annotation_columns])
    # The loader's existing spelling normalization converts restriction dtype.
    assert result.mhc_restriction.isna().tolist() == frame.mhc_restriction.isna().tolist()
    assert result.mhc_restriction.fillna("").tolist() == ["HLA-A*02:01", "", ""]


@pytest.mark.parametrize("columns", [None, ["peptide", "mhc_allele_set_size"]])
def test_alias_duplicates_collapse_without_increasing_genotype(tmp_path, monkeypatch, columns):
    path = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK"],
            "mhc_allele_set": ["HLA-B*44:01;HLA-B*44:02"],
            "mhc_allele_set_size": [2],
        }
    ).assign(assay_method="mass spectrometry", qualitative_measurement="Positive").to_parquet(
        path, index=False
    )
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    result = load_observations(columns=columns)
    assert list(result.mhc_allele_set_size) == [1]
    if columns is None:
        assert list(result.mhc_allele_set) == ["HLA-B*44:02"]
    else:
        assert set(result.columns) == set(columns)


@pytest.mark.parametrize("obs_allele", ["HLA-B*44:01", "HLA-B*44:02"])
@pytest.mark.parametrize("sample_allele", ["HLA-B*44:01", "HLA-B*44:02"])
def test_retired_and_current_designations_join_via_allele_exact(
    tmp_path, monkeypatch, obs_allele, sample_allele
):
    from hitlist.export import generate_observations_table

    path = tmp_path / "observations.parquet"
    pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK"],
            "pmid": pd.Series([99999456], dtype="Int64"),
            "mhc_restriction": [obs_allele],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
            "assay_method": ["mass spectrometry"],
            "qualitative_measurement": ["Positive"],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    monkeypatch.setattr(
        "hitlist.export.load_pmid_overrides",
        lambda: {
            99999456: {
                "study_label": "synthetic retired-allele join",
                "ms_samples": [
                    {
                        "sample_label": "B44 sample",
                        "n_samples": 1,
                        "condition": "unperturbed",
                        "mhc_class": "I",
                        "mhc": sample_allele,
                    }
                ],
            }
        },
    )
    out = generate_observations_table()
    assert out["mhc_restriction"].tolist() == [obs_allele]
    assert out["sample_mhc"].tolist() == [sample_allele]
    assert out["sample_attribution"].tolist() == ["allele_exact"]
