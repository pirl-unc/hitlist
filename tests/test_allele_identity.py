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
    assert curation.allele_to_all_serotypes("HLA-B*44:01") == ("HLA-B44", "HLA-Bw4")
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
    # The catalog reports B15 under old B*15:112; the current name has B75.
    # Both memberships survive, with the current specific assignment preferred.
    # B75 is a B15 split in the primary HLA nomenclature broad/split table.
    assert curation.allele_to_all_serotypes("HLA-B*15:112") == ("HLA-B75", "HLA-B15")
    assert curation.allele_to_all_serotypes("HLA-B*15:11") == ("HLA-B75", "HLA-B15")
    assert curation.allele_to_serotype("HLA-B*15:11") == "HLA-B75"
    assert "HLA-B*15:11" in curation.serotype_to_alleles("HLA-B15")


def test_stored_current_name_refreshes_retired_catalog_membership(tmp_path, monkeypatch):
    path = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK"],
            "mhc_restriction": ["HLA-B*15:11"],
            "serotypes": ["HLA-B75"],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    result = load_observations(serotype="HLA-B15", columns=["peptide", "serotypes"])
    assert list(result.peptide) == ["ACDEFGHIK"]
    assert list(result.serotypes) == ["HLA-B75;HLA-B15"]
    assert set(result.columns) == {"peptide", "serotypes"}


@pytest.mark.parametrize("columns", [None, ["peptide", "mhc_allele_set_size"]])
def test_alias_duplicates_collapse_without_increasing_genotype(tmp_path, monkeypatch, columns):
    path = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK"],
            "mhc_allele_set": ["HLA-B*44:01;HLA-B*44:02"],
            "mhc_allele_set_size": [2],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    result = load_observations(columns=columns)
    assert list(result.mhc_allele_set_size) == [1]
    if columns is None:
        assert list(result.mhc_allele_set) == ["HLA-B*44:02"]
    else:
        assert set(result.columns) == set(columns)
