import pytest

from hitlist.observations import (
    binding_path,
    is_binding_built,
    is_built,
    load_binding,
    load_observations,
    observations_path,
)


def test_observations_path():
    p = observations_path()
    assert p.name == "observations.parquet"


def test_binding_path():
    p = binding_path()
    assert p.name == "binding.parquet"
    assert p.parent == observations_path().parent


def test_is_binding_built_bool():
    assert isinstance(is_binding_built(), bool)


def test_is_built_false_initially():
    # May or may not be built depending on test environment
    # Just verify the function returns a bool
    assert isinstance(is_built(), bool)


def test_load_observations_not_built():
    if not is_built():
        with pytest.raises(FileNotFoundError, match="not built"):
            load_observations()


def test_load_observations_if_built():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations()
    assert len(df) > 0
    assert "peptide" in df.columns
    assert "mhc_restriction" in df.columns
    assert "mhc_class" in df.columns
    assert "src_cancer" in df.columns
    assert "source" in df.columns


def test_load_observations_class_filter():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(mhc_class="I")
    assert len(df) > 0
    assert (df["mhc_class"] == "I").all()


def test_load_observations_species_filter():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(species="Homo sapiens")
    assert len(df) > 0
    assert (df["mhc_species"] == "Homo sapiens").all()


def test_load_observations_column_select():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(columns=["peptide", "mhc_restriction"])
    assert set(df.columns) == {"peptide", "mhc_restriction"}


def test_load_binding_not_built():
    if not is_binding_built():
        with pytest.raises(FileNotFoundError, match="not built"):
            load_binding()


def test_load_observations_never_contains_binding_rows(tmp_path, monkeypatch):
    """Smoke test: the MS parquet and binding parquet never overlap by IRI.

    Builds tiny fake fixtures, points paths at tmp_path, and confirms the
    two loaders return disjoint row sets.
    """
    import pandas as pd

    ms = pd.DataFrame(
        {
            "peptide": ["AAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["ms-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_binding_assay": [False],
        }
    )
    bd = pd.DataFrame(
        {
            "peptide": ["BBB"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["bd-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_binding_assay": [True],
        }
    )
    obs_p = tmp_path / "observations.parquet"
    bd_p = tmp_path / "binding.parquet"
    ms.to_parquet(obs_p, index=False)
    bd.to_parquet(bd_p, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_p)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_p)

    ms_df = load_observations()
    bd_df = load_binding()
    assert set(ms_df["peptide"]) == {"AAA"}
    assert set(bd_df["peptide"]) == {"BBB"}
    assert not ms_df["is_binding_assay"].any()
    assert bd_df["is_binding_assay"].all()


def test_load_binding_filters(tmp_path, monkeypatch):
    """Binding loader respects the same filters as the observations loader."""
    import pandas as pd

    bd = pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "II"],
            "reference_iri": ["b1", "b2", "b3"],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["iedb", "cedar", "iedb"],
            "mhc_species": ["Homo sapiens"] * 3,
            "is_binding_assay": [True] * 3,
            "serotypes": ["HLA-A2", "HLA-B7", "HLA-A2"],
        }
    )
    bd_p = tmp_path / "binding.parquet"
    bd.to_parquet(bd_p, index=False)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_p)

    assert len(load_binding()) == 3
    assert len(load_binding(mhc_class="I")) == 2
    assert len(load_binding(source="cedar")) == 1
    assert list(load_binding(mhc_restriction="HLA-A*02:01")["peptide"]) == ["AAA", "CCC"]
    assert set(load_binding(serotype="A2")["peptide"]) == {"AAA", "CCC"}
