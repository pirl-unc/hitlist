import pytest

from hitlist.observations import is_built, load_observations, observations_path


def test_observations_path():
    p = observations_path()
    assert p.name == "observations.parquet"


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
