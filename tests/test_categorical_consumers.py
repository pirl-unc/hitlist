"""Public readers must accept categorical strings without a blank category (#605)."""

import pandas as pd
import pytest

from hitlist import downloads, export, observations, predict, qc, report


@pytest.fixture
def categorical_frame():
    return pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "CCCCCCCCC"],
            "pmid": [99999001, 99999001],
            "mhc_class": ["I", "I"],
            "mhc_restriction": pd.Categorical(["HLA-A*02:01", None]),
            "host": pd.Categorical(["Homo sapiens", None]),
            "mhc_species": pd.Categorical(["Homo sapiens", None]),
            "source_organism": pd.Categorical(["Homo sapiens", None]),
            "species": ["Homo sapiens", ""],
            "source": ["iedb", "iedb"],
            "is_monoallelic": [False, False],
            "src_cancer": [True, True],
            "src_healthy_tissue": [False, False],
            "src_healthy_cell_line": [False, False],
            "sample_label": ["sample", "sample"],
            "sample_mhc_origin": ["sample", "sample"],
            "sample_mhc": ["HLA-A*02:01", "HLA-A*02:01"],
            "cell_type": ["cell_line", "cell_line"],
            "disease": pd.Categorical(["melanoma", None]),
            "cell_line_name": pd.Categorical(["JY", None]),
        }
    )


def test_species_axes_handle_missing_categorical_values(categorical_frame):
    observations._attach_species_axes(categorical_frame)
    assert list(categorical_frame.host_organism) == ["Homo sapiens", ""]
    assert list(categorical_frame.source_species) == ["Homo sapiens", ""]
    assert not categorical_frame.is_chimeric.any()


def test_proteome_qc_handles_categories_without_blank(categorical_frame, monkeypatch):
    monkeypatch.setattr(observations, "is_built", lambda: True)
    monkeypatch.setattr(observations, "load_observations", lambda **kw: categorical_frame.copy())
    result = qc.proteome_coverage(min_rows=1)
    assert dict(zip(result.source_organism, result.n_rows)) == {"Homo sapiens": 1, "": 1}


@pytest.mark.parametrize("missing_name", [None, ""])
def test_sample_discrepancies_handle_categories_without_placeholder(
    categorical_frame, monkeypatch, missing_name
):
    categorical_frame["cell_name"] = pd.Categorical(["JY", missing_name])
    categorical_frame["mhc_class_label_suspect"] = [False, False]
    categorical_frame["mhc_class_label_severity"] = ["ok", "ok"]
    monkeypatch.setattr(observations, "is_built", lambda: True)
    monkeypatch.setattr(observations, "load_observations", lambda **kw: categorical_frame.copy())
    result = qc.discrepancies(min_rows=1, by="sample")
    assert dict(zip(result.cell_name, result.n_rows)) == {"JY": 1, "(no cell_name)": 1}


def test_prediction_selection_handles_categories_without_blank(categorical_frame, monkeypatch):
    monkeypatch.setattr(
        export, "generate_observations_table", lambda **kw: categorical_frame.copy()
    )
    assert predict.reassign_class_only_alleles().empty


def test_report_handles_categories_without_blank(categorical_frame):
    result = report.generate_report(categorical_frame)
    assert "melanoma" in result
    assert "JY" in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("assay_method", "purified"),
        ("response_measured", "MHC binding"),
        ("measurement_units", "nM"),
    ],
)
def test_binding_filters_accept_categories_without_blank(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "CCCCCCCCC"],
            "pmid": [99999001, 99999001],
            "mhc_restriction": ["HLA-A*02:01"] * 2,
            "mhc_class": ["I"] * 2,
            "mhc_species": ["Homo sapiens"] * 2,
            field: pd.Categorical([value, None]),
        }
    )
    frame.to_parquet(tmp_path / "binding.parquet", index=False)
    result = export.generate_binding_table(**{field: value})
    assert list(result.peptide) == ["AAAAAAAAA"]


@pytest.mark.parametrize("values", [["a", "b"], ["a", None], [None, None], []])
def test_shared_fill_preserves_categories_index_and_input(values):
    from hitlist.pandas_utils import fillna_scalar_safe

    original = pd.Series(
        pd.Categorical(values, categories=["a", "b"]),
        index=range(10, 10 + len(values)),
        name="labels",
    )
    before = original.copy()
    result = fillna_scalar_safe(original, "")
    assert list(result) == ["" if value is None else value for value in values]
    assert isinstance(result.dtype, pd.CategoricalDtype)
    assert "" in result.cat.categories
    assert result.index.equals(original.index)
    assert result.name == original.name
    pd.testing.assert_series_equal(original, before)
