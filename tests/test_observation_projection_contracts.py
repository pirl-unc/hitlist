"""Projection and no-hit queries preserve public derived-data contracts (#912/#914)."""

import pandas as pd
import pytest

from hitlist import downloads, mappings, observations

LOADERS = [
    observations.load_observations,
    observations.load_binding,
    observations.load_other_assays,
]


def write_index(tmp_path, monkeypatch, loader, frame):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    filename, method = {
        observations.load_observations: ("observations.parquet", "mass spectrometry"),
        observations.load_binding: ("binding.parquet", "binding assay"),
        observations.load_other_assays: ("other_assays.parquet", "x-ray crystallography"),
    }[loader]
    frame.assign(assay_method=method, qualitative_measurement="Positive").to_parquet(
        tmp_path / filename, index=False
    )


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("dtype", ["string", "category"])
@pytest.mark.parametrize("stored", [False, True])
def test_projected_quality_labels_refresh_like_full_reads(
    tmp_path, monkeypatch, loader, dtype, stored
):
    frame = pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNP", "ACDEFGH", "ACDE"],
            "mhc_class": pd.Series(["I", "II", "I", "II"], dtype=dtype),
            "mhc_restriction": ["HLA-A*02:01", "HLA-DRB1*01:01", "HLA-A*02:01", "HLA-DRB1*01:01"],
        }
    )
    if stored:
        frame["mhc_class_label_severity"] = pd.Series(
            ["implausible", "implausible", "ok", "ok"], dtype=dtype
        )
        frame["mhc_class_label_suspect"] = pd.Series(
            [True, True, False, False], dtype="category" if dtype == "category" else bool
        )
    write_index(tmp_path, monkeypatch, loader, frame)
    full = loader()
    assert full.mhc_class_label_severity.tolist() == ["ok", "ok", "implausible", "implausible"]
    assert full.mhc_class_label_suspect.tolist() == [False, False, True, True]
    for columns in (
        ["mhc_class_label_severity"],
        ["mhc_class_label_suspect"],
        ["peptide", "mhc_class_label_severity", "mhc_class_label_suspect"],
    ):
        projected = loader(columns=columns)
        pd.testing.assert_frame_equal(projected, full[columns])
        for flag in ("exclude_class_label_suspect", "exclude_class_label_implausible"):
            filtered = loader(columns=columns, **{flag: True})
            pd.testing.assert_frame_equal(filtered, full.loc[[0, 1], columns])


DERIVED_DTYPES = {
    "mhc_class_label_severity": "object",
    "mhc_class_label_suspect": "bool",
    "is_non_peptide_ligand": "bool",
    "host_organism": "str",
    "source_species": "str",
    "is_chimeric": "bool",
    "is_engineered_mhc": "bool",
    "xenograft": "bool",
    "gene_names": "str",
    "gene_ids": "str",
    "protein_ids": "str",
    "n_source_proteins": "int64",
    "n_source_genes": "int64",
    "is_binding_assay": "bool",
    "is_ms_observation": "bool",
    "ms_outcome_status": "str",
}


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("dtype", ["string", "category"])
@pytest.mark.parametrize(
    "empty_via",
    [
        "stored",
        "restriction",
        "gene",
        "peptide",
        "source",
        "serotype",
        "serotype_source",
        "quality",
        "length",
        "source_species",
    ],
)
def test_empty_queries_keep_typed_derived_projection(
    tmp_path, monkeypatch, loader, dtype, empty_via
):
    frame = pd.DataFrame(
        {
            "peptide": ["ACDEFGH" if empty_via == "quality" else "ACDEFGHIK"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
            "host": ["human"],
            "source_organism": ["human"],
            "source": ["iedb"],
            "serotypes": ["HLA-A2;HLA-A2.1"],
            "serotype_source": ["derived"],
        }
    ).astype(dtype)
    if empty_via == "stored":
        frame = frame.iloc[:0]
    write_index(tmp_path, monkeypatch, loader, frame)
    calls = []

    def mapping_lookup(**kwargs):
        calls.append(kwargs)
        assert kwargs == {"columns": ["peptide"], "gene_name": ["MISSING"]}
        return pd.DataFrame({"peptide": pd.Series(dtype=str)})

    monkeypatch.setattr(mappings, "is_mappings_built", lambda: True)
    monkeypatch.setattr(mappings, "load_peptide_mappings", mapping_lookup)
    filters = {
        "stored": {},
        "restriction": {"mhc_restriction": "HLA-B*07:02"},
        "gene": {"gene_name": "MISSING"},
        "peptide": {"peptide": []},
        "source": {"source": "MISSING"},
        "serotype": {"serotype": "HLA-B75"},
        "serotype_source": {"serotype_source": "reported"},
        "quality": {"exclude_class_label_suspect": True},
        "length": {"length_min": 100},
        "source_species": {"source_species": "mouse"},
    }[empty_via]
    columns = ["peptide", *DERIVED_DTYPES, "unsupported_source_field"]
    result = loader(columns=columns, **filters)
    assert result.empty
    assert result.columns.tolist() == columns[:-1]
    # Arrow may discard unused dictionary values when it skips all row groups;
    # preserve the dtype family without requiring unused category vocabulary.
    if dtype == "category":
        assert isinstance(result.peptide.dtype, pd.CategoricalDtype)
    else:
        assert result.peptide.dtype == frame.peptide.dtype
    for column, expected_dtype in DERIVED_DTYPES.items():
        assert result[column].dtype == pd.Series(dtype=expected_dtype).dtype, column
    assert len(calls) == (1 if empty_via == "gene" else 0)


@pytest.mark.parametrize("loader", LOADERS)
def test_no_hit_restriction_has_same_full_schema_as_empty_peptide(tmp_path, monkeypatch, loader):
    frame = pd.DataFrame(
        {"peptide": ["ACDEFGHIK"], "mhc_restriction": ["HLA-A*02:01"], "mhc_class": ["I"]}
    )
    write_index(tmp_path, monkeypatch, loader, frame)
    expected = loader(peptide=[])
    result = loader(mhc_restriction="HLA-B*07:02")
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("loader", LOADERS)
def test_empty_legacy_projection_does_not_need_missing_inputs(tmp_path, monkeypatch, loader):
    frame = pd.DataFrame({"peptide": pd.Series(dtype="string")})
    write_index(tmp_path, monkeypatch, loader, frame)
    monkeypatch.setattr(mappings, "is_mappings_built", lambda: False)
    result = loader(columns=[*DERIVED_DTYPES, "unsupported_source_field"])
    assert result.empty
    assert result.columns.tolist() == list(DERIVED_DTYPES)
    for column, expected_dtype in DERIVED_DTYPES.items():
        assert result[column].dtype == pd.Series(dtype=expected_dtype).dtype, column


@pytest.mark.parametrize("loader", LOADERS)
def test_empty_projection_preserves_stored_annotation_dtypes(tmp_path, monkeypatch, loader):
    frame = pd.DataFrame(
        {
            "peptide": pd.Series(dtype="string"),
            "gene_names": pd.Series(dtype="string"),
            "n_source_genes": pd.Series(dtype="Int32"),
            "source_score": pd.Series(dtype="float32"),
        }
    )
    write_index(tmp_path, monkeypatch, loader, frame)
    result = loader(columns=list(frame.columns))
    pd.testing.assert_frame_equal(result, frame)
