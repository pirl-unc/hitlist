"""Public membership filters share scalar/list/empty contracts (#913)."""

import pandas as pd
import pytest

from hitlist import downloads, mappings, observations

LOADERS = [
    observations.load_observations,
    observations.load_binding,
    observations.load_other_assays,
]
FILTER_VALUES = {
    "peptide": "ACDEFGHIK",
    "mhc_restriction": "HLA-A*02:01",
    "mhc_allele_in_set": "HLA-A*02:01",
    "mhc_allele_provenance": "exact",
    "restriction_evidence": "experimental",
    "serotype": "HLA-A2",
    "serotype_source": "derived",
    "source_species": "human",
    "host_species": "human",
    "gene_name": "PRAME",
    "gene_id": "ENSG1",
}
MAPPING_VALUES = {
    "peptide": "ACDEFGHIK",
    "gene_name": "PRAME",
    "gene_id": "ENSG1",
    "protein_id": "P1",
    "transcript_id": "ENST1",
    "gene_biotype": "protein_coding",
    "proteome": "human",
}


@pytest.fixture(params=["string", "category"])
def indexes(tmp_path, monkeypatch, request):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK", "KLMNPQRST"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B7"],
            "mhc_class": ["I", "I"],
            "mhc_allele_set": ["HLA-A*02:01", "HLA-B*07:02"],
            "mhc_allele_provenance": ["exact", "sample_allele_match"],
            "restriction_evidence": ["experimental", "predicted"],
            "serotypes": ["HLA-A2;HLA-A2.1", "HLA-B7"],
            "serotype_source": ["derived", "reported"],
            "source_organism": ["human", "mouse"],
            "host": ["human", "mouse"],
            "mhc_species": ["Homo sapiens", "Homo sapiens"],
            "qualitative_measurement": ["Positive", "Positive"],
        }
    ).astype(request.param)
    for filename, method in (
        ("observations.parquet", "mass spectrometry"),
        ("binding.parquet", "binding assay"),
        ("other_assays.parquet", "x-ray crystallography"),
    ):
        frame.assign(assay_method=method).to_parquet(tmp_path / filename, index=False)
    mapping = pd.DataFrame({key: [value, "OTHER"] for key, value in MAPPING_VALUES.items()})
    mapping["peptide"] = frame.peptide
    mapping.astype(request.param).to_parquet(tmp_path / "peptide_mappings.parquet", index=False)
    return tmp_path, frame


def equivalent_forms(value):
    return [value, f" {value}, ", [f" {value} "], [f",{value}, ", " "]]


@pytest.mark.parametrize("loader", LOADERS)
def test_membership_filter_equivalent_forms(indexes, loader):
    for name, value in FILTER_VALUES.items():
        expected = loader(**{name: value})
        assert expected.peptide.tolist() == ["ACDEFGHIK"], name
        for selection in equivalent_forms(value):
            for columns in (None, ["peptide"]):
                actual = loader(**{name: selection}, columns=columns)
                pd.testing.assert_frame_equal(
                    actual, expected if columns is None else expected[columns]
                )


@pytest.mark.parametrize("loader", LOADERS)
def test_membership_filter_empty_and_none(indexes, loader):
    for name in FILTER_VALUES:
        assert len(loader(**{name: None}, columns=["peptide"])) == 2
        for selection in ([], "", [" , "]):
            if name in {"mhc_restriction", "mhc_allele_in_set"}:
                with pytest.raises(ValueError, match=f"{name} filter received no usable"):
                    loader(**{name: selection}, peptide=[])
            else:
                result = loader(**{name: selection}, columns=["peptide"])
                assert result.empty, name
                assert result.columns.tolist() == ["peptide"], name


@pytest.mark.parametrize("loader", LOADERS)
def test_membership_filter_invalid_elements_are_public_errors(indexes, loader):
    for name in FILTER_VALUES:
        for selection in ([None], [0], [pd.NA], 3):
            with pytest.raises(TypeError, match=name):
                loader(**{name: selection})


@pytest.mark.parametrize("loader", LOADERS)
def test_candidate_filter_on_old_schema_does_not_silently_broaden(indexes, loader):
    path, _ = indexes
    for filename in ("observations.parquet", "binding.parquet", "other_assays.parquet"):
        stored = pd.read_parquet(path / filename).drop(columns="mhc_allele_set")
        stored.to_parquet(path / filename, index=False)
    with pytest.raises(ValueError, match=r"mhc_allele_set.*rebuild"):
        loader(mhc_allele_in_set="HLA-A*02:01", columns=["peptide"])


def test_mapping_filter_forms_and_empty_values(indexes):
    for name, value in MAPPING_VALUES.items():
        for selection in equivalent_forms(value):
            result = mappings.load_peptide_mappings(**{name: selection}, columns=["peptide"])
            assert result.peptide.tolist() == ["ACDEFGHIK"], name
        for selection in ([], "", [" , "]):
            result = mappings.load_peptide_mappings(**{name: selection}, columns=["peptide"])
            assert result.empty, name
        with pytest.raises(TypeError, match=name):
            mappings.load_peptide_mappings(**{name: [None]})


@pytest.mark.parametrize("command", ["ms", "binding"])
@pytest.mark.parametrize(
    "option,value",
    [
        ("mhc-allele-provenance", "exact"),
        ("restriction-evidence", "experimental"),
        ("serotype-source", "derived"),
    ],
)
def test_cli_vocabulary_filters_normalize_and_still_reject_typos(
    monkeypatch, command, option, value
):
    from hitlist.cli import main

    captured = []
    monkeypatch.setattr("hitlist.cli_export._export", lambda args: captured.append(args))
    monkeypatch.setattr(
        "sys.argv",
        ["hitlist", "export", command, f"--{option}", f" {value}, ", f"--{option}", value],
    )
    main()
    assert getattr(captured[-1], option.replace("-", "_")) == [value, value]
    monkeypatch.setattr("sys.argv", ["hitlist", "export", command, f"--{option}", f"{value},typo"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "command,loader",
    [("ms", observations.load_observations), ("binding", observations.load_binding)],
)
def test_cli_free_text_filters_reach_loader(indexes, monkeypatch, command, loader):
    from hitlist.cli import main

    captured = []

    def export(args):
        captured.append(
            loader(mhc_restriction=args.mhc_allele, gene_name=args.gene_name, columns=["peptide"])
        )

    monkeypatch.setattr("hitlist.cli_export._export", export)
    monkeypatch.setattr(
        "sys.argv",
        ["hitlist", "export", command, "--mhc-allele", " HLA-A*02:01, ", "--gene-name", " PRAME, "],
    )
    main()
    assert captured[0].peptide.tolist() == ["ACDEFGHIK"]
