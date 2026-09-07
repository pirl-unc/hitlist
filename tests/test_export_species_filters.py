"""Species filters select the same evidence across loaders, exports, and CLI."""

from collections import Counter

import pandas as pd
import pytest

from hitlist import cli, downloads, export
from hitlist.observations import load_binding, load_observations


@pytest.fixture
def species_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    # Distinct source, MHC, and host axes, plus unknowns and the source fallback.
    rows = pd.DataFrame(
        {
            "peptide": [letter * 9 for letter in "ACDEFG"],
            "mhc_restriction": ["HLA-A*02:01"] * 6,
            "mhc_class": ["I"] * 6,
            "mhc_species": ["Homo sapiens"] * 6,
            "source_organism": ["Homo sapiens", "Mus musculus", "", None, "", "Homo sapiens"],
            "species": ["Homo sapiens", "Homo sapiens", "Mus musculus", "Mus musculus", "", ""],
            "host": ["Homo sapiens", "Mus musculus", "Homo sapiens", "", "", "Mus musculus"],
            "pmid": pd.array([99999999] * 6, dtype="Int64"),
            "source": ["iedb"] * 6,
        }
    )
    for kind, filename in (("ms", "observations.parquet"), ("binding", "binding.parquet")):
        rows.assign(assay_iri=[f"{kind}:{i}" for i in range(6)]).to_parquet(
            tmp_path / filename, index=False
        )
    return rows


FILTER_CASES = [
    {},
    {"source_species": "mouse"},
    {"host_species": "mouse"},
    {"exclude_chimeric": True},
    {"species": "human", "source_species": "mouse", "host_species": "human"},
    {"source_species": "Homo sapiens", "exclude_chimeric": True},
    {"source_species": "mouse", "host_species": "human", "exclude_chimeric": True},
    {"host_species": "rat"},
    {"source_species": ["mouse", "human"]},
    {"source_species": []},
]


@pytest.mark.parametrize("filters", FILTER_CASES)
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize(
    "loader,exporter",
    [
        (load_observations, export.generate_observations_table),
        (load_observations, export.generate_ms_observations_table),
        (load_binding, export.generate_binding_table),
    ],
)
def test_species_export_matches_raw_evidence(species_indexes, filters, projected, loader, exporter):
    raw = loader(**filters)
    result = exporter(**filters, columns=["assay_iri"] if projected else None)
    assert Counter(result["assay_iri"]) == Counter(raw["assay_iri"])
    if projected:
        assert list(result.columns) == ["assay_iri"]
    if not filters:
        assert len(result) == 6


@pytest.mark.parametrize("filters", FILTER_CASES)
@pytest.mark.parametrize("mode", ["ms", "binding", "both"])
def test_training_species_filters_preserve_evidence(species_indexes, filters, mode):
    expected = []
    if mode in {"ms", "both"}:
        expected.extend(load_observations(**filters)["assay_iri"])
    if mode in {"binding", "both"}:
        expected.extend(load_binding(**filters)["assay_iri"])
    result = export.generate_training_table(include_evidence=mode, **filters)
    assert Counter(result["assay_iri"]) == Counter(expected)
    assert result["evidence_row_id"].is_unique


@pytest.mark.parametrize("loader", [load_observations, load_binding])
def test_chimeric_exclusion_uses_source_species_fallback(species_indexes, loader):
    raw = loader(source_species="mouse")
    assert set(raw["peptide"]) == {"CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"}
    assert raw["is_chimeric"].all()
    assert loader(source_species="mouse", exclude_chimeric=True).empty


@pytest.mark.parametrize(
    "exporter",
    [
        export.generate_ms_observations_table,
        export.generate_binding_table,
        export.generate_training_table,
    ],
)
def test_export_species_flags_match_selection(species_indexes, exporter):
    result = exporter(source_species="mouse")
    assert result["is_chimeric"].all()
    assert exporter(source_species="mouse", exclude_chimeric=True).empty


@pytest.mark.parametrize("filters", FILTER_CASES)
def test_peptide_summary_species_filters_preserve_support(species_indexes, filters):
    result = export.generate_ms_peptide_summary_table(
        mhc_allele="A*02:01", peptide=species_indexes["peptide"].tolist(), **filters
    )
    raw = load_observations(**filters)
    assert set(result["peptide"]) == set(raw["peptide"])
    assert result["n_support_rows"].sum() == len(raw)


@pytest.mark.parametrize("command", ["ms", "binding", "training", "peptide-summary"])
def test_cli_species_filters_select_expected_rows(species_indexes, command, tmp_path, monkeypatch):
    output = tmp_path / "export.csv"
    args = [
        "hitlist",
        "export",
        command,
        "--species",
        "human",
        "--source-species",
        "human",
        "--host-species",
        "human",
        "--exclude-chimeric",
        "--output",
        str(output),
    ]
    if command == "peptide-summary":
        args.extend(["--mhc-allele", "A*02:01", "--peptide", *species_indexes["peptide"]])
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    result = pd.read_csv(output)
    assert set(result["peptide"]) == {"AAAAAAAAA"}
    assert len(result) == (2 if command == "training" else 1)
