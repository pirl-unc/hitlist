"""Mixed-endpoint binding selection through the unified API and real CLI (#615)."""

import sys

import pandas as pd
import pytest

from hitlist import cli, downloads, export
from hitlist.training_bundle import verify_training_bundle

IC50 = "half maximal inhibitory concentration (IC50)"
KD = "dissociation constant KD"
AFFINITY_FILTERS = {
    "assay_method": "PURIFIED",
    "response_measured": [IC50, KD.upper()],
    "measurement_units": "nM",
    "has_quantitative_value": True,
    "quantitative_value_min": 12.5,
    "quantitative_value_max": 5000.0,
}


@pytest.fixture
def endpoint_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    rows = pd.DataFrame(
        {
            "peptide": [letter * 9 for letter in "ACDEFGHIKL"],
            "mhc_restriction": ["HLA-A*02:01"] * 10,
            "mhc_class": ["I"] * 10,
            "mhc_species": ["Homo sapiens"] * 10,
            "pmid": pd.array([99999999] * 10, dtype="Int64"),
            "source": ["iedb"] * 10,
            "assay_iri": [f"http://iedb.org/assay/{i}" for i in range(10)],
            "assay_method": ["purified MHC/direct/fluorescence"] * 9 + ["cellular MHC/direct"],
            "response_measured": [
                IC50,
                IC50,
                IC50,
                KD,
                "half maximal effective concentration (EC50)",
                "half life",
                "3D structure",
                IC50,
                IC50,
                "qualitative binding",
            ],
            "measurement_units": ["nM"] * 5 + ["min", "angstroms", "log10(nM)", "nM", ""],
            "quantitative_value": [12.5, 50.0, 5000.0, 100.0, 30.0, 12.5, 12.5, 1.1, None, None],
            "quantitative_measurement": [
                "12.5",
                "50",
                "5000",
                "100",
                "30",
                "12.5",
                "12.5",
                "1.1",
                "",
                "",
            ],
            "measurement_inequality": ["=", "<", ">", "=", "=", "=", "=", "=", "", ""],
            "qualitative_measurement": ["Positive-High", "Positive", "Negative"] + ["Positive"] * 7,
        }
    )
    rows.to_parquet(tmp_path / "binding.parquet", index=False)
    rows.iloc[[0]].assign(
        peptide="MMMMMMMMM",
        assay_iri="http://iedb.org/assay/100",
        assay_method="cellular MHC/mass spectrometry",
        response_measured="ligand presentation",
        measurement_units="",
        quantitative_value=float("nan"),
        quantitative_measurement="",
        measurement_inequality="",
    ).to_parquet(tmp_path / "observations.parquet", index=False)
    return rows


@pytest.mark.parametrize("mode", ["binding", "both", "ms"])
def test_training_binding_endpoint_selection_preserves_measurements(endpoint_indexes, mode):
    result = export.generate_training_table(
        include_evidence=mode, mhc_class="I", species="human", **AFFINITY_FILTERS
    )
    binding = result[result.evidence_kind.eq("binding")]
    expected = endpoint_indexes.iloc[:4] if mode != "ms" else endpoint_indexes.iloc[:0]
    columns = [
        "assay_iri",
        "response_measured",
        "measurement_units",
        "quantitative_value",
        "quantitative_measurement",
        "measurement_inequality",
        "qualitative_measurement",
    ]
    pd.testing.assert_frame_equal(
        binding[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert list(result[result.evidence_kind.eq("ms")].peptide) == (
        ["MMMMMMMMM"] if mode != "binding" else []
    )


@pytest.mark.parametrize(
    "filters,selected",
    [
        ({}, "ACDEFGIKL"),
        ({"assay_method": "CELLULAR"}, "L"),
        ({"response_measured": KD.upper()}, "E"),
        ({"response_measured": [IC50, KD]}, "ACDEIK"),
        ({"measurement_units": "NM"}, "ACDEFK"),
        ({"measurement_units": ["nM", "min"]}, "ACDEFGK"),
        ({"has_quantitative_value": True}, "ACDEFGI"),
        ({"has_quantitative_value": False}, "KL"),
        ({"quantitative_value_min": 100.0}, "DE"),
        ({"quantitative_value_max": 12.5}, "AGI"),
        ({"response_measured": "IC50"}, ""),
    ],
)
def test_training_binding_filters_match_existing_semantics(endpoint_indexes, filters, selected):
    result = export.generate_training_table(include_evidence="binding", **filters)
    assert list(result.peptide) == [letter * 9 for letter in selected]


def test_training_binding_filters_precede_mapping_and_projection(endpoint_indexes, monkeypatch):
    monkeypatch.setattr(
        export,
        "_load_training_mappings_for_peptides",
        lambda *a, **kw: pd.DataFrame({"peptide": ["AAAAAAAAA"] * 2, "protein_id": ["P1", "P2"]}),
    )
    result = export.generate_training_table(
        include_evidence="binding",
        **AFFINITY_FILTERS,
        map_source_proteins=True,
        columns=["peptide", "protein_id", "measurement_inequality"],
    )
    assert list(result.peptide) == ["AAAAAAAAA", "AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]
    assert list(result.measurement_inequality) == ["=", "=", "<", ">", "="]
    assert result.evidence_row_id.nunique() == 4


@pytest.mark.parametrize("destination", ["csv", "bundle"])
def test_training_binding_cli_and_bundle(endpoint_indexes, tmp_path, monkeypatch, destination):
    output = tmp_path / ("export.csv" if destination == "csv" else "bundle")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "export",
            "training",
            "--include-evidence",
            "both",
            "--class",
            "I",
            "--species",
            "human",
            "--assay-method",
            "PURIFIED",
            "--response-measured",
            IC50,
            "--response-measured",
            KD.upper(),
            "--measurement-units",
            "nM",
            "--has-quantitative-value",
            "--quantitative-value-min",
            "12.5",
            "--quantitative-value-max",
            "5000",
            "--output" if destination == "csv" else "--bundle",
            str(output),
        ],
    )
    cli.main()
    if destination == "csv":
        result = pd.read_csv(output, keep_default_na=False)
    else:
        result = pd.read_parquet(output / "training.parquet")
        manifest = verify_training_bundle(output)
        for key, value in AFFINITY_FILTERS.items():
            assert manifest["training_options"][key] == (
                [value] if key in {"assay_method", "measurement_units"} else value
            )
    assert list(result.peptide) == ["MMMMMMMMM", "AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]
    assert list(result.measurement_inequality) == ["", "=", "<", ">", "="]


@pytest.mark.parametrize("command", ["binding", "training"])
def test_training_binding_cli_qualitative_only(endpoint_indexes, tmp_path, monkeypatch, command):
    output = tmp_path / "qualitative.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "export",
            command,
            "--qualitative-only",
            "-o",
            str(output),
        ],
    )
    cli.main()
    result = pd.read_csv(output)
    assert list(result.peptide) == (["MMMMMMMMM"] if command == "training" else []) + [
        "KKKKKKKKK",
        "LLLLLLLLL",
    ]
