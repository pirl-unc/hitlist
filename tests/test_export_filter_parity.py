"""Evidence selection must agree across raw loaders and derived exports."""

import pandas as pd
import pytest

from hitlist.export import (
    generate_binding_table,
    generate_ms_observations_table,
    generate_training_table,
)
from hitlist.observations import load_binding, load_observations


@pytest.fixture
def allele_indexes(tmp_path, monkeypatch):
    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    for kind, filename in (("ms", "observations.parquet"), ("binding", "binding.parquet")):
        pd.DataFrame(
            {
                "peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD"],
                "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA class I"],
                "mhc_allele_set": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01;HLA-B*07:02"],
                "mhc_allele_provenance": ["exact", "exact", "sample_allele_match"],
                "mhc_class": ["I"] * 3,
                "mhc_species": ["Homo sapiens"] * 3,
                "pmid": pd.array([99999999] * 3, dtype="Int64"),
                "source": ["iedb"] * 3,
                "assay_iri": [f"{kind}:{i}" for i in range(3)],
            }
        ).to_parquet(tmp_path / filename, index=False)


@pytest.mark.parametrize(
    "loader,exporter",
    [(load_observations, generate_ms_observations_table), (load_binding, generate_binding_table)],
)
@pytest.mark.parametrize("allele", ["HLA-A*02:01", "A*02:01", "hla-a*02:01", ["A*02:01"]])
@pytest.mark.parametrize("provenance", [None, "exact", "sample_allele_match"])
def test_export_allele_filters_match_loader(allele_indexes, loader, exporter, allele, provenance):
    kwargs = {"mhc_allele_in_set": allele, "mhc_allele_provenance": provenance}
    raw = loader(**kwargs)
    result = exporter(**kwargs)
    assert not raw.empty
    assert list(result["assay_iri"]) == list(raw["assay_iri"])


@pytest.mark.parametrize("mode", ["ms", "binding", "both"])
@pytest.mark.parametrize(
    "alleles", [["A*02:01", "B*07:02"], "A*02:01,B*07:02", ["A*02:01,B*07:02"]]
)
def test_training_allele_filter_input_forms(allele_indexes, mode, alleles):
    result = generate_training_table(
        include_evidence=mode, mhc_allele_in_set=alleles, mhc_allele_provenance="exact"
    )
    assert set(result["peptide"]) == {"AAAAAAAAA", "CCCCCCCCC"}
    assert len(result) == (4 if mode == "both" else 2)
    assert result["evidence_row_id"].is_unique


@pytest.mark.parametrize(
    "exporter", [generate_ms_observations_table, generate_binding_table, generate_training_table]
)
@pytest.mark.parametrize("alleles", ["", [], [""]])
def test_export_rejects_empty_allele_filter(allele_indexes, exporter, alleles):
    with pytest.raises(ValueError, match="mhc_allele_in_set filter received no usable"):
        exporter(mhc_allele_in_set=alleles)
