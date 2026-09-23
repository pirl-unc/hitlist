"""Old derived donor labels must obey source-cohort scope too (#534)."""

import pandas as pd
import pytest

from hitlist.observations import load_binding, load_observations


def _stale_rows():
    monoallelic = {
        "pmid": 31844290,
        "peptide": "SLLQHLIGL",
        "assay_iri": "http://www.iedb.org/assay/534",
        "mhc_restriction": "HLA-G*01:01",
        "mhc_class": "non-classical",
        "mhc_species": "Homo sapiens",
        "mhc_allele_provenance": "exact",
        "source": "iedb",
    }
    return [
        {**monoallelic, "attributed_sample_label": "MEL3 (13240-006)"},
        {**monoallelic, "attributed_sample_label": "GBM9 (H4198 BT187)"},
        {
            **monoallelic,
            "assay_iri": "http://www.iedb.org/assay/535",
            "mhc_restriction": "HLA-A*02:01;HLA-A*03:01",
            "mhc_class": "I",
            "mhc_allele_provenance": "peptide_attribution",
            "attributed_sample_label": "MEL3 (13240-006)",
        },
    ]


def _write_rows(tmp_path, monkeypatch, rows):
    path = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: path)


@pytest.mark.parametrize("loader", [load_observations, load_binding])
@pytest.mark.parametrize("columns", [None, ["peptide"], ["attributed_sample_label"]])
def test_old_exact_rows_lose_false_labels_and_extra_donor_copies(
    tmp_path, monkeypatch, loader, columns
):
    _write_rows(tmp_path, monkeypatch, _stale_rows())
    result = loader(columns=columns)
    assert len(result) == 2
    if columns is not None:
        assert list(result.columns) == columns
    if columns is None or "attributed_sample_label" in columns:
        assert result.attributed_sample_label.tolist() == ["", "MEL3 (13240-006)"]
    if columns is None:
        assert result.mhc_restriction.tolist() == ["HLA-G*01:01", "HLA-A*02:01;HLA-A*03:01"]
        assert result.assay_iri.tolist() == [
            "http://www.iedb.org/assay/534",
            "http://www.iedb.org/assay/535",
        ]


def test_valid_multi_patient_and_unscoped_exact_rows_are_preserved(tmp_path, monkeypatch):
    patient = _stale_rows()[-1]
    bovine = {
        **_stale_rows()[0],
        "pmid": 36423003,
        "mhc_restriction": "Bota-6*014:01",
        "mhc_class": "I",
        "attributed_sample_label": "2824TP (T. parva-infected, BoLA-I A19)",
    }
    rows = [patient, {**patient, "attributed_sample_label": "GBM9 (H4198 BT187)"}, bovine]
    _write_rows(tmp_path, monkeypatch, rows)
    result = load_observations()
    assert len(result) == 3
    assert result.attributed_sample_label.tolist() == [r["attributed_sample_label"] for r in rows]


@pytest.mark.parametrize("identity", [None, ""])
def test_stale_rows_without_assay_identity_require_a_rebuild(tmp_path, monkeypatch, identity):
    rows = _stale_rows()[:2]
    for row in rows:
        if identity is None:
            row.pop("assay_iri")
        else:
            row["assay_iri"] = identity
    _write_rows(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match=r"rebuild|Rebuild"):
        load_observations(columns=["peptide"])


def test_empty_projected_query_keeps_requested_schema(tmp_path, monkeypatch):
    _write_rows(tmp_path, monkeypatch, _stale_rows())
    result = load_observations(peptide="NOTPRESENT", columns=["peptide"])
    assert result.empty
    assert result.columns.tolist() == ["peptide"]


def test_valid_patient_rows_do_not_require_repair_identity(tmp_path, monkeypatch):
    row = _stale_rows()[-1]
    row.pop("assay_iri")
    _write_rows(tmp_path, monkeypatch, [row])
    assert load_observations().attributed_sample_label.tolist() == ["MEL3 (13240-006)"]


def test_distinct_monoallelic_assays_are_not_collapsed(tmp_path, monkeypatch):
    rows = _stale_rows()[:2]
    rows.append({**rows[0], "assay_iri": "http://www.iedb.org/assay/536"})
    _write_rows(tmp_path, monkeypatch, rows)
    result = load_observations()
    assert result.assay_iri.tolist() == [
        "http://www.iedb.org/assay/534",
        "http://www.iedb.org/assay/536",
    ]
    assert result.attributed_sample_label.tolist() == ["", ""]


def test_missing_provenance_value_cannot_preserve_an_out_of_scope_label(tmp_path, monkeypatch):
    row = {**_stale_rows()[0], "mhc_allele_provenance": None}
    _write_rows(tmp_path, monkeypatch, [row])
    result = load_observations()
    assert result.attributed_sample_label.tolist() == [""]


def test_export_recovers_monoallelic_host_without_patient_genotype(tmp_path, monkeypatch):
    from hitlist.export import generate_observations_table

    rows = _stale_rows()
    for row in rows:
        row.update(
            cell_name="B cell",
            source_tissue="lymphoid",
            antigen_processing_comments="Single-HLA transfection of B721.221",
            assay_comments="",
            is_monoallelic=True,
            is_binding_assay=False,
        )
    rows[-1].update(cell_name="melanoma", source_tissue="skin", is_monoallelic=False)
    _write_rows(tmp_path, monkeypatch, rows)
    result = generate_observations_table(exclude_non_peptide_ligand=False)
    mono = result.loc[result.mhc_restriction.eq("HLA-G*01:01")]
    assert len(mono) == 1
    assert mono.sample_label.tolist() == ["721.221-HLA-G*01:01"]
    assert mono.sample_mhc.tolist() == ["HLA-G*01:01"]
    assert result.iloc[1].sample_label == "MEL3 (13240-006)"


@pytest.mark.parametrize(
    "scope", [[], "HLA class I", [None], [""], [" HLA class I"], ["HLA class I", "hla class i"]]
)
def test_source_scope_requires_explicit_distinct_restrictions(tmp_path, monkeypatch, scope):
    import yaml

    from hitlist import curation

    path = tmp_path / "pmid_overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 99999534,
                    "peptide_attributions": "mapping.csv",
                    "peptide_attribution_restrictions": scope,
                }
            ]
        )
    )
    real_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda filename: str(path) if filename == "pmid_overrides.yaml" else real_path(filename),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="peptide_attribution_restrictions"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()
