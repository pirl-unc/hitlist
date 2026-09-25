"""Cellular typing is independent of experimental MHC candidates (#520)."""

import pandas as pd
import pytest

from hitlist import curation, export

GENOTYPE = "HLA-B*35:03 HLA-B*40:02 HLA-C*04:01"
TYPING = {
    "mhc_basis": "selected_restriction",
    "mhc_genotype": GENOTYPE,
    "mhc_genotype_cell": "C1R-B*40:02",
    "mhc_genotype_complete_loci": "",
    "mhc_genotype_source": "PMID 31530632, Cell Lines",
}


def _sample(label="C1R-B40", **changes):
    return {
        "sample_label": label,
        "mhc": "HLA-B*40:02",
        "mhc_class": "I",
        **TYPING,
        **changes,
    }


def _install(monkeypatch, samples):
    entries = {31530632: {"ms_samples": samples}}
    monkeypatch.setattr(export, "load_pmid_overrides", lambda: entries)
    monkeypatch.setattr(curation, "load_pmid_overrides", lambda: entries)
    rows = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "pmid": [31530632],
            "mhc_restriction": ["HLA-B*40:02"],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
            "source": ["iedb"],
            "cell_name": [""],
            "source_tissue": [""],
            "assay_comments": [""],
            "antigen_processing_comments": [""],
            "is_binding_assay": [False],
        }
    )
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kwargs: rows.copy())


def test_cellular_background_does_not_expand_experimental_candidates(monkeypatch):
    _install(monkeypatch, [_sample()])
    sample = export.generate_ms_samples_table().iloc[0]
    observation = export.generate_observations_table().iloc[0]
    assert sample.mhc == observation.sample_mhc == "HLA-B*40:02"
    assert observation.mhc_restriction == "HLA-B*40:02"
    assert observation.sample_attribution == "allele_exact"
    for key, value in TYPING.items():
        assert sample[key] == observation[key] == value
    assert observation.mhc_genotype_reported_loci == "HLA-B;HLA-C"
    assert export.generate_ms_peptide_summary_table(
        mhc_allele="HLA-B*35:03", peptide="AAAAAAAAA"
    ).empty


def test_legacy_candidates_are_not_automatically_a_genotype(monkeypatch):
    sample = {key: value for key, value in _sample().items() if key not in TYPING}
    _install(monkeypatch, [sample])
    row = export.generate_observations_table().iloc[0]
    assert row.sample_mhc == "HLA-B*40:02"
    assert row.mhc_genotype == row.mhc_genotype_source == row.mhc_basis == ""


@pytest.mark.parametrize("changed", ["mhc_genotype", "mhc_genotype_cell"])
def test_ambiguous_cellular_typings_are_never_pooled_or_detached(monkeypatch, changed):
    other = "HLA-B*40:02" if changed == "mhc_genotype" else "Another cell"
    _install(monkeypatch, [_sample("first"), _sample("second", **{changed: other})])
    row = export.generate_observations_table().iloc[0]
    assert row.sample_label == ""
    assert row.sample_mhc == "HLA-B*40:02"
    for key in TYPING:
        if key.startswith("mhc_genotype"):
            assert row[key] == ""


def test_shared_cellular_typing_survives_unresolved_treatment(monkeypatch):
    _install(monkeypatch, [_sample("WT"), _sample("KO")])
    row = export.generate_observations_table().iloc[0]
    assert row.sample_label == ""
    assert row.mhc_genotype == GENOTYPE
    assert row.mhc_genotype_cell == "C1R-B*40:02"


def test_typing_columns_survive_projection_and_empty_samples(monkeypatch):
    _install(monkeypatch, [_sample()])
    columns = [*TYPING, "mhc_genotype_reported_loci"]
    projected = export.generate_observations_table(columns=columns)
    assert list(projected.columns) == columns
    assert projected.iloc[0].mhc_genotype == GENOTYPE
    assert set(columns) <= set(export.generate_ms_samples_table(mhc_class="II").columns)


@pytest.mark.parametrize("mode", ["ms", "binding", "both"])
def test_training_export_preserves_ms_typing_and_leaves_binding_unknown(monkeypatch, mode):
    _install(monkeypatch, [_sample()])
    monkeypatch.setattr(
        export,
        "generate_binding_table",
        lambda **kwargs: pd.DataFrame(
            {
                "peptide": ["LLLLLLLLL"],
                "pmid": [31530632],
                "mhc_restriction": ["HLA-B*40:02"],
                "source": ["iedb"],
            }
        ),
    )
    result = export.generate_training_table(
        include_evidence=mode, columns=["evidence_kind", *curation.MHC_TYPING_COLUMNS]
    )
    for row in result.to_dict("records"):
        if row["evidence_kind"] == "ms":
            assert row["mhc_genotype"] == GENOTYPE
        else:
            assert all(row[column] == "" for column in curation.MHC_TYPING_COLUMNS)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"mhc_basis": "genotype"}, "mhc_basis"),
        ({"mhc_genotype_source": ""}, "mhc_genotype_source"),
        ({"mhc_genotype_cell": ""}, "mhc_genotype_cell"),
        ({"mhc_genotype_complete_loci": "HLA-A"}, "complete_loci"),
        ({"mhc_genotype_complete_loci": "HLA-C;HLA-B"}, "complete_loci"),
        ({"mhc_genotype": ""}, "mhc_genotype"),
        ({"mhc_genotype": False}, "mhc_genotype"),
    ],
)
def test_invalid_typing_contract_is_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        curation.sample_mhc_metadata(_sample(**changes))


def test_lorente_primary_source_separates_background_and_selected_restriction():
    samples = curation.load_pmid_overrides()[31530632]["ms_samples"]
    assert len(samples) == 5
    for sample in samples:
        assert sample["mhc"] == "HLA-B*40:02"
        assert sample["mhc_genotype"] == GENOTYPE
        assert sample["mhc_basis"] == "selected_restriction"
        assert not sample.get("mhc_genotype_complete_loci")


def test_selected_restrictions_stay_out_of_cellular_background_attribution():
    assert set(curation.sample_alleles_for_pmid(31530632).values()) == {frozenset({"HLA-B*40:02"})}


def test_sarkizova_residual_c0102_does_not_expand_other_selected_alleles():
    samples = curation.load_pmid_overrides()[31844290]["ms_samples"]
    mono = [s for s in samples if s["sample_label"].startswith("721.221-")]
    assert len(mono) == 95
    for sample in mono:
        selected = curation.sample_mhc_candidates(sample["mhc"]).exact
        genotype = curation.sample_mhc_candidates(sample["mhc_genotype"]).exact
        assert len(selected) == 1
        assert genotype == selected | {"HLA-C*01:02"}
        assert sample["mhc_basis"] == "selected_restriction"
        assert not sample.get("mhc_genotype_complete_loci")


def test_gbm_partial_drb1_candidates_do_not_claim_complete_class_ii_typing():
    samples = export.generate_ms_samples_table()
    sample = samples[
        samples.pmid.eq(33592498) & samples.sample_label.eq("HROG02 CIITA-transduced (class II)")
    ].iloc[0]
    assert sample.mhc == "HLA-DRB1*03:01 HLA-DRB1*07:01"
    typed = curation.sample_mhc_candidates(sample.mhc_genotype).exact
    assert {"HLA-DQB1*02:01", "HLA-DPB1*04:01", "HLA-DRB3*01:01"} <= typed
    assert "HLA-DPA1" not in sample.mhc_genotype_complete_loci.split(";")
    assert "HLA-DRB3" not in sample.mhc_genotype_complete_loci.split(";")


def test_modc_genotype_names_p4_and_never_pools_a549_feeders():
    samples = curation.load_pmid_overrides()[35051231]["ms_samples"]
    sample = next(s for s in samples if s["sample_label"].endswith("A549 feeders"))
    assert sample["mhc_genotype_cell"] == "P4"
    alleles = curation.sample_mhc_candidates(sample["mhc_genotype"]).exact
    assert {"HLA-A*01:01", "HLA-B*51:01", "HLA-B*57:01"} <= alleles
    assert not ({"HLA-A*25:01", "HLA-A*30:01", "HLA-B*44:03"} & alleles)
    assert "HLA-DQA1" not in sample["mhc_genotype_complete_loci"]


def test_genotype_contract_is_checked_on_yaml_load(tmp_path, monkeypatch):
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump([{"pmid": 1, "ms_samples": [_sample(mhc_genotype_cell="")]}]))
    real_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda name: path if name == "pmid_overrides.yaml" else real_path(name),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match=r"PMID 1: ms_samples\[0\].*mhc_genotype_cell"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_token_qc_reads_independent_genotype(monkeypatch):
    from hitlist import qc

    monkeypatch.setattr(
        qc,
        "load_pmid_overrides",
        lambda: {1: {"ms_samples": [_sample(mhc_genotype="HLA-B*40:02 MADEUP")]}},
    )
    monkeypatch.setattr(qc, "ms_excluded_pmids", lambda: frozenset())
    findings = qc.mhc_token_audit(evidence_frames={})
    assert findings[["field", "token"]].values.tolist() == [["mhc_genotype", "MADEUP"]]


def test_filtered_and_projected_typing_matches_complete_export(tmp_path, monkeypatch):
    from hitlist import observations

    path = tmp_path / "observations.parquet"
    rows = [
        {
            "peptide": peptide,
            "pmid": 31530632,
            "mhc_restriction": "HLA-B*40:02",
            "mhc_class": "I",
            "mhc_species": "Homo sapiens",
            "source": "iedb",
            "cell_name": "C1R",
            "source_tissue": "",
            "antigen_processing_comments": "",
            "assay_comments": "",
            "is_binding_assay": False,
        }
        for peptide in ["AAAAAAAAA", "LLLLLLLLL"]
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(observations, "observations_path", lambda: path)
    columns = ["peptide", "sample_mhc", *curation.MHC_TYPING_COLUMNS]
    full = export.generate_observations_table(columns=columns).set_index("peptide")
    selected = export.generate_observations_table(peptide="AAAAAAAAA", columns=columns)
    pd.testing.assert_frame_equal(
        selected.set_index("peptide").astype(str), full.loc[["AAAAAAAAA"]].astype(str)
    )
    assert selected.iloc[0].mhc_genotype == GENOTYPE
