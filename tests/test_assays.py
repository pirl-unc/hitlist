import pandas as pd
import pytest


@pytest.mark.parametrize(
    "method,response",
    [
        ("cellular MHC/direct/fluorescence", "qualitative binding"),
        ("purified MHC/direct/fluorescence", "50% dissociation temperature"),
        ("binding assay", "dissociation constant KD"),
    ],
)
def test_structured_binding_is_not_elution(method, response):
    from hitlist.curation import is_binding_assay

    assert is_binding_assay("Positive", "", method, response)


@pytest.mark.parametrize(
    "method,response,outcome,source,modality,positive_ms",
    [
        ("cellular MHC/mass spectrometry", "ligand presentation", "Positive", "iedb", "ms", True),
        (
            "secreted MHC/mass spectrometry",
            "ligand presentation",
            "Positive-Low",
            "iedb",
            "ms",
            True,
        ),
        ("mass spectrometry", "ligand presentation", "Negative", "iedb", "ms", False),
        ("x-ray crystallography", "3D structure", "Positive", "iedb", "structural", False),
        ("electron microscopy", "3D structure", "Positive", "cedar", "structural", False),
        ("any method", "3D structure", "Positive", "iedb", "structural", False),
        ("Edman degradation", "ligand presentation", "Positive", "iedb", "non_ms_ligand", False),
        ("T cell recognition", "ligand presentation", "Positive", "iedb", "non_ms_ligand", False),
        ("coelution", "ligand presentation", "Negative", "iedb", "non_ms_ligand", False),
        ("", "", "", "iedb", "unknown", False),
        ("", "", "", "supplement", "ms", True),
        ("fluorescence", "", "Positive", "supplement", "other", False),
        ("mass spectrometry", "3D structure", "Positive", "iedb", "unknown", False),
        ("mass spectrometry", "qualitative binding", "Positive", "iedb", "unknown", False),
        (None, None, None, "iedb", "unknown", False),
    ],
)
def test_modality_is_separate_from_polarity(
    method, response, outcome, source, modality, positive_ms
):
    from hitlist.assays import assay_annotations

    record = assay_annotations(outcome, "", method, response, source)
    assert record["assay_modality"] == modality
    assert record["is_ms_observation"] == positive_ms
    assert record["is_binding_assay"] == (modality == "binding")
    assert record["assay_modality_source"]


def test_explicit_ms_overrides_binding_comment_and_tier():
    from hitlist.curation import is_binding_assay

    assert not is_binding_assay(
        "Positive-Low",
        "Also evaluated by refolding assay",
        "mass spectrometry",
        "ligand presentation",
    )


def test_ms_evidence_bundle_rejects_conflicting_response():
    from hitlist.evidence_bundle import _positive_ms

    frame = pd.DataFrame(
        {"assay_method": ["mass spectrometry"], "response_measured": ["3D structure"]}
    )
    assert not _positive_ms(frame).any()


@pytest.mark.parametrize("categorical", [False, True])
def test_arrow_filter_matches_scalar_rules_under_projection(tmp_path, categorical):
    from hitlist.assays import assay_annotations, positive_ms_expression

    rows = []
    for method in (None, "", " MaSs Spectrometry ", "fluorescence", "x-ray crystallography"):
        for response in (None, "", "ligand presentation", "qualitative binding", "3D structure"):
            for source in ("iedb", "supplement"):
                for outcome in (None, "Positive", "Positive-Low", "Negative"):
                    rows.append((method, response, source, outcome))
    frame = pd.DataFrame(
        rows, columns=["assay_method", "response_measured", "source", "qualitative_measurement"]
    )
    frame["row_id"] = range(len(frame))
    expected = [
        i
        for i, (method, response, source, outcome) in enumerate(rows)
        if assay_annotations(outcome, "", method, response, source)["is_ms_observation"]
    ]
    if categorical:
        for column in frame.columns[:-1]:
            frame[column] = frame[column].astype("category")
    path = tmp_path / "legacy.parquet"
    frame.to_parquet(path, index=False)
    result = pd.read_parquet(
        path, columns=["row_id"], filters=positive_ms_expression(frame.columns)
    )
    assert result.row_id.tolist() == expected


def test_missing_modality_fields_cannot_establish_ms(tmp_path):
    from hitlist.assays import positive_ms_expression

    path = tmp_path / "legacy.parquet"
    pd.DataFrame({"peptide": ["SLYNTVATL"], "is_binding_assay": [False]}).to_parquet(path)
    assert pd.read_parquet(
        path, filters=positive_ms_expression(["peptide", "is_binding_assay"])
    ).empty


def test_build_partitions_retains_all_source_contributors(
    tmp_path, monkeypatch, _isolated_curation_root
):
    from hitlist import builder, downloads, mappings, supplement
    from hitlist.observations import load_all_evidence, load_observations, load_other_assays
    from hitlist.provenance import load_contributors
    from tests.test_provenance import _row
    from tests.test_scanner import _write_tiny_iedb_csv

    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "indexes")
    source = tmp_path / "mixed.csv"
    cases = [
        ("mass spectrometry", "ligand presentation", "Positive-Low"),
        ("cellular MHC/direct/fluorescence", "qualitative binding", "Positive"),
        ("purified MHC/direct/fluorescence", "50% dissociation temperature", "Positive"),
        ("x-ray crystallography", "3D structure", "Positive"),
        ("electron microscopy", "3D structure", "Positive"),
        ("Edman degradation", "ligand presentation", "Positive"),
        ("mass spectrometry", "ligand presentation", "Negative"),
        ("", "", "Positive"),
    ]
    rows = []
    for index, (method, response, outcome) in enumerate(cases):
        row = _row(f"http://iedb.org/assay/{index + 1}")
        row[14], row[22], row[23] = outcome, method, response
        rows.append(row)
    _write_tiny_iedb_csv(source, [*rows, rows[3]])
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())

    def empty(path):
        frame = pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    monkeypatch.setattr(
        builder, "build_bulk_proteomics", lambda **kw: empty(builder._bulk_proteomics_path())
    )
    monkeypatch.setattr(
        builder, "build_line_expression", lambda **kw: empty(builder._line_expression_path())
    )
    mapping_inputs = {}
    monkeypatch.setattr(mappings, "build_peptide_mappings", lambda **kw: mapping_inputs.update(kw))
    builder.build_observations()
    assert len(mapping_inputs["obs_override"]) == 1
    assert len(mapping_inputs["binding_override"]) == 2
    assert len(mapping_inputs["other_override"]) == 5
    evidence = load_all_evidence()
    assert evidence.groupby("evidence_kind").size().to_dict() == {"ms": 1, "binding": 2, "other": 5}
    assert len(load_contributors(evidence.provenance_id)) == 9
    assert set(load_other_assays().assay_modality) == {
        "structural",
        "non_ms_ligand",
        "ms",
        "unknown",
    }
    assert load_observations(columns=["peptide", "is_binding_assay"]).is_binding_assay.tolist() == [
        False
    ]
    assert builder._cache_is_valid(builder._source_paths())
    builder._other_assays_path().unlink()
    assert not builder._cache_is_valid(builder._source_paths())
    with pytest.raises(ValueError, match=r"other_assays\.parquet"):
        load_contributors()


def test_legacy_projection_filters_before_pandas_and_refreshes_flags(tmp_path, monkeypatch):
    from hitlist.observations import load_observations

    path = tmp_path / "observations.parquet"
    pd.DataFrame(
        {
            "peptide": ["MS", "FLUORESCENCE", "STRUCTURE", "NEGATIVE", "MISSING"],
            "assay_method": [
                "mass spectrometry",
                "cellular MHC/direct/fluorescence",
                "x-ray crystallography",
                "mass spectrometry",
                None,
            ],
            "response_measured": [
                "ligand presentation",
                "qualitative binding",
                "3D structure",
                "ligand presentation",
                None,
            ],
            "qualitative_measurement": [
                "Positive-Low",
                "Positive",
                "Positive",
                "Negative",
                "Positive",
            ],
            "is_binding_assay": [True, False, False, False, False],
        }
    ).to_parquet(path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    read = pd.read_parquet
    materialized = []

    def capture(*args, **kwargs):
        result = read(*args, **kwargs)
        materialized.append(len(result))
        return result

    monkeypatch.setattr(pd, "read_parquet", capture)
    result = load_observations(
        columns=["peptide", "is_binding_assay", "assay_modality", "is_ms_observation"]
    )
    assert materialized == [1]
    assert result.to_dict("records") == [
        {
            "peptide": "MS",
            "is_binding_assay": False,
            "assay_modality": "ms",
            "is_ms_observation": True,
        }
    ]


def test_bundle_policy_version_preserves_legacy_interpretation():
    from hitlist.evidence_bundle import _legacy_positive_ms, _positive_ms

    rows = pd.DataFrame(
        {"assay_method": ["mass spectrometry"], "response_measured": ["3D structure"]}
    )
    assert _legacy_positive_ms(rows).all()
    assert not _positive_ms(rows).any()
