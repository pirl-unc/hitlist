"""An assay can support several observations, each with several mappings (#614)."""

import pandas as pd
import pytest

from hitlist.export import _apply_training_defaults, generate_training_table


@pytest.fixture
def donor_export(tmp_path, monkeypatch):
    from hitlist import downloads
    from hitlist.export import _TRAINING_MAPPING_COLUMNS

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    overrides = {
        99999101: {
            "species": "Homo sapiens",
            "ms_samples": [
                {
                    "sample_label": label,
                    "condition_id": arm,
                    "n_samples": 1,
                    "mhc_class": "I",
                    "mhc": "HLA-A*02:01",
                }
                for label, arm in [("Donor A", "donor_a"), ("Donor B", "donor_b")]
            ],
        }
    }
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)
    rows = pd.DataFrame(
        [
            {
                "peptide": peptide,
                "assay_iri": assay,
                "reference_iri": "reference:1",
                "pmid": 99999101,
                "attributed_sample_label": label,
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "source": "iedb",
                "mhc_species": "Homo sapiens",
                "is_monoallelic": False,
                "is_binding_assay": False,
                "qualitative_measurement": "Positive",
            }
            for peptide, assay, label in [
                ("SIINFEKL", "assay:shared", "Donor A"),
                ("SIINFEKL", "assay:shared", "Donor B"),
                ("SIINFEKL", "assay:independent", "Donor A"),
                ("NLVPMVATV", "assay:other", ""),
            ]
        ]
    )
    path = tmp_path / "observations.parquet"
    rows.assign(assay_method="mass spectrometry").to_parquet(path, index=False)
    mappings = pd.DataFrame(
        [
            {"peptide": "SIINFEKL", "protein_id": protein, "position": position}
            for protein, position in [("P1", 1), ("P2", 10)]
        ]
    ).reindex(columns=["peptide", *_TRAINING_MAPPING_COLUMNS])
    mappings.to_parquet(tmp_path / "peptide_mappings.parquet", index=False)
    return rows, path, overrides


def test_donor_observations_survive_mapping_expansion_and_projection(donor_export):
    compact = generate_training_table(include_evidence="ms", peptide="SIINFEKL")
    assert len(compact) == compact.evidence_row_id.nunique() == 3
    assert compact.evidence_source_id.nunique() == 2
    assert set(compact.evidence_source_id) == {"ms:assay:shared", "ms:assay:independent"}
    # Freeze the v1 serialization contract, not just within-run equality.
    assert compact.evidence_row_id.iloc[0] == "ms:attributed:v1:assay:shared|arm:99999101:donor_a"

    expanded = generate_training_table(
        include_evidence="ms", peptide="SIINFEKL", map_source_proteins=True
    )
    assert len(expanded) == 6
    assert set(expanded.evidence_row_id) == set(compact.evidence_row_id)
    assert expanded.groupby("evidence_row_id").size().tolist() == [2, 2, 2]
    assert expanded.groupby("evidence_row_id").protein_id.nunique().tolist() == [2, 2, 2]

    for mapped in (False, True):
        projected = generate_training_table(
            include_evidence="ms",
            peptide="SIINFEKL",
            map_source_proteins=mapped,
            columns=["peptide"],
        )
        assert projected.columns.tolist() == [
            "peptide",
            "evidence_kind",
            "evidence_row_id",
            "evidence_source_id",
            "provenance_id",
            "provenance_status",
            "lineage_context_id",
        ]
        assert set(projected.evidence_row_id) == set(compact.evidence_row_id)
        assert len(projected) == (6 if mapped else 3)


def test_donor_ids_ignore_order_filters_and_display_renames(donor_export):
    rows, path, overrides = donor_export
    original = generate_training_table(include_evidence="ms")
    key = ["assay_iri", "attributed_sample_label", "peptide"]

    def identities(frame):
        return (
            frame.astype(dict.fromkeys(key, "string")).set_index(key).evidence_row_id.sort_index()
        )

    expected = identities(original)
    rows.iloc[::-1].assign(assay_method="mass spectrometry").to_parquet(path, index=False)
    reordered = generate_training_table(include_evidence="ms")
    pd.testing.assert_series_equal(identities(reordered), expected)

    filtered = generate_training_table(include_evidence="ms", peptide="SIINFEKL")
    pd.testing.assert_series_equal(
        identities(filtered),
        expected[expected.index.get_level_values("peptide") == "SIINFEKL"],
    )
    unsplit = original[original.assay_iri == "assay:other"].iloc[0]
    assert unsplit.evidence_row_id == unsplit.evidence_source_id == "ms:assay:other"

    # A source subset with only one donor must retain that donor's identity.
    rows.iloc[[0]].assign(assay_method="mass spectrometry").to_parquet(path, index=False)
    single = generate_training_table(include_evidence="ms")
    assert single.evidence_row_id.iloc[0] == original.evidence_row_id.iloc[0]

    rows.loc[rows.attributed_sample_label == "Donor A", "attributed_sample_label"] = "Renamed A"
    overrides[99999101]["ms_samples"][0]["sample_label"] = "Renamed A"
    rows.assign(assay_method="mass spectrometry").to_parquet(path, index=False)
    renamed = generate_training_table(include_evidence="ms")
    assert renamed.evidence_row_id.tolist() == original.evidence_row_id.tolist()


def test_unresolved_attributions_use_original_labels_and_study_namespace(monkeypatch):
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: {})
    rows = pd.DataFrame(
        {
            "evidence_kind": ["ms"] * 5,
            "assay_iri": ["assay:shared"] * 5,
            "pmid": [1, 1, 2, None, None],
            "attributed_sample_label": ["Donor A", "Donor B", "Donor A", "Donor A", "Donor B"],
            # Heuristic/pool metadata must never erase explicit source labels.
            "sample_label": ["pooled"] * 5,
            "condition_id": ["pooled"] * 5,
        }
    )
    result = _apply_training_defaults(rows)
    assert result.evidence_row_id.is_unique
    assert result.evidence_source_id.nunique() == 1
    assert (
        _apply_training_defaults(rows.iloc[[1]]).evidence_row_id.iloc[0]
        == result.evidence_row_id.iloc[1]
    )


def test_ambiguous_curated_labels_do_not_select_an_arbitrary_arm(monkeypatch):
    samples = [
        {"sample_label": "same label", "condition_id": "arm_a"},
        {"sample_label": "same label", "condition_id": "arm_b"},
    ]
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: {1: {"ms_samples": samples}})
    rows = pd.DataFrame(
        [
            {
                "evidence_kind": "ms",
                "assay_iri": "assay:1",
                "pmid": 1,
                "attributed_sample_label": "same label",
            }
        ]
    )
    ambiguous = _apply_training_defaults(rows).evidence_row_id.iloc[0]
    samples.reverse()
    assert _apply_training_defaults(rows).evidence_row_id.iloc[0] == ambiguous
    samples.pop()
    assert _apply_training_defaults(rows).evidence_row_id.iloc[0] != ambiguous
    samples.clear()
    assert _apply_training_defaults(rows).evidence_row_id.iloc[0] == ambiguous


def test_attributed_ids_escape_delimiters_and_literal_percent_signs(monkeypatch):
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: {})
    rows = pd.DataFrame(
        {
            "evidence_kind": ["ms"] * 3,
            "assay_iri": ["https://example.org/assay/1|label:2:A"] * 3,
            "pmid": [1] * 3,
            "attributed_sample_label": ["A|B:C", "A%7CB%3AC", "A/B"],
        }
    )
    result = _apply_training_defaults(rows)
    assert result.evidence_row_id.tolist() == [
        "ms:attributed:v1:https://example.org/assay/1%7Clabel:2:A|label:1:A%7CB%3AC",
        "ms:attributed:v1:https://example.org/assay/1%7Clabel:2:A|label:1:A%257CB%253AC",
        "ms:attributed:v1:https://example.org/assay/1%7Clabel:2:A|label:1:A%2FB",
    ]


@pytest.mark.parametrize("kind", ["ms", "binding"])
def test_unattributed_ids_preserve_legacy_values_with_categorical_nulls(kind):
    rows = pd.DataFrame(
        {
            "evidence_kind": pd.Categorical([kind] * 3),
            "assay_iri": ["assay:1", "assay:2", "assay:3"],
            "attributed_sample_label": pd.Categorical([None, "", " "]),
        }
    )
    result = _apply_training_defaults(rows)
    expected = [f"{kind}:assay:{i}" for i in (1, 2, 3)]
    assert result.evidence_row_id.tolist() == expected
    assert result.evidence_source_id.tolist() == expected


@pytest.mark.integration
def test_sarkizova_six_observations_have_six_ids():
    from hitlist.observations import is_built

    if not is_built():
        pytest.skip("Requires the built observation index")
    result = generate_training_table(include_evidence="ms", peptide="SLLQHLIGL", mhc_class="I")
    result = result[result.pmid.astype(str) == "31844290"]
    assert len(result) == result.evidence_row_id.nunique() == 6
    assert result.evidence_source_id.nunique() == 2
    assert set(result.sample_label) == {"MEL15 (13240-015)", "MEL3 (13240-006)", "OV1 (CP-594_v1)"}
