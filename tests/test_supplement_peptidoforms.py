import json

import pandas as pd
import pytest

from hitlist.supplement import scan_supplementary


def scan(tmp_path, rows, policy=None):
    path = tmp_path / "forms.csv"
    pd.DataFrame(rows).assign(mhc_restriction="HLA-A*02:01", mhc_class="I").to_csv(
        path, index=False
    )
    entry = {"pmid": 99999999, "file": path.name, "defaults": {"species": "Homo sapiens"}}
    if policy is not None:
        entry["peptidoform_policy"] = policy
    return scan_supplementary(
        classify_source=False, entries=[entry], directory=tmp_path, allow_download=False
    )


def test_explicit_false_true_unknown_and_conflict_survive(tmp_path):
    rows = [
        {"peptide": "AAAAAAAAA", "has_ptm": "False", "peptide_modifications": ""},
        {"peptide": "AAAAAAAAA", "has_ptm": "TRUE", "peptide_modifications": "Acetyl:N-term"},
        {"peptide": "AAAAAAAAA", "has_ptm": "", "peptide_modifications": ""},
        {"peptide": "AAAAAAAAA", "has_ptm": "false", "peptide_modifications": "Acetyl:N-term"},
    ]
    result = scan(tmp_path, rows)
    assert len(result) == result.assay_iri.nunique() == 4
    assert result.has_ptm.iloc[0] == False  # noqa: E712
    assert result.has_ptm.iloc[1] == True  # noqa: E712
    assert pd.isna(result.has_ptm.iloc[2]) and pd.isna(result.has_ptm.iloc[3])
    assert result.peptidoform_review_status.tolist() == [
        "reported_unmodified",
        "reported_modified",
        "unknown",
        "conflicting",
    ]
    assert result.peptide_modifications_raw.tolist() == ["", "Acetyl:N-term", "", "Acetyl:N-term"]
    # Original unannotated identities stay stable; explicit chemistry is distinct.
    assert ":form:" not in result.assay_iri.iloc[2]
    out = tmp_path / "roundtrip.parquet"
    result.to_parquet(out, index=False)
    replay = pd.read_parquet(out)
    pd.testing.assert_frame_equal(result, replay)


def test_explicit_unmodified_text_and_modifications_only(tmp_path):
    result = scan(
        tmp_path,
        [
            {"peptide": "AAAAAAAAA", "peptide_modifications": "Unmodified"},
            {"peptide": "CCCCCCCCC", "peptide_modifications": "Carbamidomethyl:C1"},
        ],
    )
    assert result.has_ptm.tolist() == [False, True]
    assert result.peptide_modifications.iloc[0] == ""
    assert result.peptide_modifications_raw.iloc[0] == "Unmodified"


def test_empty_modification_field_does_not_assert_unmodified(tmp_path):
    result = scan(tmp_path, [{"peptide": "AAAAAAAAA", "peptide_modifications": ""}])
    assert pd.isna(result.has_ptm.iloc[0])


def test_unknown_modification_text_is_not_a_reported_modification(tmp_path):
    result = scan(tmp_path, [{"peptide": "AAAAAAAAA", "peptide_modifications": "Unknown"}])
    assert pd.isna(result.has_ptm.iloc[0])
    assert result.peptide_modifications.iloc[0] == ""
    assert result.peptide_modifications_raw.iloc[0] == "Unknown"


def test_cited_file_policy_and_row_conflict(tmp_path):
    result = scan(
        tmp_path,
        [
            {"peptide": "AAAAAAAAA", "has_ptm": ""},
            {"peptide": "CCCCCCCCC", "has_ptm": "True"},
        ],
        policy={"status": "unmodified", "reference": "source-fixture", "locator": "sheet 1 filter"},
    )
    assert result.has_ptm.iloc[0] == False  # noqa: E712
    assert pd.isna(result.has_ptm.iloc[1])
    assert result.peptidoform_review_status.tolist() == ["source_policy_unmodified", "conflicting"]
    assert json.loads(result.peptidoform_evidence.iloc[0])["policy"]["locator"] == "sheet 1 filter"


def test_uncited_file_policy_rejected(tmp_path):
    with pytest.raises(ValueError, match="reference and locator"):
        scan(tmp_path, [{"peptide": "AAAAAAAAA"}], policy={"status": "unmodified"})


def test_notation_and_distinct_modified_forms_preserved(tmp_path):
    rows = [
        {"peptide": "CCCCCCCCC", "has_ptm": "True", "peptide_extended": "C[+57]CCCCCCCC"},
        {"peptide": "CCCCCCCCC", "has_ptm": "True", "peptide_extended": "CC[+57]CCCCCCC"},
        {"peptide": "CCCCCCCCC", "has_ptm": "False", "peptide_extended": "C[+57]CCCCCCCC"},
    ]
    result = scan(tmp_path, [*rows, rows[0]])
    assert len(result) == result.assay_iri.nunique() == 3
    assert result.peptide_extended.tolist() == [r["peptide_extended"] for r in rows]
    assert result.peptidoform_review_status.iloc[2] == "conflicting"


def test_invalid_boolean_is_not_truthy(tmp_path):
    with pytest.raises(ValueError, match="has_ptm"):
        scan(tmp_path, [{"peptide": "AAAAAAAAA", "has_ptm": "maybe"}])


def test_training_export_preserves_source_chemistry(tmp_path, monkeypatch):
    from hitlist.export import generate_training_table

    frame = scan(
        tmp_path,
        [
            {"peptide": "AAAAAAAAA", "has_ptm": "False"},
            {"peptide": "CCCCCCCCC", "has_ptm": "True", "peptide_modifications": "C1:+57"},
            {"peptide": "DDDDDDDDD", "has_ptm": ""},
        ],
    )
    path = tmp_path / "observations.parquet"
    # The builder supplies the source namespace after scanning.
    frame.assign(source="supplement").to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    exported = generate_training_table(include_evidence="ms")
    assert len(exported) == 3
    values = exported.set_index("peptide")
    assert values.loc["AAAAAAAAA", "has_ptm"] == False  # noqa: E712
    assert values.loc["CCCCCCCCC", "has_ptm"] == True  # noqa: E712
    assert pd.isna(values.loc["DDDDDDDDD", "has_ptm"])
    assert values.loc["CCCCCCCCC", "peptide_modifications"] == "C1:+57"


def test_cross_source_dedup_keeps_distinct_or_unknown_chemistry(tmp_path):
    from hitlist.builder import _drop_supplementary_duplicates

    forms = scan(
        tmp_path,
        [
            {"peptide": "AAAAAAAAA", "has_ptm": "False"},
            {"peptide": "AAAAAAAAA", "has_ptm": "True", "peptide_modifications": "N-term:+42"},
            {"peptide": "AAAAAAAAA", "has_ptm": ""},
        ],
    )
    observed = forms.iloc[[0]].copy()
    result = _drop_supplementary_duplicates(forms, observed)
    assert len(result) == 2
    assert result.has_ptm.iloc[0] == True  # noqa: E712
    assert pd.isna(result.has_ptm.iloc[1])
