"""Reassignment must select an allele within each biological sample (#547)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hitlist import predict


def _observation(pmid, label, genotype, peptide="AAAAAAAAA"):
    return {
        "peptide": peptide,
        "pmid": pmid,
        "sample_label": label,
        "sample_mhc": genotype,
        "mhc_restriction": "HLA class I",
        "is_monoallelic": False,
    }


def _install_predictions(monkeypatch, rows, ranks, method="mhcflurry"):
    monkeypatch.setattr(
        "hitlist.export.generate_observations_table", lambda **kwargs: pd.DataFrame(rows)
    )
    calls = []

    def score(pairs):
        calls.append(pairs.copy())
        result = pairs.copy()
        result["presentation_percentile"] = result["allele"].map(ranks)
        result["affinity_nM"] = result["presentation_percentile"] * 100
        return result

    monkeypatch.setattr(predict, f"_predict_{method}", score)
    return calls


@pytest.mark.parametrize("method", ["mhcflurry", "netmhcpan"])
def test_shared_peptide_keeps_each_sample_and_its_own_winner(monkeypatch, method):
    rows = [
        _observation(1, "cell_A", "HLA-A*02:01 HLA-B*07:02"),
        _observation(2, "cell_B", "HLA-A*24:02 HLA-B*08:01"),
    ]
    calls = _install_predictions(
        monkeypatch,
        rows,
        {"HLA-A*02:01": 1.0, "HLA-B*07:02": 2.0, "HLA-A*24:02": 0.01, "HLA-B*08:01": 0.1},
        method,
    )
    result = predict.reassign_class_only_alleles(method=method).set_index("sample_label")
    assert set(result.index) == {"cell_A", "cell_B"}
    assert result.loc["cell_A", "best_allele"] == "HLA-A*02:01"
    assert result.loc["cell_B", "best_allele"] == "HLA-A*24:02"
    assert result.loc["cell_A", "best_affinity_nM"] == 100
    assert result.loc["cell_A", "pmid"] == 1
    assert result.loc["cell_B", "pmid"] == 2
    assert result["n_alleles_tested"].tolist() == [2, 2]
    assert len(calls) == 1 and len(calls[0]) == 4
    for row in result.itertuples():
        assert row.best_allele in row.sample_mhc.split()


def test_context_identity_includes_study_label_and_genotype(monkeypatch):
    rows = [
        _observation(1, "same_label", "HLA-A*02:01 HLA-B*07:02"),
        _observation(2, "same_label", "HLA-A*02:01 HLA-B*07:02"),
        _observation(2, "same_label", "HLA-A*24:02 HLA-B*08:01"),
        _observation(2, "other_label", "HLA-A*24:02 HLA-B*08:01"),
    ]
    calls = _install_predictions(
        monkeypatch,
        rows + rows,
        {"HLA-A*02:01": 1, "HLA-B*07:02": 2, "HLA-A*24:02": 0.01, "HLA-B*08:01": 0.1},
    )
    forward = predict.reassign_class_only_alleles()
    assert len(forward) == 4
    assert len(calls[0]) == 4
    assert not calls[0].duplicated(["peptide", "allele"]).any()
    monkeypatch.setattr(
        "hitlist.export.generate_observations_table", lambda **kwargs: pd.DataFrame(rows[::-1])
    )
    reverse = predict.reassign_class_only_alleles()
    keys = ["pmid", "sample_label", "sample_mhc"]
    pd.testing.assert_frame_equal(
        forward.sort_values(keys).reset_index(drop=True),
        reverse.sort_values(keys).reset_index(drop=True),
    )


def test_equal_scores_have_a_stable_winner(monkeypatch):
    _install_predictions(
        monkeypatch,
        [_observation(1, "cell", "HLA-B*07:02 HLA-A*02:01")],
        {"HLA-A*02:01": 0.1, "HLA-B*07:02": 0.1},
    )
    assert predict.reassign_class_only_alleles().iloc[0]["best_allele"] == "HLA-A*02:01"


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_unscored_sample_does_not_borrow_another_samples_prediction(monkeypatch, missing):
    _install_predictions(
        monkeypatch,
        [
            _observation(1, "unscored", "HLA-A*02:01"),
            _observation(2, "scored", "HLA-A*24:02"),
        ],
        {"HLA-A*02:01": missing, "HLA-A*24:02": 0.01},
    )
    result = predict.reassign_class_only_alleles().set_index("sample_label")
    assert set(result.index) == {"unscored", "scored"}
    for name in ("best_allele", "best_affinity_nM", "best_presentation_percentile"):
        assert pd.isna(result.loc["unscored", name])
    assert not result.loc["unscored", "is_strong_binder"]
    assert not result.loc["unscored", "is_weak_binder"]
    assert result.loc["scored", "best_allele"] == "HLA-A*24:02"


def test_empty_predictor_output_preserves_unscored_context(monkeypatch):
    _install_predictions(monkeypatch, [_observation(1, "cell", "HLA-A*02:01")], {})
    monkeypatch.setattr(
        predict,
        "_predict_mhcflurry",
        lambda pairs: pd.DataFrame(
            columns=["peptide", "allele", "affinity_nM", "presentation_percentile"]
        ),
    )
    result = predict.reassign_class_only_alleles()
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["best_allele"])
    assert not result.iloc[0]["is_weak_binder"]


@pytest.mark.parametrize(
    "peptide,allele", [("AAAAAAAAA", "HLA-B*07:02"), ("CCCCCCCCC", "HLA-A*02:01")]
)
def test_unrequested_prediction_cannot_supply_a_winner(monkeypatch, peptide, allele):
    _install_predictions(monkeypatch, [_observation(1, "cell", "HLA-A*02:01")], {})
    monkeypatch.setattr(
        predict,
        "_predict_mhcflurry",
        lambda pairs: pd.DataFrame(
            {
                "peptide": [peptide],
                "allele": [allele],
                "affinity_nM": [1.0],
                "presentation_percentile": [0.01],
            }
        ),
    )
    result = predict.reassign_class_only_alleles()
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["best_allele"])


def test_no_valid_length_peptides_returns_empty_without_predicting(monkeypatch):
    calls = _install_predictions(
        monkeypatch, [_observation(1, "cell", "HLA-A*02:01", peptide="AAA")], {}
    )
    result = predict.reassign_class_only_alleles()
    assert result.empty
    assert "best_allele" in result
    assert calls == []


def test_six_distinct_allele_default_does_not_truncate_larger_genotypes(monkeypatch):
    alleles = [
        "HLA-A*01:01",
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-B*08:01",
        "HLA-C*07:01",
        "HLA-C*07:02",
    ]
    rows = [
        _observation(1, "six", " ".join(alleles)),
        _observation(2, "seven", " ".join([*alleles, "HLA-A*24:02"])),
        _observation(3, "repeated", " ".join(alleles + alleles)),
    ]
    calls = _install_predictions(monkeypatch, rows, dict.fromkeys([*alleles, "HLA-A*24:02"], 0.1))
    result = predict.reassign_class_only_alleles().set_index("sample_label")
    assert set(result.index) == {"six", "repeated"}
    assert result["n_alleles_tested"].tolist() == [6, 6]
    assert len(calls[0]) == 6
    explicit = predict.reassign_class_only_alleles(max_alleles_per_sample=7)
    assert set(explicit["sample_label"]) == {"six", "seven", "repeated"}


def test_cli_uses_six_alleles_and_reports_sample_contexts(monkeypatch, capsys):
    from hitlist import cli

    seen = []

    def reassign(**kwargs):
        seen.append(kwargs)
        return pd.DataFrame({"is_strong_binder": [False, True], "is_weak_binder": [True, True]})

    monkeypatch.setattr(predict, "reassign_class_only_alleles", reassign)
    monkeypatch.setattr("sys.argv", ["hitlist", "reassign-alleles"])
    cli.main()
    assert seen[0]["max_alleles_per_sample"] == 6
    assert "2 peptide/sample contexts" in capsys.readouterr().err


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("allele", ["HLA-A*02:01", "HLA-A0201"])
def test_netmhcpan_result_preserves_requested_allele_spelling(monkeypatch, tmp_path, empty, allele):
    tokens = ["x"] * 16
    tokens[1], tokens[2], tokens[10], tokens[12], tokens[15] = (
        "HLA-A02:01",
        "AAAAAAAAA",
        "PEPLIST",
        "0.4",
        "25.0",
    )
    monkeypatch.setattr(predict, "Path", lambda path: tmp_path)
    monkeypatch.setattr(
        predict.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="" if empty else " ".join(tokens)),
    )
    result = predict._predict_netmhcpan(
        pd.DataFrame({"peptide": ["AAAAAAAAA"], "allele": [allele]})
    )
    if empty:
        assert result.empty
        assert "presentation_percentile" in result
    else:
        assert result.iloc[0]["allele"] == allele
        assert result.iloc[0]["presentation_percentile"] == 0.4


def test_netmhcpan_rejects_a_result_for_a_different_allele(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "Path", lambda path: tmp_path)
    monkeypatch.setattr(
        predict.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0 HLA-B07:02 AAAAAAAAA AAAAAAAAA 0 0 0 0 0 AAAAAAAAA PEPLIST 0.9 0.4 0.5 0.6 25"
        ),
    )
    with pytest.raises(RuntimeError, match="requested allele"):
        predict._predict_netmhcpan(
            pd.DataFrame({"peptide": ["AAAAAAAAA"], "allele": ["HLA-A*02:01"]})
        )
