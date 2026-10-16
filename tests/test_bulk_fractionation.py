"""Primary-source attribution survives legacy loaders and numeric pH filters."""

import importlib.util
import io
from pathlib import Path

import pandas as pd
import pytest

from hitlist.bulk_fractionation import (
    curate_fractionation_ph,
    experiment_fractionation_ph,
    fractionation_registry,
)
from hitlist.bulk_proteomics import _apply_ph_filter, _curate_bj_metadata, _read_parquet_cached


def legacy_rows():
    return pd.DataFrame(
        {
            "source": ["Bekker-Jensen_2017"] * 7 + ["CCLE_Nusinow_2020"],
            "reference": ["PMID:28591648"] * 7 + ["PMID:31978347"],
            "cell_line": ["HeLa"] * 8,
            "digestion_enzyme": ["Trypsin/P (cleaves K/R except before P)"] * 8,
            "n_fractions_in_run": [46, 14, 50, 12, 12, 46, 99, 12],
            "enrichment": ["none", "none", "TiO2", "TiO2", "TiO2", "none", "none", "none"],
            "fractionation_ph": [10.0, 10.0, 10.0, 8.0, 10.0, 6.0, 10.0, 10.0],
            "peptide": list("ABCDEFGH"),
            "log2_intensity": [1.25] * 8,
        }
    )


def test_reviewed_buffer_labels_unknowns_and_idempotence():
    before = legacy_rows()
    after = curate_fractionation_ph(before)
    assert after.loc[[0, 3, 4, 7], "fractionation_ph"].tolist() == [8.0, 8.0, 10.0, 10.0]
    assert after.loc[[1, 2, 5, 6], "fractionation_ph"].isna().all()
    assert after.fractionation_ph_basis.tolist() == [
        "paper_reported_buffer",
        "unresolved_reused_protocol",
        "unresolved_scope_mapping",
        "deposit_experiment_label",
        "deposit_experiment_label",
        "unresolved_scope",
        "unresolved_scope",
        "",
    ]
    assert (
        after.legacy_fractionation_ph.iloc[:7].tolist() == before.fractionation_ph.iloc[:7].tolist()
    )
    assert pd.isna(after.legacy_fractionation_ph.iloc[7])
    assert after.loc[0, "fractionation_ph_reference"].endswith("#sec4.4.4")
    pd.testing.assert_frame_equal(curate_fractionation_ph(after), after)
    columns = before.columns.drop("fractionation_ph")
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert _apply_ph_filter(after, 8).peptide.tolist() == ["A", "D"]
    assert _apply_ph_filter(after, 10).peptide.tolist() == ["E", "H"]
    assert len(_apply_ph_filter(after, None)) == len(before)


def test_cached_parquet_and_csv_projection_agree(tmp_path):
    before = legacy_rows()
    path = tmp_path / "old.parquet"
    before.rename(columns={"cell_line": "cell_line_name"}).to_parquet(path)
    stat = path.stat()
    actual = _read_parquet_cached(str(path), stat.st_mtime_ns, stat.st_size)
    expected = _curate_bj_metadata(before).rename(columns={"cell_line": "cell_line_name"})
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.reported_reference.iloc[0] == "PMID:28591648"
    assert actual.reference.iloc[0] == "PMID:28601559"


def test_string_axes_and_missing_scope_are_explicit():
    before = legacy_rows().astype(str)
    assert curate_fractionation_ph(before).fractionation_ph.iloc[0] == 8.0
    canonical = before.copy()
    canonical["digestion_enzyme"] = "Trypsin/P"
    assert curate_fractionation_ph(canonical).fractionation_ph.iloc[0] == 8.0
    incomplete = before.drop(columns="n_fractions_in_run")
    after = curate_fractionation_ph(incomplete)
    assert after.fractionation_ph.iloc[:7].isna().all()
    assert set(after.fractionation_ph_basis.iloc[:7]) == {"unresolved_scope"}


def test_ingest_uses_exact_reviewed_experiment_registry():
    path = Path(__file__).resolve().parents[1] / "scripts/ingest_bekker_jensen_2017.py"
    spec = importlib.util.spec_from_file_location("bekker_jensen_ingest_ph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for scope in fractionation_registry()["scopes"]:
        for name in scope["experiment_ids"]:
            assert module._parse_experiment(name)["fractionation_ph"] == scope["fractionation_ph"]
    assert experiment_fractionation_ph("Tryp-14Frac-A") is None
    assert experiment_fractionation_ph("Phospho-50fracs") is None
    with pytest.raises(ValueError, match="reviewed fractionation"):
        module._parse_experiment("Tryp-99fracs")
    assert module._parse_experiment("Colon-Pandey") is None


def test_packaged_unknown_or_old_default_is_not_a_qualified_scope():
    from hitlist.detectability import _packaged_observations

    with pytest.raises(ValueError, match="unresolved fractionation"):
        _packaged_observations("HeLa", "Trypsin/P", 14, "none", None, None, 100)
    with pytest.raises(ValueError, match="No curated"):
        _packaged_observations("HeLa", "Trypsin/P", 46, "none", 10.0, None, 100)


def test_ingest_unknown_ph_retains_rows_and_within_arm_abundance(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/ingest_bekker_jensen_2017.py"
    spec = importlib.util.spec_from_file_location("bekker_jensen_ingest_null_ph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = pd.DataFrame({column: ["", ""] for column in module.BASE_COLS})
    data["Sequence"] = ["AAAAAAK", "CCCCCCR"]
    data["Proteins"] = data["Leading razor protein"] = ["P1", "P2"]
    data["Gene names"] = ["G1", "G2"]
    data["Start position"], data["End position"], data["Length"] = 1, 7, 7
    for name in ("Tryp-14Frac-A", "Phospho-50fracs", "Tryp-46fracs"):
        data["Intensity " + name] = [10, 20]
    monkeypatch.setattr(
        module,
        "_fetch_peptides_txt_stream",
        lambda **kwargs: io.StringIO(data.to_csv(sep="\t", index=False)),
    )
    peptides, proteins = module.extract(verbose=True)
    assert len(peptides) == len(proteins) == 6
    for frame in (peptides, proteins):
        assert frame.fractionation_ph.isna().sum() == 4
        assert frame.loc[frame.n_fractions_in_run.eq(46), "fractionation_ph"].eq(8).all()
    assert sorted(proteins.abundance_percentile) == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
