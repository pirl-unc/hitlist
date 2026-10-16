"""Observation counts must not conflate peptides, missing metadata or MHC roles."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cohort_script():
    path = Path(__file__).parents[1] / "scripts" / "qualify_human_cohort.py"
    spec = importlib.util.spec_from_file_location("qualify_human_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort_rows():
    # Repeated peptide with distinct observations; cellular MHC sets and assay
    # restrictions intentionally differ. Missing units must remain visible.
    return pd.DataFrame(
        {
            "evidence_row_id": ["ms:1", "ms:2", "ms:3", "binding:4"],
            "peptide": ["AAAAAAAAA"] * 4,
            "evidence_kind": ["ms", "ms", "ms", "binding"],
            "mhc_class": ["I", "I", "II", "I"],
            "pmid": [1, 1, 2, 3],
            "mhc_allele_set": ["HLA-A*02:01,HLA-B*07:02"] * 3 + [""],
            "mhc_restriction": ["HLA-A*02:01", "", "HLA-DRB1*01:01", "HLA-A*02:01"],
            "measurement_units": [None, "", None, "nM"],
            "measurement_inequality": ["", "", "", ">"],
            "quantitative_value": [None, None, None, 5000],
            "donor_status": ["unknown", "unknown", "resolved", "unknown"],
        }
    )


def test_inspection_preserves_observation_counts_missingness_and_roles(tmp_path, cohort_script):
    cohort_rows().to_parquet(tmp_path / "training.parquet", index=False)
    report = cohort_script.inspection_reports(tmp_path)
    assert report["n_observations"] == report["n_training_rows"] == 4
    assert report["evidence_kinds"] == {"ms": 3, "binding": 1}
    assert report["lineage_statuses"]["donor_status"] == {"unknown": 3, "resolved": 1}
    assert report["fields"]["measurement_units"]["n_missing_observations"] == 3
    assert "cell_type" in report["unavailable_fields"]
    preview = pd.read_csv(tmp_path / "inspection.tsv", sep="\t", keep_default_na=False)
    assert len(preview) == 4
    assert (
        preview.loc[preview.evidence_row_id == "binding:4", "measurement_inequality"].item() == ">"
    )
    assert preview.loc[preview.evidence_row_id == "ms:2", "mhc_restriction"].item() == ""
    assert (
        preview.loc[preview.evidence_row_id == "ms:2", "mhc_allele_set"].item()
        == "HLA-A*02:01,HLA-B*07:02"
    )
    contexts = pd.read_csv(tmp_path / "source_contexts.tsv", sep="\t")
    assert contexts.n_observations.sum() == 4


def test_duplicate_observations_cannot_inflate_inspection_counts(tmp_path, cohort_script):
    frame = cohort_rows()
    pd.concat([frame, frame.iloc[:1]]).to_parquet(tmp_path / "training.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate observation IDs"):
        cohort_script.inspection_reports(tmp_path)


@pytest.mark.parametrize("status", ["success", "failed"])
def test_actual_receipt_materialization_preserves_success_and_failure(
    tmp_path, cohort_script, status
):
    import json

    from datacache import validate_file

    log = tmp_path / "export.log"
    log.write_text("observed export output\n")
    result = {"status": status, "error": "injected export failure" if status == "failed" else None}
    paths = cohort_script.publish_receipt(tmp_path / "receipts", result, log)
    assert json.loads(Path(paths["receipt.json"]).read_text()) == result
    assert Path(paths["export.log"]).read_bytes() == log.read_bytes()
    validate_file(paths["export.log"], cohort_script.digest(log)["sha256"])
