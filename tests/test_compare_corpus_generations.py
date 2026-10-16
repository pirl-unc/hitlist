import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_corpus_generations",
    Path(__file__).resolve().parents[1] / "scripts/compare_corpus_generations.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path):
    old = pd.DataFrame({key: ["first", "second"] for key in audit.CONTEXT})
    old["cell_line_name"] = pd.Categorical(["unknown", None])
    old["quantitative_value"] = [float("nan"), 3.5]
    new = old.copy()
    new["cell_line_name"] = pd.Categorical(["HCT116", None])
    outputs = {
        name: tmp_path / name
        for name in ("changed-cells.tsv", "field-counts.tsv", "study-counts.tsv")
    }
    return old, new, outputs


def test_changed_cells_keep_sources_and_nulls_are_equal(tmp_path):
    old, new, outputs = fixture(tmp_path)
    old.to_parquet(tmp_path / "old.parquet")
    new.to_parquet(tmp_path / "new.parquet")
    result = audit.compare(tmp_path / "old.parquet", tmp_path / "new.parquet", outputs)
    assert result["n_observations"] == 2
    assert result["n_changed_observations"] == result["n_changed_cells"] == 1
    assert result["changed_fields"] == {"cell_line_name": 1}
    changed = pd.read_csv(outputs["changed-cells.tsv"], sep="\t")
    assert changed.iloc[0].reference_iri == "first"
    assert changed.iloc[0].original_value == '"unknown"'
    assert changed.iloc[0].corrected_value == '"HCT116"'


@pytest.mark.parametrize("mismatch", ["order", "count", "schema"])
def test_nonaligned_inputs_fail_instead_of_claiming_unchanged_ids(tmp_path, mismatch):
    old, new, outputs = fixture(tmp_path)
    if mismatch == "order":
        new = new.iloc[::-1].reset_index(drop=True)
    elif mismatch == "count":
        new = new.iloc[:1]
    else:
        new = new.drop(columns="cell_line_name")
    old.to_parquet(tmp_path / "old.parquet")
    new.to_parquet(tmp_path / "new.parquet")
    with pytest.raises(ValueError, match=r"differ|changed"):
        audit.compare(tmp_path / "old.parquet", tmp_path / "new.parquet", outputs)


def test_native_training_observation_ids_are_validated_and_retained(tmp_path):
    old, new, outputs = fixture(tmp_path)
    old["evidence_row_id"] = new["evidence_row_id"] = ["ms:1", "ms:2"]
    old.to_parquet(tmp_path / "old.parquet")
    new.to_parquet(tmp_path / "new.parquet")
    result = audit.compare(tmp_path / "old.parquet", tmp_path / "new.parquet", outputs)
    assert "evidence_row_id" in result["identity_fields"]
    changes = pd.read_csv(outputs["changed-cells.tsv"], sep="\t")
    assert changes.iloc[0].evidence_row_id == "ms:1"
    new.loc[0, "evidence_row_id"] = "different-observation"
    new.to_parquet(tmp_path / "new.parquet")
    with pytest.raises(ValueError, match="identity/order changed"):
        audit.compare(tmp_path / "old.parquet", tmp_path / "new.parquet", outputs)
