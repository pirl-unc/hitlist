"""A benchmark is not a speed result if its outputs silently changed."""

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_reliability_outputs",
    Path(__file__).resolve().parents[1] / "scripts/compare_reliability_outputs.py",
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write_run(root, *, flag=True, timestamp=1, count=2):
    output = root / "output"
    output.mkdir(parents=True)
    pq.write_table(
        pa.table({"source": ["☃\nsource", None], "curated": [flag, None]}),
        output / "observations.parquet",
    )
    (output / "observations_meta.json").write_text(
        json.dumps({"n_rows": count, "parquets": {"observations": {"mtime": timestamp}}})
    )
    (root / "outputs.json").write_text(
        json.dumps(
            {
                path.name: {
                    "sha256": comparison.digest(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in output.iterdir()
            }
        )
    )


def test_complete_comparison_records_only_expected_metadata_difference(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    write_run(before)
    write_run(after, timestamp=2)
    receipt = comparison.compare(before, after)
    assert receipt["files"]["observations.parquet"]["n_rows"] == 2
    differences = receipt["files"]["observations_meta.json"]["expected_differences"]
    assert len(differences) == 1
    assert differences[0]["before"] == 1 and differences[0]["after"] == 2


@pytest.mark.parametrize("change", ["curation", "count", "corrupt", "inventory"])
def test_semantic_changes_and_untrusted_output_fail_closed(tmp_path, change):
    before, after = tmp_path / "before", tmp_path / "after"
    write_run(before)
    write_run(after, flag=change != "curation", count=3 if change == "count" else 2)
    if change == "corrupt":
        (after / "output" / "observations.parquet").write_bytes(b"corrupted")
    elif change == "inventory":
        manifest = json.loads((after / "outputs.json").read_text())
        manifest.pop("observations_meta.json")
        (after / "outputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        comparison.compare(before, after)


def test_changed_source_filename_is_not_an_allowed_relocation():
    with pytest.raises(ValueError, match="Unexpected metadata difference"):
        comparison.metadata_differences(
            {"sources": {"iedb": {"path": "/a/source.csv"}}},
            {"sources": {"iedb": {"path": "/b/other.csv"}}},
            "observations_meta.json",
        )
