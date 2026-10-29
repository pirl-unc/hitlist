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


def write_generation(root, version):
    import hashlib
    import shutil
    from unittest.mock import patch

    from datacache import materialize

    from hitlist.dataset import _write_readme

    root.mkdir()
    source = root / "source.csv"
    source.write_text("source,flag\noriginal,true\n")
    pin = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size": source.stat().st_size}

    def build(_sources, outputs):
        directory = Path(outputs["README.md"]).parent
        pq.write_table(pa.table({"flag": [True, None]}), directory / "observations.parquet")
        (directory / "observations_meta.json").write_text(json.dumps({"n_rows": 2}))
        with patch("hitlist.version.__version__", version):
            _write_readme(directory)

    published = materialize(
        root / "generation",
        {"iedb": {"path": source, **pin}},
        transform={"version": f"hitlist-{version}", "options": {"force": True}},
        outputs={
            name: {} for name in ("observations.parquet", "observations_meta.json", "README.md")
        },
        builder=build,
    )
    output = root / "output"
    output.mkdir()
    for path in Path(published["observations.parquet"]).parent.iterdir():
        if path.suffix in (".json", ".parquet"):
            shutil.copyfile(path, output / path.name)
    (root / "report.json").write_text(
        json.dumps({"status": "success", "versions": {"hitlist": version}})
    )
    (root / "inputs.json").write_text(
        json.dumps({"sources": {"iedb": {"sha256": pin["sha256"], "size_bytes": pin["size"]}}})
    )
    update_inventory(root)


def update_inventory(root):
    (root / "outputs.json").write_text(
        json.dumps(
            {
                path.name: {"sha256": comparison.digest(path), "size_bytes": path.stat().st_size}
                for path in (root / "output").iterdir()
            }
        )
    )


def test_real_generation_receipts_allow_verified_versions_paths_and_timestamps(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    write_generation(before, "2.0.44")
    write_generation(after, "2.1.0")
    result = comparison.compare(before, after)
    assert result["generation_validation"]["after"]["producer_version"] == "2.1.0"
    assert result["generation_validation"]["before"]["readme"] == "reconstructed and hash-verified"


def test_inventory_paths_are_rejected_before_generation_reconstruction(tmp_path, monkeypatch):
    before, after = tmp_path / "before", tmp_path / "after"
    for root in (before, after):
        root.mkdir()
        (root / "outputs.json").write_text(
            json.dumps({".datacache-manifest.json": {}, "../outside.parquet": {}})
        )
    calls = []
    monkeypatch.setattr(comparison, "validate_generation", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Invalid output filename"):
        comparison.compare(before, after)
    assert calls == []


@pytest.mark.parametrize("change", ["source", "output", "readme", "version", "options"])
def test_generation_receipts_cannot_hide_changed_inputs_outputs_or_policy(tmp_path, change):
    before, after = tmp_path / "before", tmp_path / "after"
    write_generation(before, "2.0.44")
    write_generation(after, "2.1.0")
    path = after / "output" / ".datacache-manifest.json"
    manifest = json.loads(path.read_text())
    if change == "source":
        manifest["sources"]["iedb"]["sha256"] = "0" * 64
    elif change == "output":
        manifest["outputs"]["observations.parquet"]["sha256"] = "0" * 64
    elif change == "readme":
        manifest["outputs"]["README.md"]["sha256"] = "0" * 64
    elif change == "version":
        manifest["definition"]["transform"]["version"] = "hitlist-99"
    else:
        manifest["definition"]["transform"]["options"]["force"] = False
    path.write_text(json.dumps(manifest))
    update_inventory(after)
    with pytest.raises(ValueError):
        comparison.compare(before, after)
