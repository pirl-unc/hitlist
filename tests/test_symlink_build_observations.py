"""Regression tests for #954: symlinked IEDB/CEDAR source paths in build_observations/publish_dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hitlist import builder, dataset, downloads, mappings, provenance
from hitlist.parquet_io import atomic_write_parquet
from tests.test_build_smoke import _write_synthetic_iedb


def test_build_observations_succeeds_with_symlinked_manifest_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_curation_root: Path
) -> None:
    """End-to-end build_observations + datacache.materialize(O_NOFOLLOW) works on symlinks (#954)."""
    from hitlist import supplement

    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n", encoding="utf-8")
    real_dir = tmp_path / "shared_exports"
    real_dir.mkdir()
    real_csv = real_dir / "cedar-mhc-ligand-full.csv"
    _write_synthetic_iedb(real_csv)

    link_csv = tmp_path / "cedar-symlink.csv"
    link_csv.symlink_to(real_csv)
    assert link_csv.is_symlink()

    data_root = downloads.data_dir()
    data_root.mkdir(parents=True, exist_ok=True)
    # Simulate a manifest.json holding an unresolved symlink path directly (exact #954 trigger).
    (data_root / "manifest.json").write_text(
        json.dumps({"datasets": {"cedar": {"path": str(link_csv)}}}),
        encoding="utf-8",
    )

    def empty_index(path: Path) -> pd.DataFrame:
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(
        builder,
        "build_bulk_proteomics",
        lambda **kw: empty_index(builder._bulk_proteomics_path()),
    )
    monkeypatch.setattr(
        builder,
        "build_line_expression",
        lambda **kw: empty_index(builder._line_expression_path()),
    )

    def tiny_mappings(**kwargs):
        path = mappings.mappings_path()
        frame = kwargs.get("obs_override")
        if frame is None:
            frame = pd.read_parquet(builder._observations_path())
        atomic_write_parquet(frame[["peptide"]], path)
        mappings.mappings_meta_path().write_text(
            json.dumps({"observations": mappings._obs_fingerprint()}),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(mappings, "build_peptide_mappings", tiny_mappings)

    obs_path = builder.build_observations(
        force=True,
        build_mappings=True,
        with_flanking=False,
        fetch_missing_proteomes=False,
    )
    assert obs_path.is_file()
    meta = json.loads((obs_path.parent / "observations_meta.json").read_text(encoding="utf-8"))
    # Metadata still records the registered manifest path for freshness checks.
    assert meta["provenance"]["sources"]["cedar"]["path"] == str(link_csv)


def test_publish_dataset_rejects_broken_symlink_and_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "hitlist_data"
    root.mkdir()

    broken = tmp_path / "broken.csv"
    broken.symlink_to(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        dataset.publish_dataset(
            {"iedb": {"path": str(broken)}},
            ("observations.parquet",),
            lambda _inputs: None,
            root=root,
        )

    target_dir = tmp_path / "a_directory"
    target_dir.mkdir()
    dir_link = tmp_path / "dir_link.csv"
    dir_link.symlink_to(target_dir)
    with pytest.raises(ValueError, match="not a regular file"):
        dataset.publish_dataset(
            {"iedb": {"path": str(dir_link)}},
            ("observations.parquet",),
            lambda _inputs: None,
            root=root,
        )


def test_build_observations_rejects_broken_symlink_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Broken symlinks fail before file_digest is invoked (#954)."""
    data_root = downloads.data_dir()
    data_root.mkdir(parents=True, exist_ok=True)

    broken = tmp_path / "broken_iedb.csv"
    broken.symlink_to(tmp_path / "missing_target.csv")
    (data_root / "manifest.json").write_text(
        json.dumps({"datasets": {"iedb": {"path": str(broken)}}}),
        encoding="utf-8",
    )

    def boom(_path):
        raise AssertionError("file_digest should not be called on a broken symlink")

    monkeypatch.setattr(provenance, "file_digest", boom)
    with pytest.raises(FileNotFoundError):
        builder.build_observations(
            force=True,
            build_mappings=False,
            with_flanking=False,
        )


def test_build_observations_passes_resolved_regular_path_to_publish_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid symlinks resolve to canonical regular files before publish_dataset (#954)."""
    data_root = downloads.data_dir()
    data_root.mkdir(parents=True, exist_ok=True)
    real_csv = tmp_path / "real_cedar.csv"
    real_csv.write_text("header\nrow\n", encoding="utf-8")
    link_csv = tmp_path / "linked_cedar.csv"
    link_csv.symlink_to(real_csv)
    (data_root / "manifest.json").write_text(
        json.dumps({"datasets": {"cedar": {"path": str(link_csv)}}}),
        encoding="utf-8",
    )

    captured_sources: dict[str, dict] = {}

    def spy_publish(sources, names, build, *, root=None, options=None):
        captured_sources.update(sources)
        return {"observations.parquet": data_root / "observations.parquet"}

    monkeypatch.setattr(dataset, "publish_dataset", spy_publish)

    out = builder.build_observations(
        force=True,
        build_mappings=False,
        with_flanking=False,
    )
    assert out == data_root / "observations.parquet"
    resolved_passed = Path(captured_sources["cedar"]["path"])
    assert resolved_passed == real_csv.resolve()
    assert not resolved_passed.is_symlink()
