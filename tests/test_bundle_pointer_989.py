"""Tests for active-bundle pointer, latest_bundle(), stale-cache warnings, and `hitlist path` (#989)."""

from __future__ import annotations

import json
import os
import sys
import warnings

import pandas as pd
import pytest

import hitlist
from hitlist import builder, cli, dataset, downloads, mappings
from hitlist.errors import DatasetError
from hitlist.parquet_io import atomic_write_parquet
from tests.test_build_smoke import _write_synthetic_iedb


@pytest.fixture
def tiny_published_cache(tmp_path, monkeypatch, _isolated_curation_root):
    from hitlist import supplement

    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    source = tmp_path / "iedb.csv"
    _write_synthetic_iedb(source)
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())

    def empty_index(path):
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder, "build_bulk_proteomics", lambda **kw: empty_index(builder._bulk_proteomics_path())
    )
    monkeypatch.setattr(
        builder, "build_line_expression", lambda **kw: empty_index(builder._line_expression_path())
    )

    def tiny_mappings(**kwargs):
        path = mappings.mappings_path()
        frame = kwargs.get("obs_override")
        if frame is None:
            frame = pd.read_parquet(builder._observations_path())
        atomic_write_parquet(frame[["peptide"]], path)
        mappings.mappings_meta_path().write_text(
            json.dumps({"observations": mappings._obs_fingerprint()})
        )
        return path

    monkeypatch.setattr(mappings, "build_peptide_mappings", tiny_mappings)
    dataset._reset_stale_root_warnings()
    return source


def test_latest_bundle_and_current_pointer_update_atomically(tiny_published_cache):
    first_obs = builder.build_observations()
    first_bundle = first_obs.parent

    assert hitlist.latest_bundle() == first_bundle
    assert dataset.bundle_path("observations") == first_obs
    assert dataset.bundle_path("ms") == first_obs
    assert dataset.bundle_path("binding") == first_bundle / "binding.parquet"
    assert dataset.bundle_path("observations.parquet", must_exist=True) == first_obs

    pointer = downloads.data_dir() / dataset.STORE_NAME / "bundles" / "current"
    assert pointer.is_symlink()
    assert os.readlink(pointer) == first_bundle.name
    assert pointer.resolve() == first_bundle.resolve()

    # Rebuild publishes a second generation and swaps `current` atomically.
    tiny_published_cache.write_text(
        tiny_published_cache.read_text().replace("33858848", "99858848")
    )
    second_obs = builder.build_observations(force=True)
    second_bundle = second_obs.parent
    assert second_bundle != first_bundle
    assert hitlist.latest_bundle() == second_bundle
    assert os.readlink(pointer) == second_bundle.name
    assert pointer.resolve() == second_bundle.resolve()

    # `prune_cache_bundles` must not count the `current` symlink as an extra bundle.
    plan = downloads.prune_cache_bundles(keep=1, apply=False)
    assert plan["n_prunable_bundles"] == 1


def test_current_pointer_falls_back_to_json_when_symlinks_unavailable(
    tiny_published_cache, monkeypatch
):
    def fail_symlink(*_args, **_kwargs):
        raise OSError("symlinks disabled")

    monkeypatch.setattr(os, "symlink", fail_symlink)
    first_obs = builder.build_observations()
    fallback = downloads.data_dir() / dataset.STORE_NAME / "bundles" / "current.json"
    assert fallback.is_file()
    payload = json.loads(fallback.read_text(encoding="utf-8"))
    assert payload["bundle"] == first_obs.parent.name
    assert payload["path"] == str(first_obs.parent)


def test_stale_root_files_detected_and_warned_once(tiny_published_cache):
    first_obs = builder.build_observations()
    root = downloads.data_dir()
    assert dataset.stale_root_files(root=root) == []

    # Drop a stale legacy root-level observations.parquet beside the published bundle.
    stale_obs = root / "observations.parquet"
    stale_obs.write_bytes(b"stale")
    assert dataset.stale_root_files(root=root) == [stale_obs]

    dataset._reset_stale_root_warnings()
    with pytest.warns(UserWarning, match="Stale root-level cache files"):
        resolved = hitlist.latest_bundle(root=root)
    assert resolved == first_obs.parent

    # Second call for the same (root, bundle) does not spam repeated warnings.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert hitlist.latest_bundle(root=root) == first_obs.parent
    assert not caught


def test_latest_bundle_requires_published_generation_unless_legacy(tmp_path):
    root = tmp_path / "empty_cache"
    with pytest.raises(dataset.DatasetGenerationError, match="No published Hitlist dataset bundle"):
        hitlist.latest_bundle(root=root)
    assert not root.exists()

    # Legacy flat cache is refused by default and accepted with allow_legacy=True.
    root.mkdir()
    (root / "observations.parquet").write_bytes(b"legacy")
    assert dataset.stale_root_files(root=root) == []
    with pytest.raises(dataset.DatasetGenerationError):
        hitlist.latest_bundle(root=root)
    assert hitlist.latest_bundle(root=root, allow_legacy=True) == root


def test_cache_layout_is_read_only(tmp_path):
    root = tmp_path / "fresh"
    layout = dataset.cache_layout(root=root)
    assert layout["root"] == root.absolute()
    assert layout["raw"] == root.absolute() / "raw"
    assert layout["proteomes"] == root.absolute() / "proteomes"
    assert layout["bundles"] == root.absolute() / "dataset" / "bundles"
    assert layout["current"] == root.absolute() / "dataset" / "bundles" / "current"
    assert layout["scratch"] == root.absolute() / "scratch"
    assert not root.exists()


def test_bundle_path_rejects_traversal_and_missing_when_checked(tiny_published_cache):
    builder.build_observations()
    with pytest.raises(DatasetError, match="Invalid bundle table name"):
        dataset.bundle_path("../escape.parquet")
    with pytest.raises(FileNotFoundError):
        dataset.bundle_path("nonexistent.parquet", must_exist=True)


def test_cli_path_command(tiny_published_cache, monkeypatch, capsys):
    obs = builder.build_observations()
    capsys.readouterr()  # discard build progress output
    root = downloads.data_dir()

    monkeypatch.setattr(sys, "argv", ["hitlist", "path"])
    cli.main()
    assert capsys.readouterr().out.strip() == str(obs.parent)

    monkeypatch.setattr(sys, "argv", ["hitlist", "path", "observations", "--check"])
    cli.main()
    assert capsys.readouterr().out.strip() == str(obs)

    # Stale root file triggers stderr warning without corrupting stdout path.
    (root / "observations.parquet").write_bytes(b"stale")
    dataset._reset_stale_root_warnings()
    monkeypatch.setattr(sys, "argv", ["hitlist", "path", "binding"])
    cli.main()
    captured = capsys.readouterr()
    assert captured.out.strip() == str(obs.parent / "binding.parquet")
    assert "stale root-level cache files" in captured.err.lower()

    # Missing file with --check exits 1 cleanly via HitlistError.
    monkeypatch.setattr(sys, "argv", ["hitlist", "path", "nonexistent.parquet", "--check"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    assert "does not exist" in capsys.readouterr().err
