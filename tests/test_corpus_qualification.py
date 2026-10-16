"""Qualification receipts retain failed attempts and reject changed prerequisites."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_corpus.py"
    spec = importlib.util.spec_from_file_location("qualification_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "recipe"
    config.mkdir()
    (config / "requirements.txt").write_text("hitlist==1.76.0\n")
    monkeypatch.setattr(module, "CONFIG", config)
    monkeypatch.setattr(module, "BASE", tmp_path / "shared")
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    monkeypatch.setattr(module.signal, "alarm", lambda *_: None)
    for name in (
        "HITLIST_DATA_DIR",
        "XDG_CACHE_HOME",
        "PYENSEMBL_CACHE_DIR",
        "HITLIST_BUILD_WORKERS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(name, str(tmp_path / "unused"))
    return module


def test_changed_prerequisite_is_rejected(runner, tmp_path, monkeypatch):
    artifact = tmp_path / "source.csv"
    artifact.write_text("captured evidence\n")
    monkeypatch.setattr(runner, "perform", lambda *_: ({"source": runner.digest(artifact)}, {}))
    result = runner.run("sources")
    root = runner.BASE / "runs" / result["receipt"]["run_id"]
    assert runner.previous(root, "sources")["status"] == "success"
    artifact.write_text("changed evidence\n")
    with pytest.raises(ValueError, match="Content mismatch"):
        runner.previous(root, "sources")


def test_failed_stage_retains_previous_generation_and_failure_log(runner, monkeypatch):
    monkeypatch.setattr(runner, "perform", lambda *_: ({}, {"qualification": "fixture"}))
    first = runner.run("indexes")
    old = Path(first["receipt_files"]["receipt.json"]["path"])
    original = old.read_bytes()

    def failed(*_):
        print("Builder reached the injected interruption")
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(runner, "perform", failed)
    with pytest.raises(RuntimeError, match="retained receipt"):
        runner.run("indexes")
    assert old.read_bytes() == original
    root = runner.BASE / "runs" / first["receipt"]["run_id"]
    from datacache import inspect_materialization

    state = inspect_materialization(root / "receipts" / "indexes", verify_files=True)
    current = Path(state.bundle)
    assert current != old.parent
    assert json.loads((current / "receipt.json").read_text())["status"] == "failed"
    assert "injected build failure" in (current / "build.log").read_text()
    with pytest.raises(ValueError, match="did not succeed"):
        runner.previous(root, "indexes")


def test_local_reference_is_verified_and_cannot_mutate_shared_input(runner, tmp_path):
    source = tmp_path / "captured.fa"
    source.write_text(">captured\nACDEFGHIK\n")
    original = os.environ["PYENSEMBL_CACHE_DIR"]
    artifacts = {source.name: runner.digest(source)}
    with runner.local_reference(artifacts, tmp_path / "scratch") as genome:
        local = Path(genome.download_cache.cache_directory_path) / source.name
        assert local.read_bytes() == source.read_bytes()
        assert local != source
        local.write_text("local indexing work\n")
    assert source.read_text() == ">captured\nACDEFGHIK\n"
    assert os.environ["PYENSEMBL_CACHE_DIR"] == original
    artifacts[source.name]["sha256"] = "0" * 64
    with (
        pytest.raises(ValueError, match="Content mismatch"),
        runner.local_reference(artifacts, tmp_path / "bad-scratch"),
    ):
        pytest.fail("Unverified reference must never reach the mapping stage")
    assert os.environ["PYENSEMBL_CACHE_DIR"] == original
