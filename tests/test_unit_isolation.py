"""A populated configured corpus must not change ordinary test behavior."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_unit_fixture_isolates_corpus_but_integration_keeps_it(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "observations.parquet").write_bytes(b"sentinel; never read")
    with (corpus / "reference.csv").open("wb") as handle:
        handle.write(b"key\nvalue\n")
        handle.seek(11 * 1024**2)
        handle.write(b"\n")
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copyfile(Path(__file__).with_name("conftest.py"), suite / "conftest.py")
    (suite / "test_probe.py").write_text("""
import os
from pathlib import Path
import pytest
import pandas as pd
from hitlist import downloads
from hitlist.observations import is_built

def test_unit():
    assert downloads.data_dir() != Path(os.environ["HITLIST_DATA_DIR"])
    assert not is_built()

@pytest.mark.integration
def test_integration():
    assert downloads.data_dir() == Path(os.environ["HITLIST_DATA_DIR"])
    assert is_built()

def test_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    assert downloads.data_dir() == tmp_path

def test_reference_read_is_blocked_before_allocation():
    with pytest.raises(RuntimeError, match="requires an integration test"):
        pd.read_csv(Path(os.environ["HITLIST_DATA_DIR"]) / "reference.csv")
""")
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-c",
            str(project / "pyproject.toml"),
            str(suite),
        ],
        cwd=suite,
        env=dict(
            os.environ, HITLIST_DATA_DIR=str(corpus), PYTHONPATH=str(project), PYTEST_ADDOPTS=""
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout
