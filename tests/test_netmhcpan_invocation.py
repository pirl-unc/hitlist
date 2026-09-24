"""NetMHCpan input batches remain private and failed commands cannot emit scores."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from hitlist import predict


@pytest.fixture(autouse=True)
def isolated_temporary_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    # Also isolate the old hard-coded path when verifying these tests before
    # the fix. Real per-invocation directories continue to use pathlib.Path.
    monkeypatch.setattr(predict, "Path", lambda path: tmp_path if path == "/tmp" else Path(path))


def _pairs(peptide):
    return pd.DataFrame({"peptide": [peptide], "allele": ["HLA-A*02:01"]})


def _output(peptide):
    return f"1 HLA-A02:01 {peptide} core 0 0 0 0 0 core PEPLIST 0.9 0.1 0.5 0.5 50"


def test_overlapping_same_allele_requests_keep_their_own_peptides(monkeypatch):
    paths = []
    nested_results = []

    def run(args, **kwargs):
        path = Path(args[args.index("-p") + 1])
        paths.append(path)
        peptide = path.read_text().strip()
        if peptide == "AAAAAAAAA":
            nested_results.append(predict._predict_netmhcpan(_pairs("CCCCCCCCC")))
        return subprocess.CompletedProcess(args, 0, stdout=_output(path.read_text().strip()))

    monkeypatch.setattr(predict.subprocess, "run", run)
    result = predict._predict_netmhcpan(_pairs("AAAAAAAAA"))

    assert result["peptide"].tolist() == ["AAAAAAAAA"]
    assert nested_results[0]["peptide"].tolist() == ["CCCCCCCCC"]
    assert len(set(paths)) == 2
    assert all(not path.exists() for path in paths)


def test_successful_prediction_removes_its_input(monkeypatch):
    paths = []

    def run(args, **kwargs):
        path = Path(args[args.index("-p") + 1])
        paths.append(path)
        return subprocess.CompletedProcess(args, 0, stdout=_output(path.read_text().strip()))

    monkeypatch.setattr(predict.subprocess, "run", run)
    result = predict._predict_netmhcpan(_pairs("AAAAAAAAA"))

    assert result["peptide"].tolist() == ["AAAAAAAAA"]
    assert result["presentation_percentile"].tolist() == [0.1]
    assert paths and all(not path.exists() for path in paths)


def test_nonzero_exit_rejects_partial_predictions_and_removes_input(monkeypatch):
    paths = []
    real_run = subprocess.run

    def run(args, **kwargs):
        paths.append(Path(args[args.index("-p") + 1]))
        # Execute a real failing process so this tests subprocess exit handling,
        # rather than a mock that merely checks for the implementation keyword.
        code = f"import sys; print({_output('AAAAAAAAA')!r}); sys.exit(7)"
        return real_run([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(predict.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        predict._predict_netmhcpan(_pairs("AAAAAAAAA"))

    assert exc.value.returncode == 7
    assert "PEPLIST" in exc.value.stdout
    assert paths and all(not path.exists() for path in paths)


@pytest.mark.parametrize("failure", ["timeout", "missing_executable"])
def test_launch_failure_removes_input(monkeypatch, failure):
    paths = []

    def run(args, **kwargs):
        paths.append(Path(args[args.index("-p") + 1]))
        assert paths[-1].read_text() == "AAAAAAAAA\n"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        raise FileNotFoundError("netMHCpan missing")

    monkeypatch.setattr(predict.subprocess, "run", run)
    error = subprocess.TimeoutExpired if failure == "timeout" else FileNotFoundError
    with pytest.raises(error):
        predict._predict_netmhcpan(_pairs("AAAAAAAAA"))

    assert paths and all(not path.exists() for path in paths)
