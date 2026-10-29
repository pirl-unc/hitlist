"""Timed-build audits reject attempted network access even if code catches it."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("paired_build_benchmark")
    monkeypatch.setattr(module, "GUARD_DIRECTORY", scripts / "benchmark_site")
    return module


def test_benchmark_audit_preserves_run_environment(tmp_path, benchmark):
    env = benchmark.audited_environment(
        tmp_path, {"HITLIST_DATA_DIR": "/run/cache", "PYTHONPATH": "/producer"}
    )
    assert env["HITLIST_DATA_DIR"] == "/run/cache"
    assert env["PYTHONPATH"].endswith(os.pathsep + "/producer")


def test_network_audit_allows_local_ipc(tmp_path, benchmark):
    code = "import socket; a,b=socket.socketpair(); a.send(b'x'); assert b.recv(1)==b'x'; a.close(); b.close()"
    process = subprocess.Popen(
        [sys.executable, "-c", code], env=benchmark.audited_environment(tmp_path)
    )
    assert process.wait(timeout=20) == 0
    benchmark.check_network_audit(tmp_path, process.pid)


@pytest.mark.parametrize(
    "operation",
    ["socket.getaddrinfo('example.invalid', 443)", "socket.socket().connect(('127.0.0.1', 9))"],
)
def test_caught_network_attempt_still_invalidates_benchmark(tmp_path, benchmark, operation):
    code = f"import socket\ntry:\n {operation}\nexcept RuntimeError:\n pass\n"
    process = subprocess.Popen(
        [sys.executable, "-c", code], env=benchmark.audited_environment(tmp_path)
    )
    assert process.wait(timeout=20) == 0
    with pytest.raises(ValueError, match="attempted network"):
        benchmark.check_network_audit(tmp_path, process.pid)


def test_network_guard_reaches_child_interpreters(tmp_path, benchmark):
    child = (
        "import socket\ntry: socket.getaddrinfo('example.invalid', 80)\nexcept RuntimeError: pass"
    )
    parent = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}],check=True)"
    process = subprocess.Popen(
        [sys.executable, "-c", parent], env=benchmark.audited_environment(tmp_path)
    )
    assert process.wait(timeout=20) == 0
    assert len(list(tmp_path.glob("*.installed"))) == 2
    with pytest.raises(ValueError, match="attempted network"):
        benchmark.check_network_audit(tmp_path, process.pid)


def test_missing_audit_marker_is_not_a_verified_run(tmp_path, benchmark):
    with pytest.raises(ValueError, match="did not install"):
        benchmark.check_network_audit(tmp_path, 123)


def test_scan_diagnostic_uses_builder_pmid_projection_and_keeps_originals(
    tmp_path, benchmark, monkeypatch
):
    import json

    import pandas as pd

    from hitlist import scanner
    from tests.test_provenance import _row
    from tests.test_scanner import _write_tiny_iedb_csv

    monkeypatch.setattr(scanner, "_scan_workers", lambda requested: 1)
    source = tmp_path / "iedb.csv"
    rows = [_row("integer"), _row("missing"), _row("noninteger")]
    rows[1][2], rows[2][2] = "", "reported-reference"
    _write_tiny_iedb_csv(source, rows)
    output = tmp_path / "result"
    output.mkdir()
    benchmark.scan_child(source, output)
    frame = pd.read_parquet(output / "observations.parquet")
    assert str(frame.pmid.dtype) == "Int64"
    assert frame.pmid.notna().sum() == 1
    links = pd.read_parquet(output / "contributors.parquet").sort_values("source_row")
    assert [json.loads(value)["pmid"] for value in links.original_fields] == [
        "99999999",
        "",
        "reported-reference",
    ]
    assert json.loads((output / "report.json").read_text())["status"] == "success"
