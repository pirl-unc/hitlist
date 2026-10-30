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


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal cleanup")
def test_interrupted_experiment_records_failure_and_reaps_child(tmp_path, benchmark):
    import json
    import signal
    import time

    config, base, producers = (tmp_path / name for name in ("config", "results", "producers"))
    for directory in (config, base, producers):
        directory.mkdir()
    for name in ("baseline-env", "candidate-env"):
        bin_directory = producers / name / "bin"
        bin_directory.mkdir(parents=True)
        (bin_directory / "python").symlink_to(sys.executable)
    for name in ("baseline.tar.gz", "producer.tar.gz", "sitecustomize.py"):
        (config / name).write_bytes(b"fixture")
    marker = tmp_path / "child.pid"
    (config / "reliability_benchmark.py").write_text(
        "import os,time\nfrom pathlib import Path\n"
        f"marker=Path({str(marker)!r})\ntemporary=marker.with_suffix('.tmp')\n"
        "temporary.write_text(str(os.getpid()))\ntemporary.replace(marker)\ntime.sleep(120)\n"
    )
    code = (
        f"import sys,signal\nsys.path.insert(0,{str(Path(benchmark.__file__).parent)!r})\n"
        "from pathlib import Path\nimport paired_build_benchmark as b\n"
        f"b.CONFIG=Path({str(config)!r}); b.BASE=Path({str(base)!r})\n"
        f"b.GUARD_DIRECTORY=b.CONFIG; b.PRODUCER_ROOT=Path({str(producers)!r})\n"
        "signal.signal(signal.SIGTERM,b.interrupted)\nb.run('interrupted',False)\n"
    )
    with (tmp_path / "parent.log").open("w") as log:
        parent = subprocess.Popen([sys.executable, "-c", code], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline:
                if parent.poll() is not None:
                    pytest.fail((tmp_path / "parent.log").read_text())
                time.sleep(0.02)
            assert marker.exists()
            child_pid = int(marker.read_text())
            parent.send_signal(signal.SIGTERM)
            assert parent.wait(timeout=15) != 0
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if parent.poll() is None:
                parent.send_signal(signal.SIGTERM)
                parent.wait(timeout=15)
    receipt = json.loads((base / "interrupted/experiment.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["active_run"] == "interrupted-1-baseline"
    assert receipt["runs"] == []
    assert "received signal 15" in (base / "interrupted/failure.txt").read_text()


def test_scan_diagnostic_uses_builder_pmid_projection_and_keeps_originals(tmp_path, benchmark):
    import json

    import pandas as pd

    from tests.test_provenance import _row
    from tests.test_scanner import _write_tiny_iedb_csv

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


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal cleanup")
@pytest.mark.parametrize("failure", ["sibling", "timeout"])
def test_source_diagnostic_reaps_all_children_on_failure(tmp_path, monkeypatch, benchmark, failure):
    source_benchmark = importlib.import_module("source_scan_benchmark")
    processes = []
    launch = subprocess.Popen

    def tracked(*args, **kwargs):
        process = launch(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(source_benchmark.subprocess, "Popen", tracked)
    with (tmp_path / "children.log").open("w") as log:
        commands = [([sys.executable, "-c", "import time; time.sleep(30)"], os.environ.copy(), log)]
        if failure == "sibling":
            commands.append(([sys.executable, "-c", "raise SystemExit(7)"], os.environ.copy(), log))
        with pytest.raises(subprocess.CalledProcessError if failure == "sibling" else TimeoutError):
            source_benchmark.run_children(commands, concurrent=True, timeout=0.5)
    assert processes
    assert all(process.poll() is not None for process in processes)


def test_source_diagnostic_preserves_cedar_identity(tmp_path, benchmark):
    import json

    import pandas as pd

    from tests.test_provenance import _row
    from tests.test_scanner import _write_tiny_iedb_csv

    source_benchmark = importlib.import_module("source_scan_benchmark")
    source = tmp_path / "cedar.csv"
    _write_tiny_iedb_csv(source, [_row("cedar:1")])
    output = tmp_path / "output"
    output.mkdir()
    source_benchmark.child("cedar", source, output)
    links = pd.read_parquet(output / "contributors.parquet")
    assert links.source_dataset.tolist() == ["cedar"]
    assert links.source_record_id.tolist() == ["record:cedar:row:1"]
    assert json.loads((output / "report.json").read_text())["status"] == "success"


def test_memory_sampler_counts_thread_groups_once_and_keeps_descendants(tmp_path, benchmark):
    profile = importlib.import_module("profile_build")
    # Tasks 2/3/4 share one address space. Task 3 alone owns process 5;
    # missing task 99 models an exit between the children read and inspection.
    for pid, group, pages, children in (
        (1, 1, 10, [2, 3, 4, 99]),
        (2, 2, 20, []),
        (3, 2, 20, [5]),
        (4, 2, 20, []),
        (5, 5, 30, []),
    ):
        path = tmp_path / str(pid)
        (path / f"task/{pid}").mkdir(parents=True)
        (path / f"task/{pid}/children").write_text(" ".join(map(str, children)))
        (path / "status").write_text(f"Name:\tpython\nTgid:\t{group}\nPid:\t{pid}\n")
        (path / "statm").write_text(f"100 {pages} 0 0 0 0 0\n")
    own, total = profile._process_tree_rss(1, tmp_path)
    assert own == 10 * os.sysconf("SC_PAGE_SIZE")
    assert total == 60 * os.sysconf("SC_PAGE_SIZE")
