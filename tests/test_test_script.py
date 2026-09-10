"""Exercise test.sh resource decisions without launching pytest recursively."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("n_free_pages", "n_speculative_pages", "probe_fails", "worker_min", "worker_max", "expected"),
    [
        (100_000, 10_000, False, 1, 0, 1),
        (100_000, 400_000, False, 1, 0, 3),
        (100_000, 400_000, True, 1, 0, 1),
        (100_000, 400_000, False, 10, 2, 2),
    ],
)
def test_worker_sizing_and_python_selection(
    tmp_path, n_free_pages, n_speculative_pages, probe_fails, worker_min, worker_max, expected
):
    stubs = {
        "uname": "echo Darwin",
        "getconf": "echo 16",
        "sysctl": "exit 1" if probe_fails else "echo 16384",
        "vm_stat": (
            f"echo 'Pages free: {n_free_pages}.'\n"
            "echo 'Pages inactive: 2000000.'\n"
            f"echo 'Pages speculative: {n_speculative_pages}.'"
        ),
        "python": 'if [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$@"',
        # A different interpreter's pytest must never be selected from PATH.
        "pytest": "exit 99",
    }
    for name, body in stubs.items():
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\n{body}\n")
        executable.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        PER_WORKER_GB="2.5",
        TEST_SH_MIN=str(worker_min),
        TEST_SH_MAX=str(worker_max),
    )
    script = Path(__file__).resolve().parents[1] / "test.sh"
    result = subprocess.run(["bash", str(script), "--all"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[:4] == ["-m", "pytest", "-n", str(expected)]
    assert "not integration" not in args
    assert "--all" not in args
    assert "--cov=hitlist/" in args
