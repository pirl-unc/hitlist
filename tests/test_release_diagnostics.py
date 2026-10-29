"""Partial and signal-terminated gates remain failures with durable evidence."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release_diagnostics.py"


@pytest.mark.parametrize("code", [0, 42, 143])
def test_gate_records_output_and_preserves_exit_status(tmp_path, code):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            f'print("last completed test", flush=True); raise SystemExit({code})',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == code
    receipt = json.loads((tmp_path / "gate.json").read_text())
    assert receipt["status"] == ("success" if code == 0 else "failed")
    assert receipt["returncode"] == code
    assert receipt["terminating_signal"] is None  # 143 is a shell status, not an OOM diagnosis
    assert "last completed test" in (tmp_path / "gate.log").read_text()
    assert receipt["before"]["files"]
    assert receipt["after"]["files"]


def test_gate_forwards_shutdown_and_retains_signal_evidence(tmp_path):
    ready = tmp_path / "ready"
    code = "from pathlib import Path; import sys, time; Path(sys.argv[1]).touch(); time.sleep(30)"
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), str(tmp_path), "--", sys.executable, "-c", code, str(ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        os.kill(process.pid, signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143
        receipt = json.loads((tmp_path / "gate.json").read_text())
        assert receipt["status"] == "failed"
        assert receipt["terminating_signal"] == signal.SIGTERM
        assert receipt["received_signals"][0]["signal"] == signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
