"""Retain release-gate output, resource snapshots and exit/signal evidence.

These are diagnostics, never approval to publish. A runner shutdown can prevent
post-job uploads, so line-buffered live Actions logs remain essential evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def snapshot():
    files = ["/proc/meminfo", "/proc/pressure/memory", "/proc/self/cgroup"]
    cgroup = Path("/sys/fs/cgroup")
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            if line.startswith("0::"):
                candidate = cgroup / line[3:].lstrip("/")
                if candidate.is_dir():
                    cgroup = candidate
    files += [
        str(cgroup / name)
        for name in (
            "memory.current",
            "memory.peak",
            "memory.max",
            "memory.events",
            "memory.swap.current",
            "memory.swap.max",
            "memory.swap.events",
        )
    ]
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "files": {}}
    for filename in files:
        try:
            result["files"][filename] = Path(filename).read_text()
        except OSError as error:
            result["files"][filename] = {"unavailable": str(error)}
    return result


def write_json(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_gate(directory, command):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "running",
        "command": command,
        "received_signals": [],
        "before": snapshot(),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_commit": os.environ.get("GITHUB_SHA"),
    }
    receipt_path = directory / "gate.json"
    write_json(receipt_path, receipt)
    started = time.monotonic()
    process = None
    previous = {}

    def forward(number, _frame):
        receipt["received_signals"].append(
            {"signal": number, "elapsed_seconds": time.monotonic() - started}
        )
        write_json(receipt_path, receipt)
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, number)

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, forward)
        with (directory / "gate.log").open("w", buffering=1) as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            returncode = process.wait()
        receipt["returncode"] = returncode
        receipt["terminating_signal"] = -returncode if returncode < 0 else None
        # Shell exit 143 alone is not proof of either SIGTERM provenance or OOM.
        receipt["status"] = (
            "success" if returncode == 0 and not receipt["received_signals"] else "failed"
        )
        if receipt["received_signals"] and returncode == 0:
            return 128 + receipt["received_signals"][0]["signal"]
        return 128 - returncode if returncode < 0 else returncode
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        receipt["after"] = snapshot()
        receipt["wall_seconds"] = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt["children"] = {
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "peak_process_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "rss_scope": "largest child process high-water mark; not simultaneous tree total",
        }
        write_json(receipt_path, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a release gate command is required")
    sys.exit(run_gate(args.directory, command))
