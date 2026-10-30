"""Compare independent source scheduling; excludes a production cross-source merge."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from paired_build_benchmark import audited_environment, check_network_audit
from profile_build import MemorySampler, elapsed, usage
from reliability_benchmark import BASE, CONFIG, digest


def run_children(commands, *, concurrent, timeout):
    """Wait for owned process groups, cancelling every sibling on failure."""
    processes = []
    deadline = time.monotonic() + timeout
    try:
        for command, environment, log in commands:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            if not concurrent:
                code = process.wait(timeout=max(0, deadline - time.monotonic()))
                if code:
                    raise subprocess.CalledProcessError(code, command)
        pending = list(processes)
        while pending:
            for process in list(pending):
                code = process.poll()
                if code is None:
                    continue
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
                pending.remove(process)
            if time.monotonic() >= deadline:
                raise TimeoutError("Independent-source diagnostic exceeded its deadline")
            if pending:
                time.sleep(0.1)
        return [process.pid for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def child(source, sample, destination):
    import pandas as pd

    import hitlist
    from hitlist.provenance import ContributorCollector
    from hitlist.scanner import scan

    destination = Path(destination)
    receipt = {"status": "failed", "source": source, "version": hitlist.__version__}
    before, memory = usage(), MemorySampler()
    try:
        with memory, ContributorCollector(max_scratch_bytes=4 * 1024**3) as collector:
            frame = scan(**{f"{source}_path": sample}, mhc_species=None, provenance=collector)
            collector.write([frame], destination / "contributors.parquet")
            frame["pmid"] = pd.to_numeric(frame["pmid"], errors="coerce").astype("Int64")
            frame.to_parquet(destination / "observations.parquet", index=False)
            receipt["scratch_bytes"] = (
                collector.db.execute("PRAGMA page_count").fetchone()[0] * 4096
            )
        receipt.update(
            status="success", n_rows=len(frame), total={**elapsed(before), **memory.total}
        )
        receipt["outputs"] = {
            name: digest(destination / name)
            for name in ("contributors.parquet", "observations.parquet")
        }
    finally:
        (destination / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")


def run(label, n_rows):
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Invalid experiment label")
    if not 1 <= n_rows <= 500000:
        raise ValueError("This diagnostic is limited to 500,000 rows per source")
    destination = BASE / label
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "failed",
        "scope": "independent source processes; excludes production cross-source graph/partition merge",
        "n_rows_per_source_limit": n_rows,
        "order": ["serial", "parallel", "parallel", "serial"],
        "producer": digest(CONFIG / "producer.tar.gz"),
        "recipe": digest(Path(__file__)),
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "allocation": {"cpu_limit": 8, "memory_limit_mib": 65536, "nonpreemptible": True},
        "runs": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="hitlist-source-pair-") as temporary:
            scratch = Path(temporary)
            pins = json.loads((CONFIG / "sources.json").read_text())
            receipt["sources"], receipt["samples"] = {}, {}
            for source in ("iedb", "cedar"):
                pin = pins[source]
                path = Path("/data") / pin["remote_path"]
                if digest(path) != {key: pin[key] for key in ("sha256", "size_bytes")}:
                    raise ValueError(f"Pinned source changed: {source}")
                sample = scratch / f"{source}.csv"
                with path.open(newline="") as handle, sample.open("w", newline="") as output:
                    csv.writer(output).writerows(itertools.islice(csv.reader(handle), n_rows + 2))
                receipt["sources"][source] = pin
                receipt["samples"][source] = digest(sample)
            for index, variant in enumerate(receipt["order"], 1):
                output = destination / f"{index}-{variant}"
                output.mkdir()
                commands, outputs = [], []
                with contextlib.ExitStack() as stack:
                    for source in ("iedb", "cedar"):
                        target = output / source
                        target.mkdir()
                        cache = scratch / f"cache-{index}-{source}"
                        cache.mkdir()
                        environment = audited_environment(
                            target / "network",
                            {
                                **os.environ,
                                "HITLIST_DATA_DIR": str(cache),
                                "XDG_CACHE_HOME": str(cache),
                                "TMPDIR": str(cache),
                                "OMP_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "PYTHONUNBUFFERED": "1",
                            },
                        )
                        log = stack.enter_context((target / "run.log").open("w"))
                        commands.append(
                            (
                                [
                                    sys.executable,
                                    str(Path(__file__)),
                                    "--child",
                                    source,
                                    "--sample",
                                    str(scratch / f"{source}.csv"),
                                    "--destination",
                                    str(target),
                                ],
                                environment,
                                log,
                            )
                        )
                        outputs.append(target)
                    before, memory = usage(), MemorySampler()
                    with memory:
                        pids = run_children(commands, concurrent=variant == "parallel", timeout=900)
                    result = {
                        "variant": variant,
                        "total": {**elapsed(before), **memory.total},
                        "sources": {},
                    }
                    for pid, target in zip(pids, outputs):
                        check_network_audit(target / "network", pid)
                        report = json.loads((target / "report.json").read_text())
                        if report["status"] != "success":
                            raise ValueError("Incomplete source scan")
                        for name, pin in report["outputs"].items():
                            if digest(target / name) != pin:
                                raise ValueError("Retained source output differs from its receipt")
                        result["sources"][target.name] = report
                    receipt["runs"].append(result)
                    (destination / "experiment.json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
            baseline = receipt["runs"][0]["sources"]
            for result in receipt["runs"][1:]:
                for source, report in result["sources"].items():
                    if report["outputs"] != baseline[source]["outputs"]:
                        raise ValueError(f"Independent-source outputs changed: {source}")
            receipt["status"] = "success"
    except BaseException:
        (destination / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (destination / "experiment.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":

    def interrupted(signum, _frame):
        raise SystemExit(f"Source diagnostic received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?")
    parser.add_argument("--rows", type=int, default=500000)
    parser.add_argument("--child", choices=("iedb", "cedar"))
    parser.add_argument("--sample")
    parser.add_argument("--destination")
    args = parser.parse_args()
    if args.child:
        child(args.child, args.sample, args.destination)
    else:
        run(args.label, args.rows)
