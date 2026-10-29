"""Run baseline/candidate/candidate/baseline on one bounded remote allocation."""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from reliability_benchmark import BASE, CONFIG, digest

GUARD_DIRECTORY = CONFIG / "network_guard"


def audited_environment(directory, base=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ) if base is None else base
    return {
        **base,
        "HITLIST_BENCHMARK_DENY_NETWORK": "1",
        "HITLIST_BENCHMARK_NETWORK_LOG": str(directory),
        "PYTHONPATH": str(GUARD_DIRECTORY) + os.pathsep + base.get("PYTHONPATH", ""),
    }


def check_network_audit(directory, pid):
    directory = Path(directory)
    if not (directory / f"{pid}.installed").is_file():
        raise ValueError("Timed interpreter did not install the socket audit")
    attempted = sorted(str(path) for path in directory.glob("*.jsonl") if path.stat().st_size)
    if attempted:
        raise ValueError(f"Timed benchmark attempted network access: {attempted}")


def scan_child(sample, destination):
    import importlib.metadata

    import pandas as pd
    from profile_build import MemorySampler, elapsed, usage

    import hitlist
    from hitlist.provenance import ContributorCollector
    from hitlist.scanner import scan

    destination = Path(destination)
    options = {"iedb_path": Path(sample), "mhc_species": None, "classify_source": True}
    if "n_workers" in inspect.signature(scan).parameters:
        options["n_workers"] = None
    receipt = {
        "status": "failed",
        "scope": "500,000-row diagnostic; not a full build",
        "hitlist_version": hitlist.__version__,
        "metadata_version": importlib.metadata.version("hitlist"),
        "python": sys.version,
        "sample": digest(sample),
        "versions": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "observations_projection": "scanner output with the builder's numeric Int64 PMID normalization; original cells retained in contributors",
    }
    if receipt["hitlist_version"] != receipt["metadata_version"]:
        raise ValueError("Imported producer and installed metadata disagree")
    before, memory = usage(), MemorySampler()
    try:
        with memory, ContributorCollector() as collector:
            frame = scan(**options, provenance=collector)
            collector.write([frame], destination / "contributors.parquet")
            frame["pmid"] = pd.to_numeric(frame["pmid"], errors="coerce").astype("Int64")
            frame.to_parquet(destination / "observations.parquet", index=False)
        receipt.update(
            status="success", n_rows=len(frame), total={**elapsed(before), **memory.total}
        )
        receipt["outputs"] = {
            name: digest(destination / name)
            for name in ("contributors.parquet", "observations.parquet")
        }
    finally:
        (destination / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("SCAN_TOTAL " + json.dumps(receipt, sort_keys=True), flush=True)


def run(label, scan_pair):
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Invalid experiment label")
    destination = BASE / label
    destination.mkdir(parents=True, exist_ok=False)
    variants = {
        "baseline": ("/opt/baseline-env/bin/python", CONFIG / "baseline.tar.gz"),
        "candidate": ("/opt/candidate-env/bin/python", CONFIG / "producer.tar.gz"),
    }
    for executable, archive in variants.values():
        if not Path(executable).is_file() or not archive.is_file():
            raise ValueError("Both producer archives and isolated environments are required")
    receipt = {
        "status": "failed",
        "scope": "scan diagnostic" if scan_pair else "full forced build",
        "order": ["baseline", "candidate", "candidate", "baseline"],
        "allocation": {
            "hostname": platform.node(),
            "cpu_limit": 8,
            "memory_limit_mib": 65536,
            "mapping_workers": 2,
            "requested_scan_workers": 4,
            "nonpreemptible": True,
        },
        "archives": {name: digest(archive) for name, (_, archive) in variants.items()},
        "recipes": {path.name: digest(path) for path in CONFIG.glob("*.py")},
        "runs": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="hitlist-paired-") as temporary:
            scratch = Path(temporary)
            sample = scratch / "iedb-prefix.csv"
            if scan_pair:
                pin = json.loads((CONFIG / "sources.json").read_text())["iedb"]
                source_path = Path("/data") / pin["remote_path"]
                if digest(source_path) != {key: pin[key] for key in ("sha256", "size_bytes")}:
                    raise ValueError("Pinned source bytes changed")
                with source_path.open(newline="") as source, sample.open("w", newline="") as output:
                    csv.writer(output).writerows(itertools.islice(csv.reader(source), 500002))
                receipt["sample"] = digest(sample)
            for index, variant in enumerate(receipt["order"], 1):
                executable, archive = variants[variant]
                run_label = f"{label}-{index}-{variant}"
                output = BASE / run_label
                cache = scratch / run_label
                environment = {
                    **os.environ,
                    "HITLIST_BENCHMARK_PRODUCER_ARCHIVE": str(archive),
                    "HITLIST_BENCHMARK_AUDIT_NETWORK": "1",
                    "HITLIST_BUILD_WORKERS": "2",
                    "HITLIST_SCAN_WORKERS": "4",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                    "HITLIST_DATA_DIR": str(cache / "data"),
                    "XDG_CACHE_HOME": str(cache / "cache"),
                }
                if scan_pair:
                    output.mkdir(parents=True, exist_ok=False)
                    environment = audited_environment(output / "network", environment)
                    command = [
                        executable,
                        str(Path(__file__)),
                        "--scan-child",
                        str(sample),
                        "--destination",
                        str(output),
                    ]
                else:
                    command = [executable, str(CONFIG / "reliability_benchmark.py"), run_label]
                print(f"PAIR_RUN {index} {variant} {run_label}", flush=True)
                with (destination / f"{index}-{variant}.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        cwd="/",
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=3000 if scan_pair else 5100)
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                if scan_pair:
                    check_network_audit(output / "network", process.pid)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                result = json.loads((output / "report.json").read_text())
                if result["status"] != "success":
                    raise ValueError("Incomplete build cannot enter the comparison")
                receipt["runs"].append(
                    {"label": run_label, "variant": variant, "total": result["total"]}
                )
                (destination / "experiment.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print("PAIR_RESULT " + json.dumps(receipt["runs"][-1]), flush=True)
                shutil.rmtree(cache, ignore_errors=True)
            baseline = BASE / receipt["runs"][0]["label"]
            for item in receipt["runs"][1:]:
                output = BASE / item["label"]
                if scan_pair:
                    for name in ("contributors.parquet", "observations.parquet"):
                        if digest(baseline / name) != digest(output / name):
                            raise ValueError(f"Paired output bytes differ: {output / name}")
                else:
                    subprocess.run(
                        [
                            variants["candidate"][0],
                            str(CONFIG / "compare_reliability_outputs.py"),
                            str(baseline),
                            str(output),
                            str(destination / f"{item['label']}-comparison.json"),
                        ],
                        check=True,
                        timeout=1800,
                    )
            receipt["status"] = "success"
    except BaseException:
        (destination / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (destination / "experiment.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?")
    parser.add_argument("--scan-pair", action="store_true")
    parser.add_argument("--scan-child")
    parser.add_argument("--destination")
    args = parser.parse_args()
    if args.scan_child:
        scan_child(args.scan_child, args.destination)
    else:
        run(args.label, args.scan_pair)
