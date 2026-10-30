"""Bounded remote transport for repeatable full-build benchmarks.

HITLIST_BENCHMARK_ARCHIVE must identify a git archive of the measured producer.
Both variants use the same committed requirements, inputs and instrumentation.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("hitlist-reliability-benchmark")
volume = modal.Volume.from_name("presto-data", create_if_missing=False)
image = None
if modal.is_local():
    archive = Path(os.environ["HITLIST_BENCHMARK_ARCHIVE"]).resolve()
    # Preserve the original full-build image across paired measurements. The
    # test image needs Git and uses a supported Debian base for fresh apt installs.
    testing = os.environ.get("HITLIST_BENCHMARK_TEST_IMAGE") == "1"
    base = (
        modal.Image.from_registry("python:3.12-slim-bookworm").apt_install("time", "git")
        if testing
        else modal.Image.debian_slim(python_version="3.12").apt_install("time")
    )
    image = (
        base.pip_install("setuptools==84.2.1")
        .pip_install("pyinstrument==5.1.3")
        .pip_install_from_requirements(
            str(ROOT / "tasks/reliability-benchmark-requirements.txt"),
            extra_options="--require-hashes",
        )
        .add_local_file(str(archive), "/opt/benchmark/producer.tar.gz", copy=True)
        .run_commands(
            "mkdir -p /opt/producer && tar -xzf /opt/benchmark/producer.tar.gz -C /opt/producer",
            "python -m pip install --no-deps --no-build-isolation -e /opt/producer",
            "python -m pip check",
        )
    )
    baseline_archive = os.environ.get("HITLIST_BENCHMARK_BASELINE_ARCHIVE")
    if baseline_archive:
        image = image.add_local_file(
            str(Path(baseline_archive).resolve()), "/opt/benchmark/baseline.tar.gz", copy=True
        ).run_commands(
            "mkdir -p /opt/baseline && tar -xzf /opt/benchmark/baseline.tar.gz -C /opt/baseline",
            "python -m pip uninstall -y hitlist",
            "python -m venv --system-site-packages /opt/baseline-env",
            "python -m venv --system-site-packages /opt/candidate-env",
            "/opt/baseline-env/bin/python -m pip install --no-deps --no-build-isolation -e /opt/baseline",
            "/opt/candidate-env/bin/python -m pip install --no-deps --no-build-isolation -e /opt/producer",
            "/opt/baseline-env/bin/python -m pip check",
            "/opt/candidate-env/bin/python -m pip check",
        )
    image = (
        image.add_local_file(
            str(ROOT / "tasks/reliability-benchmark-requirements.txt"),
            "/opt/benchmark/requirements.txt",
        )
        .add_local_file(
            str(ROOT / "tasks/corpus-qualification-sources.json"), "/opt/benchmark/sources.json"
        )
        .add_local_file(
            str(ROOT / "tasks/reliability-benchmark-reference.json"),
            "/opt/benchmark/reference.json",
        )
        .add_local_file(
            str(ROOT / "scripts/reliability_benchmark.py"),
            "/opt/benchmark/reliability_benchmark.py",
        )
        .add_local_file(str(ROOT / "scripts/profile_build.py"), "/opt/benchmark/profile_build.py")
        .add_local_file(
            str(ROOT / "scripts/profile_mapping_workers.py"),
            "/opt/benchmark/profile_mapping_workers.py",
        )
        .add_local_file(str(ROOT / "scripts/profile_scan.py"), "/opt/benchmark/profile_scan.py")
        .add_local_file(
            str(ROOT / "scripts/profile_csv_serialization.py"),
            "/opt/benchmark/profile_csv_serialization.py",
        )
        .add_local_file(
            str(ROOT / "scripts/compare_reliability_outputs.py"),
            "/opt/benchmark/compare_reliability_outputs.py",
        )
        .add_local_file(
            str(ROOT / "scripts/paired_build_benchmark.py"),
            "/opt/benchmark/paired_build_benchmark.py",
        )
        .add_local_file(
            str(ROOT / "scripts/source_scan_benchmark.py"),
            "/opt/benchmark/source_scan_benchmark.py",
        )
        .add_local_file(
            str(ROOT / "scripts/benchmark_site/sitecustomize.py"),
            "/opt/benchmark/network_guard/sitecustomize.py",
        )
    )


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(65536, 65536),
    ephemeral_disk=524288,
    timeout=5400,
    max_containers=1,
    retries=0,
    nonpreemptible=True,
)
def execute(label: str, profile: bool):
    import subprocess
    import sys

    command = [sys.executable, "/opt/benchmark/reliability_benchmark.py", label]
    if profile:
        command.append("--profile")
    try:
        subprocess.run(command, check=True, timeout=5100)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(
    label: str,
    profile: bool = False,
    scan_sample: bool = False,
    scan_workers: int = 1,
    scan_rows: int = 100000,
    codec_sample: bool = False,
    serialization_sample: bool = False,
    paired: bool = False,
    scan_pair: bool = False,
    source_pair: bool = False,
    builder_source_pair: bool = False,
    candidate_first: bool = False,
    require_source_workers: bool = False,
    test_pair: bool = False,
    include_serial: bool = False,
    background: bool = False,
    test_suite: bool = False,
    compare_before: str = "",
    compare_after: str = "",
):
    if paired or scan_pair or test_pair or source_pair or builder_source_pair:
        if background:
            call = execute_pair.spawn(
                label,
                scan_pair,
                test_pair,
                include_serial,
                source_pair or builder_source_pair,
                builder_source_pair,
                candidate_first,
                require_source_workers,
            )
            print(f"Submitted benchmark function call: {call.object_id}", flush=True)
        else:
            execute_pair.remote(
                label,
                scan_pair,
                test_pair,
                include_serial,
                source_pair or builder_source_pair,
                builder_source_pair,
                candidate_first,
                require_source_workers,
            )
    elif compare_before or compare_after:
        compare_outputs.remote(label, compare_before, compare_after)
    elif test_suite:
        measure_tests.remote(label)
    elif scan_sample or codec_sample or serialization_sample:
        diagnose_scan.remote(label, codec_sample, serialization_sample, scan_workers, scan_rows)
    else:
        execute.remote(label, profile)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(65536, 65536),
    ephemeral_disk=524288,
    timeout=14400,
    max_containers=1,
    retries=0,
    nonpreemptible=True,
)
def execute_pair(
    label: str,
    scan_only: bool,
    tests_only: bool = False,
    include_serial: bool = False,
    source_only: bool = False,
    builder_sources: bool = False,
    candidate_first: bool = False,
    require_source_workers: bool = False,
):
    """Alternate variants in fresh processes on one allocation, with a finite deadline."""
    import contextlib
    import signal
    import subprocess
    import sys

    if source_only and (
        scan_only or tests_only or include_serial or candidate_first or require_source_workers
    ):
        raise ValueError("Choose one source-scheduling diagnostic scope")
    recipe = "source_scan_benchmark.py" if source_only else "paired_build_benchmark.py"
    command = [sys.executable, f"/opt/benchmark/{recipe}", label]
    if candidate_first:
        command.append("--candidate-first")
    if require_source_workers:
        command.append("--require-source-workers")
    if builder_sources:
        command.append("--builder")
    if scan_only:
        command.append("--scan-pair")
    if tests_only:
        command.append("--test-pair")
    if include_serial:
        command.append("--include-serial")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=14100)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        try:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        finally:
            volume.commit()


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(65536, 65536),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=1,
    retries=0,
)
def diagnose_scan(
    label: str,
    codecs_only: bool = False,
    serialization_only: bool = False,
    workers: int = 1,
    n_rows: int = 100000,
):
    """Locate row-level hotspots; this sample is not a full-build speed result."""
    import subprocess
    import sys

    script = "profile_csv_serialization.py" if serialization_only else "profile_scan.py"
    command = [sys.executable, f"/opt/benchmark/{script}", label]
    if codecs_only:
        command.append("--codecs-only")
    if not serialization_only:
        command.extend(["--rows", str(n_rows), "--workers", str(workers)])
    try:
        subprocess.run(
            command,
            check=True,
            timeout=1650,
        )
    finally:
        volume.commit()


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(65536, 65536),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=1,
    retries=0,
    nonpreemptible=True,
)
def measure_tests(label: str):
    """Measure the unchanged test.sh admission/gates in fresh isolated caches."""
    import hashlib
    import importlib.metadata
    import json
    import platform
    import shutil
    import subprocess
    import tempfile
    import time

    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Invalid label")
    destination = Path("/data/hitlist-reliability-20261028") / label
    destination.mkdir(parents=True, exist_ok=False)
    if shutil.which("git") is None:
        raise RuntimeError("Run test measurements with HITLIST_BENCHMARK_TEST_IMAGE=1")
    receipt = {
        "status": "failed",
        "test_workers": 2,
        "command": "./test.sh",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "os_release": Path("/etc/os-release").read_text(),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="hitlist-test-benchmark-") as temporary:
            env = {
                **os.environ,
                "HITLIST_DATA_DIR": temporary + "/data",
                "XDG_CACHE_HOME": temporary + "/cache",
                "TEST_SH_MAX": "2",
                "PYTHONUNBUFFERED": "1",
            }
            with Path("/opt/benchmark/producer.tar.gz").open("rb") as source:
                receipt["producer_archive_sha256"] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
            receipt["versions"] = {
                dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            }
            started = time.monotonic()
            with (destination / "tests.log").open("w") as log:
                process = subprocess.Popen(
                    ["/usr/bin/time", "-v", "-o", str(destination / "resource.txt"), "./test.sh"],
                    cwd="/opt/producer",
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            receipt.update(returncode=code, wall_seconds=time.monotonic() - started)
            if code:
                raise subprocess.CalledProcessError(code, "./test.sh")
            receipt["status"] = "success"
    finally:
        (destination / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=1800,
    max_containers=1,
    retries=0,
    nonpreemptible=True,
)
def compare_outputs(label: str, before: str, after: str):
    import subprocess
    import sys

    for value in (label, before, after):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in value):
            raise ValueError("Invalid comparison label")
    root = Path("/data/hitlist-reliability-20261028")
    destination = root / label
    destination.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run(
            [
                sys.executable,
                "/opt/benchmark/compare_reliability_outputs.py",
                str(root / before),
                str(root / after),
                str(destination / "comparison.json"),
            ],
            check=True,
            timeout=1650,
        )
    finally:
        volume.commit()
