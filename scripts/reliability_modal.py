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
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("time")
        .pip_install("setuptools==84.2.1")
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
        .add_local_file(
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
        .add_local_file(str(ROOT / "scripts/profile_scan.py"), "/opt/benchmark/profile_scan.py")
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
def main(label: str, profile: bool = False, scan_sample: bool = False):
    if scan_sample:
        diagnose_scan.remote(label)
    else:
        execute.remote(label, profile)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(16384, 16384),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=1,
    retries=0,
)
def diagnose_scan(label: str):
    """Locate row-level hotspots; this sample is not a full-build speed result."""
    import subprocess
    import sys

    try:
        subprocess.run(
            [sys.executable, "/opt/benchmark/profile_scan.py", label],
            check=True,
            timeout=1650,
        )
    finally:
        volume.commit()
