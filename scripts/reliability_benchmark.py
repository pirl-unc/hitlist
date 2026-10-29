"""Run a pinned full build on private scratch; retain bounded reports and outputs.

Invoked by reliability_modal.py. Source acquisition, verification and reference
preparation are outside the measured warm-cache build. Every run has fresh output.
Preparation pins human Ensembl 112. The build requests cache-only references;
baselines before #865 may nevertheless attempt missing Ensembl downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path

CONFIG = Path("/opt/benchmark")
BASE = Path("/data/hitlist-reliability-20261028")


def digest(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "size_bytes": path.stat().st_size}


def checked_copy(source, target, pin):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(target) != {key: pin[key] for key in ("sha256", "size_bytes")}:
        raise ValueError(f"Input integrity failure: {source}")


def prepare(scratch):
    """Prepare current-library indexes from pinned raw files, never old pickles."""
    from pyensembl import EnsemblRelease

    from hitlist.downloads import fetch_all_data_assets, register

    sources = json.loads((CONFIG / "sources.json").read_text())
    for name, pin in sources.items():
        target = scratch / "inputs" / pin["filename"]
        checked_copy(Path("/data") / pin["remote_path"], target, pin)
        register(name, target, description="Frozen benchmark input; export date unverified")
    assets = fetch_all_data_assets(verbose=True)
    reference = json.loads((CONFIG / "reference.json").read_text())
    genome = EnsemblRelease(112, species="homo_sapiens")
    cache = Path(genome.download_cache.cache_directory_path)
    for name, pin in reference.items():
        checked_copy(pin["path"], cache / name, pin)
    # Pin raw sequences across producer variants. Generate caches with the same
    # locked pyensembl version on POSIX scratch, outside the measured build.
    genome.index(show_progress=True)
    genome.db.connection.close()
    with tarfile.open(CONFIG / "producer.tar.gz") as archive:
        source_commit = archive.pax_headers.get("comment")
    return {
        "source_commit": source_commit,
        "sources": sources,
        "reference": reference,
        "assets": {name: digest(path) for name, path in assets.items()},
        "code_archive": digest(CONFIG / "producer.tar.gz"),
        "requirements": digest(CONFIG / "requirements.txt"),
        "recipes": {path.name: digest(path) for path in sorted(CONFIG.glob("*.py"))},
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unavailable",
        ),
        "reference_scope": "prepared human Ensembl 112; cache-only build requested (see #865 for older producers)",
        "resources": {
            "cpu_limit": 8,
            "memory_limit_mib": 65536,
            "mapping_workers": 2,
            "nonpreemptible": True,
        },
        "preparation_excluded": "source acquisition/copy/verification and pyensembl reference indexing",
        "build_cache_state": "fresh outputs; warm raw sources and pyensembl reference; cold Hitlist mapping index",
    }


def run(label, profiled):
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Use a lowercase alphanumeric run label")
    destination = BASE / label
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix="hitlist-benchmark-") as directory:
            scratch = Path(directory)
            os.environ.update(
                HITLIST_DATA_DIR=str(scratch / "hitlist"),
                XDG_CACHE_HOME=str(scratch / "cache"),
                PYENSEMBL_CACHE_DIR=str(scratch / "references"),
                HITLIST_BUILD_WORKERS="2",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            context = prepare(scratch)
            (destination / "inputs.json").write_text(json.dumps(context, indent=2) + "\n")
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(destination / "resource.txt"),
                sys.executable,
                str(CONFIG / "profile_build.py"),
                "--cached-references-only",
                "--report",
                str(destination / "report.json"),
                "--context",
                str(destination / "inputs.json"),
            ]
            if profiled:
                command += ["--sampling-out", str(destination / "build.pyisession")]
            # Capture output on disk and stream it. A subprocess has a fresh
            # process peak; preparation cannot contaminate its timing/RSS.
            with (destination / "build.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
            report = json.loads((destination / "report.json").read_text())
            if report["status"] != "success":
                raise ValueError("Build did not produce a successful receipt")
            output = Path(report["output"]).parent
            retained = destination / "output"
            retained.mkdir()
            inventory = {}
            for path in sorted(output.iterdir()):
                if path.suffix in (".parquet", ".json") and path.is_file():
                    shutil.copy2(path, retained / path.name)
                    inventory[path.name] = digest(retained / path.name)
            (destination / "outputs.json").write_text(json.dumps(inventory, indent=2) + "\n")
    except BaseException:
        (destination / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run(args.label, args.profile)
