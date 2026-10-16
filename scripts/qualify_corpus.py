"""Bounded, separately cached reconstruction with the published 1.76.0 producer.

Run through qualification_modal.py. This runner never imports the source checkout
as the producer. Each stage records its actual environment, inputs and outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("environment", "sources", "indexes", "reference", "mappings", "snapshot")
BASE = Path("/data/hitlist-qualification")
CONFIG = Path("/opt/qualification")


def digest(path):
    path = Path(path)
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path}")
    return {"sha256": value.hexdigest(), "size_bytes": after.st_size, "path": str(path)}


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def validate(files):
    for pin in files.values():
        actual = digest(pin["path"])
        if any(actual[key] != pin[key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"Content mismatch: {pin['path']}")


def previous(root, stage):
    from datacache import inspect_materialization

    state = inspect_materialization(root / "receipts" / stage, verify_files=True)
    if state.status != "available":
        raise ValueError(f"A verified {stage} receipt is required: {state.status}")
    receipt = json.loads((Path(state.bundle) / "receipt.json").read_text())
    if receipt["status"] != "success":
        raise ValueError(f"The preceding {stage} stage did not succeed")
    validate(receipt["artifacts"])
    return receipt


def dataset_inventory():
    from hitlist.dataset import INDEX_FILES, MAPPING_FILES, dataset_snapshot

    with dataset_snapshot() as directory:
        return {
            name: digest(directory / name)
            for name in (*INDEX_FILES, *MAPPING_FILES)
            if (directory / name).is_file()
        }


def perform(stage, root):
    import hitlist

    if hitlist.__version__ != "1.76.0":
        raise ValueError(f"Expected the published Hitlist 1.76.0, found {hitlist.__version__}")
    checked = subprocess.run(
        [sys.executable, "-m", "pip", "check"], check=True, capture_output=True, text=True
    )
    print(checked.stdout, end="")
    if stage == "environment":
        return {}, {"producer_import": hitlist.__file__}
    previous(root, "environment")
    if stage == "sources":
        from datacache import install_bundle

        from hitlist.downloads import data_assets, fetch_all_data_assets, register

        pins = json.loads((CONFIG / "sources.json").read_text())
        sources = {
            name: {"path": str(Path("/data") / pin["remote_path"]), **pin}
            for name, pin in pins.items()
        }
        validate(sources)
        paths = install_bundle(
            root / "inputs",
            {
                name: {"path": pin["path"], "sha256": pin["sha256"], "size": pin["size_bytes"]}
                for name, pin in sources.items()
            },
        )
        for name, path in paths.items():
            register(name, path, description="Pinned qualification source; export date unverified")
        assets = fetch_all_data_assets(verbose=True)
        return {
            **{name: digest(path) for name, path in paths.items()},
            **{f"asset:{name}": digest(path) for name, path in assets.items()},
        }, {"source_pins": pins, "paper_asset_registry": data_assets(), "depmap_registered": False}
    if stage == "indexes":
        from hitlist.builder import build_observations

        previous(root, "sources")
        build_observations(
            with_flanking=False,
            build_mappings=False,
            fetch_missing_proteomes=False,
            use_uniprot_search=False,
        )
        return dataset_inventory(), {"mapping_status": "not_built"}
    if stage == "reference":
        from pyensembl import EnsemblRelease

        genome = EnsemblRelease(112, species="homo_sapiens")
        genome.download(show_progress=True)
        files = [genome.gtf_path, *genome.protein_fasta_paths, *genome.transcript_fasta_paths]
        artifacts = {Path(path).name: digest(path) for path in files}
        genome.index(show_progress=True)
        validate(artifacts)
        return artifacts, {"species": "homo_sapiens", "ensembl_release": 112}
    if stage == "mappings":
        from hitlist.mappings import build_peptide_mappings

        previous(root, "indexes")
        previous(root, "reference")
        build_peptide_mappings(release=112, fetch_missing=False, use_uniprot=False)
        return dataset_inventory(), {
            "reference_scope": "human Ensembl 112 only; other missing references are limitations",
            "fetch_missing": False,
            "use_uniprot": False,
        }
    if stage == "snapshot":
        from ci_corpus import snapshot_corpus, verify_manifest

        from hitlist.downloads import data_dir

        previous(root, "mappings")
        directory = snapshot_corpus(data_dir(), cache_dir=root / "portable")
        verify_manifest(directory)
        return {
            path.name: digest(path) for path in sorted(directory.iterdir()) if path.is_file()
        }, {"portable_directory": str(directory), "qualification": "build_only"}
    raise ValueError(stage)


class Tee:
    def __init__(self, live, saved):
        self.live = live
        self.saved = saved

    def write(self, value):
        self.live.write(value)
        self.saved.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.live.flush()
        self.saved.flush()


def deadline(*_):
    raise TimeoutError("Qualification stage reached its 85-minute work deadline")


def run(stage):
    from datacache import materialize

    recipes = {path.name: digest(path) for path in sorted(CONFIG.iterdir()) if path.is_file()}
    identity = hashlib.sha256(
        encode({name: pin["sha256"] for name, pin in recipes.items()})
    ).hexdigest()
    root = BASE / "runs" / identity
    os.environ.update(
        HITLIST_DATA_DIR=str(root / "hitlist"),
        XDG_CACHE_HOME=str(root / "cache"),
        PYENSEMBL_CACHE_DIR=str(root / "references"),
        HITLIST_BUILD_WORKERS="2",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(5100)
    started = time.monotonic()
    receipt = {
        "schema_version": 1,
        "run_id": identity,
        "stage": stage,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "recipes": recipes,
        "versions": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_limit": 8,
        "memory_limit_mib": 65536,
        "gpu": "none",
        "disk_free_bytes_before": shutil.disk_usage(root).free,
        "options": {
            name: os.environ[name]
            for name in os.environ
            if name.startswith(("HITLIST_", "PYENSEMBL_"))
        },
        "artifacts": {},
    }
    error = None
    with tempfile.TemporaryDirectory(prefix="qualification-log-") as temporary:
        log = Path(temporary) / "build.log"
        with (
            log.open("w") as stream,
            contextlib.redirect_stdout(Tee(sys.stdout, stream)),
            contextlib.redirect_stderr(Tee(sys.stderr, stream)),
        ):
            try:
                receipt["artifacts"], receipt["details"] = perform(stage, root)
                receipt["status"] = "success"
            except Exception as caught:
                error = caught
                receipt["status"] = "failed"
                receipt["error"] = f"{type(caught).__name__}: {caught}"
                traceback.print_exc()
        signal.alarm(0)
        receipt.update(
            runtime_seconds=time.monotonic() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            rss_scope="runner process only; worker processes excluded",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            disk_free_bytes_after=shutil.disk_usage(root).free,
        )

        def publish(inputs, outputs):
            Path(outputs["receipt.json"]).write_bytes(encode(receipt))
            shutil.copyfile(log, outputs["build.log"])

        paths = materialize(
            root / "receipts" / stage,
            {
                name: {"path": pin["path"], "sha256": pin["sha256"], "size": pin["size_bytes"]}
                for name, pin in recipes.items()
            },
            transform={"version": "qualification-stage-v1", "options": {"stage": stage}},
            outputs={"receipt.json": {}, "build.log": {}},
            builder=publish,
            force=True,
        )
    result = {
        "receipt": receipt,
        "receipt_files": {name: digest(path) for name, path in paths.items()},
    }
    print(encode(result).decode())
    if error:
        raise RuntimeError(
            f"{stage} failed; retained receipt at {paths['receipt.json']}"
        ) from error
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    run(parser.parse_args().stage)
