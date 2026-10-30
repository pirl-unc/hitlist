"""Bounded independent IEDB/CEDAR workers for the full-data builder."""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .provenance import ContributorCollector, ProvenanceStorageError

_GIB = 1024**3
_RESULT_BYTES = 64 * 1024
_PARTITIONS = ("ms", "binding", "other", "empty")


def _available_memory():
    """Conservative free capacity; unknown probes always select serial execution."""
    try:
        if sys.platform == "linux":
            available = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            for limit_name, used_name in (
                ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                (
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                ),
            ):
                if Path(limit_name).exists():
                    limit = Path(limit_name).read_text().strip()
                    if limit != "max":
                        available = min(
                            available, max(0, int(limit) - int(Path(used_name).read_text()))
                        )
            return available
        if sys.platform == "darwin":
            probe = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
            pages = {}
            for line in probe.stdout.splitlines():
                name, separator, value = line.partition(":")
                if separator and name in {"Pages free", "Pages speculative"}:
                    pages[name] = int(value.strip().rstrip("."))
            return (pages["Pages free"] + pages.get("Pages speculative", 0)) * os.sysconf(
                "SC_PAGE_SIZE"
            )
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError):
        return None
    return None


def _available_cpus():
    count = os.cpu_count() or 1
    try:
        if sys.platform == "linux":
            if Path("/sys/fs/cgroup/cpu.max").exists():
                quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
                if quota != "max":
                    count = min(count, max(1, int(quota) // int(period)))
            elif Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").exists():
                quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
                if quota > 0:
                    count = min(count, max(1, quota // period))
    except (OSError, ValueError, ZeroDivisionError):
        return 1
    return count


def _admit_workers(sources):
    requested = int(
        os.environ.get("HITLIST_SCAN_WORKERS", os.environ.get("HITLIST_BUILD_WORKERS", "2"))
    )
    if requested < 1:
        raise ValueError("HITLIST_SCAN_WORKERS must be a positive integer")
    if requested == 1 or len(sources) != 2 or _available_cpus() < 2:
        return False
    available = _available_memory()
    if available is None:
        return False
    try:
        needed = 4 * _GIB + sum(max(28 * _GIB, 3 * path.stat().st_size) for _, path in sources)
    except OSError:
        return False
    return available >= needed


class _QuotaSink(io.RawIOBase):
    """Enforce the worker's aggregate IPC reservation before every file write."""

    def __init__(self, handle, remaining, collector):
        super().__init__()
        self.handle, self.remaining, self.collector = handle, remaining, collector

    def writable(self):
        return True

    def tell(self):
        return self.handle.tell()

    def write(self, data):
        if len(data) > self.remaining[0]:
            raise self.collector._storage_error(
                "source table buffers reached their scratch reservation"
            )
        self.collector._check_space(force=True)
        written = self.handle.write(data)
        self.remaining[0] -= written
        return written


def _write_result(directory, result):
    payload = (json.dumps(result) + "\n").encode()
    if len(payload) > _RESULT_BYTES:
        raise ProvenanceStorageError("Source worker result exceeds its scratch reservation")
    (directory / "result.json").write_bytes(payload)


def _worker(name, source, directory, quota, reserve, data_directory):
    import pyarrow as pa

    from . import scanner
    from .builder import _scan_source_tables
    from .downloads import set_data_dir

    directory = Path(directory)
    set_data_dir(data_directory)
    scanner._tqdm = None
    try:
        with ContributorCollector(
            scratch_dir=directory, max_scratch_bytes=quota, min_free_bytes=reserve
        ) as collector:
            tables = list(_scan_source_tables(name, Path(source), collector))
            tables[-1] = pa.Table.from_pandas(tables[-1], preserve_index=False)
            remaining, written = [quota], []
            for partition, table in zip(_PARTITIONS, tables):
                if table is None:
                    continue
                with (
                    (directory / f"{partition}.arrow").open("wb") as handle,
                    pa.PythonFile(_QuotaSink(handle, remaining, collector), mode="w") as sink,
                    pa.ipc.new_stream(sink, table.schema) as writer,
                ):
                    writer.write_table(table)
                written.append(partition)
            sources = collector._export_snapshot(directory / "graph.sqlite")
            result = {"status": "success", "partitions": written, "sources": sources}
            collector._check_space(force=True)
            _write_result(directory, result)
    except BaseException as error:
        result = {
            "status": "failed",
            "error": "capacity" if isinstance(error, ProvenanceStorageError) else "worker",
            "message": f"{type(error).__name__}: {error}"[:4000],
        }
        _write_result(directory, result)
        raise


def _wait_workers(jobs):
    pending = list(jobs)
    while pending:
        for process, directory in list(pending):
            process.join(timeout=0)
            if process.exitcode is None:
                continue
            if process.exitcode:
                result_path = directory / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                message = result.get(
                    "message", f"Source worker exited with code {process.exitcode}"
                )
                if result.get("error") == "capacity":
                    raise ProvenanceStorageError(message)
                raise RuntimeError(message)
            pending.remove((process, directory))
        if pending:
            time.sleep(0.1)


def _stop_workers(jobs):
    for process, _ in jobs:
        if process.is_alive():
            process.terminate()
    for process, _ in jobs:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def _parallel_sources(sources, provenance, quota, original_limit):
    import pyarrow as pa

    from .downloads import data_dir

    results = []
    with tempfile.TemporaryDirectory(
        prefix="hitlist-scan-", dir=provenance.scratch_dir
    ) as temporary:
        root = Path(temporary)
        jobs = []
        try:
            for name, source in sources:
                directory = root / name
                directory.mkdir()
                process = multiprocessing.get_context("spawn").Process(
                    target=_worker,
                    args=(
                        name,
                        str(source),
                        str(directory),
                        quota,
                        provenance.min_free_bytes,
                        str(data_dir()),
                    ),
                )
                process.start()
                jobs.append((process, directory))
            try:
                _wait_workers(jobs)
            except ProvenanceStorageError:
                # Nothing has been imported yet. Release all child reservations
                # before a serial retry using the original collector budget.
                return None
        finally:
            _stop_workers(jobs)
        parent_limit = original_limit - 4 * quota - 2 * _RESULT_BYTES
        for name, source in sources:
            directory = root / name
            result = json.loads((directory / "result.json").read_text())
            if (
                result.get("status") != "success"
                or set(result.get("sources", {})) != {name}
                or result["sources"][name].get("path") != str(source)
                or not set(result.get("partitions", ())) <= set(_PARTITIONS)
                or "empty" not in result.get("partitions", ())
                or len(result["partitions"]) != len(set(result["partitions"]))
            ):
                raise ValueError("Invalid source worker result")
            expected_files = {"graph.sqlite", "result.json"} | {
                f"{partition}.arrow" for partition in result["partitions"]
            }
            if {path.name for path in directory.iterdir()} != expected_files:
                raise ValueError("Source worker file inventory differs from its result")
            tables = []
            for partition in _PARTITIONS:
                path = directory / f"{partition}.arrow"
                if partition not in result["partitions"]:
                    tables.append(None)
                    continue
                # Owned buffers allow removal on every supported platform;
                # memory-mapped tables would keep the reservation's files open.
                with path.open("rb") as handle:
                    tables.append(pa.ipc.open_stream(handle).read_all())
                path.unlink()
            tables[-1] = tables[-1].to_pandas()
            parent_limit += quota
            provenance._set_scratch_limit(parent_limit)
            graph = directory / "graph.sqlite"
            provenance._import_snapshot(graph, result["sources"])
            graph.unlink()
            parent_limit += quota
            provenance._set_scratch_limit(parent_limit)
            results.append((name, tuple(tables)))
    return results


def scan_sources(paths, provenance):
    """Return deterministic source partitions with serial fallback before import."""
    from .builder import _scan_source_tables

    sources = [(name, Path(paths[name])) for name in ("iedb", "cedar") if name in paths]
    original_limit = provenance.max_scratch_bytes
    quota = (original_limit - 2 * _RESULT_BYTES) // (5 * 4096) * 4096
    n_pages = provenance.db.execute("PRAGMA page_count").fetchone()[0]
    if not provenance.sources and quota >= n_pages * 4096 and _admit_workers(sources):
        provenance._set_scratch_limit(original_limit - 4 * quota - 2 * _RESULT_BYTES)
        print("Scanning IEDB and CEDAR with two source workers.")
        results = _parallel_sources(sources, provenance, quota, original_limit)
        provenance._set_scratch_limit(original_limit)
        if results is not None:
            return results
        print(
            "Source worker scratch reservations were insufficient; retrying source scans serially."
        )
    return [(name, _scan_source_tables(name, path, provenance)) for name, path in sources]
