"""Independent source workers preserve source partitions and contributor graphs."""

import multiprocessing

import pandas as pd
import pyarrow as pa
import pytest

from hitlist import _source_scan as workers
from hitlist.builder import _CONCAT_PROMOTE_OPTIONS, _drop_duplicate_iris
from hitlist.provenance import ContributorCollector, ProvenanceStorageError
from tests.test_provenance import _row
from tests.test_scanner import _write_tiny_iedb_csv


def _sources(tmp_path, *, large=False):
    paths = {name: tmp_path / f"{name}.csv" for name in ("iedb", "cedar")}
    for name, path in paths.items():
        rows = [_row("shared"), _row("shared", "copy metadata"), _row(""), _row(f"{name}:only")]
        rows[2][2] = "reported-reference"
        if large:
            rows = [_row(f"{name}:{index}") for index in range(10)]
            for index, row in enumerate(rows):
                row[15] = "unrelated comment " + "x" * 90000 + str(index)
        _write_tiny_iedb_csv(path, rows)
    return paths


def _build(paths, scratch, output, monkeypatch, parallel, *, limit=64 * 1024**2):
    monkeypatch.setattr(workers, "_admit_workers", lambda _: parallel)
    with ContributorCollector(
        scratch_dir=scratch, max_scratch_bytes=limit, min_free_bytes=0
    ) as collector:
        sources = workers.scan_sources(paths, collector)
        assert collector.max_scratch_bytes == limit
        tables = [parts[0] for _, parts in sources if parts[0] is not None]
        frame = pa.concat_tables(tables, promote_options=_CONCAT_PROMOTE_OPTIONS).to_pandas()
        frame = _drop_duplicate_iris(frame, "MS", provenance=collector)
        collector.write([frame], output)
        assert not list(scratch.glob("hitlist-scan-*"))
    return sources, frame


def test_parallel_sources_match_serial_partitions_and_cross_source_graph(tmp_path, monkeypatch):
    paths = _sources(tmp_path)
    serial, serial_frame = _build(
        paths, tmp_path / "scratch", tmp_path / "serial.parquet", monkeypatch, False
    )
    parallel, parallel_frame = _build(
        paths, tmp_path / "scratch", tmp_path / "parallel.parquet", monkeypatch, True
    )
    assert [name for name, _ in parallel] == ["iedb", "cedar"]
    for (_, left), (_, right) in zip(serial, parallel):
        for a, b in zip(left[:-1], right[:-1]):
            if a is None:
                assert b is None
            else:
                pd.testing.assert_frame_equal(a.to_pandas(), b.to_pandas())
        pd.testing.assert_frame_equal(left[-1], right[-1])
    pd.testing.assert_frame_equal(serial_frame, parallel_frame)
    assert (tmp_path / "serial.parquet").read_bytes() == (
        tmp_path / "parallel.parquet"
    ).read_bytes()


def test_ipc_reservation_exhaustion_retries_serial_before_graph_import(
    tmp_path, monkeypatch, capsys
):
    paths = _sources(tmp_path, large=True)
    _, expected = _build(
        paths,
        tmp_path / "scratch",
        tmp_path / "serial.parquet",
        monkeypatch,
        False,
        limit=2 * 1024**2,
    )
    _, actual = _build(
        paths,
        tmp_path / "scratch",
        tmp_path / "retry.parquet",
        monkeypatch,
        True,
        limit=2 * 1024**2,
    )
    assert "retrying source scans serially" in capsys.readouterr().out
    pd.testing.assert_frame_equal(actual, expected)
    assert (tmp_path / "serial.parquet").read_bytes() == (tmp_path / "retry.parquet").read_bytes()


def test_worker_failure_reaps_children_and_preserves_published_output(tmp_path, monkeypatch):
    paths = _sources(tmp_path)
    paths["cedar"].write_text("unrecognized\nheader\n")
    output = tmp_path / "published.parquet"
    output.write_bytes(b"published")
    before = {process.pid for process in multiprocessing.active_children()}
    monkeypatch.setattr(workers, "_admit_workers", lambda _: True)
    scratch = tmp_path / "scratch"
    with (
        pytest.raises(RuntimeError, match="resolve column"),
        ContributorCollector(
            scratch_dir=scratch, max_scratch_bytes=64 * 1024**2, min_free_bytes=0
        ) as collector,
    ):
        workers.scan_sources(paths, collector)
    assert {process.pid for process in multiprocessing.active_children()} <= before
    assert output.read_bytes() == b"published"
    assert not list(scratch.iterdir())


@pytest.mark.parametrize(
    "available,cpus,requested,expected",
    [
        (None, 8, "2", False),
        (59 * 1024**3, 8, "2", False),
        (60 * 1024**3, 8, "2", True),
        (64 * 1024**3, 1, "2", False),
        (64 * 1024**3, 8, "1", False),
        (64 * 1024**3, 8, "8", True),
    ],
)
def test_source_worker_admission_preserves_headroom(
    tmp_path, monkeypatch, available, cpus, requested, expected
):
    paths = _sources(tmp_path)
    monkeypatch.setattr(workers, "_available_memory", lambda: available)
    monkeypatch.setattr(workers, "_available_cpus", lambda: cpus)
    monkeypatch.setenv("HITLIST_SCAN_WORKERS", requested)
    assert workers._admit_workers(list(paths.items())) is expected
    assert not workers._admit_workers(list(paths.items())[:1])


def test_snapshot_merge_offsets_payload_blocks_and_retains_donor_links(tmp_path):
    roots = []
    sources = {}
    for dataset in ("a", "b"):
        source = tmp_path / f"{dataset}.csv"
        source.write_text("source\n")
        sources[dataset] = source
    for mode in ("serial", "merged"):
        roots = []
        with ContributorCollector(scratch_dir=tmp_path) as parent:
            for dataset, source in sources.items():

                def capture(collector, dataset=dataset, source=source):
                    collector.register_source(dataset, source)
                    collector._CAPTURE_BATCH_ROWS = 2
                    first = collector.record(dataset, 1, {"pmid": "001"}, ["first", "é"])
                    copy = collector.record(dataset, 2, {}, ["copy"])
                    collector.redirect(copy, first, "assay_copy")
                    collector.redirect(first, copy, "database_copy")
                    return [collector.observe(first, label) for label in ("donor a", "donor b")]

                if mode == "serial":
                    roots.extend(capture(parent))
                else:
                    with ContributorCollector(scratch_dir=tmp_path) as child:
                        roots.extend(capture(child))
                        snapshot = tmp_path / f"{dataset}.sqlite"
                        metadata = child._export_snapshot(snapshot)
                    parent._import_snapshot(snapshot, metadata)
                    snapshot.unlink()
            # Capture after importing both snapshots to exercise the new offset.
            roots.append(parent.observe(parent.record("later", 1, {}, ["after imports"])))
            parent.write([pd.DataFrame({"provenance_id": roots})], tmp_path / f"{mode}.parquet")
    assert (tmp_path / "serial.parquet").read_bytes() == (tmp_path / "merged.parquet").read_bytes()


def test_worker_result_reservation_is_bounded(tmp_path):
    with pytest.raises(ProvenanceStorageError, match="reservation"):
        workers._write_result(tmp_path, {"value": "x" * workers._RESULT_BYTES})
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("failure", ["cancel", "identity", "inventory", "corrupt"])
def test_source_failure_before_import_cleans_owned_scratch(tmp_path, monkeypatch, failure):
    import json
    import sqlite3

    paths = _sources(tmp_path)
    scratch = tmp_path / "scratch"
    before = {process.pid for process in multiprocessing.active_children()}
    wait = workers._wait_workers

    def fail(jobs):
        if failure == "cancel":
            raise KeyboardInterrupt
        wait(jobs)
        directory = jobs[0][1]
        if failure == "corrupt":
            (directory / "graph.sqlite").write_bytes(b"not SQLite")
        elif failure == "inventory":
            (directory / "unexpected").touch()
        else:
            path = directory / "result.json"
            data = json.loads(path.read_text())
            data["sources"]["iedb"]["path"] = "a different source"
            path.write_text(json.dumps(data))

    monkeypatch.setattr(workers, "_admit_workers", lambda _: True)
    monkeypatch.setattr(workers, "_wait_workers", fail)
    error = {"cancel": KeyboardInterrupt, "corrupt": sqlite3.DatabaseError}.get(failure, ValueError)
    with (
        pytest.raises(error),
        ContributorCollector(scratch_dir=scratch, min_free_bytes=0) as collector,
    ):
        workers.scan_sources(paths, collector)
    assert {process.pid for process in multiprocessing.active_children()} <= before
    assert not list(scratch.iterdir())


@pytest.mark.parametrize(
    "contents,expected",
    [
        (
            {
                "/proc/meminfo": "MemAvailable: 999999999 kB\n",
                "/sys/fs/cgroup/memory.max": "64000",
                "/sys/fs/cgroup/memory.current": "7000",
            },
            57000,
        ),
        (
            {"/proc/meminfo": "MemAvailable: 999999999 kB\n", "/sys/fs/cgroup/memory.max": "bad"},
            None,
        ),
        ({"/proc/meminfo": "MemAvailable: 1000 kB\n", "/sys/fs/cgroup/memory.max": "max"}, 1024000),
        ({"/proc/meminfo": "MemTotal: 1000 kB\n"}, None),
    ],
)
def test_memory_admission_caps_host_capacity_and_fails_closed(monkeypatch, contents, expected):
    monkeypatch.setattr(workers.sys, "platform", "linux")
    monkeypatch.setattr(workers.Path, "exists", lambda path: str(path) in contents)
    monkeypatch.setattr(workers.Path, "read_text", lambda path: contents[str(path)])
    assert workers._available_memory() == expected
