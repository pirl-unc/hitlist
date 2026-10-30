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


def _sources(tmp_path, *, large=False, compressible=False):
    paths = {name: tmp_path / f"{name}.csv" for name in ("iedb", "cedar")}
    for name, path in paths.items():
        rows = [_row("shared"), _row("shared", "copy metadata"), _row(""), _row(f"{name}:only")]
        rows[2][2] = "reported-reference"
        if large:
            rows = [_row(f"{name}:{index}") for index in range(10)]
            for index, row in enumerate(rows):
                if compressible:
                    comment = "x" * 90000 + str(index)
                else:
                    import random
                    import string

                    comment = "".join(
                        random.Random(index).choices(string.ascii_letters + string.digits, k=90000)
                    )
                row[15] = "unrelated comment " + comment
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
    contents = {
        "/proc/self/cgroup": "0::/\n",
        "/proc/self/mountinfo": "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        **contents,
    }
    monkeypatch.setattr(workers.sys, "platform", "linux")
    monkeypatch.setattr(workers.Path, "exists", lambda path: str(path) in contents)
    monkeypatch.setattr(workers.Path, "read_text", lambda path: contents[str(path)])
    assert workers._available_memory() == expected


@pytest.mark.parametrize("unified", [True, False])
def test_nested_cgroup_and_ancestor_limits_bound_admission(monkeypatch, unified):
    gib = 1024**3
    root = "/sys/fs/cgroup"
    contents = {
        "/proc/meminfo": f"MemAvailable: {200 * gib // 1024} kB\n",
        "/proc/self/cgroup": "0::/service/task\n" if unified else "4:memory,cpu:/service/task\n",
        "/proc/self/mountinfo": f"1 0 0:1 / {root} rw - cgroup2 cgroup rw\n"
        if unified
        else f"1 0 0:1 / {root} rw - cgroup cgroup rw,memory,cpu\n",
    }
    for location, limit, used, cpus in [
        ("", 100, 4, 8),
        ("/service", 40, 10, 1),
        ("/service/task", 64, 2, 4),
    ]:
        path = root + location
        if unified:
            contents.update(
                {
                    path + "/memory.max": str(limit * gib),
                    path + "/memory.current": str(used * gib),
                    path + "/cpu.max": f"{cpus * 100000} 100000",
                }
            )
        else:
            contents.update(
                {
                    path + "/memory.limit_in_bytes": str(limit * gib),
                    path + "/memory.usage_in_bytes": str(used * gib),
                    path + "/cpu.cfs_quota_us": str(cpus * 100000),
                    path + "/cpu.cfs_period_us": "100000",
                }
            )
    monkeypatch.setattr(workers.sys, "platform", "linux")
    monkeypatch.setattr(workers.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.setattr(workers.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(workers.Path, "exists", lambda path: str(path) in contents)
    monkeypatch.setattr(workers.Path, "read_text", lambda path: contents[str(path)])
    assert workers._available_memory() == 30 * gib
    assert workers._available_cpus() == 1
    contents["/proc/self/cgroup"] = "invalid"
    assert workers._available_memory() is None
    assert workers._available_cpus() == 1


def test_hybrid_cgroup_prefers_the_controllers_v1_membership(monkeypatch):
    contents = {
        "/proc/self/cgroup": "0::/unified\n4:memory,cpu:/service/task\n",
        "/proc/self/mountinfo": (
            "1 0 0:1 / /sys/fs/cgroup/unified rw - cgroup2 cgroup rw\n"
            "2 0 0:2 /service /sys/fs/cgroup/legacy rw - cgroup cgroup rw,memory,cpu\n"
        ),
    }
    monkeypatch.setattr(workers.Path, "read_text", lambda path: contents[str(path)])
    unified, directories = workers._cgroup_directories("memory")
    assert not unified
    assert list(map(str, directories)) == [
        "/sys/fs/cgroup/legacy/task",
        "/sys/fs/cgroup/legacy",
    ]


def test_owned_empty_schema_releases_arrow_buffers_and_preserves_metadata():
    frame = pd.DataFrame(
        {
            "value": ["a" * 1000000, "b" * 1000000, "c" * 1000000],
            "nullable": pd.Series(["a", None, "b"], dtype="string[pyarrow]"),
            "category": pd.Categorical(["b", "a", "b"], categories=["b", "a"], ordered=True),
            "integer": pd.Series([1, None, 3], dtype="Int64"),
            "time": pd.to_datetime(["2026-01-01"] * 3, utc=True),
        }
    )
    frame.index.name = "source row"
    frame.columns.name = "source field"
    frame.attrs = {"origin": {"dataset": "fixture"}}
    original = frame.iloc[:0].copy()
    assert original["value"].array._pa_array.chunk(0).buffers()[-1].size == 3000000
    owned = workers._owned_empty_schema(original)
    pd.testing.assert_frame_equal(original, owned)
    assert owned.attrs == original.attrs
    assert owned.attrs is not original.attrs
    for name in ("value", "nullable"):
        for chunk in owned[name].array._pa_array.chunks:
            assert chunk.buffers()[-1] is None or chunk.buffers()[-1].size == 0
    with pytest.raises(ValueError, match="empty source schema"):
        workers._owned_empty_schema(frame)


def test_compressed_source_transfers_fit_the_existing_reservation(tmp_path, monkeypatch, capsys):
    paths = _sources(tmp_path, large=True, compressible=True)
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
        tmp_path / "parallel.parquet",
        monkeypatch,
        True,
        limit=2 * 1024**2,
    )
    output = capsys.readouterr().out
    assert "two source workers" in output
    assert "retrying source scans serially" not in output
    pd.testing.assert_frame_equal(actual, expected)
    assert (tmp_path / "serial.parquet").read_bytes() == (
        tmp_path / "parallel.parquet"
    ).read_bytes()
