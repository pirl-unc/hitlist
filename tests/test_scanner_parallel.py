"""Ordered annotation workers preserve scanner and contributor contracts."""

import os

import pandas as pd
import pytest

from hitlist import scanner
from hitlist.downloads import data_dir, set_data_dir
from hitlist.provenance import ContributorCollector
from tests.test_provenance import _row
from tests.test_scanner import _write_tiny_iedb_csv


def _record_worker(batch, options, capture):
    # This function is imported afresh in a spawned worker. The marker proves
    # both actual execution outside the parent and its resolved cache context.
    marker = data_dir() / f"scan-worker-{os.getpid()}.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(data_dir()))
    return scanner._annotate_batch(batch, options, capture)


def _fail_worker(batch, options, capture):
    raise RuntimeError("deliberate annotation failure")


@pytest.mark.parametrize("mhc_species", [None, "Homo sapiens"])
@pytest.mark.parametrize("capture", [False, True])
def test_parallel_scan_preserves_order_filters_donors_and_contributors(
    tmp_path, monkeypatch, threaded_parent, mhc_species, capture
):
    # Fixtures have a handful of rows. Test the worker boundary independently
    # of the full-corpus admission policy, which has its own tests below.
    monkeypatch.setattr(scanner, "_scan_workers", lambda requested: requested)
    monkeypatch.setattr(scanner, "_SCAN_BATCH_ROWS", 2)
    iedb, cedar = tmp_path / "iedb.csv", tmp_path / "cedar.csv"
    donor = _row("donor")
    donor[2], donor[5], donor[19] = "31844290", "SLLQHLIGL", "HLA class I"
    excluded = _row("filtered-first")
    excluded[19] = "H-2-Kb"
    rows = [
        _row("first"),
        _row("second"),
        excluded,
        donor,
        _row("first", "duplicate across batches"),
        _row("", "blank 1"),
        _row("", "blank 2"),
        _row("filtered-first"),
        _row("short")[:6],
    ]
    # Preserve every raw cell, including ones outside the recognized projection.
    rows[0].extend(['é 😀\t"\\', "unmapped\nmultiline"])
    _write_tiny_iedb_csv(iedb, rows)
    _write_tiny_iedb_csv(cedar, [_row("second", "cross-source duplicate"), _row("cedar")])
    target = tmp_path / "explicit-parent-data"
    set_data_dir(target)

    def execute(n_workers):
        options = {
            "iedb_path": iedb,
            "cedar_path": cedar,
            "mhc_species": mhc_species,
            "n_workers": n_workers,
        }
        if not capture:
            return scanner.scan(**options)
        with ContributorCollector(scratch_dir=tmp_path) as collector:
            frame = scanner.scan(**options, provenance=collector)
            collector.write([frame], tmp_path / f"contributors-{n_workers}.parquet")
            return frame

    expected = execute(1)
    original_worker = scanner._annotate_batch
    monkeypatch.setattr(scanner, "_annotate_batch", _record_worker)
    actual = execute(2)
    monkeypatch.setattr(scanner, "_annotate_batch", original_worker)
    pd.testing.assert_frame_equal(actual, expected)
    assert len(actual[actual.assay_iri == "donor"]) == 3
    assert len(actual[actual.assay_iri == ""]) == 2
    assert len(actual[actual.assay_iri == "filtered-first"]) == (1 if mhc_species is None else 0)
    markers = list(target.glob("scan-worker-*.txt"))
    assert markers
    assert all(path.read_text() == str(target) for path in markers)
    assert all(path.stem != f"scan-worker-{os.getpid()}" for path in markers)
    if capture:
        assert (tmp_path / "contributors-1.parquet").read_bytes() == (
            tmp_path / "contributors-2.parquet"
        ).read_bytes()


def test_failed_worker_closes_source_and_removes_owned_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "_scan_workers", lambda requested: requested)
    monkeypatch.setattr(scanner, "_SCAN_BATCH_ROWS", 1)
    monkeypatch.setattr(scanner, "_annotate_batch", _fail_worker)
    source = tmp_path / "source.csv"
    _write_tiny_iedb_csv(source, [_row(str(index)) for index in range(30)])
    scratch = tmp_path / "scratch"
    handles = []
    original_open = scanner._open_csv

    def open_source(path):
        result = original_open(path)
        handles.append(result[-1]._fh)
        return result

    monkeypatch.setattr(scanner, "_open_csv", open_source)
    with (
        pytest.raises(RuntimeError, match="deliberate annotation failure"),
        ContributorCollector(scratch_dir=scratch) as collector,
    ):
        scanner.scan(iedb_path=source, provenance=collector, n_workers=2)
    assert handles and all(handle.closed for handle in handles)
    assert list(scratch.iterdir()) == []


def test_scan_batch_limits_keep_order_and_isolate_oversized_input(monkeypatch):
    monkeypatch.setattr(scanner, "_SCAN_BATCH_ROWS", 3)
    monkeypatch.setattr(scanner, "_SCAN_BATCH_BYTES", 1500)
    columns = {"payload": 0}
    inputs = [
        ("iedb", columns, (index, ["x" * size], str(index), False))
        for index, size in enumerate([1, 1, 1, 1, 5000, 1])
    ]
    inputs.append(("cedar", {"payload": 0}, (1, ["last"], "cedar:1", False)))
    batches = list(scanner._scan_batches(iter(inputs)))
    assert [(source, layout, item) for source, layout, batch in batches for item in batch] == inputs
    assert all(len(batch) <= 3 for _, _, batch in batches)
    assert ("iedb", columns, [inputs[4][2]]) in batches
    assert batches[-1] == ("cedar", {"payload": 0}, [inputs[-1][2]])


@pytest.mark.parametrize(
    "available,expected", [(None, 1), (0, 1), (32, 1), (37, 2), (42, 4), (128, 4)]
)
def test_worker_admission_reserves_parent_and_worker_memory(monkeypatch, available, expected):
    monkeypatch.setattr(
        scanner, "_scan_available_bytes", lambda: None if available is None else available * 1024**3
    )
    monkeypatch.setattr(scanner.os, "cpu_count", lambda: 8)
    assert scanner._scan_workers(20) == expected


def test_worker_override_and_unknown_capacity(monkeypatch):
    monkeypatch.setattr(scanner, "_scan_available_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(scanner.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "1")
    monkeypatch.delenv("HITLIST_SCAN_WORKERS", raising=False)
    assert scanner._scan_workers(None) == 1
    monkeypatch.setenv("HITLIST_SCAN_WORKERS", "2")
    assert scanner._scan_workers(None) == 2
    assert scanner._scan_workers(1) == 1
    monkeypatch.setattr(scanner, "_scan_available_bytes", lambda: None)
    assert scanner._scan_workers(None) == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_worker_requests_fail(value):
    with pytest.raises(ValueError, match="positive integer"):
        scanner._scan_workers(value)
