"""Bounded real-subprocess and unit regressions for build-worker SIGTERM cleanup (#881)."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hitlist._worker_lifecycle import ParentTerminated, graceful_sigterm_scope

_POSIX_SIGTERM = hasattr(signal, "SIGTERM") and sys.platform != "win32"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pids(receipt_dir: Path, expected: int, *, timeout: float = 15.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = sorted(receipt_dir.glob("*.pid"))
        if len(files) >= expected:
            pids = []
            ready = True
            for path in files:
                text = path.read_text().strip()
                if not text:
                    ready = False
                    break
                pids.append(int(text))
            if ready and len(pids) == expected:
                return pids
        time.sleep(0.05)
    raise TimeoutError(
        f"Timed out waiting for {expected} worker PID receipts in {receipt_dir} "
        f"(found {sorted(p.name for p in receipt_dir.glob('*.pid'))})"
    )


def _wait_until_dead(pids: list[int], *, timeout: float = 10.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _pid_is_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _pid_is_alive(pid)]


@pytest.mark.skipif(not _POSIX_SIGTERM, reason="POSIX SIGTERM semantics required")
def test_graceful_sigterm_scope_preserves_custom_and_ignored_handlers():
    previous = signal.getsignal(signal.SIGTERM)
    observed: list[int] = []

    def custom_handler(signum, _frame):
        observed.append(signum)

    try:
        signal.signal(signal.SIGTERM, custom_handler)
        with graceful_sigterm_scope() as installed:
            assert installed is False
            assert signal.getsignal(signal.SIGTERM) is custom_handler
        assert signal.getsignal(signal.SIGTERM) is custom_handler

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with graceful_sigterm_scope() as installed:
            assert installed is False
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with graceful_sigterm_scope() as outer:
            assert outer is True
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
            with graceful_sigterm_scope() as inner:
                assert inner is False
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.skipif(not _POSIX_SIGTERM, reason="POSIX SIGTERM semantics required")
def test_graceful_sigterm_scope_is_noop_in_non_main_thread():
    results: list[bool] = []
    errors: list[BaseException] = []

    def runner():
        try:
            with graceful_sigterm_scope() as installed:
                results.append(installed)
        except BaseException as error:  # pragma: no cover
            errors.append(error)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert not errors
    assert results == [False]


def test_parent_terminated_carries_posix_signal_exit_code():
    err = ParentTerminated(15)
    assert isinstance(err, SystemExit)
    assert err.signum == 15
    assert err.code == 143


def test_prefetch_supervised_terminates_pool_on_keyboard_interrupt(monkeypatch):
    """KeyboardInterrupt coverage stays distinct from parent-only SIGTERM (#881)."""
    from hitlist import mappings

    events: list[str] = []

    class FakeResult:
        def get(self, timeout=None):
            raise KeyboardInterrupt

    class FakePool:
        def apply_async(self, *_args, **_kwargs):
            return FakeResult()

        def terminate(self):
            events.append("terminate")

        def close(self):
            events.append("close")

        def join(self):
            events.append("join")

    class FakeContext:
        def Pool(self, processes=1):
            return FakePool()

    monkeypatch.setattr("multiprocessing.get_context", lambda _name: FakeContext())
    with pytest.raises(KeyboardInterrupt):
        mappings._supervise_prefetch_tasks(
            [("fake", ("fake",), {"kind": "uniprot"})], release=112, verbose=False
        )
    assert events == ["terminate", "join"]


# Module-level picklable worker targets for bounded real-subprocess SIGTERM tests.
def _slow_source_worker(name, _source, directory, _quota, _reserve, _data_dir):
    receipt_dir = Path(os.environ["HITLIST_TEST_RECEIPT_DIR"])
    (receipt_dir / f"{name}.pid").write_text(f"{os.getpid()}\n")
    (Path(directory) / "partial.arrow").write_bytes(b"incomplete")
    time.sleep(60)


def _slow_prefetch_worker(label, _entry, _data_dir, _release):
    receipt_dir = Path(os.environ["HITLIST_TEST_RECEIPT_DIR"])
    (receipt_dir / f"{label}.pid").write_text(f"{os.getpid()}\n")
    time.sleep(60)
    return label, True, ""


def _slow_mapping_worker(tag: str) -> str:
    receipt_dir = Path(os.environ["HITLIST_TEST_RECEIPT_DIR"])
    (receipt_dir / f"{tag}.pid").write_text(f"{os.getpid()}\n")
    time.sleep(60)
    return tag


@pytest.mark.skipif(not _POSIX_SIGTERM, reason="POSIX SIGTERM semantics required")
def test_parallel_sources_parent_only_sigterm_stops_workers_and_cleans_scratch(tmp_path):
    """Real subprocess: signal ONLY the parent PID; owned source workers and
    scratch must vanish while an existing published generation stays intact."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    published = tmp_path / "published_generation"
    published.mkdir()
    sentinel = published / "observations.parquet"
    sentinel.write_text("previous-generation-intact\n")
    for name in ("iedb", "cedar"):
        (tmp_path / f"{name}.csv").write_text("x\n")

    repo_root = str(Path(__file__).resolve().parents[1])
    script = tmp_path / "parent_source_scan.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "from pathlib import Path\n"
        "from hitlist import _source_scan\n"
        "from hitlist.provenance import ContributorCollector\n"
        "from tests.test_worker_sigterm import _slow_source_worker\n"
        "if __name__ == '__main__':\n"
        "    _source_scan._worker = _slow_source_worker\n"
        f"    scratch = Path({str(scratch)!r})\n"
        f"    sources = [('iedb', Path({str(tmp_path / 'iedb.csv')!r})), "
        f"('cedar', Path({str(tmp_path / 'cedar.csv')!r}))]\n"
        "    with ContributorCollector(scratch_dir=scratch, min_free_bytes=0) as collector:\n"
        "        _source_scan._parallel_sources(sources, collector, 1024 * 1024, 10 * 1024 * 1024)\n"
    )
    env = {
        **os.environ,
        "HITLIST_TEST_RECEIPT_DIR": str(receipt_dir),
        "PYTHONPATH": repo_root,
    }
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_pids: list[int] = []
    try:
        worker_pids = _wait_for_pids(receipt_dir, expected=2, timeout=20.0)
        assert all(_pid_is_alive(pid) for pid in worker_pids)
        # Signal ONLY the parent PID (never a process group or worker PID).
        os.kill(proc.pid, signal.SIGTERM)
        code = proc.wait(timeout=15.0)
        assert code == 128 + int(signal.SIGTERM)
        survivors = _wait_until_dead(worker_pids, timeout=10.0)
        assert not survivors, f"Worker PIDs survived parent-only SIGTERM: {survivors}"
        assert not list(scratch.iterdir()), f"Leftover scratch: {list(scratch.iterdir())}"
        assert sentinel.read_text() == "previous-generation-intact\n"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        for pid in worker_pids:
            if _pid_is_alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(not _POSIX_SIGTERM, reason="POSIX SIGTERM semantics required")
def test_mapping_prefetch_and_pool_parent_only_sigterm_stop_owned_workers(tmp_path):
    """Real subprocess: parent-only SIGTERM stops owned mapping prefetch and
    ProcessPoolExecutor workers promptly (#881)."""
    repo_root = str(Path(__file__).resolve().parents[1])

    # 1. Supervised proteome prefetch worker
    prefetch_receipts = tmp_path / "prefetch_receipts"
    prefetch_receipts.mkdir()
    prefetch_script = tmp_path / "parent_prefetch.py"
    prefetch_script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "from hitlist.mappings import _supervise_prefetch_tasks\n"
        "from tests.test_worker_sigterm import _slow_prefetch_worker\n"
        "if __name__ == '__main__':\n"
        "    _supervise_prefetch_tasks(\n"
        "        [('human', ('Homo sapiens',), {'kind': 'uniprot'})],\n"
        "        release=112,\n"
        "        verbose=False,\n"
        "        deadline_seconds=60.0,\n"
        "        worker_target=_slow_prefetch_worker,\n"
        "    )\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(prefetch_script)],
        cwd=repo_root,
        env={
            **os.environ,
            "HITLIST_TEST_RECEIPT_DIR": str(prefetch_receipts),
            "PYTHONPATH": repo_root,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids: list[int] = []
    try:
        pids = _wait_for_pids(prefetch_receipts, expected=1, timeout=20.0)
        assert all(_pid_is_alive(pid) for pid in pids)
        os.kill(proc.pid, signal.SIGTERM)
        assert proc.wait(timeout=15.0) == 128 + int(signal.SIGTERM)
        assert not _wait_until_dead(pids, timeout=10.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        for pid in pids:
            if _pid_is_alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)

    # 2. ProcessPoolExecutor mapping workers guarded by graceful_sigterm_scope + terminate_process_pool
    pool_receipts = tmp_path / "pool_receipts"
    pool_receipts.mkdir()
    pool_script = tmp_path / "parent_pool.py"
    pool_script.write_text(
        "import multiprocessing as mp\n"
        "import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from hitlist._worker_lifecycle import graceful_sigterm_scope, terminate_process_pool\n"
        "from tests.test_worker_sigterm import _slow_mapping_worker\n"
        "if __name__ == '__main__':\n"
        "    with graceful_sigterm_scope():\n"
        "        pool = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'))\n"
        "        completed = False\n"
        "        try:\n"
        "            list(pool.map(_slow_mapping_worker, ['w1', 'w2'], chunksize=1))\n"
        "            completed = True\n"
        "        finally:\n"
        "            if completed:\n"
        "                pool.shutdown(wait=True)\n"
        "            else:\n"
        "                terminate_process_pool(pool)\n"
    )
    proc2 = subprocess.Popen(
        [sys.executable, str(pool_script)],
        cwd=repo_root,
        env={
            **os.environ,
            "HITLIST_TEST_RECEIPT_DIR": str(pool_receipts),
            "PYTHONPATH": repo_root,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids2: list[int] = []
    try:
        pids2 = _wait_for_pids(pool_receipts, expected=2, timeout=20.0)
        assert all(_pid_is_alive(pid) for pid in pids2)
        os.kill(proc2.pid, signal.SIGTERM)
        assert proc2.wait(timeout=15.0) == 128 + int(signal.SIGTERM)
        assert not _wait_until_dead(pids2, timeout=10.0)
    finally:
        if proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=5.0)
        for pid in pids2:
            if _pid_is_alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
