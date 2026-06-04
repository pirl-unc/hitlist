"""On-disk caches for sharing expensive session fixtures across
pytest-xdist workers.  See ``tests/conftest.py::full_observations_df``.

The pattern: first arrival builds the value and writes it to disk;
subsequent callers (other xdist workers) read the cache.  POSIX
exclusive ``flock`` serializes the critical section so only one builder
runs even when N workers race in at startup.

Two serializers:

- :func:`load_or_build_pickled` — generic (any picklable object).  Each
  reader ``pickle.load``\\s into its **private heap**, so N workers hold
  N copies.

- :func:`load_or_build_mmapped_arrow` — pandas-DataFrame-specific, backed
  by Arrow IPC + ``memory_map``.  Numeric / bool / dictionary-encoded
  columns stay in the file's mmap'd pages, which the OS unified buffer
  cache shares **once** across every worker that maps the file — no
  per-worker heap copy (#262).  Object/string columns still materialize
  per worker (unavoidable without consuming Arrow directly).

Module name deliberately omits the ``test_`` prefix so pytest does not
collect it.  Tests for the helpers live in ``tests/test_xdist_cache.py``.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    import pandas as pd

T = TypeVar("T")


def load_or_build_pickled(cache_path: Path, builder: Callable[[], T]) -> T:
    """Return the cached pickle at ``cache_path``, or build via ``builder`` and cache.

    Concurrency: serializes all callers (across processes) on a POSIX
    ``flock`` keyed off ``cache_path.with_suffix(suffix + '.lock')``.
    The first arrival pays the full ``builder()`` cost; subsequent
    callers pay only a pickle read.  The lock is held for the whole
    critical section (build + write *or* read) — read parallelism
    would shave a few seconds in the read-only case but isn't worth
    the shared/exclusive complexity given the build dominates.

    Atomicity: writes go to a sibling ``.tmp`` and are renamed into
    place via ``os.replace``.  If ``builder()`` or the dump raises,
    the partial ``.tmp`` is unlinked so a future caller doesn't see
    stale half-written state.

    POSIX-only: relies on ``fcntl.flock``.  Matches the project's
    bash ``./test.sh`` reality (no Windows CI).
    """
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        value = builder()
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception:
            # Don't leave a half-written .tmp behind — a future caller
            # would see no cache (correct) but the .tmp would accumulate.
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise
        return value


def load_or_build_mmapped_arrow(
    cache_path: Path,
    builder: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """DataFrame variant of :func:`load_or_build_pickled` backed by Arrow IPC.

    The first arrival builds the frame and writes it as an Arrow IPC
    (Feather v2) file; subsequent callers ``memory_map`` that file so the
    OS shares its pages across all workers.  ``to_pandas(split_blocks=True)``
    keeps numeric / bool / dictionary columns zero-copy (each column wraps
    the shared mmap buffer rather than being consolidated into a fresh
    private block), which is what actually delivers the cross-worker memory
    saving (#262).  Object/string columns still materialize per worker.

    Same concurrency / atomicity contract as :func:`load_or_build_pickled`:
    a POSIX ``flock`` serializes build-or-read; the write goes to a sibling
    ``.tmp`` renamed via ``os.replace``.

    ``builder`` must return a ``pandas.DataFrame``.

    The returned frame is effectively **read-only**: its zero-copy columns
    wrap shared mmap pages, so in-place column mutation raises
    ``ValueError: assignment destination is read-only``.  Tests must copy /
    mask before mutating (they already do — that's the fixture contract).

    Every xdist worker — including the one that builds — returns through
    the same ``memory_map`` read path, so all workers get an *identical*
    frame (same dtypes, same read-only buffers).  Returning the builder's
    own in-memory frame instead would make a test behave differently
    depending on which worker it landed on (writable RangeIndex on the
    builder vs read-only Int64 index on readers) — a nondeterministic
    footgun under ``-n auto``.  The single redundant read on the builder
    worker is negligible next to the build it just paid for.

    Lifetime note: the returned frame's zero-copy columns keep the
    underlying mmap alive via pyarrow's buffer refcounting, so the local
    ``source`` handle going out of scope here does not invalidate them.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    def _read_mmapped() -> pd.DataFrame:
        source = pa.memory_map(str(cache_path), "r")
        table = ipc.open_file(source).read_all()
        return table.to_pandas(split_blocks=True, zero_copy_only=False)

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            return _read_mmapped()

        df = builder()
        table = pa.Table.from_pandas(df, preserve_index=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            # Comma-form (not parenthesized) ``with`` — parenthesized
            # context managers are a SyntaxError on Python 3.9.
            with pa.OSFile(str(tmp_path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
            os.replace(tmp_path, cache_path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise
        # Read back through mmap too, so the builder worker returns the same
        # read-only, mmap-backed frame every reader worker sees.
        return _read_mmapped()
