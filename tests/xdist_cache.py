"""On-disk pickle cache for sharing expensive session fixtures across
pytest-xdist workers.  See ``tests/conftest.py::full_observations_df``.

The pattern: first arrival builds the value and pickles it to disk;
subsequent callers (other xdist workers) read the cached pickle.  POSIX
exclusive ``flock`` serializes the critical section so only one builder
runs even when N workers race in at startup.

Module name deliberately omits the ``test_`` prefix so pytest does not
collect it.  Tests for the helper live in ``tests/test_xdist_cache.py``.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pickle
from pathlib import Path
from typing import Callable, TypeVar

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
