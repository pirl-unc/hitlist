"""Shared fixtures for the test suite.

The ``full_observations_df`` fixture is the dominant cost driver for the
integration tier: ``generate_observations_table()`` materializes the full
~4.4M-row enriched table.  Post-vectorization (#244) it takes ~33s and
peaks at several GB of working memory.

Under pytest-xdist, ``scope="session"`` is *per-worker*, so a 10-worker
``./test.sh --all`` previously rebuilt the table 10x in parallel — same
33s x 10 of CPU time, *and* a peak resident set of ~50 GB across workers
that OOM'd 32 GB Macs (issue #244).

This module shares the build across xdist workers via an on-disk pickle
in the session-shared tmp dir (``tmp_path_factory.getbasetemp().parent``
is shared by all workers within a single pytest invocation).  The first
worker to acquire the file lock builds and writes the pickle; later
workers find the cache populated and read it.  One build + N cheap loads
instead of N builds.
"""

from __future__ import annotations

import fcntl
import os
import pickle
from pathlib import Path

import pytest


def _build_full_observations_df():
    from hitlist.export import generate_observations_table

    return generate_observations_table()


def _load_or_build_shared(cache_path: Path):
    """Build the table on cache miss, otherwise read the cached pickle.

    Serializes all workers on a POSIX exclusive ``flock``.  The first
    arrival pays the build cost; the rest read the pickle.  The lock is
    held for the whole critical section (build + write *or* read) — read
    parallelism would shave a few seconds but isn't worth the
    shared/exclusive complexity given the build dominates.
    """
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        df = _build_full_observations_df()
        # Atomic write: pickle to .tmp, then rename.  Avoids leaving a
        # half-written cache file if the build process is killed mid-dump.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        return df


@pytest.fixture(scope="session")
def full_observations_df(tmp_path_factory, worker_id):
    """Built observations table with no filters applied.

    Tests that need filtered views should copy / mask this DataFrame
    rather than calling ``generate_observations_table()`` again.

    Under pytest-xdist this fixture is shared across workers via an
    on-disk pickle (see module docstring) — one worker pays the build
    cost, the rest read the cache.
    """
    from hitlist.observations import is_built

    if not is_built():
        pytest.skip("Observations table not built")

    if worker_id == "master":
        # Serial run (no xdist) — just build once via pytest's own
        # session-scoped fixture caching.
        return _build_full_observations_df()

    # ``getbasetemp().parent`` is the session-wide tmp dir shared by all
    # xdist workers within a single pytest invocation.  Per-invocation
    # (not persistent across runs) so we never read a stale cache from a
    # previous test run.
    cache_path = tmp_path_factory.getbasetemp().parent / "full_observations_df.pkl"
    return _load_or_build_shared(cache_path)


def pytest_collection_modifyitems(config, items):
    """Auto-tag tests that depend on the built observations corpus.

    Any test that requests ``full_observations_df`` is implicitly an
    integration test — it cannot run without the built parquet, takes
    seconds-to-minutes per call after the session fixture warms, and
    is the dominant cost driver for ``./test.sh``. Marking them
    automatically means the default ``-m "not integration"`` filter
    just works without each test author remembering to add the tag.

    Tests that internally call ``hitlist.observations.is_built()`` and
    branch on the result aren't auto-marked here because some of them
    intentionally test the not-built error path; those carry an
    explicit ``@pytest.mark.integration`` decorator instead.
    """
    integration = pytest.mark.integration
    for item in items:
        if "full_observations_df" in getattr(item, "fixturenames", ()):
            item.add_marker(integration)
