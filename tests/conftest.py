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

The cache helper itself lives in ``tests.xdist_cache`` so it has a
public surface that can be unit-tested independently of the fixture.
"""

from __future__ import annotations

import pytest

from tests.xdist_cache import load_or_build_pickled


def _build_full_observations_df():
    from hitlist.export import generate_observations_table

    return generate_observations_table()


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
        # Serial run (no xdist) — build inline via pytest's own
        # session-scoped fixture caching.
        #
        # NOTE: this branch must NOT fall through to the xdist cache
        # path below.  Under xdist, ``tmp_path_factory.getbasetemp()``
        # is the *per-worker* basetemp (e.g. ``.../pytest-N/popen-gw0/``)
        # and ``.parent`` is the per-invocation session root
        # (``.../pytest-N/``) — safe.  Under "master" (no xdist),
        # ``getbasetemp()`` IS the per-invocation root, so ``.parent``
        # would resolve to the persistent ``/tmp/pytest-of-<user>/`` dir
        # that pytest reuses across runs — an inappropriate place to
        # cache, since stale entries from earlier invocations could
        # poison the current run.
        return _build_full_observations_df()

    # xdist: share the build across workers via on-disk pickle.  The
    # session-shared root is per-invocation (see comment above), so
    # there's no stale-cache concern from previous runs.
    cache_path = tmp_path_factory.getbasetemp().parent / "full_observations_df.pkl"
    return load_or_build_pickled(cache_path, _build_full_observations_df)


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
