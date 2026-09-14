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

from pathlib import Path

import pytest

from tests.mhcgnomes_floor_check import check as _check_mhcgnomes_floor
from tests.xdist_cache import load_or_build_mmapped_arrow


def pytest_configure(config):
    """Fail the whole session immediately, with one clear message, when the
    installed ``mhcgnomes`` predates this project's declared floor (#467) --
    instead of 200+ scattered ``AttributeError``\\ s with nothing connecting
    any one traceback back to the actual cause. Runs once, before collection.
    """
    message = _check_mhcgnomes_floor()
    if message:
        pytest.exit(message, returncode=1)


@pytest.fixture
def _isolated_curation_root(tmp_path, monkeypatch):
    """Shared base for a test-isolated curation YAML tree (#471, #474).

    Copies the packaged ``pmid_overrides.yaml`` / ``tissue_categories.yaml``
    / ``monoallelic_lines.yaml`` / ``cell_lines.yaml`` into an isolated temp
    dir and points ``curation._data_path`` / ``cell_name_parser._registry_path``
    at it, clearing every curation ``lru_cache`` on setup and teardown.

    Every curation loader this touches is process-global, keyed on no
    arguments or on a PMID that can collide with a real one, so a value
    cached under this fixture's fake data survives ``monkeypatch``'s own
    teardown (which only restores ``_data_path``, not whatever got cached
    while it was patched) and leaks into whatever test runs next on the
    same xdist worker. #448's ``curation_referencing_uncached_asset`` shipped
    without this and stayed safe only because nothing in its call graph
    happened to reach ``load_pmid_overrides()`` -- until #471 gave
    ``load_supplementary_manifest`` an indirect path to it, and the leak
    failed 23 unrelated tests on the very next deploy. One fixture with an
    unconditional clear, reused everywhere a test needs this, is the version
    that survives the next such addition -- see #474.

    Callers overwrite ``pmid_overrides.yaml`` with their own fake content
    before the test body runs; nothing here calls a cached loader in
    between, so the caller's final content is always what gets read first.
    """
    from hitlist import cell_name_parser, curation

    data_root = tmp_path / "curation"
    data_root.mkdir()
    for name in ("pmid_overrides.yaml", "tissue_categories.yaml", "monoallelic_lines.yaml"):
        (data_root / name).write_bytes(Path(curation._data_path(name)).read_bytes())
    (data_root / "cell_lines.yaml").write_bytes(cell_name_parser._registry_path().read_bytes())
    monkeypatch.setattr(curation, "_data_path", lambda name: str(data_root / name))
    monkeypatch.setattr(cell_name_parser, "_registry_path", lambda: data_root / "cell_lines.yaml")
    curation._clear_curation_caches()
    yield data_root
    curation._clear_curation_caches()


def _build_full_observations_df():
    from hitlist.export import generate_observations_table

    return generate_observations_table()


@pytest.fixture(scope="session")
def full_observations_df(tmp_path_factory, worker_id):
    """Built observations table with no filters applied.

    Tests that need filtered views should copy / mask this DataFrame
    rather than calling ``generate_observations_table()`` again.

    Under pytest-xdist this fixture is shared across workers via an
    on-disk Arrow IPC file read with ``memory_map`` (see module
    docstring) — one worker pays the build cost, the rest mmap the
    cache so numeric / categorical columns share one copy across all
    workers instead of N private heap copies (#262).
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

    # xdist: share the build across workers via an mmap'd Arrow IPC file.
    # The session-shared root is per-invocation (see comment above), so
    # there's no stale-cache concern from previous runs.
    cache_path = tmp_path_factory.getbasetemp().parent / "full_observations_df.arrow"
    return load_or_build_mmapped_arrow(cache_path, _build_full_observations_df)


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
