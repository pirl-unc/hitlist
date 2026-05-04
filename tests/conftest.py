"""Shared fixtures for the test suite.

The ``full_observations_df`` fixture is the big lever here: building the
full enriched observations table takes 30s+ end-to-end, and several tests
in ``test_export.py`` were each rebuilding it independently. Promoting it
to a session-scoped fixture cuts ~250s off a serial run and still works
under ``pytest-xdist`` (each worker pays the cost once instead of N times).

Any test that takes ``full_observations_df`` is automatically tagged
``pytest.mark.integration`` by ``pytest_collection_modifyitems`` so the
default ``./test.sh`` invocation (``-m "not integration"``) skips it.
``./test.sh --all`` and ``./deploy.sh`` drop the filter and run the full
matrix. See issue #223.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def full_observations_df():
    """Built observations table with no filters applied.

    Tests that need filtered views should copy / mask this DataFrame
    rather than calling ``generate_observations_table()`` again.
    """
    from hitlist.observations import is_built

    if not is_built():
        pytest.skip("Observations table not built")

    from hitlist.export import generate_observations_table

    return generate_observations_table()


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
