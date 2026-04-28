"""Shared fixtures for the test suite.

The ``full_observations_df`` fixture is the big lever here: building the
full enriched observations table takes 30s+ end-to-end, and several tests
in ``test_export.py`` were each rebuilding it independently. Promoting it
to a session-scoped fixture cuts ~250s off a serial run and still works
under ``pytest-xdist`` (each worker pays the cost once instead of N times).
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
