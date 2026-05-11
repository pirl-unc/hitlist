"""Tests for ``tests.xdist_cache.load_or_build_pickled``.

Exercises the cache helper through its public API rather than its
private internals.  The contract under test:

* First caller invokes the builder and caches the result on disk.
* Subsequent callers (with the same ``cache_path``) get the cached
  result without invoking the builder.
* If the builder raises, no partial cache file is left behind.
"""

from __future__ import annotations

import pytest

from tests.xdist_cache import load_or_build_pickled


def test_first_call_invokes_builder_and_writes_cache(tmp_path):
    cache = tmp_path / "v.pkl"
    calls: list[int] = []

    def builder():
        calls.append(1)
        return {"answer": 42, "letters": ["a", "b"]}

    out = load_or_build_pickled(cache, builder)

    assert out == {"answer": 42, "letters": ["a", "b"]}
    assert len(calls) == 1
    assert cache.is_file()


def test_second_call_uses_cache_and_skips_builder(tmp_path):
    cache = tmp_path / "v.pkl"
    calls: list[int] = []

    def builder():
        calls.append(1)
        return {"answer": 42}

    out1 = load_or_build_pickled(cache, builder)
    out2 = load_or_build_pickled(cache, builder)

    assert out1 == out2
    assert len(calls) == 1  # builder ran exactly once


def test_cache_value_round_trips_complex_objects(tmp_path):
    """Nested dict / list / tuple / None survive pickle round-trip."""
    cache = tmp_path / "v.pkl"
    payload = {
        "nested": {"deep": [1, 2, 3, None]},
        "tuple": (4, 5, "six"),
        "missing": None,
    }
    out = load_or_build_pickled(cache, lambda: payload)
    out_again = load_or_build_pickled(cache, lambda: pytest.fail("should not run"))
    assert out == payload
    assert out_again == payload


def test_builder_failure_leaves_no_cache(tmp_path):
    """If the builder raises, no cache file or .tmp is left behind so
    a future caller will retry."""
    cache = tmp_path / "v.pkl"

    def boom():
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        load_or_build_pickled(cache, boom)

    assert not cache.is_file()
    assert not (tmp_path / "v.pkl.tmp").is_file()


def test_builder_failure_then_success_caches_success(tmp_path):
    """After a failed build, a subsequent call retries the builder and
    caches the successful result."""
    cache = tmp_path / "v.pkl"

    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        load_or_build_pickled(cache, boom)

    out = load_or_build_pickled(cache, lambda: "second-try")
    assert out == "second-try"
    # Subsequent call hits the cache.
    out_again = load_or_build_pickled(cache, lambda: pytest.fail("cache miss"))
    assert out_again == "second-try"


def test_cache_path_is_created_for_missing_parent(tmp_path):
    """``load_or_build_pickled`` mkdirs the parent if it doesn't exist."""
    cache = tmp_path / "subdir" / "v.pkl"
    assert not cache.parent.exists()
    load_or_build_pickled(cache, lambda: 1)
    assert cache.is_file()


def test_no_leftover_tmp_after_successful_build(tmp_path):
    """The atomic-rename leaves no .tmp file behind on success."""
    cache = tmp_path / "v.pkl"
    load_or_build_pickled(cache, lambda: "ok")
    assert not (tmp_path / "v.pkl.tmp").is_file()
