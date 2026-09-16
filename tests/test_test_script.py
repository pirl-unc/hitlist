"""Exercise test.sh resource decisions without launching pytest recursively."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "test.sh"


def _stub_env(
    tmp_path, n_free_pages, n_speculative_pages, probe_fails, worker_min, worker_max, **extra_env
):
    stubs = {
        "uname": "echo Darwin",
        "getconf": "echo 16",
        "sysctl": "exit 1" if probe_fails else "echo 16384",
        "vm_stat": (
            f"echo 'Pages free: {n_free_pages}.'\n"
            "echo 'Pages inactive: 2000000.'\n"
            f"echo 'Pages speculative: {n_speculative_pages}.'"
        ),
        "python": 'if [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$@"',
        # A different interpreter's pytest must never be selected from PATH.
        "pytest": "exit 99",
    }
    for name, body in stubs.items():
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\n{body}\n")
        executable.chmod(0o755)

    return dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        PER_WORKER_GB="2.5",
        INTEGRATION_PER_WORKER_GB="5",
        TEST_SH_MIN=str(worker_min),
        TEST_SH_MAX=str(worker_max),
        **extra_env,
    )


def _split_invocations(args):
    """Split one process's flattened stdout back into per-invocation arg lists.

    Each ``python -m pytest -n <workers> ...`` invocation starts with the
    same three-token prefix, which never recurs mid-invocation (the only
    other ``-m`` is pytest's own marker flag, never followed by ``-n``).
    """
    starts = [
        i
        for i in range(len(args) - 2)
        if args[i : i + 2] == ["-m", "pytest"] and args[i + 2] == "-n"
    ]
    assert starts == sorted(starts) and starts and starts[0] == 0, args
    bounds = [*starts, len(args)]
    return [args[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def _marker(invocation):
    # invocation[:4] is always ["-m", "pytest", "-n", "<workers>"]; the next
    # "-m" (if any) is pytest's own marker-expression flag.
    for i in range(4, len(invocation) - 1):
        if invocation[i] == "-m":
            return invocation[i + 1]
    return None


@pytest.mark.parametrize(
    (
        "n_free_pages",
        "n_speculative_pages",
        "probe_fails",
        "worker_min",
        "worker_max",
        "expected_light",
        "expected_integration",
    ),
    [
        (100_000, 10_000, False, 1, 0, 1, 1),
        (100_000, 400_000, False, 1, 0, 3, 1),
        (100_000, 400_000, True, 1, 0, 1, 1),
        (100_000, 400_000, False, 10, 2, 2, 2),
    ],
)
def test_all_runs_two_passes_with_independent_worker_budgets(
    tmp_path,
    n_free_pages,
    n_speculative_pages,
    probe_fails,
    worker_min,
    worker_max,
    expected_light,
    expected_integration,
):
    """``--all`` must run non-integration and integration tests as two
    separate pytest processes (#483), each re-probing available memory and
    sizing workers off its own per-worker budget."""
    env = _stub_env(
        tmp_path, n_free_pages, n_speculative_pages, probe_fails, worker_min, worker_max
    )
    result = subprocess.run(["bash", str(SCRIPT), "--all"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert "--all" not in args

    light, integration = _split_invocations(args)

    assert light[:4] == ["-m", "pytest", "-n", str(expected_light)]
    assert _marker(light) == "not integration"
    assert "--cov=hitlist/" in light
    assert "--cov-append" not in light
    assert "--cov-report=term-missing" not in light

    assert integration[:4] == ["-m", "pytest", "-n", str(expected_integration)]
    assert _marker(integration) == "integration"
    assert "--cov=hitlist/" in integration
    assert "--cov-append" in integration
    assert "--cov-report=term-missing" in integration


def test_default_invocation_is_a_single_non_integration_pass(tmp_path):
    """Without ``--all``, test.sh must stay a single pytest process (the
    common local/dev-loop case) filtered to non-integration tests."""
    env = _stub_env(tmp_path, 100_000, 400_000, probe_fails=False, worker_min=1, worker_max=0)
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()

    invocations = _split_invocations(args)
    assert len(invocations) == 1
    (invocation,) = invocations

    assert invocation[:4] == ["-m", "pytest", "-n", "3"]
    assert _marker(invocation) == "not integration"
    assert "--cov=hitlist/" in invocation
    assert "--cov-report=term-missing" in invocation
    assert "--cov-append" not in invocation


def test_extra_args_are_forwarded_to_every_pass(tmp_path):
    """Args after --all/other flags (e.g. a -k filter) must reach pytest in
    both passes, not just the first."""
    env = _stub_env(tmp_path, 100_000, 400_000, probe_fails=False, worker_min=1, worker_max=0)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--all", "-k", "foo"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()

    light, integration = _split_invocations(args)
    assert "-k" in light and "foo" in light
    assert "-k" in integration and "foo" in integration
