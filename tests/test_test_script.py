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
        (100_000, 400_000, False, 1, 0, 3, 1),
        (100_000, 400_000, True, 1, 0, 1, 1),
        # Enough real memory backs the final (post TEST_SH_MIN-floor,
        # TEST_SH_MAX-ceiling) worker count in both passes here -- unlike an
        # earlier version of this case (10_000 speculative pages), which
        # relied on TEST_SH_MAX clamping 10 back down to 2 while only ever
        # having memory for 3-4, and now correctly aborts instead (#483).
        (100_000, 700_000, False, 10, 2, 2, 2),
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


def test_aborts_with_a_clear_message_when_memory_cant_cover_even_one_worker(tmp_path):
    """#483: when available memory can't cover TEST_SH_MIN workers at the
    pass's own per-worker budget, test.sh must refuse to start and say why,
    rather than silently forcing TEST_SH_MIN and letting the OS SIGKILL
    pytest later with no useful signal."""
    env = _stub_env(tmp_path, 50_000, 50_000, probe_fails=False, worker_min=1, worker_max=0)
    result = subprocess.run(["bash", str(SCRIPT), "--all"], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "only 1.5" in result.stderr
    assert "need ~2.5GB for 1 worker(s)" in result.stderr
    assert "lower TEST_SH_MIN" in result.stderr


def test_light_pass_runs_but_integration_pass_aborts_on_its_own_higher_budget(tmp_path):
    """The two passes must be gated independently: enough memory for the
    light pass's 2.5GB/worker budget but not the integration pass's
    5GB/worker budget must run the light pass and abort before the second."""
    env = _stub_env(tmp_path, 150_000, 50_000, probe_fails=False, worker_min=1, worker_max=0)
    result = subprocess.run(["bash", str(SCRIPT), "--all"], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    args = result.stdout.splitlines()
    (light,) = _split_invocations(args)
    assert _marker(light) == "not integration"
    assert "only 3.0" in result.stderr
    assert "need ~5.0GB" in result.stderr


def test_probe_unavailable_still_proceeds_rather_than_aborting(tmp_path):
    """When the memory probe itself fails, there's no evidence of scarcity
    to abort on -- must keep falling back to mem_cap=1, not refuse to run."""
    env = _stub_env(tmp_path, 100, 100, probe_fails=True, worker_min=1, worker_max=0)
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "probe unavailable" in result.stderr


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
