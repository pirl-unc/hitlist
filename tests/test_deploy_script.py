"""Tests for deployment's phase-preflight retry policy (#526).

deploy.sh calls ./lint.sh and ./test.sh via relative paths, so exercising
it means running it from a temp directory that holds fake versions of
those scripts (plus fake python/twine on PATH so it never touches a real
build or a real PyPI upload) -- same stubbing approach as
tests/test_test_script.py uses for test.sh's own external commands.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).resolve().parents[1] / "deploy.sh"


def _stub_deploy_dir(tmp_path, *, test_sh_body: str) -> dict:
    """Copy deploy.sh into tmp_path with a fake lint.sh (always passes)
    and the given fake test.sh body. Fake python/twine on PATH so the
    build/upload steps (unreached if test.sh keeps failing, harmless
    no-ops if reached) never touch anything real."""
    shutil.copy(DEPLOY_SCRIPT, tmp_path / "deploy.sh")
    (tmp_path / "deploy.sh").chmod(0o755)

    (tmp_path / "lint.sh").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "lint.sh").chmod(0o755)

    (tmp_path / "test.sh").write_text(test_sh_body)
    (tmp_path / "test.sh").chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then echo 1.0.0; exit 0; fi\n'
        'if [ "$1" = "-m" ]; then mkdir -p dist && touch dist/fake.whl dist/fake.tar.gz; exit 0; fi\n'
        "exit 0\n"
    )
    (bin_dir / "python").chmod(0o755)
    (bin_dir / "twine").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "twine").chmod(0o755)

    return dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        DEPLOY_TEST_RETRY_DELAY_SECONDS="0",
    )


def _run(tmp_path, env, *args):
    return subprocess.run(
        ["bash", "deploy.sh", *args], cwd=tmp_path, env=env, capture_output=True, text=True
    )


def test_deploy_delegates_bounded_memory_retries_to_the_test_runner(tmp_path):
    """The test runner owns phase progress; deploy must invoke it just once."""
    arguments = tmp_path / "test_arguments"
    delay = tmp_path / "retry_delay"
    body = (
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" >> "{arguments}"\n'
        f'printf "%s" "$TEST_SH_MEMORY_RETRY_DELAY_SECONDS" > "{delay}"\n'
        "exit 0\n"
    )
    env = _stub_deploy_dir(tmp_path, test_sh_body=body)
    env["DEPLOY_TEST_RETRY_DELAY_SECONDS"] = "13"
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert arguments.read_text().splitlines() == ["--all", "--retry-memory"]
    assert delay.read_text() == "13"
    assert "Deploy complete" in result.stdout


def test_test_runner_failure_stops_deploy_without_replaying_any_phase(tmp_path):
    """Real failures or exhausted preflights must never reach build/upload."""
    counts_file = tmp_path / "test_sh_calls"
    body = (
        "#!/bin/sh\n"
        f'n=$(( $(cat "{counts_file}" 2>/dev/null || echo 0) + 1 ))\n'
        f'echo "$n" > "{counts_file}"\n'
        "exit 42\n"
    )
    env = _stub_deploy_dir(tmp_path, test_sh_body=body)
    result = _run(tmp_path, env)
    assert result.returncode == 42
    assert counts_file.read_text().strip() == "1"
    assert "Deploy complete" not in result.stdout
    assert not (tmp_path / "dist" / "fake.whl").exists()  # never reached the build step


def test_no_retry_message_or_delay_when_the_first_attempt_passes(tmp_path):
    """The common case: test.sh passes first try. No retry noise, no
    sleep, straight through to a successful deploy."""
    counts_file = tmp_path / "test_sh_calls"
    body = (
        "#!/bin/sh\n"
        f'n=$(( $(cat "{counts_file}" 2>/dev/null || echo 0) + 1 ))\n'
        f'echo "$n" > "{counts_file}"\n'
        "exit 0\n"
    )
    env = _stub_deploy_dir(tmp_path, test_sh_body=body)
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert counts_file.read_text().strip() == "1"  # only ran once
    assert "retrying once" not in result.stderr
    assert "Deploy complete" in result.stdout


def test_build_only_runs_all_gates_without_invoking_upload(tmp_path):
    arguments = tmp_path / "test_arguments"
    body = f'#!/bin/sh\nprintf "%s\\n" "$@" > "{arguments}"\n'
    env = _stub_deploy_dir(tmp_path, test_sh_body=body)
    (tmp_path / "bin" / "twine").write_text("#!/bin/sh\nexit 99\n")
    result = _run(tmp_path, env, "--build-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert arguments.read_text().splitlines() == ["--all", "--retry-memory"]
    assert (tmp_path / "dist" / "fake.whl").exists()
    assert "PyPI upload remains required" in result.stdout
    assert "Deploy complete" not in result.stdout


def test_build_only_test_failure_cannot_produce_artifacts(tmp_path):
    env = _stub_deploy_dir(tmp_path, test_sh_body="#!/bin/sh\nexit 42\n")
    result = _run(tmp_path, env, "--build-only")
    assert result.returncode == 42
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("gate", ["lint", "license"])
def test_build_only_cannot_ignore_other_release_gate_failures(tmp_path, gate):
    env = _stub_deploy_dir(tmp_path, test_sh_body="#!/bin/sh\nexit 0\n")
    if gate == "lint":
        (tmp_path / "lint.sh").write_text("#!/bin/sh\nexit 31\n")
    else:
        python = tmp_path / "bin" / "python"
        body = python.read_text()
        python.write_text(
            body.replace(
                "#!/bin/sh\n",
                '#!/bin/sh\nif [ "$1" = "scripts/check_distribution_license.py" ]; then exit 31; fi\n',
            )
        )
    result = _run(tmp_path, env, "--build-only")
    assert result.returncode == 31
    assert "Build complete" not in result.stdout
    assert "Deploy complete" not in result.stdout


def test_unknown_deployment_option_fails_before_any_tests(tmp_path):
    marker = tmp_path / "tests_started"
    env = _stub_deploy_dir(tmp_path, test_sh_body=f'#!/bin/sh\ntouch "{marker}"\n')
    result = _run(tmp_path, env, "--skip-tests")
    assert result.returncode == 2
    assert not marker.exists()
    assert not (tmp_path / "dist").exists()
