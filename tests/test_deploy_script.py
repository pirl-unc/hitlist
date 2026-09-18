"""Tests for deploy.sh's retry-once-after-a-delay behavior (#483).

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


def _run(tmp_path, env):
    return subprocess.run(
        ["bash", "deploy.sh"], cwd=tmp_path, env=env, capture_output=True, text=True
    )


def test_retries_once_after_a_transient_failure_and_then_succeeds(tmp_path):
    """A first test.sh failure followed by a passing retry must let the
    deploy proceed (reach the build/upload steps) rather than aborting."""
    counts_file = tmp_path / "test_sh_calls"
    body = (
        "#!/bin/sh\n"
        f'n=$(( $(cat "{counts_file}" 2>/dev/null || echo 0) + 1 ))\n'
        f'echo "$n" > "{counts_file}"\n'
        '[ "$n" -eq 1 ] && exit 1\n'
        "exit 0\n"
    )
    env = _stub_deploy_dir(tmp_path, test_sh_body=body)
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert counts_file.read_text().strip() == "2"  # first call failed, second (retry) ran
    assert "Deploy complete" in result.stdout


def test_gives_up_after_the_retry_also_fails(tmp_path):
    """Two consecutive test.sh failures must abort the deploy for real --
    the retry is a single attempt, not a loop that could mask a genuine
    break forever."""
    counts_file = tmp_path / "test_sh_calls"
    body = (
        "#!/bin/sh\n"
        f'n=$(( $(cat "{counts_file}" 2>/dev/null || echo 0) + 1 ))\n'
        f'echo "$n" > "{counts_file}"\n'
        "exit 1\n"
    )
    env = _stub_deploy_dir(tmp_path, test_sh_body=body)
    result = _run(tmp_path, env)
    assert result.returncode != 0
    assert counts_file.read_text().strip() == "2"  # exactly one retry, not more
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
