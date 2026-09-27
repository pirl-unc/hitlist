"""develop.sh's install check: exactly one editable hitlist install of the checkout (#553).

Every environment here is a fixture under tmp_path: fake ``*.dist-info`` /
``*.egg-info`` directories on an explicit search path, so no test reads or
touches a real virtualenv.
"""

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_dev_install.py"
SPEC = importlib.util.spec_from_file_location("check_dev_install", SCRIPT)
check_dev_install = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_dev_install)
audit = check_dev_install.audit
console_script = check_dev_install.console_script

CURRENT = "1.62.56"
PYC_TAG = "cpython-312"


def _tag(version):
    return version.replace(".", "_")


def _editable_install(site, version, source):
    """Lay out what pip's PEP 660 editable install of ``source`` leaves in ``site``.

    Its RECORD, like pip's, lists the finder's bytecode inside the
    ``__pycache__`` directory every other package in site-packages shares.
    """
    site.mkdir(parents=True, exist_ok=True)
    info = site / f"hitlist-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: hitlist\nVersion: {version}\n")
    direct_url = {"url": source.as_uri(), "dir_info": {"editable": True}}
    (info / "direct_url.json").write_text(json.dumps(direct_url))
    finder = f"__editable___hitlist_{_tag(version)}_finder"
    hooks = [f"__editable__.hitlist-{version}.pth", f"{finder}.py"]
    (site / "__pycache__").mkdir(exist_ok=True)
    (site / "__pycache__" / f"{finder}.{PYC_TAG}.pyc").write_bytes(b"")
    (site / "__pycache__" / f"six.{PYC_TAG}.pyc").write_bytes(b"")  # another package's
    for hook in hooks:
        (site / hook).write_text("")
    record = [
        "../../../bin/hitlist",
        *hooks,
        f"__pycache__/{finder}.{PYC_TAG}.pyc",
        *(f"{info.name}/{name}" for name in ("METADATA", "direct_url.json", "RECORD")),
    ]
    (info / "RECORD").write_text("".join(f"{entry},,\n" for entry in record))
    return info


def _egg_info(directory, version, files=()):
    """An egg-info: the one setuptools generates in a checkout, or a legacy install's."""
    info = directory / "hitlist.egg-info"
    info.mkdir(parents=True)
    (info / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: hitlist\nVersion: {version}\n")
    for name, text in dict(files).items():
        (info / name).write_text(text)
    return info


def _snapshot(*directories):
    return sorted(path for directory in directories for path in directory.rglob("*"))


@pytest.fixture
def checkout(tmp_path):
    root = (tmp_path / "checkout").resolve()
    (root / "hitlist").mkdir(parents=True)
    (root / "hitlist" / "__init__.py").write_text("")
    (root / "hitlist" / "version.py").write_text(f'__version__ = "{CURRENT}"\n')
    return root


@pytest.fixture
def site(tmp_path):
    return (tmp_path / "site-packages").resolve()


def _audit(root, site, version=CURRENT, module_file=None, extra_path=()):
    if module_file is None:
        module_file = root / "hitlist" / "__init__.py"
    return audit(root, version, module_file, [str(root), str(site), *map(str, extra_path)])


def test_healthy_editable_install_passes(checkout, site):
    """The egg-info setuptools regenerates at the current version is not a duplicate."""
    info = _editable_install(site, CURRENT, checkout)
    _egg_info(checkout, CURRENT)
    result = _audit(checkout, site)
    assert result.problems == []
    assert result.to_move == []
    assert result.reinstall is None
    assert result.kept == info
    assert result.n_distributions == 2


def test_issue_553_extra_registrations_are_listed_file_by_file(checkout, site):
    """The reported environment: a stale checkout egg-info plus five registrations."""
    _egg_info(checkout, "1.62.21")
    stale_versions = ("1.62.21", "1.62.42", "1.62.43", "1.62.44")
    for version in (*stale_versions, CURRENT):
        _editable_install(site, version, checkout)
    # From inside the checkout, importlib.metadata answers with the stale
    # egg-info: the #553 symptom that import-path and CLI checks cannot see.
    first = next(importlib.metadata.distributions(name="hitlist", path=[str(checkout), str(site)]))
    assert first.version == "1.62.21"
    before = _snapshot(checkout, site)

    result = _audit(checkout, site)

    assert result.kept == site / f"hitlist-{CURRENT}.dist-info"
    assert result.n_distributions == 6
    assert result.reinstall is None
    assert sorted(result.problems) == sorted(
        [f"{checkout / 'hitlist.egg-info'}: version 1.62.21, not the checkout's {CURRENT}"]
        + [
            f"{site / f'hitlist-{version}.dist-info'}: extra registration; "
            f"version {version}, not the checkout's {CURRENT}"
            for version in stale_versions
        ]
    )
    # Each extra's metadata, hooks and its own finder bytecode -- never the
    # shared __pycache__ directory (other packages' bytecode lives there), the
    # kept registration, the checkout's sources, or bin/hitlist.
    expected = [checkout / "hitlist.egg-info"]
    for version in stale_versions:
        finder = f"__editable___hitlist_{_tag(version)}_finder"
        expected += [
            site / f"__editable__.hitlist-{version}.pth",
            site / f"{finder}.py",
            site / "__pycache__" / f"{finder}.{PYC_TAG}.pyc",
            site / f"hitlist-{version}.dist-info",
        ]
    assert sorted(result.to_move) == sorted(expected)
    assert site / "__pycache__" not in result.to_move
    assert _snapshot(checkout, site) == before


def test_sole_stale_registration_is_reinstalled_not_moved(checkout, site):
    """Right after a version bump the only registration is stale; moving it breaks import."""
    info = _editable_install(site, "1.62.55", checkout)
    result = _audit(checkout, site)
    assert result.to_move == []
    assert result.kept is None
    assert result.problems == [f"{info}: version 1.62.55, not the checkout's {CURRENT}"]
    assert result.reinstall == (
        f"re-run ./develop.sh in {checkout}; it reinstalls over {info}, so leave that one in place"
    )


def test_sole_registration_of_another_checkout_warns_about_the_shared_env(tmp_path, checkout, site):
    """Run from a worktree against a venv installed from main: never list main's registration."""
    main = (tmp_path / "main").resolve()
    info = _editable_install(site, CURRENT, main)
    result = _audit(checkout, site)
    assert result.to_move == []
    assert result.problems == [f"{info}: editable install of {main}"]
    assert f"re-points everyone using this environment from {main}" in result.reinstall


def test_second_matching_registration_is_a_duplicate(tmp_path, checkout, site):
    first = _editable_install(site, CURRENT, checkout)
    second_site = (tmp_path / "user-site").resolve()
    second = _editable_install(second_site, CURRENT, checkout)
    result = _audit(checkout, site, extra_path=[second_site])
    assert result.kept == first
    assert result.problems == [f"{second}: extra registration; duplicate of {first}"]
    assert second in result.to_move
    assert result.reinstall is None


@pytest.mark.parametrize(
    "direct_url",
    ['{"url": "file:///x", "dir_info": {"edita', '{"dir_info": {"editable": true}}', "[]"],
    ids=["truncated", "no-url", "not-an-object"],
)
def test_malformed_direct_url_is_a_problem_not_a_crash(checkout, site, direct_url):
    """A half-written registration, e.g. from an interrupted or concurrent install."""
    info = _editable_install(site, CURRENT, checkout)
    (info / "direct_url.json").write_text(direct_url)
    result = _audit(checkout, site)
    assert len(result.problems) == 1
    assert result.problems[0].startswith(f"{info}: unreadable direct_url.json (")
    assert result.reinstall is not None


def test_registration_without_metadata_is_a_problem_not_a_crash(checkout, site):
    info = _editable_install(site, CURRENT, checkout)
    (info / "METADATA").unlink()
    result = _audit(checkout, site)
    assert result.problems == [f"{info}: no METADATA or PKG-INFO version (half-written?)"]


def test_absolute_and_outside_record_entries_are_never_listed(checkout, site):
    _editable_install(site, CURRENT, checkout)
    stale = _editable_install(site, "1.62.21", checkout)
    with (stale / "RECORD").open("a") as record:
        record.write(f"{site.anchor},,\n{site.parent},,\n../outside.txt,,\n")
    (site.parent / "outside.txt").write_text("")
    result = _audit(checkout, site)
    assert all(path.is_relative_to(site) and path != site for path in result.to_move)


def test_non_editable_install_shadowing_the_checkout_lists_its_package(checkout, site):
    """A regular wheel install wins the import over the editable finder; list its package too."""
    (site / "hitlist").mkdir(parents=True)
    (site / "hitlist" / "__init__.py").write_text("")
    wheel = site / "hitlist-1.62.21.dist-info"
    wheel.mkdir()
    (wheel / "METADATA").write_text("Metadata-Version: 2.1\nName: hitlist\nVersion: 1.62.21\n")
    (wheel / "RECORD").write_text(
        "hitlist/__init__.py,,\nhitlist-1.62.21.dist-info/METADATA,,\n../../../bin/hitlist,,\n"
    )
    _editable_install(site, CURRENT, checkout)
    result = _audit(checkout, site, module_file=site / "hitlist" / "__init__.py")
    assert result.problems[0] == (
        f"import hitlist resolves to {site / 'hitlist' / '__init__.py'}, not {checkout / 'hitlist'}"
    )
    assert result.problems[1] == (
        f"{wheel}: extra registration; "
        f"version 1.62.21, not the checkout's {CURRENT}; not an editable install"
    )
    assert sorted(result.to_move) == [site / "hitlist", wheel]


def test_legacy_egg_info_install_lists_the_package_it_installed(checkout, site):
    """installed-files.txt, relative to the egg-info, names the package that keeps shadowing."""
    _editable_install(site, CURRENT, checkout)
    (site / "hitlist").mkdir()
    (site / "hitlist" / "__init__.py").write_text("")
    legacy = _egg_info(
        site,
        "1.40.0",
        {"installed-files.txt": "../hitlist/__init__.py\nPKG-INFO\n", "top_level.txt": "hitlist\n"},
    )
    result = _audit(checkout, site)
    assert sorted(result.to_move) == [site / "hitlist", legacy]


def test_build_tree_egg_info_elsewhere_never_lists_that_checkouts_sources(tmp_path, checkout, site):
    """An egg-info with no install record (another checkout on sys.path) lists only itself."""
    _editable_install(site, CURRENT, checkout)
    other = (tmp_path / "other-checkout").resolve()
    (other / "hitlist").mkdir(parents=True)
    (other / "hitlist" / "__init__.py").write_text("")
    foreign = _egg_info(
        other, "1.50.0", {"SOURCES.txt": "hitlist/__init__.py\n", "top_level.txt": "hitlist\n"}
    )
    result = _audit(checkout, site, extra_path=[other])
    assert result.to_move == [foreign]


@pytest.mark.parametrize("module_file", [None, "elsewhere"])
def test_import_that_misses_the_checkout_is_a_problem(tmp_path, checkout, site, module_file):
    _editable_install(site, CURRENT, checkout)
    if module_file == "elsewhere":
        module_file = tmp_path / "elsewhere" / "hitlist" / "__init__.py"
        expected = f"import hitlist resolves to {module_file}, not {checkout / 'hitlist'}"
    else:
        expected = "import hitlist finds no regular package"
    result = audit(checkout, CURRENT, module_file, [str(checkout), str(site)])
    assert result.problems == [expected]


def test_checkout_metadata_alone_is_not_an_install(checkout, site):
    """Importable only through sys.path (e.g. PYTHONPATH): nothing is installed."""
    _egg_info(checkout, CURRENT)
    result = _audit(checkout, site)
    assert result.problems == [
        "no installed hitlist metadata for importlib.metadata or pip to find"
    ]
    assert result.to_move == []
    assert result.reinstall == f"re-run ./develop.sh in {checkout} to install this checkout"


def _stub(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def test_console_script_of_this_interpreter_passes(tmp_path):
    cli = _stub(tmp_path / "bin" / "hitlist", f"#!/bin/sh\necho 'hitlist {CURRENT}'\n")
    assert console_script(str(cli), CURRENT, str(tmp_path / "bin")) == (f"hitlist {CURRENT}", [])


def test_console_script_with_a_dead_shebang_is_a_problem_not_a_crash(tmp_path):
    """A leftover bin/hitlist whose interpreter was deleted."""
    cli = _stub(tmp_path / "bin" / "hitlist", "#!/nonexistent/venv/bin/python\n")
    reported, problems = console_script(str(cli), CURRENT, str(tmp_path / "bin"))
    assert reported == "did not run"
    assert len(problems) == 1
    assert problems[0].startswith(f"`{cli} --version` cannot run: ")


def test_console_script_failure_reports_exit_code_and_stderr(tmp_path):
    cli = _stub(tmp_path / "bin" / "hitlist", "#!/bin/sh\necho 'ImportError: boom' >&2\nexit 3\n")
    _, problems = console_script(str(cli), CURRENT, str(tmp_path / "bin"))
    assert problems == [f"`{cli} --version` exited 3: ImportError: boom"]


def test_same_version_console_script_from_another_env_is_a_problem(tmp_path):
    other = _stub(
        tmp_path / "other-env" / "bin" / "hitlist", f"#!/bin/sh\necho 'hitlist {CURRENT}'\n"
    )
    _, problems = console_script(str(other), CURRENT, str(tmp_path / "this-env" / "bin"))
    assert problems == [
        f"`hitlist` on PATH is {other}, not this interpreter's (in {tmp_path / 'this-env' / 'bin'})"
    ]


def _run_script(checkout, site, tmp_path, cli_body=None):
    """Run a copy of the script as if it lived in ``checkout``, isolated from this venv.

    ``-S`` keeps the real site-packages (and its real hitlist registration)
    off sys.path; PYTHONPATH supplies the fixture checkout and site directory.
    """
    (checkout / "scripts").mkdir()
    shutil.copy(SCRIPT, checkout / "scripts" / SCRIPT.name)
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    if cli_body is not None:
        _stub(stub_bin / "hitlist", cli_body)
    env = {"PATH": str(stub_bin), "PYTHONPATH": os.pathsep.join([str(checkout), str(site)])}
    return subprocess.run(
        [sys.executable, "-S", str(checkout / "scripts" / SCRIPT.name)],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
    )


def test_script_passes_a_healthy_fixture_without_importing_hitlist(tmp_path, checkout, site):
    """Importing hitlist has side effects (#579); the check must only locate it."""
    marker = tmp_path / "imported"
    (checkout / "hitlist" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    _editable_install(site, CURRENT, checkout)
    _egg_info(checkout, CURRENT)
    result = _run_script(checkout, site, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"metadata          -> {CURRENT}  editable install of {checkout}" in result.stdout
    assert f"import hitlist    -> {checkout / 'hitlist' / '__init__.py'}" in result.stdout
    assert not marker.exists()


def test_script_fails_on_stale_metadata_without_removing_it(tmp_path, checkout, site):
    _editable_install(site, CURRENT, checkout)
    stale = _editable_install(site, "1.62.21", checkout)
    stale_egg_info = _egg_info(checkout, "1.62.21")
    before = _snapshot(checkout, site)
    result = _run_script(checkout, site, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "3 hitlist distributions visible" in result.stderr
    assert f"{stale}: extra registration; version 1.62.21" in result.stderr
    assert f"       {stale_egg_info}\n" in result.stderr
    assert "Nothing was removed" in result.stderr
    assert "Traceback" not in result.stderr
    assert set(before) <= set(_snapshot(checkout, site))  # nothing removed


def test_script_reports_metadata_even_when_the_cli_cannot_run(tmp_path, checkout, site):
    _editable_install(site, CURRENT, checkout)
    result = _run_script(checkout, site, tmp_path, cli_body="#!/nonexistent/venv/bin/python\n")
    assert result.returncode == 1
    assert f"metadata          -> {CURRENT}" in result.stdout
    assert "--version` cannot run: " in result.stderr
    assert "Traceback" not in result.stderr


def test_script_diagnoses_an_unimportable_package(tmp_path, checkout, site):
    _editable_install(site, CURRENT, checkout)
    (checkout / "hitlist" / "__init__.py").unlink()  # leaves only a namespace directory
    result = _run_script(checkout, site, tmp_path)
    assert result.returncode == 1
    assert "import hitlist    -> namespace package" in result.stdout
    assert "import hitlist finds no regular package" in result.stderr


@pytest.mark.parametrize("check_status", [0, 1])
def test_develop_sh_exits_with_the_check(tmp_path, check_status):
    """develop.sh must fail when the post-install check fails, not just print it."""
    shutil.copy(REPO / "develop.sh", tmp_path / "develop.sh")
    calls = tmp_path / "python_calls"
    _stub(tmp_path / "uv", "#!/bin/sh\nexit 0\n")
    _stub(
        tmp_path / "python", f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{calls}"\nexit {check_status}\n'
    )
    env = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        VIRTUAL_ENV=str(tmp_path / "venv"),
    )
    result = subprocess.run(
        ["bash", "develop.sh"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == check_status, result.stdout + result.stderr
    assert calls.read_text().splitlines() == ["scripts/check_dev_install.py"]


def test_lint_and_format_cover_the_script_through_one_list(tmp_path):
    """lint.sh and format.sh pass no paths, so pyproject's include list is the only list."""
    calls = tmp_path / "ruff_calls"
    _stub(tmp_path / "ruff", f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{calls}"\n')
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    for script in ("lint.sh", "format.sh"):
        subprocess.run(["bash", script], cwd=REPO, env=env, check=True, capture_output=True)
    assert calls.read_text().splitlines() == ["check", "format --check", "format"]
    listed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--show-files"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert str(SCRIPT) in listed.splitlines()
