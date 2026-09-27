"""develop.sh's install check: exactly one editable hitlist install of the checkout (#553).

Every environment here is a fixture under tmp_path: fake ``*.dist-info`` /
``*.egg-info`` directories discovered by ``importlib.metadata`` from an
explicit search path, so no test reads or touches a real virtualenv.
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

CURRENT = "1.62.56"


def _editable_install(site, version, source):
    """Lay out what a PEP 660 editable install of ``source`` leaves in ``site``."""
    site.mkdir(parents=True, exist_ok=True)
    info = site / f"hitlist-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: hitlist\nVersion: {version}\n")
    direct_url = {"url": source.as_uri(), "dir_info": {"editable": True}}
    (info / "direct_url.json").write_text(json.dumps(direct_url))
    hooks = [
        f"__editable__.hitlist-{version}.pth",
        f"__editable___hitlist_{version.replace('.', '_')}_finder.py",
    ]
    for hook in hooks:
        (site / hook).write_text("")
    record = [*hooks, *(f"{info.name}/{name}" for name in ("METADATA", "RECORD"))]
    record.append("../../../bin/hitlist")
    (info / "RECORD").write_text("".join(f"{entry},,\n" for entry in record))
    return info


def _egg_info(directory, version):
    """The metadata setuptools generates inside the checkout during a build."""
    info = directory / "hitlist.egg-info"
    info.mkdir()
    (info / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: hitlist\nVersion: {version}\n")
    return info


def _snapshot(*directories):
    return sorted(path for directory in directories for path in directory.rglob("*"))


@pytest.fixture
def checkout(tmp_path):
    root = (tmp_path / "checkout").resolve()
    (root / "hitlist").mkdir(parents=True)
    (root / "hitlist" / "__init__.py").write_text(f'__version__ = "{CURRENT}"\n')
    return root


@pytest.fixture
def site(tmp_path):
    return (tmp_path / "site-packages").resolve()


def _audit(root, site, version=CURRENT, module_file=None):
    module_file = module_file or root / "hitlist" / "__init__.py"
    return audit(root, version, module_file, [str(root), str(site)])


def test_healthy_editable_install_passes(checkout, site):
    """The egg-info setuptools regenerates at the current version is not a duplicate."""
    info = _editable_install(site, CURRENT, checkout)
    _egg_info(checkout, CURRENT)
    result = _audit(checkout, site)
    assert result.problems == []
    assert result.to_move == []
    assert result.kept == info
    assert result.n_distributions == 2


def test_issue_553_stale_and_duplicate_metadata_is_named_not_removed(checkout, site):
    """The reported environment: a stale checkout egg-info plus five registrations."""
    _egg_info(checkout, "1.62.21")
    stale_versions = ("1.62.21", "1.62.42", "1.62.43", "1.62.44")
    for version in (*stale_versions, CURRENT):
        _editable_install(site, version, checkout)
    # From inside the checkout, importlib.metadata answers with the stale
    # egg-info: the #553 symptom that import-path and CLI checks cannot see.
    search_path = [str(checkout), str(site)]
    first = next(importlib.metadata.distributions(name="hitlist", path=search_path))
    assert first.version == "1.62.21"
    before = _snapshot(checkout, site)

    result = _audit(checkout, site)

    assert result.kept == site / f"hitlist-{CURRENT}.dist-info"
    assert result.n_distributions == 6
    assert sorted(result.problems) == sorted(
        [f"{checkout / 'hitlist.egg-info'}: version 1.62.21, not the imported {CURRENT}"]
        + [
            f"{site / f'hitlist-{version}.dist-info'}: version {version}, not the imported {CURRENT}"
            for version in stale_versions
        ]
    )
    # Each stale registration's metadata and editable hooks -- never the kept
    # registration, the checkout's sources, or the console script its RECORD
    # lists outside the site directory, which every install shares.
    expected = [checkout / "hitlist.egg-info"]
    for version in stale_versions:
        expected += [
            site / f"__editable___hitlist_{version.replace('.', '_')}_finder.py",
            site / f"__editable__.hitlist-{version}.pth",
            site / f"hitlist-{version}.dist-info",
        ]
    assert sorted(result.to_move) == sorted(expected)
    assert _snapshot(checkout, site) == before


def test_registration_of_another_checkout_is_rejected(tmp_path, checkout, site):
    """Same version, wrong source: e.g. a sibling worktree installed last."""
    other = tmp_path / "other-worktree"
    info = _editable_install(site, CURRENT, other)
    result = _audit(checkout, site)
    assert result.kept is None
    assert result.problems == [
        f"{info}: editable install of {other.resolve()}",
        f"no installed hitlist {CURRENT} is an editable install of {checkout}",
    ]


def test_second_matching_registration_is_a_duplicate(tmp_path, checkout, site):
    first = _editable_install(site, CURRENT, checkout)
    second_site = (tmp_path / "user-site").resolve()
    second = _editable_install(second_site, CURRENT, checkout)
    result = audit(
        checkout,
        CURRENT,
        checkout / "hitlist" / "__init__.py",
        [str(checkout), str(site), str(second_site)],
    )
    assert result.kept == first
    assert result.problems == [f"{second}: duplicate of {first}"]
    assert second in result.to_move


def test_non_editable_install_shadowing_the_checkout_is_rejected(checkout, site):
    """A regular wheel install wins the import; its package directory is listed too."""
    (site / "hitlist").mkdir(parents=True)
    (site / "hitlist" / "__init__.py").write_text('__version__ = "1.62.21"\n')
    info = site / "hitlist-1.62.21.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: hitlist\nVersion: 1.62.21\n")
    (info / "RECORD").write_text(
        "hitlist/__init__.py,,\nhitlist-1.62.21.dist-info/METADATA,,\n../../../bin/hitlist,,\n"
    )
    result = _audit(checkout, site, "1.62.21", site / "hitlist" / "__init__.py")
    assert result.problems == [
        f"import hitlist resolves to {site / 'hitlist' / '__init__.py'}, not {checkout / 'hitlist'}",
        f"{info}: not an editable install",
        f"no installed hitlist 1.62.21 is an editable install of {checkout}",
    ]
    assert result.to_move == [site / "hitlist", info]


def test_checkout_metadata_alone_is_not_an_install(checkout, site):
    """Importable only through sys.path (e.g. PYTHONPATH): nothing is installed."""
    _egg_info(checkout, CURRENT)
    result = _audit(checkout, site)
    assert result.problems == [
        f"no installed hitlist {CURRENT} is an editable install of {checkout}"
    ]
    assert result.to_move == []


def _run_script(checkout, site, tmp_path, cli_version=CURRENT):
    """Run a copy of the script as if it lived in ``checkout``, isolated from this venv.

    ``-S`` keeps the real site-packages (and its real hitlist registration)
    off sys.path; PYTHONPATH supplies the fixture checkout and site directory.
    """
    (checkout / "scripts").mkdir()
    shutil.copy(SCRIPT, checkout / "scripts" / SCRIPT.name)
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    (stub_bin / "hitlist").write_text(f"#!/bin/sh\necho 'hitlist {cli_version}'\n")
    (stub_bin / "hitlist").chmod(0o755)
    env = {"PATH": str(stub_bin), "PYTHONPATH": os.pathsep.join([str(checkout), str(site)])}
    return subprocess.run(
        [sys.executable, "-S", str(checkout / "scripts" / SCRIPT.name)],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
    )


def test_script_passes_a_healthy_fixture(tmp_path, checkout, site):
    _editable_install(site, CURRENT, checkout)
    _egg_info(checkout, CURRENT)
    result = _run_script(checkout, site, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"metadata          -> {CURRENT}  editable install of {checkout}" in result.stdout
    assert result.stderr == ""


def test_script_fails_on_stale_metadata_without_removing_it(tmp_path, checkout, site):
    _editable_install(site, CURRENT, checkout)
    stale = _editable_install(site, "1.62.21", checkout)
    stale_egg_info = _egg_info(checkout, "1.62.21")
    result = _run_script(checkout, site, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "3 hitlist distributions visible" in result.stderr
    assert f"{stale}: version 1.62.21, not the imported {CURRENT}" in result.stderr
    assert f"  {stale_egg_info}\n" in result.stderr
    assert "Nothing was removed" in result.stderr
    assert stale.is_dir()
    assert stale_egg_info.is_dir()


def test_script_fails_when_the_console_script_is_another_install(tmp_path, checkout, site):
    _editable_install(site, CURRENT, checkout)
    result = _run_script(checkout, site, tmp_path, cli_version="1.55.0")
    assert result.returncode == 1
    assert "`hitlist` on PATH reports 'hitlist 1.55.0'; another install shadows it" in result.stderr


@pytest.mark.parametrize("check_status", [0, 1])
def test_develop_sh_exits_with_the_check(tmp_path, check_status):
    """develop.sh must fail when the post-install check fails, not just print it."""
    shutil.copy(REPO / "develop.sh", tmp_path / "develop.sh")
    calls = tmp_path / "python_calls"
    stubs = {
        "uv": "exit 0",
        "python": f'printf "%s\\n" "$*" >> "{calls}"\nexit {check_status}',
    }
    for name, body in stubs.items():
        (tmp_path / name).write_text(f"#!/bin/sh\n{body}\n")
        (tmp_path / name).chmod(0o755)
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
