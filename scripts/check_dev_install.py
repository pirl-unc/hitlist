"""Fail unless this environment's hitlist is one editable install of this checkout (#553).

./develop.sh runs this after installing; run it directly any time.  Importing
hitlist and running ``hitlist --version`` show only which *code* runs.
``importlib.metadata`` (and pip, and anything else asking which hitlist is
installed) reads distribution metadata stored separately from that code, which
can disagree with it and can exist more than once.  #553 found an environment
whose import was 1.62.56 from the intended checkout while
``importlib.metadata.version("hitlist")`` said 1.62.21, site-packages held
editable registrations for five versions, and ``pip check`` passed.

Every ``hitlist`` metadata directory visible to this interpreter must carry the
checkout's version, and exactly one of them must be an installed editable
install of this checkout.  The checkout itself is searched too: ``python -c``
and ``python -m pytest`` started there put it first on sys.path, where
setuptools leaves a generated ``hitlist.egg-info`` that
``pip install --no-build-isolation -e .`` does not refresh.

This never imports hitlist -- the import has side effects (#579), and a broken
install must still be diagnosed -- and never removes anything: on a mismatch it
exits 1 and says what to move aside and whether to reinstall.  It is a
standalone script, not a hitlist module, so a stale installed copy cannot be
the code that checks itself.
"""

import csv
import email
import importlib.metadata
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hitlist"


class Registration(NamedTuple):
    path: Path  # the metadata directory
    installed: bool  # False for the checkout's own generated egg-info
    source: Optional[Path]  # where an editable install points, if it is one
    reasons: list  # why it does not match the checkout; empty when it does


class Audit(NamedTuple):
    n_distributions: int
    kept: Optional[Path]  # the one matching installed registration, if any
    problems: list
    to_move: list  # extra registrations and stale checkout metadata
    reinstall: Optional[str]  # why ./develop.sh must be re-run, if it must


def metadata_dirs(search_path):
    """Every hitlist metadata directory on ``search_path``, in lookup order.

    Matched by name as importlib.metadata matches them, case-insensitively:
    ``hitlist-<version>.dist-info``, ``hitlist.egg-info`` and
    ``hitlist-<version>[-pyX.Y].egg-info``.
    """
    found = []
    for entry in search_path:
        directory = Path(entry or os.curdir)
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            stem, suffix = os.path.splitext(child.name.lower())
            if (
                suffix in (".dist-info", ".egg-info")
                and stem.partition("-")[0] == PACKAGE
                and child.is_dir()
                and child.resolve() not in found
            ):
                found.append(child.resolve())
    return found


def editable_source(text):
    """Where a PEP 610 ``direct_url.json`` says an editable install points, or None."""
    if text is None:
        return None
    direct_url = json.loads(text)
    if not direct_url.get("dir_info", {}).get("editable"):
        return None
    return Path(url2pathname(urlparse(direct_url["url"]).path)).resolve()


def examine(path, root, version):
    """Compare one metadata directory with the checkout at ``root``."""
    distribution = importlib.metadata.Distribution.at(path)
    reasons = []
    text = distribution.read_text("METADATA") or distribution.read_text("PKG-INFO")
    found_version = email.message_from_string(text)["Version"] if text else None
    if found_version is None:
        reasons.append("no METADATA or PKG-INFO version (half-written?)")
    elif found_version != version:
        reasons.append(f"version {found_version}, not the checkout's {version}")
    installed = path.parent != root
    source = None
    if installed:
        try:
            source = editable_source(distribution.read_text("direct_url.json"))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            reasons.append(f"unreadable direct_url.json ({error!r}; half-written?)")
        else:
            if source is None:
                reasons.append("not an editable install")
            elif source != root:
                reasons.append(f"editable install of {source}")
    return Registration(path, installed, source, reasons)


def owned_paths(path):
    """The metadata directory plus what its install record says it put beside it.

    A wheel's RECORD (relative to site-packages) or a legacy install's
    installed-files.txt (relative to the egg-info) lists every installed file.
    Only the metadata directory itself and the hitlist package directory are
    listed whole; anything else -- the editable finder's bytecode in the shared
    ``__pycache__``, say -- is listed file by file.  Entries outside the site
    directory (``../../../bin/hitlist``, absolute paths) are skipped: other
    installs share them.  A build-tree egg-info has no install record, so a
    checkout's sources are never listed.
    """
    site = path.parent
    distribution = importlib.metadata.Distribution.at(path)
    if path.suffix == ".dist-info":
        rows = csv.reader((distribution.read_text("RECORD") or "").splitlines())
        base, entries = site, [row[0] for row in rows if row]
    else:
        base, entries = path, (distribution.read_text("installed-files.txt") or "").splitlines()
    owned = {path}
    for entry in entries:
        if not entry or os.path.isabs(entry):
            continue
        target = Path(os.path.normpath(base / entry))
        if target == site or not target.is_relative_to(site):
            continue
        top = target.relative_to(site).parts[0]
        owned.add(site / top if top in (path.name, PACKAGE) else target)
    return {owned_path for owned_path in owned if owned_path.exists()}


def audit(root, version, module_file, search_path):
    """Compare where ``import hitlist`` would load from, and all hitlist metadata, with root."""
    problems = []
    if module_file is None:
        problems.append(f"import {PACKAGE} finds no regular package")
    elif Path(module_file).resolve().parent != root / PACKAGE:
        problems.append(f"import {PACKAGE} resolves to {module_file}, not {root / PACKAGE}")
    registrations = [examine(path, root, version) for path in metadata_dirs(search_path)]
    # The registration to keep is the one a reinstall repairs in place: a
    # matching one, else one of this checkout, else whichever is found first.
    # Every other installed registration is an extra and gets moved aside.
    ranked = sorted(
        (r for r in registrations if r.installed), key=lambda r: (bool(r.reasons), r.source != root)
    )
    keep = ranked[0] if ranked else None
    kept_paths = owned_paths(keep.path) if keep else set()
    to_move = []
    for registration in registrations:
        if registration is keep:
            continue
        if not registration.installed:
            if registration.reasons:
                problems.append(f"{registration.path}: {'; '.join(registration.reasons)}")
                to_move.append(registration.path)
            continue
        reasons = registration.reasons or [f"duplicate of {keep.path}"]
        problems.append(f"{registration.path}: extra registration; {'; '.join(reasons)}")
        to_move.extend(sorted(owned_paths(registration.path) - kept_paths))
    reinstall = None
    if keep is None:
        problems.append(f"no installed {PACKAGE} metadata for importlib.metadata or pip to find")
        reinstall = f"re-run ./develop.sh in {root} to install this checkout"
    elif keep.reasons:
        problems.append(f"{keep.path}: {'; '.join(keep.reasons)}")
        reinstall = (
            f"re-run ./develop.sh in {root}; it reinstalls over {keep.path}, "
            "so leave that one in place"
        )
        if keep.source is not None and keep.source != root:
            reinstall += (
                f". That re-points everyone using this environment from {keep.source} "
                f"to {root}; use a separate environment if {keep.source} must keep working"
            )
    kept = keep.path if keep and not keep.reasons else None
    return Audit(len(registrations), kept, problems, to_move, reinstall)


def console_script(cli, version, scripts_dir):
    """What ``hitlist --version`` on PATH reports, and what is wrong with it."""
    problems = []
    if Path(cli).resolve().parent != Path(scripts_dir).resolve():
        problems.append(f"`{PACKAGE}` on PATH is {cli}, not this interpreter's (in {scripts_dir})")
    try:
        completed = subprocess.run([cli, "--version"], capture_output=True, text=True)
    except OSError as error:
        problems.append(
            f"`{cli} --version` cannot run: {error} (also what a missing #! interpreter reports)"
        )
        return "did not run", problems
    reported = completed.stdout.strip()
    if completed.returncode != 0:
        stderr = completed.stderr.strip().replace("\n", "\n      ")
        problems.append(f"`{cli} --version` exited {completed.returncode}: {stderr}")
    elif reported != f"{PACKAGE} {version}":
        problems.append(f"`{cli} --version` reports {reported!r}, not '{PACKAGE} {version}'")
    return reported, problems


def main():
    version = runpy.run_path(str(ROOT / PACKAGE / "version.py"))["__version__"]
    spec = importlib.util.find_spec(PACKAGE)
    module_file = spec.origin if spec else None
    if spec is None:
        located = "not importable"
    else:
        located = module_file or f"namespace package {list(spec.submodule_search_locations)}"
    print(f"python            -> {sys.executable}")
    print(f"checkout          -> {version}  ({ROOT})")
    print(f"import {PACKAGE}    -> {located}")
    result = audit(ROOT, version, module_file, [str(ROOT), *sys.path])
    if result.kept is not None:
        print(f"metadata          -> {version}  editable install of {ROOT}  ({result.kept})")
    problems = list(result.problems)
    cli = shutil.which(PACKAGE)
    if cli is None:
        print(f"WARNING: no `{PACKAGE}` on PATH")
    else:
        reported, cli_problems = console_script(cli, version, sysconfig.get_path("scripts"))
        print(f"`{PACKAGE}` on PATH -> {cli}  ({reported})")
        problems += cli_problems
    if not problems:
        return 0
    sys.stdout.flush()  # keep the report above the error when both are piped
    plural = "" if result.n_distributions == 1 else "s"
    print(
        f"\nERROR: this environment's {PACKAGE} does not match {ROOT} at {version} "
        f"({result.n_distributions} {PACKAGE} distribution{plural} visible; #553):",
        file=sys.stderr,
    )
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    steps = []
    if result.to_move:
        paths = "".join(f"\n       {path}" for path in result.to_move)
        steps.append(f"Move these out of the environment (or delete them):{paths}")
    if result.reinstall:
        steps.append(result.reinstall[0].upper() + result.reinstall[1:] + ".")
    if steps:
        print("Nothing was removed. To fix:", file=sys.stderr)
        for step_number, step in enumerate(steps, start=1):
            print(f"  {step_number}. {step}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
