"""Fail unless this environment's hitlist is one editable install of this checkout (#553).

./develop.sh runs this after installing; run it directly any time.  Importing
hitlist and running ``hitlist --version`` show only which *code* runs.
``importlib.metadata`` (and pip, and anything else asking which hitlist is
installed) reads distribution metadata stored separately from that code, which
can disagree with it and can exist more than once.  #553 found an environment
whose import was 1.62.56 from the intended checkout while
``importlib.metadata.version("hitlist")`` said 1.62.21, site-packages held
editable registrations for five versions, and ``pip check`` passed.

Every ``hitlist`` distribution visible to this interpreter must carry the
imported version, and exactly one of them must be an installed (site-packages)
editable install of this checkout.  The checkout itself is searched too:
``python -c`` and ``python -m pytest`` started there put it first on sys.path,
where setuptools leaves a generated ``hitlist.egg-info`` that
``pip install --no-build-isolation -e .`` does not refresh.

Nothing is ever removed.  On a mismatch this exits 1 and lists what to move
aside.  It is a standalone script rather than a hitlist module so that a stale
installed copy of hitlist cannot be the code that checks itself.
"""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

import hitlist

ROOT = Path(__file__).resolve().parents[1]


class Audit(NamedTuple):
    n_distributions: int
    kept: Optional[Path]
    problems: list
    to_move: list


def metadata_dir(distribution):
    # importlib.metadata has no public accessor for where a distribution's
    # metadata lives; PathDistribution has kept it in ``_path`` since 3.8.
    return Path(distribution._path).resolve()


def editable_source(distribution):
    """The directory an editable install points at (PEP 610), or None."""
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if not direct_url.get("dir_info", {}).get("editable"):
        return None
    return Path(url2pathname(urlparse(direct_url["url"]).path)).resolve()


def registration(path, distribution):
    """The metadata directory plus the files its RECORD placed beside it.

    Only a wheel install's RECORD lists what it put in site-packages; an
    egg-info's file list names the checkout's own sources.  Entries outside
    the directory (``../../../bin/hitlist``) are shared by every install.
    """
    names = {path.name}
    if path.suffix == ".dist-info":
        names.update(
            entry.parts[0] for entry in distribution.files or () if not str(entry).startswith("..")
        )
    return sorted(path.parent / name for name in names if (path.parent / name).exists())


def audit(root, version, module_file, search_path):
    """Compare every hitlist distribution on ``search_path`` with the imported code."""
    problems = []
    to_move = []
    if Path(module_file).resolve().parent != root / "hitlist":
        problems.append(f"import hitlist resolves to {module_file}, not {root / 'hitlist'}")
    distributions = {}
    for distribution in importlib.metadata.distributions(name="hitlist", path=search_path):
        distributions.setdefault(metadata_dir(distribution), distribution)
    kept = None
    for path, distribution in distributions.items():
        reasons = []
        if distribution.version != version:
            reasons.append(f"version {distribution.version}, not the imported {version}")
        if path.parent != root:
            source = editable_source(distribution)
            if source is None:
                reasons.append("not an editable install")
            elif source != root:
                reasons.append(f"editable install of {source}")
            elif not reasons and kept is None:
                kept = path
            elif not reasons:
                reasons.append(f"duplicate of {kept}")
        if reasons:
            problems.append(f"{path}: {'; '.join(reasons)}")
            to_move.extend(registration(path, distribution))
    if kept is None:
        problems.append(f"no installed hitlist {version} is an editable install of {root}")
    return Audit(len(distributions), kept, problems, to_move)


def main():
    version = hitlist.__version__
    print(f"python            -> {sys.executable}")
    print(f"import hitlist    -> {version}  ({hitlist.__file__})")
    result = audit(ROOT, version, hitlist.__file__, [str(ROOT), *sys.path])
    problems = list(result.problems)
    if result.kept is not None:
        print(f"metadata          -> {version}  editable install of {ROOT}  ({result.kept})")
    cli = shutil.which("hitlist")
    if cli is None:
        print("WARNING: no `hitlist` on PATH")
    else:
        reported = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"`hitlist` on PATH -> {cli}  ({reported})")
        if reported != f"hitlist {version}":
            problems.append(f"`hitlist` on PATH reports {reported!r}; another install shadows it")
    if not problems:
        return 0
    sys.stdout.flush()  # keep the report above the error when both are piped
    print(
        f"\nERROR: hitlist here is not one editable install of {ROOT} at {version} "
        f"({result.n_distributions} hitlist distributions visible; #553):",
        file=sys.stderr,
    )
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    if result.to_move:
        print(
            "Nothing was removed. Move these out of the environment (or delete them), "
            "then re-run ./develop.sh:",
            file=sys.stderr,
        )
        for path in result.to_move:
            print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
