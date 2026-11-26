# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data management for IEDB, CEDAR, HPA, viral proteomes, and other external datasets.

Tracks downloaded data files across sessions with metadata (source URL,
download date, file size, row count where applicable). Supports both
auto-fetchable datasets (UniProt proteomes, HPA downloads) and manually
downloaded datasets (IEDB/CEDAR behind terms-of-use).

Storage location: ``datacache``'s cache dir for the ``hitlist`` subdir
(``~/Library/Caches/hitlist`` on macOS, ``~/.cache/hitlist`` on Linux), or the
legacy ``~/.hitlist`` when an install already has data there.  Override with the
``HITLIST_DATA_DIR`` env var or :func:`set_data_dir`; ``hitlist data dirs``
prints what resolved and why.  See :func:`data_dir` for the full order (#291).

Python API::

    from hitlist.downloads import register, get_path, fetch, info, list_datasets
    from hitlist.downloads import download_to_file

    register("iedb", "/data/mhc_ligand_full.csv")
    path = get_path("iedb")
    fetch("hpv16")
    info("iedb")  # detailed metadata
    list_datasets()
    # progress bar + cache reporting + optional .zip/.gz decompression:
    download_to_file(url, dest, label="hpa", decompress=True)
    # version-pinned datasets with a provenance manifest (see tsarina's
    # reference data) -- consumers register their own defs + cache dir:
    reg = VersionedDatasetRegistry(MY_DATASETS, cache_dir=my_cache_dir)
    reg.ensure("hpa_rna_consensus", version="v23")

CLI::

    hitlist data register iedb /data/mhc_ligand_full.csv
    hitlist data fetch hpv16
    hitlist data list
    hitlist data dirs   # every location hitlist reads/writes, and why
    hitlist data info iedb
    hitlist data path iedb
    hitlist data refresh hpv16
    hitlist data remove iedb
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from datacache import FileValidationError, VersionedFileRegistry
from filelock import FileLock

from .errors import (
    DatasetError,
    DownloadError,
    HitlistError,
    InsufficientSpaceError,
    IntegrityError,
    OfflineError,
)

# Socket inactivity bound; mapping prefetch separately enforces a wall-clock
# deadline (#255/#402). Transfer, retry and publication belong to datacache.
_DOWNLOAD_SOCKET_TIMEOUT = 300.0


@contextmanager
def _typed_download_errors(url: str):
    """Re-raise datacache failures as :class:`DownloadError`/:class:`IntegrityError`.

    ``FileValidationError`` (size/SHA-256/corrupt or empty file) is an integrity
    failure; upstream ``InsufficientSpace`` / ``InsufficientSpaceError`` (once
    ``openvax/datacache#136`` lands) maps to :class:`InsufficientSpaceError`;
    other ``OSError`` (HTTP, timeout, disk) and ``ValueError`` (empty or
    oversized response) are download failures. The original exception stays on
    ``__cause__``; anything else is a bug and propagates untouched.
    """
    try:
        yield
    except FileValidationError as error:
        raise IntegrityError(f"Failed to download {url}: {error}") from error
    except (OSError, ValueError) as error:
        if type(error).__name__ in {"InsufficientSpace", "InsufficientSpaceError"}:
            raise InsufficientSpaceError(
                f"Failed to download {url}: {error}",
                path=getattr(error, "path", None),
                required_bytes=getattr(error, "required_bytes", None),
                available_bytes=getattr(error, "available_bytes", None),
            ) from error
        raise DownloadError(f"Failed to download {url}: {error}") from error


OPENVAX_OFFLINE_ENV_VAR = "OPENVAX_OFFLINE"
DATACACHE_OFFLINE_ENV_VAR = "DATACACHE_OFFLINE"
_OFFLINE_TRUTHY = frozenset({"1", "true", "yes", "on"})
_offline_override: bool | None = None


def set_offline(enabled: bool | None) -> None:
    """Enable, disable, or clear (``None``) process-wide offline mode (#960)."""
    global _offline_override
    _offline_override = None if enabled is None else bool(enabled)


def is_offline(*, offline: bool | None = None) -> bool:
    """Return True when network downloads are blocked via ``--offline``, ``OPENVAX_OFFLINE``, or ``DATACACHE_OFFLINE`` (#960)."""
    if offline is not None:
        return bool(offline)
    if _offline_override is not None:
        return _offline_override
    for var in (OPENVAX_OFFLINE_ENV_VAR, DATACACHE_OFFLINE_ENV_VAR):
        raw = os.environ.get(var, "").strip().lower()
        if raw in _OFFLINE_TRUTHY:
            return True
    return False


def _require_online(target: str, *, offline: bool | None = None) -> None:
    if is_offline(offline=offline):
        raise OfflineError(
            f"Offline mode is enabled ({OPENVAX_OFFLINE_ENV_VAR}/{DATACACHE_OFFLINE_ENV_VAR}/--offline); "
            f"refusing to download {target}"
        )


def download_to_file(
    url: str,
    dest: Path | str,
    *,
    label: str = "",
    verbose: bool = True,
    force: bool = False,
    decompress: bool = False,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    resume: bool = False,
    offline: bool | None = None,
) -> Path:
    """Fetch into the existing cache path using datacache's atomic downloader.

    Raw bytes are the default, including archives and HTML. With ``decompress``
    enabled, expand a literal .gz/.zip URL unless dest retains its archive suffix.
    Integrity expectations describe installed bytes and are checked on reuse too.
    ``resume`` requires a raw HTTP(S) download and expected_size; prefer a trusted
    SHA-256. Without one datacache requires strong ETags on all accepted responses.
    Cache hits never write provenance; new transfers record it for inspection.
    """
    import datacache

    dest = Path(dest)
    name = label or dest.name
    cached = dest.exists() and not force
    if not cached:
        _require_online(f"{name} from {url}", offline=offline)
    if verbose and not cached:
        print(f"  [{name}] downloading from {url}", file=sys.stderr)
    # Preserve hitlist's literal-URL opt-in, including query-bearing endpoints
    # whose query ends in an archive name (IEDB/CEDAR).
    source, target = url.lower(), dest.name.lower()
    expand = decompress and any(
        source.endswith(suffix) and not target.endswith(suffix) for suffix in (".gz", ".zip")
    )
    with _typed_download_errors(url):
        datacache.fetch_file(
            url,
            destination=dest,
            force=force,
            decompress=expand,
            raw=not expand,
            timeout=_DOWNLOAD_SOCKET_TIMEOUT,
            show_progress=verbose and sys.stderr.isatty(),
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            resume=resume,
            record_provenance=True,
        )
    if verbose and cached:
        print(f"  [{name}] already cached ({dest.stat().st_size:,} bytes)", file=sys.stderr)
    return dest


# ── Data directory ──────────────────────────────────────────────────────────
#
# hitlist keeps data in two places on purpose, and ``hitlist data dirs`` prints
# both (#291):
#
#   * :func:`data_dir` — everything this install *builds or downloads for
#     itself*: the observations/binding/bulk_proteomics/line_expression
#     parquets, ``manifest.json``, cached proteomes, the HGNC gene cache.
#   * :func:`data_asset_dir` — the read-only paper-derived CSVs mirrored to the
#     data-assets release and fetched through ``datacache`` (#303).
#
# On a fresh install the two are the same directory (datacache's cache dir).
# They diverge only when ``$HITLIST_DATA_DIR``/:func:`set_data_dir` moves the
# first one, or when this install still uses the legacy ``~/.hitlist``.
#
# Nothing in here prints or writes.  ``data_dir()`` is the body of
# ``observations_path()``, ``binding_path()``, ``mappings_path()``,
# ``_manifest_path()`` and ``genes._cache_path()``, and
# ``observations_cache_is_current()`` documents itself as doing "nothing else:
# no output, no writes" — so the legacy-location notice lives at the CLI entry
# point (:func:`legacy_data_dir_notice`), not here.

#: Name of the pre-#291 data directory under ``$HOME``.
_LEGACY_DATA_DIR_NAME = ".hitlist"

#: Every value :func:`data_dir_origin` can return, in priority order.  The CLI
#: renders one label per origin and a drift guard asserts it covers this tuple,
#: so adding a rule here fails loudly instead of printing a bare token.
DATA_DIR_ORIGINS = ("override", "env", "openvax_env", "legacy", "default")

#: Environment variable that relocates :func:`data_dir`.
DATA_DIR_ENV_VAR = "HITLIST_DATA_DIR"

#: Ecosystem-wide cache root honored as a fallback when HITLIST_DATA_DIR is unset (#960).
OPENVAX_DATA_CACHE_ENV_VAR = "OPENVAX_DATA_CACHE"

_override_data_dir: Path | None = None

#: Memoized ``{resolution inputs: (path, origin)}``.  ``data_dir()`` is called
#: thousands of times per process and rule 3 costs a directory scan, so the
#: answer is cached under the inputs that can change it — all of them
#: in-process lookups, no syscalls.  See :func:`reset_data_dir_cache`.
_data_dir_cache: dict[tuple[str | None, str, str, str], tuple[Path, str]] = {}

#: Cap on the memo above, which is keyed on environment values.
_DATA_DIR_CACHE_MAX_ENTRIES = 32


def reset_data_dir_cache() -> None:
    """Forget the memoized :func:`data_dir` resolution.

    :func:`set_data_dir` calls this.  Call it directly after *creating* the
    legacy directory's first artifact inside a long-lived process that has
    already resolved the path, which is otherwise the one way the cache can go
    stale (the cache key covers the override, the environment and ``$HOME``,
    but not the filesystem).
    """
    _data_dir_cache.clear()


def set_data_dir(path: str | Path) -> None:
    """Override the data directory for this session.

    Parameters
    ----------
    path
        Directory to use for all data storage. Created on first write, not
        here — resolving a path never touches the filesystem (see
        :func:`data_dir`).

    Example
    -------
    >>> from hitlist.downloads import set_data_dir
    >>> set_data_dir("/data/shared/hitlist")
    """
    global _override_data_dir
    _override_data_dir = Path(path)
    reset_data_dir_cache()


def legacy_data_dir() -> Path:
    """Return ``~/.hitlist``: where hitlist kept its data before #291.

    Resolved on every call rather than at import so that pointing ``$HOME``
    elsewhere (tests, containers) is respected.
    """
    return Path.home() / _LEGACY_DATA_DIR_NAME


def _holds_a_file(path: Path | str) -> bool:
    """True if *path* is a directory containing at least one regular file."""
    try:
        with os.scandir(path) as entries:
            return any(entry.is_file() for entry in entries)
    except OSError:
        return False


def legacy_data_dir_is_populated() -> bool:
    """True when ``~/.hitlist`` holds data, rather than just empty folders.

    **Populated** means the directory exists and holds either a regular file at
    the top level, or a subdirectory that holds a regular file.

    It is deliberately a *structural* test and not a list of known artifact
    names.  A closed list drifts: ``genes._cache_path()`` writes real HGNC
    lookups into ``<data dir>/gene_cache/hgnc_lookups.json`` without ever
    touching ``manifest.json``, so a user who only called
    :func:`hitlist.genes.resolve_hgnc_symbol` would have read as "empty" and
    been silently relocated — and the next artifact added to the tree would
    have repeated it.

    It is equally deliberately not "the directory exists" or "the directory is
    non-empty": ``data_dir()`` used to ``mkdir`` its result on *every* call,
    and ``_proteomes_dir()`` / ``genes._cache_path()`` still create their
    subdirectories eagerly, so plenty of installs have a ``~/.hitlist``
    containing nothing but empty folders.  Those fall through to the new
    default instead of pinning themselves to the legacy location forever.

    One level of nesting is enough for every directory hitlist creates
    (``proteomes/``, ``gene_cache/``, ``proteome_index_cache/``).
    """
    d = legacy_data_dir()
    try:
        with os.scandir(d) as it:
            entries = list(it)
    except OSError:
        return False
    # Files first: on a real corpus the answer is one dirent lookup away, and
    # it avoids descending into a 7k-entry index cache to learn what
    # ``observations.parquet`` already says.
    return any(e.is_file() for e in entries) or any(
        e.is_dir() and _holds_a_file(e.path) for e in entries
    )


def _resolve_openvax_cache_dir() -> Path | None:
    """``$OPENVAX_DATA_CACHE/hitlist`` when set, else ``None`` (#960)."""
    raw = os.environ.get(OPENVAX_DATA_CACHE_ENV_VAR, "").strip()
    return Path(raw).expanduser() / "hitlist" if raw else None


def data_asset_dir() -> Path:
    """Directory :func:`fetch_data_asset` caches mirrored data assets in (#303, #960).

    Honors ``$OPENVAX_DATA_CACHE/hitlist`` when set so HPC users redirecting the
    shared OpenVax cache do not spill mirrored CSVs into ``~/.cache/hitlist``;
    otherwise delegates to ``datacache.get_data_dir("hitlist")``.
    """
    openvax = _resolve_openvax_cache_dir()
    if openvax is not None:
        return openvax
    from datacache import get_data_dir

    return Path(get_data_dir("hitlist"))


def default_data_dir() -> Path:
    """The data directory a fresh install resolves to.

    ``~/Library/Caches/hitlist`` on macOS, ``~/.cache/hitlist`` on Linux — the
    convention pyensembl and the rest of the openvax ecosystem already use
    (#291).

    Identical to :func:`data_asset_dir` by construction, because datacache
    resolves both the same way: on a default install an install's built indexes
    and its mirrored assets live in one directory, which is the whole point.
    The two names are kept apart because they answer different questions — "where
    do my indexes go?" is moved by ``$HITLIST_DATA_DIR`` and "where does
    datacache put fetched assets?" is not.
    """
    return data_asset_dir()


def _resolve_env_data_dir() -> Path | None:
    """``$HITLIST_DATA_DIR`` as a path, or ``None`` when it does not apply.

    Whitespace is stripped and ``~`` expanded.  An unset, empty or
    whitespace-only value means *unset*: before #291 an empty value resolved to
    ``Path("")`` — the process's current working directory — which scattered
    indexes wherever the shell happened to be and was never anything but a
    misconfiguration.
    """
    raw = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    return Path(raw).expanduser() if raw else None


def resolve_data_dir() -> tuple[Path, str]:
    """Return ``(directory, origin)`` without touching the filesystem.

    ``origin`` is one of :data:`DATA_DIR_ORIGINS`.  Callers that want both —
    the CLI prints the path *and* why it resolved there — use this rather than
    calling :func:`data_dir` and :func:`data_dir_origin` in turn, which would
    run the resolution twice and could disagree with itself.
    """
    key = (
        str(_override_data_dir) if _override_data_dir is not None else None,
        os.environ.get(DATA_DIR_ENV_VAR, ""),
        os.environ.get(OPENVAX_DATA_CACHE_ENV_VAR, ""),
        os.path.expanduser("~"),
    )
    cached = _data_dir_cache.get(key)
    if cached is not None:
        return cached
    if _override_data_dir is not None:
        resolved = (Path(_override_data_dir), "override")
    else:
        env = _resolve_env_data_dir()
        if env is not None:
            resolved = (env, "env")
        elif (openvax := _resolve_openvax_cache_dir()) is not None:
            resolved = (openvax, "openvax_env")
        elif legacy_data_dir_is_populated():
            resolved = (legacy_data_dir(), "legacy")
        else:
            resolved = (default_data_dir(), "default")
    if len(_data_dir_cache) >= _DATA_DIR_CACHE_MAX_ENTRIES:
        # Keys come from the environment, so a process that keeps changing
        # $HITLIST_DATA_DIR (a test session does) would otherwise grow this
        # forever. Only the newest key is ever read twice in a row.
        _data_dir_cache.clear()
    _data_dir_cache[key] = resolved
    return resolved


def data_dir() -> Path:
    """Return the hitlist data directory. Resolution only — silent, creates nothing.

    Priority:

    1. :func:`set_data_dir` override
    2. ``$HITLIST_DATA_DIR``, used as the directory itself, as it always has
       been — stripped and ``~``-expanded; empty means unset
    3. an existing, *populated* ``~/.hitlist`` — the legacy location from
       before #291.  It stays fully supported, so an install that already has a
       corpus there keeps using it; see :func:`legacy_data_dir_is_populated`.
    4. ``datacache.get_data_dir(subdir="hitlist")`` — the openvax-ecosystem
       cache dir a fresh install uses (#291).

    Resolving stays side-effect free.  ``import hitlist`` may not touch the
    filesystem (#579); ``observations_cache_is_current()`` promises "no output,
    no writes" and is one of this function's callers; and a ``mkdir`` here would
    make an empty ``~/.hitlist`` look populated to rule 3 forever after.  The
    write path creates the directory (``parquet_io.atomic_write_parquet``,
    ``_save_manifest``, ``download_to_file``), and the CLI — not the library —
    reports rule 3 via :func:`legacy_data_dir_notice`.
    """
    return resolve_data_dir()[0]


def data_dir_origin() -> str:
    """Which of :func:`data_dir`'s rules chose the current directory.

    One of :data:`DATA_DIR_ORIGINS` — what ``hitlist data dirs`` reports so a
    user can see *why* their data is where it is.
    """
    return resolve_data_dir()[1]


def legacy_data_dir_notice() -> str | None:
    """The "you are on the legacy location" message, or ``None``.

    Returns text only when :func:`data_dir` resolved by rule 3.  It is a
    *return value*, not a print, so that the library stays silent and only the
    CLI entry point decides to show it (once, in ``main()``).
    """
    path, origin = resolve_data_dir()
    if origin != "legacy":
        return None
    return (
        f"[hitlist] Using the legacy data directory {path} — it holds this "
        f"install's built indexes and stays fully supported; nothing has to move.\n"
        f"[hitlist] New installs default to {default_data_dir()}. To move this "
        f"one, set {DATA_DIR_ENV_VAR} to the new location and copy the directory's "
        f"*entire* contents there — moving only the parquets leaves "
        f"manifest.json behind, which keeps rule 3 pointing at a now-empty "
        f"corpus. `hitlist data dirs` shows every location."
    )


def _manifest_path() -> Path:
    return data_dir() / "manifest.json"


def load_manifest() -> dict:
    p = _manifest_path()
    if not p.exists():
        return {"datasets": {}}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError, OSError):
        # The manifest is a regenerable cache.  Tolerate a transient empty/partial
        # read (another build worker mid-write) or a pre-existing corrupt file
        # rather than crashing the build — missing entries just get re-fetched.
        # See #331.
        return {"datasets": {}}


def _save_manifest(manifest: dict) -> None:
    # Atomic write: serialize to a unique temp file in the same dir, then
    # os.replace() into place.  os.replace is atomic on POSIX/Windows, so a
    # concurrent load_manifest() reader always sees a COMPLETE file (old or new),
    # never a half-written/empty one — fixes the parallel-build race (#331).
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(manifest, indent=2, default=str) + "\n")
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def manifest_transaction() -> Iterator[dict]:
    """Locked read-modify-write of ``manifest.json`` (#883).

    Holds an exclusive ``filelock.FileLock`` on ``.manifest.json.lock`` (next to
    the manifest) while it loads the manifest, yields the dict for the caller to
    mutate in place, and atomically saves it on a clean exit.  If the body
    raises, nothing is saved.  Concurrent processes/threads therefore never drop
    each other's entries, which the atomic ``os.replace`` alone (#331) cannot
    prevent.

    Keep the body short: do downloads and other slow work *before* entering the
    transaction.  The lock is not re-entrant across separate calls, so never nest
    one ``manifest_transaction()`` inside another (``register`` included).
    For pure reads use :func:`load_manifest`.
    """
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(p.with_name("." + p.name + ".lock"))):
        manifest = load_manifest()
        yield manifest
        _save_manifest(manifest)


# ── Known datasets ──────────────────────────────────────────────────────────

# ``min_size_bytes`` is a floor well below the real artifact, there to reject the
# empty/error bodies upstreams answer with HTTP 200 (#350, #882), not to pin a size.
FETCHABLE_DATASETS: dict[str, dict[str, str | int]] = {
    # Viral proteomes (auto-downloadable from UniProt REST API; UPIDs and NCBI
    # taxon IDs verified against rest.uniprot.org/proteomes/{upid}, #977)
    "hpv16": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000009251&format=fasta",
        "filename": "hpv16.fasta",
        "proteome_id": "UP000009251",
        "expected_taxon_id": 333760,
        "description": "HPV-16 proteome (UniProt UP000009251)",
        "usage": "Peptide generation for cervical/oropharyngeal cancer viral targets",
        "min_size_bytes": 500,
    },
    "hpv18": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000009109&format=fasta",
        "filename": "hpv18.fasta",
        "proteome_id": "UP000009109",
        "expected_taxon_id": 333761,
        "description": "HPV-18 proteome (UniProt UP000009109)",
        "usage": "Peptide generation for cervical cancer viral targets",
        "min_size_bytes": 500,
    },
    "ebv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000153037&format=fasta",
        "filename": "ebv.fasta",
        "proteome_id": "UP000153037",
        "expected_taxon_id": 10377,
        "description": "EBV/HHV-4 proteome (UniProt UP000153037)",
        "usage": "Peptide generation for Burkitt lymphoma, NPC, Hodgkin lymphoma",
        "min_size_bytes": 500,
    },
    "htlv1": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000001061&format=fasta",
        "filename": "htlv1.fasta",
        "proteome_id": "UP000001061",
        "expected_taxon_id": 11927,
        "description": "HTLV-1 proteome (UniProt UP000001061)",
        "usage": "Peptide generation for adult T-cell leukemia/lymphoma",
        "min_size_bytes": 500,
    },
    "hbv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000007930&format=fasta",
        "filename": "hbv.fasta",
        "proteome_id": "UP000007930",
        "expected_taxon_id": 490133,
        "description": "HBV proteome (UniProt UP000007930)",
        "usage": "Peptide generation for hepatocellular carcinoma",
        "min_size_bytes": 500,
    },
    "hcv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000518&format=fasta",
        "filename": "hcv.fasta",
        "proteome_id": "UP000000518",
        "expected_taxon_id": 63746,
        "description": "HCV proteome (UniProt UP000000518)",
        "usage": "Peptide generation for hepatocellular carcinoma, B-cell lymphoma",
        "min_size_bytes": 500,
    },
    "kshv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000942&format=fasta",
        "filename": "kshv.fasta",
        "proteome_id": "UP000000942",
        "expected_taxon_id": 868565,
        "description": "KSHV/HHV-8 proteome (UniProt UP000000942)",
        "usage": "Peptide generation for Kaposi sarcoma",
        "min_size_bytes": 500,
    },
    "mcpyv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000170366&format=fasta",
        "filename": "mcpyv.fasta",
        "proteome_id": "UP000170366",
        "expected_taxon_id": 493803,
        "description": "MCPyV proteome (UniProt UP000170366)",
        "usage": "Peptide generation for Merkel cell carcinoma",
        "min_size_bytes": 500,
    },
    "hiv1": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000002241&format=fasta",
        "filename": "hiv1.fasta",
        "proteome_id": "UP000002241",
        "expected_taxon_id": 11706,
        "description": "HIV-1 proteome (UniProt UP000002241)",
        "usage": "Peptide generation for Kaposi sarcoma, lymphoma (indirect)",
        "min_size_bytes": 500,
    },
    # IEDB / CEDAR MHC-ligand exports. The downloader.php endpoints serve the
    # zip directly (no enforced login), so fetch() streams + unzips them like
    # any other fetchable dataset. The `terms` URL drives a usage/citation
    # notice on fetch, since these are terms-of-use-governed sources.
    "iedb": {
        "url": "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip",
        "filename": "mhc_ligand_full.csv",
        "description": "IEDB MHC ligand full export",
        "usage": "Mass spec evidence for peptide-MHC presentation.",
        "terms": "https://www.iedb.org/",
        "min_size_bytes": 50_000_000,
        # Approximate disk needed once extracted; a full-disk ENOSPC midway (#884) is
        # worse than refusing to start.  Optional: omit when unknown.
        "size_bytes": 7_800_000_000,
    },
    "cedar": {
        # CEDAR serves the export under the same member name IEDB uses,
        # on the CEDAR host.  ``doc/cedar_mhc_ligand_full.zip`` is not a
        # filename CEDAR serves, and ``downloader.php`` answers HTTP 200
        # with a zero-length body for any unrecognized name rather than
        # 404 — so the old URL failed identically to a server outage and
        # the retry loop could not tell the difference (#350).
        "url": "https://cedar.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip",
        "filename": "cedar-mhc-ligand-full.csv",
        "description": "CEDAR MHC ligand full export",
        "usage": "Additional mass spec evidence (companion to IEDB).",
        "terms": "https://cedar.iedb.org/",
        "min_size_bytes": 10_000_000,
    },
    "netmhcstabpan": {
        "url": "https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt",
        "filename": "netmhcstabpan-1.0-Stability.txt",
        "description": "NetMHCstabpan 1.0 pMHC-I half-life archive (PMID 27402703)",
        "usage": "Source-qualified pMHC-I dissociation half-lives in hours (#875).",
        "terms": "https://pubmed.ncbi.nlm.nih.gov/27402703/",
        "min_size_bytes": 799_880,
        "expected_size": 799_880,
        "expected_sha256": "5f2c6e82d198705a66974b4e42a0ed1412124c4b1e8f7033c7cd43bbdac07f67",
    },
}

# Immutable files from doi:10.25452/figshare.plus.27993248.v1 (CC BY 4.0).
_DEPMAP_FILES = {
    "depmap_rna": (51065489, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"),
    "depmap_rna_transcript": (51065534, "OmicsExpressionTranscriptsTPMLogp1Profile.csv"),
    "depmap_models": (51065297, "Model.csv"),
    "depmap_profiles": (51065723, "OmicsProfiles.csv"),
    "depmap_default_profiles": (51065339, "OmicsDefaultModelProfiles.csv"),
}
FETCHABLE_DATASETS.update(
    {
        key: {
            "url": f"https://ndownloader.figshare.com/files/{file_id}",
            "filename": filename,
            "description": f"DepMap 24Q4 {filename}",
            "usage": "Optional line RNA expression; fetch depmap downloads companions and builds the index.",
            "terms": "https://doi.org/10.25452/figshare.plus.27993248.v1",
            "min_size_bytes": 100_000,
        }
        for key, (file_id, filename) in _DEPMAP_FILES.items()
    }
)

MANUAL_DATASETS: dict[str, dict[str, str]] = {
    "hpa_bulk": {
        "download_url": "https://www.proteinatlas.org/download/proteinatlas.tsv.zip",
        "description": "HPA proteinatlas.tsv bulk summary",
        "expected_filename": "proteinatlas.tsv",
        "usage": "RNA tissue specificity, distribution, nTPM per gene for CTA restriction analysis.",
    },
    "hpa_rna": {
        "download_url": "https://www.proteinatlas.org/download/rna_tissue_consensus.tsv.zip",
        "description": "HPA RNA tissue consensus (50 tissues)",
        "expected_filename": "rna_tissue_consensus.tsv",
        "usage": "Per-tissue nTPM values for deflated reproductive fraction computation.",
    },
    "hpa_protein": {
        "download_url": "https://www.proteinatlas.org/download/normal_tissue.tsv.zip",
        "description": "HPA normal tissue IHC (63 tissues)",
        "expected_filename": "normal_tissue.tsv",
        "usage": "Protein-level tissue expression for CTA restriction analysis.",
    },
    "tcrvdb": {
        "download_url": "https://tcrvdb.nkiai.nl",
        "description": "TCRvdb functionally validated paired TCR-pMHC database (access by request; DOI 10.1016/j.immuni.2026.09.007)",
        "expected_filename": "2026-10-01_TCRvdb_ttd_gil_ylq_nlv_glc.csv",
        "usage": "Paired alpha/beta TCR functional specificity evidence via hitlist.load_tcrvdb_functional_evidence() (#722).",
    },
    "miyamae_2026_dla88": {
        "download_url": "https://doi.org/10.64898/2026.10.06.756904",
        "description": (
            "Miyamae 2026 ten-allele canine DLA-88 ligand atlas "
            "(bioRxiv license=cc_no; local acquisition only, #677)"
        ),
        "expected_filename": "miyamae_2026_dla88_ligands.tsv",
        "usage": "Offline access-aware loading via hitlist.load_miyamae_2026_dla88_ligands().",
    },
    "lang_2026_splenic_dla_dr": {
        "download_url": "https://academic.oup.com/jimmunol/article/215/9/vkag243/8812665",
        "description": (
            "Lang 2026 splenic DLA-DR MS peptide table "
            "(PMID 42752862; pending authoritative MS deposit, #676)"
        ),
        "expected_filename": "lang_2026_splenic_dla_dr_peptides.tsv",
        "usage": "Offline access-aware loading via hitlist.load_lang_2026_splenic_dla_dr_ligands().",
    },
    "peptidomicsdb_metadata": {
        "download_url": "http://www.inbirg.com/ppd/download/",
        "description": (
            "PeptidomicsDB project/sample/dataset metadata workbooks "
            "(10.1093/nar/gkag1032; audited before native-peptide admission, #811)"
        ),
        "expected_filename": "sample-download.xlsx",
        "usage": "Capture-vs-compartment qualification via hitlist.qualify_peptidomicsdb_samples().",
    },
}


# ── Species / viral proteome registry ───────────────────────────────────────
#
# Maps canonical species names (from normalize_species()) to reference
# proteomes.  For "ensembl" species, callers use pyensembl directly
# (ProteomeIndex.from_ensembl).  For "uniprot" species, we download the
# reference proteome FASTA from UniProt's REST API and cache it locally.
#
# Proteome IDs: https://www.uniprot.org/proteomes/
#
# Keys must match the output of ``curation.normalize_species()``.

_UNIPROT_QUERY_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?query={query}&format=fasta&compressed=false"
)
_UNIPROT_PROTEOME_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=proteome:{proteome_id}&format=fasta&compressed=false"
)
_UNIPROT_PROTEOME_META_URL = "https://rest.uniprot.org/proteomes/{proteome_id}?format=json"


SPECIES_PROTEOMES: dict[str, dict[str, str | int]] = {
    # Ensembl-supported (pyensembl)
    "Homo sapiens": {"kind": "ensembl", "release": 112, "species": "human"},
    "Mus musculus": {"kind": "ensembl", "release": 112, "species": "mouse"},
    "Rattus norvegicus": {"kind": "ensembl", "release": 112, "species": "rat"},
    # UniProt reference proteomes (auto-downloaded; UPIDs and NCBI taxon IDs
    # verified against rest.uniprot.org/proteomes/{upid}, #977)
    "Sarcophilus harrisii": {
        "kind": "uniprot",
        "proteome_id": "UP000007648",
        "expected_taxon_id": 9305,
    },
    "Canis lupus": {
        "kind": "uniprot",
        "proteome_id": "UP000805418",
        "expected_taxon_id": 9615,
    },
    "Bos taurus": {
        "kind": "uniprot",
        "proteome_id": "UP000009136",
        "expected_taxon_id": 9913,
    },
    "Gallus gallus": {
        "kind": "uniprot",
        "proteome_id": "UP000000539",
        "expected_taxon_id": 9031,
    },
    "Sus scrofa": {
        "kind": "uniprot",
        "proteome_id": "UP000008227",
        "expected_taxon_id": 9823,
    },
    "Macaca mulatta": {
        "kind": "uniprot",
        "proteome_id": "UP000006718",
        "expected_taxon_id": 9544,
    },
    "Equus caballus": {
        "kind": "uniprot",
        "proteome_id": "UP000002281",
        "expected_taxon_id": 9796,
    },
    "Pan troglodytes": {
        "kind": "uniprot",
        "proteome_id": "UP000002277",
        "expected_taxon_id": 9598,
    },
    # Trichosurus vulpecula (brushtail possum, OX=9337) is intentionally NOT
    # registered: UniProt Proteomes has no proteome for Trichosurus (previously
    # UP000504604 erroneously resolved to Sesamum indicum / sesame seed, #977).
    # IEDB genus-abbreviated names — treat as the type species
    "Sus sp.": {
        "kind": "uniprot",
        "proteome_id": "UP000008227",
        "expected_taxon_id": 9823,
    },  # → Sus scrofa
    "Canis sp.": {
        "kind": "uniprot",
        "proteome_id": "UP000805418",
        "expected_taxon_id": 9615,
    },  # → Canis lupus
    "Bos sp.": {
        "kind": "uniprot",
        "proteome_id": "UP000009136",
        "expected_taxon_id": 9913,
    },  # → Bos taurus
    "Rattus sp.": {"kind": "ensembl", "release": 112, "species": "rat"},
    # Parasites / plants / other with curated UPIDs
    "Theileria parva": {
        "kind": "uniprot",
        "proteome_id": "UP000001949",
        "expected_taxon_id": 5875,
    },
    "Ascaris suum": {
        "kind": "uniprot",
        "proteome_id": "UP001319243",
        "expected_taxon_id": 6253,
    },
    "Ascaris lumbricoides": {
        "kind": "uniprot",
        "proteome_id": "UP000036681",
        "expected_taxon_id": 6252,
    },
    # Bacteria — canonical reference strains.
    "Mycobacterium tuberculosis": {
        "kind": "uniprot",
        "proteome_id": "UP000001584",
        "expected_taxon_id": 83332,
    },  # H37Rv
    "Mycobacterium tuberculosis H37Rv": {
        "kind": "uniprot",
        "proteome_id": "UP000001584",
        "expected_taxon_id": 83332,
    },
    # Apicomplexan parasites — 3D7 is the canonical P. falciparum
    # reference used by virtually all immunology and antimalarial
    # epitope work.
    "Plasmodium falciparum": {
        "kind": "uniprot",
        "proteome_id": "UP000001450",
        "expected_taxon_id": 36329,
    },  # 3D7
}


# Viral proteomes keyed by the IEDB ``source_organism`` string we observe.
# For matching, we lowercase both sides and check substring inclusion of the
# registry key.  This tolerates IEDB variations (e.g. "Epstein-Barr virus
# (strain B95-8)" matches "epstein-barr virus").
VIRAL_PROTEOMES: dict[str, dict[str, str | int]] = {
    # SARS coronaviruses — "sars-cov" must come before "sars-cov-2" check below
    # but we sort by key length at lookup time to match most-specific first
    "severe acute respiratory syndrome coronavirus 2": {
        "proteome_id": "UP000464024",
        "key": "sars-cov-2",
        "expected_taxon_id": 2697049,
    },
    "sars-cov2": {
        "proteome_id": "UP000464024",
        "key": "sars-cov-2",
        "expected_taxon_id": 2697049,
    },
    "sars-cov-2": {
        "proteome_id": "UP000464024",
        "key": "sars-cov-2",
        "expected_taxon_id": 2697049,
    },
    "severe acute respiratory syndrome coronavirus": {
        "proteome_id": "UP000000354",
        "key": "sars-cov-1",
        "expected_taxon_id": 694009,
    },
    "sars-cov1": {
        "proteome_id": "UP000000354",
        "key": "sars-cov-1",
        "expected_taxon_id": 694009,
    },
    "sars-cov-1": {
        "proteome_id": "UP000000354",
        "key": "sars-cov-1",
        "expected_taxon_id": 694009,
    },
    "sars coronavirus": {
        "proteome_id": "UP000000354",
        "key": "sars-cov-1",
        "expected_taxon_id": 694009,
    },
    # Retroviruses / Herpesviruses
    "human immunodeficiency virus 1": {
        "proteome_id": "UP000002241",
        "key": "hiv1",
        "expected_taxon_id": 11706,
    },
    "hiv-1": {
        "proteome_id": "UP000002241",
        "key": "hiv1",
        "expected_taxon_id": 11706,
    },
    "human t-lymphotropic virus 1": {
        "proteome_id": "UP000001061",
        "key": "htlv1",
        "expected_taxon_id": 11927,
    },
    "human t-cell leukemia virus 1": {
        "proteome_id": "UP000001061",
        "key": "htlv1",
        "expected_taxon_id": 11927,
    },
    "epstein-barr virus": {
        "proteome_id": "UP000153037",
        "key": "ebv",
        "expected_taxon_id": 10377,
    },
    "human gammaherpesvirus 4": {
        "proteome_id": "UP000153037",
        "key": "ebv",
        "expected_taxon_id": 10377,
    },
    "human herpesvirus 4": {
        "proteome_id": "UP000153037",
        "key": "ebv",
        "expected_taxon_id": 10377,
    },
    "human betaherpesvirus 5": {
        "proteome_id": "UP000000938",
        "key": "hcmv",
        "expected_taxon_id": 295027,
    },
    "human cytomegalovirus": {
        "proteome_id": "UP000000938",
        "key": "hcmv",
        "expected_taxon_id": 295027,
    },
    "human herpesvirus 5": {
        "proteome_id": "UP000000938",
        "key": "hcmv",
        "expected_taxon_id": 295027,
    },
    "human gammaherpesvirus 8": {
        "proteome_id": "UP000000942",
        "key": "kshv",
        "expected_taxon_id": 868565,
    },
    "human herpesvirus 8": {
        "proteome_id": "UP000000942",
        "key": "kshv",
        "expected_taxon_id": 868565,
    },
    "kaposi": {
        "proteome_id": "UP000000942",
        "key": "kshv",
        "expected_taxon_id": 868565,
    },
    "human alphaherpesvirus 1": {
        "proteome_id": "UP000009294",
        "key": "hsv-1",
        "expected_taxon_id": 10299,
    },
    "herpes simplex virus type 1": {
        "proteome_id": "UP000009294",
        "key": "hsv-1",
        "expected_taxon_id": 10299,
    },
    "human herpesvirus 1": {
        "proteome_id": "UP000009294",
        "key": "hsv-1",
        "expected_taxon_id": 10299,
    },
    "human alphaherpesvirus 2": {
        "proteome_id": "UP000001874",
        "key": "hsv-2",
        "expected_taxon_id": 10315,
    },
    "herpes simplex virus type 2": {
        "proteome_id": "UP000001874",
        "key": "hsv-2",
        "expected_taxon_id": 10315,
    },
    "human herpesvirus 2": {
        "proteome_id": "UP000001874",
        "key": "hsv-2",
        "expected_taxon_id": 10315,
    },
    "human betaherpesvirus 6b": {
        "proteome_id": "UP000006930",
        "key": "hhv-6b",
        "expected_taxon_id": 36351,
    },
    "human herpesvirus 6b": {
        "proteome_id": "UP000006930",
        "key": "hhv-6b",
        "expected_taxon_id": 36351,
    },
    "murid betaherpesvirus 1": {
        "proteome_id": "UP000008774",
        "key": "mcmv",
        "expected_taxon_id": 10367,
    },
    "murid herpesvirus 1": {
        "proteome_id": "UP000008774",
        "key": "mcmv",
        "expected_taxon_id": 10367,
    },
    "murine cytomegalovirus": {
        "proteome_id": "UP000008774",
        "key": "mcmv",
        "expected_taxon_id": 10367,
    },
    # Hepatitis
    "hepatitis b virus": {
        "proteome_id": "UP000007930",
        "key": "hbv",
        "expected_taxon_id": 490133,
    },
    "hepatitis c virus": {
        "proteome_id": "UP000000518",
        "key": "hcv",
        "expected_taxon_id": 63746,
    },
    "hepacivirus hominis": {
        "proteome_id": "UP000000518",
        "key": "hcv",
        "expected_taxon_id": 63746,
    },
    # Papillomaviruses
    "human papillomavirus type 16": {
        "proteome_id": "UP000009251",
        "key": "hpv16",
        "expected_taxon_id": 333760,
    },
    "human papillomavirus 16": {
        "proteome_id": "UP000009251",
        "key": "hpv16",
        "expected_taxon_id": 333760,
    },
    "human papillomavirus type 18": {
        "proteome_id": "UP000009109",
        "key": "hpv18",
        "expected_taxon_id": 333761,
    },
    "human papillomavirus 18": {
        "proteome_id": "UP000009109",
        "key": "hpv18",
        "expected_taxon_id": 333761,
    },
    # Influenza
    "influenza a virus": {
        "proteome_id": "UP000009255",
        "key": "influenza-a",
        "expected_taxon_id": 211044,
    },
    "influenza b virus": {
        "proteome_id": "UP000008158",
        "key": "influenza-b",
        "expected_taxon_id": 518987,
    },
    # Poxviruses
    "vaccinia virus": {
        "proteome_id": "UP000000344",
        "key": "vaccinia",
        "expected_taxon_id": 10254,
    },
    "orf virus": {
        "proteome_id": "UP000000870",
        "key": "orf",
        "expected_taxon_id": 647330,
    },
    # Animal pathogens
    "canine distemper virus": {
        "proteome_id": "UP000007634",
        "key": "cdv",
        "expected_taxon_id": 11233,
    },
    "african swine fever virus": {
        "proteome_id": "UP000000624",
        "key": "asfv",
        "expected_taxon_id": 10498,
    },
    "porcine reproductive and respiratory syndrome virus": {
        "proteome_id": "UP000006687",
        "key": "prrsv",
        "expected_taxon_id": 11049,
    },
    "wobbly possum disease virus": {
        "proteome_id": "UP000203107",
        "key": "wpdv",
        "expected_taxon_id": 1118369,
    },
    "peste-des-petits-ruminants virus": {
        "proteome_id": "UP000100083",
        "key": "pprv",
        "expected_taxon_id": 2593991,
    },
    # Respiratory / other human viruses
    "human respiratory syncytial virus": {
        "proteome_id": "UP000002472",
        "key": "rsv",
        "expected_taxon_id": 79692,
    },
    "human orthopneumovirus": {
        "proteome_id": "UP000002472",
        "key": "rsv",
        "expected_taxon_id": 79692,
    },
    "human metapneumovirus": {
        "proteome_id": "UP000001398",
        "key": "hmpv",
        "expected_taxon_id": 694067,
    },
    "zika virus": {
        "proteome_id": "UP000054557",
        "key": "zika",
        "expected_taxon_id": 64320,
    },
    "rotavirus a": {
        "proteome_id": "UP000001119",
        "key": "rotavirus-a",
        "expected_taxon_id": 450149,
    },
    "lymphocytic choriomeningitis virus": {
        "proteome_id": "UP000002474",
        "key": "lcmv",
        "expected_taxon_id": 11624,
    },
    # Polyomaviruses
    "betapolyomavirus hominis": {
        "proteome_id": "UP000008475",
        "key": "bkv",
        "expected_taxon_id": 1891762,
    },
    "bk polyomavirus": {
        "proteome_id": "UP000008475",
        "key": "bkv",
        "expected_taxon_id": 1891762,
    },
    "merkel cell polyomavirus": {
        "proteome_id": "UP000170366",
        "key": "mcpyv",
        "expected_taxon_id": 493803,
    },
    "alphapolyomavirus muris": {
        "proteome_id": "UP000116380",
        "key": "mpyv",
        "expected_taxon_id": 179241,
    },
    # Adeno-associated viruses (gene-therapy capsid context — #213).
    # Each AAV serotype has a distinct UniProt reference proteome.
    # The "virus - 6" key handles IEDB's hyphen-space-hyphen variant
    # ("Adeno-associated virus - 6") which the substring matcher would
    # otherwise miss against the canonical "virus 6" key. AAV1 and
    # AAV9 are intentionally NOT registered: UniProt has no clean
    # serotype-9 reference proteome (a UniProt-side gap), and its
    # AAV1 hit (UP000232962) resolves to "California sea lion AAV1"
    # which is the wrong organism for the human gene-therapy /
    # immunopeptidome context the IEDB rows come from.
    "adeno-associated virus 2": {
        "proteome_id": "UP000180764",
        "key": "aav-2",
        "expected_taxon_id": 648242,
    },
    # AAV-6 (UP000119472) and AAV-8 (UP000201958) are Non-Reference proteomes
    # in UniProt, so ``query=proteome:UP...`` returns 0 bytes on UniProtKB
    # stream, whereas ``query=organism_id:<taxon>`` streams their exact Rep +
    # VP1 capsid FASTA entries (#977).
    "adeno-associated virus 6": {
        "proteome_id": "UP000119472",
        "key": "aav-6",
        "expected_taxon_id": 68558,
        "uniprot_query": "organism_id:68558",
    },
    "adeno-associated virus - 6": {
        "proteome_id": "UP000119472",
        "key": "aav-6",
        "expected_taxon_id": 68558,
        "uniprot_query": "organism_id:68558",
    },
    "adeno-associated virus 8": {
        "proteome_id": "UP000201958",
        "key": "aav-8",
        "expected_taxon_id": 202813,
        "uniprot_query": "organism_id:202813",
    },
}


_UNIPROT_OX_RE = re.compile(r"\bOX=(\d+)\b")
_UNIPROT_OS_RE = re.compile(r"\bOS=(.+?)(?:\s+OX=|\s+GN=|\s+PE=|\s+SV=|$)")
# Historical UPIDs that were replaced in #977 because they pointed to the wrong
# organism or a deprecated Non-Reference proteome returning 0 bytes.
_REPLACED_WRONG_UPIDS: frozenset[str] = frozenset(
    {
        "UP000006729",  # Populus trichocarpa (was hpv16)
        "UP000006728",  # Saccharopolyspora erythraea (was hpv18)
        "UP000002063",  # Methanocaldococcus vulcanius (was htlv1)
        "UP000126453",  # Influenza A virus (was hbv)
        "UP000009113",  # HPV type 26 (was kshv)
        "UP000116695",  # Influenza A virus (was mcpyv)
        "UP000117312",  # Influenza A virus (was canine distemper virus)
        "UP000006706",  # Aspergillus niger (was PRRSV)
        "UP000147130",  # Influenza A virus (was wobbly possum disease virus)
        "UP000007212",  # IHNV (was alphapolyomavirus muris)
        "UP000504604",  # Sesamum indicum (was Trichosurus vulpecula)
        "UP000002254",  # Deprecated Non-Reference Canis lupus proteome (0-byte stream)
    }
)


def _expected_taxon_for_upid(upid: str) -> int | None:
    """Return the curated NCBI taxon ID for a UniProt proteome ID, if registered."""
    for entry in SPECIES_PROTEOMES.values():
        if entry.get("proteome_id") == upid and entry.get("expected_taxon_id") is not None:
            return int(entry["expected_taxon_id"])
    for entry in VIRAL_PROTEOMES.values():
        if entry.get("proteome_id") == upid and entry.get("expected_taxon_id") is not None:
            return int(entry["expected_taxon_id"])
    for ds in FETCHABLE_DATASETS.values():
        if ds.get("proteome_id") == upid and ds.get("expected_taxon_id") is not None:
            return int(ds["expected_taxon_id"])
    return None


def validate_uniprot_fasta_headers(
    path: str | Path,
    *,
    expected_taxon_id: int | Iterable[int] | None = None,
    expected_organism: str | None = None,
    require_taxon_header: bool = False,
    max_headers: int = 200,
) -> dict[str, Any]:
    """Validate that a UniProt FASTA file contains FASTA headers matching the expected taxon (#977).

    Inspects up to ``max_headers`` header lines (lines starting with ``>``) and
    extracts UniProt ``OS=`` organism names and ``OX=`` NCBI taxonomy IDs.

    Validation rules:
    - Always requires at least one FASTA header (rejects empty, HTML, or JSON
      error payloads with :class:`IntegrityError`).
    - When ``expected_taxon_id`` is provided (an ``int`` or iterable of accepted
      ``int`` taxon IDs) and any inspected header carries an ``OX=`` tag (true
      for all real UniProt FASTA downloads), raises :class:`IntegrityError`
      unless at least one expected taxon ID appears in ``observed_taxon_ids``.
    - When ``require_taxon_header=True``, raises :class:`IntegrityError` if no
      inspected header carries an ``OX=`` tag.  Defaults to ``False`` so minimal
      synthetic FASTA fixtures without ``OX=`` annotations remain usable.
    """
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise IntegrityError(f"UniProt FASTA file does not exist: {fasta_path}")

    expected_ids: set[int] = set()
    if expected_taxon_id is not None:
        if isinstance(expected_taxon_id, int):
            expected_ids.add(expected_taxon_id)
        else:
            expected_ids.update(int(tid) for tid in expected_taxon_id)

    n_headers_checked = 0
    observed_taxon_ids: set[int] = set()
    observed_organisms: set[str] = set()
    try:
        with open(fasta_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                n_headers_checked += 1
                ox_match = _UNIPROT_OX_RE.search(line)
                if ox_match:
                    observed_taxon_ids.add(int(ox_match.group(1)))
                os_match = _UNIPROT_OS_RE.search(line)
                if os_match:
                    observed_organisms.add(os_match.group(1).strip())
                if n_headers_checked >= max_headers:
                    break
    except OSError as exc:
        raise IntegrityError(f"Could not read UniProt FASTA {fasta_path}: {exc}") from exc

    org_note = f" for {expected_organism!r}" if expected_organism else ""
    if n_headers_checked == 0:
        raise IntegrityError(
            f"UniProt FASTA {fasta_path.name}{org_note} contains no FASTA headers "
            "(empty, truncated, or non-FASTA error response)"
        )

    has_taxon_headers = bool(observed_taxon_ids)
    if require_taxon_header and not has_taxon_headers:
        raise IntegrityError(
            f"UniProt FASTA {fasta_path.name}{org_note} has {n_headers_checked} FASTA header(s) "
            "but no UniProt OX=<taxon_id> annotations"
        )

    if expected_ids and has_taxon_headers and not (observed_taxon_ids & expected_ids):
        observed_org_list = sorted(observed_organisms)[:3]
        org_summary = f" (OS={observed_org_list})" if observed_org_list else ""
        raise IntegrityError(
            f"UniProt FASTA {fasta_path.name} taxon mismatch{org_note}: "
            f"expected OX={sorted(expected_ids)}, "
            f"observed OX={sorted(observed_taxon_ids)}{org_summary}"
        )

    return {
        "path": str(fasta_path),
        "n_headers_checked": n_headers_checked,
        "observed_taxon_ids": sorted(observed_taxon_ids),
        "observed_organisms": sorted(observed_organisms),
        "expected_taxon_ids": sorted(expected_ids),
        "has_taxon_headers": has_taxon_headers,
    }


def _proteome_filename_expectations() -> dict[str, dict[str, Any]]:
    """Map canonical proteome filenames under ``proteomes/`` to expected taxon metadata."""
    by_filename: dict[str, dict[str, Any]] = {}
    for species_name, entry in SPECIES_PROTEOMES.items():
        if entry.get("kind") != "uniprot":
            continue
        by_filename[_safe_filename(species_name)] = {
            "expected_organism": species_name,
            "proteome_id": entry.get("proteome_id"),
            "expected_taxon_id": entry.get("expected_taxon_id"),
        }
    for viral_name, entry in VIRAL_PROTEOMES.items():
        by_filename.setdefault(
            _safe_filename(viral_name),
            {
                "expected_organism": viral_name,
                "proteome_id": entry.get("proteome_id"),
                "expected_taxon_id": entry.get("expected_taxon_id"),
            },
        )
    return by_filename


def audit_cached_proteomes(*, max_headers: int = 200) -> dict[str, Any]:
    """Audit cached UniProt FASTA proteomes on disk for taxon mismatches and leftover tmp files (#977).

    Inspects both ``<data_dir>/proteomes/*.fasta`` and cached viral FASTAs from
    ``FETCHABLE_DATASETS``.  Read-only: never deletes or downloads files.
    """
    manifest = load_manifest()
    proteomes_manifest = manifest.get("proteomes", {})
    datasets_manifest = manifest.get("datasets", {})
    filename_expectations = _proteome_filename_expectations()

    # Map resolved paths to their manifest metadata when available.
    manifest_by_path: dict[str, tuple[str, dict[str, Any]]] = {}
    for canonical, entry in proteomes_manifest.items():
        raw_path = entry.get("path")
        if raw_path and entry.get("kind") == "uniprot":
            manifest_by_path[str(Path(raw_path).resolve())] = (canonical, entry)

    targets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    leftover_tmp_files: list[Path] = []

    proteomes_dir = data_dir() / "proteomes"
    if proteomes_dir.is_dir():
        for child in sorted(proteomes_dir.iterdir()):
            if not child.is_file():
                continue
            if child.name.endswith(".tmp") or child.name.endswith(".staging"):
                leftover_tmp_files.append(child)
                continue
            if child.suffix != ".fasta":
                continue
            resolved_key = str(child.resolve())
            seen_paths.add(resolved_key)
            meta = filename_expectations.get(child.name, {})
            canonical_name = meta.get("expected_organism")
            proteome_id = meta.get("proteome_id")
            expected_taxon_id = meta.get("expected_taxon_id")
            if resolved_key in manifest_by_path:
                m_name, m_entry = manifest_by_path[resolved_key]
                canonical_name = canonical_name or m_name
                m_upid = m_entry.get("proteome_id")
                if expected_taxon_id is None:
                    # Prefer current registry lookup when the manifest entry has no
                    # distinct strain-specific UPID or used a known-replaced UPID.
                    looked_up = lookup_proteome(m_name, use_uniprot=False, allow_network=False)
                    if (
                        looked_up is not None
                        and looked_up.get("kind") == "uniprot"
                        and (
                            m_upid is None
                            or m_upid == looked_up.get("proteome_id")
                            or m_upid in _REPLACED_WRONG_UPIDS
                        )
                    ):
                        proteome_id = looked_up.get("proteome_id") or proteome_id
                        expected_taxon_id = looked_up.get("expected_taxon_id")
                if proteome_id is None:
                    proteome_id = m_upid
                if expected_taxon_id is None:
                    expected_taxon_id = m_entry.get("expected_taxon_id")
                    if expected_taxon_id is None and proteome_id:
                        expected_taxon_id = _expected_taxon_for_upid(proteome_id)
            targets.append(
                {
                    "path": child,
                    "source": "proteomes",
                    "expected_organism": canonical_name or child.stem,
                    "proteome_id": proteome_id,
                    "expected_taxon_id": expected_taxon_id,
                }
            )

    base_dir = data_dir()
    for name, ds in sorted(FETCHABLE_DATASETS.items()):
        if ds.get("expected_taxon_id") is None:
            continue
        candidate_paths = [base_dir / ds["filename"]]
        ds_entry = datasets_manifest.get(name)
        if ds_entry and ds_entry.get("path"):
            candidate_paths.append(Path(ds_entry["path"]))
        for tmp_name in (f"{ds['filename']}.tmp", f".{ds['filename']}.staging"):
            tmp_candidate = base_dir / tmp_name
            if tmp_candidate.is_file():
                leftover_tmp_files.append(tmp_candidate)
        for cand in candidate_paths:
            if not cand.is_file():
                continue
            resolved_key = str(cand.resolve())
            if resolved_key in seen_paths:
                continue
            seen_paths.add(resolved_key)
            targets.append(
                {
                    "path": cand,
                    "source": f"dataset:{name}",
                    "expected_organism": ds.get("description", name),
                    "proteome_id": ds.get("proteome_id"),
                    "expected_taxon_id": ds["expected_taxon_id"],
                }
            )

    n_valid_files = 0
    mismatches: list[dict[str, Any]] = []
    for target in targets:
        try:
            validate_uniprot_fasta_headers(
                target["path"],
                expected_taxon_id=target["expected_taxon_id"],
                expected_organism=target["expected_organism"],
                require_taxon_header=False,
                max_headers=max_headers,
            )
            n_valid_files += 1
        except IntegrityError as exc:
            observed_taxon_ids: list[int] = []
            observed_organisms: list[str] = []
            try:
                info_dict = validate_uniprot_fasta_headers(
                    target["path"],
                    expected_taxon_id=None,
                    require_taxon_header=False,
                    max_headers=max_headers,
                )
                observed_taxon_ids = info_dict["observed_taxon_ids"]
                observed_organisms = info_dict["observed_organisms"]
            except IntegrityError:
                pass
            mismatches.append(
                {
                    "path": str(target["path"]),
                    "source": target["source"],
                    "expected_organism": target["expected_organism"],
                    "proteome_id": target["proteome_id"],
                    "expected_taxon_id": target["expected_taxon_id"],
                    "observed_taxon_ids": observed_taxon_ids,
                    "observed_organisms": observed_organisms,
                    "error": str(exc),
                }
            )

    unique_tmp_files = sorted({str(p) for p in leftover_tmp_files})
    return {
        "n_files_checked": len(targets),
        "n_valid_files": n_valid_files,
        "n_mismatched_files": len(mismatches),
        "n_leftover_tmp_files": len(unique_tmp_files),
        "mismatches": mismatches,
        "leftover_tmp_files": unique_tmp_files,
    }


def _proteomes_dir() -> Path:
    d = data_dir() / "proteomes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename(species: str) -> str:
    """Convert a species name to a filesystem-safe filename."""
    safe = species.lower().replace("/", "_").replace("\\", "_")
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in safe)
    # Collapse runs of underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") + ".fasta"


_UNIPROT_PROTEOME_SEARCH_URL = "https://rest.uniprot.org/proteomes/search"
_PROTEOME_TYPE_RANK = {
    "reference and representative proteome": 0,
    "reference proteome": 1,
    "representative proteome": 2,
    "other proteome": 3,
    "redundant proteome": 4,
}


# Organism strings that are known placeholders/noise — never send to UniProt.
_ORGANISM_DENYLIST: set[str] = {
    "",
    "unidentified",
    "unknown",
    "unclassified",
    "mixed",
    "various",
    "not available",
    "n/a",
    "na",
}


_NON_STREAMABLE_PROTEOME_TYPES: frozenset[str] = frozenset(
    {
        "non reference proteome",
        "redundant proteome",
        "excluded proteome",
    }
)
_ORGANISM_STOPWORDS: frozenset[str] = frozenset(
    {
        "virus",
        "strain",
        "isolate",
        "type",
        "subtype",
        "serotype",
        "subsp",
        "var",
        "sp",
        "spp",
        "group",
        "human",
        "mouse",
        "murine",
        "bovine",
        "porcine",
        "canine",
        "avian",
        "disease",
        "syndrome",
    }
)


def _organism_name_tokens(name: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", name.lower())
        if len(tok) >= 3 and tok not in _ORGANISM_STOPWORDS
    }


def _candidate_matches_organism(query_organism: str, candidate: dict[str, Any]) -> bool:
    """Guard against UniProt free-text search matching host/description of another organism (#977)."""
    tax = candidate.get("taxonomy") or {}
    names = [tax.get("scientificName") or "", tax.get("commonName") or ""]
    names.extend(tax.get("synonyms") or [])
    combined = " ".join(n.lower() for n in names if n)
    if not combined:
        return False
    q_lower = query_organism.lower().strip()
    if q_lower in combined:
        return True
    q_tokens = _organism_name_tokens(query_organism)
    c_tokens = _organism_name_tokens(combined)
    if not q_tokens:
        return True
    return bool(q_tokens & c_tokens)


def resolve_proteome_via_uniprot(
    organism: str,
    timeout: int = 15,
) -> dict | None:
    """Query UniProt REST to find the best reference proteome for an organism.

    Returns a dict with ``proteome_id``, ``scientific_name``, ``taxon_id``,
    ``proteome_type``, and ``protein_count`` fields.  Prefers "Reference
    and representative" over plain "Representative" proteomes and skips
    non-streamable ("Non Reference", "Redundant", "Excluded") proteomes as well
    as free-text hits whose taxonomy names do not match *organism* (#977).
    Returns ``None`` if no match is found or if the organism is on the denylist
    (e.g. ``"unidentified"``).

    The raw organism string is used as a free-text query, so strain
    suffixes like ``"(strain B95-8)"`` are tolerated.

    Raises on a transient transport failure (timeout / 5xx / DNS) so the caller
    can tell "UniProt is down" apart from "no such proteome" and avoid caching a
    permanent negative for what is really a temporary outage. Returns ``None``
    only for a *genuine* empty/denylisted result.
    """
    import json
    import urllib.parse
    import urllib.request

    if not organism:
        return None
    cleaned = organism.strip()
    if cleaned.lower() in _ORGANISM_DENYLIST:
        return None
    query = urllib.parse.quote(cleaned)
    url = f"{_UNIPROT_PROTEOME_SEARCH_URL}?query={query}&format=json&size=10"
    # NB: transport/parse errors deliberately propagate (see docstring).
    with urllib.request.urlopen(url, timeout=timeout) as r:
        payload = json.load(r)

    results = payload.get("results", [])
    if not results:
        return None

    filtered = [
        p
        for p in results
        if (p.get("proteomeType") or "").lower() not in _NON_STREAMABLE_PROTEOME_TYPES
        and _candidate_matches_organism(cleaned, p)
    ]
    if not filtered:
        return None

    def sort_key(p: dict) -> tuple:
        ptype = (p.get("proteomeType") or "").lower()
        rank = _PROTEOME_TYPE_RANK.get(ptype, 99)
        # Within same rank, prefer lower taxon id (parent species often
        # has a smaller ID than strain-specific entries) and higher count
        tax_id = int(p.get("taxonomy", {}).get("taxonId") or 1_000_000_000)
        count = int(p.get("proteinCount") or 0)
        return (rank, tax_id, -count)

    best = min(filtered, key=sort_key)
    tax = best.get("taxonomy", {})
    return {
        "proteome_id": best.get("id"),
        "scientific_name": tax.get("scientificName") or organism,
        "taxon_id": tax.get("taxonId"),
        "proteome_type": best.get("proteomeType"),
        "protein_count": best.get("proteinCount"),
    }


def _resolve_uniprot_proteome_taxon(upid: str, timeout: int = 15) -> int | None:
    """Query UniProt Proteomes metadata JSON for a UPID's NCBI taxon ID."""
    import urllib.request

    url = _UNIPROT_PROTEOME_META_URL.format(proteome_id=upid)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = json.load(r)
    except Exception:
        return None
    tax_id = (payload.get("taxonomy") or {}).get("taxonId")
    return int(tax_id) if tax_id is not None else None


def _find_existing_proteome_by_upid(proteomes: dict, proteome_id: str) -> dict | None:
    """Find a previously-downloaded proteome with the same UniProt ID."""
    for entry in proteomes.values():
        if (
            entry.get("kind") == "uniprot"
            and entry.get("proteome_id") == proteome_id
            and entry.get("path")
        ):
            return entry
    return None


def _uniprot_cache() -> dict:
    """Load the manifest's uniprot_resolutions section."""
    manifest = load_manifest()
    return manifest.get("uniprot_resolutions", {})


def _save_uniprot_cache_entry(organism: str, entry: dict | None) -> None:
    with manifest_transaction() as manifest:
        cache = manifest.setdefault("uniprot_resolutions", {})
        cache[organism] = {
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            **(entry or {"not_found": True}),
        }


def lookup_proteome(
    species_or_organism: str,
    use_uniprot: bool = False,
    allow_network: bool = True,
) -> dict | None:
    """Resolve a species or IEDB source_organism string to a proteome registry entry.

    Applies in order:
    1. Curated ``SPECIES_PROTEOMES`` registry (normalized via mhcgnomes)
    2. Curated ``VIRAL_PROTEOMES`` substring match
    3. (Optional) cached UniProt resolution when ``use_uniprot=True``
    4. (Optional) UniProt REST lookup when ``use_uniprot=True`` and
       ``allow_network=True``. Negative results are cached to avoid re-querying.

    Returns ``None`` if no proteome is registered/discoverable.
    """
    if not species_or_organism:
        return None

    from .curation import normalize_species

    canonical = normalize_species(species_or_organism)
    if canonical in SPECIES_PROTEOMES:
        entry = dict(SPECIES_PROTEOMES[canonical])
        entry["canonical_species"] = canonical
        return entry

    # Viral fallback: substring match on raw organism string.
    # Longer keys are checked first so "sars-cov-2" wins over "sars-cov".
    lowered = species_or_organism.lower()
    for viral_key in sorted(VIRAL_PROTEOMES, key=len, reverse=True):
        if viral_key in lowered:
            viral_entry = VIRAL_PROTEOMES[viral_key]
            viral_result: dict[str, Any] = {
                "kind": "uniprot",
                "proteome_id": viral_entry["proteome_id"],
                "canonical_species": species_or_organism,
                "key": viral_entry["key"],
                "expected_taxon_id": viral_entry.get("expected_taxon_id"),
            }
            if viral_entry.get("uniprot_query"):
                viral_result["uniprot_query"] = viral_entry["uniprot_query"]
            return viral_result

    # Species-name substring fallback (handles strain suffixes like
    # "Theileria parva strain Muguga" → Theileria parva).  Skip
    # generic genus-only entries ("Sus sp.", "Canis sp.") to avoid
    # spurious matches.
    for species_key in sorted(SPECIES_PROTEOMES, key=len, reverse=True):
        if species_key.endswith(" sp."):
            continue
        if species_key.lower() in lowered:
            entry = dict(SPECIES_PROTEOMES[species_key])
            entry["canonical_species"] = species_key
            return entry

    if not use_uniprot:
        return None

    # UniProt REST fallback — with manifest-cached results
    cache = _uniprot_cache()
    cached = cache.get(species_or_organism)
    if cached is not None:
        if cached.get("not_found"):
            return None
        return {
            "kind": "uniprot",
            "proteome_id": cached["proteome_id"],
            "canonical_species": cached.get("scientific_name", species_or_organism),
            "source": "uniprot_search",
            "taxon_id": cached.get("taxon_id"),
            "expected_taxon_id": cached.get("taxon_id"),
            "proteome_type": cached.get("proteome_type"),
        }

    if not allow_network:
        return None

    try:
        resolved = resolve_proteome_via_uniprot(species_or_organism)
    except Exception:
        # Transient UniProt failure: return None WITHOUT caching a negative, so
        # a later run retries instead of permanently excluding this organism.
        return None
    _save_uniprot_cache_entry(species_or_organism, resolved)
    if resolved is None:
        return None
    return {
        "kind": "uniprot",
        "proteome_id": resolved["proteome_id"],
        "canonical_species": resolved["scientific_name"],
        "source": "uniprot_search",
        "taxon_id": resolved.get("taxon_id"),
        "expected_taxon_id": resolved.get("taxon_id"),
        "proteome_type": resolved.get("proteome_type"),
    }


def fetch_species_proteome(
    species: str,
    force: bool = False,
    verbose: bool = True,
    use_uniprot: bool = False,
    fetch_missing: bool = True,
) -> Path | None:
    """Fetch (or return cached) reference proteome FASTA for a species.

    Returns the local FASTA path.  For Ensembl-supported species this is
    a sentinel (marker file) indicating the caller should use pyensembl
    instead.  Returns ``None`` if no proteome is registered.

    Parameters
    ----------
    species
        Any species or source_organism string.
    force
        Re-download even if already cached.
    verbose
        Print progress messages.
    use_uniprot
        Fall back to UniProt REST search for organisms not in the curated
        registry.  Resolved mappings are cached in the manifest to avoid
        re-querying.
    fetch_missing
        Download an uncached FASTA when True. When False, cached registry and
        UniProt-resolution entries may be reused but no network call is made.
    """
    entry = lookup_proteome(
        species,
        use_uniprot=use_uniprot,
        allow_network=fetch_missing,
    )
    if entry is None:
        return None

    canonical = entry.get("canonical_species", species)
    proteomes = load_manifest().get("proteomes", {})

    # Ensembl species: pyensembl manages its own cache — no FASTA to download
    if entry["kind"] == "ensembl":
        with manifest_transaction() as manifest:
            manifest.setdefault("proteomes", {})[canonical] = {
                "kind": "ensembl",
                "species": entry.get("species", canonical),
                "release": entry.get("release", 112),
                "registered": datetime.now(timezone.utc).isoformat(),
            }
        return None  # caller should use ProteomeIndex.from_ensembl()

    # UniProt species: download the FASTA (dedup by UPID — multiple strain
    # variants often resolve to the same UniProt reference proteome)
    proteome_id = entry["proteome_id"]
    expected_taxon_id = entry.get("expected_taxon_id")
    if expected_taxon_id is None:
        expected_taxon_id = entry.get("taxon_id") or _expected_taxon_for_upid(proteome_id)
    uniprot_query = entry.get("uniprot_query")
    url = (
        _UNIPROT_QUERY_URL.format(query=uniprot_query)
        if uniprot_query
        else _UNIPROT_PROTEOME_URL.format(proteome_id=proteome_id)
    )

    def _manifest_record(path: Path, *, registered: str | None = None) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "kind": "uniprot",
            "proteome_id": proteome_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "source_url": url,
            "canonical_species": canonical,
            "registered": registered or datetime.now(timezone.utc).isoformat(),
        }
        if expected_taxon_id is not None:
            rec["expected_taxon_id"] = int(expected_taxon_id)
        return rec

    existing = _find_existing_proteome_by_upid(proteomes, proteome_id)
    if existing is not None and Path(existing["path"]).exists() and not force:
        dest = Path(existing["path"])
        try:
            validate_uniprot_fasta_headers(
                dest,
                expected_taxon_id=expected_taxon_id,
                expected_organism=canonical,
            )
        except IntegrityError as exc:
            if not fetch_missing:
                raise
            print(
                f"  [{canonical}] cached FASTA rejected ({exc}); re-downloading",
                file=sys.stderr,
            )
        else:
            if verbose:
                print(
                    f"  [{canonical}] reusing cached FASTA from "
                    f"{existing.get('canonical_species', '?')} ({dest.stat().st_size:,} bytes)"
                )
            with manifest_transaction() as manifest:
                manifest.setdefault("proteomes", {})[canonical] = _manifest_record(dest)
            return dest

    fname = _safe_filename(canonical)
    dest = _proteomes_dir() / fname

    if dest.exists() and not force:
        try:
            validate_uniprot_fasta_headers(
                dest,
                expected_taxon_id=expected_taxon_id,
                expected_organism=canonical,
            )
        except IntegrityError as exc:
            if not fetch_missing:
                raise
            print(
                f"  [{canonical}] cached FASTA rejected ({exc}); re-downloading",
                file=sys.stderr,
            )
        else:
            if verbose:
                print(f"  [{canonical}] already cached ({dest.stat().st_size:,} bytes)")
            with manifest_transaction() as manifest:
                manifest.setdefault("proteomes", {})[canonical] = _manifest_record(
                    dest,
                    registered=proteomes.get(canonical, {}).get("registered"),
                )
            return dest

    if not fetch_missing:
        return None

    if verbose:
        print(f"  [{canonical}] fetching UniProt {proteome_id} ...")
    staging = dest.with_name(f".{dest.name}.staging")
    try:
        download_to_file(url, staging, label=canonical, verbose=verbose, force=True)
        validate_uniprot_fasta_headers(
            staging,
            expected_taxon_id=expected_taxon_id,
            expected_organism=canonical,
        )
        staging.replace(dest)
    except Exception:
        staging.unlink(missing_ok=True)
        raise

    size = dest.stat().st_size
    if verbose:
        print(f"  [{canonical}] downloaded {size:,} bytes → {dest}")

    with manifest_transaction() as manifest:
        manifest.setdefault("proteomes", {})[canonical] = _manifest_record(dest)
    return dest


def list_proteomes() -> dict:
    """Return the proteomes section of the manifest."""
    return load_manifest().get("proteomes", {})


def fetch_proteome_by_upid(
    upid: str,
    label: str | None = None,
    force: bool = False,
    verbose: bool = True,
    fetch_missing: bool = True,
    *,
    expected_taxon_id: int | None = None,
) -> Path | None:
    """Fetch (or return cached) a UniProt reference proteome by UPID.

    Unlike ``fetch_species_proteome`` which requires the organism to be
    in the curated registry, this fetches any UPID directly.  Used by
    the ``reference_proteomes`` override on ``ms_samples`` for per-sample
    viral/custom proteomes.

    Parameters
    ----------
    upid
        UniProt proteome ID (e.g. ``"UP000153037"``).
    label
        Optional human-readable name for logging and the filename.
        Defaults to the UPID.
    force
        Re-download even if already cached.
    verbose
        Print progress messages.
    fetch_missing
        Download the FASTA when it is not already cached. When False, return
        ``None`` without network access for a cache miss.
    expected_taxon_id
        Optional NCBI taxon ID to validate against FASTA ``OX=`` headers.
        Falls back to the curated taxon ID when *upid* is in the registry.
    """
    if not upid:
        return None
    proteomes = load_manifest().get("proteomes", {})
    resolved_expected_taxon = (
        expected_taxon_id if expected_taxon_id is not None else _expected_taxon_for_upid(upid)
    )
    name = label or upid

    # Dedup by UPID if we already have this proteome under another key
    existing = _find_existing_proteome_by_upid(proteomes, upid)
    if existing is not None and Path(existing["path"]).exists() and not force:
        path = Path(existing["path"])
        try:
            validate_uniprot_fasta_headers(
                path,
                expected_taxon_id=resolved_expected_taxon,
                expected_organism=name,
            )
        except IntegrityError as exc:
            if not fetch_missing:
                raise
            print(
                f"  [{name}] cached FASTA rejected ({exc}); re-downloading",
                file=sys.stderr,
            )
        else:
            if verbose:
                print(f"  [{name}] reusing cached {path.name} ({path.stat().st_size:,} bytes)")
            return path

    fname = _safe_filename(name)
    dest = _proteomes_dir() / fname
    url = _UNIPROT_PROTEOME_URL.format(proteome_id=upid)

    cached_ok = False
    if dest.exists() and not force:
        try:
            validate_uniprot_fasta_headers(
                dest,
                expected_taxon_id=resolved_expected_taxon,
                expected_organism=name,
            )
        except IntegrityError as exc:
            if not fetch_missing:
                raise
            print(
                f"  [{name}] cached FASTA rejected ({exc}); re-downloading",
                file=sys.stderr,
            )
        else:
            cached_ok = True
            if verbose:
                print(f"  [{name}] already cached ({dest.stat().st_size:,} bytes)")

    if not cached_ok:
        if not fetch_missing:
            return None
        if verbose:
            print(f"  [{name}] fetching UniProt {upid} ...")
        staging = dest.with_name(f".{dest.name}.staging")
        try:
            download_to_file(url, staging, label=name, verbose=verbose, force=True)
            try:
                validate_uniprot_fasta_headers(
                    staging,
                    expected_taxon_id=resolved_expected_taxon,
                    expected_organism=name,
                )
            except IntegrityError as exc:
                # UniProtKB ``query=proteome:UP...`` returns 0 bytes for Non-Reference
                # proteomes (e.g. Measles Edmonston B UP000100252); fall back to
                # ``organism_id:<taxon_id>`` when metadata resolves a taxon (#977).
                if "contains no FASTA headers" not in str(exc):
                    raise
                fallback_taxon = resolved_expected_taxon or _resolve_uniprot_proteome_taxon(upid)
                if fallback_taxon is None:
                    raise
                url = _UNIPROT_QUERY_URL.format(query=f"organism_id:{fallback_taxon}")
                resolved_expected_taxon = fallback_taxon
                if verbose:
                    print(
                        f"  [{name}] UniProt proteome:{upid} returned empty stream; "
                        f"falling back to organism_id:{fallback_taxon}"
                    )
                download_to_file(url, staging, label=name, verbose=verbose, force=True)
                validate_uniprot_fasta_headers(
                    staging,
                    expected_taxon_id=resolved_expected_taxon,
                    expected_organism=name,
                )
            staging.replace(dest)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
        if verbose:
            print(f"  [{name}] downloaded {dest.stat().st_size:,} bytes → {dest}")

    rec: dict[str, Any] = {
        "kind": "uniprot",
        "proteome_id": upid,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "source_url": url,
        "canonical_species": name,
        "registered": datetime.now(timezone.utc).isoformat(),
    }
    if resolved_expected_taxon is not None:
        rec["expected_taxon_id"] = int(resolved_expected_taxon)
    with manifest_transaction() as manifest:
        manifest.setdefault("proteomes", {})[name] = rec
    return dest


# ── Mirrored data assets (issue #303) ───────────────────────────────────────
# Every important data CSV is mirrored to the GitHub "data-assets-v1" release
# for backup + high availability, and fetched on demand into the datacache cache
# dir (datacache.get_data_dir('hitlist'); openvax ecosystem, #291).  The registry
# lives in hitlist/data/data_assets.yaml.  Large files (``bundled: false``) are
# excluded from the PyPI wheel and ALWAYS fetched; small files ship in the wheel
# for out-of-box use and are mirrored only for backup / ``data fetch-all``.


@lru_cache(maxsize=1)
def _data_assets_registry() -> dict:
    """Load the data-assets manifest (base_url + per-file sha256/source)."""
    from importlib.resources import files as _ir_files

    from .curation_yaml import load_curation_yaml

    doc = load_curation_yaml(_ir_files("hitlist.data") / "data_assets.yaml") or {}
    base_url = doc.get("base_url", "")
    assets = {
        a["filename"]: {
            "sha256": a["sha256"],
            "size_bytes": a["size_bytes"],
            "source": a.get("source", ""),
            "bundled": bool(a.get("bundled", True)),
            "url": f"{base_url}/{a['filename']}",
        }
        for a in doc.get("assets", [])
    }
    return {"base_url": base_url, "assets": assets}


def data_assets() -> dict[str, dict]:
    """Return the full mirrored-data-asset registry, keyed by filename."""
    return dict(_data_assets_registry()["assets"])


# Backwards-compatible alias for callers that imported the dict form.
class _ExternalDataAssetsView:
    def __contains__(self, key: object) -> bool:
        return key in _data_assets_registry()["assets"]

    def __getitem__(self, key: str) -> dict:
        return _data_assets_registry()["assets"][key]

    def __iter__(self):
        return iter(_data_assets_registry()["assets"])

    def keys(self):
        return _data_assets_registry()["assets"].keys()


EXTERNAL_DATA_ASSETS = _ExternalDataAssetsView()


def fetch_data_asset(
    filename: str,
    *,
    force: bool = False,
    verbose: bool = True,
    offline: bool | None = None,
) -> Path:
    """Fetch a mirrored data asset into the datacache cache dir (#303, #960).

    Uses ``datacache.fetch_file`` (openvax ecosystem cache; #291) so the file is
    stored under ``data_asset_dir()`` and reused across runs.

    datacache verifies the registry's size and SHA-256 before publication and on
    reuse, bounds socket inactivity, resumes interrupted raw HTTP transfers, and
    records source provenance. ``force=True`` explicitly repairs a corrupt cache.
    """
    assets = _data_assets_registry()["assets"]
    if filename not in assets:
        raise KeyError(f"unknown data asset {filename!r}; known: {sorted(assets)}")
    import datacache

    meta = assets[filename]
    dest = data_asset_dir() / filename
    if force or not dest.exists():
        _require_online(filename, offline=offline)
        _ensure_enough_space(filename, preflight(filename, force=True))
    if verbose:
        print(f"Fetching {filename} ({meta['source']}) via datacache...", file=sys.stderr)
    fetch_kwargs: dict[str, Any] = {
        "force": force,
        "timeout": _DOWNLOAD_SOCKET_TIMEOUT,
        "expected_sha256": meta["sha256"],
        "expected_size": meta["size_bytes"],
        "raw": True,
        "resume": os.name == "posix",
        "record_provenance": True,
        "show_progress": verbose,
    }
    if _resolve_openvax_cache_dir() is not None:
        fetch_kwargs["destination"] = dest
    else:
        fetch_kwargs["filename"] = filename
        fetch_kwargs["subdir"] = "hitlist"
    with _typed_download_errors(meta["url"]):
        return Path(datacache.fetch_file(meta["url"], **fetch_kwargs))


def fetch_all_data_assets(
    *,
    force: bool = False,
    verbose: bool = True,
    offline: bool | None = None,
) -> dict[str, Path]:
    """Fetch (and checksum-verify) EVERY mirrored data asset into the cache dir.

    Backs the ``hitlist data fetch-all`` command — one call to pull the complete
    set of paper-derived CSVs from the GitHub data-assets release. Returns a
    ``{filename: cached_path}`` map. Re-uses cached copies unless ``force``.

    A failed asset does not stop the rest: after every asset was attempted, a
    summary is printed and a :class:`DownloadError` listing each failure is raised.
    """
    assets = _data_assets_registry()["assets"]
    out: dict[str, Path] = {}
    failures: dict[str, HitlistError] = {}
    for i, filename in enumerate(sorted(assets), 1):
        if verbose:
            print(f"[{i}/{len(assets)}] {filename}", file=sys.stderr)
        try:
            if offline is None:
                out[filename] = fetch_data_asset(filename, force=force, verbose=verbose)
            else:
                out[filename] = fetch_data_asset(
                    filename, force=force, verbose=verbose, offline=offline
                )
        except HitlistError as error:
            failures[filename] = error
            print(f"  FAILED {filename}: {error}", file=sys.stderr)
    if verbose and out:
        total = sum(p.stat().st_size for p in out.values())
        cache_dir = next(iter(out.values())).parent
        print(
            f"Done — {len(out)} assets cached in {cache_dir} ({total / 1e6:.1f} MB).",
            file=sys.stderr,
        )
    if failures:
        if all(isinstance(err, OfflineError) for err in failures.values()):
            first_err = next(iter(failures.values()))
            raise OfflineError(
                f"Offline mode blocked {len(failures)} of {len(assets)} data assets: "
                + ", ".join(list(failures)[:5])
            ) from first_err
        listing = "; ".join(f"{name} ({error})" for name, error in failures.items())
        print(f"{len(failures)} of {len(assets)} assets failed.", file=sys.stderr)
        raise DownloadError(f"{len(failures)} of {len(assets)} data assets failed: {listing}")
    return out


def packaged_or_fetched(packaged_path, filename: str) -> Path:
    """Return the packaged data file if it exists locally (source / editable
    install), otherwise fetch the externalized copy via :func:`fetch_data_asset`.

    ``packaged_path`` may be a ``pathlib.Path`` or an ``importlib.resources``
    Traversable; only its string form is used for the existence check.
    """
    p = Path(str(packaged_path))
    if p.is_file():
        return p
    return fetch_data_asset(filename)


def packaged_or_cached(packaged_path, filename: str) -> Path | None:
    """Like :func:`packaged_or_fetched`, but never downloads.

    Returns the packaged file, else the copy a previous
    :func:`fetch_data_asset` left in the datacache dir, else ``None``. For
    callers that must not reach the network: the cache-validity predicates
    (#448) resolve every fingerprinted input through here.
    """
    p = Path(str(packaged_path))
    if p.is_file():
        return p
    # The same path fetch_data_asset's ``datacache.fetch_file(..., subdir="hitlist")``
    # writes to, asked of datacache itself so the two cannot drift.
    # ``expected_path`` (datacache >= 1.11.1) is documented to resolve the fetch
    # destination "without filesystem access or mutations", so — unlike
    # ``datacache.build_path``, which still creates the parent — it leaves the
    # filesystem alone.
    from datacache import expected_path

    cached = Path(expected_path(filename=filename, subdir="hitlist"))
    return cached if cached.is_file() else None


# ── Core API ────────────────────────────────────────────────────────────────


def register(
    name: str,
    path: str | Path,
    description: str | None = None,
    *,
    preserve_symlink: bool = True,
) -> Path:
    """Register a local file path for a named dataset.

    By default (``preserve_symlink=True``), symlinks are recorded by their
    normalized absolute path rather than dereferenced, so stable symlinks
    (e.g. ``iedb_latest.csv -> iedb_2026_11.csv``) stay live and report
    ``symlink_status == "valid_symlink"`` consistently with :func:`info`,
    :func:`preflight`, and ``build_observations()`` (#954, #958, #964).
    Target existence is always validated via ``Path.resolve(strict=True)``.
    Pass ``preserve_symlink=False`` to store the dereferenced target path.
    """
    normalized = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = normalized.resolve(strict=True)
    except FileNotFoundError as exc:
        if normalized.is_symlink():
            raise FileNotFoundError(
                f"Broken symlink: {normalized} -> {os.readlink(normalized)}"
            ) from exc
        raise FileNotFoundError(f"File not found: {normalized}") from exc

    p = normalized if preserve_symlink else resolved

    desc = description
    if desc is None and name in MANUAL_DATASETS:
        desc = MANUAL_DATASETS[name]["description"]
    if desc is None and name in FETCHABLE_DATASETS:
        desc = FETCHABLE_DATASETS[name]["description"]

    with manifest_transaction() as manifest:
        manifest.setdefault("datasets", {})[name] = {
            "path": str(p),
            "registered": datetime.now(timezone.utc).isoformat(),
            "size_bytes": resolved.stat().st_size,
            "description": desc or "",
            "source": "registered",
        }
    return p


def _depmap_file_is_registered(key: str) -> bool:
    registered = load_manifest().get("datasets", {}).get(key, {})
    return bool(registered.get("path")) and Path(registered["path"]).exists()


def unregistered_depmap_files() -> list[str]:
    """File names ``hitlist data fetch depmap`` would download before rebuilding.

    :func:`fetch` reuses every bundle file that is registered and still on
    disk and downloads only the rest, so an empty list means the command is a
    purely local rebuild of the line-expression index.  Answering this per
    file lets a "rebuild your index" message say exactly what a fetch costs.
    """
    return [
        filename
        for key, (_file_id, filename) in _DEPMAP_FILES.items()
        if not _depmap_file_is_registered(key)
    ]


def _unfetchable(name: str) -> DatasetError:
    """Why *name* cannot be fetched automatically: a manual download, or no such dataset."""
    if name in MANUAL_DATASETS:
        info = MANUAL_DATASETS[name]
        return DatasetError(
            f"'{name}' requires manual download from:\n"
            f"  {info['download_url']}\n"
            f"Then register the downloaded {info['expected_filename']} file "
            f"under dataset name '{name}'."
        )
    return DatasetError(f"Unknown dataset '{name}'. Available: {sorted(available_datasets())}")


def _free_bytes_for_path(dest: Path) -> int:
    """Free bytes on the filesystem hosting the nearest existing ancestor of *dest*."""
    probe = dest.parent
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def _ensure_enough_space(name: str, plan: dict) -> None:
    """Raise :class:`InsufficientSpaceError` when a declared size exceeds free space."""
    if plan["enough_space"] is False:
        target_dir = Path(plan["path"]).parent
        raise InsufficientSpaceError(
            f"Not enough free space for '{name}': needs about "
            f"{plan['expected_size_bytes'] / 1e9:.1f} GB, {plan['free_bytes'] / 1e9:.1f} GB "
            f"free under {target_dir}",
            path=target_dir,
            required_bytes=plan["expected_size_bytes"],
            available_bytes=plan["free_bytes"],
        )


def fetch(name: str, force: bool = False, *, offline: bool | None = None) -> Path:
    """Download a fetchable dataset."""
    if name == "depmap":
        from .builder import build_line_expression

        for key in _DEPMAP_FILES:
            if not force and _depmap_file_is_registered(key):
                continue
            fetch(key, force=force, offline=offline)
        build_line_expression(verbose=True)
        from .line_expression import line_expression_path

        output = line_expression_path()
        register("depmap", output, description="DepMap 24Q4 line-expression index")
        return output
    if name not in FETCHABLE_DATASETS:
        if name in data_assets():
            asset_dest = data_asset_dir() / name
            if force or not asset_dest.exists():
                _require_online(name, offline=offline)
            if offline is None:
                return fetch_data_asset(name, force=force)
            return fetch_data_asset(name, force=force, offline=offline)
        from ._dataset_catalog import fetch_managed, get_managed_dataset

        managed = get_managed_dataset(name)
        if managed is not None:
            if force or not managed.exists_fn():
                _require_online(name, offline=offline)
            return fetch_managed(name, force=force)
        raise _unfetchable(name)

    ds = FETCHABLE_DATASETS[name]
    dest = data_dir() / ds["filename"]
    entry = load_manifest().get("datasets", {}).get(name)

    if dest.exists() and not force:
        problem = _cached_file_problem(name, dest, entry)
        if problem is None:
            if entry is None:
                print(
                    f"  [{name}] adopting existing {dest} (no recorded checksum); "
                    f"re-run with --force to re-download and record one",
                    file=sys.stderr,
                )
                register(name, dest, ds["description"])
            return dest
        print(f"  [{name}] cached file rejected ({problem}); re-downloading", file=sys.stderr)
        force = True

    _require_online(name, offline=offline)
    if entry is not None and not Path(entry["path"]).exists():
        print(
            f"  [{name}] registered file {entry['path']} is missing; downloading again",
            file=sys.stderr,
        )
    _ensure_enough_space(name, preflight(name, force=True))
    if ds.get("terms"):
        print(
            f"  [{name}] from a terms-of-use-governed source — please review "
            f"usage/citation terms: {ds['terms']}",
            file=sys.stderr,
        )
    print(f"Downloading {ds['description']}...", file=sys.stderr)
    # decompress=True unzips/gunzips the IEDB/CEDAR archives into their CSV; it
    # is a no-op for the plain-FASTA viral proteomes (download_to_file only
    # decompresses when the URL is a .zip/.gz the dest filename doesn't carry).
    download_kwargs: dict[str, Any] = {
        "label": name,
        "force": force,
        "decompress": True,
        "expected_sha256": ds.get("expected_sha256"),
        "expected_size": ds.get("expected_size"),
    }
    if offline is not None:
        download_kwargs["offline"] = offline
    download_to_file(ds["url"], dest, **download_kwargs)
    size = dest.stat().st_size
    if size < ds["min_size_bytes"]:
        # Drop it so a later cache hit cannot mistake the empty body for good data.
        dest.unlink()
        raise IntegrityError(
            f"Downloaded '{name}' from {ds['url']} is only {size:,} bytes "
            f"(expected at least {ds['min_size_bytes']:,}); the server likely returned "
            f"an empty or error body"
        )
    if ds.get("expected_taxon_id") is not None:
        try:
            validate_uniprot_fasta_headers(
                dest,
                expected_taxon_id=ds["expected_taxon_id"],
                expected_organism=ds.get("description", name),
            )
        except IntegrityError:
            dest.unlink(missing_ok=True)
            raise

    with manifest_transaction() as manifest:
        manifest["datasets"][name] = {
            "path": str(dest),
            "registered": datetime.now(timezone.utc).isoformat(),
            "size_bytes": size,
            "sha256": _file_sha256(dest),
            "description": ds["description"],
            "source": ds["url"],
        }
    print(f"  Saved to {dest} ({size:,} bytes)", file=sys.stderr)
    return dest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_path_symlink_state(path: str | Path) -> dict[str, Any]:
    """Inspect a local path consistently with ``build_observations()`` (#954, #958).

    Returns ``is_symlink``, ``resolved_path`` (canonical path when resolvable, else ``None``),
    and ``symlink_status`` in ``{"regular_file", "valid_symlink", "broken_symlink",
    "non_regular_target", "missing"}``.
    """
    p = Path(path)
    is_link = p.is_symlink()
    if is_link:
        try:
            resolved = p.resolve(strict=True)
        except OSError:
            return {
                "is_symlink": True,
                "resolved_path": None,
                "symlink_status": "broken_symlink",
            }
        if not resolved.is_file():
            return {
                "is_symlink": True,
                "resolved_path": str(resolved),
                "symlink_status": "non_regular_target",
            }
        return {
            "is_symlink": True,
            "resolved_path": str(resolved),
            "symlink_status": "valid_symlink",
        }
    if not p.exists():
        return {
            "is_symlink": False,
            "resolved_path": None,
            "symlink_status": "missing",
        }
    resolved = p.resolve()
    if not resolved.is_file():
        return {
            "is_symlink": False,
            "resolved_path": str(resolved),
            "symlink_status": "non_regular_target",
        }
    return {
        "is_symlink": False,
        "resolved_path": str(resolved),
        "symlink_status": "regular_file",
    }


def preflight(name: str, *, force: bool = False) -> dict:
    """Report what :func:`fetch` would do for *name* without touching the network.

    Keys: ``name``; ``path`` (where the file lands); ``registered``; ``exists``;
    ``stale`` (registered, but the registered file is gone); ``would_download``;
    ``expected_size_bytes`` (``None`` when unknown); ``free_bytes`` on the target
    filesystem; ``enough_space`` (``None`` when the size is unknown); ``terms``
    (a terms-of-use URL, or ``None``).  ``hitlist data fetch --dry-run`` prints this.
    """
    if name in FETCHABLE_DATASETS:
        ds = FETCHABLE_DATASETS[name]
        dest = data_dir() / ds["filename"]
        expected = ds.get("size_bytes") or ds.get("expected_size")
        terms = ds.get("terms")
    elif name in data_assets():
        dest = data_asset_dir() / name
        expected, terms = data_assets()[name]["size_bytes"], None
    elif name == "depmap":
        from .line_expression import line_expression_path

        output = line_expression_path()
        files = []
        for key in sorted(_DEPMAP_FILES):
            member_plan = preflight(key, force=force)
            files.append(
                {
                    "name": key,
                    "path": member_plan["path"],
                    "exists": member_plan["exists"],
                    "would_download": member_plan["would_download"],
                    "expected_size_bytes": member_plan["expected_size_bytes"],
                }
            )
        sizes = [f["expected_size_bytes"] for f in files]
        expected_total = sum(sizes) if sizes and all(s is not None for s in sizes) else None
        free = _free_bytes_for_path(output)
        entry = load_manifest().get("datasets", {}).get("depmap")
        exists = output.exists()
        return {
            "name": "depmap",
            "kind": "bundle",
            "path": str(output),
            "registered": entry is not None,
            "exists": exists,
            "stale": entry is not None and not Path(entry["path"]).exists(),
            "would_download": force or not exists or any(f["would_download"] for f in files),
            "expected_size_bytes": expected_total,
            "free_bytes": free,
            "enough_space": None if expected_total is None else free >= expected_total,
            "terms": "https://doi.org/10.25452/figshare.plus.27993248.v1",
            "files": files,
        }
    else:
        from ._dataset_catalog import get_managed_dataset, preflight_managed

        if get_managed_dataset(name) is not None:
            return preflight_managed(name, force=force)
        raise _unfetchable(name)
    entry = load_manifest().get("datasets", {}).get(name)
    free = _free_bytes_for_path(dest)
    check_path = Path(entry["path"]) if entry is not None else dest
    link_state = _inspect_path_symlink_state(check_path)
    reg_valid = entry is not None and link_state["symlink_status"] in (
        "regular_file",
        "valid_symlink",
    )
    dest_state = _inspect_path_symlink_state(dest) if entry is not None else link_state
    dest_valid = dest_state["symlink_status"] in ("regular_file", "valid_symlink")
    return {
        "name": name,
        "path": str(dest),
        "registered": entry is not None,
        "exists": dest_valid,
        "stale": entry is not None and not reg_valid,
        "would_download": force or not dest_valid,
        "expected_size_bytes": expected,
        "free_bytes": free,
        "enough_space": None if expected is None else free >= expected,
        "terms": terms,
        "is_symlink": link_state["is_symlink"],
        "resolved_path": link_state["resolved_path"],
        "symlink_status": link_state["symlink_status"],
    }


def _cached_file_problem(name: str, dest: Path, entry: dict | None) -> str | None:
    """Why the cached file can't be reused, or None; checks the recorded size and sha256."""
    size = dest.stat().st_size
    ds = FETCHABLE_DATASETS[name]
    floor = ds["min_size_bytes"]
    if size < floor:
        return f"{size:,} bytes, below the {floor:,}-byte minimum"
    if "expected_size" in ds and size != ds["expected_size"]:
        return f"{size:,} bytes on disk, {ds['expected_size']:,} pinned"
    if "expected_sha256" in ds and _file_sha256(dest) != ds["expected_sha256"]:
        return "sha256 differs from the pinned source value"
    if ds.get("expected_taxon_id") is not None:
        try:
            validate_uniprot_fasta_headers(
                dest,
                expected_taxon_id=ds["expected_taxon_id"],
                expected_organism=ds.get("description", name),
            )
        except IntegrityError as exc:
            return str(exc)
    if entry is None or Path(entry["path"]) != dest:
        return None
    if "size_bytes" in entry and entry["size_bytes"] != size:
        return f"{size:,} bytes on disk, {entry['size_bytes']:,} recorded"
    if "sha256" in entry and entry["sha256"] != _file_sha256(dest):
        return "sha256 differs from the recorded value"
    return None


def get_path(name: str) -> Path:
    """Resolve a dataset name to its local file path."""
    manifest = load_manifest()
    entry = manifest.get("datasets", {}).get(name)
    if entry is None and name in data_assets():
        path = data_asset_dir() / name
        if not path.exists():
            raise FileNotFoundError(
                f"Mirrored asset missing: {path}; fetch with hitlist data fetch {name}"
            )
        return path
    if entry is None:
        from ._dataset_catalog import get_managed_dataset, get_managed_path

        if get_managed_dataset(name) is not None:
            return get_managed_path(name)
        hint = ""
        if name in FETCHABLE_DATASETS:
            hint = f"\n  Fetch with: hitlist data fetch {name}"
        elif name in MANUAL_DATASETS:
            info = MANUAL_DATASETS[name]
            hint = (
                f"\n  Download from: {info['download_url']}"
                f"\n  Register the downloaded {info['expected_filename']} file "
                f"under dataset name '{name}'."
            )
        raise KeyError(f"Dataset '{name}' not registered.{hint}")

    p = Path(entry["path"])
    if not p.exists():
        raise FileNotFoundError(
            f"Registered path for '{name}' no longer exists: {p}\nRe-register or re-fetch."
        )
    return p


def info(name: str) -> dict:
    """Get detailed metadata for a registered dataset."""
    import datacache

    manifest = load_manifest()
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
        if name in data_assets():
            meta = data_assets()[name]
            state = datacache.inspect_file(
                data_asset_dir() / name, expected_size=meta["size_bytes"]
            )
            return {
                **meta,
                "type": "mirrored asset",
                "exists": state.status != "missing",
                "path": state.path,
                "status": "installed" if state.status == "available" else state.status,
                "cache_status": state.status,
                "verified": state.verified,
                "source_url": state.source_url,
                "fetched_at": state.fetched_at,
                "recorded_sha256": state.recorded_sha256,
                "error": str(state.error) if state.error else None,
            }
        # Return known info even if not registered
        if name == "depmap":
            return {
                "description": "DepMap 24Q4 gene/transcript expression and index (~4.7 GB)",
                "datasets": list(_DEPMAP_FILES),
                "exists": False,
                "status": "not installed",
                "type": "bundle + build",
            }
        if name in FETCHABLE_DATASETS:
            return {
                **FETCHABLE_DATASETS[name],
                "exists": False,
                "status": "not installed",
                "type": "auto-fetch",
            }
        if name in MANUAL_DATASETS:
            return {
                **MANUAL_DATASETS[name],
                "exists": False,
                "status": "not installed",
                "type": "manual download",
            }
        from ._dataset_catalog import get_managed_dataset

        managed = get_managed_dataset(name)
        if managed is not None:
            return managed.info_fn()
        raise KeyError(f"Unknown dataset '{name}'")
    result = dict(entry)
    link_state = _inspect_path_symlink_state(entry["path"])
    result.update(link_state)
    if link_state["symlink_status"] == "broken_symlink":
        result.update(
            exists=False,
            status="missing",
            cache_status="missing",
            verified=False,
            source_url=None,
            fetched_at=None,
            recorded_sha256=None,
            error=f"Broken symlink: {entry['path']}",
        )
    elif link_state["symlink_status"] == "non_regular_target":
        result.update(
            exists=False,
            status="invalid",
            cache_status="invalid",
            verified=False,
            source_url=None,
            fetched_at=None,
            recorded_sha256=None,
            error=(
                f"Registered path {entry['path']} resolves to {link_state['resolved_path']}, "
                "which is not a regular file"
            ),
        )
    else:
        inspect_target = link_state["resolved_path"] or entry["path"]
        state = datacache.inspect_file(inspect_target)
        result.update(
            exists=state.status != "missing",
            status="installed" if state.status == "available" else state.status,
            cache_status=state.status,
            verified=state.verified,
            source_url=state.source_url,
            fetched_at=state.fetched_at,
            recorded_sha256=state.recorded_sha256,
            error=str(state.error) if state.error else None,
        )
    if name in FETCHABLE_DATASETS:
        result["type"] = "auto-fetch"
        result["usage"] = FETCHABLE_DATASETS[name].get("usage", "")
    elif name in MANUAL_DATASETS:
        result["type"] = "manual download"
        result["usage"] = MANUAL_DATASETS[name].get("usage", "")
    return result


def list_datasets() -> dict[str, dict]:
    """Return all registered/fetched datasets; ``exists`` says whether the file is still there."""
    return {
        name: {**entry, "exists": Path(entry["path"]).exists()}
        for name, entry in load_manifest().get("datasets", {}).items()
    }


def list_cache_files(*, verify: bool = False, include_unregistered: bool = True) -> list[dict]:
    """Inspect files across all cache locations; see :mod:`hitlist.cache_inventory`."""
    from .cache_inventory import list_cache_files as inventory

    return inventory(verify=verify, include_unregistered=include_unregistered)


DATASET_CATEGORIES: tuple[str, ...] = (
    "core_immunopeptidomics",
    "viral_proteomes",
    "reference_proteomes",
    "expression_and_proteomics",
    "tcell_and_atlas",
    "perturbation_and_calibration",
    "canine_and_cross_species",
    "supplementary_immunopeptidomics",
)

_VIRAL_DATASET_NAMES = frozenset(
    {"ebv", "hbv", "hcv", "hiv1", "hpv16", "hpv18", "htlv1", "kshv", "mcpyv"}
)
_EXPRESSION_PREFIXES = ("depmap", "hpa_", "ccle_", "bekker_jensen_", "bekker-jensen-", "gm12878_")
_PERTURBATION_NAMES = frozenset(
    {
        "bergseng-2015-dq",
        "dodm-ratio-35085787",
        "stopfer-2021-iso18",
        "pollock-2021-neotoma",
        "sarango-2022-pxd024417-sdrf",
        "sanz-bravo-2018b-erap2-oe",
        "pxd024809-dilution-observer",
        "pxd052314-goldberg-mapp-metadata",
        "pxd052314-goldberg-mapp",
        "pxd004886-ms-calibration",
        "maben-2021-erap1-source-data",
        "pxd058121-intact-degradomics",
        "pxd018722-sahu-2021-psmd2-metadata",
        "perturbation-quantities",
        "dlbcl-35561310-supplements",
        "pxd013455",
        "pxd013455-sources",
        "netmhcstabpan",
    }
)


def dataset_categories() -> tuple[str, ...]:
    """Return the canonical structured dataset catalog categories (#958)."""
    return DATASET_CATEGORIES


def _classify_catalog_entry(name: str, *, kind: str) -> tuple[str, str, str]:
    """Return ``(category, group, species)`` for a dataset catalog name (#958)."""
    if name in ("iedb", "cedar"):
        return "core_immunopeptidomics", "core", "multi-species"
    if name in _VIRAL_DATASET_NAMES:
        return "viral_proteomes", "viral", "viral"
    if name.startswith("uniprot/"):
        return "reference_proteomes", "uniprot", "Homo sapiens"
    if name.startswith("canine-bulk"):
        return "canine_and_cross_species", "canine_bulk", "Canis lupus familiaris"
    if name in ("miyamae_2026_dla88", "lang_2026_splenic_dla_dr"):
        return "canine_and_cross_species", "canine_dla", "Canis lupus familiaris"
    if name in ("tcell-assays", "hla-ligand-atlas", "hla-ligand-atlas-crosswalk", "tcrvdb"):
        return (
            "tcell_and_atlas",
            "tcr_functional" if name == "tcrvdb" else "atlas_and_tcell",
            "Homo sapiens",
        )
    if name in _PERTURBATION_NAMES:
        species = "Mus musculus" if name == "pollock-2021-neotoma" else "Homo sapiens"
        return "perturbation_and_calibration", "source_qualified", species
    if name == "peptidomicsdb_metadata":
        return "perturbation_and_calibration", "peptidomics_qualification", "multi-species"
    if name.startswith(_EXPRESSION_PREFIXES):
        group = (
            "depmap"
            if name.startswith("depmap")
            else ("hpa" if name.startswith("hpa_") else "proteomics_expression")
        )
        return "expression_and_proteomics", group, "Homo sapiens"
    if kind == "mirrored asset":
        return "supplementary_immunopeptidomics", "mirrored_study_csv", "Homo sapiens"
    return "supplementary_immunopeptidomics", "other", "Homo sapiens"


def available_dataset_catalog(*, category: str | None = None) -> dict[str, dict[str, Any]]:
    """Return structured metadata (category, group, species, kind, description) per dataset (#958).

    Parameters
    ----------
    category
        Optional category name from :func:`dataset_categories` to filter results.
    """
    if category is not None and category not in DATASET_CATEGORIES:
        raise ValueError(
            f"Unknown dataset category {category!r}; choose from {list(DATASET_CATEGORIES)}"
        )

    from ._dataset_catalog import managed_datasets

    entries: dict[str, dict[str, Any]] = {}

    def _add(name: str, description: str, kind: str) -> None:
        if name in entries:
            return
        cat, group, species = _classify_catalog_entry(name, kind=kind)
        if category is not None and cat != category:
            return
        entries[name] = {
            "name": name,
            "category": cat,
            "group": group,
            "species": species,
            "kind": kind,
            "description": description,
        }

    _add(
        "depmap", "DepMap 24Q4 gene/transcript expression + companions (~4.7 GB)", "bundle + build"
    )
    for name, ds in FETCHABLE_DATASETS.items():
        _add(name, ds["description"], "auto-fetch")
    for name, ds in MANUAL_DATASETS.items():
        _add(name, ds["description"], "manual download")
    for name, asset in data_assets().items():
        _add(name, asset["source"], "mirrored asset")
    for name, managed in managed_datasets().items():
        _add(name, managed.description, managed.cache_kind)

    return dict(sorted(entries.items()))


def available_datasets() -> dict[str, str]:
    """Return all known dataset names with descriptions."""
    return {
        name: f"{meta['description']} [{meta['kind']}]"
        for name, meta in available_dataset_catalog().items()
    }


def refresh(name: str, *, offline: bool | None = None) -> Path:
    """Re-download a fetchable dataset."""
    return fetch(name, force=True, offline=offline)


def verify_cache_files(*, include_unregistered: bool = False) -> dict[str, Any]:
    """Verify cached/registered files and trusted mirrored-asset hashes (#960)."""
    rows = list_cache_files(verify=True, include_unregistered=include_unregistered)
    failed_rows = [
        row
        for row in rows
        if row["status"] in ("corrupt", "invalid", "inaccessible")
        or (row["status"] == "missing" and "registered dataset" in row.get("kinds", []))
    ]
    verified_rows = [row for row in rows if row.get("verified")]
    return {
        "ok": len(failed_rows) == 0,
        "n_files": len(rows),
        "n_verified": len(verified_rows),
        "n_failed": len(failed_rows),
        "files": rows,
    }


def _tree_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            with contextlib.suppress(OSError):
                total += (Path(dirpath) / fname).stat().st_size
    return total


def _list_datacache_bundles(store_path: Path) -> list[Path]:
    """List bundle generation directories in a datacache store across datacache 1.30+ (#960)."""
    import datacache

    lister = getattr(datacache, "list_bundles", None)
    if lister is None:
        try:
            from datacache import bundle_store

            lister = getattr(bundle_store, "list_bundles", None)
        except Exception:
            lister = None
    if lister is not None:
        try:
            return [Path(p) for p in lister(store_path)]
        except Exception:
            return []
    # Fallback for datacache versions without public list_bundles: detect bundle directories
    # carrying .datacache-manifest.json under store_path.
    if not store_path.is_dir():
        return []
    found = [
        child
        for child in sorted(store_path.iterdir())
        if child.is_dir()
        and not child.name.startswith(".")
        and (child / ".datacache-manifest.json").is_file()
    ]
    return found


def _prune_datacache_bundles(store_path: Path, *, keep: int) -> list[Path]:
    """Invoke datacache.prune_bundles (or bundle_store.prune_bundles) cleanly across versions."""
    import datacache

    pruner = getattr(datacache, "prune_bundles", None)
    if pruner is None:
        try:
            from datacache import bundle_store

            pruner = getattr(bundle_store, "prune_bundles", None)
        except Exception:
            pruner = None
    if pruner is not None:
        return [Path(p) for p in pruner(store_path, keep=keep)]
    deleted = _list_datacache_bundles(store_path)[:-keep]
    for bundle in deleted:
        if bundle.is_dir() and not bundle.is_symlink():
            shutil.rmtree(bundle)
    return deleted


def prune_cache_bundles(
    *,
    keep: int = 1,
    apply: bool = False,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Plan or execute pruning of superseded datacache bundle generations (#960).

    Defaults to dry-run (``apply=False``). When ``apply=True``, invokes
    ``datacache.prune_bundles(store_path, keep=keep)`` under each store's lock.
    """
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be a positive integer; the current bundle is always kept")

    base = Path(root) if root is not None else data_dir()
    candidate_stores: list[Path] = []
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                bundles = _list_datacache_bundles(child)
                if bundles:
                    candidate_stores.append(child)

    stores_report: list[dict[str, Any]] = []
    total_prunable = 0
    total_deleted = 0
    total_reclaimable_bytes = 0

    for store_path in candidate_stores:
        bundles = _list_datacache_bundles(store_path)
        prunable = bundles[:-keep]
        kept = bundles[-keep:]
        store_reclaimable = sum(_tree_size_bytes(b) for b in prunable)
        total_prunable += len(prunable)
        total_reclaimable_bytes += store_reclaimable

        deleted_paths: list[str] = []
        if apply and prunable:
            deleted_paths = [str(p) for p in _prune_datacache_bundles(store_path, keep=keep)]
            total_deleted += len(deleted_paths)

        stores_report.append(
            {
                "store": str(store_path),
                "n_bundles": len(bundles),
                "n_kept": len(kept),
                "n_prunable": len(prunable),
                "n_deleted": len(deleted_paths),
                "reclaimable_bytes": store_reclaimable,
                "kept_bundles": [str(b) for b in kept],
                "prunable_bundles": [str(b) for b in prunable],
                "deleted_bundles": deleted_paths,
            }
        )

    return {
        "dry_run": not apply,
        "keep": keep,
        "root": str(base),
        "n_stores": len(stores_report),
        "n_prunable_bundles": total_prunable,
        "n_deleted_bundles": total_deleted,
        "reclaimable_bytes": total_reclaimable_bytes,
        "stores": stores_report,
    }


def remove(name: str, delete_file: bool = False) -> bool:
    """Remove a dataset from the registry. Optionally delete the file.

    Parameters
    ----------
    name
        Dataset name to unregister.
    delete_file
        If True, also delete the file on disk.

    Returns
    -------
    bool
        ``True`` if a dataset was registered or cached under *name* and removed,
        ``False`` if no such dataset existed (so callers can distinguish a real
        removal from a typo instead of silently reporting success).
    """
    with manifest_transaction() as manifest:
        entry = manifest.get("datasets", {}).pop(name, None)
    if entry is not None:
        if delete_file:
            p = Path(entry["path"])
            if p.exists():
                p.unlink()
                print(f"Deleted {p}")
        return True
    if name in data_assets():
        p = data_asset_dir() / name
        if not p.exists():
            return False
        if delete_file:
            p.unlink()
            print(f"Deleted {p}")
        return True
    from ._dataset_catalog import get_managed_dataset

    managed = get_managed_dataset(name)
    if managed is not None:
        return managed.remove_fn(delete_file)
    return False


# ── Versioned dataset registry ───────────────────────────────────────────────
#
# A reusable layer over ``download_to_file`` for datasets that are pinned to an
# explicit version (e.g. a particular upstream release). Adds a name->spec
# registry, per-version URLs with a pinned default, a JSON provenance manifest
# (sha256/bytes/url/downloaded_at), and a caller-supplied cache directory.
#
# Consumers register their own dataset definitions and point it at their own
# cache namespace, so the machinery lives in one place instead of being
# re-implemented per package. (tsarina's HPA/NCBI reference data uses this.)


class VersionedDatasetError(RuntimeError):
    """Unknown dataset/version, or a download failure, in a registry."""


class VersionedDatasetRegistry(VersionedFileRegistry):
    """Hitlist-compatible fixed-path registry backed by datacache.

    Keeps the filename/urls/default_version mapping, callable cache_dir,
    configurable error_cls, Path results and legacy root manifest/status keys.
    Existing files are reused in place. Shared datacache machinery owns version
    resolution, cache paths, writer locking and atomic receipt publication.
    """

    def __init__(self, datasets, *, cache_dir, error_cls=VersionedDatasetError):
        super().__init__(datasets, cache_dir=cache_dir, error_cls=error_cls)

    def download(
        self, name: str, version: str | None = None, *, force: bool = False, verbose: bool = True
    ) -> Path:
        # Snapshot a dynamic root once, including status messages and the
        # transfer/receipt transaction. The shared object carries no mutable
        # installation state; creating this bound view is inexpensive.
        version = self.resolve_version(name, version)
        root = Path(self._cache_dir())
        registry = VersionedFileRegistry(
            self._datasets, cache_dir=lambda: root, error_cls=self._error_cls
        )
        dest = registry.local_path(name, version)
        url = self._datasets[name]["urls"][version]
        cached = dest.exists() and not force
        if verbose:
            if cached:
                print(f"  [{name}] already cached ({dest.stat().st_size:,} bytes)")
            else:
                print(f"  [{name}] downloading from {url}")
        expand = any(
            url.lower().endswith(suffix) and not dest.name.lower().endswith(suffix)
            for suffix in (".gz", ".zip")
        )
        return registry.download(
            name,
            version,
            force=force,
            raw=not expand,
            decompress=expand,
            timeout=_DOWNLOAD_SOCKET_TIMEOUT,
            show_progress=verbose and sys.stderr.isatty(),
            record_provenance=True,
        )

    def ensure(self, name: str, version: str | None = None) -> Path:
        """Reuse silently, or download with the established status messages."""
        version = self.resolve_version(name, version)
        root = Path(self._cache_dir())
        registry = type(self)(self._datasets, cache_dir=lambda: root, error_cls=self._error_cls)
        path = registry.local_path(name, version)
        return path if path.exists() else registry.download(name, version)
