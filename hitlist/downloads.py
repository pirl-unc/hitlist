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
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from tqdm.auto import tqdm

# ── Download helper (timeout + retry) ────────────────────────────────────────
#
# UniProt / HPA / IEDB FASTAs and TSVs are fetched over plain HTTP.  We use
# ``urlopen(timeout=...)`` + ``shutil.copyfileobj`` rather than
# ``urlretrieve`` (which takes no timeout) so a stalled TCP connection raises
# ``socket.timeout`` instead of blocking forever — important now that the
# parallel mapping pre-fetch (#254) downloads serially in the orchestrator,
# where one hung connection would stall every worker.  See issue #255.

# Connect/read socket timeout for a single download operation, in seconds.
# This is a safety invariant, not a user-tuning knob: accepting arbitrary
# process-state values allowed NaN/inf/negative timeouts that either disabled
# the guard or crashed inside ``socket.settimeout``.  Mapping prefetch has a
# separate hard wall-clock deadline supervised from another process (#402).
_DOWNLOAD_SOCKET_TIMEOUT = 300.0

# Backoff (seconds) before each retry.  Its length is the retry count, so the
# total number of attempts is ``len(_DOWNLOAD_RETRY_BACKOFF) + 1``.  Tests
# monkeypatch this to disable sleeping.
_DOWNLOAD_RETRY_BACKOFF: tuple[float, ...] = (5.0, 30.0)

# Streaming read size for downloads — caps memory at one chunk (vs reading the
# whole response) and gives the progress bar a smooth update cadence.
_DOWNLOAD_CHUNK_SIZE = 1 << 16  # 64 KiB


def _download_to_file(url: str, dest: Path, *, label: str = "", verbose: bool = True) -> None:
    """Download ``url`` to ``dest`` atomically, with a timeout and retries.

    Streams the response into a sibling ``.tmp`` file and ``shutil.move``s it
    into place only on success, so a partial download never clobbers a good
    cached file.  Transient failures (timeouts, connection resets, transient
    HTTP errors) are retried with the backoff schedule in
    ``_DOWNLOAD_RETRY_BACKOFF``.  Raises ``RuntimeError`` if every attempt
    fails.
    """
    timeout = _DOWNLOAD_SOCKET_TIMEOUT
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    what = label or url
    last_err: Exception | None = None
    attempts = len(_DOWNLOAD_RETRY_BACKOFF) + 1
    try:
        for attempt in range(attempts):
            try:
                # Note: comma form, not parenthesized — parenthesized context
                # managers are a SyntaxError on Python 3.9, which we still
                # support.
                with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
                    # Real HTTP responses expose Content-Length via .headers;
                    # fall back to an indeterminate bar when it's absent.
                    headers = getattr(resp, "headers", None)
                    raw = headers.get("Content-Length") if headers is not None else None
                    total = int(raw) if raw and str(raw).isdigit() else None
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=what,
                        leave=False,
                        # Quiet under verbose=False and on non-TTYs (CI, pipes,
                        # the pytest capture) — tqdm with disable=True is a cheap
                        # pass-through, so the chunked copy stays the hot path.
                        disable=not verbose or not sys.stderr.isatty(),
                    ) as bar:
                        for chunk in iter(lambda: resp.read(_DOWNLOAD_CHUNK_SIZE), b""):
                            fh.write(chunk)
                            bar.update(len(chunk))
                # A clean 200 with an empty body (e.g. a withdrawn UniProt
                # proteome ID) would otherwise be moved into place and cached
                # as a permanent "valid" 0-byte file. Treat it as a retryable
                # OSError instead of silently succeeding.
                if tmp.stat().st_size == 0:
                    raise OSError(f"empty response body from {url}")
                shutil.move(str(tmp), str(dest))
                return
            except (urllib.error.URLError, OSError) as err:
                # socket.timeout is an OSError subclass, so a stalled
                # connection lands here instead of hanging forever.
                last_err = err
                if tmp.exists():
                    tmp.unlink()
                # A 4xx is a permanent client error (bad/withdrawn proteome
                # ID, wrong URL) — retrying just wastes the backoff window, so
                # fail fast.  Timeouts, connection resets, and 5xx are
                # transient and worth retrying.
                permanent = isinstance(err, urllib.error.HTTPError) and 400 <= err.code < 500
                if permanent or attempt >= len(_DOWNLOAD_RETRY_BACKOFF):
                    break
                backoff = _DOWNLOAD_RETRY_BACKOFF[attempt]
                if verbose:
                    print(
                        f"  [{what}] download failed ({err}); retrying in "
                        f"{backoff:g}s ({attempt + 1}/{len(_DOWNLOAD_RETRY_BACKOFF)})"
                    )
                time.sleep(backoff)
    finally:
        if tmp.exists():
            tmp.unlink()
    raise RuntimeError(f"Failed to download {url}: {last_err}") from last_err


def _is_compressed(url: str, dest: Path) -> bool:
    """True if *url* is a ``.zip``/``.gz`` archive to expand into *dest*.

    Mirrors datacache's heuristic: only decompress when *dest* doesn't itself
    carry the archive suffix, so a deliberately-kept ``foo.gz`` cache file is
    left compressed.
    """
    u = url.lower()
    name = dest.name.lower()
    return (u.endswith(".zip") and not name.endswith(".zip")) or (
        u.endswith(".gz") and not name.endswith(".gz")
    )


def _decompress_to(src: Path, dest: Path) -> None:
    """Expand a downloaded ``.zip``/``.gz`` *src* into *dest* atomically.

    Streams into a sibling ``.tmp`` then ``shutil.move``s it into place, so a
    partial/failed decompress never clobbers a good cached file.  ``.zip``
    archives extract the member matching ``dest.name`` if present, else the
    largest member (matching datacache's behaviour).  The compression kind is
    inferred from *src*'s suffix.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        if src.name.lower().endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                names = z.namelist()
                if not names:
                    raise RuntimeError(f"empty zip archive: {src}")
                member = (
                    dest.name
                    if dest.name in names
                    else max(z.infolist(), key=lambda i: i.file_size).filename
                )
                with z.open(member) as zf, open(tmp, "wb") as fh:
                    shutil.copyfileobj(zf, fh)
        else:  # .gz
            with gzip.open(src, "rb") as gz, open(tmp, "wb") as fh:
                shutil.copyfileobj(gz, fh)
        shutil.move(str(tmp), str(dest))
    finally:
        if tmp.exists():
            tmp.unlink()


def download_to_file(
    url: str,
    dest: Path | str,
    *,
    label: str = "",
    verbose: bool = True,
    force: bool = False,
    decompress: bool = False,
) -> Path:
    """Download *url* to *dest* with a progress bar and cache reporting.

    The reusable entry point behind hitlist's (and tsarina's) fetch commands —
    bundles cache reuse, status messaging, a streaming ``tqdm`` progress bar,
    and optional decompression.  Returns the local path.

    - Reuses a cached *dest* unless ``force``, printing a one-line cache-status
      message when ``verbose``.
    - Streams the transfer (chunked, never buffering the whole file in memory)
      with the timeout + retry + atomic-move semantics of the underlying
      downloader; the progress bar is suppressed on non-TTYs / ``verbose=False``.
    - When ``decompress`` and *url* is a ``.zip``/``.gz`` archive whose suffix
      *dest* doesn't carry, expands it into *dest* (streamed to disk).
    """
    dest = Path(dest)
    name = label or dest.name
    if dest.exists() and not force:
        if verbose:
            print(f"  [{name}] already cached ({dest.stat().st_size:,} bytes)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  [{name}] downloading from {url}")

    if decompress and _is_compressed(url, dest):
        # Fetch the archive alongside dest, then expand it in.  Both the
        # download and the decompress write through their own .tmp + move, so a
        # failure at either step leaves the prior cache (if any) intact.
        suffix = ".zip" if url.lower().endswith(".zip") else ".gz"
        archive = dest.with_name(dest.name + suffix)
        try:
            _download_to_file(url, archive, label=name, verbose=verbose)
            _decompress_to(archive, dest)
        finally:
            if archive.exists():
                archive.unlink()
    else:
        _download_to_file(url, dest, label=name, verbose=verbose)
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
DATA_DIR_ORIGINS = ("override", "env", "legacy", "default")

#: Environment variable that relocates :func:`data_dir`.
DATA_DIR_ENV_VAR = "HITLIST_DATA_DIR"

_override_data_dir: Path | None = None

#: Memoized ``{resolution inputs: (path, origin)}``.  ``data_dir()`` is called
#: thousands of times per process and rule 3 costs a directory scan, so the
#: answer is cached under the three inputs that can change it — all of them
#: in-process lookups, no syscalls.  See :func:`reset_data_dir_cache`.
_data_dir_cache: dict[tuple[str | None, str, str], tuple[Path, str]] = {}

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


def data_asset_dir() -> Path:
    """Directory :func:`fetch_data_asset` caches mirrored data assets in (#303).

    This is datacache's own cache dir for the ``hitlist`` subdir, asked of
    datacache so it cannot drift from what ``datacache.fetch_file`` actually
    writes to.  datacache resolves a fetch destination with no ``envkey`` (see
    ``datacache.expected_path``), so — unlike :func:`data_dir` — this one is
    *not* moved by ``$HITLIST_DATA_DIR``.
    """
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


def _load_manifest() -> dict:
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
    # concurrent _load_manifest() reader always sees a COMPLETE file (old or new),
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


# ── Known datasets ──────────────────────────────────────────────────────────

FETCHABLE_DATASETS: dict[str, dict[str, str]] = {
    # Viral proteomes (auto-downloadable from UniProt REST API)
    "hpv16": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000006729&format=fasta",
        "filename": "hpv16.fasta",
        "description": "HPV-16 proteome (UniProt UP000006729)",
        "usage": "Peptide generation for cervical/oropharyngeal cancer viral targets",
    },
    "hpv18": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000006728&format=fasta",
        "filename": "hpv18.fasta",
        "description": "HPV-18 proteome (UniProt UP000006728)",
        "usage": "Peptide generation for cervical cancer viral targets",
    },
    "ebv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000153037&format=fasta",
        "filename": "ebv.fasta",
        "description": "EBV/HHV-4 proteome (UniProt UP000153037)",
        "usage": "Peptide generation for Burkitt lymphoma, NPC, Hodgkin lymphoma",
    },
    "htlv1": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000002063&format=fasta",
        "filename": "htlv1.fasta",
        "description": "HTLV-1 proteome (UniProt UP000002063)",
        "usage": "Peptide generation for adult T-cell leukemia/lymphoma",
    },
    "hbv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000126453&format=fasta",
        "filename": "hbv.fasta",
        "description": "HBV proteome (UniProt UP000126453)",
        "usage": "Peptide generation for hepatocellular carcinoma",
    },
    "hcv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000518&format=fasta",
        "filename": "hcv.fasta",
        "description": "HCV proteome (UniProt UP000000518)",
        "usage": "Peptide generation for hepatocellular carcinoma, B-cell lymphoma",
    },
    "kshv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000009113&format=fasta",
        "filename": "kshv.fasta",
        "description": "KSHV/HHV-8 proteome (UniProt UP000009113)",
        "usage": "Peptide generation for Kaposi sarcoma",
    },
    "mcpyv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000116695&format=fasta",
        "filename": "mcpyv.fasta",
        "description": "MCPyV proteome (UniProt UP000116695)",
        "usage": "Peptide generation for Merkel cell carcinoma",
    },
    "hiv1": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000002241&format=fasta",
        "filename": "hiv1.fasta",
        "description": "HIV-1 proteome (UniProt UP000002241)",
        "usage": "Peptide generation for Kaposi sarcoma, lymphoma (indirect)",
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

_UNIPROT_PROTEOME_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=proteome:{proteome_id}&format=fasta&compressed=false"
)


SPECIES_PROTEOMES: dict[str, dict[str, str | int]] = {
    # Ensembl-supported (pyensembl)
    "Homo sapiens": {"kind": "ensembl", "release": 112, "species": "human"},
    "Mus musculus": {"kind": "ensembl", "release": 112, "species": "mouse"},
    "Rattus norvegicus": {"kind": "ensembl", "release": 112, "species": "rat"},
    # UniProt reference proteomes (auto-downloaded)
    "Sarcophilus harrisii": {"kind": "uniprot", "proteome_id": "UP000007648"},
    "Canis lupus": {"kind": "uniprot", "proteome_id": "UP000002254"},
    "Bos taurus": {"kind": "uniprot", "proteome_id": "UP000009136"},
    "Gallus gallus": {"kind": "uniprot", "proteome_id": "UP000000539"},
    "Sus scrofa": {"kind": "uniprot", "proteome_id": "UP000008227"},
    "Macaca mulatta": {"kind": "uniprot", "proteome_id": "UP000006718"},
    "Equus caballus": {"kind": "uniprot", "proteome_id": "UP000002281"},
    "Pan troglodytes": {"kind": "uniprot", "proteome_id": "UP000002277"},
    "Trichosurus vulpecula": {"kind": "uniprot", "proteome_id": "UP000504604"},
    # IEDB genus-abbreviated names — treat as the type species
    "Sus sp.": {"kind": "uniprot", "proteome_id": "UP000008227"},  # → Sus scrofa
    "Canis sp.": {"kind": "uniprot", "proteome_id": "UP000002254"},  # → Canis lupus
    "Bos sp.": {"kind": "uniprot", "proteome_id": "UP000009136"},  # → Bos taurus
    "Rattus sp.": {"kind": "ensembl", "release": 112, "species": "rat"},
    # Parasites / plants / other with curated UPIDs
    "Theileria parva": {"kind": "uniprot", "proteome_id": "UP000001949"},
    "Ascaris suum": {"kind": "uniprot", "proteome_id": "UP000036681"},
    "Ascaris lumbricoides": {"kind": "uniprot", "proteome_id": "UP000036681"},
    # Bacteria — canonical reference strains.
    "Mycobacterium tuberculosis": {"kind": "uniprot", "proteome_id": "UP000001584"},  # H37Rv
    "Mycobacterium tuberculosis H37Rv": {"kind": "uniprot", "proteome_id": "UP000001584"},
    # Apicomplexan parasites — 3D7 is the canonical P. falciparum
    # reference used by virtually all immunology and antimalarial
    # epitope work.
    "Plasmodium falciparum": {"kind": "uniprot", "proteome_id": "UP000001450"},  # 3D7
}


# Viral proteomes keyed by the IEDB ``source_organism`` string we observe.
# For matching, we lowercase both sides and check substring inclusion of the
# registry key.  This tolerates IEDB variations (e.g. "Epstein-Barr virus
# (strain B95-8)" matches "epstein-barr virus").
VIRAL_PROTEOMES: dict[str, dict[str, str]] = {
    # SARS coronaviruses — "sars-cov" must come before "sars-cov-2" check below
    # but we sort by key length at lookup time to match most-specific first
    "severe acute respiratory syndrome coronavirus 2": {
        "proteome_id": "UP000464024",
        "key": "sars-cov-2",
    },
    "sars-cov2": {"proteome_id": "UP000464024", "key": "sars-cov-2"},
    "sars-cov-2": {"proteome_id": "UP000464024", "key": "sars-cov-2"},
    "severe acute respiratory syndrome coronavirus": {
        "proteome_id": "UP000000354",
        "key": "sars-cov-1",
    },
    "sars-cov1": {"proteome_id": "UP000000354", "key": "sars-cov-1"},
    "sars-cov-1": {"proteome_id": "UP000000354", "key": "sars-cov-1"},
    "sars coronavirus": {"proteome_id": "UP000000354", "key": "sars-cov-1"},
    # Herpes viruses
    "human immunodeficiency virus 1": {"proteome_id": "UP000002241", "key": "hiv1"},
    "hiv-1": {"proteome_id": "UP000002241", "key": "hiv1"},
    "epstein-barr virus": {"proteome_id": "UP000153037", "key": "ebv"},
    "human gammaherpesvirus 4": {"proteome_id": "UP000153037", "key": "ebv"},
    "human herpesvirus 4": {"proteome_id": "UP000153037", "key": "ebv"},
    "human betaherpesvirus 5": {"proteome_id": "UP000000938", "key": "hcmv"},
    "human cytomegalovirus": {"proteome_id": "UP000000938", "key": "hcmv"},
    "human herpesvirus 5": {"proteome_id": "UP000000938", "key": "hcmv"},
    "human gammaherpesvirus 8": {"proteome_id": "UP000009113", "key": "kshv"},
    "human herpesvirus 8": {"proteome_id": "UP000009113", "key": "kshv"},
    "kaposi": {"proteome_id": "UP000009113", "key": "kshv"},
    "human alphaherpesvirus 1": {"proteome_id": "UP000009294", "key": "hsv-1"},
    "herpes simplex virus type 1": {"proteome_id": "UP000009294", "key": "hsv-1"},
    "human herpesvirus 1": {"proteome_id": "UP000009294", "key": "hsv-1"},
    "human alphaherpesvirus 2": {"proteome_id": "UP000001874", "key": "hsv-2"},
    "herpes simplex virus type 2": {"proteome_id": "UP000001874", "key": "hsv-2"},
    "human herpesvirus 2": {"proteome_id": "UP000001874", "key": "hsv-2"},
    "human betaherpesvirus 6b": {"proteome_id": "UP000006930", "key": "hhv-6b"},
    "human herpesvirus 6b": {"proteome_id": "UP000006930", "key": "hhv-6b"},
    "murid betaherpesvirus 1": {"proteome_id": "UP000008774", "key": "mcmv"},
    "murid herpesvirus 1": {"proteome_id": "UP000008774", "key": "mcmv"},
    "murine cytomegalovirus": {"proteome_id": "UP000008774", "key": "mcmv"},
    # Hepatitis
    "hepatitis b virus": {"proteome_id": "UP000126453", "key": "hbv"},
    "hepatitis c virus": {"proteome_id": "UP000000518", "key": "hcv"},
    "hepacivirus hominis": {"proteome_id": "UP000000518", "key": "hcv"},
    # Papillomaviruses
    "human papillomavirus type 16": {"proteome_id": "UP000006729", "key": "hpv16"},
    "human papillomavirus 16": {"proteome_id": "UP000006729", "key": "hpv16"},
    "human papillomavirus type 18": {"proteome_id": "UP000006728", "key": "hpv18"},
    "human papillomavirus 18": {"proteome_id": "UP000006728", "key": "hpv18"},
    # Influenza
    "influenza a virus": {"proteome_id": "UP000009255", "key": "influenza-a"},
    "influenza b virus": {"proteome_id": "UP000008158", "key": "influenza-b"},
    # Poxviruses
    "vaccinia virus": {"proteome_id": "UP000000344", "key": "vaccinia"},
    "orf virus": {"proteome_id": "UP000000870", "key": "orf"},
    # Animal pathogens
    "canine distemper virus": {"proteome_id": "UP000117312", "key": "cdv"},
    "african swine fever virus": {"proteome_id": "UP000000624", "key": "asfv"},
    "porcine reproductive and respiratory syndrome virus": {
        "proteome_id": "UP000006706",
        "key": "prrsv",
    },
    "wobbly possum disease virus": {"proteome_id": "UP000147130", "key": "wpdv"},
    "peste-des-petits-ruminants virus": {"proteome_id": "UP000100083", "key": "pprv"},
    # Respiratory / other human viruses
    "human respiratory syncytial virus": {"proteome_id": "UP000002472", "key": "rsv"},
    "human orthopneumovirus": {"proteome_id": "UP000002472", "key": "rsv"},
    "human metapneumovirus": {"proteome_id": "UP000001398", "key": "hmpv"},
    "zika virus": {"proteome_id": "UP000054557", "key": "zika"},
    "rotavirus a": {"proteome_id": "UP000001119", "key": "rotavirus-a"},
    "lymphocytic choriomeningitis virus": {"proteome_id": "UP000002474", "key": "lcmv"},
    # Polyomaviruses
    "betapolyomavirus hominis": {"proteome_id": "UP000008475", "key": "bkv"},
    "bk polyomavirus": {"proteome_id": "UP000008475", "key": "bkv"},
    "alphapolyomavirus muris": {"proteome_id": "UP000007212", "key": "mpyv"},
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
    "adeno-associated virus 2": {"proteome_id": "UP000180764", "key": "aav-2"},
    "adeno-associated virus 6": {"proteome_id": "UP000119472", "key": "aav-6"},
    "adeno-associated virus - 6": {"proteome_id": "UP000119472", "key": "aav-6"},
    "adeno-associated virus 8": {"proteome_id": "UP000201958", "key": "aav-8"},
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


def resolve_proteome_via_uniprot(
    organism: str,
    timeout: int = 15,
) -> dict | None:
    """Query UniProt REST to find the best reference proteome for an organism.

    Returns a dict with ``proteome_id``, ``scientific_name``, ``taxon_id``,
    ``proteome_type``, and ``protein_count`` fields.  Prefers "Reference
    and representative" over plain "Representative" proteomes.  Returns
    ``None`` if no match is found or if the organism is on the denylist
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

    def sort_key(p: dict) -> tuple:
        ptype = (p.get("proteomeType") or "").lower()
        rank = _PROTEOME_TYPE_RANK.get(ptype, 99)
        # Within same rank, prefer lower taxon id (parent species often
        # has a smaller ID than strain-specific entries) and higher count
        tax_id = int(p.get("taxonomy", {}).get("taxonId") or 1_000_000_000)
        count = int(p.get("proteinCount") or 0)
        return (rank, tax_id, -count)

    best = min(results, key=sort_key)
    tax = best.get("taxonomy", {})
    return {
        "proteome_id": best.get("id"),
        "scientific_name": tax.get("scientificName") or organism,
        "taxon_id": tax.get("taxonId"),
        "proteome_type": best.get("proteomeType"),
        "protein_count": best.get("proteinCount"),
    }


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
    manifest = _load_manifest()
    return manifest.get("uniprot_resolutions", {})


def _save_uniprot_cache_entry(organism: str, entry: dict | None) -> None:
    manifest = _load_manifest()
    cache = manifest.setdefault("uniprot_resolutions", {})
    cache[organism] = {
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        **(entry or {"not_found": True}),
    }
    _save_manifest(manifest)


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
            return {
                "kind": "uniprot",
                "proteome_id": viral_entry["proteome_id"],
                "canonical_species": species_or_organism,
                "key": viral_entry["key"],
            }

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
    manifest = _load_manifest()
    proteomes = manifest.setdefault("proteomes", {})

    # Ensembl species: pyensembl manages its own cache — no FASTA to download
    if entry["kind"] == "ensembl":
        proteomes[canonical] = {
            "kind": "ensembl",
            "species": entry.get("species", canonical),
            "release": entry.get("release", 112),
            "registered": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest)
        return None  # caller should use ProteomeIndex.from_ensembl()

    # UniProt species: download the FASTA (dedup by UPID — multiple strain
    # variants often resolve to the same UniProt reference proteome)
    proteome_id = entry["proteome_id"]
    url = _UNIPROT_PROTEOME_URL.format(proteome_id=proteome_id)

    existing = _find_existing_proteome_by_upid(proteomes, proteome_id)
    if existing is not None and Path(existing["path"]).exists() and not force:
        dest = Path(existing["path"])
        if verbose:
            print(
                f"  [{canonical}] reusing cached FASTA from "
                f"{existing.get('canonical_species', '?')} ({dest.stat().st_size:,} bytes)"
            )
        proteomes[canonical] = {
            "kind": "uniprot",
            "proteome_id": proteome_id,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "source_url": url,
            "canonical_species": canonical,
            "registered": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest)
        return dest

    fname = _safe_filename(canonical)
    dest = _proteomes_dir() / fname

    if dest.exists() and not force:
        if verbose:
            print(f"  [{canonical}] already cached ({dest.stat().st_size:,} bytes)")
        proteomes[canonical] = {
            "kind": "uniprot",
            "proteome_id": proteome_id,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "source_url": url,
            "canonical_species": canonical,
            "registered": proteomes.get(canonical, {}).get(
                "registered", datetime.now(timezone.utc).isoformat()
            ),
        }
        _save_manifest(manifest)
        return dest

    if not fetch_missing:
        return None

    if verbose:
        print(f"  [{canonical}] fetching UniProt {proteome_id} ...")
    _download_to_file(url, dest, label=canonical, verbose=verbose)

    size = dest.stat().st_size
    if verbose:
        print(f"  [{canonical}] downloaded {size:,} bytes → {dest}")

    proteomes[canonical] = {
        "kind": "uniprot",
        "proteome_id": proteome_id,
        "path": str(dest),
        "size_bytes": size,
        "source_url": url,
        "registered": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
    return dest


def list_proteomes() -> dict:
    """Return the proteomes section of the manifest."""
    return _load_manifest().get("proteomes", {})


def fetch_proteome_by_upid(
    upid: str,
    label: str | None = None,
    force: bool = False,
    verbose: bool = True,
    fetch_missing: bool = True,
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
    """
    if not upid:
        return None
    manifest = _load_manifest()
    proteomes = manifest.setdefault("proteomes", {})

    # Dedup by UPID if we already have this proteome under another key
    existing = _find_existing_proteome_by_upid(proteomes, upid)
    if existing is not None and Path(existing["path"]).exists() and not force:
        path = Path(existing["path"])
        if verbose:
            print(f"  [{label or upid}] reusing cached {path.name} ({path.stat().st_size:,} bytes)")
        return path

    name = label or upid
    fname = _safe_filename(name)
    dest = _proteomes_dir() / fname
    url = _UNIPROT_PROTEOME_URL.format(proteome_id=upid)

    if dest.exists() and not force:
        if verbose:
            print(f"  [{name}] already cached ({dest.stat().st_size:,} bytes)")
    else:
        if not fetch_missing:
            return None
        if verbose:
            print(f"  [{name}] fetching UniProt {upid} ...")
        _download_to_file(url, dest, label=name, verbose=verbose)
        if verbose:
            print(f"  [{name}] downloaded {dest.stat().st_size:,} bytes → {dest}")

    proteomes[name] = {
        "kind": "uniprot",
        "proteome_id": upid,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "source_url": url,
        "canonical_species": name,
        "registered": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
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


def fetch_data_asset(filename: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Fetch a mirrored data asset into the datacache cache dir (#303).

    Uses ``datacache.fetch_file`` (openvax ecosystem cache; #291) so the file is
    stored under ``datacache.get_data_dir('hitlist')`` and reused across runs.

    Every guard is delegated rather than re-implemented (#590):

    - ``timeout`` is :data:`_DOWNLOAD_SOCKET_TIMEOUT`, the same bound the
      hand-rolled downloader has carried since #255/#402. Without it
      ``fetch_file`` defaults to ``timeout=None`` and a stalled TCP connection
      hangs with no wall-clock limit -- the exact failure those issues were
      filed for, reintroduced on this newer path.
    - ``expected_sha256`` validates the staged bytes *before* they are published
      into the cache, and revalidates a cache hit without re-hashing the file
      here. The previous hand-rolled loop published a corrupt file, noticed, and
      re-fetched; it also re-hashed all 26 assets on every
      :func:`fetch_all_data_assets` call. A mismatch now raises
      ``datacache.FileValidationError`` naming the path and reason; ``force=True``
      is the documented repair.
    - ``show_progress`` reaches the assets that are never packaged and therefore
      always downloaded (#341), where silence reads as a hang.
    """
    assets = _data_assets_registry()["assets"]
    if filename not in assets:
        raise KeyError(f"unknown data asset {filename!r}; known: {sorted(assets)}")
    import datacache

    meta = assets[filename]
    if verbose:
        print(f"Fetching {filename} ({meta['source']}) via datacache...")
    return Path(
        datacache.fetch_file(
            meta["url"],
            filename=filename,
            subdir="hitlist",
            force=force,
            timeout=_DOWNLOAD_SOCKET_TIMEOUT,
            expected_sha256=meta["sha256"],
            show_progress=verbose,
        )
    )


def fetch_all_data_assets(*, force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Fetch (and checksum-verify) EVERY mirrored data asset into the cache dir.

    Backs the ``hitlist data fetch-all`` command — one call to pull the complete
    set of paper-derived CSVs from the GitHub data-assets release. Returns a
    ``{filename: cached_path}`` map. Re-uses cached copies unless ``force``.
    """
    assets = _data_assets_registry()["assets"]
    out: dict[str, Path] = {}
    for i, filename in enumerate(sorted(assets), 1):
        if verbose:
            print(f"[{i}/{len(assets)}] {filename}")
        out[filename] = fetch_data_asset(filename, force=force, verbose=verbose)
    if verbose and out:
        total = sum(p.stat().st_size for p in out.values())
        cache_dir = next(iter(out.values())).parent
        print(f"Done — {len(out)} assets cached in {cache_dir} ({total / 1e6:.1f} MB).")
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


def register(name: str, path: str | Path, description: str | None = None) -> Path:
    """Register a local file path for a named dataset."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    manifest = _load_manifest()
    desc = description
    if desc is None and name in MANUAL_DATASETS:
        desc = MANUAL_DATASETS[name]["description"]
    if desc is None and name in FETCHABLE_DATASETS:
        desc = FETCHABLE_DATASETS[name]["description"]

    manifest["datasets"][name] = {
        "path": str(p),
        "registered": datetime.now(timezone.utc).isoformat(),
        "size_bytes": p.stat().st_size,
        "description": desc or "",
        "source": "registered",
    }
    _save_manifest(manifest)
    return p


def _depmap_file_is_registered(key: str) -> bool:
    registered = _load_manifest().get("datasets", {}).get(key, {})
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


def fetch(name: str, force: bool = False) -> Path:
    """Download a fetchable dataset."""
    if name == "depmap":
        from .builder import build_line_expression

        for key in _DEPMAP_FILES:
            if not force and _depmap_file_is_registered(key):
                continue
            fetch(key, force=force)
        build_line_expression(verbose=True)
        output = data_dir() / "line_expression.parquet"
        register("depmap", output, description="DepMap 24Q4 line-expression index")
        return output
    if name not in FETCHABLE_DATASETS:
        if name in MANUAL_DATASETS:
            info = MANUAL_DATASETS[name]
            raise ValueError(
                f"'{name}' requires manual download from:\n"
                f"  {info['download_url']}\n"
                f"Then register the downloaded {info['expected_filename']} file "
                f"under dataset name '{name}'."
            )
        available = sorted(available_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    ds = FETCHABLE_DATASETS[name]
    dest = data_dir() / ds["filename"]

    if dest.exists() and not force:
        manifest = _load_manifest()
        if name not in manifest.get("datasets", {}):
            register(name, dest, ds["description"])
        return dest

    print(f"Downloading {ds['description']}...")
    # decompress=True unzips/gunzips the IEDB/CEDAR archives into their CSV; it
    # is a no-op for the plain-FASTA viral proteomes (download_to_file only
    # decompresses when the URL is a .zip/.gz the dest filename doesn't carry).
    download_to_file(ds["url"], dest, label=name, force=force, decompress=True)
    if ds.get("terms"):
        print(
            f"  [{name}] from a terms-of-use-governed source — please review "
            f"usage/citation terms: {ds['terms']}",
            file=sys.stderr,
        )

    manifest = _load_manifest()
    manifest["datasets"][name] = {
        "path": str(dest),
        "registered": datetime.now(timezone.utc).isoformat(),
        "size_bytes": dest.stat().st_size,
        "description": ds["description"],
        "source": ds["url"],
    }
    _save_manifest(manifest)
    print(f"  Saved to {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def get_path(name: str) -> Path:
    """Resolve a dataset name to its local file path."""
    manifest = _load_manifest()
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
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
    manifest = _load_manifest()
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
        # Return known info even if not registered
        if name == "depmap":
            return {
                "description": "DepMap 24Q4 gene/transcript expression and index (~4.7 GB)",
                "datasets": list(_DEPMAP_FILES),
                "status": "not installed",
                "type": "bundle + build",
            }
        if name in FETCHABLE_DATASETS:
            return {**FETCHABLE_DATASETS[name], "status": "not installed", "type": "auto-fetch"}
        if name in MANUAL_DATASETS:
            return {**MANUAL_DATASETS[name], "status": "not installed", "type": "manual download"}
        raise KeyError(f"Unknown dataset '{name}'")
    result = dict(entry)
    result["status"] = "installed"
    if name in FETCHABLE_DATASETS:
        result["type"] = "auto-fetch"
        result["usage"] = FETCHABLE_DATASETS[name].get("usage", "")
    elif name in MANUAL_DATASETS:
        result["type"] = "manual download"
        result["usage"] = MANUAL_DATASETS[name].get("usage", "")
    return result


def list_datasets() -> dict[str, dict]:
    """Return all registered/fetched datasets."""
    return dict(_load_manifest().get("datasets", {}))


def available_datasets() -> dict[str, str]:
    """Return all known dataset names with descriptions."""
    result = {
        "depmap": "DepMap 24Q4 gene/transcript expression + companions (~4.7 GB) [bundle + build]"
    }
    for name, ds in FETCHABLE_DATASETS.items():
        result[name] = ds["description"] + " [auto-fetch]"
    for name, ds in MANUAL_DATASETS.items():
        result[name] = ds["description"] + " [manual download]"
    return result


def refresh(name: str) -> Path:
    """Re-download a fetchable dataset."""
    return fetch(name, force=True)


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
        ``True`` if a dataset was registered under *name* and removed,
        ``False`` if no such dataset existed (so callers can distinguish a real
        removal from a typo instead of silently reporting success).
    """
    manifest = _load_manifest()
    entry = manifest.get("datasets", {}).pop(name, None)
    if entry is None:
        return False
    _save_manifest(manifest)
    if delete_file:
        p = Path(entry["path"])
        if p.exists():
            p.unlink()
            print(f"Deleted {p}")
    return True


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


class VersionedDatasetRegistry:
    """Download + cache for versioned, version-pinned external datasets.

    Parameters
    ----------
    datasets
        Mapping of ``name -> spec`` where each spec has::

            {
                "filename": "local_name.tsv",      # name on disk (post-decompress)
                "urls": {"v23": "https://...zip", "latest": "https://..."},
                "default_version": "v23",          # used when caller passes version=None
                "description": "...",              # optional, for status()
            }

    cache_dir
        Zero-arg callable returning the cache root :class:`~pathlib.Path`
        (created on demand by the caller). The on-disk layout is
        ``<cache>/<name>/<version>/<filename>`` plus a ``<cache>/manifest.json``
        provenance file.
    error_cls
        Exception type raised for unknown datasets/versions and download
        failures. Defaults to :class:`VersionedDatasetError`; consumers may pass
        their own subclass to preserve their public error type.
    """

    def __init__(self, datasets, *, cache_dir, error_cls=VersionedDatasetError):
        self._datasets = datasets
        self._cache_dir = cache_dir
        self._error_cls = error_cls

    # -- dataset / version resolution --

    def _dataset(self, name: str) -> dict:
        try:
            return self._datasets[name]
        except KeyError:
            known = ", ".join(sorted(self._datasets))
            raise self._error_cls(f"unknown dataset {name!r}; known: {known}") from None

    def resolve_version(self, name: str, version: str | None = None) -> str:
        """Return the concrete version for *name*, applying its default."""
        spec = self._dataset(name)
        if version is None:
            version = spec["default_version"]
        if version not in spec["urls"]:
            avail = ", ".join(sorted(spec["urls"]))
            raise self._error_cls(f"{name!r} has no version {version!r}; available: {avail}")
        return version

    # -- cache paths / manifest --

    def _manifest_path(self) -> Path:
        return self._cache_dir() / "manifest.json"

    def _read_manifest(self) -> dict:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_manifest(self, manifest: dict) -> None:
        # Atomic write (temp + os.replace), same as the module-level
        # _save_manifest (#331): an interrupted/concurrent write must never
        # truncate manifest.json, or _read_manifest silently returns {} and all
        # provenance (sha256/bytes/url) vanishes.
        p = self._manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, str(p))
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def local_path(self, name: str, version: str | None = None) -> Path:
        """Expected cache path for *name*/*version* (may not exist yet)."""
        version = self.resolve_version(name, version)
        spec = self._dataset(name)
        return self._cache_dir() / name / version / spec["filename"]

    def is_cached(self, name: str, version: str | None = None) -> bool:
        return self.local_path(name, version).exists()

    # -- fetch --

    def download(
        self, name: str, version: str | None = None, *, force: bool = False, verbose: bool = True
    ) -> Path:
        """Download *name*/*version* into the cache and record it in the manifest.

        A cached copy is reused unless ``force``. The transfer + ``.zip``/``.gz``
        decompression are delegated to :func:`download_to_file`.
        """
        version = self.resolve_version(name, version)
        spec = self._dataset(name)
        dest = self.local_path(name, version)
        url = spec["urls"][version]

        was_cached = dest.exists() and not force
        try:
            download_to_file(url, dest, label=name, verbose=verbose, force=force, decompress=True)
        except Exception as e:  # surface network/HTTP/decompress failures uniformly
            raise self._error_cls(f"failed to download {name} ({url}): {e}") from e

        # A cache hit needs no manifest churn (and no fresh sha256 of a large
        # file); download_to_file already printed the cache-status line.
        if was_cached:
            return dest

        manifest = self._read_manifest()
        manifest[name] = {
            "version": version,
            "url": url,
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write_manifest(manifest)
        return dest

    def ensure(self, name: str, version: str | None = None) -> Path:
        """Return a local path to *name*/*version*, downloading if absent."""
        path = self.local_path(name, version)
        return path if path.exists() else self.download(name, version)

    def status(self) -> list[dict]:
        """Return one status row per dataset (for a ``... list`` CLI command)."""
        manifest = self._read_manifest()
        rows = []
        for name, spec in sorted(self._datasets.items()):
            default_v = spec["default_version"]
            path = self._cache_dir() / name / default_v / spec["filename"]
            record = manifest.get(name, {})
            rows.append(
                {
                    "name": name,
                    "description": spec.get("description", ""),
                    "default_version": default_v,
                    "available_versions": sorted(spec["urls"]),
                    "cached": path.exists(),
                    "cached_version": record.get("version") if record else None,
                    "bytes": record.get("bytes") if path.exists() else None,
                    "downloaded_at": record.get("downloaded_at") if path.exists() else None,
                    "path": str(path),
                }
            )
        return rows
