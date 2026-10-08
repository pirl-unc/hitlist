"""Release-pinned UniProt FASTAs with verified, bounded, opt-in caching.

Reference identity is not evidence that a study searched that reference.
The catalog describes sequence assets; search contracts remain study-specific.
"""

from __future__ import annotations

import copy
import os
import re
import stat
import tempfile
from pathlib import Path

import datacache
from filelock import FileLock

from .curation_yaml import load_curation_yaml

DEFAULT_COLLECTION = "human"
DEFAULT_MAX_ASSET_BYTES = 256 * 2**20
DEFAULT_MAX_CACHE_BYTES = 2**30
# Conservative reserve for transfer state and atomic provenance publication.
_TRANSFER_HEADROOM_BYTES = 2 * 2**20
_CHUNK_BYTES = 2**20
_CATALOG_PATH = Path(__file__).parent / "data" / "uniprot_references.yaml"


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _component(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value) is None:
        raise ValueError(f"Invalid UniProt cache path component: {value!r}")
    return value


def uniprot_catalog() -> dict:
    """Return the bundled, pinned catalog; never contact UniProt or create a cache."""
    catalog = load_curation_yaml(_CATALOG_PATH)
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported UniProt catalog schema")
    notice = catalog.get("license_statement", "")
    if not isinstance(notice, str) or len(notice.encode()) > 65536:
        raise ValueError("UniProt license statement must be text of at most 64 KiB")
    for name, collection in catalog["collections"].items():
        _component(name)
        if collection["default_release"] not in collection["releases"]:
            raise ValueError(f"Missing default release for UniProt collection {name}")
        for release, entry in collection["releases"].items():
            if re.fullmatch(r"\d{4}_\d{2}", release) is None:
                raise ValueError(f"UniProt releases must be explicit YYYY_NN versions: {release}")
            _component(entry["filename"])
            if not entry["filename"].endswith(".fasta"):
                raise ValueError("Managed UniProt assets must be plain FASTA files")
            _positive_integer("size_bytes", entry["size_bytes"])
            if re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
                raise ValueError("UniProt assets require a pinned SHA256")
            if not entry["url"].startswith("https://"):
                raise ValueError("UniProt assets require an HTTPS download URL")
    return catalog


def uniprot_cache_dir() -> Path:
    """Cache root under ``data_dir()``; read-only and compatible with relocation."""
    from .downloads import data_dir

    return data_dir() / "uniprot"


def _reference(release, collection):
    catalog = uniprot_catalog()
    collections = catalog["collections"]
    if collection not in collections:
        raise ValueError(
            f"Unknown UniProt collection {collection!r}; available: {sorted(collections)}"
        )
    spec = collections[collection]
    release = spec["default_release"] if release is None else release
    if release not in spec["releases"]:
        raise ValueError(
            f"Unknown UniProt release {release!r} for {collection}; available: {sorted(spec['releases'])}"
        )
    return {
        **copy.deepcopy(spec["releases"][release]),
        "collection": collection,
        "release": release,
        "default_release": spec["default_release"],
        "taxonomy_id": spec["taxonomy_id"],
        "selection": copy.deepcopy(spec["selection"]),
        "license_statement": catalog.get("license_statement", ""),
    }


def _path(reference):
    return (
        uniprot_cache_dir() / reference["collection"] / reference["release"] / reference["filename"]
    )


def _check_path(path):
    # A configured data directory may itself be a link. Within our managed
    # subtree, links could escape the budget or redirect deletion/publication.
    root = uniprot_cache_dir()
    for relative in reversed(path.relative_to(root).parents):
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"Symlinks are not supported in the UniProt cache: {candidate}")
    if root.is_symlink() or path.is_symlink():
        raise ValueError(f"Symlinks are not supported in the UniProt cache: {path}")


def uniprot_info(release=None, *, collection=DEFAULT_COLLECTION, verify=False) -> dict:
    """Describe a release and its local state, without downloading or writing.

    ``verify=True`` checks the trusted SHA256. Otherwise only size is checked;
    a historical receipt is never reported as a fresh integrity verification.
    """
    reference = _reference(release, collection)
    path = _path(reference)
    try:
        _check_path(path)
    except ValueError as error:
        return {
            **reference,
            "path": str(path),
            "status": "corrupt",
            "verified": False,
            "error": str(error),
        }
    state = datacache.inspect_file(
        path,
        expected_size=reference["size_bytes"],
        expected_sha256=reference["sha256"] if verify else None,
    )
    return {
        **reference,
        "path": str(path),
        "status": state.status,
        "verified": state.verified,
        "fetched_at": state.fetched_at,
        "error": str(state.error) if state.error else None,
    }


def list_uniprot_references(*, collection=None, verify=False) -> list[dict]:
    """List every catalog release, including absent and corrupt local versions."""
    collections = uniprot_catalog()["collections"]
    if collection is not None and collection not in collections:
        raise ValueError(f"Unknown UniProt collection {collection!r}")
    return [
        uniprot_info(release, collection=name, verify=verify)
        for name, spec in sorted(collections.items())
        if collection is None or name == collection
        for release in sorted(spec["releases"])
    ]


def uniprot_path(release=None, *, collection=DEFAULT_COLLECTION) -> Path:
    """Return an existing, checksum-verified FASTA. Never download implicitly."""
    reference = _reference(release, collection)
    path = _path(reference)
    _check_path(path)
    datacache.validate_file(path, reference["sha256"], reference["size_bytes"])
    return path


def _cache_bytes():
    """Count all regular bytes, including hidden partials and unrelated files.

    Refuse unknown file types and traversal errors rather than undercounting.
    This is a logical byte budget, not a filesystem block/quota measurement.
    """
    root = uniprot_cache_dir()
    _check_path(root)
    total = 0

    def fail(error):
        raise error

    for directory, dirs, files in os.walk(root, onerror=fail):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Unsupported file type in UniProt cache: {path}")
    return total


def _write_license(path, statement):
    """Retain UniProt's required copyright statement with each installed copy."""
    if not statement:
        return
    destination = path.with_suffix(".license.txt")
    _check_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".uniprot-license-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(statement)
            handle.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def fetch_uniprot_reference(
    release=None,
    *,
    collection=DEFAULT_COLLECTION,
    force=False,
    max_asset_bytes=DEFAULT_MAX_ASSET_BYTES,
    max_cache_bytes=DEFAULT_MAX_CACHE_BYTES,
    verbose=True,
) -> Path:
    """Fetch an explicit release, or the catalog's pinned default.

    Cache hits are checksum verified. Corruption raises; ``force=True`` repairs
    explicitly. All writers serialize, and preflight includes existing files,
    partials, a full new asset and 2 MiB transfer/control headroom. There is no
    automatic eviction. Limits cover the UniProt subtree, not other Hitlist
    data or downstream indexes. New transfers require a POSIX local filesystem
    for datacache's bounded resumable downloader; offline reads are portable.
    """
    _positive_integer("max_asset_bytes", max_asset_bytes)
    _positive_integer("max_cache_bytes", max_cache_bytes)
    reference = _reference(release, collection)
    if reference["size_bytes"] > max_asset_bytes:
        raise ValueError(f"UniProt asset exceeds max_asset_bytes={max_asset_bytes}")
    path = _path(reference)
    _check_path(path)
    if not force and path.exists():
        return uniprot_path(release, collection=collection)
    if os.name != "posix":
        raise NotImplementedError(
            "Bounded UniProt downloads require a POSIX local filesystem; "
            "catalog inspection and existing verified FASTAs remain readable."
        )
    root = uniprot_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".hitlist-uniprot.lock"
    _check_path(lock)
    with FileLock(str(lock)):
        _check_path(path)
        if not force and path.exists():
            return uniprot_path(release, collection=collection)
        required = _cache_bytes() + reference["size_bytes"] + _TRANSFER_HEADROOM_BYTES
        if required > max_cache_bytes:
            raise ValueError(
                f"UniProt cache requires up to {required} bytes; max_cache_bytes={max_cache_bytes}. "
                "Remove a release explicitly or raise the budget. No files were evicted."
            )

        def bound_transfer(done, total):
            if done > reference["size_bytes"]:
                raise ValueError("UniProt transfer exceeds its pinned size_bytes")

        _write_license(path, reference["license_statement"])
        datacache.fetch_file(
            reference["url"],
            destination=path,
            expected_sha256=reference["sha256"],
            expected_size=reference["size_bytes"],
            raw=True,
            force=force,
            resume=True,
            record_provenance=True,
            chunk_size=_CHUNK_BYTES,
            progress_callback=bound_transfer,
            show_progress=verbose,
            timeout=60,
        )
    return path


def remove_uniprot_reference(release, *, collection=DEFAULT_COLLECTION) -> bool:
    """Delete one catalog FASTA and its transfer state, leaving other files alone.

    An explicit release is required. Empty control directories/locks remain to
    keep concurrent download/removal coordination valid.
    """
    if release is None:
        raise ValueError("Removal requires an explicit UniProt release")
    reference = _reference(release, collection)
    path = _path(reference)
    _check_path(path)
    root = uniprot_cache_dir()
    if not root.exists():
        return False
    lock = root / ".hitlist-uniprot.lock"
    _check_path(lock)
    with FileLock(str(lock)):
        _check_path(path)
        existed = path.exists()
        if os.name == "posix":
            datacache.discard_partial(path)
        path.unlink(missing_ok=True)
        path.with_suffix(".license.txt").unlink(missing_ok=True)
        datacache.provenance.remove(path)
    return existed


def uniprot_reference_for_digest(sha256, size_bytes) -> dict | None:
    """Identify exact catalog bytes, including copies outside the managed cache.

    Callers must compute the digest of the actual input first. This adds asset
    provenance only; it does not validate a study's searched-reference claim.
    """
    for name, spec in uniprot_catalog()["collections"].items():
        for release, entry in spec["releases"].items():
            if entry["sha256"] == sha256 and entry["size_bytes"] == size_bytes:
                return _reference(release, name)
    return None
