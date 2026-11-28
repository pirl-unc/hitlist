"""Coherent, immutable index generations in the shared datacache directory.

Hitlist owns the index schema and scientific checks. Datacache owns staging,
input copies, locking, hashes, receipts and complete-directory publication.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from datacache import inspect_materialization, materialize

from .downloads import data_dir
from .errors import DatasetError

STORE_NAME = "dataset"
INDEX_FILES = (
    "observations.parquet",
    "binding.parquet",
    "other_assays.parquet",
    "observation_contributors.parquet",
    "bulk_proteomics.parquet",
    "line_expression.parquet",
    "observations_meta.json",
)
MAPPING_FILES = ("peptide_mappings.parquet", "peptide_mappings_meta.json")
# Compact dimension/summary tables written by new builds (#990, #992).
# Kept outside required INDEX_FILES so pre-2.38.0 published bundles stay valid.
SUMMARY_FILES = ("samples.parquet", "alleles.parquet")
STALE_ROOT_CANDIDATES = (*INDEX_FILES, *MAPPING_FILES)

# Friendly table aliases accepted by ``bundle_path()`` and ``hitlist path <table>`` (#989).
BUNDLE_TABLE_ALIASES: dict[str, str] = {
    "observations": "observations.parquet",
    "ms": "observations.parquet",
    "ms_ligands": "observations.parquet",
    "binding": "binding.parquet",
    "other_assays": "other_assays.parquet",
    "other": "other_assays.parquet",
    "contributors": "observation_contributors.parquet",
    "observation_contributors": "observation_contributors.parquet",
    "mappings": "peptide_mappings.parquet",
    "peptide_mappings": "peptide_mappings.parquet",
    "primary_flanks": "primary_flanks.parquet",
    "flanks": "primary_flanks.parquet",
    "samples": "samples.parquet",
    "cohorts": "samples.parquet",
    "alleles": "alleles.parquet",
    "proteomes": "proteomes.parquet",
    "bulk_proteomics": "bulk_proteomics.parquet",
    "bulk": "bulk_proteomics.parquet",
    "line_expression": "line_expression.parquet",
    "expression": "line_expression.parquet",
    "meta": "observations_meta.json",
    "observations_meta": "observations_meta.json",
    "mappings_meta": "peptide_mappings_meta.json",
    "peptide_mappings_meta": "peptide_mappings_meta.json",
    "manifest": ".datacache-manifest.json",
    "readme": "README.md",
}

# Context-local pins support nested API calls and independent reader threads.
_pins = ContextVar("hitlist_dataset_pins", default=None)
_staging = ContextVar("hitlist_dataset_staging", default=None)
_warned_stale_roots: set[tuple[Path, Path]] = set()


class DatasetGenerationError(DatasetError):
    """A claimed published generation is incomplete or unreadable."""


def dataset_cache_check(check):
    """Keep cache predicates boolean while ordinary reads explain corruption."""

    @wraps(check)
    def checked(*args, **kwargs):
        try:
            with dataset_snapshot():
                return check(*args, **kwargs)
        except DatasetGenerationError:
            return False

    return checked


def _root(root=None):
    return Path(data_dir() if root is None else root).absolute()


def _inspection(root, *, verify_files: bool = False):
    state = inspect_materialization(root / STORE_NAME, verify_files=verify_files)
    if state.status not in {"available", "missing"}:
        raise DatasetGenerationError(
            f"Invalid Hitlist dataset at {state.path}: {state.error}"
        ) from state.error
    if state.status == "available" and not set(INDEX_FILES) <= set(state.files):
        raise DatasetGenerationError(f"Incomplete Hitlist dataset generation: {state.bundle}")
    return state


def stale_root_files(*, root=None) -> list[Path]:
    """Return legacy root-level index files shadowed by a published bundle (#989).

    Returns an empty list if no published bundle exists under ``root`` (pure
    legacy cache) or if no legacy root-level parquet/metadata files are present.
    Never creates or modifies directories.
    """
    resolved = _root(root)
    if not resolved.is_dir():
        return []
    state = inspect_materialization(resolved / STORE_NAME, verify_files=False)
    if state.status != "available":
        return []
    return [resolved / name for name in STALE_ROOT_CANDIDATES if (resolved / name).exists()]


def _reset_stale_root_warnings() -> None:
    """Clear the per-process stale-root warning cache (used by tests)."""
    _warned_stale_roots.clear()


def _warn_if_stale_root_files(root: Path, bundle: Path) -> None:
    key = (root, bundle)
    if key in _warned_stale_roots:
        return
    if not root.is_dir():
        return
    stale = [root / name for name in STALE_ROOT_CANDIDATES if (root / name).exists()]
    if not stale:
        return
    _warned_stale_roots.add(key)
    names = ", ".join(p.name for p in stale[:4])
    if len(stale) > 4:
        names += f", ... ({len(stale)} total)"
    warnings.warn(
        f"Stale root-level cache files ({names}) in {root} coexist with newer published "
        f"bundle {bundle}. Use hitlist.latest_bundle() or `hitlist path` instead of reading "
        f"root-level *.parquet files directly (#989).",
        UserWarning,
        stacklevel=3,
    )


def cache_layout(*, root=None) -> dict[str, Path]:
    """Return the canonical cache directory paths without creating them (#989)."""
    resolved = _root(root)
    bundles_dir = resolved / STORE_NAME / "bundles"
    return {
        "root": resolved,
        "raw": resolved / "raw",
        "proteomes": resolved / "proteomes",
        "bundles": bundles_dir,
        "current": bundles_dir / "current",
        "scratch": resolved / "scratch",
    }


def _update_current_pointer(bundle_dir: Path) -> Path:
    """Atomically point ``<bundles_dir>/current`` at ``bundle_dir`` (#989).

    Uses a relative symlink swapped atomically via ``os.replace``. On platforms
    where symlink creation is unavailable, falls back to writing
    ``current.json`` atomically in the same directory.
    """
    bundles_dir = bundle_dir.parent
    pointer = bundles_dir / "current"
    tmp = bundles_dir / f".current.{os.getpid()}.tmp"
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(bundle_dir.name, tmp)
        os.replace(tmp, pointer)
        return pointer
    except OSError:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        fallback = bundles_dir / "current.json"
        fallback_tmp = bundles_dir / f".current.{os.getpid()}.json.tmp"
        payload = json.dumps({"bundle": bundle_dir.name, "path": str(bundle_dir)}, indent=2) + "\n"
        fallback_tmp.write_text(payload, encoding="utf-8")
        os.replace(fallback_tmp, fallback)
        return fallback


def _current(root):
    state = _inspection(root)
    if state.status == "available":
        bundle = Path(state.bundle)
        _warn_if_stale_root_files(root, bundle)
        return bundle
    return root


def latest_bundle(
    *,
    root=None,
    allow_legacy: bool = False,
    verify: bool = False,
    warn_stale: bool = True,
) -> Path:
    """Return the active verified Hitlist bundle directory (#989).

    Respects an active :func:`dataset_snapshot` pin when ``verify=False``.
    Raises :class:`DatasetGenerationError` if no published bundle exists
    (unless ``allow_legacy=True`` and a flat ``observations.parquet`` exists
    at ``root``). Never creates directories.
    """
    resolved = _root(root)
    pins = _pins.get() or {}
    if resolved in pins and not verify:
        pinned = pins[resolved]
        if pinned != resolved or (allow_legacy and (resolved / "observations.parquet").exists()):
            return pinned
    state = _inspection(resolved, verify_files=verify)
    if state.status == "available":
        bundle = Path(state.bundle)
        if warn_stale:
            _warn_if_stale_root_files(resolved, bundle)
        return bundle
    if allow_legacy and (resolved / "observations.parquet").exists():
        return resolved
    raise DatasetGenerationError(
        f"No published Hitlist dataset bundle found under {resolved / STORE_NAME}. "
        "Run `hitlist build` first (or pass allow_legacy=True / --legacy for flat root caches)."
    )


def bundle_path(
    table: str | None = None,
    *,
    root=None,
    bundle: str | Path | None = None,
    allow_legacy: bool = False,
    verify: bool = False,
    must_exist: bool = False,
    warn_stale: bool = True,
) -> Path:
    """Resolve the active bundle directory or a named file inside it (#989).

    ``table`` may be a short alias (e.g. ``"observations"``, ``"ms"``,
    ``"binding"``, ``"mappings"``, ``"samples"``, ``"alleles"``,
    ``"proteomes"``) or an exact filename inside the bundle.
    """
    if bundle is not None:
        base = Path(bundle).expanduser().resolve()
        if not base.is_dir():
            raise DatasetGenerationError(f"Bundle directory does not exist: {base}")
    else:
        base = latest_bundle(
            root=root,
            allow_legacy=allow_legacy,
            verify=verify,
            warn_stale=warn_stale,
        )
    if table is None:
        return base
    token = str(table).strip()
    if not token or "/" in token or "\\" in token or token in {".", ".."}:
        raise DatasetError(f"Invalid bundle table name: {table!r}")
    filename = BUNDLE_TABLE_ALIASES.get(token.lower(), token)
    target = base / filename
    if must_exist and not target.exists():
        raise FileNotFoundError(f"Bundle file {filename!r} does not exist in {base}")
    return target


def dataset_path(name, *, root=None):
    """Resolve an index in the pinned/current generation, or the legacy layout.

    No file-level fallback: a missing file in a generation is never taken from
    an older generation or from the flat cache. Reads never create directories.
    """
    root = _root(root)
    pins = _pins.get() or {}
    directory = pins[root] if root in pins else _current(root)
    return directory / name


@contextmanager
def dataset_snapshot(*, root=None):
    """Pin related Hitlist reads to one generation; nested calls inherit it.

    Use this around several API calls that must agree. Returned generation
    paths remain usable after later builds; automatic pruning is never done.
    Legacy flat files remain readable but cannot provide snapshot isolation
    against older Hitlist writers that overwrite those files in place.
    """
    root = _root(root)
    pins = _pins.get() or {}
    if root in pins:
        yield pins[root]
        return
    directory = _current(root)
    token = _pins.set({**pins, root: directory})
    try:
        yield directory
    finally:
        _pins.reset(token)


@contextmanager
def _stage(directory, root):
    pin = _pins.set({**(_pins.get() or {}), root: directory})
    stage = _staging.set(root)
    try:
        yield
    finally:
        _staging.reset(stage)
        _pins.reset(pin)


def is_staging(*, root=None):
    return _staging.get() == _root(root)


def _write_readme(directory):
    from .version import __version__

    names = sorted(path.name for path in directory.iterdir() if path.is_file())
    metadata_path = directory / "observations_meta.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    sources = metadata.get("provenance", {}).get("sources", {})
    text = [
        "# Hitlist dataset generation",
        "",
        f"Written with Hitlist {__version__}. This directory is immutable after publication.",
        "",
        "Datacache's `.datacache-manifest.json` lists every output with its byte size",
        "and SHA-256, the source receipts and transform identity. Observed hashes",
        "establish consistency; they do not by themselves establish scientific validity.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "| --- | --- |",
    ]
    descriptions = {
        "observations.parquet": "Positive MHC ligand MS observations",
        "binding.parquet": "Binding assays; preserve endpoints, units and censoring",
        "other_assays.parquet": "Other/negative/unknown assay evidence retained for inspection",
        "observation_contributors.parquet": "All source contributors to retained observations",
        "peptide_mappings.parquet": "Alternative protein occurrences; not independent observations",
        "peptide_mappings_meta.json": "Mapping settings and reference coverage",
        "samples.parquet": "Compact cohort/sample summary index with peptide and flank counts",
        "alleles.parquet": "Compact canonical MHC allele dimension table with modality counts",
        "proteomes.parquet": "Host reference proteome registry with SHA-256 and mapping counts",
        "primary_flanks.parquet": "Deterministic canonical-isoform primary host flanks per (peptide, proteome_id)",
        "bulk_proteomics.parquet": "Bulk MS; fraction counts are not detectability labels",
        "line_expression.parquet": "Cell-line expression evidence",
        "observations_meta.json": "Build inputs, curation identity and contributor contract",
    }
    text.extend(
        f"| `{name}` | {descriptions.get(name, 'Generation documentation')} |"
        for name in names
        if name != "README.md"
    )
    text.extend(["", "## Registered evidence sources", ""])
    for name, entry in sorted(sources.items()):
        text.extend(
            [
                f"- **{name}**: `{entry.get('path', '')}`",
                f"  SHA-256: `{entry.get('sha256', '')}`; bytes: {entry.get('size_bytes', 'unknown')}.",
            ]
        )
    text.extend(
        [
            "",
            "## Inspection and interpretation",
            "",
            "Use `hitlist.dataset_snapshot()` around related API calls. Source PMIDs,",
            "assay IRIs and original source fields remain in the tables/contributors.",
            "Cell types, cell conditions, cellular MHC typing and experimental restriction",
            "sets have distinct meanings; retain their explicit columns and uncertainty.",
            "Missing metadata is unknown, not permission to infer donor/specimen identity.",
            "",
            "This is an index generation, not a qualified training release. Frozen",
            "training bundles must additionally record label policies, partition membership,",
            "coverage and leakage audits. No training-readiness claim follows from a build.",
            "",
            "Prior generations and registered input files are retained. Rebuilds need",
            "space for source copies, new outputs and scratch in addition to retained data.",
            "Publication is an atomic directory rename on a local filesystem; it does not",
            "guarantee recovery from arbitrary filesystem or power-loss damage.",
            "",
        ]
    )
    (directory / "README.md").write_text("\n".join(text))


def _validate_generation():
    from .mappings import _obs_fingerprint, mappings_meta_path
    from .provenance import _contributor_contract

    if _contributor_contract(verify_hashes=True) is None:
        raise ValueError("A dataset generation requires captured contributor provenance")
    metadata = mappings_meta_path()
    if metadata.exists():
        value = json.loads(metadata.read_text())
        if value.get("observations") != _obs_fingerprint():
            raise ValueError("Mappings do not match the staged evidence indexes")


def _resolve_materialization_sources(sources):
    """Resolve symlinked local source paths before datacache.materialize(O_NOFOLLOW) (#954)."""
    resolved_sources = {}
    for name, spec in sources.items():
        if isinstance(spec, dict) and "path" in spec:
            raw_path = Path(spec["path"])
            resolved = raw_path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(
                    f"Materialization source {name!r} at {raw_path} resolved to {resolved}, "
                    "which is not a regular file"
                )
            resolved_sources[name] = {**spec, "path": str(resolved)}
        else:
            resolved_sources[name] = spec
    return resolved_sources


def publish_dataset(sources, names, build, *, root=None, options=None):
    """Build declared indexes in private datacache staging, then publish once."""
    from .version import __version__

    root = _root(root)
    resolved_sources = _resolve_materialization_sources(sources)
    outputs = {name: {} for name in (*names, "README.md")}

    def staged(source_paths, output_paths):
        directory = Path(output_paths["README.md"]).parent
        with _stage(directory, root):
            build({name: Path(path) for name, path in source_paths.items()})
            _validate_generation()
            _write_readme(directory)

    published = {
        name: Path(path)
        for name, path in materialize(
            root / STORE_NAME,
            resolved_sources,
            transform={"version": f"hitlist-{__version__}", "options": options or {}},
            outputs=outputs,
            builder=staged,
            force=True,
            download_options={"show_progress": False},
        ).items()
    }
    if published:
        bundle_dir = next(iter(published.values())).parent
        _update_current_pointer(bundle_dir)
    return published


def update_dataset(names, write, *, root=None, options=None, metadata_updates=None):
    """Publish changed sidecars with the other files from one parent generation.

    Internal build calls already in staging, and legacy flat standalone builds,
    call the writer directly. A changed parent aborts rather than losing an
    intervening update. The caller can retry against the new current generation.
    """
    root = _root(root)
    if is_staging(root=root):
        return write()
    state = _inspection(root)
    if state.status == "missing":
        return write()
    parent = Path(state.bundle)
    preserved = set(state.files) - set(names) - {"README.md"}
    receipt = json.loads((parent / ".datacache-manifest.json").read_text())
    sources = {
        name: {"path": state.files[name].path, **receipt["outputs"][name]} for name in preserved
    }
    result = []

    def build(inputs):
        if _current(root) != parent:
            raise RuntimeError("Dataset generation changed before update; retry the build")
        for name, source in inputs.items():
            target = dataset_path(name, root=root)
            shutil.copyfile(source, target)
            # Datacache's private input copy has a fresh timestamp. Existing
            # freshness contracts refer to the unchanged published bytes.
            shutil.copystat(parent / name, target)
        result.append(write())
        from .builder import _parquet_fingerprints

        metadata = dataset_path("observations_meta.json", root=root)
        if metadata.exists():
            value = json.loads(metadata.read_text())
            value["parquets"] = _parquet_fingerprints()
            value.update(metadata_updates or {})
            metadata.write_text(json.dumps(value, indent=2) + "\n")

    published = publish_dataset(
        sources,
        sorted(preserved | set(names)),
        build,
        root=root,
        options={"parent_generation": parent.name, "update": list(names), **(options or {})},
    )
    value = result[0]
    return published[value.name] if isinstance(value, Path) else value
