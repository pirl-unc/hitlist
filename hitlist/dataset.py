"""Coherent, immutable index generations in the shared datacache directory.

Hitlist owns the index schema and scientific checks. Datacache owns staging,
input copies, locking, hashes, receipts and complete-directory publication.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from datacache import inspect_materialization, materialize

from .downloads import data_dir

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
# Context-local pins support nested API calls and independent reader threads.
_pins = ContextVar("hitlist_dataset_pins", default=None)
_staging = ContextVar("hitlist_dataset_staging", default=None)


class DatasetGenerationError(ValueError):
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


def _inspection(root):
    state = inspect_materialization(root / STORE_NAME, verify_files=False)
    if state.status not in {"available", "missing"}:
        raise DatasetGenerationError(
            f"Invalid Hitlist dataset at {state.path}: {state.error}"
        ) from state.error
    if state.status == "available" and not set(INDEX_FILES) <= set(state.files):
        raise DatasetGenerationError(f"Incomplete Hitlist dataset generation: {state.bundle}")
    return state


def _current(root):
    state = _inspection(root)
    return Path(state.bundle) if state.status == "available" else root


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


def publish_dataset(sources, names, build, *, root=None, options=None):
    """Build declared indexes in private datacache staging, then publish once."""
    from .version import __version__

    root = _root(root)
    outputs = {name: {} for name in (*names, "README.md")}

    def staged(source_paths, output_paths):
        directory = Path(output_paths["README.md"]).parent
        with _stage(directory, root):
            build({name: Path(path) for name, path in source_paths.items()})
            _validate_generation()
            _write_readme(directory)

    return {
        name: Path(path)
        for name, path in materialize(
            root / STORE_NAME,
            sources,
            transform={"version": f"hitlist-{__version__}", "options": options or {}},
            outputs=outputs,
            builder=staged,
            force=True,
            download_options={"show_progress": False},
        ).items()
    }


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
