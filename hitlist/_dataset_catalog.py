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

"""Unified catalog adapter for managed external datasets and bundles.

Connects release-pinned and multi-file datasets (UniProt references, HLA Ligand
Atlas + community crosswalk, T-cell audit exports, Stopfer perturbation
quantities, canine bulk proteomics, PXD013455 portable bundle / source recovery,
and DLBCL PMID 35561310 supplements) to the single ``hitlist.downloads`` /
``hitlist data`` surface (``available_datasets``, ``preflight``, ``fetch``,
``refresh``, ``info``, ``get_path``, ``remove``, and ``list_cache_files``).
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datacache

from .curation_yaml import load_curation_yaml
from .errors import DownloadError, HitlistError, InsufficientSpaceError, IntegrityError


@dataclass(frozen=True)
class ManagedDataset:
    """Descriptor for one catalog-managed dataset or multi-file bundle."""

    name: str
    kind: str  # "file" or "bundle"
    description: str
    terms: str | None
    path_fn: Callable[[], Path]
    files_fn: Callable[[], list[dict[str, Any]]]
    exists_fn: Callable[[], bool]
    fetch_fn: Callable[[bool], Path]
    info_fn: Callable[[], dict[str, Any]]
    remove_fn: Callable[[bool], bool]
    cache_kind: str = "managed dataset"


def _free_bytes_for_path(dest: Path) -> int:
    probe = dest.parent if dest.suffix else dest
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def _build_plan(dataset: ManagedDataset, *, force: bool = False) -> dict[str, Any]:
    path = dataset.path_fn()
    files = dataset.files_fn()
    exists = dataset.exists_fn()
    expected = (
        sum(f["expected_size_bytes"] for f in files)
        if files and all(f["expected_size_bytes"] is not None for f in files)
        else None
    )
    free = _free_bytes_for_path(path)
    would_download = force or not exists
    plan: dict[str, Any] = {
        "name": dataset.name,
        "kind": dataset.kind,
        "path": str(path),
        "registered": exists,
        "exists": exists,
        "stale": False,
        "would_download": would_download,
        "expected_size_bytes": expected,
        "free_bytes": free,
        "enough_space": None if expected is None else free >= expected,
        "terms": dataset.terms,
    }
    if dataset.kind == "bundle":
        plan["files"] = [
            {
                "name": f["name"],
                "path": str(f["path"]),
                "exists": f["exists"],
                "would_download": force or not f["exists"],
                "expected_size_bytes": f["expected_size_bytes"],
            }
            for f in files
        ]
    return plan


def _wrap_managed_errors(name: str, fn: Callable[[], Path]) -> Path:
    try:
        return fn()
    except HitlistError:
        raise
    except datacache.FileValidationError as error:
        raise IntegrityError(f"Failed to fetch '{name}': {error}") from error
    except (OSError, ValueError, NotImplementedError) as error:
        if type(error).__name__ in {"InsufficientSpace", "InsufficientSpaceError"}:
            raise InsufficientSpaceError(
                f"Failed to fetch '{name}': {error}",
                path=getattr(error, "path", None),
                required_bytes=getattr(error, "required_bytes", None),
                available_bytes=getattr(error, "available_bytes", None),
            ) from error
        raise DownloadError(f"Failed to fetch '{name}': {error}") from error


def _remove_tree(root: Path, *, delete_file: bool) -> bool:
    if not root.exists():
        return False
    if not delete_file:
        return True
    if root.is_dir():
        shutil.rmtree(root)
    else:
        root.unlink(missing_ok=True)
    return True


def _bundle_generation_path(root: Path, assets: dict[str, dict[str, Any]]) -> Path:
    state = datacache.inspect_bundle(root, assets, verify_files=False)
    if state.status == "available" and state.bundle:
        return Path(state.bundle)
    return root


def _uniprot_datasets() -> dict[str, ManagedDataset]:
    from .uniprot import (
        fetch_uniprot_reference,
        remove_uniprot_reference,
        uniprot_catalog,
        uniprot_info,
    )

    catalog = uniprot_catalog()
    terms = catalog.get("license_statement") or "https://www.uniprot.org/help/license"
    out: dict[str, ManagedDataset] = {}

    for collection, spec in sorted(catalog["collections"].items()):
        default_release = spec["default_release"]
        entries: list[tuple[str, str, str]] = [
            (
                f"uniprot/{collection}",
                default_release,
                f"UniProt {collection} reference FASTA (default release {default_release})",
            )
        ]
        for release in sorted(spec["releases"]):
            entries.append(
                (
                    f"uniprot/{collection}/{release}",
                    release,
                    f"UniProt {collection} reference FASTA ({release})",
                )
            )
        for name, release, desc in entries:

            def _make(
                _name: str = name,
                _collection: str = collection,
                _release: str = release,
                _desc: str = desc,
            ) -> ManagedDataset:
                def _info() -> dict[str, Any]:
                    raw = uniprot_info(_release, collection=_collection, verify=False)
                    exists = raw["status"] != "missing"
                    return {
                        **raw,
                        "name": _name,
                        "type": "managed reference",
                        "exists": exists,
                        "status": "installed" if raw["status"] == "available" else raw["status"],
                        "terms": terms,
                    }

                def _path() -> Path:
                    return Path(_info()["path"])

                def _files() -> list[dict[str, Any]]:
                    raw = uniprot_info(_release, collection=_collection, verify=False)
                    return [
                        {
                            "name": raw["filename"],
                            "path": Path(raw["path"]),
                            "exists": raw["status"] == "available",
                            "expected_size_bytes": raw["size_bytes"],
                        }
                    ]

                def _exists() -> bool:
                    return uniprot_info(_release, collection=_collection, verify=False)[
                        "status"
                    ] == ("available")

                def _fetch(force: bool) -> Path:
                    return fetch_uniprot_reference(
                        _release, collection=_collection, force=force, verbose=True
                    )

                def _remove(delete_file: bool) -> bool:
                    if not delete_file:
                        return _exists()
                    return remove_uniprot_reference(_release, collection=_collection)

                return ManagedDataset(
                    name=_name,
                    kind="file",
                    description=_desc,
                    terms=terms,
                    path_fn=_path,
                    files_fn=_files,
                    exists_fn=_exists,
                    fetch_fn=_fetch,
                    info_fn=_info,
                    remove_fn=_remove,
                    cache_kind="UniProt reference",
                )

            out[name] = _make()
    return out


def _datacache_bundle_dataset(
    *,
    name: str,
    description: str,
    terms: str | None,
    root_fn: Callable[[], Path],
    assets_fn: Callable[[], dict[str, dict[str, Any]]],
    fetch_bundle_fn: Callable[[], dict[str, Path]],
) -> ManagedDataset:
    def _inspect() -> datacache.BundleInspection:
        return datacache.inspect_bundle(root_fn(), assets_fn(), verify_files=False)

    def _path() -> Path:
        return _bundle_generation_path(root_fn(), assets_fn())

    def _files() -> list[dict[str, Any]]:
        state = _inspect()
        base = Path(state.bundle) if state.status == "available" and state.bundle else root_fn()
        installed = state.status == "available"
        return [
            {
                "name": filename,
                "path": base / filename,
                "exists": installed and (base / filename).is_file(),
                "expected_size_bytes": spec.get("size_bytes", spec.get("size")),
            }
            for filename, spec in sorted(assets_fn().items())
        ]

    def _exists() -> bool:
        return _inspect().status == "available"

    def _fetch(force: bool) -> Path:
        if force and root_fn().exists():
            shutil.rmtree(root_fn())
        paths = fetch_bundle_fn()
        first = next(iter(paths.values()))
        return Path(first).parent

    def _info() -> dict[str, Any]:
        state = _inspect()
        files = _files()
        expected = sum(f["expected_size_bytes"] or 0 for f in files)
        return {
            "name": name,
            "type": "managed bundle",
            "description": description,
            "path": str(_path()),
            "exists": state.status == "available",
            "status": "installed" if state.status == "available" else state.status,
            "cache_status": state.status,
            "verified": state.verified,
            "size_bytes": expected,
            "files": [f["name"] for f in files],
            "terms": terms,
            "error": str(state.error) if state.error else None,
        }

    def _remove(delete_file: bool) -> bool:
        return _remove_tree(root_fn(), delete_file=delete_file)

    return ManagedDataset(
        name=name,
        kind="bundle",
        description=description,
        terms=terms,
        path_fn=_path,
        files_fn=_files,
        exists_fn=_exists,
        fetch_fn=_fetch,
        info_fn=_info,
        remove_fn=_remove,
        cache_kind="managed bundle",
    )


def _atlas_datasets() -> dict[str, ManagedDataset]:
    from .atlas import (
        CATALOG_PATH,
        CROSSWALK_CATALOG_PATH,
        atlas_crosswalk_sources,
        atlas_sources,
    )
    from .downloads import data_dir
    from .tissue_blacklist import ATLAS_RELEASE

    atlas_meta = load_curation_yaml(CATALOG_PATH)
    crosswalk_meta = load_curation_yaml(CROSSWALK_CATALOG_PATH)

    return {
        "hla-ligand-atlas": _datacache_bundle_dataset(
            name="hla-ligand-atlas",
            description=f"HLA Ligand Atlas {ATLAS_RELEASE} primary donor/peptide tables",
            terms=f"{atlas_meta.get('license', 'CC-BY-4.0')} ({atlas_meta.get('source_url', '')})",
            root_fn=lambda: data_dir() / "hla-ligand-atlas" / ATLAS_RELEASE / "sources",
            assets_fn=lambda: atlas_meta["files"],
            fetch_bundle_fn=atlas_sources,
        ),
        "hla-ligand-atlas-crosswalk": _datacache_bundle_dataset(
            name="hla-ligand-atlas-crosswalk",
            description="HLA Ligand Atlas PXD019643 community SDRF & PRIDE file inventory",
            terms="https://www.ebi.ac.uk/pride/archive/projects/PXD019643",
            root_fn=lambda: data_dir() / "hla-ligand-atlas" / ATLAS_RELEASE / "community-crosswalk",
            assets_fn=lambda: crosswalk_meta["files"],
            fetch_bundle_fn=atlas_crosswalk_sources,
        ),
    }


def _tcell_datasets() -> dict[str, ManagedDataset]:
    from .downloads import data_dir
    from .tcell_assays import CATALOG_PATH, tcell_assay_sources

    meta = load_curation_yaml(CATALOG_PATH)
    return {
        "tcell-assays": _datacache_bundle_dataset(
            name="tcell-assays",
            description="IEDB & CEDAR T-cell v3 audit exports (2026-10-20 snapshot)",
            terms="https://www.iedb.org/ and https://cedar.iedb.org/",
            root_fn=lambda: data_dir() / "functional-evidence" / "tcell-availability-2026-10-20",
            assets_fn=lambda: meta["files"],
            fetch_bundle_fn=tcell_assay_sources,
        )
    }


def _perturbation_datasets() -> dict[str, ManagedDataset]:
    from .downloads import data_dir
    from .perturbation_quantities import fetch_perturbation_sources, source_contract

    contract = source_contract()
    assets = contract["assets"]
    terms = contract.get("publication_url") or "https://doi.org/10.1038/s41467-020-16588-9"

    def _root() -> Path:
        return data_dir() / "perturbation_sources"

    def _files() -> list[dict[str, Any]]:
        root = _root()
        return [
            {
                "name": asset["filename"],
                "path": root / asset["filename"],
                "exists": (root / asset["filename"]).is_file(),
                "expected_size_bytes": asset["size_bytes"],
            }
            for asset in assets
        ]

    def _exists() -> bool:
        return all(f["exists"] for f in _files())

    def _fetch(force: bool) -> Path:
        fetch_perturbation_sources(force=force, verbose=True)
        return _root()

    def _info() -> dict[str, Any]:
        files = _files()
        installed = all(f["exists"] for f in files)
        return {
            "name": "perturbation-quantities",
            "type": "managed bundle",
            "description": "Stopfer 2020 palbociclib perturbation quantity workbooks (PMID 32488085)",
            "path": str(_root()),
            "exists": installed,
            "status": "installed" if installed else "not installed",
            "size_bytes": sum(a["size_bytes"] for a in assets),
            "files": [a["filename"] for a in assets],
            "terms": terms,
        }

    return {
        "perturbation-quantities": ManagedDataset(
            name="perturbation-quantities",
            kind="bundle",
            description="Stopfer 2020 palbociclib perturbation quantity workbooks (PMID 32488085)",
            terms=terms,
            path_fn=_root,
            files_fn=_files,
            exists_fn=_exists,
            fetch_fn=_fetch,
            info_fn=_info,
            remove_fn=lambda delete_file: _remove_tree(_root(), delete_file=delete_file),
            cache_kind="managed bundle",
        )
    }


def _dlbcl_datasets() -> dict[str, ManagedDataset]:
    from .downloads import data_dir
    from .drug_dm_qualification import dlbcl_supplement_contract, fetch_dlbcl_supplements

    contract = dlbcl_supplement_contract()
    workbooks = list(contract["workbooks"].values())
    terms = f"https://doi.org/{contract['doi']}"

    def _root() -> Path:
        return data_dir() / "dlbcl_35561310_supplements"

    def _files() -> list[dict[str, Any]]:
        root = _root()
        return [
            {
                "name": wb["filename"],
                "path": root / wb["filename"],
                "exists": (root / wb["filename"]).is_file(),
                "expected_size_bytes": wb["size_bytes"],
            }
            for wb in workbooks
        ]

    def _exists() -> bool:
        return all(f["exists"] for f in _files())

    def _fetch(force: bool) -> Path:
        fetch_dlbcl_supplements(force=force, verbose=True)
        return _root()

    def _info() -> dict[str, Any]:
        files = _files()
        installed = all(f["exists"] for f in files)
        return {
            "name": "dlbcl-35561310-supplements",
            "type": "managed bundle",
            "description": "Bourne 2022 DLBCL Tables 3 & 5 workbooks (PMID 35561310 / PMC9327544)",
            "path": str(_root()),
            "exists": installed,
            "status": "installed" if installed else "not installed",
            "size_bytes": sum(wb["size_bytes"] for wb in workbooks),
            "files": [wb["filename"] for wb in workbooks],
            "terms": terms,
        }

    return {
        "dlbcl-35561310-supplements": ManagedDataset(
            name="dlbcl-35561310-supplements",
            kind="bundle",
            description="Bourne 2022 DLBCL Tables 3 & 5 workbooks (PMID 35561310 / PMC9327544)",
            terms=terms,
            path_fn=_root,
            files_fn=_files,
            exists_fn=_exists,
            fetch_fn=_fetch,
            info_fn=_info,
            remove_fn=lambda delete_file: _remove_tree(_root(), delete_file=delete_file),
            cache_kind="managed bundle",
        )
    }


def _canine_bulk_datasets() -> dict[str, ManagedDataset]:
    from .canine_bulk import canine_bulk_catalog, fetch_canine_bulk_asset
    from .downloads import data_dir

    catalog = canine_bulk_catalog()["sources"]
    out: dict[str, ManagedDataset] = {}

    for source_id, spec in sorted(catalog.items()):
        version = spec["version"]
        terms = f"{spec.get('license', '')} (DOI: {spec.get('study', '')})".strip()
        bundle_name = f"canine-bulk/{source_id}"

        def _make_bundle(
            _source_id: str = source_id,
            _version: str = version,
            _spec: dict[str, Any] = spec,
            _terms: str = terms,
            _bundle_name: str = bundle_name,
        ) -> ManagedDataset:
            def _root() -> Path:
                return data_dir() / "canine_bulk" / _source_id / _version

            def _files() -> list[dict[str, Any]]:
                root = _root()
                return [
                    {
                        "name": asset_spec["filename"],
                        "path": root / asset_spec["filename"],
                        "exists": (root / asset_spec["filename"]).is_file(),
                        "expected_size_bytes": asset_spec["size_bytes"],
                    }
                    for _asset_key, asset_spec in sorted(_spec["assets"].items())
                ]

            def _exists() -> bool:
                return all(f["exists"] for f in _files())

            def _fetch(force: bool) -> Path:
                if force and _root().exists():
                    shutil.rmtree(_root())
                for asset_key in sorted(_spec["assets"]):
                    fetch_canine_bulk_asset(_source_id, asset_key)
                return _root()

            def _info() -> dict[str, Any]:
                files = _files()
                installed = all(f["exists"] for f in files)
                return {
                    "name": _bundle_name,
                    "type": "managed bundle",
                    "description": f"Canine bulk proteomics {_source_id} ({_version})",
                    "path": str(_root()),
                    "exists": installed,
                    "status": "installed" if installed else "not installed",
                    "size_bytes": sum(a["size_bytes"] for a in _spec["assets"].values()),
                    "files": [a["filename"] for a in _spec["assets"].values()],
                    "terms": _terms,
                }

            return ManagedDataset(
                name=_bundle_name,
                kind="bundle",
                description=f"Canine bulk proteomics {_source_id} ({_version})",
                terms=_terms,
                path_fn=_root,
                files_fn=_files,
                exists_fn=_exists,
                fetch_fn=_fetch,
                info_fn=_info,
                remove_fn=lambda delete_file: _remove_tree(_root(), delete_file=delete_file),
                cache_kind="managed bundle",
            )

        out[bundle_name] = _make_bundle()

        for asset_key, asset_spec in sorted(spec["assets"].items()):
            asset_name = f"canine-bulk/{source_id}/{asset_key}"

            def _make_asset(
                _source_id: str = source_id,
                _version: str = version,
                _asset_key: str = asset_key,
                _asset_spec: dict[str, Any] = asset_spec,
                _terms: str = terms,
                _asset_name: str = asset_name,
            ) -> ManagedDataset:
                def _path() -> Path:
                    return (
                        data_dir() / "canine_bulk" / _source_id / _version / _asset_spec["filename"]
                    )

                def _files() -> list[dict[str, Any]]:
                    path = _path()
                    return [
                        {
                            "name": _asset_spec["filename"],
                            "path": path,
                            "exists": path.is_file(),
                            "expected_size_bytes": _asset_spec["size_bytes"],
                        }
                    ]

                def _fetch(force: bool) -> Path:
                    if force and _path().exists():
                        _path().unlink()
                    return fetch_canine_bulk_asset(_source_id, _asset_key)

                def _info() -> dict[str, Any]:
                    path = _path()
                    installed = path.is_file()
                    return {
                        "name": _asset_name,
                        "type": "managed dataset",
                        "description": f"Canine bulk {_source_id} {_asset_key} ({_asset_spec['filename']})",
                        "path": str(path),
                        "exists": installed,
                        "status": "installed" if installed else "not installed",
                        "size_bytes": _asset_spec["size_bytes"],
                        "sha256": _asset_spec["sha256"],
                        "terms": _terms,
                    }

                return ManagedDataset(
                    name=_asset_name,
                    kind="file",
                    description=f"Canine bulk {_source_id} {_asset_key} ({_asset_spec['filename']})",
                    terms=_terms,
                    path_fn=_path,
                    files_fn=_files,
                    exists_fn=lambda: _path().is_file(),
                    fetch_fn=_fetch,
                    info_fn=_info,
                    remove_fn=lambda delete_file: _remove_tree(_path(), delete_file=delete_file),
                    cache_kind="managed dataset",
                )

            out[asset_name] = _make_asset()

    return out


def _pxd013455_datasets() -> dict[str, ManagedDataset]:
    from .downloads import data_dir
    from .pxd013455 import CATALOG as RAW_CATALOG
    from .pxd013455 import recover_pxd013455_sources
    from .pxd013455_download import PUBLICATION, load_pxd013455_bundle

    pub = load_curation_yaml(PUBLICATION)
    raw_cat = load_curation_yaml(RAW_CATALOG)

    def _pub_root() -> Path:
        return data_dir() / "scoped-proteomics" / "pxd013455-hela-tryp46" / "published-download"

    def _pub_inspect() -> datacache.ArchiveInspection:
        return datacache.inspect_archive(
            _pub_root(),
            pub["url"],
            expected_sha256=pub["sha256"],
            expected_size=pub["size_bytes"],
            verify_files=False,
        )

    def _pub_path() -> Path:
        state = _pub_inspect()
        if state.status == "available" and state.bundle:
            return Path(state.bundle)
        return _pub_root()

    def _pub_files() -> list[dict[str, Any]]:
        state = _pub_inspect()
        return [
            {
                "name": Path(pub["url"]).name,
                "path": _pub_path(),
                "exists": state.status == "available",
                "expected_size_bytes": pub["size_bytes"],
            }
        ]

    def _pub_info() -> dict[str, Any]:
        state = _pub_inspect()
        return {
            "name": "pxd013455",
            "type": "managed bundle",
            "description": "PXD013455 HeLa tryptic digest portable publication bundle",
            "path": str(_pub_path()),
            "exists": state.status == "available",
            "status": "installed" if state.status == "available" else state.status,
            "cache_status": state.status,
            "verified": state.verified,
            "size_bytes": pub["size_bytes"],
            "sha256": pub["sha256"],
            "terms": "https://www.ebi.ac.uk/pride/archive/projects/PXD013455",
            "error": str(state.error) if state.error else None,
        }

    def _raw_root() -> Path:
        return data_dir() / "scoped-proteomics" / "pxd013455-hela-tryp46"

    def _raw_files() -> list[dict[str, Any]]:
        root = _raw_root()
        items: list[dict[str, Any]] = []
        for fname, spec in sorted({**raw_cat["files"], **raw_cat["supporting_files"]}.items()):
            dest = root / "source-files" / spec.get("original_name", fname)
            items.append(
                {
                    "name": dest.name,
                    "path": dest,
                    "exists": dest.is_file(),
                    "expected_size_bytes": spec["size_bytes"],
                }
            )
        for mname, spec in sorted(raw_cat["members"].items()):
            mdir = root / "result-members" / mname
            items.append(
                {
                    "name": f"{mname}.txt",
                    "path": mdir,
                    "exists": mdir.is_dir() and any(mdir.iterdir()),
                    "expected_size_bytes": spec["size_bytes"] + spec["compressed_size_bytes"],
                }
            )
        return items

    def _raw_info() -> dict[str, Any]:
        files = _raw_files()
        installed = all(f["exists"] for f in files)
        return {
            "name": "pxd013455-sources",
            "type": "managed bundle",
            "description": "PXD013455 HeLa raw PRIDE/Europe PMC recovery sources (~3.7 GB expanded)",
            "path": str(_raw_root()),
            "exists": installed,
            "status": "installed" if installed else "not installed",
            "size_bytes": sum(f["expected_size_bytes"] for f in files),
            "files": [f["name"] for f in files],
            "terms": "https://www.ebi.ac.uk/pride/archive/projects/PXD013455",
        }

    def _raw_fetch(force: bool) -> Path:
        if force:
            for sub in ("source-files", "result-members", "paper-supplements"):
                target = _raw_root() / sub
                if target.exists():
                    shutil.rmtree(target)
        recover_pxd013455_sources()
        return _raw_root()

    return {
        "pxd013455": ManagedDataset(
            name="pxd013455",
            kind="bundle",
            description="PXD013455 HeLa tryptic digest portable publication bundle",
            terms="https://www.ebi.ac.uk/pride/archive/projects/PXD013455",
            path_fn=_pub_path,
            files_fn=_pub_files,
            exists_fn=lambda: _pub_inspect().status == "available",
            fetch_fn=lambda force: load_pxd013455_bundle(force=force),
            info_fn=_pub_info,
            remove_fn=lambda delete_file: _remove_tree(_pub_root(), delete_file=delete_file),
            cache_kind="managed bundle",
        ),
        "pxd013455-sources": ManagedDataset(
            name="pxd013455-sources",
            kind="bundle",
            description="PXD013455 HeLa raw PRIDE/Europe PMC recovery sources (~3.7 GB expanded)",
            terms="https://www.ebi.ac.uk/pride/archive/projects/PXD013455",
            path_fn=_raw_root,
            files_fn=_raw_files,
            exists_fn=lambda: all(f["exists"] for f in _raw_files()),
            fetch_fn=_raw_fetch,
            info_fn=_raw_info,
            remove_fn=lambda delete_file: _remove_tree(_raw_root(), delete_file=delete_file),
            cache_kind="managed bundle",
        ),
    }


def _bergseng_datasets() -> dict[str, ManagedDataset]:
    from .bergseng_2015_dq import bergseng_2015_source_contract, fetch_bergseng_2015_source
    from .downloads import data_dir

    contract = bergseng_2015_source_contract()
    asset = contract["asset"]

    def root_fn() -> Path:
        return data_dir() / "bergseng_2015_dq"

    def path_fn() -> Path:
        return root_fn() / asset["filename"]

    def _files() -> list[dict[str, Any]]:
        target = path_fn()
        return [
            {
                "name": asset["filename"],
                "path": target,
                "exists": target.is_file(),
                "expected_size_bytes": asset["size_bytes"],
            }
        ]

    def _info() -> dict[str, Any]:
        target = path_fn()
        return {
            "name": "bergseng-2015-dq",
            "type": "managed file",
            "description": "Bergseng 2015 (PMID 25502872, CC-BY 4.0) HLA-DQ run-level MS1 ion volumes",
            "path": str(target),
            "exists": target.is_file(),
            "status": "installed" if target.is_file() else "not installed",
            "size_bytes": asset["size_bytes"],
            "files": [asset["filename"]],
            "terms": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4297300/",
        }

    return {
        "bergseng-2015-dq": ManagedDataset(
            name="bergseng-2015-dq",
            kind="file",
            description="Bergseng 2015 (PMID 25502872, CC-BY 4.0) HLA-DQ run-level MS1 ion volumes",
            terms="https://pmc.ncbi.nlm.nih.gov/articles/PMC4297300/",
            path_fn=path_fn,
            files_fn=_files,
            exists_fn=lambda: path_fn().is_file(),
            fetch_fn=lambda force: fetch_bergseng_2015_source(force=force),
            info_fn=_info,
            remove_fn=lambda delete_file: _remove_tree(root_fn(), delete_file=delete_file),
            cache_kind="managed source",
        )
    }


def _dodm_ratio_datasets() -> dict[str, ManagedDataset]:
    from .dohm_ratio_peptidomics import dodm_ratio_source_contract, fetch_dodm_ratio_source
    from .downloads import data_dir

    contract = dodm_ratio_source_contract()
    asset = contract["asset"]

    def root_fn() -> Path:
        return data_dir() / "dodm_ratio_35085787"

    def path_fn() -> Path:
        return root_fn() / asset["filename"]

    def _files() -> list[dict[str, Any]]:
        target = path_fn()
        return [
            {
                "name": asset["filename"],
                "path": target,
                "exists": target.is_file(),
                "expected_size_bytes": asset["size_bytes"],
            }
        ]

    def _info() -> dict[str, Any]:
        target = path_fn()
        return {
            "name": "dodm-ratio-35085787",
            "type": "managed file",
            "description": "DO:DM ratio MHC-II immunopeptidomics Dataset S3 (PMID 35085787, PXD024392, CC-BY 4.0)",
            "path": str(target),
            "exists": target.is_file(),
            "status": "installed" if target.is_file() else "not installed",
            "size_bytes": asset["size_bytes"],
            "files": [asset["filename"]],
            "terms": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10329146/",
        }

    return {
        "dodm-ratio-35085787": ManagedDataset(
            name="dodm-ratio-35085787",
            kind="file",
            description="DO:DM ratio MHC-II immunopeptidomics Dataset S3 (PMID 35085787, PXD024392, CC-BY 4.0)",
            terms="https://pmc.ncbi.nlm.nih.gov/articles/PMC10329146/",
            path_fn=path_fn,
            files_fn=_files,
            exists_fn=lambda: path_fn().is_file(),
            fetch_fn=lambda force: fetch_dodm_ratio_source(force=force),
            info_fn=_info,
            remove_fn=lambda delete_file: _remove_tree(root_fn(), delete_file=delete_file),
            cache_kind="managed source",
        )
    }


def _iso18_and_neotoma_datasets() -> dict[str, ManagedDataset]:
    from .downloads import data_dir
    from .drug_studies_qualification import (
        drug_studies_qualification_contract,
        fetch_neotoma_2021_sources,
    )
    from .iso18_calibration import (
        fetch_iso18_calibration_sources,
        iso18_calibration_source_contract,
    )

    iso_contract = iso18_calibration_source_contract()
    iso_keys = [
        "sd01",
        "sd03",
        "sd06",
        "sd07",
        "pride_direct_spike",
        "pride_hipmhc_spikein",
        "pride_skmel5_dmso_meki",
    ]
    neotoma_pin = drug_studies_qualification_contract()["neotoma_2021_source_pin"]

    def iso_root() -> Path:
        return data_dir() / "stopfer_2021_iso18"

    def _iso_files() -> list[dict[str, Any]]:
        root = iso_root()
        out = []
        for k in iso_keys:
            fname = iso_contract["verified_assets"][k]["filename"]
            p = root / fname
            out.append(
                {"name": fname, "path": p, "exists": p.is_file(), "expected_size_bytes": None}
            )
        return out

    def _iso_exists() -> bool:
        return all(item["exists"] for item in _iso_files())

    def _iso_info() -> dict[str, Any]:
        files = _iso_files()
        installed = all(item["exists"] for item in files)
        return {
            "name": "stopfer-2021-iso18",
            "type": "managed bundle",
            "description": (
                "Stopfer 2021 (PMID 34497125, PXD024917) Iso18 known-dose/raw-signal calibration "
                "and SKMEL5 binimetinib vs DMSO Skyline transitions"
            ),
            "path": str(iso_root()),
            "exists": installed,
            "status": "installed" if installed else "not installed",
            "files": [item["name"] for item in files],
            "terms": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8449407/",
        }

    def _iso_fetch(force: bool) -> Path:
        fetch_iso18_calibration_sources(dest_dir=iso_root(), force=force)
        return iso_root()

    def neotoma_root() -> Path:
        return data_dir() / "pollock_2021_neotoma"

    def _neotoma_files() -> list[dict[str, Any]]:
        root = neotoma_root()
        out = []
        for meta in neotoma_pin["verified_assets"].values():
            p = root / meta["filename"]
            out.append(
                {
                    "name": meta["filename"],
                    "path": p,
                    "exists": p.is_file(),
                    "expected_size_bytes": None,
                }
            )
        return out

    def _neotoma_exists() -> bool:
        return all(item["exists"] for item in _neotoma_files())

    def _neotoma_info() -> dict[str, Any]:
        files = _neotoma_files()
        installed = all(item["exists"] for item in files)
        return {
            "name": "pollock-2021-neotoma",
            "type": "managed bundle",
            "description": (
                "Pollock 2021 (PMID 34129938, NeoToma2021@f7df59fd) idAdpgkG MC38 factorial "
                "dTAG-13/doxycycline TOMAHAQ MS3 and untargeted TMT6 MS2 tables"
            ),
            "path": str(neotoma_root()),
            "exists": installed,
            "status": "installed" if installed else "not installed",
            "files": [item["name"] for item in files],
            "terms": "https://github.com/sbpollock/NeoToma2021",
        }

    def _neotoma_fetch(force: bool) -> Path:
        fetch_neotoma_2021_sources(dest_dir=neotoma_root(), force=force)
        return neotoma_root()

    return {
        "stopfer-2021-iso18": ManagedDataset(
            name="stopfer-2021-iso18",
            kind="bundle",
            description=(
                "Stopfer 2021 (PMID 34497125, PXD024917) Iso18 known-dose/raw-signal calibration "
                "and SKMEL5 binimetinib vs DMSO Skyline transitions"
            ),
            terms="https://pmc.ncbi.nlm.nih.gov/articles/PMC8449407/",
            path_fn=iso_root,
            files_fn=_iso_files,
            exists_fn=_iso_exists,
            fetch_fn=_iso_fetch,
            info_fn=_iso_info,
            remove_fn=lambda delete_file: _remove_tree(iso_root(), delete_file=delete_file),
            cache_kind="managed source",
        ),
        "pollock-2021-neotoma": ManagedDataset(
            name="pollock-2021-neotoma",
            kind="bundle",
            description=(
                "Pollock 2021 (PMID 34129938, NeoToma2021@f7df59fd) idAdpgkG MC38 factorial "
                "dTAG-13/doxycycline TOMAHAQ MS3 and untargeted TMT6 MS2 tables"
            ),
            terms="https://github.com/sbpollock/NeoToma2021",
            path_fn=neotoma_root,
            files_fn=_neotoma_files,
            exists_fn=_neotoma_exists,
            fetch_fn=_neotoma_fetch,
            info_fn=_neotoma_info,
            remove_fn=lambda delete_file: _remove_tree(neotoma_root(), delete_file=delete_file),
            cache_kind="managed source",
        ),
    }


def _sarango_2022_datasets() -> dict[str, ManagedDataset]:
    from .downloads import data_dir
    from .sarango_2022_pxd024417 import fetch_sarango_2022_sdrf, sarango_2022_source_contract

    contract = sarango_2022_source_contract()
    asset = contract["pride_inventory"]["verified_sdrf_asset"]

    def root_fn() -> Path:
        return data_dir() / "sarango_2022_pxd024417"

    def path_fn() -> Path:
        return root_fn() / asset["filename"]

    def _files() -> list[dict[str, Any]]:
        target = path_fn()
        return [
            {
                "name": asset["filename"],
                "path": target,
                "exists": target.is_file(),
                "expected_size_bytes": asset["size_bytes"],
            }
        ]

    def _info() -> dict[str, Any]:
        target = path_fn()
        return {
            "name": "sarango-2022-pxd024417-sdrf",
            "type": "managed file",
            "description": (
                "Sarango 2022 (PMID 36215666, PXD024417) community-annotated SDRF run/replicate map"
            ),
            "path": str(target),
            "exists": target.is_file(),
            "status": "installed" if target.is_file() else "not installed",
            "size_bytes": asset["size_bytes"],
            "files": [asset["filename"]],
            "terms": "https://www.ebi.ac.uk/pride/archive/projects/PXD024417",
        }

    return {
        "sarango-2022-pxd024417-sdrf": ManagedDataset(
            name="sarango-2022-pxd024417-sdrf",
            kind="file",
            description=(
                "Sarango 2022 (PMID 36215666, PXD024417) community-annotated SDRF run/replicate map"
            ),
            terms="https://www.ebi.ac.uk/pride/archive/projects/PXD024417",
            path_fn=path_fn,
            files_fn=_files,
            exists_fn=lambda: path_fn().is_file(),
            fetch_fn=lambda force: fetch_sarango_2022_sdrf(force=force),
            info_fn=_info,
            remove_fn=lambda delete_file: _remove_tree(root_fn(), delete_file=delete_file),
            cache_kind="managed source",
        )
    }


def managed_datasets() -> dict[str, ManagedDataset]:
    """Return every catalog-managed external dataset and bundle."""
    entries: dict[str, ManagedDataset] = {}
    for loader in (
        _uniprot_datasets,
        _atlas_datasets,
        _tcell_datasets,
        _perturbation_datasets,
        _dlbcl_datasets,
        _canine_bulk_datasets,
        _pxd013455_datasets,
        _bergseng_datasets,
        _dodm_ratio_datasets,
        _iso18_and_neotoma_datasets,
        _sarango_2022_datasets,
    ):
        entries.update(loader())
    return entries


def get_managed_dataset(name: str) -> ManagedDataset | None:
    """Look up *name* in the managed external dataset catalog."""
    return managed_datasets().get(name)


def preflight_managed(name: str, *, force: bool = False) -> dict[str, Any]:
    """Return the uniform preflight plan dict for managed dataset *name*."""
    dataset = get_managed_dataset(name)
    if dataset is None:
        raise KeyError(f"Unknown managed dataset '{name}'")
    return _build_plan(dataset, force=force)


def fetch_managed(name: str, *, force: bool = False) -> Path:
    """Preflight space, print terms notice, and fetch managed dataset *name*."""
    from .downloads import _ensure_enough_space

    dataset = get_managed_dataset(name)
    if dataset is None:
        raise KeyError(f"Unknown managed dataset '{name}'")
    plan = _build_plan(dataset, force=force)
    if plan["would_download"]:
        _ensure_enough_space(name, plan)
        if dataset.terms:
            print(
                f"  [{name}] from a terms-of-use-governed source — please review "
                f"usage/citation terms: {dataset.terms}",
                file=sys.stderr,
            )
    return _wrap_managed_errors(name, lambda: dataset.fetch_fn(force))


def get_managed_path(name: str) -> Path:
    """Return the installed path for managed dataset *name*, or raise FileNotFoundError."""
    dataset = get_managed_dataset(name)
    if dataset is None:
        raise KeyError(f"Unknown managed dataset '{name}'")
    if not dataset.exists_fn():
        raise FileNotFoundError(
            f"Managed dataset '{name}' is not installed; fetch with: hitlist data fetch {name}"
        )
    return dataset.path_fn()


def installed_managed_cache_entries() -> list[tuple[Path, str, str, dict[str, Any]]]:
    """Return installed managed-bundle files for :func:`hitlist.cache_inventory.list_cache_files`.

    Skips UniProt references (which ``cache_inventory`` already lists with
    symlink-escape checks) and uninstalled entries (so ``include_unregistered=False``
    never emits missing-bundle noise).
    """
    rows: list[tuple[Path, str, str, dict[str, Any]]] = []
    for name, dataset in sorted(managed_datasets().items()):
        if name.startswith("uniprot/"):
            continue
        if not dataset.exists_fn():
            continue
        for item in dataset.files_fn():
            path = Path(item["path"])
            if path.is_file():
                meta = (
                    {"size_bytes": item["expected_size_bytes"]}
                    if item["expected_size_bytes"] is not None
                    else {}
                )
                rows.append((path, dataset.cache_kind, name, meta))
    return rows
