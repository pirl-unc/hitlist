"""Unified YAML-backed study qualification registry and pinned-asset fetcher (#986).

Replaces the 12 legacy per-paper ``hitlist/*_qualification.py`` modules with one
validated YAML registry (:file:`hitlist/data/study_qualifications.yaml`) and one
shared contract/asset loader.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import tempfile
import urllib.request
import zipfile
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from .curation_yaml import load_curation_yaml
from .errors import IntegrityError

_DATA_DIR = Path(__file__).parent / "data"
_REGISTRY_PATH = _DATA_DIR / "study_qualifications.yaml"

VALID_STATUSES = frozenset(
    {
        "admitted",
        "audit_only",
        "qualified_pending_public_data",
        "embargoed_or_restricted",
    }
)
VALID_FETCH_MODES = frozenset({"none", "direct_files", "zip_extract", "remote_zip_range"})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_registry_raw() -> dict[str, dict[str, Any]]:
    raw = load_curation_yaml(_REGISTRY_PATH)
    if not isinstance(raw, dict) or not isinstance(raw.get("studies"), dict):
        raise ValueError(f"Invalid study_qualifications.yaml structure in {_REGISTRY_PATH}")
    studies: dict[str, dict[str, Any]] = {}
    for study_id, entry in raw["studies"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Study {study_id!r} must map to a dict")
        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Study {study_id!r} has invalid status {status!r}")
        fetch_mode = entry.get("fetch_mode")
        if fetch_mode not in VALID_FETCH_MODES:
            raise ValueError(f"Study {study_id!r} has invalid fetch_mode {fetch_mode!r}")
        contract_rel = entry.get("contract_file")
        if not contract_rel or not (_DATA_DIR / contract_rel).is_file():
            raise ValueError(
                f"Study {study_id!r} references missing contract_file {contract_rel!r}"
            )
        assets_rel = entry.get("assets_file")
        if assets_rel and not (_DATA_DIR / assets_rel).is_file():
            raise ValueError(f"Study {study_id!r} references missing assets_file {assets_rel!r}")
        studies[str(study_id)] = dict(entry, study_id=str(study_id))
    return studies


def load_study_qualifications() -> dict[str, dict[str, Any]]:
    """Return a deep copy of the validated study qualification registry."""
    return copy.deepcopy(_load_registry_raw())


def list_study_qualifications() -> list[str]:
    """Return sorted registered study qualification identifiers."""
    return sorted(_load_registry_raw())


def get_study_qualification(study_id: str) -> dict[str, Any]:
    """Return registry metadata for ``study_id`` (raises ``KeyError`` if unknown)."""
    registry = _load_registry_raw()
    if study_id not in registry:
        raise KeyError(f"Unknown study qualification {study_id!r}; known: {sorted(registry)}")
    return copy.deepcopy(registry[study_id])


def _load_data_file(relative_path: str) -> dict[str, Any]:
    path = _DATA_DIR / relative_path
    if path.suffix in {".yaml", ".yml"}:
        data = load_curation_yaml(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_study_contract(study_id: str) -> dict[str, Any]:
    """Load the pinned contract dict for ``study_id``."""
    meta = get_study_qualification(study_id)
    data = _load_data_file(meta["contract_file"])
    section = meta.get("contract_section")
    if section:
        return copy.deepcopy(data[section])
    return copy.deepcopy(data)


def load_study_assets_manifest(study_id: str) -> dict[str, Any]:
    """Load the pinned asset manifest dict for ``study_id``."""
    meta = get_study_qualification(study_id)
    assets_rel = meta.get("assets_file")
    if not assets_rel:
        raise ValueError(f"Study {study_id!r} does not declare a separate assets_file")
    return copy.deepcopy(_load_data_file(assets_rel))


def study_assets_manifest_path(study_id: str) -> Path:
    """Return the absolute path to the pinned ``assets_file`` for ``study_id``."""
    meta = get_study_qualification(study_id)
    assets_rel = meta.get("assets_file")
    if not assets_rel:
        raise ValueError(f"Study {study_id!r} does not declare a separate assets_file")
    return _DATA_DIR / assets_rel


# ── Shared HTTP Range ZIP extraction (PXD007679 / PXD004452) ─────────────────


def _http_range_get(url: str, start: int, end_inclusive: int, *, timeout: float = 60.0) -> bytes:
    expected_len = end_inclusive - start + 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", None)
        if status is not None and status != 206:
            raise OSError(f"Expected HTTP 206 Partial Content for {url}, got HTTP {status}")
        data = resp.read(expected_len + 1)
    if len(data) != expected_len:
        raise OSError(
            f"HTTP range length mismatch for {url} ({start}-{end_inclusive}): "
            f"expected {expected_len}, got {len(data)}"
        )
    return data


def _parse_zip_central_directory(cd_bytes: bytes) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    pos = 0
    total = len(cd_bytes)
    while pos + 46 <= total:
        if struct.unpack_from("<I", cd_bytes, pos)[0] != 0x02014B50:
            break
        (
            _vm,
            _vn,
            _fl,
            method,
            _mt,
            _md,
            crc32,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
            comment_len,
            _ds,
            _ia,
            _ea,
            local_offset,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd_bytes, pos + 4)
        name_start = pos + 46
        name = cd_bytes[name_start : name_start + name_len].decode("utf-8", errors="replace")
        extra = cd_bytes[name_start + name_len : name_start + name_len + extra_len]
        if uncomp_size == 0xFFFFFFFF or comp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            epos = 0
            while epos + 4 <= len(extra):
                hdr_id, data_size = struct.unpack_from("<HH", extra, epos)
                epos += 4
                chunk = extra[epos : epos + data_size]
                epos += data_size
                if hdr_id == 0x0001:
                    cpos = 0
                    if uncomp_size == 0xFFFFFFFF and cpos + 8 <= len(chunk):
                        uncomp_size = struct.unpack_from("<Q", chunk, cpos)[0]
                        cpos += 8
                    if comp_size == 0xFFFFFFFF and cpos + 8 <= len(chunk):
                        comp_size = struct.unpack_from("<Q", chunk, cpos)[0]
                        cpos += 8
                    if local_offset == 0xFFFFFFFF and cpos + 8 <= len(chunk):
                        local_offset = struct.unpack_from("<Q", chunk, cpos)[0]
                    break
        entries.append(
            {
                "name": name,
                "method": method,
                "crc32": crc32,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "local_offset": local_offset,
            }
        )
        pos = name_start + name_len + extra_len + comment_len
    return entries


def _extract_remote_zip_members(
    url: str,
    archive_size: int,
    member_suffixes: dict[str, str],
    *,
    range_getter: Any = None,
) -> dict[str, bytes]:
    """Extract selected members from a remote ZIP/ZIP64 archive via HTTP Range."""
    fetch_range = range_getter or _http_range_get
    tail_size = min(archive_size, 65536 + 128)
    tail = fetch_range(url, archive_size - tail_size, archive_size - 1)
    eocd_rel = tail.rfind(b"PK\x05\x06")
    if eocd_rel < 0:
        raise ValueError(f"ZIP EOCD signature not found in remote archive {url}")
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd_rel + 12)
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        loc_rel = tail.rfind(b"PK\x06\x07", 0, eocd_rel)
        if loc_rel < 0:
            raise ValueError(f"ZIP64 EOCD locator missing in remote archive {url}")
        zip64_offset = struct.unpack_from("<Q", tail, loc_rel + 8)[0]
        z64 = fetch_range(url, zip64_offset, zip64_offset + 55)
        if z64[:4] != b"PK\x06\x06":
            raise ValueError(f"ZIP64 EOCD record signature mismatch in {url}")
        cd_size, cd_offset = struct.unpack_from("<QQ", z64, 40)

    entries = _parse_zip_central_directory(fetch_range(url, cd_offset, cd_offset + cd_size - 1))
    extracted: dict[str, bytes] = {}
    for target_name, suffix in member_suffixes.items():
        norm_suffix = suffix.lstrip("/").lower()
        match = next(
            (
                e
                for e in entries
                if e["name"].replace("\\", "/").lstrip("/").lower() == norm_suffix
                or e["name"].replace("\\", "/").lstrip("/").lower().endswith("/" + norm_suffix)
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Member matching suffix {suffix!r} not found in {url}")
        local_offset = match["local_offset"]
        lfh = fetch_range(url, local_offset, local_offset + 29)
        if lfh[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP local file header for {match['name']}")
        lfh_name_len, lfh_extra_len = struct.unpack_from("<HH", lfh, 26)
        data_start = local_offset + 30 + lfh_name_len + lfh_extra_len
        comp_bytes = (
            b""
            if match["comp_size"] == 0
            else fetch_range(url, data_start, data_start + match["comp_size"] - 1)
        )
        if match["method"] == 0:
            raw = comp_bytes
        elif match["method"] == 8:
            raw = zlib.decompress(comp_bytes, -zlib.MAX_WBITS)
        else:
            raise ValueError(f"Unsupported ZIP compression method {match['method']}")
        if len(raw) != match["uncomp_size"]:
            raise ValueError(f"Decompressed size mismatch for {target_name}")
        if (zlib.crc32(raw) & 0xFFFFFFFF) != match["crc32"]:
            raise ValueError(f"CRC32 mismatch for {target_name}")
        extracted[target_name] = raw
    return extracted


def _fetch_remote_zip_members_verified(
    url: str,
    archive_size: int | None,
    members: dict[str, dict[str, Any]],
    target_dir: Path,
    *,
    force: bool = False,
    offline: bool | None = None,
    range_getter: Any = None,
) -> Path:
    from .downloads import _require_online

    target_dir.mkdir(parents=True, exist_ok=True)
    needed = {
        fname: meta["member_suffix"]
        for fname, meta in members.items()
        if force or not (target_dir / fname).is_file()
    }
    if needed:
        _require_online(f"remote ZIP members from {url}", offline=offline)
        size = archive_size
        if not size:
            if range_getter is not None:
                raise ValueError("archive_size required with custom range_getter")
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                size = int(resp.headers["Content-Length"])
        extracted = _extract_remote_zip_members(url, int(size), needed, range_getter=range_getter)
        for fname, payload in extracted.items():
            meta = members[fname]
            if _sha256_bytes(payload) != meta["sha256"] or len(payload) != int(meta["size_bytes"]):
                raise IntegrityError(f"Integrity mismatch for {fname}")
            tmp = target_dir / f".{fname}.part"
            tmp.write_bytes(payload)
            tmp.replace(target_dir / fname)

    for fname, meta in members.items():
        path = target_dir / fname
        if _sha256_file(path) != meta["sha256"] or path.stat().st_size != int(meta["size_bytes"]):
            raise IntegrityError(f"Cached {fname} failed SHA-256 verification")
    return target_dir


# ── Unified Study Asset Fetcher ──────────────────────────────────────────────


def dlbcl_supplement_contract() -> dict[str, Any]:
    """Return the pinned Bourne 2022 DLBCL Tables 3 & 5 supplement contract (#901)."""
    return load_study_contract("dlbcl_35561310")


def fetch_dlbcl_supplements(
    cache_directory: str | Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Fetch and SHA-256-verify PMC9327544 Tables 3 & 5 workbooks from Europe PMC (#901)."""
    from .downloads import data_dir, download_to_file

    contract = dlbcl_supplement_contract()
    workbooks = contract["workbooks"]
    directory = (
        Path(cache_directory)
        if cache_directory is not None
        else data_dir() / "dlbcl_35561310_supplements"
    )
    directory.mkdir(parents=True, exist_ok=True)
    targets = {k: (directory / meta["filename"], meta) for k, meta in workbooks.items()}
    if not force and all(
        p.is_file()
        and _sha256_file(p) == meta["sha256"]
        and p.stat().st_size == int(meta["size_bytes"])
        for p, meta in targets.values()
    ):
        return {meta["filename"]: p for p, meta in targets.values()}

    source_url = workbooks["table_3_class_i"]["source_url"]
    with tempfile.TemporaryDirectory(dir=directory, prefix=".dlbcl-fetch-") as tmp:
        archive_path = Path(tmp) / "PMC9327544_supplementaryFiles.zip"
        download_to_file(
            source_url,
            archive_path,
            label="dlbcl-35561310-supplements",
            force=True,
            verbose=verbose,
            decompress=False,
        )
        with zipfile.ZipFile(archive_path) as zf:
            members = {Path(name).name: name for name in zf.namelist()}
            staged: list[tuple[Path, Path]] = []
            for path, meta in targets.values():
                member = members.get(meta["filename"])
                if member is None:
                    raise IntegrityError(f"Archive {source_url} missing {meta['filename']!r}")
                staged_path = Path(tmp) / meta["filename"]
                with zf.open(member) as src, staged_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if _sha256_file(staged_path) != meta["sha256"] or staged_path.stat().st_size != int(
                    meta["size_bytes"]
                ):
                    raise IntegrityError(f"DLBCL checksum mismatch for {meta['filename']}")
                staged.append((staged_path, path))
            for staged_path, final_path in staged:
                os.replace(staged_path, final_path)
    return {meta["filename"]: p for p, meta in targets.values()}


def fetch_study_assets(
    study_id: str,
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
    offline: bool | None = None,
    include_optional: bool = False,
    range_getter: Any = None,
) -> Path:
    """Fetch and SHA-256-verify pinned public assets for ``study_id``."""
    from .downloads import data_dir, download_to_file

    meta = get_study_qualification(study_id)
    subdir = meta.get("cache_subdir")
    if not subdir or meta["fetch_mode"] == "none":
        raise ValueError(f"Study {study_id!r} has no auto-downloadable asset bundle")
    target_dir = Path(dest_dir) if dest_dir is not None else data_dir() / subdir

    if study_id == "dlbcl_35561310":
        fetch_dlbcl_supplements(cache_directory=target_dir, force=force, verbose=verbose)
        return target_dir

    if study_id == "drug_studies_903":
        pin = load_study_contract("drug_studies_903")["neotoma_2021_source_pin"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for asset in pin["verified_assets"].values():
            download_to_file(
                asset["url"],
                target_dir / asset["filename"],
                label=f"NeoToma2021/{asset['filename']}",
                force=force,
                expected_sha256=asset["sha256"],
                offline=offline,
            )
        return target_dir

    if study_id == "erap2_oe_29769354":
        contract = load_study_contract("erap2_oe_29769354")
        archive = contract["pride_inventory"]["lentiviral_oe_archive"]
        _fetch_remote_zip_members_verified(
            archive["url"],
            int(archive["size_bytes"]),
            archive["verified_extracted_members"],
            target_dir,
            force=force,
            offline=offline,
            range_getter=range_getter,
        )
        sdrf = contract["pride_inventory"]["verified_sdrf_asset"]
        download_to_file(
            sdrf["url"],
            target_dir / sdrf["filename"],
            label=f"PXD007679/{sdrf['filename']}",
            force=force,
            expected_sha256=sdrf["sha256"],
            expected_size=int(sdrf["size_bytes"]),
            offline=offline,
        )
        return target_dir

    if study_id == "bekker_jensen_2017":
        contract = load_study_contract("bekker_jensen_2017")
        archive = contract["archive"]
        members = {k: archive["verified_members"][k] for k in ("parameters.txt", "summary.txt")}
        return _fetch_remote_zip_members_verified(
            archive["url"],
            int(archive.get("size_bytes", 0)) or None,
            members,
            target_dir,
            force=force,
            offline=offline,
            range_getter=range_getter,
        )

    if study_id == "mapp_unified":
        manifest = load_study_assets_manifest("mapp_unified")
        target_dir.mkdir(parents=True, exist_ok=True)
        large_roles = {"complete_publisher_supplement", "original_mapp_search_results"}
        for asset in manifest["assets"]:
            if not include_optional and asset.get("role") in large_roles:
                continue
            url = asset["source_url"]
            if url.startswith("ftp://ftp.pride.ebi.ac.uk/"):
                url = "https://" + url[len("ftp://") :]
            download_to_file(
                url,
                target_dir / asset["filename"],
                label=f"PXD052314/{asset['filename']}",
                force=force,
                expected_sha256=asset["sha256"],
                expected_size=int(asset["size_bytes"]),
                offline=offline,
            )
        return target_dir

    if study_id == "pxd058121_intact_degradomics":
        manifest = load_study_assets_manifest("pxd058121_intact_degradomics")
        target_dir.mkdir(parents=True, exist_ok=True)
        for asset in manifest["assets"]:
            download_to_file(
                asset["source_url"],
                target_dir / asset["filename"],
                label=f"PXD058121/{asset['filename']}",
                force=force,
                expected_sha256=asset["sha256"],
                expected_size=int(asset["size_bytes"]),
                offline=offline,
            )
        return target_dir

    if study_id == "pxd018722_sahu_psmd2":
        manifest = load_study_assets_manifest("pxd018722_sahu_psmd2")
        target_dir.mkdir(parents=True, exist_ok=True)
        allowed_roles = {"original_metadata", "primary_publication"}
        if include_optional:
            allowed_roles.add("original_reference")
        for asset in manifest["assets"]:
            if asset["role"] not in allowed_roles or "#" in asset["source_url"]:
                continue
            download_to_file(
                asset["source_url"],
                target_dir / asset["filename"],
                label=f"PXD018722/{asset['filename']}",
                force=force,
                expected_sha256=asset["sha256"],
                expected_size=int(asset["size_bytes"]),
                offline=offline,
            )
        return target_dir

    if study_id == "erap1_biochemical":
        asset = load_study_contract("erap1_biochemical")["sources"]["maben_2021"]["archive_asset"]
        target_dir.mkdir(parents=True, exist_ok=True)
        download_to_file(
            asset["url"],
            target_dir / asset["filename"],
            label="Maben2021 ERAP1 Source Data",
            force=force,
            expected_sha256=asset["sha256"],
            expected_size=int(asset["size_bytes"]),
            offline=offline,
        )
        return target_dir

    if study_id == "non_mhc_calibration":
        manifest = load_study_assets_manifest("non_mhc_calibration")
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, asset in sorted(manifest["assets"].items()):
            download_to_file(
                asset["url"],
                target_dir / filename,
                label=f"PXD004886/{filename}",
                force=force,
                expected_sha256=asset["sha256"],
                expected_size=int(asset["size"]),
                offline=offline,
            )
        return target_dir

    raise ValueError(f"Unsupported study asset bundle {study_id!r}")
