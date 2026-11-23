"""Source-qualified Sanz-Bravo 2018b ERAP2 overexpression peptidomics (PMID 29769354, PXD007679).

Implements #715 (and closes the final child of #905):
- Disambiguates the four biological/experimental arms in PMC6072538 / PRIDE PXD007679:
  1. PF97387 nontransduced (natural ERAP2-negative parental LCL; 50198-50206 .rar panel)
  2. PF-GFP (PF97387 + GFP lentiviral control, A29(-)-1..3, Seq56011..Seq56013_QE2.raw)
  3. PF-ERAP2 (PF97387 + ERAP2 lentiviral OE, A29(+)-1..3, Seq56014..Seq56016_QE2.raw)
  4. GM19397 / GM19452 natural ERAP2-positive LCLs (50198-50206 .rar panel)
- Recovers the author-deposited MaxQuant 1.5.8.3 quantitative tables directly from PRIDE
  ``PXD007679/A29-19-3-18.zip`` via HTTP Range ZIP central-directory reads (~1 MB wire
  transfer instead of downloading the 6.12 GB archive), verified against pinned SHA-256s.
- Preserves all 6,308 original MaxQuant peptide rows, all 6 run intensities, Match Between Runs
  provenance (``By MS/MS`` vs ``By matching``), shared multi-arm memberships, condition-specific
  detections (never treated as measured negatives), zero-intensity rows, and explicit exclusion
  of the pooled 4,999-row IEDB table from OE-versus-control training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import struct
import tempfile
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "sanz_bravo_2018b_erap2_29769354.json"
)
PACKAGE_FAMILY = "sanz_bravo_2018b_erap2_oe"
PACKAGE_SCHEMA_VERSION = 1

_RUN_SPEC: tuple[dict[str, Any], ...] = (
    {
        "experiment_label": "A29(-)-1",
        "raw_file": "Seq56011_QE2",
        "replicate_index": 1,
        "arm_id": "pf97387_gfp_lentiviral_control",
        "arm_label": "PF-GFP",
        "condition_control": "mock",
        "condition_overexpression_genes": "GFP",
        "condition_transduction": "lentiviral_GFP_control",
    },
    {
        "experiment_label": "A29(-)-2",
        "raw_file": "Seq56012_QE2",
        "replicate_index": 2,
        "arm_id": "pf97387_gfp_lentiviral_control",
        "arm_label": "PF-GFP",
        "condition_control": "mock",
        "condition_overexpression_genes": "GFP",
        "condition_transduction": "lentiviral_GFP_control",
    },
    {
        "experiment_label": "A29(-)-3",
        "raw_file": "Seq56013_QE2",
        "replicate_index": 3,
        "arm_id": "pf97387_gfp_lentiviral_control",
        "arm_label": "PF-GFP",
        "condition_control": "mock",
        "condition_overexpression_genes": "GFP",
        "condition_transduction": "lentiviral_GFP_control",
    },
    {
        "experiment_label": "A29(+)-1",
        "raw_file": "Seq56014_QE2",
        "replicate_index": 1,
        "arm_id": "pf97387_erap2_lentiviral_oe",
        "arm_label": "PF-ERAP2",
        "condition_control": "treated",
        "condition_overexpression_genes": "ERAP2",
        "condition_transduction": "lentiviral_ERAP2",
    },
    {
        "experiment_label": "A29(+)-2",
        "raw_file": "Seq56015_QE2",
        "replicate_index": 2,
        "arm_id": "pf97387_erap2_lentiviral_oe",
        "arm_label": "PF-ERAP2",
        "condition_control": "treated",
        "condition_overexpression_genes": "ERAP2",
        "condition_transduction": "lentiviral_ERAP2",
    },
    {
        "experiment_label": "A29(+)-3",
        "raw_file": "Seq56016_QE2",
        "replicate_index": 3,
        "arm_id": "pf97387_erap2_lentiviral_oe",
        "arm_label": "PF-ERAP2",
        "condition_control": "treated",
        "condition_overexpression_genes": "ERAP2",
        "condition_transduction": "lentiviral_ERAP2",
    },
)


def erap2_oe_source_contract() -> dict[str, Any]:
    """Return the pinned Sanz-Bravo 2018b (PMID 29769354, PXD007679) source contract."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_range_get(url: str, start: int, end_inclusive: int, *, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_zip_central_directory(
    cd_bytes: bytes,
) -> list[dict[str, Any]]:
    """Parse ZIP/ZIP64 Central Directory entries from raw CD bytes."""
    entries: list[dict[str, Any]] = []
    pos = 0
    total = len(cd_bytes)
    while pos + 46 <= total:
        sig = struct.unpack_from("<I", cd_bytes, pos)[0]
        if sig != 0x02014B50:
            break
        (
            _ver_made,
            _ver_need,
            _flags,
            method,
            _mtime,
            _mdate,
            crc32,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _int_attr,
            _ext_attr,
            local_offset,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd_bytes, pos + 4)
        name_start = pos + 46
        name = cd_bytes[name_start : name_start + name_len].decode("utf-8", errors="replace")
        extra = cd_bytes[name_start + name_len : name_start + name_len + extra_len]
        # Parse ZIP64 extended information extra field (header ID 0x0001) if needed.
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
                        cpos += 8
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
    """Extract selected members from a remote ZIP/ZIP64 archive using HTTP Range requests.

    Reads only the tail EOCD/ZIP64 locator, the Central Directory, and the exact compressed
    byte ranges for requested members — avoiding multi-GB full-archive downloads.
    """
    fetch_range = range_getter or _http_range_get
    tail_size = min(archive_size, 65536 + 128)
    tail_start = archive_size - tail_size
    tail = fetch_range(url, tail_start, archive_size - 1)

    eocd_rel = tail.rfind(b"PK\x05\x06")
    if eocd_rel < 0:
        raise ValueError(f"ZIP EOCD signature not found in remote archive {url}")
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd_rel + 12)

    # Check ZIP64 EOCD locator (PK\x06\x07) immediately preceding EOCD when 32-bit fields overflow.
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        loc_rel = tail.rfind(b"PK\x06\x07", 0, eocd_rel)
        if loc_rel < 0:
            raise ValueError(f"ZIP64 EOCD locator missing in remote archive {url}")
        zip64_eocd_offset = struct.unpack_from("<Q", tail, loc_rel + 8)[0]
        z64 = fetch_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
        if z64[:4] != b"PK\x06\x06":
            raise ValueError(f"ZIP64 EOCD record signature mismatch in {url}")
        cd_size, cd_offset = struct.unpack_from("<QQ", z64, 40)

    cd_bytes = fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    entries = _parse_zip_central_directory(cd_bytes)

    extracted: dict[str, bytes] = {}
    for target_name, suffix in member_suffixes.items():
        norm_suffix = suffix.lstrip("/").lower()
        match = None
        for entry in entries:
            norm_name = entry["name"].replace("\\", "/").lstrip("/").lower()
            if norm_name == norm_suffix or norm_name.endswith("/" + norm_suffix):
                match = entry
                break
        if match is None:
            raise ValueError(f"Member matching suffix {suffix!r} not found in remote archive {url}")
        local_offset = match["local_offset"]
        lfh = fetch_range(url, local_offset, local_offset + 29)
        if lfh[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP local file header for {match['name']}")
        lfh_name_len, lfh_extra_len = struct.unpack_from("<HH", lfh, 26)
        data_start = local_offset + 30 + lfh_name_len + lfh_extra_len
        comp_size = match["comp_size"]
        comp_bytes = (
            b"" if comp_size == 0 else fetch_range(url, data_start, data_start + comp_size - 1)
        )
        if match["method"] == 0:
            raw = comp_bytes
        elif match["method"] == 8:
            raw = zlib.decompress(comp_bytes, -zlib.MAX_WBITS)
        else:
            raise ValueError(
                f"Unsupported ZIP compression method {match['method']} for {match['name']}"
            )
        if len(raw) != match["uncomp_size"]:
            raise ValueError(
                f"Decompressed size mismatch for {target_name}: "
                f"expected {match['uncomp_size']}, got {len(raw)}"
            )
        if (zlib.crc32(raw) & 0xFFFFFFFF) != match["crc32"]:
            raise ValueError(f"CRC32 mismatch for {target_name}")
        extracted[target_name] = raw
    return extracted


def fetch_erap2_oe_maxquant_sources(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    include_sdrf: bool = True,
    offline: bool | None = None,
    range_getter: Any = None,
) -> Path:
    """Fetch and SHA-256-verify the PXD007679 MaxQuant PF-GFP vs PF-ERAP2 files (~2.6 MB).

    Extracts ``peptides.txt``, ``summary.txt``, ``parameters.txt``, and ``mqpar.xml`` from
    PRIDE ``PXD007679/A29-19-3-18.zip`` using HTTP Range requests (downloading ~1 MB over the
    wire instead of the 6.12 GB archive), plus optionally ``PXD007679_community_annotated.sdrf.tsv``.
    """
    from .downloads import _require_online, data_dir, download_to_file
    from .errors import IntegrityError

    contract = erap2_oe_source_contract()
    target_dir = (
        Path(dest_dir) if dest_dir is not None else data_dir() / "sanz_bravo_2018b_erap2_29769354"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_meta = contract["pride_inventory"]["lentiviral_oe_archive"]
    members = archive_meta["verified_extracted_members"]

    needed: dict[str, str] = {}
    for fname, meta in members.items():
        out_path = target_dir / fname
        if force or not out_path.is_file():
            needed[fname] = meta["member_suffix"]

    if needed:
        _require_online(
            f"PXD007679 MaxQuant tables from {archive_meta['url']}",
            offline=offline,
        )
        extracted = _extract_remote_zip_members(
            archive_meta["url"],
            int(archive_meta["size_bytes"]),
            needed,
            range_getter=range_getter,
        )
        for fname, payload in extracted.items():
            meta = members[fname]
            actual_sha = _sha256_bytes(payload)
            if actual_sha != meta["sha256"] or len(payload) != int(meta["size_bytes"]):
                raise IntegrityError(
                    f"Integrity mismatch for {fname} from {archive_meta['filename']}: "
                    f"expected sha256={meta['sha256']} size={meta['size_bytes']}, "
                    f"got sha256={actual_sha} size={len(payload)}"
                )
            tmp = target_dir / f".{fname}.part"
            tmp.write_bytes(payload)
            tmp.replace(target_dir / fname)

    # Verify already-present files too so corrupt cache entries are never silently reused.
    for fname, meta in members.items():
        path = target_dir / fname
        actual_sha = _sha256_file(path)
        if actual_sha != meta["sha256"] or path.stat().st_size != int(meta["size_bytes"]):
            raise IntegrityError(
                f"Cached {fname} failed SHA-256 verification: expected {meta['sha256']}, "
                f"got {actual_sha}"
            )

    if include_sdrf:
        sdrf = contract["pride_inventory"]["verified_sdrf_asset"]
        download_to_file(
            sdrf["url"],
            target_dir / sdrf["filename"],
            label="PXD007679 SDRF",
            force=force,
            expected_sha256=sdrf["sha256"],
            expected_size=int(sdrf["size_bytes"]),
            offline=offline,
        )

    return target_dir


def _parse_float(raw: str) -> float | None:
    text = raw.strip()
    if not text or text.lower() in {"nan", "na", "null"}:
        return None
    val = float(text)
    return val if math.isfinite(val) else None


def parse_erap2_oe_peptides(
    peptides_path: str | Path,
    *,
    include_reverse_or_contaminant: bool = True,
    allow_unverified_checksum: bool = False,
) -> dict[str, pd.DataFrame]:
    """Parse PXD007679/A29-19-3-18.zip ``peptides.txt`` into normalized evidence relations.

    Returns a dict with three DataFrames:
    - ``peptides``: peptide-level quantitative summary with arm membership status,
      per-arm positive/MSMS/MBR run counts, arm mean intensities, and log2 fold-change.
    - ``run_observations``: long-format per-run observations preserving raw intensities,
      explicit zero vs positive status (never measured negatives), and MBR provenance
      (``By MS/MS`` vs ``By matching``).
    - ``arm_memberships``: explicit multi-membership relation (one row per supported
      ``(peptide, arm_id)`` pair; shared peptides produce two rows, one per arm).
    """
    contract = erap2_oe_source_contract()
    pinned = contract["pride_inventory"]["lentiviral_oe_archive"]["verified_extracted_members"][
        "peptides.txt"
    ]
    path = Path(peptides_path)
    actual_sha = _sha256_file(path)
    if not allow_unverified_checksum and actual_sha != pinned["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {pinned['sha256']}, got {actual_sha}"
        )

    peptide_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required_cols = {
            "Sequence",
            "Length",
            "Proteins",
            "Leading razor protein",
            "Gene names",
            "PEP",
            "Score",
            "Reverse",
            "Potential contaminant",
            "id",
            *(f"Intensity {r['experiment_label']}" for r in _RUN_SPEC),
            *(f"Identification type {r['experiment_label']}" for r in _RUN_SPEC),
        }
        missing_cols = sorted(required_cols - set(reader.fieldnames or ()))
        if missing_cols:
            raise ValueError(f"Missing required MaxQuant columns in {path.name}: {missing_cols}")

        for source_row_idx, row in enumerate(reader, start=2):
            is_reverse = row["Reverse"].strip() == "+"
            is_contaminant = row["Potential contaminant"].strip() == "+"
            if not include_reverse_or_contaminant and (is_reverse or is_contaminant):
                continue

            seq = row["Sequence"].strip()
            mq_id = row["id"].strip()

            gfp_vals: list[float] = []
            erap2_vals: list[float] = []
            n_gfp_msms = 0
            n_gfp_mbr = 0
            n_erap2_msms = 0
            n_erap2_mbr = 0

            for spec in _RUN_SPEC:
                exp = spec["experiment_label"]
                raw_int = row[f"Intensity {exp}"].strip()
                parsed_int = _parse_float(raw_int)
                id_type = row[f"Identification type {exp}"].strip()
                is_pos = parsed_int is not None and parsed_int > 0.0
                value_status = (
                    "positive" if is_pos else ("zero" if parsed_int == 0.0 else "missing")
                )
                is_mbr = id_type == "By matching"
                if spec["arm_label"] == "PF-GFP":
                    if is_pos and parsed_int is not None:
                        gfp_vals.append(parsed_int)
                    if id_type == "By MS/MS":
                        n_gfp_msms += 1
                    elif is_mbr:
                        n_gfp_mbr += 1
                else:
                    if is_pos and parsed_int is not None:
                        erap2_vals.append(parsed_int)
                    if id_type == "By MS/MS":
                        n_erap2_msms += 1
                    elif is_mbr:
                        n_erap2_mbr += 1

                run_rows.append(
                    {
                        "study_id": "sanz_bravo_2018b_erap2_29769354",
                        "pmid": 29769354,
                        "pride_accession": "PXD007679",
                        "source_asset": path.name,
                        "source_sha256": actual_sha,
                        "source_row": source_row_idx,
                        "maxquant_id": mq_id,
                        "peptide": seq,
                        "mhc_class": "I",
                        "mhc_restriction": "HLA-A*29:02",
                        "raw_file": spec["raw_file"],
                        "experiment_label": exp,
                        "replicate_index": spec["replicate_index"],
                        "arm_id": spec["arm_id"],
                        "arm_label": spec["arm_label"],
                        "condition_control": spec["condition_control"],
                        "condition_overexpression_genes": spec["condition_overexpression_genes"],
                        "condition_transduction": spec["condition_transduction"],
                        "intensity_raw": raw_int,
                        "intensity": parsed_int,
                        "value_status": value_status,
                        "identification_type": id_type,
                        "is_mbr_transferred": is_mbr,
                        "is_reverse": is_reverse,
                        "is_contaminant": is_contaminant,
                        "nondetection_is_measured_negative": False,
                    }
                )

            n_gfp_pos = len(gfp_vals)
            n_erap2_pos = len(erap2_vals)
            if is_reverse or is_contaminant:
                status = "excluded_reverse_or_contaminant"
            elif n_gfp_pos > 0 and n_erap2_pos > 0:
                status = "shared_both_arms"
            elif n_gfp_pos > 0:
                status = "pf_gfp_only"
            elif n_erap2_pos > 0:
                status = "pf_erap2_only"
            else:
                status = "zero_intensity_all_runs"

            gfp_mean = sum(gfp_vals) / n_gfp_pos if n_gfp_pos > 0 else None
            erap2_mean = sum(erap2_vals) / n_erap2_pos if n_erap2_pos > 0 else None
            log2_fc = (
                math.log2(erap2_mean / gfp_mean)
                if (
                    gfp_mean is not None
                    and erap2_mean is not None
                    and gfp_mean > 0
                    and erap2_mean > 0
                )
                else None
            )
            shared_direct_msms = (
                status == "shared_both_arms" and n_gfp_msms > 0 and n_erap2_msms > 0
            )
            requires_mbr = status == "shared_both_arms" and not shared_direct_msms

            peptide_rows.append(
                {
                    "study_id": "sanz_bravo_2018b_erap2_29769354",
                    "pmid": 29769354,
                    "pride_accession": "PXD007679",
                    "source_asset": path.name,
                    "source_sha256": actual_sha,
                    "source_row": source_row_idx,
                    "maxquant_id": mq_id,
                    "peptide": seq,
                    "length": int(row["Length"]) if row["Length"].strip().isdigit() else len(seq),
                    "mhc_class": "I",
                    "mhc_restriction": "HLA-A*29:02",
                    "proteins": row["Proteins"].strip(),
                    "leading_razor_protein": row["Leading razor protein"].strip(),
                    "gene_names": row["Gene names"].strip(),
                    "pep": _parse_float(row["PEP"]),
                    "score": _parse_float(row["Score"]),
                    "is_reverse": is_reverse,
                    "is_contaminant": is_contaminant,
                    "n_pf_gfp_positive_runs": n_gfp_pos,
                    "n_pf_erap2_positive_runs": n_erap2_pos,
                    "n_pf_gfp_msms_runs": n_gfp_msms,
                    "n_pf_erap2_msms_runs": n_erap2_msms,
                    "n_pf_gfp_mbr_runs": n_gfp_mbr,
                    "n_pf_erap2_mbr_runs": n_erap2_mbr,
                    "arm_membership_status": status,
                    "pf_gfp_mean_intensity": gfp_mean,
                    "pf_erap2_mean_intensity": erap2_mean,
                    "log2_fold_change_erap2_over_gfp": log2_fc,
                    "shared_direct_msms_both_arms": shared_direct_msms,
                    "requires_mbr_transfer": requires_mbr,
                    "parent_reference_rna_is_measured_arm_rna": False,
                }
            )

            if not (is_reverse or is_contaminant):
                if n_gfp_pos > 0:
                    membership_rows.append(
                        {
                            "study_id": "sanz_bravo_2018b_erap2_29769354",
                            "pmid": 29769354,
                            "source_row": source_row_idx,
                            "peptide": seq,
                            "mhc_restriction": "HLA-A*29:02",
                            "arm_id": "pf97387_gfp_lentiviral_control",
                            "arm_label": "PF-GFP",
                            "condition_control": "mock",
                            "condition_overexpression_genes": "GFP",
                            "condition_transduction": "lentiviral_GFP_control",
                            "n_positive_runs": n_gfp_pos,
                            "n_msms_runs": n_gfp_msms,
                            "n_mbr_runs": n_gfp_mbr,
                            "arm_mean_intensity": gfp_mean,
                            "is_shared_with_paired_arm": n_erap2_pos > 0,
                        }
                    )
                if n_erap2_pos > 0:
                    membership_rows.append(
                        {
                            "study_id": "sanz_bravo_2018b_erap2_29769354",
                            "pmid": 29769354,
                            "source_row": source_row_idx,
                            "peptide": seq,
                            "mhc_restriction": "HLA-A*29:02",
                            "arm_id": "pf97387_erap2_lentiviral_oe",
                            "arm_label": "PF-ERAP2",
                            "condition_control": "treated",
                            "condition_overexpression_genes": "ERAP2",
                            "condition_transduction": "lentiviral_ERAP2",
                            "n_positive_runs": n_erap2_pos,
                            "n_msms_runs": n_erap2_msms,
                            "n_mbr_runs": n_erap2_mbr,
                            "arm_mean_intensity": erap2_mean,
                            "is_shared_with_paired_arm": n_gfp_pos > 0,
                        }
                    )

    return {
        "peptides": pd.DataFrame(peptide_rows),
        "run_observations": pd.DataFrame(run_rows),
        "arm_memberships": pd.DataFrame(membership_rows),
    }


def erap2_oe_contrast_recipe(
    tables_or_path: dict[str, pd.DataFrame] | str | Path,
    *,
    require_direct_msms_both_arms: bool = False,
    min_replicates_per_arm: int = 1,
    iedb_pooled_peptides: set[str] | list[str] | None = None,
    allow_unverified_checksum: bool = False,
) -> dict[str, Any]:
    """Build the declared PF-ERAP2 vs PF-GFP contrast recipe with explicit exclusions (#715).

    Separates source admission from quantitative contrast eligibility:
    - Admitted shared contrasts require positive intensity in both ``PF-GFP`` and ``PF-ERAP2``
      (meeting ``min_replicates_per_arm`` and optionally ``require_direct_msms_both_arms``).
    - Condition-specific detections (``pf_gfp_only``, ``pf_erap2_only``) are preserved separately
      as supported single-arm detections — NEVER converted into measured negative controls.
    - Reverse hits, contaminants, zero-intensity rows, and replicate/MBR-filtered rows are retained
      explicitly in ``exclusions`` with their disposition reason.
    """
    if isinstance(tables_or_path, (str, Path)):
        p = Path(tables_or_path)
        if p.is_dir():
            tables = {
                "peptides": pd.read_parquet(p / "peptides.parquet"),
                "run_observations": pd.read_parquet(p / "run_observations.parquet"),
                "arm_memberships": pd.read_parquet(p / "arm_memberships.parquet"),
            }
        else:
            tables = parse_erap2_oe_peptides(p, allow_unverified_checksum=allow_unverified_checksum)
    else:
        tables = tables_or_path

    peptides = tables["peptides"].copy()
    if min_replicates_per_arm < 1 or min_replicates_per_arm > 3:
        raise ValueError("min_replicates_per_arm must be between 1 and 3")

    pooled_set = {str(x).strip().upper() for x in (iedb_pooled_peptides or ())}
    if pooled_set:
        peptides["in_pooled_iedb_29769354"] = peptides["peptide"].str.upper().isin(pooled_set)
    else:
        peptides["in_pooled_iedb_29769354"] = False

    dispositions: list[str] = []
    reasons: list[str] = []
    for row in peptides.itertuples(index=False):
        status = row.arm_membership_status
        if status == "excluded_reverse_or_contaminant":
            dispositions.append("excluded")
            reasons.append("reverse_or_contaminant")
        elif status == "zero_intensity_all_runs":
            dispositions.append("excluded")
            reasons.append("zero_intensity_all_runs")
        elif status in ("pf_gfp_only", "pf_erap2_only"):
            dispositions.append("condition_specific_detection")
            reasons.append(status)
        elif (
            row.n_pf_gfp_positive_runs < min_replicates_per_arm
            or row.n_pf_erap2_positive_runs < min_replicates_per_arm
        ):
            dispositions.append("excluded")
            reasons.append("insufficient_replicates_per_arm")
        elif require_direct_msms_both_arms and not bool(row.shared_direct_msms_both_arms):
            dispositions.append("excluded")
            reasons.append("mbr_only_in_at_least_one_arm")
        else:
            dispositions.append("admitted_shared_contrast")
            reasons.append("supported_shared_quantitative_contrast")

    peptides["recipe_disposition"] = dispositions
    peptides["recipe_reason"] = reasons

    admitted = peptides[peptides["recipe_disposition"] == "admitted_shared_contrast"].reset_index(
        drop=True
    )
    condition_specific = peptides[
        peptides["recipe_disposition"] == "condition_specific_detection"
    ].reset_index(drop=True)
    exclusions = peptides[peptides["recipe_disposition"] == "excluded"].reset_index(drop=True)

    clean = peptides[peptides["arm_membership_status"] != "excluded_reverse_or_contaminant"]
    shared = clean[clean["arm_membership_status"] == "shared_both_arms"]

    summary = {
        "study_id": "sanz_bravo_2018b_erap2_29769354",
        "pmid": 29769354,
        "pride_accession": "PXD007679",
        "control_arm_id": "pf97387_gfp_lentiviral_control",
        "treatment_arm_id": "pf97387_erap2_lentiviral_oe",
        "require_direct_msms_both_arms": bool(require_direct_msms_both_arms),
        "min_replicates_per_arm": int(min_replicates_per_arm),
        "n_source_rows": len(peptides),
        "n_excluded_reverse_or_contaminant": int(
            (peptides["arm_membership_status"] == "excluded_reverse_or_contaminant").sum()
        ),
        "n_clean_peptides": len(clean),
        "n_shared_peptides": len(shared),
        "n_shared_direct_msms_both_arms": int(shared["shared_direct_msms_both_arms"].sum()),
        "n_shared_mbr_assisted": int(shared["requires_mbr_transfer"].sum()),
        "n_pf_gfp_only": int((clean["arm_membership_status"] == "pf_gfp_only").sum()),
        "n_pf_erap2_only": int((clean["arm_membership_status"] == "pf_erap2_only").sum()),
        "n_zero_intensity_all_runs": int(
            (clean["arm_membership_status"] == "zero_intensity_all_runs").sum()
        ),
        "n_admitted_shared_contrasts": len(admitted),
        "n_condition_specific_detections": len(condition_specific),
        "n_excluded_from_contrast": len(exclusions),
        "n_arm_membership_rows": len(tables["arm_memberships"]),
        "n_run_observation_rows": len(tables["run_observations"]),
        "n_overlapping_pooled_iedb_peptides": int(clean["in_pooled_iedb_29769354"].sum()),
        "pooled_iedb_rows_eligible_for_oe_contrast": False,
        "ms_nondetection_is_measured_negative": False,
        "parent_reference_rna_is_measured_arm_rna": False,
    }
    return {
        "admitted_shared_contrasts": admitted,
        "condition_specific_detections": condition_specific,
        "exclusions": exclusions,
        "arm_memberships": tables["arm_memberships"],
        "run_observations": tables["run_observations"],
        "summary": summary,
    }


def write_erap2_oe_package(
    source_dir: str | Path,
    dest_dir: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> Path:
    """Write a manifest-hashed Parquet package for Sanz-Bravo 2018b PXD007679 ERAP2 OE."""
    src = Path(source_dir)
    dst = Path(dest_dir)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    peptides_path = src / "peptides.txt"
    if not peptides_path.is_file():
        raise FileNotFoundError(f"Expected MaxQuant peptides.txt in {src}")

    tables = parse_erap2_oe_peptides(
        peptides_path,
        include_reverse_or_contaminant=True,
        allow_unverified_checksum=allow_unverified_checksum,
    )
    recipe = erap2_oe_contrast_recipe(tables)
    contract = erap2_oe_source_contract()

    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}-", dir=dst.parent))
    try:
        for name, df in tables.items():
            df.to_parquet(staging / f"{name}.parquet", index=False)
        (staging / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "contrast_recipe.json").write_text(
            json.dumps(recipe["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files_map = {
            p.name: {"sha256": _sha256_file(p), "size_bytes": p.stat().st_size}
            for p in sorted(staging.iterdir())
            if p.is_file()
        }
        from .version import __version__

        manifest = {
            "package_family": PACKAGE_FAMILY,
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "pmid": 29769354,
            "pride_accession": "PXD007679",
            "source_peptides_sha256": _sha256_file(peptides_path),
            "summary": recipe["summary"],
            "files": files_map,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(dst)
        return dst / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_erap2_oe_package(package_dir: str | Path) -> dict[str, Any]:
    """Verify integrity, hashes, and relation conservation of a written ERAP2 OE package."""
    pkg = Path(package_dir)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("package_family") != PACKAGE_FAMILY:
        raise ValueError(f"Unexpected package_family: {manifest.get('package_family')!r}")
    for fname, expected in manifest.get("files", {}).items():
        path = pkg / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing package file: {fname}")
        actual_sha = _sha256_file(path)
        actual_size = path.stat().st_size
        if actual_sha != expected["sha256"] or actual_size != expected["size_bytes"]:
            raise ValueError(f"Package file integrity mismatch: {fname}")
    peptides = pd.read_parquet(pkg / "peptides.parquet")
    runs = pd.read_parquet(pkg / "run_observations.parquet")
    memberships = pd.read_parquet(pkg / "arm_memberships.parquet")
    if len(peptides) * len(_RUN_SPEC) != len(runs):
        raise ValueError("Run observation count does not conserve peptides x 6 runs")
    expected_memberships = int(
        peptides[peptides["arm_membership_status"] != "excluded_reverse_or_contaminant"][
            "arm_membership_status"
        ]
        .map(
            {
                "shared_both_arms": 2,
                "pf_gfp_only": 1,
                "pf_erap2_only": 1,
                "zero_intensity_all_runs": 0,
            }
        )
        .sum()
    )
    if len(memberships) != expected_memberships:
        raise ValueError(
            f"Arm membership conservation mismatch: expected {expected_memberships}, "
            f"got {len(memberships)}"
        )
    return manifest
